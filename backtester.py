"""
backtester.py
==============
Простой векторизованный бэктест-фреймворк.

Принимает:
- DataFrame с OHLC + колонкой 'signal' (-1/0/+1 — позиция на каждом баре)
- Параметры инструмента (спред, размер пипса, своп)

Возвращает:
- DataFrame с пошаговыми результатами (equity, drawdown и т.д.)
- dict с агрегированными метриками (Sharpe, max DD, win rate и т.д.)

Логика:
- Сигнал на баре N используется для входа на закрытии бара N
- Позиция держится до изменения сигнала
- На каждой смене позиции: издержки = спред (входит дважды — вход и выход)
- Своп начисляется ежедневно за overnight (грубое приближение)
"""

import numpy as np
import pandas as pd


def backtest(df, signal_col="signal",
             spread_pips=2.0, pip_size=0.0001,
             swap_long_per_day=0.0, swap_short_per_day=0.0,
             risk_per_trade_pct=1.0, capital=10000.0,
             atr_periods=14, atr_stop_mult=2.0):
    """
    Бэктест с учётом издержек.

    Параметры:
    ----------
    df : DataFrame с колонками ['timestamps', 'open', 'high', 'low', 'close', signal_col]
    signal_col : имя колонки с сигналами (-1, 0, +1)
    spread_pips : спред в пипсах для round-trip (вход + выход) — будет учтён по сторонам
    pip_size : размер пипса (0.0001 для большинства FX, 0.01 для JPY-пар, 0.1 для XAU/USD)
    swap_long_per_day : своп для лонга, в пунктах цены, минус = списание
    swap_short_per_day : своп для шорта, аналогично
    risk_per_trade_pct : % капитала на сделку (используется для sizing)
    capital : стартовый капитал
    atr_periods : период ATR для sizing/стопов
    atr_stop_mult : сколько ATR используется как стоп (для волатильность-адаптивного sizing)

    Возвращает:
    -----------
    df_out : DataFrame с пошаговыми результатами
    metrics : dict с агрегированными метриками
    """
    df = df.copy().reset_index(drop=True)
    df["signal"] = df[signal_col].fillna(0).astype(int)

    # ATR для определения размера стопа и нормализации
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"]  - df["close"].shift(1)).abs(),
    }).max(axis=1)
    df["atr"] = tr.rolling(atr_periods).mean()

    # Доходность одного бара (если бы держали позицию весь бар)
    df["bar_return"] = df["close"].pct_change().fillna(0)

    # Позиция, которую держим НА бар N — это сигнал, поданный НА бар N-1
    # (мы видим сигнал на закрытии бара N, и входим на следующем баре)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Сделка происходит когда позиция меняется
    df["position_change"] = df["position"].diff().fillna(0).abs()
    # 0 -> 1 или 0 -> -1: одна сторона издержек (вход)
    # 1 -> -1: две стороны (закрытие + новый вход), и т.д.
    # Упрощение: каждое ненулевое изменение = 1 спред
    df["trade_cost_pct"] = (df["position_change"] > 0).astype(int) * (spread_pips * pip_size / df["close"])

    # Доходность за бар с учётом позиции
    df["strategy_return"] = df["position"] * df["bar_return"] - df["trade_cost_pct"]

    # Свопы — упрощённо, начисляются на каждом баре пропорционально
    # (для D1 это 1 раз в день, для H4 — 6 раз в день, разделим)
    bars_per_day = _infer_bars_per_day(df["timestamps"])
    swap_per_bar_long  = swap_long_per_day  / bars_per_day / df["close"]
    swap_per_bar_short = swap_short_per_day / bars_per_day / df["close"]
    df["swap_pct"] = np.where(df["position"] > 0, swap_per_bar_long,
                     np.where(df["position"] < 0, swap_per_bar_short, 0.0))
    df["strategy_return"] += df["swap_pct"]

    # Equity curve
    df["equity"] = capital * (1 + df["strategy_return"]).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

    # Метрики
    metrics = _compute_metrics(df, capital, bars_per_day)
    return df, metrics


def _infer_bars_per_day(timestamps):
    """Грубо определяет число баров в сутки по медианному интервалу."""
    diff = timestamps.diff().dropna()
    if len(diff) == 0:
        return 1.0
    median_seconds = diff.dt.total_seconds().median()
    if median_seconds <= 0:
        return 1.0
    return 86400 / median_seconds


def _compute_metrics(df, capital, bars_per_day):
    final_equity = df["equity"].iloc[-1]
    total_return = (final_equity / capital - 1) * 100

    # Сколько торговых лет — для аннуализации
    days = (df["timestamps"].iloc[-1] - df["timestamps"].iloc[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1.0

    cagr = (final_equity / capital) ** (1 / years) - 1 if final_equity > 0 else -1

    # Sharpe (annualized)
    ret = df["strategy_return"].replace([np.inf, -np.inf], 0).fillna(0)
    sharpe = (ret.mean() / ret.std() * np.sqrt(bars_per_day * 252)) if ret.std() > 0 else 0

    # Max drawdown
    max_dd = df["drawdown"].min() * 100

    # Подсчёт сделок: каждое ненулевое изменение позиции — это полу-сделка
    # Полную сделку считаем как пару (вход + выход)
    n_position_changes = (df["position_change"] > 0).sum()
    n_trades = n_position_changes // 2  # грубо

    # Win rate и avg R: считаем по периодам где позиция != 0
    # Группируем по "удержание позиции"
    df["trade_id"] = (df["position_change"] > 0).cumsum() * (df["position"] != 0)
    trades = []
    for tid, group in df[df["trade_id"] > 0].groupby("trade_id"):
        ret_sum = group["strategy_return"].sum()
        trades.append(ret_sum)

    if trades:
        trades = pd.Series(trades)
        win_rate = (trades > 0).mean() * 100
        avg_win = trades[trades > 0].mean() * 100 if (trades > 0).any() else 0
        avg_loss = trades[trades < 0].mean() * 100 if (trades < 0).any() else 0
        profit_factor = (trades[trades > 0].sum() / -trades[trades < 0].sum()
                         if (trades < 0).any() and trades[trades < 0].sum() != 0 else float('inf'))
    else:
        win_rate = avg_win = avg_loss = 0
        profit_factor = 0

    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "sharpe":           round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades":         int(len(trades)),
        "win_rate_pct":     round(win_rate, 1),
        "avg_win_pct":      round(avg_win, 3),
        "avg_loss_pct":     round(avg_loss, 3),
        "profit_factor":    round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "final_equity":     round(final_equity, 2),
        "years":            round(years, 1),
    }
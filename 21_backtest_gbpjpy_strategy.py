"""
21_backtest_gbpjpy_strategy.py
================================
Полноценный бэктест стратегии:
  GBP/JPY M15
  Сигнал: RSI(14) > level (75 или 80) во флэтовые часы [0,1,2,5,6,7,22,23]
  Действие: SHORT (fade перекупленности)
  Выход: SL × ATR, TP × ATR, или таймаут N баров

Бэктест с реалистичным учётом:
  - Издержки 1.7 пипса (спред + комиссия FxPro) на каждую сделку
  - Стоп и тейк проверяются БАРНО (high/low бара)
  - При срабатывании двух уровней в одном баре — консервативная оценка (SL первым)
  - Реальные просадки в пипсах и %

Перебор параметров:
  - RSI level: 75, 80
  - SL: 1.0, 1.5, 2.0 × ATR
  - TP: 1.0, 1.5, 2.0 × ATR (или фиксированно)
  - Max hold: 8, 16 баров

Walk-forward: train 2022-2024, test 2025-2026.

Запуск:
    python 21_backtest_gbpjpy_strategy.py
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_FILE  = "./data/fxpro/gbpjpy_m15.csv"
OUTPUT_DIR = "./data/fxpro/gbpjpy_strategy"

PIP_SIZE  = 0.01
COST_PIPS = 1.7

FLAT_HOURS = [0, 1, 2, 5, 6, 7, 22, 23]

# Параметры стратегии (сетка)
RSI_LEVELS    = [75, 80]
SL_MULTS      = [1.0, 1.5, 2.0]    # SL = ATR × mult
TP_MULTS      = [1.0, 1.5, 2.0]    # TP = ATR × mult
MAX_HOLD_BARS = [8, 16]

# Train/test split
TRAIN_END = "2024-12-31"

# Начальный капитал (для оценки в %)
CAPITAL_USD = 1000.0
LOT_SIZE    = 0.01

# Грубо: 1 пипс GBPJPY на 0.01 лота при курсе USDJPY~150 ≈ $0.07
# Точно зависит от курса, но для оценок этого достаточно.
PIP_VALUE_USD = 0.067


def load():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} не найден.")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["hour"] = df["timestamps"].dt.hour
    return df


def rsi_func(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_func(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate(df, rsi_level, sl_mult, tp_mult, max_hold):
    """
    Симулирует стратегию бар за баром.

    Сигнал на баре i (close уже сформирован):
      - RSI(close)[i] > rsi_level И час[i] во FLAT_HOURS
      - входим SHORT на open следующего бара (i+1)
    Стоп: entry + ATR[i] × sl_mult
    Тейк: entry - ATR[i] × tp_mult
    Таймаут: закрытие по close через max_hold баров

    Если в одном баре сработали и SL и TP — берём SL (консервативно).

    Returns: list dict с информацией о сделках
    """
    df = df.copy()
    df["rsi"] = rsi_func(df["close"], 14)
    df["atr"] = atr_func(df, 14)

    open_  = df["open"].values
    high   = df["high"].values
    low    = df["low"].values
    close  = df["close"].values
    rsi_v  = df["rsi"].values
    atr_v  = df["atr"].values
    hour_v = df["hour"].values
    ts     = df["timestamps"].values
    n = len(df)

    trades = []
    i = 14  # пропускаем первые бары пока ATR/RSI не сформированы
    in_position_until = -1

    while i < n - max_hold - 1:
        # Пропускаем если в позиции
        if i <= in_position_until:
            i += 1
            continue

        # Условие сигнала
        if (rsi_v[i] > rsi_level) and (hour_v[i] in FLAT_HOURS) and not np.isnan(atr_v[i]):
            atr_i = atr_v[i]
            # Входим на open следующего бара
            entry_idx = i + 1
            entry_price = open_[entry_idx]
            sl_price = entry_price + atr_i * sl_mult
            tp_price = entry_price - atr_i * tp_mult

            # Симулируем бары от entry_idx до entry_idx + max_hold
            exit_reason = "timeout"
            exit_idx = entry_idx + max_hold
            exit_price = close[min(exit_idx, n - 1)]

            for j in range(entry_idx, min(entry_idx + max_hold + 1, n)):
                # Сначала проверяем SL (консервативно)
                if high[j] >= sl_price:
                    exit_reason = "SL"
                    exit_idx = j
                    exit_price = sl_price
                    break
                if low[j] <= tp_price:
                    exit_reason = "TP"
                    exit_idx = j
                    exit_price = tp_price
                    break

            # PnL в пипсах (short: entry - exit)
            pnl_pips = (entry_price - exit_price) / PIP_SIZE - COST_PIPS

            trades.append({
                "signal_idx": i,
                "entry_idx":  entry_idx,
                "exit_idx":   exit_idx,
                "signal_time": ts[i],
                "entry_time":  ts[entry_idx],
                "exit_time":   ts[min(exit_idx, n - 1)],
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "sl_price":    sl_price,
                "tp_price":    tp_price,
                "atr_at_signal": atr_i,
                "reason":      exit_reason,
                "pnl_pips":    pnl_pips,
                "bars_held":   exit_idx - entry_idx,
            })
            in_position_until = exit_idx

        i += 1

    return trades


def metrics(trades, label=""):
    if not trades:
        return None
    pnls = np.array([t["pnl_pips"] for t in trades])
    cumsum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumsum)
    dd = cumsum - peak

    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    reasons = pd.Series([t["reason"] for t in trades]).value_counts().to_dict()

    return {
        "label":         label,
        "n_trades":      n,
        "total_pips":    round(pnls.sum(), 1),
        "avg_pips":      round(pnls.mean(), 2),
        "win_rate":      round((pnls > 0).mean() * 100, 1),
        "avg_win":       round(wins.mean(), 2) if len(wins) else 0,
        "avg_loss":      round(losses.mean(), 2) if len(losses) else 0,
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if len(losses) and losses.sum() < 0 else float("inf"),
        "max_dd_pips":   round(dd.min(), 1),
        "max_dd_pct":    round(dd.min() * PIP_VALUE_USD / CAPITAL_USD * 100, 2),
        "total_usd":     round(pnls.sum() * PIP_VALUE_USD, 2),
        "exits": reasons,
        "trades_per_year": round(n / 4.0, 0),  # ~4 года данных
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load()
    print(f"Загружено {len(df)} баров: {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
    print(f"Капитал для оценок: ${CAPITAL_USD}, лот {LOT_SIZE}, ~${PIP_VALUE_USD} за 1 пипс")
    print()

    df_train = df[df["timestamps"] <= TRAIN_END].reset_index(drop=True)
    df_test  = df[df["timestamps"] >  TRAIN_END].reset_index(drop=True)
    print(f"TRAIN: {len(df_train)} баров")
    print(f"TEST:  {len(df_test)} баров")

    # Прогон сетки на TRAIN
    print("\n" + "=" * 110)
    print("ОПТИМИЗАЦИЯ НА TRAIN")
    print("=" * 110)
    print(f"{'RSI':>3} {'SL':>4} {'TP':>4} {'Hold':>4} | "
          f"{'N':>4} {'Total':>7} {'Avg':>6} {'WR%':>5} {'PF':>5} "
          f"{'MaxDD_p':>8} {'MaxDD%':>7} {'$':>7}")
    print("-" * 110)

    train_results = []
    for rsi_lvl in RSI_LEVELS:
        for sl in SL_MULTS:
            for tp in TP_MULTS:
                for hold in MAX_HOLD_BARS:
                    trades = simulate(df_train, rsi_lvl, sl, tp, hold)
                    m = metrics(trades, f"r{rsi_lvl}_sl{sl}_tp{tp}_h{hold}")
                    if m is None or m["n_trades"] < 20:
                        continue
                    train_results.append({
                        "rsi": rsi_lvl, "sl": sl, "tp": tp, "hold": hold, **m
                    })
                    print(f"{rsi_lvl:>3} {sl:>4.1f} {tp:>4.1f} {hold:>4} | "
                          f"{m['n_trades']:>4} {m['total_pips']:>+7.1f} "
                          f"{m['avg_pips']:>+6.2f} {m['win_rate']:>5.1f} "
                          f"{m['profit_factor']:>5.2f} "
                          f"{m['max_dd_pips']:>+8.1f} {m['max_dd_pct']:>+7.1f} "
                          f"{m['total_usd']:>+7.2f}")

    df_tr = pd.DataFrame(train_results)
    if len(df_tr) == 0:
        print("ERROR: нет результатов")
        return

    # Топ-5 по total_pips на TRAIN (с фильтром: PF > 1, DD% не катастрофический)
    df_tr_good = df_tr[(df_tr["profit_factor"] > 1.0) & (df_tr["max_dd_pct"] > -30)].copy()
    if len(df_tr_good) == 0:
        print("\nНи одна конфигурация не прошла базовые фильтры PF>1 и DD>-30%.")
        df_tr_good = df_tr.copy()

    top = df_tr_good.sort_values("total_pips", ascending=False).head(5)

    print("\n" + "=" * 110)
    print("ТОП-5 КОНФИГУРАЦИЙ TRAIN (PF>1, DD>-30%) -> ВАЛИДАЦИЯ НА TEST")
    print("=" * 110)
    print(f"{'RSI':>3} {'SL':>4} {'TP':>4} {'Hold':>4} | "
          f"{'TR_N':>4} {'TR_pips':>8} {'TR_$':>7} {'TR_PF':>5} {'TR_DD%':>7} | "
          f"{'TE_N':>4} {'TE_pips':>8} {'TE_$':>7} {'TE_PF':>5} {'TE_DD%':>7} | {'OK?'}")
    print("-" * 110)

    final_results = []
    for _, r in top.iterrows():
        trades_te = simulate(df_test, int(r["rsi"]), r["sl"], r["tp"], int(r["hold"]))
        m_te = metrics(trades_te, "test") or {
            "n_trades": 0, "total_pips": 0, "total_usd": 0,
            "profit_factor": 0, "max_dd_pct": 0
        }

        ok = "✓✓" if (m_te["total_pips"] > 0 and m_te["profit_factor"] > 1.0) else (
             "~" if m_te["total_pips"] > -50 else "✗")

        print(f"{int(r['rsi']):>3} {r['sl']:>4.1f} {r['tp']:>4.1f} {int(r['hold']):>4} | "
              f"{int(r['n_trades']):>4} {r['total_pips']:>+8.1f} {r['total_usd']:>+7.2f} "
              f"{r['profit_factor']:>5.2f} {r['max_dd_pct']:>+7.1f} | "
              f"{m_te['n_trades']:>4} {m_te['total_pips']:>+8.1f} {m_te['total_usd']:>+7.2f} "
              f"{m_te['profit_factor']:>5.2f} {m_te['max_dd_pct']:>+7.1f} | {ok}")

        final_results.append({
            "rsi": int(r["rsi"]), "sl": r["sl"], "tp": r["tp"], "hold": int(r["hold"]),
            **{f"tr_{k}": v for k, v in r.items() if k not in ["rsi","sl","tp","hold"]},
            **{f"te_{k}": v for k, v in m_te.items()},
            "ok": ok,
        })

    pd.DataFrame(final_results).to_csv(
        os.path.join(OUTPUT_DIR, "top5_validation.csv"), index=False)

    # Если есть хоть один ✓✓ — детальный анализ лучшего
    good = [r for r in final_results if r["ok"] == "✓✓"]
    if good:
        best = max(good, key=lambda x: x["te_total_pips"])
        print("\n" + "=" * 110)
        print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ КОНФИГУРАЦИИ: "
              f"RSI>{best['rsi']}, SL={best['sl']}xATR, TP={best['tp']}xATR, hold={best['hold']}")
        print("=" * 110)
        # Прогон на весь датасет для общей картины
        trades_all = simulate(df, best["rsi"], best["sl"], best["tp"], best["hold"])
        m_all = metrics(trades_all, "all")
        print(f"  Всего сделок:       {m_all['n_trades']} ({m_all['trades_per_year']:.0f} в год)")
        print(f"  Win rate:           {m_all['win_rate']}%")
        print(f"  Avg win / loss:     +{m_all['avg_win']} / {m_all['avg_loss']} пипс")
        print(f"  Profit factor:      {m_all['profit_factor']}")
        print(f"  Total:              {m_all['total_pips']} пипс = ${m_all['total_usd']}")
        print(f"  Max drawdown:       {m_all['max_dd_pips']} пипс = {m_all['max_dd_pct']}% капитала")
        print(f"  Распределение выходов: {m_all['exits']}")

        # Годовая разбивка
        df_trades = pd.DataFrame(trades_all)
        df_trades["year"] = pd.to_datetime(df_trades["entry_time"]).dt.year
        print(f"\n  По годам:")
        yearly = df_trades.groupby("year")["pnl_pips"].agg(["count", "sum", "mean"])
        for year, row in yearly.iterrows():
            print(f"    {year}: {int(row['count']):>4} сделок, "
                  f"{row['sum']:>+7.1f} пипс (${row['sum']*PIP_VALUE_USD:>+6.2f}), "
                  f"avg {row['mean']:>+5.2f}")
    else:
        print("\n⚠️  Ни одна из топ-5 конфигураций TRAIN не прошла валидацию на TEST.")
        print("    Это значит — параметры подгоняются под TRAIN, реального edge меньше.")

    print("\n" + "=" * 110)
    print("ВЫВОДЫ")
    print("=" * 110)
    print("""
Что искать:
  - Топ-5 на TRAIN: PF>1.3, DD%<20%, разумное число сделок (>100)
  - Из этих топ-5, на TEST: должна быть хотя бы 1 конфигурация ✓✓
  - В детальном анализе: win_rate 35-50%, годовая прибыль > 0 в большинстве лет

Если ✓✓ есть — у нас рабочая стратегия для demo торговли.
Если только ~ — есть смысл оптимизировать дальше (другие фильтры).
Если все ✗ — реалистичный SL/TP убил тот edge что мы видели в forward returns.
""")


if __name__ == "__main__":
    main()
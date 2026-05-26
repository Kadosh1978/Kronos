"""
02_eda_and_split.py
===================
EDA по xauusd_h1.csv + walk-forward сплит на train/val/test.

Что делает:
1. Загружает датасет, базовая статистика
2. Проверяет пропуски (выходные, праздники, дыры в данных)
3. Анализирует non-stationarity по годовым окнам (режимы рынка)
4. Делает walk-forward сплит train/val/test по времени
5. Сохраняет сплиты в отдельные CSV
6. Опционально рисует графики (если установлен matplotlib)

Запуск:
    python 02_eda_and_split.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ========== Настройки ==========
INPUT_FILE  = "./data/xauusd_h1.csv"
OUTPUT_DIR  = "./data"

# Walk-forward сплит по времени.
# 70/15/15 — стандартное соотношение для финансовых временных рядов.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC = 0.15 (остаток)

# Если хочешь обрезать старые данные (например только с 2019),
# поставь сюда дату начала. None = брать всё.
START_DATE_FILTER = None  # например: "2019-01-01"

PLOT = True   # построить графики (если есть matplotlib)
# ===============================


def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: файл {INPUT_FILE} не найден. Сначала запусти 01_download_xauusd_history.py")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if START_DATE_FILTER:
        before = len(df)
        df = df[df["timestamps"] >= START_DATE_FILTER].reset_index(drop=True)
        print(f"Фильтр по дате {START_DATE_FILTER}: {before} -> {len(df)} баров")

    return df


def basic_stats(df):
    print("\n" + "=" * 60)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Период:  {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
    print(f"Баров:   {len(df)}")
    print(f"Цена:    min={df['close'].min():.2f}, max={df['close'].max():.2f}, "
          f"mean={df['close'].mean():.2f}")

    # Логарифмические доходности
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    print(f"\nЛог-доходности (час):")
    print(f"  std:   {df['log_ret'].std():.5f}  (~{df['log_ret'].std()*100:.3f}% / час)")
    print(f"  min:   {df['log_ret'].min():.5f}  (худший час)")
    print(f"  max:   {df['log_ret'].max():.5f}  (лучший час)")

    # True range / ATR — мера диапазона бара в долларах
    df["tr"] = df["high"] - df["low"]
    print(f"\nДиапазон бара (high-low), USD:")
    print(f"  mean:  {df['tr'].mean():.2f}")
    print(f"  p50:   {df['tr'].median():.2f}")
    print(f"  p95:   {df['tr'].quantile(0.95):.2f}")
    print(f"  max:   {df['tr'].max():.2f}")


def check_gaps(df):
    """Проверяем пропуски в часовой сетке."""
    print("\n" + "=" * 60)
    print("ПРОПУСКИ В ВРЕМЕННОЙ СЕТКЕ")
    print("=" * 60)

    diff = df["timestamps"].diff()
    one_hour = pd.Timedelta(hours=1)

    normal = (diff == one_hour).sum()
    short  = ((diff < one_hour) & (diff > pd.Timedelta(0))).sum()
    long_gaps = diff[diff > one_hour]

    print(f"Нормальных интервалов (1ч): {normal:>7} ({normal/len(df)*100:.1f}%)")
    print(f"Коротких (< 1ч):            {short:>7}")
    print(f"Длинных (> 1ч):             {len(long_gaps):>7}")

    if len(long_gaps) > 0:
        # Группируем длинные пропуски по размеру
        gap_hours = long_gaps.dt.total_seconds() / 3600
        print(f"\nРаспределение длинных пропусков (часы):")
        print(f"  Выходные (~50-65ч):    {((gap_hours >= 40) & (gap_hours <= 70)).sum():>5}")
        print(f"  Праздники (70-200ч):   {((gap_hours > 70) & (gap_hours <= 200)).sum():>5}")
        print(f"  Большие дыры (>200ч):  {(gap_hours > 200).sum():>5}")

        big_gaps = df.loc[long_gaps[gap_hours > 200].index]
        if len(big_gaps) > 0:
            print(f"\n  Большие дыры (возможно проблемы с данными):")
            for idx in big_gaps.index[:5]:
                prev = df["timestamps"].iloc[idx-1]
                curr = df["timestamps"].iloc[idx]
                print(f"    {prev} -> {curr} ({(curr-prev).total_seconds()/3600:.0f}ч)")


def regime_analysis(df):
    """Сравниваем статистику по годам — насколько режимы рынка меняются."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ НЕСТАЦИОНАРНОСТИ ПО ГОДАМ")
    print("=" * 60)
    print("(показывает, насколько разные режимы рынка в разные годы)")

    df["year"] = df["timestamps"].dt.year
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_ret"] = df["log_ret"].abs()

    stats = df.groupby("year").agg(
        bars=("close", "count"),
        price_mean=("close", "mean"),
        price_std=("close", "std"),
        vol_hourly=("log_ret", "std"),      # волатильность
        avg_abs_ret=("abs_ret", "mean"),
        tick_vol_mean=("volume", "mean"),
    )
    stats["vol_annual_%"] = stats["vol_hourly"] * np.sqrt(24*365) * 100

    pd.set_option("display.float_format", lambda x: f"{x:>10.3f}")
    print(stats.to_string())
    pd.reset_option("display.float_format")

    print("\nИнтерпретация:")
    print(f"  Если vol_annual_% сильно различается между годами -> non-stationarity.")
    print(f"  Если tick_vol_mean растёт -> микроструктура меняется.")


def split_data(df):
    """Walk-forward сплит по времени."""
    print("\n" + "=" * 60)
    print("СПЛИТ TRAIN / VAL / TEST (walk-forward по времени)")
    print("=" * 60)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    for name, part in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n  {name:5}: {len(part):>6} баров  "
              f"[{part['timestamps'].iloc[0]} -> {part['timestamps'].iloc[-1]}]")
        print(f"         цена: {part['close'].min():.2f} - {part['close'].max():.2f}, "
              f"средняя {part['close'].mean():.2f}")

    print("\nВАЖНО:")
    print("  - TEST — самый свежий период, на нём НИКОГДА не обучаемся.")
    print("  - Сравни диапазоны цен — если в TEST цены сильно выше TRAIN,")
    print("    модель будет видеть 'невиданное' и это нормально для walk-forward.")

    # Сохраняем только колонки, которые ест Kronos
    keep_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = part[keep_cols].copy()
        path = os.path.join(OUTPUT_DIR, f"xauusd_h1_{name}.csv")
        out.to_csv(path, index=False)
        print(f"  Сохранено: {path}")

    return train_df, val_df, test_df


def make_plots(df, train_df, val_df, test_df):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib не установлен — графики пропущены.")
        print("       pip install matplotlib  если хочешь увидеть визуально.")
        return

    print("\n[plot] Рисую графики...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Цена с разметкой сплитов
    ax = axes[0]
    ax.plot(df["timestamps"], df["close"], lw=0.5, color="black")
    ax.axvline(train_df["timestamps"].iloc[-1], color="orange", ls="--", label="train | val")
    ax.axvline(val_df["timestamps"].iloc[-1],   color="red",    ls="--", label="val | test")
    ax.set_title("XAU/USD H1 — цена и сплиты")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Часовая волатильность (rolling std логдоходностей, окно ~1 месяц)
    ax = axes[1]
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    rolling_vol = df["log_ret"].rolling(24*30).std() * np.sqrt(24*365) * 100
    ax.plot(df["timestamps"], rolling_vol, lw=0.7, color="darkblue")
    ax.set_title("Скользящая годовая волатильность (% , окно ~30 дней)")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    # 3. Tick volume по месяцам
    ax = axes[2]
    monthly_vol = df.set_index("timestamps")["volume"].resample("ME").mean()
    ax.plot(monthly_vol.index, monthly_vol.values, color="darkgreen")
    ax.set_title("Средний tick_volume (помесячно) — индикатор изменения микроструктуры")
    ax.set_ylabel("ticks/bar")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "eda_overview.png")
    plt.savefig(out_path, dpi=100)
    print(f"[plot] Сохранено: {out_path}")
    plt.close()


def main():
    df = load_data()
    basic_stats(df)
    check_gaps(df)
    regime_analysis(df)
    train_df, val_df, test_df = split_data(df)
    if PLOT:
        make_plots(df, train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("ГОТОВО. Следующий шаг — zero-shot инференс Kronos на test'е.")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
12_download_eurusd_fxpro.py
============================
Скачивает историю EUR/USD с FxPro в M5, M15, H1.

Зачем разные ТФ:
- M5  — для скальпинга и быстрых стратегий
- M15 — для интрадей-моментум стратегий
- H1  — для контекста (тренд на H1 — фильтр для входов на M5)

Сохраняет в отдельную папку data/fxpro/ чтобы не путать со старыми данными
Альфа-Форекса.

Запуск:
    python 12_download_eurusd_fxpro.py
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: пакет MetaTrader5 не установлен.")
    sys.exit(1)


# ========== Настройки ==========
SYMBOL     = "EURUSD"
OUTPUT_DIR = "./data/fxpro"

# M5 за 10 лет = ~700К баров. Это нормально для современного железа.
# Если брокер столько не даёт — скачаем что есть.
DATE_FROM = datetime(2014, 1, 1)
DATE_TO   = datetime.now()

TIMEFRAMES = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}

# Размер чанка для скачивания. Меньше для M5 — больше баров в день.
CHUNK_DAYS = {
    "M5":  60,    # 60 дней M5 = ~16K баров
    "M15": 180,   # 180 дней M15 = ~16K баров
    "H1":  365,   # 365 дней H1 = ~6K баров
}
# ===============================


def init_mt5():
    if not mt5.initialize():
        print(f"ERROR: MT5 не подключился: {mt5.last_error()}")
        print("Убедись что MT5 терминал FxPro запущен и залогинен.")
        sys.exit(1)

    acc = mt5.account_info()
    term = mt5.terminal_info()
    is_demo = "demo" in (acc.server or "").lower() if acc else False

    print("=" * 60)
    print(f"Терминал:    {term.company if term else '?'}")
    print(f"Сервер:      {acc.server if acc else '?'}")
    print(f"Тип счёта:   {'DEMO ✓' if is_demo else '⚠️  REAL'}")
    print(f"Баланс:      {acc.balance} {acc.currency}" if acc else "")
    print("=" * 60)

    if not is_demo:
        print("\n⚠️  Чтение данных безопасно, но рекомендуется использовать demo.")
        print("Продолжаю (только чтение)...\n")


def check_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"ERROR: символ {symbol} не найден.")
        mt5.shutdown()
        sys.exit(1)
    if not info.visible:
        mt5.symbol_select(symbol, True)
    print(f"Символ: {symbol}")
    print(f"  Digits: {info.digits}, spread: {info.spread} points")
    return info


def download_chunked(symbol, tf_const, tf_name):
    """Качает историю годовыми чанками для надёжности."""
    chunks = []
    current = DATE_FROM
    chunk_days = CHUNK_DAYS[tf_name]
    chunk_delta = timedelta(days=chunk_days)

    total_chunks = (DATE_TO - DATE_FROM).days // chunk_days + 1
    chunk_idx = 0

    while current < DATE_TO:
        chunk_idx += 1
        chunk_end = min(current + chunk_delta, DATE_TO)
        rates = mt5.copy_rates_range(symbol, tf_const, current, chunk_end)

        if rates is not None and len(rates) > 0:
            chunks.append(pd.DataFrame(rates))
            # Прогресс каждые 10 чанков
            if chunk_idx % 10 == 0 or chunk_idx == total_chunks:
                so_far = sum(len(c) for c in chunks)
                print(f"    [{chunk_idx:>3}/{total_chunks}] {current.date()}: всего {so_far} баров")

        current = chunk_end

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["timestamps"] = pd.to_datetime(df["time"], unit="s")

    out = df[["timestamps", "open", "high", "low", "close", "tick_volume", "real_volume"]].copy()
    out.rename(columns={"tick_volume": "volume", "real_volume": "amount"}, inplace=True)
    if (out["amount"] == 0).all():
        out["amount"] = out["volume"] * out["close"]
    return out


def main():
    init_mt5()
    check_symbol(SYMBOL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nСкачиваю историю {SYMBOL} c {DATE_FROM.date()} по {DATE_TO.date()}\n")

    summary = []
    for tf_name, tf_const in TIMEFRAMES.items():
        print(f"--- {tf_name} ---")
        df = download_chunked(SYMBOL, tf_const, tf_name)

        if df is None or len(df) == 0:
            print(f"  {tf_name}: пусто, пропуск")
            continue

        fname = f"{SYMBOL.lower()}_{tf_name.lower()}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(path, index=False)

        actual_start = df["timestamps"].iloc[0]
        actual_end = df["timestamps"].iloc[-1]
        size_mb = os.path.getsize(path) / 1024**2

        print(f"  {tf_name}: {len(df):>7} баров")
        print(f"         период: {actual_start} -> {actual_end}")
        print(f"         файл: {path} ({size_mb:.1f} MB)")
        print()

        summary.append({
            "timeframe": tf_name,
            "bars": len(df),
            "start": actual_start,
            "end": actual_end,
            "size_mb": round(size_mb, 1),
            "file": fname,
        })

    if summary:
        df_sum = pd.DataFrame(summary)
        sum_path = os.path.join(OUTPUT_DIR, "_summary.csv")
        df_sum.to_csv(sum_path, index=False)
        print(f"Сводка: {sum_path}")

        # Сравнение глубины истории
        print("\nРеальная глубина истории у FxPro:")
        for s in summary:
            years = (s["end"] - s["start"]).days / 365.25
            print(f"  {s['timeframe']}: {years:.1f} лет ({s['bars']} баров)")

    mt5.shutdown()
    print("\nГотово.")


if __name__ == "__main__":
    main()
"""
08_download_fx_history.py
==========================
Скачивает историю нескольких FX-пар и металлов с MT5 в H4 и D1.
Цель — данные для бэктестов трендовых/carry стратегий.

Тестирует разные суффиксы (rfd, без суффикса, _ и т.д.) — у Альфа-Форекса
символы обычно с суффиксом 'rfd'.

Запуск:
    python 08_download_fx_history.py

Перед запуском: MT5 терминал запущен и залогинен (DEMO!).
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
# Базовые имена пар без суффиксов. Скрипт сам подберёт суффикс брокера.
BASE_SYMBOLS = [
    "USDJPY",   # мажор, классическая трендовая пара
    "GBPUSD",   # мажор, волатильный
    "AUDUSD",   # мажор, товарная валюта
    "USDCAD",   # мажор, товарная валюта
    "EURJPY",   # кросс йены, исторически трендовый
    "GBPJPY",   # кросс йены, "виджет" — большие диапазоны
    "XAUUSD",   # золото — оставляем для сравнения
]

# Возможные суффиксы у разных брокеров.
SUFFIX_CANDIDATES = ["rfd", "", ".", "_", "m", ".a", ".pro"]

TIMEFRAMES = {
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

DATE_FROM = datetime(2014, 1, 1)
DATE_TO   = datetime.now()

OUTPUT_DIR = "./data/fx"
# ===============================


def init_mt5():
    if not mt5.initialize():
        print(f"ERROR: не подключиться к MT5: {mt5.last_error()}")
        print("Запусти MT5 терминал и убедись что залогинен.")
        sys.exit(1)

    acc = mt5.account_info()
    term = mt5.terminal_info()
    is_demo = "demo" in (acc.server or "").lower() if acc else False
    print("=" * 60)
    print(f"Терминал:    {term.company if term else '?'}")
    print(f"Счёт:        {acc.login} ({acc.server})  [{'DEMO' if is_demo else 'REAL'}]")
    print(f"Баланс:      {acc.balance} {acc.currency}")
    print("=" * 60)

    if not is_demo:
        print("\n⚠️  ВНИМАНИЕ: ты не на demo-счёте!")
        print("   Сейчас будем только читать данные — это безопасно.")
        print("   Но для будущей разработки переключись на demo.")
        resp = input("Продолжить чтение данных с REAL-счёта? (yes/no): ").strip().lower()
        if resp != "yes":
            mt5.shutdown()
            sys.exit(0)


def find_symbol(base):
    """Ищет правильное имя символа у текущего брокера."""
    for suf in SUFFIX_CANDIDATES:
        candidate = base + suf
        info = mt5.symbol_info(candidate)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(candidate, True)
            return candidate
    return None


def download(symbol, timeframe, tf_name):
    """Качает историю по годовым чанкам и собирает в один DataFrame."""
    chunks = []
    current = DATE_FROM
    chunk_size = timedelta(days=365)

    while current < DATE_TO:
        chunk_end = min(current + chunk_size, DATE_TO)
        rates = mt5.copy_rates_range(symbol, timeframe, current, chunk_end)
        if rates is not None and len(rates) > 0:
            chunks.append(pd.DataFrame(rates))
        current = chunk_end

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["timestamps"] = pd.to_datetime(df["time"], unit="s")

    out = df[["timestamps", "open", "high", "low", "close", "tick_volume", "real_volume"]].copy()
    out.rename(columns={"tick_volume": "volume", "real_volume": "amount"}, inplace=True)
    # У FX/CFD real_volume = 0 — заменяем
    if (out["amount"] == 0).all():
        out["amount"] = out["volume"] * out["close"]
    return out


def main():
    init_mt5()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nИщу символы у брокера...")
    resolved = {}
    for base in BASE_SYMBOLS:
        sym = find_symbol(base)
        if sym:
            info = mt5.symbol_info(sym)
            spread_pts = info.spread
            print(f"  {base:8} -> {sym:14}  spread={spread_pts} points  digits={info.digits}")
            resolved[base] = sym
        else:
            print(f"  {base:8} -> НЕ НАЙДЕН")

    if not resolved:
        print("Ни один символ не найден. Проверь Market Watch.")
        mt5.shutdown()
        sys.exit(1)

    print(f"\nКачаю историю {DATE_FROM.date()} -> {DATE_TO.date()}...")
    print()

    summary = []
    for base, sym in resolved.items():
        for tf_name, tf_const in TIMEFRAMES.items():
            df = download(sym, tf_const, tf_name)
            if df is None or len(df) == 0:
                print(f"  {base:8} {tf_name}: пусто")
                continue

            fname = f"{base.lower()}_{tf_name.lower()}.csv"
            path = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path, index=False)

            print(f"  {base:8} {tf_name}: {len(df):>6} баров  "
                  f"[{df['timestamps'].iloc[0].date()} -> {df['timestamps'].iloc[-1].date()}]  "
                  f"-> {fname}")
            summary.append({
                "symbol": base,
                "broker_symbol": sym,
                "timeframe": tf_name,
                "bars": len(df),
                "start": df["timestamps"].iloc[0],
                "end": df["timestamps"].iloc[-1],
                "file": fname,
            })

    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "_summary.csv"), index=False)
    print(f"\nСводка: {os.path.join(OUTPUT_DIR, '_summary.csv')}")
    print(f"Всего файлов: {len(summary)}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
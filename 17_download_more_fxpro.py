"""
17_download_more_fxpro.py
==========================
Скачивает GBP/JPY и XAU/USD с FxPro в M15 и H1.
Чтобы прогнать те же тесты edge discovery что и на EUR/USD.

Запуск:
    python 17_download_more_fxpro.py
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


OUTPUT_DIR = "./data/fxpro"
DATE_FROM = datetime(2014, 1, 1)
DATE_TO = datetime.now()

SYMBOLS = ["GBPJPY", "XAUUSD"]
SYMBOL_FALLBACKS = {
    "GBPJPY": ["GBPJPY", "GBP/JPY", "GBPJPYm"],
    "XAUUSD": ["XAUUSD", "GOLD", "XAU/USD", "XAUUSDm"],
}

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

CHUNK_DAYS = {"M15": 180, "H1": 365}


def init_mt5():
    if not mt5.initialize():
        print(f"ERROR: MT5 не подключился: {mt5.last_error()}")
        sys.exit(1)
    acc = mt5.account_info()
    term = mt5.terminal_info()
    is_demo = "demo" in (acc.server or "").lower() if acc else False
    print(f"Терминал: {term.company if term else '?'}")
    print(f"Сервер:   {acc.server if acc else '?'}  [{'DEMO' if is_demo else 'REAL'}]")
    print()


def find_symbol(base):
    for c in SYMBOL_FALLBACKS.get(base, [base]):
        info = mt5.symbol_info(c)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(c, True)
            return c
    return None


def download_chunked(symbol, tf_const, tf_name):
    chunks = []
    current = DATE_FROM
    chunk_delta = timedelta(days=CHUNK_DAYS[tf_name])

    while current < DATE_TO:
        chunk_end = min(current + chunk_delta, DATE_TO)
        rates = mt5.copy_rates_range(symbol, tf_const, current, chunk_end)
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
    if (out["amount"] == 0).all():
        out["amount"] = out["volume"] * out["close"]
    return out


def main():
    init_mt5()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Проверяю символы...")
    resolved = {}
    for base in SYMBOLS:
        sym = find_symbol(base)
        if sym:
            info = mt5.symbol_info(sym)
            print(f"  {base} -> {sym}  spread={info.spread} points, digits={info.digits}")
            resolved[base] = sym
        else:
            print(f"  {base} -> НЕ НАЙДЕН")
    print()

    for base, sym in resolved.items():
        for tf_name, tf_const in TIMEFRAMES.items():
            df = download_chunked(sym, tf_const, tf_name)
            if df is None or len(df) == 0:
                print(f"  {base} {tf_name}: пусто")
                continue
            fname = f"{base.lower()}_{tf_name.lower()}.csv"
            path = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path, index=False)
            years = (df['timestamps'].iloc[-1] - df['timestamps'].iloc[0]).days / 365.25
            print(f"  {base} {tf_name}: {len(df):>7} баров, {years:.1f} лет -> {fname}")

    mt5.shutdown()
    print("\nГотово.")


if __name__ == "__main__":
    main()
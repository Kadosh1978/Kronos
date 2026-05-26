"""
01_download_xauusd_history.py
=============================
Скачивает историю XAU/USD H1 из MetaTrader 5 (Альфа-Форекс)
и сохраняет в CSV в формате, пригодном для Kronos.

Требования:
- Python 3.10 или 3.11
- pip install MetaTrader5 pandas
- MetaTrader 5 терминал установлен и ЗАПУЩЕН
- Авторизован в demo или реальный счёт Альфа-Форекс

Запуск:
    python 01_download_xauusd_history.py
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd 

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: пакет MetaTrader5 не установлен.")
    print("Поставь: pip install MetaTrader5")
    sys.exit(1)


# ========== Настройки ==========
# Возможные имена золота у разных брокеров. Скрипт переберёт по очереди.
# У Альфа-Форекса обычно "XAUUSD", но проверим всё.
SYMBOL_CANDIDATES = [
    "XAUUSDrfd", "XAU/USD", "XAU_USD", "GOLD",
    "XAUUSD.", "XAUUSD_", "XAUUSDm", "XAUUSD.a",
]

TIMEFRAME = mt5.TIMEFRAME_H1     # часовой таймфрейм
DATE_FROM = datetime(2015, 1, 1) # начало истории (поправь если у брокера глубина другая)
DATE_TO   = datetime.now()       # до текущего момента

OUTPUT_DIR  = "./data"
OUTPUT_FILE = "xauusd_h1.csv"


def init_mt5():
    """Подключается к запущенному MT5 терминалу."""
    if not mt5.initialize():
        err = mt5.last_error()
        print(f"ERROR: не удалось подключиться к MT5: {err}")
        print("Проверь: терминал запущен? Залогинен в аккаунт?")
        sys.exit(1)

    term = mt5.terminal_info()
    acc = mt5.account_info()
    print("=" * 60)
    print(f"MT5 терминал: {term.name if term else 'unknown'}")
    print(f"Компания:     {term.company if term else 'unknown'}")
    if acc:
        print(f"Счёт:         {acc.login} ({acc.server})")
        print(f"Тип:          {'DEMO' if 'demo' in (acc.server or '').lower() else 'REAL/UNKNOWN'}")
        print(f"Валюта:       {acc.currency}, баланс: {acc.balance}")
    else:
        print("WARNING: не залогинен ни в один счёт")
    print("=" * 60)


def find_symbol():
    """Подбирает правильное имя символа золота у текущего брокера."""
    for s in SYMBOL_CANDIDATES:
        info = mt5.symbol_info(s)
        if info is not None:
            # Активируем в Market Watch если не виден
            if not info.visible:
                mt5.symbol_select(s, True)
                info = mt5.symbol_info(s)
            print(f"\nНайден символ: {s}")
            print(f"  Описание:        {info.description}")
            print(f"  Спред (points):  {info.spread}")
            print(f"  Digits:          {info.digits}")
            print(f"  Контракт:        {info.trade_contract_size}")
            print(f"  Мин. лот:        {info.volume_min}")
            return s

    # Не нашли стандартных имён — ищем по подстроке
    print("\nСтандартные имена не подошли. Ищу по подстроке XAU/GOLD...")
    all_symbols = mt5.symbols_get()
    matches = [
        s.name for s in all_symbols
        if "XAU" in s.name.upper() or "GOLD" in s.name.upper()
    ]
    if matches:
        print("Найдены кандидаты в Market Watch:")
        for m in matches:
            print(f"  - {m}")
        print("\nДобавь правильное имя в SYMBOL_CANDIDATES в начале скрипта и запусти снова.")
    else:
        print("Золото у брокера не найдено вообще.")
        print("Открой MT5 -> правая панель Market Watch -> правый клик -> 'Show All'.")
    mt5.shutdown()
    sys.exit(1)


def download_history(symbol):
    """
    Качает историю по годовым чанкам.
    У MT5 есть лимит на размер ответа, поэтому идём кусками.
    """
    chunks = []
    current = DATE_FROM
    chunk_size = timedelta(days=365)

    print(f"\nКачаю {symbol} {DATE_FROM.date()} -> {DATE_TO.date()}...")
    while current < DATE_TO:
        chunk_end = min(current + chunk_size, DATE_TO)
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, current, chunk_end)

        if rates is None or len(rates) == 0:
            print(f"  {current.date()} -> {chunk_end.date()}: пусто")
        else:
            df_chunk = pd.DataFrame(rates)
            chunks.append(df_chunk)
            print(f"  {current.date()} -> {chunk_end.date()}: {len(df_chunk):>6} баров")

        current = chunk_end

    if not chunks:
        print("\nERROR: история пустая. Возможно у брокера нет такой глубины — попробуй позже стартовую дату.")
        mt5.shutdown()
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Приводим к формату, который ест Kronos: timestamps, open, high, low, close, volume, amount
    df["timestamps"] = pd.to_datetime(df["time"], unit="s")
    df_out = df[["timestamps", "open", "high", "low", "close", "tick_volume", "real_volume"]].copy()
    df_out.rename(columns={"tick_volume": "volume", "real_volume": "amount"}, inplace=True)

    # У FX/CFD real_volume обычно нули — заполняем аппроксимацией
    if (df_out["amount"] == 0).all():
        print("\nINFO: real_volume = 0 на всём датасете (норма для FX/CFD у Альфа-Форекса).")
        print("      Заполняю amount = volume * close для совместимости с Kronos.")
        df_out["amount"] = df_out["volume"] * df_out["close"]

    return df_out


def main():
    init_mt5()
    symbol = find_symbol()
    df = download_history(symbol)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"Файл:    {out_path}")
    print(f"Период:  {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
    print(f"Баров:   {len(df)}")
    print(f"\nПервые 3 строки:")
    print(df.head(3).to_string())
    print(f"\nПоследние 3 строки:")
    print(df.tail(3).to_string())
    print(f"\nСтатистика по close:")
    print(df["close"].describe().to_string())

    mt5.shutdown()


if __name__ == "__main__":
    main()
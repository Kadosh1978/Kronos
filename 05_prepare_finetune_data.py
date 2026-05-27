"""
05_prepare_finetune_data.py
============================
Готовит данные для файнтюна Kronos:
1. Объединяет xauusd_h1_train.csv + xauusd_h1_val.csv в один файл
2. Переупорядочивает колонки в порядок: open, close, high, low, volume, amount
   (как в их эталоне HK_ali_09988_kline_5min_all.csv)
3. Сохраняет в data/xauusd_h1_finetune.csv

TEST не трогаем — он останется только для финальной оценки после файнтюна.

Запуск:
    python 05_prepare_finetune_data.py
"""

import os
import pandas as pd

TRAIN_FILE = "./data/xauusd_h1_train.csv"
VAL_FILE   = "./data/xauusd_h1_val.csv"
OUTPUT_FILE = "./data/xauusd_h1_finetune.csv"


def main():
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE):
        raise FileNotFoundError("Файлы train/val не найдены. Запусти сначала 02_eda_and_split.py")

    df_train = pd.read_csv(TRAIN_FILE, parse_dates=["timestamps"])
    df_val   = pd.read_csv(VAL_FILE,   parse_dates=["timestamps"])

    print(f"Train: {len(df_train)} строк, {df_train['timestamps'].min()} -> {df_train['timestamps'].max()}")
    print(f"Val:   {len(df_val)} строк, {df_val['timestamps'].min()} -> {df_val['timestamps'].max()}")

    # Объединяем по времени (train идёт первым, val после него — это walk-forward)
    df = pd.concat([df_train, df_val], ignore_index=True)
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Переупорядочиваем колонки в формат Kronos finetune_csv:
    # timestamps, open, close, high, low, volume, amount
    df = df[["timestamps", "open", "close", "high", "low", "volume", "amount"]]

    # Сохраняем
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nОбъединённый файл: {OUTPUT_FILE}")
    print(f"Итого: {len(df)} строк")
    print(f"Период: {df['timestamps'].min()} -> {df['timestamps'].max()}")
    print(f"\nПервые 3 строки:")
    print(df.head(3).to_string(index=False))
    print(f"\nПоследние 3 строки:")
    print(df.tail(3).to_string(index=False))

    print(f"\nЭтот файл будет поделён 0.85/0.15 для train/val при файнтюне.")
    print(f"TEST остаётся в data/xauusd_h1_test.csv и НЕ участвует в обучении.")


if __name__ == "__main__":
    main()
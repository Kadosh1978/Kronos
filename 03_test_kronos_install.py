"""
03_test_kronos_install.py
=========================
Проверяет что Kronos корректно установлен и работает на твоём железе.

Что делает:
1. Проверяет CUDA / GPU
2. Загружает Kronos-small с HuggingFace (один раз скачает в кэш)
3. Берёт одно окно из TEST: контекст 512 баров + следующие 24 бара
4. Делает предсказание pretrained моделью (zero-shot, БЕЗ файнтюна)
5. Сравнивает с реальными значениями
6. Замеряет скорость inference
7. Рисует график (если есть matplotlib)

Запуск:
    python 03_test_kronos_install.py

Перед запуском:
- pip install torch с CUDA (см. инструкции в чате)
- pip install transformers huggingface_hub einops safetensors
- git clone https://github.com/shiyu-coder/Kronos.git Kronos_repo
- pip install -r Kronos_repo/requirements.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ========== Настройки ==========
KRONOS_REPO_PATH = "./Kronos_repo"  # путь к клону репо Kronos
TEST_FILE        = "./data/xauusd_h1_test.csv"

# Размеры окон (стандарт Kronos)
CONTEXT_LENGTH    = 512   # сколько баров истории даём модели
PREDICTION_LENGTH = 24    # на сколько баров вперёд предсказываем (24ч = 1 сутки)

# С какого момента в TEST брать пример (0 = с начала, негативное = с конца)
START_OFFSET = 5000       # середина TEST для разнообразия

# Параметры sampling'a (см. документацию Kronos)
T            = 1.0        # temperature
TOP_P        = 0.9        # nucleus sampling
SAMPLE_COUNT = 1          # сколько траекторий генерировать (1 для теста скорости)

DEVICE = "cuda"           # "cuda" или "cpu" (будет проверено)
# ===============================


def check_environment():
    print("=" * 60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch:    {torch.__version__}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA версия:   {torch.version.cuda}")
            print(f"GPU:           {torch.cuda.get_device_name(0)}")
            print(f"VRAM total:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            global DEVICE
            DEVICE = "cuda"
        else:
            print("WARNING: CUDA недоступна — будем работать на CPU (медленнее в ~20-50 раз).")
            DEVICE = "cpu"
    except ImportError:
        print("ERROR: PyTorch не установлен.")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    if not os.path.isdir(KRONOS_REPO_PATH):
        print(f"\nERROR: папка {KRONOS_REPO_PATH} не найдена.")
        print(f"       git clone https://github.com/shiyu-coder/Kronos.git Kronos_repo")
        sys.exit(1)

    # Добавляем репо в путь чтобы импортировать model/
    sys.path.insert(0, os.path.abspath(KRONOS_REPO_PATH))

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print(f"\nKronos импортирован из: {KRONOS_REPO_PATH}")
    except ImportError as e:
        print(f"\nERROR: не удалось импортировать Kronos: {e}")
        print(f"       Проверь что в {KRONOS_REPO_PATH}/model/ есть нужные файлы.")
        sys.exit(1)

    return Kronos, KronosTokenizer, KronosPredictor


def load_model(Kronos, KronosTokenizer, KronosPredictor):
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА МОДЕЛИ (первый раз — скачает ~100MB с HuggingFace)")
    print("=" * 60)

    t0 = time.time()
    print("Загружаю токенизатор: NeoQuasar/Kronos-Tokenizer-base ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    print("Загружаю модель:      NeoQuasar/Kronos-small ...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_context=CONTEXT_LENGTH,
    )
    print(f"Загрузка заняла: {time.time()-t0:.1f} сек")

    # Проверим память на GPU
    if DEVICE == "cuda":
        import torch
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM использовано после загрузки: {used:.2f} GB")

    return predictor


def prepare_window():
    """Берёт одно окно из TEST: 512 баров контекста + 24 бара истины."""
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ОКНА ИЗ TEST")
    print("=" * 60)

    df = pd.read_csv(TEST_FILE, parse_dates=["timestamps"])
    print(f"Загружено {len(df)} баров из {TEST_FILE}")

    start = START_OFFSET
    end_context = start + CONTEXT_LENGTH
    end_pred    = end_context + PREDICTION_LENGTH

    if end_pred > len(df):
        print(f"ERROR: START_OFFSET={START_OFFSET} слишком большой, выходит за границы.")
        sys.exit(1)

    context_df = df.iloc[start:end_context].reset_index(drop=True)
    truth_df   = df.iloc[end_context:end_pred].reset_index(drop=True)

    print(f"Контекст: {context_df['timestamps'].iloc[0]} -> {context_df['timestamps'].iloc[-1]} ({len(context_df)} баров)")
    print(f"Истина:   {truth_df['timestamps'].iloc[0]} -> {truth_df['timestamps'].iloc[-1]} ({len(truth_df)} баров)")
    print(f"Последняя цена контекста: {context_df['close'].iloc[-1]:.2f}")
    print(f"Истинная цена через 24ч:  {truth_df['close'].iloc[-1]:.2f}")
    print(f"Истинное движение:        {(truth_df['close'].iloc[-1]/context_df['close'].iloc[-1]-1)*100:+.2f}%")

    return context_df, truth_df


def predict(predictor, context_df, truth_df):
    print("\n" + "=" * 60)
    print("ИНФЕРЕНС (zero-shot, без файнтюна)")
    print("=" * 60)

    # Kronos ожидает на вход DataFrame с колонками open,high,low,close,volume,amount
    # и timestamps отдельно
    x_cols = ["open", "high", "low", "close", "volume", "amount"]
    x_df = context_df[x_cols]
    x_timestamps    = context_df["timestamps"]
    pred_timestamps = truth_df["timestamps"]

    print(f"Запускаю inference на {DEVICE}...")
    t0 = time.time()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamps,
        y_timestamp=pred_timestamps,
        pred_len=PREDICTION_LENGTH,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"Inference занял: {elapsed:.2f} сек")
    print(f"Скорость: {PREDICTION_LENGTH/elapsed:.1f} баров/сек")

    return pred_df, elapsed


def evaluate(context_df, truth_df, pred_df):
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)

    last_price = context_df["close"].iloc[-1]
    true_close = truth_df["close"].values
    pred_close = pred_df["close"].values

    # MAPE по close
    mape = np.mean(np.abs((pred_close - true_close) / true_close)) * 100

    # Directional accuracy по часовым изменениям
    true_dirs = np.sign(np.diff(np.insert(true_close, 0, last_price)))
    pred_dirs = np.sign(np.diff(np.insert(pred_close, 0, last_price)))
    dir_acc = (true_dirs == pred_dirs).mean() * 100

    # Финальное направление (за весь горизонт)
    true_final_dir = np.sign(true_close[-1] - last_price)
    pred_final_dir = np.sign(pred_close[-1] - last_price)
    final_correct = "✓ ВЕРНО" if true_final_dir == pred_final_dir else "✗ НЕВЕРНО"

    print(f"MAPE по close:                  {mape:.3f}%")
    print(f"Directional accuracy (часовая): {dir_acc:.1f}%")
    print(f"Направление за {PREDICTION_LENGTH}ч:")
    print(f"  истина:   {(true_close[-1]/last_price-1)*100:+.2f}%")
    print(f"  прогноз:  {(pred_close[-1]/last_price-1)*100:+.2f}%")
    print(f"  {final_correct}")

    print(f"\nВнимание: ЭТО ОДНО ОКНО. Не делай выводов о модели по нему.")
    print(f"Цель этого скрипта — проверить что технически всё работает.")

    return mape, dir_acc


def make_plot(context_df, truth_df, pred_df):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Показываем последние 100 баров контекста для читаемости
    ctx_tail = context_df.tail(100)
    ax.plot(ctx_tail["timestamps"], ctx_tail["close"], label="Контекст (close)", color="black", lw=1)
    ax.plot(truth_df["timestamps"], truth_df["close"], label="Истина", color="green", lw=2)
    ax.plot(truth_df["timestamps"], pred_df["close"].values, label="Прогноз Kronos", color="red", lw=2, ls="--")

    ax.axvline(ctx_tail["timestamps"].iloc[-1], color="gray", ls=":", alpha=0.5)
    ax.set_title(f"Kronos zero-shot prediction (горизонт {PREDICTION_LENGTH}ч)")
    ax.set_ylabel("XAU/USD")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = "./data/test_inference.png"
    plt.savefig(out, dpi=100)
    print(f"\nГрафик сохранён: {out}")
    plt.close()


def main():
    Kronos, KronosTokenizer, KronosPredictor = check_environment()
    predictor = load_model(Kronos, KronosTokenizer, KronosPredictor)
    context_df, truth_df = prepare_window()
    pred_df, elapsed = predict(predictor, context_df, truth_df)
    evaluate(context_df, truth_df, pred_df)
    make_plot(context_df, truth_df, pred_df)

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"Время одного предсказания: {elapsed:.2f} сек")
    print(f"Если хочешь оценить всю TEST-выборку ({10051 // PREDICTION_LENGTH} окон),")
    print(f"это займёт примерно {elapsed * 10051 // PREDICTION_LENGTH / 60:.1f} минут.")


if __name__ == "__main__":
    main()
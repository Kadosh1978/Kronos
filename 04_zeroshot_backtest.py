"""
04_zeroshot_backtest.py
========================
Полный zero-shot бэктест pretrained Kronos-small на TEST-выборке.

Цель: получить baseline качества модели БЕЗ файнтюна.
Это потом будем сравнивать с метриками после файнтюна.

Что делает:
1. Скользящее окно по всему TEST файлу
2. Для каждого окна: контекст 512 баров -> прогноз 24 бара (sample_count траекторий)
3. Считает метрики на разных горизонтах (1ч, 6ч, 12ч, 24ч)
4. Сравнивает с naive baseline (цена не меняется)
5. Сохраняет все предсказания в CSV для дальнейшего анализа

Запуск:
    python 04_zeroshot_backtest.py

Время: ~8 мин при SAMPLE_COUNT=1, ~80 мин при SAMPLE_COUNT=10, ~2.5ч при 20.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ========== Настройки ==========
KRONOS_REPO_PATH = "./Kronos_repo"
TEST_FILE        = "./data/xauusd_h1_test.csv"
OUTPUT_DIR       = "./data"

CONTEXT_LENGTH    = 512   # сколько баров истории даём модели
PREDICTION_LENGTH = 24    # на сколько баров вперёд предсказываем
STEP              = 24    # шаг между окнами (24 = непересекающиеся окна)

# sample_count — сколько траекторий генерировать на окно.
# 1  — быстро (8 мин), но без оценки распределения
# 10 — хорошо (80 мин), даёт нормальную оценку среднего
# 20 — лучше (2.5ч), оценка квантилей
# Начни с 1 для проверки скрипта, потом перезапусти с 10.
SAMPLE_COUNT = 1

T     = 1.0   # temperature
TOP_P = 0.9   # nucleus sampling

# Ограничение количества окон (None = все).
# Для отладки поставь 20 — будет 30 сек.
MAX_WINDOWS = None

# Сохранять прогресс каждые N окон (на случай прерывания)
CHECKPOINT_EVERY = 50

DEVICE = "cuda"
# ===============================


def setup():
    if not os.path.isdir(KRONOS_REPO_PATH):
        print(f"ERROR: {KRONOS_REPO_PATH} не найдена.")
        sys.exit(1)
    sys.path.insert(0, os.path.abspath(KRONOS_REPO_PATH))

    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA недоступна, переключаюсь на CPU.")
        global DEVICE
        DEVICE = "cpu"

    from model import Kronos, KronosTokenizer, KronosPredictor

    print("Загружаю модель...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=DEVICE, max_context=CONTEXT_LENGTH)
    print(f"Модель загружена на {DEVICE}.")
    return predictor


def build_windows(df):
    """Строит список индексов окон (start, end_context, end_pred)."""
    n = len(df)
    windows = []
    start = 0
    while start + CONTEXT_LENGTH + PREDICTION_LENGTH <= n:
        windows.append((start, start + CONTEXT_LENGTH, start + CONTEXT_LENGTH + PREDICTION_LENGTH))
        start += STEP

    if MAX_WINDOWS is not None:
        windows = windows[:MAX_WINDOWS]
    return windows


def run_backtest(predictor, df, windows):
    x_cols = ["open", "high", "low", "close", "volume", "amount"]

    results = []   # список dict с метаданными окна и предсказаниями
    truths  = []   # список истинных close
    preds   = []   # список предсказанных close (mean по sample_count)

    t_start = time.time()

    for i, (s, ec, ep) in enumerate(windows):
        ctx = df.iloc[s:ec]
        truth = df.iloc[ec:ep]

        try:
            pred_df = predictor.predict(
                df=ctx[x_cols],
                x_timestamp=ctx["timestamps"],
                y_timestamp=truth["timestamps"],
                pred_len=PREDICTION_LENGTH,
                T=T,
                top_p=TOP_P,
                sample_count=SAMPLE_COUNT,
                verbose=False,
            )
        except Exception as e:
            print(f"\n[window {i}] ОШИБКА: {e}")
            continue

        # pred_df содержит средние значения по sample_count — это то что отдаёт KronosPredictor
        truths.append(truth["close"].values)
        preds.append(pred_df["close"].values)

        results.append({
            "window_idx": i,
            "ctx_start":  ctx["timestamps"].iloc[0],
            "ctx_end":    ctx["timestamps"].iloc[-1],
            "pred_start": truth["timestamps"].iloc[0],
            "pred_end":   truth["timestamps"].iloc[-1],
            "last_close": ctx["close"].iloc[-1],
            "true_close_final": truth["close"].iloc[-1],
            "pred_close_final": pred_df["close"].iloc[-1],
        })

        # Прогресс
        elapsed = time.time() - t_start
        done = i + 1
        total = len(windows)
        eta = elapsed / done * (total - done)
        if done % 10 == 0 or done == total:
            print(f"  [{done:>4}/{total}] elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                  f"last_pred={pred_df['close'].iloc[-1]:.2f}  true={truth['close'].iloc[-1]:.2f}")

        # Чекпоинт
        if done % CHECKPOINT_EVERY == 0:
            save_checkpoint(results, truths, preds)

    return results, np.array(truths), np.array(preds)


def save_checkpoint(results, truths, preds):
    """Сохраняет промежуточный прогресс — на случай если прервётся."""
    path = os.path.join(OUTPUT_DIR, "backtest_checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump({"results": results, "truths": truths, "preds": preds}, f)


def compute_metrics(df_test, results, truths, preds):
    """
    Считает метрики на разных горизонтах.
    truths и preds имеют форму [n_windows, PREDICTION_LENGTH].
    """
    print("\n" + "=" * 70)
    print("МЕТРИКИ ZERO-SHOT BASELINE")
    print("=" * 70)

    n_windows = len(truths)
    print(f"Окон оценено: {n_windows}")
    print(f"Период:       {results[0]['pred_start']} -> {results[-1]['pred_end']}")

    # Последние цены контекста для расчёта направлений
    last_prices = np.array([r["last_close"] for r in results])  # [n_windows]

    # Naive baseline: предсказать "цена не изменится"
    # т.е. предсказание = последняя цена контекста
    naive = np.tile(last_prices[:, None], (1, PREDICTION_LENGTH))  # [n_windows, 24]

    print(f"\n{'Горизонт':>10} | {'MAPE Kronos':>12} | {'MAPE Naive':>11} | "
          f"{'DirAcc Kronos':>14} | {'DirAcc Naive':>13} | {'Лучше naive?':>14}")
    print("-" * 95)

    horizons = [1, 3, 6, 12, 24]
    for h in horizons:
        if h > PREDICTION_LENGTH:
            continue
        idx = h - 1  # индекс h-го бара

        # MAPE на горизонте h
        mape_kr   = np.mean(np.abs((preds[:, idx] - truths[:, idx]) / truths[:, idx])) * 100
        mape_naive = np.mean(np.abs((naive[:, idx] - truths[:, idx]) / truths[:, idx])) * 100

        # Directional accuracy на горизонте h
        # направление = знак (цена_через_h - последняя_цена_контекста)
        true_dir = np.sign(truths[:, idx] - last_prices)
        pred_dir = np.sign(preds[:, idx]  - last_prices)
        # naive direction всегда 0 (не меняется) — это дегенеративный случай.
        # Для честности будем считать naive_dir = 0 как "правильное"
        # только когда true_dir тоже 0 (т.е. флэт). Поскольку флэта почти не бывает,
        # naive directional accuracy ~ 0%. Поэтому показываем DirAcc только для Kronos.
        dir_acc_kr = (true_dir == pred_dir).mean() * 100
        # для naive показываем долю флэт-баров
        naive_dir_acc = (true_dir == 0).mean() * 100

        better = "ДА" if (dir_acc_kr > 50 and mape_kr < mape_naive) else "нет"

        print(f"{h:>9}ч | {mape_kr:>11.4f}% | {mape_naive:>10.4f}% | "
              f"{dir_acc_kr:>13.1f}% | {naive_dir_acc:>12.1f}% | {better:>14}")

    # Дополнительная статистика: насколько модель уверенная
    print(f"\nДополнительная диагностика:")
    pred_changes_24h = (preds[:, -1] - last_prices) / last_prices * 100
    true_changes_24h = (truths[:, -1] - last_prices) / last_prices * 100
    print(f"  Истинное движение за 24ч:  std = {true_changes_24h.std():.3f}%, "
          f"mean = {true_changes_24h.mean():+.3f}%")
    print(f"  Прогноз движения за 24ч:   std = {pred_changes_24h.std():.3f}%, "
          f"mean = {pred_changes_24h.mean():+.3f}%")
    print(f"  Корреляция прогноза и истины (24ч): {np.corrcoef(pred_changes_24h, true_changes_24h)[0,1]:+.4f}")

    print(f"\nКлюч к интерпретации:")
    print(f"  - MAPE Kronos должен быть НИЖЕ MAPE Naive — иначе модель хуже ничего")
    print(f"  - DirAcc > 50%  означает edge в направлении")
    print(f"  - DirAcc 52-55% — слабый сигнал, после файнтюна может улучшиться")
    print(f"  - DirAcc < 50% — модель систематически ошибается (zero-shot не работает на FX/золоте)")
    print(f"  - Корреляция движений > 0.05 — есть слабый сигнал; > 0.15 — хороший")


def save_predictions(results, truths, preds):
    """Сохраняет полные предсказания для последующего анализа."""
    rows = []
    for i, r in enumerate(results):
        for h in range(PREDICTION_LENGTH):
            rows.append({
                "window_idx": i,
                "ctx_end_ts": r["ctx_end"],
                "horizon": h + 1,
                "true_close": float(truths[i, h]),
                "pred_close": float(preds[i, h]),
                "last_close": r["last_close"],
            })
    df_out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "zeroshot_predictions.csv")
    df_out.to_csv(path, index=False)
    print(f"\nПолные предсказания сохранены: {path}  ({len(df_out)} строк)")


def make_plot(results, truths, preds):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(results)
    last_prices = np.array([r["last_close"] for r in results])
    pred_24h = (preds[:, -1] - last_prices) / last_prices * 100
    true_24h = (truths[:, -1] - last_prices) / last_prices * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # 1. Scatter: прогноз vs истина (24ч change %)
    ax = axes[0]
    ax.scatter(true_24h, pred_24h, alpha=0.4, s=15)
    lim = max(abs(true_24h).max(), abs(pred_24h).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="идеал (y=x)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Истинное движение за 24ч, %")
    ax.set_ylabel("Прогноз Kronos движения за 24ч, %")
    ax.set_title(f"Прогноз vs истина (24ч). Корр = {np.corrcoef(pred_24h, true_24h)[0,1]:+.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Временной ряд: истинная цена и средний прогноз за 24ч
    ax = axes[1]
    times = [r["pred_end"] for r in results]
    ax.plot(times, [r["true_close_final"] for r in results], label="Истина (close через 24ч)", color="green", lw=1)
    ax.plot(times, [r["pred_close_final"] for r in results], label="Прогноз Kronos", color="red", lw=1, alpha=0.7)
    ax.set_ylabel("XAU/USD")
    ax.set_title("Истина vs прогноз во времени (горизонт 24ч)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "zeroshot_backtest.png")
    plt.savefig(out, dpi=100)
    print(f"График: {out}")
    plt.close()


def main():
    df = pd.read_csv(TEST_FILE, parse_dates=["timestamps"])
    print(f"Загружено {len(df)} баров из {TEST_FILE}")

    predictor = setup()

    windows = build_windows(df)
    print(f"\nОкон для оценки: {len(windows)}")
    print(f"Параметры: CONTEXT={CONTEXT_LENGTH}, PRED={PREDICTION_LENGTH}, "
          f"STEP={STEP}, SAMPLE_COUNT={SAMPLE_COUNT}")
    print(f"Ожидаемое время: ~{len(windows) * 1.14 * SAMPLE_COUNT / 60:.1f} минут")
    print("-" * 70)

    results, truths, preds = run_backtest(predictor, df, windows)

    if len(results) == 0:
        print("ERROR: ни одного окна не получилось обсчитать.")
        sys.exit(1)

    compute_metrics(df, results, truths, preds)
    save_predictions(results, truths, preds)
    make_plot(results, truths, preds)

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()
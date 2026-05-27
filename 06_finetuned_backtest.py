"""
06_finetuned_backtest.py
=========================
Бэктест файнтюненой Kronos-модели на TEST-выборке.
Сравнивает с zero-shot baseline из 04_zeroshot_backtest.py.

Запуск:
    python 06_finetuned_backtest.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# ========== Настройки ==========
KRONOS_REPO_PATH = "./Kronos_repo"
TEST_FILE        = "./data/xauusd_h1_test.csv"
OUTPUT_DIR       = "./data"

# ПУТИ К ФАЙНТЮНЕНЫМ ВЕСАМ
FINETUNED_TOKENIZER = "C:/Users/Kadosh/Kronos/pretrained/Kronos-Tokenizer-base"
FINETUNED_MODEL     = "C:/Users/Kadosh/Kronos/finetuned/xauusd_h1_v2/basemodel/best_model"

CONTEXT_LENGTH    = 512
PREDICTION_LENGTH = 24
STEP              = 24
SAMPLE_COUNT      = 1
T                 = 1.0
TOP_P             = 0.9
MAX_WINDOWS       = None     # все ~397 окон
CHECKPOINT_EVERY  = 50
DEVICE            = "cuda"
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

    # Проверяем что файнтюненые веса существуют
    if not os.path.exists(FINETUNED_TOKENIZER):
        print(f"ERROR: токенайзер не найден: {FINETUNED_TOKENIZER}")
        sys.exit(1)
    if not os.path.exists(FINETUNED_MODEL):
        print(f"ERROR: предиктор не найден: {FINETUNED_MODEL}")
        sys.exit(1)

    print("Загружаю ФАЙНТЮНЕНЫЕ модели...")
    print(f"  Tokenizer: {FINETUNED_TOKENIZER}")
    print(f"  Model:     {FINETUNED_MODEL}")
    tokenizer = KronosTokenizer.from_pretrained(FINETUNED_TOKENIZER)
    model = Kronos.from_pretrained(FINETUNED_MODEL)
    predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=DEVICE, max_context=CONTEXT_LENGTH)
    print(f"Модель загружена на {DEVICE}.")
    return predictor


def build_windows(df):
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
    results, truths, preds = [], [], []
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
                T=T, top_p=TOP_P, sample_count=SAMPLE_COUNT,
                verbose=False,
            )
        except Exception as e:
            print(f"\n[window {i}] ОШИБКА: {e}")
            continue

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

        elapsed = time.time() - t_start
        done, total = i + 1, len(windows)
        eta = elapsed / done * (total - done)
        if done % 10 == 0 or done == total:
            print(f"  [{done:>4}/{total}] elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                  f"last_pred={pred_df['close'].iloc[-1]:.2f}  true={truth['close'].iloc[-1]:.2f}")

    return results, np.array(truths), np.array(preds)


def compute_metrics(results, truths, preds):
    print("\n" + "=" * 75)
    print("МЕТРИКИ ФАЙНТЮНЕНОЙ МОДЕЛИ (сравнение с zero-shot baseline)")
    print("=" * 75)
    print(f"Окон оценено: {len(truths)}")
    print(f"Период:       {results[0]['pred_start']} -> {results[-1]['pred_end']}")

    last_prices = np.array([r["last_close"] for r in results])
    naive = np.tile(last_prices[:, None], (1, PREDICTION_LENGTH))

    # baseline числа из 04_zeroshot_backtest.py
    BASELINE = {
        1:  {"mape": 0.3057, "diracc": 51.1},
        3:  {"mape": 0.5102, "diracc": 54.2},
        6:  {"mape": 0.6984, "diracc": 52.1},
        12: {"mape": 1.0216, "diracc": 52.4},
        24: {"mape": 1.8240, "diracc": 48.6},
    }

    print(f"\n{'Гор':>4} | {'MAPE FT':>9} | {'MAPE ZS':>9} | {'MAPE Naive':>11} | "
          f"{'DirAcc FT':>10} | {'DirAcc ZS':>10} | {'Δ DirAcc':>9} | {'Лучше ZS?':>10}")
    print("-" * 100)

    horizons = [1, 3, 6, 12, 24]
    for h in horizons:
        if h > PREDICTION_LENGTH:
            continue
        idx = h - 1
        mape_ft    = np.mean(np.abs((preds[:, idx] - truths[:, idx]) / truths[:, idx])) * 100
        mape_naive = np.mean(np.abs((naive[:, idx] - truths[:, idx]) / truths[:, idx])) * 100
        true_dir = np.sign(truths[:, idx] - last_prices)
        pred_dir = np.sign(preds[:, idx]  - last_prices)
        dir_acc_ft = (true_dir == pred_dir).mean() * 100

        bl_mape   = BASELINE[h]["mape"]
        bl_diracc = BASELINE[h]["diracc"]
        delta = dir_acc_ft - bl_diracc

        better = "ДА" if (dir_acc_ft > bl_diracc and mape_ft < bl_mape) else "нет"

        print(f"{h:>3}ч | {mape_ft:>8.4f}% | {bl_mape:>8.4f}% | {mape_naive:>10.4f}% | "
              f"{dir_acc_ft:>9.1f}% | {bl_diracc:>9.1f}% | {delta:>+8.1f} | {better:>10}")

    # Дополнительная диагностика
    pred_changes = (preds[:, -1] - last_prices) / last_prices * 100
    true_changes = (truths[:, -1] - last_prices) / last_prices * 100
    print(f"\nКалибровка волатильности (горизонт 24ч):")
    print(f"  Истинное движение:   std = {true_changes.std():.3f}%, mean = {true_changes.mean():+.3f}%")
    print(f"  Прогноз FT:          std = {pred_changes.std():.3f}%, mean = {pred_changes.mean():+.3f}%")
    print(f"  (Baseline ZS было:   std = 2.088%, mean = -0.429%)")
    print(f"  Корреляция FT-прогноза с истиной (24ч): {np.corrcoef(pred_changes, true_changes)[0,1]:+.4f}")
    print(f"  (Baseline ZS было:   +0.0166)")


def save_predictions(results, truths, preds):
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
    path = os.path.join(OUTPUT_DIR, "finetuned_predictions.csv")
    df_out.to_csv(path, index=False)
    print(f"\nПолные предсказания сохранены: {path}")


def make_plot(results, truths, preds):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    last_prices = np.array([r["last_close"] for r in results])
    pred_24h = (preds[:, -1] - last_prices) / last_prices * 100
    true_24h = (truths[:, -1] - last_prices) / last_prices * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    ax.scatter(true_24h, pred_24h, alpha=0.4, s=15)
    lim = max(abs(true_24h).max(), abs(pred_24h).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="идеал (y=x)")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Истинное движение за 24ч, %")
    ax.set_ylabel("Прогноз FT-модели, %")
    ax.set_title(f"FINETUNED: Прогноз vs истина (24ч). Корр = {np.corrcoef(pred_24h, true_24h)[0,1]:+.3f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    times = [r["pred_end"] for r in results]
    ax.plot(times, [r["true_close_final"] for r in results], label="Истина", color="green", lw=1)
    ax.plot(times, [r["pred_close_final"] for r in results], label="Прогноз FT", color="blue", lw=1, alpha=0.7)
    ax.set_ylabel("XAU/USD")
    ax.set_title("Истина vs прогноз файнтюненой модели (24ч)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "finetuned_backtest.png")
    plt.savefig(out, dpi=100)
    print(f"График: {out}")
    plt.close()


def main():
    df = pd.read_csv(TEST_FILE, parse_dates=["timestamps"])
    print(f"Загружено {len(df)} баров из {TEST_FILE}")

    predictor = setup()
    windows = build_windows(df)
    print(f"\nОкон для оценки: {len(windows)}")
    print(f"Ожидаемое время: ~{len(windows) * 1.0 / 60:.1f} минут\n")

    results, truths, preds = run_backtest(predictor, df, windows)
    if len(results) == 0:
        print("ERROR: ни одного окна.")
        sys.exit(1)

    compute_metrics(results, truths, preds)
    save_predictions(results, truths, preds)
    make_plot(results, truths, preds)
    print("\n" + "=" * 75)
    print("ГОТОВО")
    print("=" * 75)


if __name__ == "__main__":
    main()
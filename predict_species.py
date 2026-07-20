"""Predict penguin species with TabFM and report accuracy.

Repeatedly holds out a random subset of rows, fits TabFM (in-context) on the
remaining rows, predicts species for the held-out rows, and reports accuracy.

Usage:
    python predict_species.py N_REPEATS TEST_PERCENT [--n-estimators N]

    N_REPEATS     how many times to randomly select test rows (e.g. 5)
    TEST_PERCENT  percent of rows used as the test set each time (e.g. 20)
    --n-estimators  TabFM ensemble size; higher is slower but can be more
                    accurate (default: 8)
"""

import argparse
import os
import time

# Hide Hugging Face download/progress noise; must be set before
# huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# On Apple Silicon, run ops the MPS backend doesn't support on the CPU
# instead of erroring; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from pycm import ConfusionMatrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabfm import TabFMClassifier
from tabfm.src.pytorch import tabfm_v1_0_0

CSV_PATH = "penguins.csv"
TARGET = "species"
BASE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict penguin species with TabFM and report accuracy."
    )
    parser.add_argument(
        "n_repeats",
        type=int,
        help="how many times to randomly select test rows",
    )
    parser.add_argument(
        "test_percent",
        type=float,
        help="percent of rows to hold out as the test set (0-100)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="TabFM ensemble size; higher is slower but can be more "
             "accurate (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.n_repeats < 1:
        parser.error("N_REPEATS must be at least 1")
    if not 0 < args.test_percent < 100:
        parser.error("TEST_PERCENT must be between 0 and 100 (exclusive)")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1")
    return args


def pick_device() -> str:
    """Pick the best available device: NVIDIA CUDA, Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device):
    """Load the pretrained TabFM classification model.

    tabfm 1.0.0 looks for pytorch_model.bin, but the Hugging Face repo now
    ships model.safetensors instead, so fall back to loading that directly.
    """
    try:
        return tabfm_v1_0_0.load("classification", device=device)
    except FileNotFoundError:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from tabfm.src.pytorch.model import TabFM

        base = snapshot_download(repo_id=tabfm_v1_0_0.HF_REPO_ID)
        state_dict = load_file(
            os.path.join(base, "classification", "model.safetensors")
        )
        model = TabFM(**tabfm_v1_0_0.ClassificationConfig().to_dict())
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model


def main() -> None:
    args = parse_args()

    # "NA" strings in the CSV become NaN; TabFM handles missing values itself,
    # so rows with missing measurements are kept.
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Loaded {len(df)} rows, {X.shape[1]} features, "
          f"{y.nunique()} species: {', '.join(sorted(y.unique()))}")

    device = pick_device()
    print(f"Loading TabFM model on {device} (downloads weights on first run)...")
    model = load_model(device)

    actual_all = []
    predicted_all = []
    accuracies = []
    for i in range(args.n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_percent / 100, random_state=BASE_SEED + i
        )
        clf = TabFMClassifier(
            model=model,
            n_estimators=args.n_estimators,
            random_state=BASE_SEED + i,
        )
        start = time.time()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        actual_all.extend(y_test.tolist())
        predicted_all.extend(preds.tolist())
        print(f"Run {i + 1}/{args.n_repeats}: "
              f"train={len(X_train)} test={len(X_test)} "
              f"accuracy={acc:.4f} ({time.time() - start:.1f}s)")

    accuracies = np.array(accuracies)
    print(f"\nAccuracy over {args.n_repeats} run(s) "
          f"with {args.test_percent:g}% test rows: "
          f"mean={accuracies.mean():.4f} std={accuracies.std():.4f} "
          f"min={accuracies.min():.4f} max={accuracies.max():.4f}")

    print(f"\nConfusion matrix and statistics "
          f"(pooled over all {args.n_repeats} run(s)):\n")
    cm = ConfusionMatrix(actual_vector=actual_all, predict_vector=predicted_all)
    cm.print_matrix()
    cm.stat()


if __name__ == "__main__":
    main()

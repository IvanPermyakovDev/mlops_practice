#!/usr/bin/env python3
"""Download dataset from network and build train/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def save_arrays(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "X.npy", x)
    np.save(path / "y.npy", y)


def save_dataset_variants(path: Path, x: np.ndarray, y: np.ndarray, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    save_arrays(path, x, y)
    noisy = x + rng.normal(0.0, 0.2, size=x.shape).astype(np.float32)
    np.save(path / "X_noisy.npy", noisy)
    np.save(path / "y_noisy.npy", y)

    anomalous = x.copy()
    y_anomalous = y.copy()
    anomaly_count = max(1, len(x) // 20)
    idx = rng.choice(len(x), size=anomaly_count, replace=False)
    anomalous[idx] += rng.normal(0.0, 3.0, size=(anomaly_count, x.shape[1])).astype(np.float32)
    y_anomalous[idx] = 1 - y_anomalous[idx]
    np.save(path / "X_anomaly.npy", anomalous)
    np.save(path / "y_anomaly.npy", y_anomalous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset and split into train/test.")
    parser.add_argument("--output-dir", default="lab1/data", help="Directory to store generated data")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)

    dataset = fetch_openml(name="breast-w", version=1, as_frame=False, parser="liac-arff")
    x = dataset.data.astype(np.float32)
    if np.isnan(x).any():
        col_means = np.nanmean(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = col_means[inds[1]]
    y_raw = dataset.target
    y = np.where(y_raw == "malignant", 1, 0).astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    save_dataset_variants(root / "train", x_train, y_train, seed=101)
    save_dataset_variants(root / "test", x_test, y_test, seed=202)


if __name__ == "__main__":
    main()

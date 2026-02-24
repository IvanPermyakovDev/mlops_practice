#!/usr/bin/env python3
"""Fit preprocessing on train set and apply it to train/test sets."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(path / "X.npy")
    y = np.load(path / "y.npy")
    return x, y


def save_arrays(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "X.npy", x.astype(np.float32))
    np.save(path / "y.npy", y.astype(np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale train/test datasets with StandardScaler.")
    parser.add_argument("--input-dir", default="lab1/data", help="Directory with raw train/test subfolders")
    parser.add_argument("--output-dir", default="lab1/processed", help="Directory for transformed data")
    parser.add_argument("--scaler-path", default="lab1/artifacts/scaler.pkl", help="Path to save fitted scaler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    scaler_path = Path(args.scaler_path)

    x_train, y_train = load_arrays(raw_root / "train")
    x_test, y_test = load_arrays(raw_root / "test")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    save_arrays(out_root / "train", x_train_scaled, y_train)
    save_arrays(out_root / "test", x_test_scaled, y_test)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare features and targets for model training and testing."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale numeric features and encode labels for the Iris dataset."
    )
    parser.add_argument(
        "--input-dir",
        default="lab2/data/raw",
        help="Directory containing raw train.csv and test.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="lab2/data/processed",
        help="Directory where processed CSV files will be stored",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="lab2/artifacts",
        help="Directory where preprocessing artifacts will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train.csv")
    test_df = pd.read_csv(input_dir / "test.csv")

    scaler = StandardScaler()
    encoder = LabelEncoder()

    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    x_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_train = encoder.fit_transform(train_df[TARGET_COLUMN])
    y_test = encoder.transform(test_df[TARGET_COLUMN])

    processed_train = pd.DataFrame(x_train, columns=FEATURE_COLUMNS)
    processed_train["target"] = y_train
    processed_test = pd.DataFrame(x_test, columns=FEATURE_COLUMNS)
    processed_test["target"] = y_test

    processed_train.to_csv(output_dir / "train.csv", index=False)
    processed_test.to_csv(output_dir / "test.csv", index=False)

    with (artifacts_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)
    with (artifacts_dir / "label_encoder.pkl").open("wb") as file:
        pickle.dump(encoder, file)

    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    main()

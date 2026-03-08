#!/usr/bin/env python3
"""Download a raw dataset for lab2 and split it into train and test parts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)
COLUMN_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Iris dataset and create train/test CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default="lab2/data/raw",
        help="Directory where train.csv and test.csv will be stored",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES).dropna()

    train_df, test_df = train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset["species"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"Saved raw data to {output_dir} "
        f"(train={len(train_df)} rows, test={len(test_df)} rows)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a multiclass model for lab2 and persist it."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multiclass logistic regression model."
    )
    parser.add_argument(
        "--input-path",
        default="lab2/data/processed/train.csv",
        help="Path to processed training CSV file",
    )
    parser.add_argument(
        "--model-path",
        default="lab2/artifacts/model.pkl",
        help="Path where the trained model will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.input_path)
    x_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    model = LogisticRegression(max_iter=400, random_state=42)
    model.fit(x_train, y_train)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the trained model on the processed test set."""

from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model and print the test accuracy."
    )
    parser.add_argument(
        "--input-path",
        default="lab2/data/processed/test.csv",
        help="Path to processed test CSV file",
    )
    parser.add_argument(
        "--model-path",
        default="lab2/artifacts/model.pkl",
        help="Path to the trained model file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_df = pd.read_csv(args.input_path)
    x_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    with open(args.model_path, "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model test accuracy is: {accuracy:.3f}")


if __name__ == "__main__":
    main()

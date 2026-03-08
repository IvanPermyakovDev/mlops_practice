#!/usr/bin/env python3
"""Shared helpers for training and inference in lab3."""

from __future__ import annotations

from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path("lab3/artifacts/model.pkl")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def train_and_save_model(model_path: Path = MODEL_PATH) -> Path:
    dataset = load_iris()
    x = dataset.data
    y = dataset.target

    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(x, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    return model_path


def load_model(model_path: Path = MODEL_PATH) -> LogisticRegression:
    with model_path.open("rb") as file:
        model = pickle.load(file)
    return model

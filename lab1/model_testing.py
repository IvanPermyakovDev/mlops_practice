#!/usr/bin/env python3
"""Evaluate trained PyTorch model on test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(path / "X.npy").astype(np.float32)
    y = np.load(path / "y.npy").astype(np.int64)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test binary classifier and print accuracy.")
    parser.add_argument("--input-dir", default="lab1/processed/test", help="Path to preprocessed test data")
    parser.add_argument("--model-path", default="lab1/artifacts/model.pt", help="Path to saved model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_test, y_test = load_arrays(Path(args.input_dir))

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = BinaryClassifier(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).to(torch.int64).numpy()

    accuracy = (preds == y_test).mean()
    print(f"Model test accuracy is: {accuracy:.3f}")


if __name__ == "__main__":
    main()

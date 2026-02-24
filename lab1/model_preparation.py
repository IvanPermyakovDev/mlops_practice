#!/usr/bin/env python3
"""Train a simple PyTorch classifier and save model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    y = np.load(path / "y.npy").astype(np.float32)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary classifier with torch.")
    parser.add_argument("--input-dir", default="lab1/processed/train", help="Path to preprocessed train data")
    parser.add_argument("--model-path", default="lab1/artifacts/model.pt", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    x_train, y_train = load_arrays(Path(args.input_dir))

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = BinaryClassifier(input_dim=x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1]}, model_path)


if __name__ == "__main__":
    main()

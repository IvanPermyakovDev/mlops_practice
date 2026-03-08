#!/usr/bin/env python3
"""Train the model used by the lab3 microservice."""

from __future__ import annotations

from lab3.model import train_and_save_model


def main() -> None:
    model_path = train_and_save_model()
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

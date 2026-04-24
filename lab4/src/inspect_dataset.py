from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a short summary for a DVC-tracked CSV.")
    parser.add_argument("csv_path", nargs="?", default="lab4/data/titanic.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    print(f"path: {csv_path}")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"columns: {', '.join(df.columns)}")
    print(f"missing_age: {int(df['Age'].isna().sum())}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

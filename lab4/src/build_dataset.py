from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost.datasets import titanic


def load_titanic() -> pd.DataFrame:
    train_df, _ = titanic()
    return train_df


def selected_features() -> pd.DataFrame:
    df = load_titanic()
    return df.loc[:, ["Pclass", "Sex", "Age"]].copy()


def fill_age(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    mean_age = result["Age"].mean()
    result["Age"] = result["Age"].fillna(round(mean_age, 2))
    return result


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    encoded = pd.get_dummies(result["Sex"], prefix="Sex", dtype=int)
    return pd.concat([result, encoded], axis=1)


def build(version: str) -> pd.DataFrame:
    df = selected_features()
    if version in {"age_filled", "sex_encoded"}:
        df = fill_age(df)
    if version == "sex_encoded":
        df = encode_sex(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Titanic dataset version for DVC lab.")
    parser.add_argument(
        "--version",
        choices=["selected", "age_filled", "sex_encoded"],
        required=True,
        help="Dataset version to materialize.",
    )
    parser.add_argument(
        "--output",
        default="lab4/data/titanic.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build(args.version)
    dataset.to_csv(output, index=False)

    print(
        f"Saved {args.version} dataset to {output}: "
        f"{dataset.shape[0]} rows, {dataset.shape[1]} columns"
    )
    print(f"Columns: {', '.join(dataset.columns)}")
    print(f"Missing Age values: {int(dataset['Age'].isna().sum())}")


if __name__ == "__main__":
    main()

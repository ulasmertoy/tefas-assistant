"""Quick test: load fund features from parquet into Pydantic objects."""

import math
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from schemas.fund import FundMetrics

FEATURES_FILE = (
    Path(__file__).parent.parent / "data" / "processed" / "funds_features.parquet"
)


def nan_to_none(value):
    """Convert NaN to None for Pydantic compatibility."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df)} funds from parquet\n")

    # Try loading each row into a FundMetrics object
    metrics_objects = []
    errors = []

    for _, row in df.iterrows():
        try:
            metrics = FundMetrics(
                code=row["code"],
                title=row["title"],
                return_1y=nan_to_none(row["return_1y"]),
                return_2y=nan_to_none(row["return_2y"]),
                return_4y=nan_to_none(row["return_4y"]),
                volatility=nan_to_none(row["volatility"]),
                sharpe=nan_to_none(row["sharpe"]),
                sortino=nan_to_none(row["sortino"]),
                max_drawdown=nan_to_none(row["max_drawdown"]),
            )
            metrics_objects.append(metrics)
        except ValidationError as e:
            errors.append((row["code"], str(e)))

    print(f"Successfully parsed: {len(metrics_objects)} funds")
    print(f"Validation errors: {len(errors)} funds")

    if errors:
        print("\nFirst 3 errors:")
        for code, err in errors[:3]:
            print(f"  {code}: {err[:100]}...")

    # Print first object to see Pydantic in action
    if metrics_objects:
        print("\nFirst fund (as Pydantic object):")
        print(metrics_objects[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
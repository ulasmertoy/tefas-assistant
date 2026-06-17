"""
fetch_tcmb_rates.py — Fetch TCMB weighted-average cost of funding (AOFM) from EVDS,
save to local parquet. This serves as the risk-free rate series for Sharpe/Sortino.

Run ONCE (or when updating). Pipeline reads the parquet, never EVDS directly.
Uses evdspy package (handles EVDS 3 migration + auth). Key read from .env.

Usage: python src/fetch_tcmb_rates.py
"""

import logging
from pathlib import Path

import pandas as pd
from evdspy import get_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERIES_CODE = "TP.APIFON4"   # TCMB weighted-average cost of funding (effective policy rate)
START_DATE = "01-05-2021"
END_DATE = "18-05-2026"

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "raw" / "tcmb_rates.parquet"


def fetch_policy_rate() -> pd.DataFrame:
    logger.info(f"Fetching EVDS series {SERIES_CODE} (monthly)...")
    df = get_series(SERIES_CODE, start_date=START_DATE, end_date=END_DATE,
                    frequency="monthly", cache=False)

    # Tidy: rename columns, parse dates, keep rate as decimal
    df = df.rename(columns={SERIES_CODE: "rate_pct", "Tarih_string": "month"})
    df["date"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["rate"] = df["rate_pct"] / 100.0   # 19.0 -> 0.19
    df = df[["date", "rate"]].sort_values("date").reset_index(drop=True)

    logger.info(f"Fetched {len(df)} monthly rate points: "
                f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    df = fetch_policy_rate()
    print(df.head().to_string())
    print("...")
    print(df.tail().to_string())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    logger.info(f"Saved: {OUT_FILE}")
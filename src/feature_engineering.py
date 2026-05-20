import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "raw" / "funds_raw.parquet"
FEATURES_FILE = DATA_DIR / "processed" / "funds_features.parquet"


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load raw parquet and clean."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["price"] > 0]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("code")["price"].pct_change()
    logger.info(f"Loaded and cleaned: {df.shape}")
    return df


def compute_returns(group: pd.DataFrame) -> pd.Series:
    """Compute 1Y, 2Y, 4Y returns for a single fund."""
    today = group["date"].max()
    price_today = group[group["date"] == today]["price"].values[0]

    results = {}
    for years, key in [(1, "return_1y"), (2, "return_2y"), (4, "return_4y")]:
        cutoff = today - pd.DateOffset(years=years)
        past_prices = group[group["date"] <= cutoff]["price"]
        if past_prices.empty:
            results[key] = np.nan
        else:
            results[key] = (price_today - past_prices.iloc[-1]) / past_prices.iloc[-1]

    return pd.Series(results)


def compute_risk_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute volatility, Sharpe, Sortino, max drawdown."""
    returns = group["daily_return"].dropna()

    if len(returns) < 30:
        return pd.Series({
            "volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan
        })

    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else np.nan

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return pd.Series({
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all features into a single dataframe."""
    logger.info("Computing returns...")
    returns_df = df.groupby("code").apply(compute_returns, include_groups=False).reset_index()

    logger.info("Computing risk metrics...")
    risk_df = df.groupby("code").apply(compute_risk_metrics, include_groups=False).reset_index()

    features_df = returns_df.merge(risk_df, on="code")

    title_map = df.groupby("code")["title"].last()
    features_df = features_df.merge(title_map, on="code")

    logger.info(f"Features built: {features_df.shape}")
    return features_df


if __name__ == "__main__":
    df = load_and_clean(RAW_FILE)
    features_df = build_features(df)

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_FILE, index=False)
    logger.info(f"Saved: {FEATURES_FILE}")
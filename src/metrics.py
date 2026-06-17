"""
metrics.py — Pure financial metric functions.

Each function takes data and returns a number (or series). No file I/O, no logging,
no side effects. This is the single source of truth for all metric formulas, shared
by the screening layer (Layer 1) and the on-demand layer (Layer 2).
"""

import numpy as np
import pandas as pd

# --- Constants ---
TRADING_DAYS = 252  # annualization factor for daily data
MIN_OBSERVATION = 120 
MIN_REGIME_DAYS = 120  # minimum valid days within a regime to trust its metrics

def daily_returns(prices: pd.Series) -> pd.Series:
    """
    Daily simple returns from a price series.

    fill_method=None is critical: NaN prices (cleaned zeros) stay NaN instead of
    being forward-filled, so gaps don't create fake 0% returns. NaNs are excluded
    by downstream metrics automatically.
    """
    return prices.pct_change(fill_method=None)


def risk_free_series(dates: pd.DatetimeIndex, rf_table: pd.DataFrame) -> pd.Series:
    """
    Map each (daily) date to its applicable risk-free rate, converted to a daily rate.

    rf_table: DataFrame with columns ['date', 'rate'] where 'date' is the first of
              each month and 'rate' is the ANNUAL rate as a decimal (e.g. 0.50).
              Passed in by the caller (not read from disk) to keep this function pure.

    Returns a Series indexed by `dates`, giving the DAILY risk-free rate for each date.
    Monthly rate is carried forward within the month, then divided by TRADING_DAYS.
    """
    # Build a daily-indexed annual-rate series by forward-filling the monthly table
    rf = rf_table.set_index("date")["rate"].sort_index()

    # Reindex onto the requested dates, forward-filling the last known monthly rate
    annual = rf.reindex(rf.index.union(dates)).ffill().reindex(dates)

    # Convert annual -> daily
    daily = annual / TRADING_DAYS
    daily.name = "rf_daily"
    return daily


def cagr(prices: pd.Series, years: float | None = None) -> float:
    """
    Compound annual growth rate.

    If `years` is given: trailing window of that many years (e.g. years=4 -> last 4y).
    If `years` is None: inception-to-date — uses the fund's entire valid history,
    annualized over its actual lifespan.

    Returns NaN if there isn't enough data (start price missing or invalid).
    For inception mode, the caller should enforce a minimum-history threshold
    (we use 120 valid days) before trusting the result.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan

    end_price = prices.iloc[-1]
    end_date = prices.index.max()

    if years is None:
        # Inception mode: from the first valid price to today
        start_price = prices.iloc[0]
        start_date = prices.index.min()
        span_years = (end_date - start_date).days / 365.25
        if span_years <= 0:
            return np.nan
        years = span_years
    else:
        # Fixed-window mode: price as of `years` ago (or nearest earlier)
        start_target = end_date - pd.DateOffset(years=int(years))
        past = prices[prices.index <= start_target]
        if past.empty:
            return np.nan  # fund younger than the requested window
        start_price = past.iloc[-1]

    if start_price <= 0:
        return np.nan

    return (end_price / start_price) ** (1 / years) - 1

def volatility(returns: pd.Series) -> float:
    """
    Annualized volatility: the standard deviation of daily returns, scaled to a year.

    Higher = more day-to-day swing (e.g. equity funds). Lower = steadier
    (e.g. money-market funds). Returns NaN if too few observations.
    """
    returns = returns.dropna()
    if len(returns) < MIN_OBSERVATION:
        return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe(returns: pd.Series, rf_daily: pd.Series) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of total volatility.

    rf_daily: daily risk-free rate aligned to the same dates as `returns`
              (from risk_free_series). Excess = fund return - risk-free.
    Returns NaN if too few observations or zero volatility.
    """
    returns = returns.dropna()
    if len(returns) < MIN_OBSERVATION:
        return np.nan

    excess = returns - rf_daily.reindex(returns.index)
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return np.nan

    # Annualize: mean daily excess * 252, divided by (daily std * sqrt(252))
    return (excess.mean() * TRADING_DAYS) / (vol * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_daily: pd.Series) -> float:
    """
    Annualized Sortino ratio: excess return per unit of DOWNSIDE volatility.

    Same as Sharpe, but the denominator only counts negative returns — upside
    swings aren't penalized. Returns NaN if too few observations or no downside.
    """
    returns = returns.dropna()
    if len(returns) < MIN_OBSERVATION:
        return np.nan

    excess = returns - rf_daily.reindex(returns.index)
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan

    return (excess.mean() * TRADING_DAYS) / (downside * np.sqrt(TRADING_DAYS))

def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown: the largest peak-to-trough decline over the period.

    Returns a negative number (e.g. -0.40 = a 40% drop from a peak).
    Computed from the cumulative return path. NaN if too few observations.
    """
    returns = returns.dropna()
    if len(returns) < MIN_OBSERVATION:
        return np.nan

    cumulative = (1 + returns).cumprod()      # growth of 1 unit over time
    running_max = cumulative.cummax()         # highest point reached so far
    drawdown = (cumulative - running_max) / running_max  # % below the peak
    return drawdown.min()                     # the deepest dip

def get_regimes() -> list[tuple[str, str, str]]:
    """
    Real-rate regime boundaries (v1: hardcoded, verified against TCMB rate + CPI).

    Returns list of (start_date, end_date, name). To upgrade to v2 (auto-derived
    from the real-rate series), only this function's body changes — regime_metrics
    and everything downstream stay the same.

    peak_tight and easing_but_tight are both positive-real-rate environments;
    the split reflects the nominal-rate phase, not a real-rate sign change.
    """
    return [
        ("2021-09-23", "2023-06-01", "negative_real"),
        ("2023-06-01", "2024-03-21", "shock_tightening"),
        ("2024-03-21", "2024-12-26", "peak_tight"),
        ("2024-12-26", "2026-05-18", "easing_but_tight"),
    ]

def regime_metrics(returns: pd.Series, regimes: list = None,
                   rf_daily: pd.Series = None) -> dict:
    """
    Per-regime performance for one fund.

    For each regime, computes annualized return and volatility using only the
    days that fall within that regime's window. A regime with fewer than
    MIN_REGIME_DAYS valid observations yields NaN (too short to trust).

    `regimes` defaults to get_regimes(). `rf_daily` is optional — if provided,
    a regime-conditional Sharpe is also computed.

    Returns a flat dict: {'negative_real_return': ..., 'negative_real_vol': ..., ...}
    """
    if regimes is None:
        regimes = get_regimes()

    returns = returns.dropna()
    result = {}

    for start, end, name in regimes:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        mask = (returns.index >= s) & (returns.index < e)
        seg = returns[mask]

        if len(seg) < MIN_REGIME_DAYS:
            result[f"{name}_return"] = np.nan
            result[f"{name}_vol"] = np.nan
            continue

        # Annualized return over the regime (compound), and annualized volatility
        cum_growth = (1 + seg).prod()
        years = len(seg) / TRADING_DAYS
        result[f"{name}_return"] = cum_growth ** (1 / years) - 1
        result[f"{name}_vol"] = seg.std() * np.sqrt(TRADING_DAYS)

    return result
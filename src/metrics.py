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


def max_drawdown_window(prices: pd.Series, years: float | None = None,
                        min_days: int = MIN_OBSERVATION) -> float:
    """
    Maximum drawdown WITHIN a trailing window — peak-to-trough over `years`.

    Why this exists alongside max_drawdown(): the plain version measures the
    deepest dip over a fund's ENTIRE history. Placing that next to a 1-year
    return compares two different periods — a fund's 2022 crash gets attributed
    to a return earned in the last 12 months. This function scopes the drawdown
    to the SAME window as cagr(prices, years), so return and risk are measured
    over one consistent period.

    Takes PRICES (not returns), to slice the window by date the same way cagr
    does, then normalizes to the window's own starting price so the peak is the
    window peak — not an all-time peak carried in from before the window.

    years=None  -> entire valid history (matches max_drawdown's scope).
    years=1     -> trailing 1 year (pairs with return_1y).

    Returns a negative number (e.g. -0.10 = a 10% drop). NaN if the window has
    fewer than `min_days` valid observations.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan

    if years is not None:
        start_target = prices.index.max() - pd.DateOffset(years=int(years))
        prices = prices[prices.index >= start_target]

    if len(prices) < min_days:
        return np.nan

    cumulative = prices / prices.iloc[0]      # normalize to window start
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


PERIOD_WINDOWS = {
    "1A": pd.DateOffset(months=1),
    "3A": pd.DateOffset(months=3),
    "6A": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


def period_returns(prices: pd.Series) -> dict:
    """
    Standard trailing-period simple returns: 1A, 3A, 6A, 1Y, and YBB (year-to-date).

    Why this exists alongside cagr(): cagr() annualizes — it answers "what rate
    would this compound at over a year". That's the wrong question for a 1-month
    or 3-month window (a 2% move in 30 days is not "24% annualized" in any
    meaningful sense to a user comparing funds today). period_returns() answers
    the simpler question users actually ask first: "what has this fund actually
    done in the last N", as a plain cumulative return, no annualization.

    For each window, the start price is the nearest available price ON OR BEFORE
    the target date (same "nearest earlier price" rule cagr() uses for its
    trailing-window mode), so gaps/holidays don't break the calculation.

    YBB (yılbaşından bugüne / year-to-date): start price is the nearest price on
    or before December 31 of the previous year.

    Returns a dict, e.g. {"1A": 0.021, "3A": 0.084, "6A": 0.041, "1Y": 0.452,
    "YBB": 0.118}. Any window the fund is too young for (or missing data around
    the target date) is NaN — never silently dropped, so the caller can render
    "veri yok" explicitly instead of a blank cell.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return {**{k: np.nan for k in PERIOD_WINDOWS}, "YBB": np.nan}

    end_price = prices.iloc[-1]
    end_date = prices.index.max()

    result = {}
    for label, offset in PERIOD_WINDOWS.items():
        start_target = end_date - offset
        past = prices[prices.index <= start_target]
        if past.empty:
            result[label] = np.nan
            continue
        start_price = past.iloc[-1]
        result[label] = np.nan if start_price <= 0 else (end_price / start_price - 1)

    ytd_target = pd.Timestamp(year=end_date.year - 1, month=12, day=31)
    past_ytd = prices[prices.index <= ytd_target]
    if past_ytd.empty or past_ytd.iloc[-1] <= 0:
        result["YBB"] = np.nan
    else:
        result["YBB"] = end_price / past_ytd.iloc[-1] - 1

    return result


# --------------------------------------------------------------------------- #
# Piyasa rejimleri
#
# Son rejimin bitişi SABİT DEĞİL — veriden gelir. Eskiden "2026-12-31" yazıyordu
# ve etiketi "Ara'24–May'26" idi; donmuş veride bu doğruydu, canlı veride ikisi
# de yanlış. Veri ilerledikçe etiket geride kalıyordu ve explainer.py o etiketi
# okuyup kullanıcıya yanlış tarih aralığı söylüyordu. 2026 sonunu geçtiğimizde
# ise yeni günler hiçbir rejime düşmeyecek, sessizce tablodan kaybolacaktı.
#
# Asıl tasarım hatası şuydu: sınırlar bir yerde, etiketler başka yerde elle
# yazılmıştı. Bağımsız oldukları için ayrışabiliyorlardı. Artık etiket
# sınırlardan TÜRETİLİYOR — ikisi yapısal olarak ayrışamaz.
# --------------------------------------------------------------------------- #
_MONTHS_TR = {1: "Oca", 2: "Şub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz",
              7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki", 11: "Kas", 12: "Ara"}

_REGIME_NAMES = {
    "negative_real":    "negatif reel faiz",
    "shock_tightening": "şok sıkılaşma",
    "peak_tight":       "zirve sıkılık",
    "easing_but_tight": "temkinli gevşeme",
}


def get_regimes(end_date=None) -> list[tuple[str, str, str]]:
    """Rejim sınırları. Son rejimin bitişi `end_date` (verinin son günü).

    end_date verilmezse bugün kullanılır. Çağıranın veri sonunu geçmesi tercih
    edilir: iş birkaç gün çökerse duvar saati veriden ileri gider ve etiket
    olmayan bir dönemi varmış gibi gösterir.

    Not: regime_metrics maskesi `< end` olduğu için son gün dışarıda kalır.
    500+ günlük yıllıklandırılmış metriklerde bir günün etkisi ihmal edilebilir.
    """
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today()
    return [
        ("2021-09-23", "2023-06-01", "negative_real"),
        ("2023-06-01", "2024-03-21", "shock_tightening"),
        ("2024-03-21", "2024-12-26", "peak_tight"),
        ("2024-12-26", end.strftime("%Y-%m-%d"), "easing_but_tight"),
    ]


def _stamp(d) -> str:
    """2023-06-01 -> Haz'23"""
    d = pd.Timestamp(d)
    return f"{_MONTHS_TR[d.month]}'{d.strftime('%y')}"


def regime_labels(end_date=None) -> dict[str, str]:
    """Kullanıcıya gösterilen etiketler — sınırlardan türetilir, elle yazılmaz.

    Eski REGIME_LABELS sabiti bilerek kaldırıldı: import eden kod hata versin,
    sessizce eski tarihi göstermesin.
    """
    return {
        name: f"{_REGIME_NAMES[name]} ({_stamp(start)}–{_stamp(end)})"
        for start, end, name in get_regimes(end_date)
    }

def regime_metrics(returns: pd.Series, regimes: list = None,
                   rf_daily: pd.Series = None) -> dict:
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
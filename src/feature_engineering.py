"""
feature_engineering.py — Orchestration layer (Layer 1: screening).

Pipeline:  load_and_clean -> load_rf -> build_features -> health_checks -> write

Architecture rule (handoff §2.2): this module owns ALL side effects and
decisions — file I/O, what to clean, what counts as real history, which funds to
exclude, when the result is trustworthy. metrics.py stays pure and never imports
from here. Dependency is one-directional: feature_engineering -> metrics.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (
    MIN_OBSERVATION,
    daily_returns,
    risk_free_series,
    cagr,
    volatility,
    sharpe,
    sortino,
    max_drawdown,
    get_regimes,
    regime_metrics,
)

logger = logging.getLogger(__name__)

# --- Policy constants (the "decisions", all in one place) ---
MAX_MID_ZERO_RATIO = 0.15      # exclude fund if >15% of its active range is zero (GPZ:
                               # pervasive, not localized -> can't repair, untrustworthy)
MAX_DAILY_RETURN = 2.0         # POST-REPAIR safety: if a single-day move is STILL > 200%
                               # after glitch repair, the fund is unrepairable -> exclude
YOUNG_LEAGUE_MAX_DAYS = 365    # history_days < this -> "young" league (§5)

# --- Glitch repair (Stage 2b): blank economically-impossible single-day prints ---
# The data is faithful to TEFAS, but TEFAS itself sometimes publishes a bad NAV on
# a single day (e.g. NBH 2022-02-17 = 0.000406 between two ~0.25 days; Fintables
# shows the exact same print). We NaN those days for the return math — we do NOT
# trim a date range and do NOT drop the fund. The fund and its real history stay.
GLITCH_MEDIAN_WINDOW = 61      # local-level reference (~3 trading months), centered
GLITCH_HI = 2.0                # price > 2.0x local median  -> spike outlier
GLITCH_LO = 0.5                # price < 0.5x local median  -> near-zero / crash outlier
GLITCH_MAX_JUMP = 1.0          # |single-day return| > 100%  -> impossible jump
GLITCH_PASSES = 6              # iterate so multi-day cluster edges re-settle & get caught

# Health-gate HARD ranges: a value outside these (and not NaN) blocks the write.
# Only INPUTS that can be genuinely impossible are hard-gated. Derived ratios
# (Sharpe/Sortino) have no impossible range — extreme values are a real product
# of the high risk-free environment, so they are reported as warnings, not blocked.
HARD_RANGES = {
    "volatility": (0.0, 1.5),     # 0–150% annualized; >150% means corrupt data
    "max_drawdown": (-1.0, 0.0),  # a drop, never positive, never worse than -100%
}
RETURN_RANGE = (-1.0, 50.0)       # any CAGR/return col: can't lose >100%; 50 = bug ceiling

# Soft thresholds (reported, never block):
SHARPE_EXTREME = 3.0              # |Sharpe| above this is flagged (expected in high RF)
SORTINO_EXTREME = 8.0
MM_VOL_WARN = 0.25                # money-market cohort median vol above this -> warning
STALE_RATIO_WARN = 0.30           # >30% zero-return days -> stale prices, vol unreliable


# --------------------------------------------------------------------------- #
# Glitch detection (pure helper — also handy to import in eda.ipynb)
# --------------------------------------------------------------------------- #
def detect_bad_days(prices: pd.Series,
                    median_window: int = GLITCH_MEDIAN_WINDOW,
                    hi: float = GLITCH_HI, lo: float = GLITCH_LO,
                    max_jump: float = GLITCH_MAX_JUMP,
                    passes: int = GLITCH_PASSES) -> pd.Series:
    """Flag economically-impossible single-day prints on one fund's price series.

    Two NEIGHBOUR-RELATIVE signals — never an absolute threshold. A high price is
    not 'bad' if it got there smoothly (the TLY lesson: TLY climbs to 6000+ and is
    never flagged). 'Bad' means a value that breaks away from its neighbours and
    reverts:
      1. local-level outlier : price far off its ~3-month local median
                               (> hi*median spike, or < lo*median near-zero/crash)
      2. impossible jump     : a single-day move no NAV can make (|return| > max_jump)

    Iterated: blanking the worst prints lets the median/returns re-settle, so the
    edges of a multi-day cluster (NBH's Feb-2022 free-fall) get caught on later
    passes. On real, smooth funds nothing is flagged and it returns on pass 1.

    Returns
    -------
    pd.Series[bool]  same index as `prices`; True = blank this day's price.
    """
    work = prices.replace(0, np.nan).copy()
    bad = pd.Series(False, index=prices.index)
    for _ in range(passes):
        med = work.rolling(median_window, center=True, min_periods=5).median()
        ratio = work / med
        outlier = ((ratio > hi) | (ratio < lo)).fillna(False)
        jump = (work.pct_change(fill_method=None).abs() > max_jump).fillna(False)
        new = (outlier | jump) & ~bad
        if not new.any():
            break
        bad |= new
        work[bad] = np.nan
    return bad


# --------------------------------------------------------------------------- #
# Stage 1: load + clean
# --------------------------------------------------------------------------- #
def load_and_clean(funds_path, max_mid_zero_ratio: float = MAX_MID_ZERO_RATIO,
                   max_daily_return: float = MAX_DAILY_RETURN):
    """Load raw funds parquet and clean it, per fund.

    Stage 1  (cut)    : keep only [first real NAV .. last real NAV]; drop the
                        rectangular-grid placeholder rows outside that window.
    Stage 2  (null)   : inside the active range, price == 0 -> NaN (no fake 0% return).
    Stage 2b (repair) : NaN economically-impossible single-day prints (detect_bad_days)
                        — a spike/near-zero that reverts, or a >100% one-day jump.
                        The fund is KEPT; only those days are blanked, so its real
                        history (incl. genuine dips like NBH's) survives. This
                        replaces the old blunt "drop the whole fund on a spike" rule.
    Stage 3  (exclude): drop the fund ONLY when it is unusable even after repair —
                        mid_zero_ratio > max_mid_zero_ratio (GPZ: zeros so pervasive
                        the series is untrustworthy) OR a single-day move is STILL
                        > max_daily_return after repair (unrepairable; should be rare).

    history_days = valid observations on the cleaned series (active.dropna(), after
    both null + repair), so it speaks the same language as metrics.py's
    len(returns.dropna()) gate (~n, not exactly n).

    Returns
    -------
    clean : dict[str, pd.Series]  code -> cleaned date-indexed prices (survivors only)
    meta  : pd.DataFrame (index=code)  diagnostics for every fund incl. dropped ones
    """
    df = pd.read_parquet(funds_path)
    df["date"] = pd.to_datetime(df["date"])
    titles = df.groupby("code")["title"].first()

    clean: dict[str, pd.Series] = {}
    rows = []

    for code, g in df.sort_values("date").groupby("code"):
        s = g.set_index("date")["price"]
        nonzero = s[(s.notna()) & (s != 0)]

        if nonzero.empty:
            rows.append(dict(code=code, lead_cut=len(s), trail_cut=0, mid_nulled=0,
                             glitch_repaired=0, active_span=0, mid_zero_ratio=np.nan,
                             max_daily=np.nan, history_days=0, dropped=True,
                             reason="no_valid_price"))
            continue

        first, last = nonzero.index.min(), nonzero.index.max()
        active = s[(s.index >= first) & (s.index <= last)].copy()      # Stage 1
        lead_cut = int((s.index < first).sum())
        trail_cut = int((s.index > last).sum())

        mid_mask = (active == 0)                                       # Stage 2
        mid_nulled = int(mid_mask.sum())
        active[mid_mask] = np.nan
        ratio = mid_nulled / len(active)                              # GPZ signal (pervasive zeros)

        bad_mask = detect_bad_days(active)                            # Stage 2b: repair
        glitch_repaired = int(bad_mask.sum())
        active[bad_mask] = np.nan

        history_days = int(active.dropna().shape[0])
        ret = active.pct_change(fill_method=None)
        max_daily = float(ret.abs().max()) if ret.notna().any() else np.nan

        # Stage 3: exclude only the unusable. Pervasive zeros first (GPZ), then the
        # rare fund still corrupt after repair (safety; the health gate is the net).
        if ratio > max_mid_zero_ratio:
            reason = f"mid_zero_ratio>{max_mid_zero_ratio}"
        elif pd.notna(max_daily) and max_daily > max_daily_return:
            reason = "unrepairable_spike"
        else:
            reason = ""

        row = dict(code=code, lead_cut=lead_cut, trail_cut=trail_cut,
                   mid_nulled=mid_nulled, glitch_repaired=glitch_repaired,
                   active_span=len(active), mid_zero_ratio=ratio, max_daily=max_daily,
                   history_days=history_days, dropped=bool(reason), reason=reason)
        rows.append(row)
        if not reason:
            clean[code] = active

    meta = pd.DataFrame(rows).set_index("code")
    meta["title"] = titles
    logger.info("load_and_clean: kept %d, dropped %d (%s) | glitch-repaired days: %d across %d funds",
                int((~meta["dropped"]).sum()), int(meta["dropped"].sum()),
                ", ".join(f"{r}:{n}" for r, n in meta.loc[meta['dropped'], 'reason'].value_counts().items()),
                int(meta["glitch_repaired"].sum()), int((meta["glitch_repaired"] > 0).sum()))
    return clean, meta


# --------------------------------------------------------------------------- #
# Stage 2: load risk-free table (read from disk here; metrics gets it as a param)
# --------------------------------------------------------------------------- #
def load_rf(rf_path) -> pd.DataFrame:
    """Read the EVDS risk-free parquet (columns date, rate). FE reads it from disk
    and passes it into metrics as a parameter, so metrics.py stays pure (§2.3)."""
    rf = pd.read_parquet(rf_path)
    rf["date"] = pd.to_datetime(rf["date"])
    missing = {"date", "rate"} - set(rf.columns)
    if missing:
        raise ValueError(f"rf table missing columns: {missing}")
    return rf[["date", "rate"]].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 3: build the feature table (calls pure metrics on cleaned series)
# --------------------------------------------------------------------------- #
def build_features(clean: dict, meta: pd.DataFrame, rf_table: pd.DataFrame,
                   regimes=None) -> pd.DataFrame:
    """One row per surviving fund. FE only ORCHESTRATES; every formula lives in
    metrics.py. return_inception is computed only when history_days >= 120 (the
    caller-enforced floor the inception docstring refers to)."""
    if regimes is None:
        regimes = get_regimes()

    rows = []
    for code, prices in clean.items():
        history_days = int(meta.loc[code, "history_days"])
        returns = daily_returns(prices)
        rf_daily = risk_free_series(returns.index, rf_table)

        row = {
            "code": code,
            "title": meta.loc[code, "title"],
            "history_days": history_days,
            "league": "young" if history_days < YOUNG_LEAGUE_MAX_DAYS else "mature",
            "return_1y": cagr(prices, 1),
            "return_2y": cagr(prices, 2),
            "return_4y": cagr(prices, 4),
            "return_inception": cagr(prices, None) if history_days >= MIN_OBSERVATION else np.nan,
            "volatility": volatility(returns),
            "sharpe": sharpe(returns, rf_daily),
            "sortino": sortino(returns, rf_daily),
            "max_drawdown": max_drawdown(returns),
        }
        row.update(regime_metrics(returns, regimes, rf_daily=None))
        rows.append(row)

    return pd.DataFrame(rows).set_index("code")

# --------------------------------------------------------------------------- #
# Stage 4: health gate ("silent wrongness" is the dangerous bug — §10)
# --------------------------------------------------------------------------- #
def health_checks(features: pd.DataFrame, clean: dict = None):
    """Return (passed, report).

    HARD gate (blocks the write) = an IMPOSSIBLE value: volatility / return /
    drawdown outside its physically-possible range. After Stage 2b repair these
    should never fire; if one does, it means a fund slipped through unrepaired.

    SOFT report (warns, never blocks): extreme-but-real Sharpe/Sortino (a product
    of the ~40% risk-free environment, not a bug), money-market spot check, and
    stale-price funds (vol unreliable). NaNs are counted and reported, not blocked.
    """
    report = {"hard_violations": [], "warnings": []}

    # --- HARD: impossible input values ---
    for col, (lo, hi) in HARD_RANGES.items():
        s = features[col].dropna()
        bad = s[(s < lo) | (s > hi)]
        for code, v in bad.items():
            report["hard_violations"].append((code, col, float(v)))

    return_cols = [c for c in features.columns
                   if c.startswith("return_") or c.endswith("_return")]
    lo, hi = RETURN_RANGE
    for col in return_cols:
        s = features[col].dropna()
        bad = s[(s < lo) | (s > hi)]
        for code, v in bad.items():
            report["hard_violations"].append((code, col, float(v)))

    # --- NaN audit (expected NaNs are fine; we surface counts) ---
    nan_counts = features.isna().sum()
    report["nan_counts"] = nan_counts[nan_counts > 0].to_dict()

    # --- SOFT: extreme Sharpe / Sortino (reported, not blocked) ---
    n_sharpe = int((features["sharpe"].abs() > SHARPE_EXTREME).sum())
    n_sortino = int((features["sortino"].dropna().abs() > SORTINO_EXTREME).sum())
    if n_sharpe:
        report["warnings"].append(
            f"{n_sharpe} fon |Sharpe|>{SHARPE_EXTREME:g} (yüksek RF + düşük vol, beklenen)")
    if n_sortino:
        report["warnings"].append(
            f"{n_sortino} fon |Sortino|>{SORTINO_EXTREME:g} (sakin fonda payda minik, beklenen)")

    # --- SOFT: money-market gold-standard spot check (by title) ---
    mm = features[features["title"].str.contains(
        r"PARA P[İI]YASASI|L[İI]K[İI]T", case=False, na=False, regex=True)]
    report["money_market_n"] = len(mm)
    if len(mm):
        report["mm_median_vol"] = float(mm["volatility"].median())
        if report["mm_median_vol"] > MM_VOL_WARN:
            report["warnings"].append(
                f"para piyasası medyan vol {report['mm_median_vol']:.3f} > {MM_VOL_WARN}")

    # --- SOFT: stale-price funds (vol unreliable) — needs cleaned series ---
    if clean is not None:
        stale = {}
        for code in features.index:
            r = clean[code].pct_change(fill_method=None).dropna()
            if len(r) >= MIN_OBSERVATION:
                stale[code] = float((r == 0).mean())
        stale_funds = sorted((c for c, v in stale.items() if v > STALE_RATIO_WARN),
                             key=lambda c: -stale[c])
        report["stale_funds"] = stale_funds
        if stale_funds:
            report["warnings"].append(
                f"{len(stale_funds)} fon seyrek fiyatlı (>%{int(STALE_RATIO_WARN*100)} sıfır getiri) "
                f"— bu fonlarda vol güvenilmez: {', '.join(stale_funds[:8])}"
                + (" ..." if len(stale_funds) > 8 else ""))

    report["passed"] = len(report["hard_violations"]) == 0
    return report["passed"], report


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run(funds_path, rf_path, out_path=None, write: bool = True):
    """Full Layer-1 pipeline. Output is written ONLY if the health gate passes."""
    clean, meta = load_and_clean(funds_path)
    rf_table = load_rf(rf_path)
    features = build_features(clean, meta, rf_table)
    passed, report = health_checks(features, clean)

    for w in report["warnings"]:
        logger.warning("health: %s", w)

    if write and out_path:
        if passed:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(out_path)
            logger.info("Health gate PASSED -> wrote %s (%d funds)", out_path, len(features))
        else:
            logger.error("Health gate FAILED (%d impossible values) -> output NOT written",
                         len(report["hard_violations"]))
    return features, meta, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    base = Path(__file__).parent.parent / "data"
    run(base / "raw" / "funds_raw.parquet",
        base / "raw" / "tcmb_rates.parquet",
        base / "processed" / "funds_features.parquet")
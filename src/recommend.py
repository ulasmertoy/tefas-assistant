"""
recommend.py — Layer 2: risk-profile screening / ranking.

Reads the Layer-1 output (funds_features.parquet) and, given a risk profile,
returns funds ranked for that profile. Pure ranking logic; the only I/O is the
optional load_features() convenience reader.

Boundary rule (mirrors feature_engineering): the INPUT is a typed Pydantic model
(RiskProfile) so it is JSON-serialisable and agent/API-ready; the heavy lifting
inside is plain pandas. Two exits: recommend() returns a ranked DataFrame (handy
for notebooks/debug), while build_response() wraps it into the typed, JSON-ready
RecommendationResponse that the API/agent speaks (the seam, kept in this file).

Design decisions (validated against real TEFAS data):
  * Risk is controlled by a VOLATILITY BAND (vol_min..vol_max), not just a ceiling.
    A ceiling alone collapses every profile onto the same low-vol, high-Sharpe
    funds (Sharpe rewards low vol in a ~40% risk-free environment). The band makes
    the three profiles genuinely differentiate.
  * Category is NOT a risk proxy — most TEFAS categories span the whole vol range
    (Serbest: 1.3%..64%). Category is used only as a LABEL and an optional intent
    filter (allowed_categories), never to set risk.
  * Ranking is always RISK-ADJUSTED (Sharpe), never raw return (the TLY trap:
    +500,000% is currency/inflation/leverage, not skill). Conservative additionally
    leads with the smallest drawdown.
  * Young (<1y history) funds have fragile metrics, so they are league-separated:
    only the 'aggressive' profile surfaces them, and as a clearly-separate list.
"""
import math

import pandas as pd

# Pydantic contracts live in one place (schemas.py): the input RiskProfile and the
# output FundRecommendation / RecommendationResponse. recommend.py is the engine +
# the builder that turns its pandas output into those typed models.
from schemas import RankBy, RiskProfile, FundRecommendation, RecommendationResponse


PRESETS: dict[str, RiskProfile] = {
    "conservative": RiskProfile(name="conservative", vol_min=0.00, vol_max=0.10,
                                rank_by=RankBy.DRAWDOWN),
    "balanced":     RiskProfile(name="balanced",     vol_min=0.08, vol_max=0.20,
                                rank_by=RankBy.SHARPE),
    "aggressive":   RiskProfile(name="aggressive",   vol_min=0.15, vol_max=0.45,
                                rank_by=RankBy.SHARPE, include_young=True),
}


# --------------------------------------------------------------------------- #
# Category (label + optional intent filter — derived from the fund title)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = str(s).upper()
    for a, b in [("İ", "I"), ("Ş", "S"), ("Ğ", "G"), ("Ü", "U"), ("Ö", "O"), ("Ç", "C")]:
        s = s.replace(a, b)
    return s


# First match wins; order matters (more specific umbrellas first).
_CATEGORY_RULES = [
    ("Para Piyasası",  ["PARA PIYASASI", "LIKIT"]),
    ("Kıymetli Maden", ["KIYMETLI MADEN", "ALTIN", "GUMUS"]),
    ("Katılım",        ["KATILIM"]),
    ("Hisse Senedi",   ["HISSE"]),
    ("Borçlanma",      ["BORCLANMA", "TAHVIL", "BONO", "EUROBOND"]),
    ("Endeks",         ["ENDEKS"]),
    ("Fon Sepeti",     ["FON SEPETI"]),
    ("Değişken",       ["DEGISKEN"]),
    ("Karma",          ["KARMA"]),
    ("Serbest",        ["SERBEST"]),
    ("Gayrimenkul",    ["GAYRIMENKUL"]),
    ("Girişim",        ["GIRISIM"]),
]


def categorize(title: str) -> str:
    """Map a TEFAS fund title to an umbrella category. MVP: keyword-based on the
    title (0 'Diğer' on the current 899-fund universe). Prefer an official TEFAS
    category field if one becomes available."""
    t = _norm(title)
    for label, keywords in _CATEGORY_RULES:
        if any(k in t for k in keywords):
            return label
    return "Diğer"


# --------------------------------------------------------------------------- #
# Ranking core (pure pandas)
# --------------------------------------------------------------------------- #
_FRONT_COLS = ["title", "category", "league", "rank", "volatility", "sharpe",
               "max_drawdown", "return_1y", "history_days"]


def _rank(pool: pd.DataFrame, rank_by: RankBy, league: str) -> pd.DataFrame:
    """Sort one league's eligible pool and stamp the league + a 1-based rank."""
    if pool.empty:
        return pool.assign(league=league, rank=pd.Series(dtype="int64"))
    if rank_by == RankBy.DRAWDOWN:
        # smallest drop first (max_drawdown is negative); Sharpe breaks ties
        ordered = pool.sort_values(["max_drawdown", "sharpe"], ascending=[False, False])
    else:
        ordered = pool.sort_values("sharpe", ascending=False)
    return ordered.assign(league=league, rank=range(1, len(ordered) + 1))


def screen_funds(features: pd.DataFrame, profile: RiskProfile) -> pd.DataFrame:
    """Eligible funds for `profile`, ranked. Takes the feature table + a RiskProfile,
    returns a ranked DataFrame. The `league` column separates the 'mature' list
    (always) from the 'young' list (present only when the profile asks for it)."""
    df = features.copy()
    if "category" not in df.columns:
        df["category"] = df["title"].map(categorize)

    # eligibility: volatility band + computable vol + optional category whitelist
    elig = df[df["volatility"].notna()
              & df["sharpe"].notna()        #ekledim
              & df["max_drawdown"].notna()  #ekledim
              & (df["volatility"] >= profile.vol_min)
              & (df["volatility"] <= profile.vol_max)]
    if profile.allowed_categories is not None:
        elig = elig[elig["category"].isin(profile.allowed_categories)]

    mature = elig[elig["history_days"] >= profile.min_history_days]
    pools = [_rank(mature, profile.rank_by, "mature")]
    if profile.include_young:
        young = elig[elig["history_days"] < profile.min_history_days]
        pools.append(_rank(young, profile.rank_by, "young"))

    ranked = pd.concat(pools)
    front = [c for c in _FRONT_COLS if c in ranked.columns]
    return ranked[front + [c for c in ranked.columns if c not in front]]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def resolve_profile(profile: "str | RiskProfile") -> RiskProfile:
    """Accept a preset name ('balanced') or a ready RiskProfile (free query / agent)."""
    if isinstance(profile, RiskProfile):
        return profile
    if isinstance(profile, str):
        try:
            return PRESETS[profile]
        except KeyError:
            raise ValueError(f"unknown profile '{profile}'; valid: {list(PRESETS)}")
    raise TypeError(f"profile must be str or RiskProfile, got {type(profile).__name__}")


def recommend(features: pd.DataFrame, profile: "str | RiskProfile" = "balanced",
              top_n: "int | None" = None) -> pd.DataFrame:
    """Ranked recommendations for a profile.

    profile : preset name ('conservative'|'balanced'|'aggressive') or a RiskProfile.
    top_n   : keep only the top N per league (None = all eligible).

    Returns a ranked DataFrame (Pydantic RecommendationResponse wrapping is the
    next layer). For 'aggressive' the result also includes a separate young list
    (rows where league == 'young')."""
    profile = resolve_profile(profile)
    ranked = screen_funds(features, profile)
    if top_n is not None:
        ranked = ranked.groupby("league", group_keys=False).head(top_n)
    return ranked


def load_features(path) -> pd.DataFrame:
    """Convenience reader for the Layer-1 output (funds_features.parquet)."""
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Builder: pandas DataFrame -> typed, JSON-serialisable RecommendationResponse
# (the API/agent seam — pure pandas stays inside, the contract lives at the edge)
# --------------------------------------------------------------------------- #
def _num_or_none(x) -> "float | None":
    """JSON-safe number: NaN/None -> None, else a plain float."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _to_fund(code, row) -> FundRecommendation:
    """Map one ranked DataFrame row to a FundRecommendation."""
    return FundRecommendation(
        code=str(code),
        title=row["title"],
        category=row["category"],
        league=row["league"],
        rank=int(row["rank"]),
        volatility=float(row["volatility"]),
        sharpe=float(row["sharpe"]),
        max_drawdown=float(row["max_drawdown"]),
        return_1y=_num_or_none(row.get("return_1y")),
        history_days=int(row["history_days"]),
    )


def build_response(features: pd.DataFrame, profile: "str | RiskProfile" = "balanced",
                   top_n: "int | None" = None) -> RecommendationResponse:
    """Run the engine and wrap the result in the typed RecommendationResponse the
    API/agent speaks. `total_eligible` counts all funds that matched the profile
    BEFORE the top_n trim, so a consumer can say 'showing N of total_eligible'."""
    profile = resolve_profile(profile)
    ranked = screen_funds(features, profile)
    total_eligible = len(ranked)
    if top_n is not None:
        ranked = ranked.groupby("league", group_keys=False).head(top_n)

    mature = [_to_fund(c, r) for c, r in ranked[ranked["league"] == "mature"].iterrows()]
    young = [_to_fund(c, r) for c, r in ranked[ranked["league"] == "young"].iterrows()]
    return RecommendationResponse(profile=profile, total_eligible=total_eligible,
                                  mature=mature, young=young)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).parent.parent / "data"
    feats = load_features(base / "processed" / "funds_features.parquet")

    for name in ("conservative", "balanced", "aggressive"):
        prof = PRESETS[name]
        res = recommend(feats, name, top_n=5)
        mature = res[res["league"] == "mature"]
        print(f"\n=== {name.upper()}  (vol {prof.vol_min:.0%}-{prof.vol_max:.0%}, "
              f"{prof.rank_by.value}) ===")
        for code, r in mature.iterrows():
            print(f"  {code}  {r['category']:14} vol={r['volatility']*100:4.1f}%  "
                  f"Sharpe={r['sharpe']:5.2f}  {r['title'][:42]}")
        if prof.include_young:
            print("  --- young (separate list) ---")
            for code, r in res[res["league"] == "young"].iterrows():
                print(f"  {code}  {r['category']:14} vol={r['volatility']*100:4.1f}%  "
                      f"Sharpe={r['sharpe']:5.2f}  {r['title'][:42]}")
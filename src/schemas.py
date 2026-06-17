"""
schemas.py — shared Pydantic contracts for the TEFAS recommender.

Single place for the typed data models that cross layer boundaries: the
recommendation engine (recommend.py), the API endpoints, and the LLM explainer
all import from here. Keeping every contract in one module avoids circular
imports and gives one answer to "what shape is the data?".

These models are the agent/API seam: JSON-serialisable in (RiskProfile) and out
(RecommendationResponse), so a tool call or HTTP request speaks them directly.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Input: how the user's risk appetite is expressed
# --------------------------------------------------------------------------- #
class RankBy(str, Enum):
    SHARPE = "sharpe"                     # risk-adjusted return (default)
    DRAWDOWN = "drawdown_then_sharpe"     # smallest drop first, Sharpe as tie-break


class RiskProfile(BaseModel):
    """Resolved NUMERIC criteria. The engine sees only these numbers, never a tier
    name — so the 3 presets, a future free-form query, and an agent all produce the
    same shape and reuse the same engine.

    Risk is a volatility BAND (vol_min..vol_max), not just a ceiling: a ceiling
    alone collapses every profile onto the same low-vol, high-Sharpe funds."""
    name: str
    vol_min: float = Field(default=0.0, ge=0.0)   # band floor (0 for conservative)
    vol_max: float = Field(gt=0.0)                # band ceiling
    min_history_days: int = 365                   # league floor ('mature' threshold)
    include_young: bool = False                   # also return a SEPARATE young list
    rank_by: RankBy = RankBy.SHARPE
    allowed_categories: list[str] | None = None   # None = no category filter

    @model_validator(mode="after")
    def _check_band(self) -> "RiskProfile":
        if self.vol_min >= self.vol_max:
            raise ValueError(f"vol_min ({self.vol_min}) must be < vol_max ({self.vol_max})")
        return self


# --------------------------------------------------------------------------- #
# Output: one fund, and the full response
# --------------------------------------------------------------------------- #
class FundRecommendation(BaseModel):
    """One ranked fund — a flat, JSON-friendly snapshot of the row the engine picked,
    carrying just what a consumer (API / agent / LLM explainer) needs to show and
    explain it."""
    code: str
    title: str
    category: str
    league: Literal["mature", "young"]
    rank: int = Field(ge=1)
    volatility: float = Field(ge=0.0)
    sharpe: float
    max_drawdown: float
    return_1y: float | None = None        # None when the fund is younger than 1y
    history_days: int = Field(ge=0)


class RecommendationResponse(BaseModel):
    """The full answer for one profile. `mature` is the primary list; `young` is the
    separate short-history list (non-empty only when the profile opts in — i.e.
    aggressive). `total_eligible` is how many funds matched the profile before any
    top-N trim, so a consumer can say 'showing 10 of 221'."""
    profile: RiskProfile
    total_eligible: int = Field(ge=0)
    mature: list[FundRecommendation]
    young: list[FundRecommendation] = []
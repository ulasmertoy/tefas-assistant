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

# LLM explainer: TEXT-ONLY output (no numbers, on purpose)
class ExplainedFund(BaseModel):
    """The LLM's explanation for ONE fund — text only, by design. There is NO
    numeric field here: the model physically cannot emit a Sharpe/volatility
    value, so it cannot get one wrong. The real numbers stay in FundRecommendation;
    the two are merged by `code`. `code` is the join key — and also lets us verify
    the LLM didn't invent a fund that wasn't in the input."""
    code: str
    explanation: str = Field(min_length=1, max_length=400)  # a sentence or two, no numbers

class RecommendRequest(BaseModel):
    risk: Literal["Temkinli", "Dengeli", "Agresif"]
    vade: Literal["1 yıldan kısa", "1–3 yıl", "3 yıl+"]
    tur: Literal["Farketmez", "Katılım (faizsiz)", "Hisse ağırlıklı"]
    user_note: str = Field(default="", max_length=500)
    top_n: int = Field(default=5, ge=1, le=20)

class ExplainedResponse(BaseModel):
    """The full text layer the LLM returns for one recommendation set. `summary` is
    one overall rationale for the profile; `funds` carries one ExplainedFund per fund
    we asked it to explain. `note` is an OPTIONAL caveat — filled only when the user's
    request conflicts with the candidates (e.g. wants low volatility but picked an
    aggressive profile); null otherwise. Merged with RecommendationResponse (by `code`)
    to build the on-screen card: engine numbers on top, this text below."""
    summary: str = Field(min_length=1, max_length=600)
    funds: list[ExplainedFund]
    note: str | None = Field(default=None, max_length=400)   # çelişki/uyarı; yoksa null


# MERGE output: engine numbers + LLM text combined (API's final response)
class FundCard(BaseModel):
    """The output of a fund: numbers from the engine + description from the LLM, combined via code.
    `explanation` might be None — if the model has omitted the description for that fund."""
    code: str
    title: str
    category: str
    league: Literal["mature", "young"]
    volatility: float
    sharpe: float
    max_drawdown: float
    return_1y: float | None = None
    explanation: str | None = None

class RecommendationResult(BaseModel):
    """Endpoint'in döndürdüğü nihai paket: genel özet + opsiyonel uyarı +
    birleşmiş fon kartları + 'yüksek getiri, yüksek risk' vitrini (sayı-only)."""
    summary: str
    note: str | None = None
    total_eligible: int = Field(ge=0)
    cards: list[FundCard]
    high_return_flagged: list[FundCard] = []

class RecommendationResponse(BaseModel):
    """The full answer for one profile. `mature` is the primary list; `young` is the
    separate short-history list (non-empty only when the profile opts in — i.e.
    aggressive). `high_return_flagged` is the 'high return, high risk' showcase:
    funds whose volatility EXCEEDS the profile's ceiling but whose 1y return is high
    — surfaced (numbers only, no recommendation) so the user sees why they were cut.
    `total_eligible` is how many funds matched the profile before any top-N trim."""
    profile: RiskProfile
    total_eligible: int = Field(ge=0)
    mature: list[FundRecommendation]
    young: list[FundRecommendation] = []
    high_return_flagged: list[FundRecommendation] = []
"""
explainer.py — Layer 3: the LLM explanation layer.

Sits ON TOP of the deterministic engine (recommend.py). The engine computes the
numbers; this layer turns a user's free-text risk appetite into a grounded,
plain-Turkish explanation of the picked funds. The engine never imports this
file — the dependency points one way (explainer -> recommend -> schemas), so the
deterministic core stays free of any LLM/network dependency.

Three design pillars (each a deliberate, interview-ready choice):
  1. Structured output — the LLM's answer is forced into the Pydantic
     ExplainedResponse schema via a tool, then RE-VALIDATED on our side.
  2. Grounding by construction — ExplainedResponse has NO numeric field, so the
     model physically cannot emit a wrong Sharpe/volatility. Numbers come only
     from the engine; we merge them back by `code`.
  3. Tool use — the model calls the engine (`recommend_funds`) as a tool: it
     chooses the profile, the engine does the math. The model orchestrates; it
     does not compute. This is the "agentic" seam.
"""
import pandas as pd
import anthropic

from recommend import build_response, load_features
from schemas import RecommendationResponse, ExplainedResponse


# Sonnet = sensible default for a tool-use loop (reliable tool calls, fair price).
# For this fairly structured task you can drop to "claude-haiku-4-5-20251001" to
# cut cost; reach for Opus only if quality demands it.
MODEL = "claude-sonnet-4-6"

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # repo kökündeki .env -> ANTHROPIC_API_KEY
# The SDK reads ANTHROPIC_API_KEY from the environment automatically. NEVER put
# the key in code — set it in your shell / .env (and Streamlit secrets on deploy).
client = anthropic.Anthropic()


SYSTEM_PROMPT = """Sen bir TEFAS fon asistanısın. Görevin, kullanıcının risk \
iştahına göre fon önermek ve neden uygun olduklarını sade Türkçe ile açıklamak.

Adımlar:
1. Kullanıcının risk iştahını oku ve en uygun profille ('conservative', \
'balanced' veya 'aggressive') `recommend_funds` aracını çağır.
2. Araç fonları DÖNDÜRDÜKTEN sonra `submit_explanation` aracını çağır: her fon \
için 1-2 cümlelik, neden o profile uyduğunu anlatan bir açıklama + genel bir özet.

KRİTİK KURALLAR:
- Açıklamalarında ASLA somut sayı yazma (Sharpe, volatilite, getiri, düşüş). \
Niteliksel anlat: "düşük oynaklık", "güçlü riske göre getiri", "sınırlı düşüş". \
Sayılar kullanıcıya ayrıca gösteriliyor.
- "sıfır", "hiç", "tam" gibi kesin değer iddialarından kaçın; \
yön/derece olarak anlat (örn. "çok sınırlı düşüş", "son derece düşük oynaklık").
- Yalnızca `recommend_funds`'un döndürdüğü fonları açıkla. Fon UYDURMA.
- Açıklamalar Türkçe olsun."""


# --------------------------------------------------------------------------- #
# Tool definitions (the contract the model calls against)
# --------------------------------------------------------------------------- #
# Tool 1 — the engine, exposed with JSON-friendly params only. We expose the 3
# PRESETS as an enum (not raw vol bands): the model picks a profile, it cannot
# construct a nonsensical band.
RECOMMEND_TOOL = {
    "name": "recommend_funds",
    "description": "Kullanıcının risk profiline göre TEFAS fonlarını sıralayıp "
                   "döndürür. Tüm sayılar (Sharpe, volatilite vb.) buradan gelir.",
    "input_schema": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "enum": ["conservative", "balanced", "aggressive"],
                "description": "Kullanıcının risk iştahına en uygun profil.",
            },
            "top_n": {
                "type": "integer", "minimum": 1,
                "description": "Lig başına kaç fon döndürülsün (örn. 5).",
            },
        },
        "required": ["profile"],
    },
}

# Tool 2 — the model's structured OUTPUT. Its input_schema IS our Pydantic
# ExplainedResponse: the model "calls" this tool to hand us a typed, text-only
# answer. Pydantic generates the JSON schema, so the shape lives in ONE place.
SUBMIT_TOOL = {
    "name": "submit_explanation",
    "description": "Fonlar için sade Türkçe açıklamaları ve genel özeti gönder. "
                   "Sayı içermez — sadece metin.",
    "input_schema": ExplainedResponse.model_json_schema(),
}


def _run_recommend(tool_input: dict, features: pd.DataFrame) -> RecommendationResponse:
    """Execute the recommend_funds tool: call the deterministic engine, return the
    typed response (kept so we can merge the numbers back later)."""
    return build_response(features, profile=tool_input["profile"],
                          top_n=tool_input.get("top_n", 5))


# --------------------------------------------------------------------------- #
# Merge: engine numbers (FundRecommendation) + LLM text (ExplainedFund), by code
# --------------------------------------------------------------------------- #
def merge_by_code(rec: RecommendationResponse, exp: ExplainedResponse) -> list[dict]:
    """Join the engine's numeric funds with the LLM's text on `code`. Every number
    is from the engine, every sentence from the LLM. An explanation whose code is
    NOT in the engine output (a hallucination) is simply dropped — it can never
    reach the screen."""
    text_by_code = {e.code: e.explanation for e in exp.funds}
    cards = []
    for fund in rec.mature + rec.young:
        cards.append({
            "code": fund.code, "title": fund.title, "category": fund.category,
            "league": fund.league, "volatility": fund.volatility,
            "sharpe": fund.sharpe, "max_drawdown": fund.max_drawdown,
            "return_1y": fund.return_1y,
            "explanation": text_by_code.get(fund.code),  # None if model skipped it
        })
    return cards


# --------------------------------------------------------------------------- #
# Public entry point: the tool-use loop
# --------------------------------------------------------------------------- #
def explain(user_query: str, features: pd.DataFrame, max_turns: int = 4
            ) -> "tuple[RecommendationResponse, ExplainedResponse]":
    """Run the agentic loop for one user request: the model calls recommend_funds
    (engine does the math), then submit_explanation (its text-only answer). We
    drive the loop, execute the engine tool, and validate the explanation. Returns
    BOTH the engine response (numbers) and the explanation (text)."""
    messages = [{"role": "user", "content": user_query}]
    rec_response: "RecommendationResponse | None" = None

    for _ in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=[RECOMMEND_TOOL, SUBMIT_TOOL], messages=messages,
        )

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            raise RuntimeError("Model bir araç çağırmadı; prompt'u kontrol et.")

        # Terminal action: the model handed us its explanation -> validate & done.
        for block in tool_uses:
            if block.name == "submit_explanation":
                if rec_response is None:
                    raise RuntimeError("Model açıklamadan önce recommend_funds çağırmadı.")
                return rec_response, ExplainedResponse.model_validate(block.input)

        # Otherwise: run the engine tool and feed the result back.
        tool_results = []
        for block in tool_uses:
            if block.name == "recommend_funds":
                rec_response = _run_recommend(block.input, features)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": rec_response.model_dump_json(),
                })
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"{max_turns} turda açıklama üretilemedi.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).parent.parent / "data"
    feats = load_features(base / "processed" / "funds_features.parquet")

    rec, exp = explain("Çok risk almak istemiyorum, paramı korumak önceliğim.", feats)
    print("ÖZET:", exp.summary, "\n")
    for card in merge_by_code(rec, exp):
        print(f"{card['code']}  vol={card['volatility']*100:.1f}%  Sharpe={card['sharpe']:.2f}")
        print(f"  → {card['explanation']}\n")
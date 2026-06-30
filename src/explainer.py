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
from schemas import (RecommendationResponse, ExplainedResponse,
                     FundCard, RecommendationResult)   
from recommend import build_response, load_features
from pydantic import ValidationError   # dosyanın başına

# Sonnet = sensible default for a tool-use loop (reliable tool calls, fair price).
# For this fairly structured task you can drop to "claude-haiku-4-5-20251001" to
# cut cost; reach for Opus only if quality demands it.
MODEL = "claude-haiku-4-5-20251001"

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
def merge_by_code(rec: RecommendationResponse, exp: ExplainedResponse
                  ) -> RecommendationResult:
    """Engine'in sayılarını (FundRecommendation) LLM'in metniyle (ExplainedFund)
    `code` üzerinden birleştirir. Her sayı engine'den, her cümle LLM'den. Engine
    çıktısında OLMAYAN bir kod (hallucination) sessizce düşer — ekranı göremez."""
    text_by_code = {e.code: e.explanation for e in exp.funds}
    cards = [
        FundCard(
            code=fund.code,
            title=fund.title,
            category=fund.category,
            league=fund.league,
            volatility=fund.volatility,
            sharpe=fund.sharpe,
            max_drawdown=fund.max_drawdown,
            return_1y=fund.return_1y,
            explanation=text_by_code.get(fund.code),   # model atladıysa None
        )
        for fund in rec.mature + rec.young
    ]
    flagged = [
        FundCard(
            code=f.code, title=f.title, category=f.category, league=f.league,
            volatility=f.volatility, sharpe=f.sharpe, max_drawdown=f.max_drawdown,
            return_1y=f.return_1y, explanation=None,   # açıklama yok, bilerek
        )
        for f in rec.high_return_flagged
    ]
    return RecommendationResult(
        summary=exp.summary,
        note=exp.note,
        total_eligible=rec.total_eligible,
        cards=cards,
        high_return_flagged=flagged,
    )


# --------------------------------------------------------------------------- #
# Public entry point: the tool-use loop
# --------------------------------------------------------------------------- #
class NoToolCallError(RuntimeError):
    """Model araç çağırmadı (çelişkili/kapsam dışı sorgu); taşıdığı metin kullanıcıya gösterilir."""

def explain(user_query: str, features: pd.DataFrame, max_turns: int = 6
            ) -> "tuple[RecommendationResponse, ExplainedResponse]":
    """Agentic loop: model recommend_funds'u çağırır (motor hesaplar), sonra
    submit_explanation'ı çağırır (sadece metin). Doğrulama başarısız olursa hatayı
    modele geri besleyip kendini düzeltmesini sağlarız. Motor yanıtı (sayılar) +
    açıklama (metin) birlikte döner."""
    messages = [{"role": "user", "content": user_query}]
    rec_response: "RecommendationResponse | None" = None

    for _ in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=[RECOMMEND_TOOL, SUBMIT_TOOL], messages=messages,
        )

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            raise NoToolCallError(text or "Model bir öneri üretmedi.")
        
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []

        for block in tool_uses:
            if block.name == "recommend_funds":
                rec_response = _run_recommend(block.input, features)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": rec_response.model_dump_json(),
                })
            elif block.name == "submit_explanation":
                if rec_response is None:
                    raise RuntimeError("Model açıklamadan önce recommend_funds çağırmadı.")
                try:
                    return rec_response, ExplainedResponse.model_validate(block.input)
                except ValidationError as e:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "is_error": True,
                        "content": f"Şema hatası: {e}. submit_explanation'ı TÜM zorunlu "
                                   f"alanlarla (özellikle 'summary') tekrar çağır.",
                    })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"{max_turns} turda geçerli açıklama üretilemedi.")

# explain'in HEMEN ALTINA ekle — mevcut explain'i SİLME

_SELECT_SYSTEM = """Sana zaten seçilmiş TEFAS fonları verilecek. Görevin bu fonları
AÇIKLAMAK — fon seçmek/eklemek değil. Her fon için 1-2 cümlelik, neden bu risk
profiline uygun olduğunu anlatan niteliksel bir açıklama, bir de genel bir özet üret.
KURAL: açıklamalarda ASLA sayı yazma (Sharpe, volatilite, getiri, düşüş);
"düşük oynaklık", "güçlü riske göre getiri" gibi niteliksel anlat. Türkçe yaz."""

_SELECT_SYSTEM_NOTE = """Sana profil filtrelerinden geçmiş TEFAS fon adayları ve
kullanıcının kısa bir notu verilecek. Görevin:
- Verilen fonların TÜMÜNÜ açıkla — fon eleme/çıkarma YOK; her fon için 1-2 cümlelik
  niteliksel açıklama yaz.
- 'summary' alanında, kullanıcının notuna göre bu fonların ona ne kadar uyduğunu
  değerlendir; nota en çok uyanları öne çıkar.
- Kullanıcının notu eldeki adaylarla ÇELİŞİYORSA (ör. "yüksek getiri ama düşük
  oynaklık" istemiş, oysa agresif profil seçtiği için tüm adaylar yüksek oynaklıklı),
  bu çelişkiyi 'note' alanına kısa ve net yaz: kullanıcıyı uyar ve ne yapabileceğini
  söyle (ör. "düşük oynaklık önceliğinse Dengeli ya da Temkinli profili seç").
  Çelişki yoksa 'note' alanını boş (null) bırak.
KURAL: hiçbir yerde sayı yazma (Sharpe, volatilite, getiri, düşüş); "düşük oynaklık",
"güçlü riske göre getiri" gibi niteliksel anlat. Yüksek oynaklıklı bir fonu "düşük
oynaklık" diye SUNMA — verilere sadık kal. Türkçe yaz."""

def explain_selected(rec: "RecommendationResponse", user_note: str = "",
                     max_turns: int = 3) -> "ExplainedResponse":
    """Not boşsa: verilen fonların TÜMÜNÜ açıklar (eski davranış).
    Not doluysa: adaylar arasından nota göre en uygunları SEÇİP açıklar."""
    note = (user_note or "").strip()
    if note:
        system = _SELECT_SYSTEM_NOTE
        user_msg = f"Kullanıcının notu: {note}\n\nAday fonlar:\n{rec.model_dump_json()}"
    else:
        system = _SELECT_SYSTEM
        user_msg = "Aşağıdaki seçilmiş fonları açıkla:\n" + rec.model_dump_json()

    messages = [{"role": "user", "content": user_msg}]
    for _ in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=3000, system=system,
            tools=[SUBMIT_TOOL],
            tool_choice={"type": "tool", "name": "submit_explanation"},
            messages=messages,
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        try:
            return ExplainedResponse.model_validate(block.input)
        except ValidationError as e:
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"Şema hatası: {e}. submit_explanation'ı TÜM zorunlu "
                           f"alanlarla (özellikle 'summary') tekrar çağır.",
            }]})
    raise RuntimeError(f"{max_turns} turda geçerli açıklama üretilemedi.")

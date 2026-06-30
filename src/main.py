"""
main.py — FastAPI servis katmanı.
Engine (recommend.py) + LLM explainer (explainer.py) tek bir HTTP endpoint'inde
birleşiyor. Streamlit artık doğrudan Python import etmek yerine bu API'yi çağırır.
"""
import logging                          # YENİ: print yerine "ciddi" kayıt tutmak için
import time                             # YENİ: her isteğin kaç saniye sürdüğünü ölçmek için
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request

from recommend import build_profile, build_response, load_features
from explainer import explain_selected, merge_by_code
from schemas import RecommendRequest, RecommendationResult


# ── Logging kurulumu ─────────────────────────────────────────────
# Tek satırlık temel ayar: zaman damgası + seviye + mesaj formatı.
# Artık print() yerine logger.info(...) / logger.warning(...) kullanacağız.
# Bir şey patladığında geriye dönüp "ne oldu, ne zaman, hangi adımda" diye bakabilmenin tek yolu bu.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tefas-api")

# Veri sunucu açılırken BİR KEZ yüklenir, bu sözlükte tutulur.
state: dict = {}

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "funds_features.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["features"] = load_features(DATA_PATH)            # açılışta yükle
    logger.info("Veri yüklendi: %d fon", len(state["features"]))  # YENİ: açılış kaydı
    yield                                                   # sunucu burada çalışır
    state.clear()                                           # kapanışta temizle
    logger.info("Sunucu kapandı, state temizlendi.")        # YENİ: kapanış kaydı


app = FastAPI(title="TEFAS Fund Recommender", lifespan=lifespan)


# ── Latency middleware ───────────────────────────────────────────
# "Middleware" = her isteğin ÖNÜNE ve ARKASINA takılan küçük bir kod.
# Burada yaptığı tek şey: isteğin başlangıç saatini al → endpoint çalışsın →
# bitiş saatini al → farkı (süreyi) logla. ~10 satırlık observability.
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)                     # asıl endpoint burada çalışır
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.0f ms)",
                request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ── Dependency: veriyi enjekte et ────────────────────────────────
# Global state'i doğrudan okumak yerine bunu kullanmak FastAPI'nin idiomatik yolu.
# Avantajı: testte bu fonksiyonu override edip sahte (mock) veri besleyebilirsin,
# endpoint'in kodunu hiç değiştirmeden. Aynı global'i okuyor, sadece daha temiz.
def get_features():
    return state["features"]


@app.get("/health")
def health():
    """Sunucu ayakta mı, veri yüklü mü?"""
    return {"status": "ok", "funds_loaded": len(state.get("features", []))}


@app.post("/recommend")
def recommend_endpoint(
    request: RecommendRequest,
    features=Depends(get_features),           # YENİ: state["features"] yerine enjeksiyon
) -> RecommendationResult:
    """Üç UI cevabını alır → profil kurar → engine çalışır → LLM açıklar → MERGE."""
    profile = build_profile(request.risk, request.vade, request.tur)
    engine_result = build_response(features, profile, top_n=request.top_n)

    if not engine_result.mature and not engine_result.young:
        logger.warning("Profile uygun fon yok: %s", profile)  # YENİ: bağlamlı uyarı
        raise HTTPException(status_code=404, detail="Bu profile uygun fon bulunamadı.")

    # ── LLM fallback ─────────────────────────────────────────────
    # MERGE mimarisinin özü: deterministik sayılar doğrunun kaynağı, LLM sadece süs.
    # O yüzden LLM çökse (rate limit / network / API hatası) bile sayısal öneri DÖNMELİ.
    # Kullanıcı çıplak bir 500 görmek yerine geçerli öneriyi alır; sadece açıklama prose'u eksik olur.
    try:
        explained = explain_selected(engine_result, user_note=request.user_note)
    except Exception as exc:
        logger.error("LLM açıklayıcı çöktü, sayısal sonuçla devam: %s", exc)
        explained = None                                    # prose boş, sayılar tam

    return merge_by_code(engine_result, explained)
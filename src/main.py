"""
main.py — FastAPI servis katmanı.
Engine (recommend.py) + LLM explainer (explainer.py) tek bir HTTP endpoint'inde
birleşiyor. Streamlit artık doğrudan Python import etmek yerine bu API'yi çağırır.
"""
import logging
import os                          # YENİ: print yerine "ciddi" kayıt tutmak için
import time                             # YENİ: her isteğin kaç saniye sürdüğünü ölçmek için
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from recommend import build_profile, build_response, load_features
from explainer import explain_selected, merge_by_code
from schemas import RecommendRequest, RecommendationResult
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# Metrik tablosu ayri bir data repo'da yasiyor, Actions her gun guncelliyor.
# Boylece veri guncellemesi ile kod deploy'u birbirinden ayrisiyor.
FEATURES_URL = os.getenv(
    "FEATURES_URL",
    "https://raw.githubusercontent.com/ulasmertoy/tefas-data/main/"
    "data/processed/funds_features.parquet",
)
LOCAL_FALLBACK = Path(__file__).parent.parent / "data" / "processed" / "funds_features.parquet"


def load_features_with_fallback():
    """Once uzak kaynagi dene, olmazsa image'daki kopyaya dus.

    Veri guncelligi onemli ama SERVISIN AYAKTA KALMASI daha onemli: GitHub'a
    ulasilamadiginda dunku veriyle calisan bir API, hic acilmayan bir API'den
    iyidir. Sessizce degil - log'da hangi kaynagin kullanildigi yaziyor.
    """
    try:
        df = load_features(FEATURES_URL)
        logger.info("Veri kaynagi: UZAK (%s)", FEATURES_URL)
        return df
    except Exception as exc:
        logger.error("Uzak veri okunamadi (%s) - yerel kopyaya dusuluyor", exc)
        df = load_features(LOCAL_FALLBACK)
        logger.warning("Veri kaynagi: YEREL yedek - guncel olmayabilir")
        return df

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["features"] = load_features_with_fallback()
    feats = state["features"]
    data_end = feats["data_end"].iloc[0] if "data_end" in feats.columns else None
    logger.info("Veri yuklendi: %d fon | veri sonu: %s", len(feats), data_end)
    yield                                                   # sunucu burada çalışır
    state.clear()                                           # kapanışta temizle
    logger.info("Sunucu kapandı, state temizlendi.")        # YENİ: kapanış kaydı

# IP başına istek sayacı, bellekte tutulur (tek worker → paylaşım derdi yok).
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="TEFAS Fund Recommender", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React (CRA) — yerel geliştirme
        "http://localhost:5173",     # React (Vite) — yerel geliştirme
        "http://localhost:8501",     # Streamlit — yerel geliştirme
    ],
    allow_origin_regex=r"https://.*\.streamlit\.app",   # Streamlit Cloud
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    feats = state.get("features")
    data_end = None
    if feats is not None and "data_end" in feats.columns and len(feats):
        data_end = str(feats["data_end"].iloc[0].date())
    return {"status": "ok",
            "funds_loaded": len(feats) if feats is not None else 0,
            "data_end": data_end}

@app.post("/recommend")
@limiter.limit("10/minute")                    # ← @app.post'un ALTINDA olmalı, sırası önemli
def recommend_endpoint(
    request: Request,                          # Starlette Request — slowapi IP'yi buradan okur, adı 'request' ZORUNLU
    payload: RecommendRequest,                 # gövde artık 'payload'
    features=Depends(get_features),
) -> RecommendationResult:
    """Üç UI cevabını alır → profil kurar → engine çalışır → LLM açıklar → MERGE."""
    profile = build_profile(payload.risk, payload.vade, payload.tur)
    engine_result = build_response(features, profile, top_n=payload.top_n)

    if not engine_result.mature and not engine_result.young:
        logger.warning("Profile uygun fon yok: %s", profile)
        raise HTTPException(status_code=404, detail="Bu profile uygun fon bulunamadı.")

# ── LLM fallback ─────────────────────────────────────────────
    # MERGE mimarisinin özü: deterministik sayılar doğrunun kaynağı, LLM sadece süs.
    # O yüzden LLM çökse (rate limit / network / API hatası) bile sayısal öneri DÖNMELİ.
    # Kullanıcı çıplak bir 500 görmek yerine geçerli öneriyi alır; sadece açıklama prose'u eksik olur.

    try:
        explained = explain_selected(engine_result, user_note=payload.user_note)
    except Exception as exc:
        logger.error("LLM açıklayıcı çöktü, sayısal sonuçla devam: %s", exc)
        explained = None

    return merge_by_code(engine_result, explained)

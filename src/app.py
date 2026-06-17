"""
app.py — minimal Streamlit UI for the TEFAS fund recommender.

Run from a terminal (NOT the VS Code play button):
    streamlit run src/app.py

It imports the Layer-2 engine (recommend.py) directly and reads the Layer-1
output (funds_features.parquet). No API/Docker needed for this MVP — Streamlit
calls the Python engine in-process and renders the result.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import recommend as rc

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "funds_features.parquet"

# UI label -> preset key
PROFILE_LABELS = {
    "Temkinli": "conservative",
    "Dengeli": "balanced",
    "Agresif": "aggressive",
}

st.set_page_config(page_title="TEFAS Fon Asistanı", page_icon="📊", layout="wide")


@st.cache_data
def get_features(path: str) -> pd.DataFrame:
    """Load the feature table once and cache it across reruns."""
    return rc.load_features(path)


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ranked engine output -> a clean, Turkish-headed display table."""
    return pd.DataFrame({
        "Sıra": df["rank"].values,
        "Kod": df.index,
        "Fon": df["title"].values,
        "Kategori": df["category"].values,
        "Vol %": (df["volatility"] * 100).round(1).values,
        "Sharpe": df["sharpe"].round(2).values,
        "Max Düşüş %": (df["max_drawdown"] * 100).round(1).values,
        "1Y Getiri %": (df["return_1y"] * 100).round(1).values,
        "Geçmiş (gün)": df["history_days"].values,
    })


# --- header ---
st.title("📊 TEFAS Fon Tarama Asistanı")
st.caption("Risk profiline göre Türk yatırım fonlarını tarar ve risk-ayarlı (Sharpe) sıralar.")

# --- data ---
if not FEATURES_PATH.exists():
    st.error(f"Özellik tablosu bulunamadı:\n`{FEATURES_PATH}`\n\n"
             "Önce `feature_engineering.py`'yi çalıştırıp `funds_features.parquet` üret.")
    st.stop()

features = get_features(str(FEATURES_PATH))

# --- controls ---
c1, c2 = st.columns([2, 1])
with c1:
    label = st.radio("Risk profili", list(PROFILE_LABELS), horizontal=True)
with c2:
    top_n = st.slider("Kaç fon gösterilsin?", 5, 50, 15)

profile_key = PROFILE_LABELS[label]
prof = rc.PRESETS[profile_key]
st.caption(f"**{label}** — volatilite bandı %{prof.vol_min * 100:.0f}–%{prof.vol_max * 100:.0f} · "
           f"sıralama: {prof.rank_by.value}")

# --- results ---
result = rc.recommend(features, profile_key, top_n=top_n)
mature = result[result["league"] == "mature"]
young = result[result["league"] == "young"]

st.subheader(f"Önerilen fonlar ({len(mature)})")
st.dataframe(display_table(mature), hide_index=True, use_container_width=True)

if prof.include_young and len(young):
    st.subheader(f"⚠️ Genç fonlar — ayrı liste ({len(young)})")
    st.caption("Geçmişi 1 yıldan kısa: metrikleri kırılgan, yüksek potansiyel ama az kanıt.")
    st.dataframe(display_table(young), hide_index=True, use_container_width=True)

# --- disclaimer (SPK) ---
st.divider()
st.caption("ℹ️ Bu araç yalnızca **bilgilendirme ve tarama** amaçlıdır; **yatırım tavsiyesi değildir.** "
           "Geçmiş performans gelecek getiriyi garanti etmez. Veri kaynağı: TEFAS.")
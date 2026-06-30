"""
app.py — TEFAS Fon Tarama Asistanı (Streamlit UI)

Çalıştır:
    streamlit run src/app.py

Artık motoru DOĞRUDAN çağırmaz: FastAPI servisine (main.py) HTTP isteği atar.
Tüm iş (build_profile → build_response → explain_selected → merge_by_code) API
tarafında yapılır; bu dosya yalnızca formu gösterir, isteği atar, sonucu basar.

Form seçenekleri (risk/vade/tür etiketleri) saf veridir — istemcide kalır.
"""
import os

import pandas as pd
import requests
import streamlit as st

# Form etiketleri saf veri (motor değil), istemcide kalabilir.
# recommend.py içindeki sözlüklerle BİREBİR aynı anahtarlar olmalı.
PROFILE_LABELS = ["Temkinli", "Dengeli", "Agresif"]
VADE_LABELS = ["1 yıldan kısa", "1–3 yıl", "3 yıl+"]
TUR_LABELS = ["Farketmez", "Katılım (faizsiz)", "Hisse ağırlıklı"]

# API adresi. Lokalde varsayılan; Docker/deploy'da ortam değişkeniyle ezilir.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TEFAS Fon Asistanı", page_icon="📊", layout="wide")


# -------------------------------------------------------------- API çağrısı  #
def fetch_recommendation(risk: str, vade: str, tur: str,
                         user_note: str, top_n: int) -> dict:
    """FastAPI /recommend endpoint'ine istek atar, JSON sözlüğü döndürür.
    Hataları çağıran tarafa yükseltir (orada kullanıcıya gösterilir)."""
    resp = requests.post(
        f"{API_URL}/recommend",
        json={"risk": risk, "vade": vade, "tur": tur,
              "user_note": user_note, "top_n": top_n},
        timeout=60,   # LLM çağrısı yavaş olabilir
    )
    resp.raise_for_status()   # 4xx/5xx -> HTTPError
    return resp.json()


# -------------------------------------------------------------- yardımcı --- #
def cards_to_table(cards: list[dict]) -> pd.DataFrame:
    """Kart listesini temiz, Türkçe başlıklı bir tabloya çevir."""
    return pd.DataFrame([{
        "Sıra": i + 1,
        "Kod": c["code"],
        "Fon": c["title"],
        "Kategori": c["category"],
        "Vol %": round(c["volatility"] * 100, 1),
        "Sharpe": round(c["sharpe"], 2),
        "Max Düşüş %": round(c["max_drawdown"] * 100, 1),
        "1Y Getiri %": round(c["return_1y"] * 100, 1) if c["return_1y"] is not None else None,
    } for i, c in enumerate(cards)])


def render_cards(cards: list[dict]) -> None:
    """Önce tablo (sayılar), altında her fonun açıklaması (metin)."""
    mature = [c for c in cards if c["league"] == "mature"]
    young = [c for c in cards if c["league"] == "young"]

    if mature:
        st.subheader(f"Önerilen fonlar ({len(mature)})")
        st.dataframe(cards_to_table(mature), hide_index=True, width="stretch")
        for c in mature:
            if c["explanation"]:
                with st.expander(f"💬 {c['code']} — {c['title']}"):
                    st.write(c["explanation"])

    if young:
        st.subheader(f"⚠️ Genç fonlar ({len(young)})")
        st.caption("Geçmişi eşiğin altında: metrikleri kırılgan, yüksek potansiyel ama az kanıt.")
        st.dataframe(cards_to_table(young), hide_index=True, width="stretch")
        for c in young:
            if c["explanation"]:
                with st.expander(f"💬 {c['code']} — {c['title']}"):
                    st.write(c["explanation"])

def render_flagged(flagged: list[dict]) -> None:
    """'Yüksek getiri, yüksek risk' vitrini: sayılar + sabit uyarı. Açıklama YOK —
    bunlar öneri değil, kullanıcının 'neden yüksek getiriler yok?' sorusuna cevap."""
    if not flagged:
        return

    st.subheader(f"🔥 Yüksek getiri, yüksek risk ({len(flagged)})")
    st.warning(
        "Bu fonlar daha yüksek getiri sağladı, ancak oynaklıkları agresif profilin "
        "üst sınırını aştığı için **önerilmiyorlar.** Yüksek getiri genellikle yüksek "
        "risk ve sert düşüşlerle gelir — Sharpe oranı ve maksimum düşüş sütunlarına bakın."
    )
    st.dataframe(cards_to_table(flagged), hide_index=True, width="stretch")

# --------------------------------------------------------------- başlık ---- #
st.title("📊 TEFAS Fon Tarama Asistanı")
st.caption("Birkaç soruyla risk profilini al, fonları risk-ayarlı (Sharpe) tara; "
           "eklediğin notla sana en uygunları seçip sade bir açıklama üret.")


# ----------------------------------------------------- TEK FORM: öneri ---- #
st.subheader("Sana uygun fonları bulalım")

with st.form("oneri"):
    col1, col2 = st.columns(2)
    with col1:
        risk = st.radio("1) Risk iştahın?", PROFILE_LABELS, index=1, horizontal=True)
        vade = st.radio("2) Yatırım vaden?", VADE_LABELS, index=1, horizontal=True)
    with col2:
        tur = st.radio("3) Fon türü tercihin?", TUR_LABELS, horizontal=True)
        top_n = st.slider("Kaç aday taransın?", 5, 30, 12)

    user_note = st.text_area(
        "4) İstersen bir cümle ekle — neyi önemsiyorsun? (boş bırakabilirsin)",
        placeholder="örn. sert düşüşlerden kaçınmak istiyorum, son 1 yıl getirisi de iyi olsun",
        height=80,
    )
    submitted = st.form_submit_button("Bana fon öner")

if submitted:
    note = (user_note or "").strip()
    spinner_msg = "Sana en uygunları seçip açıklıyorum..." if note else "Adaylar taranıyor..."
    with st.spinner(spinner_msg):
        try:
            result = fetch_recommendation(risk, vade, tur, note, top_n)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 404:
                st.warning("Bu filtrelere uyan fon bulamadım. "
                           "Vade ya da tür kısıtını gevşetmeyi dene.")
            elif status == 422:
                st.error("Geçersiz seçim gönderildi. Form değerlerini kontrol et.")
            else:
                st.error(f"Sunucu hatası ({status}). Az sonra tekrar dene.")
            st.stop()
        except requests.exceptions.RequestException:
            st.error("API'ye ulaşılamadı. Sunucunun çalıştığından emin ol "
                     f"(`{API_URL}`).")
            st.stop()

    if result["summary"]:
        st.markdown(result["summary"])
    if result["note"]:          # çelişki/uyarı varsa sarı kutuda öne çıkar
        st.warning(result["note"])
    render_cards(result["cards"])
    render_flagged(result.get("high_return_flagged", []))


# --------------------------------------------------------------- footer ---- #
st.divider()
st.caption("ℹ️ Bu araç yalnızca **bilgilendirme ve tarama** amaçlıdır; **yatırım tavsiyesi "
           "değildir.** Geçmiş performans gelecek getiriyi garanti etmez. Veri kaynağı: TEFAS.")
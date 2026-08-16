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
import plotly.graph_objects as go

# Form etiketleri saf veri (motor değil), istemcide kalabilir.
# recommend.py içindeki sözlüklerle BİREBİR aynı anahtarlar olmalı.
PROFILE_LABELS = ["Temkinli", "Dengeli", "Agresif"]
VADE_LABELS = ["1 yıldan kısa", "1–3 yıl", "3 yıl+"]
TUR_LABELS = ["Farketmez", "Katılım (faizsiz)", "Hisse ağırlıklı"]

# Erişim etiketleri — schemas.ACCESS_LABELS ile birebir aynı olmalı.
ACCESS_LABELS = {
    "public": "Herkese açık",
    "qualified": "Nitelikli yatırımcı",
    "private": "Özel fon",
}

# API adresi. Lokalde varsayılan; Docker/deploy'da ortam değişkeniyle ezilir.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
@st.cache_resource
def wake_backend():
    """Sayfa ilk açıldığında backend'i uyandır (Render free tier uyku sorunu)."""
    try:
        requests.get(f"{API_URL}/health", timeout=3)
    except Exception:
        pass  # uyanmadıysa da devam et; asıl istek zaten tekrar deneyecek
    return True

wake_backend()

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
        "Erişim": ACCESS_LABELS.get(c.get("access", "public"), "—"),
        "Vol %": round(c["volatility"] * 100, 1),
        "Sharpe": round(c["sharpe"], 2),
        "Max Düşüş %": round(c["max_drawdown"] * 100, 1),
        "1A %": round(c["return_1m"] * 100, 1) if c.get("return_1m") is not None else None,
        "3A %": round(c["return_3m"] * 100, 1) if c.get("return_3m") is not None else None,
        "6A %": round(c["return_6m"] * 100, 1) if c.get("return_6m") is not None else None,
        "YBB %": round(c["return_ytd"] * 100, 1) if c.get("return_ytd") is not None else None,
        "1Y Getiri %": round(c["return_1y"] * 100, 1) if c["return_1y"] is not None else None,
    } for i, c in enumerate(cards)])

def _fund_details(c: dict) -> None:
    """Expander içeriği: LLM açıklaması (varsa) + standart periyot getirileri +
    deterministik dönemsel (rejim) tablo. LLM çökse bile (explanation None) her
    ikisi de motordan gelir, görünür."""
    if c["explanation"]:
        st.write(c["explanation"])

    st.caption("Standart getiriler")
    st.dataframe(pd.DataFrame([{
        "Periyot": label,
        "Getiri %": round(c[key] * 100, 1) if c.get(key) is not None else "veri yok",
    } for label, key in [
        ("1 Ay", "return_1m"), ("3 Ay", "return_3m"), ("6 Ay", "return_6m"),
        ("Yılbaşından bugüne", "return_ytd"), ("1 Yıl", "return_1y"),
    ]]), hide_index=True, width="stretch")

    regime = c.get("regime") or []
    if regime:
        st.caption("Dönemsel davranış")
        st.dataframe(pd.DataFrame([{
            "Dönem": r["label"],
            "Getiri %": round(r["ret"] * 100, 1) if r["ret"] is not None else "veri yok",
            "Oynaklık %": round(r["vol"] * 100, 1) if r["vol"] is not None else "veri yok",
        } for r in regime]), hide_index=True, width="stretch")

def render_metrics(cards: list[dict], total_eligible: int) -> None:
    """Önerilen fonların özet metrikleri: 4 kart halinde üst şerit."""
    mature = [c for c in cards if c["league"] == "mature"]
    pool = mature or cards
    if not pool:
        return
    avg_sharpe = sum(c["sharpe"] for c in pool) / len(pool)
    avg_vol = sum(c["volatility"] for c in pool) / len(pool)
    avg_dd = sum(c["max_drawdown"] for c in pool) / len(pool)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Taranan fon", f"{total_eligible}")
    k2.metric("Ortalama Sharpe", f"{avg_sharpe:.2f}")
    k3.metric("Ortalama oynaklık", f"%{avg_vol * 100:.1f}")
    k4.metric("Ortalama max düşüş", f"%{abs(avg_dd) * 100:.1f}")


def render_scatter(cards: list[dict], flagged: list[dict]) -> None:
    """Risk-getiri saçılımı: x=oynaklık, y=1y getiri."""
    def _points(items):
        xs, ys, txt = [], [], []
        for c in items:
            if c["return_1y"] is None:
                continue
            xs.append(c["volatility"] * 100)
            ys.append(c["return_1y"] * 100)
            txt.append(f"{c['code']} — {c['title']}<br>Sharpe: {c['sharpe']:.2f} · Max düşüş: %{c['max_drawdown']*100:.1f}")
        return xs, ys, txt
    fig = go.Figure()
    mx, my, mt = _points(cards)  # tüm önerilenler (mature + young)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", name="Önerilen",
        marker=dict(size=13, color="#E8B339", line=dict(width=1, color="#0F1419")),
        text=mt, hovertemplate="%{text}<br>Oynaklık: %{x:.1f}%<br>Getiri: %{y:.1f}%<extra></extra>"))
    if flagged:
        fx, fy, ft = _points(flagged)
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="markers", name="Yüksek risk",
            marker=dict(size=13, color="#D9534F", symbol="diamond", line=dict(width=1, color="#0F1419")),
            text=ft, hovertemplate="%{text}<br>Oynaklık: %{x:.1f}%<br>Getiri: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Risk – Getiri Dağılımı",
        xaxis_title="Oynaklık (%)  →  daha riskli", yaxis_title="1 yıllık getiri (%)",
        plot_bgcolor="#0F1419", paper_bgcolor="#0F1419", font=dict(color="#E6E9ED"),
        xaxis=dict(gridcolor="#2A333D"), yaxis=dict(gridcolor="#2A333D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=420, margin=dict(t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)


def render_cards(cards: list[dict]) -> None:
    """Önce tablo (sayılar), altında her fonun açıklaması (metin)."""
    mature = [c for c in cards if c["league"] == "mature"]
    young = [c for c in cards if c["league"] == "young"]

    if mature:
        st.subheader(f"Önerilen fonlar ({len(mature)})")
        st.dataframe(cards_to_table(mature), hide_index=True, width="stretch")
        for c in mature:
            if c["explanation"] or c.get("regime"):
                with st.expander(f"💬 {c['code']} — {c['title']}"):
                    _fund_details(c)

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

def render_restricted(items: list[dict]) -> None:
    """Aynı risk profiline uyan ama satın alınamayan fonlar. ÖNERİ DEĞİL —
    şeffaflık için: kullanıcı 'bu bantta başka ne var?' diye merak ederse
    görsün, ama alamayacağı bir fonu tavsiye olarak almasın."""
    if not items:
        return

    st.subheader(f"🔒 Erişimi kısıtlı fonlar ({len(items)})")
    st.info(
        "Bu fonlar seçtiğin risk profiline uyuyor ancak **herkes tarafından satın "
        "alınamaz.** Serbest fonlar yalnızca nitelikli yatırımcılara, özel fonlar "
        "ise belirli bir yatırımcı grubuna sunulur. Bilgi amaçlı listeleniyorlar; "
        "önerilen fonlar listesine dahil değiller."
    )
    st.dataframe(cards_to_table(items), hide_index=True, width="stretch")


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
        top_n = st.slider("Kaç aday taransın?", 3, 10, 5)

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

    flagged = result.get("high_return_flagged", [])
    render_metrics(result["cards"], result.get("total_eligible", 0))
    st.divider()
    if result["summary"]:
        st.markdown(result["summary"])
    if result["note"]:          # çelişki/uyarı varsa sarı kutuda öne çıkar
        st.warning(result["note"])
    render_scatter(result["cards"], flagged)
    render_cards(result["cards"])
    render_flagged(flagged)
    render_restricted(result.get("qualified_only", []))


# --------------------------------------------------------------- footer ---- #
st.divider()
st.caption("ℹ️ Bu araç yalnızca **bilgilendirme ve tarama** amaçlıdır; **yatırım tavsiyesi "
           "değildir.** Geçmiş performans gelecek getiriyi garanti etmez. Veri kaynağı: TEFAS.")
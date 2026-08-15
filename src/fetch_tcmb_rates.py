"""
fetch_tcmb_rates.py — TCMB ağırlıklı ortalama fonlama maliyetini (AOFM) EVDS'ten
çeker; Sharpe/Sortino'nun risksiz faiz serisi bu.

Canlı pipeline için üç değişiklik yapıldı:
  1. END_DATE artık sabit değil — bugüne kadar çekiyor. Eski sürümde 10-07-2026'da
     donmuştu ve metrics.risk_free_series() ffill kullandığı için bu SESSİZ bir
     hataydı: hata fırlatmaz, NaN üretmez, son bilinen oranı sonsuza kadar taşır.
     Sharpe yavaşça kayar, kimse fark etmez.
  2. Yazmadan önce doğrulama kapısı var. EVDS bozuk/boş dönerse iyi tabloyu ezmez.
  3. refresh_rates() çağrılabilir bir fonksiyon — ingest.py günlük işte kullanıyor.

Çalıştır (elle):  python src/fetch_tcmb_rates.py
EVDS anahtarı .env'den okunuyor (evdspy hallediyor).
"""
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from evdspy import get_series

logger = logging.getLogger(__name__)

SERIES_CODE = "TP.APIFON4"      # TCMB ağırlıklı ortalama fonlama maliyeti
START_DATE = "01-05-2021"       # EVDS formatı: GG-AA-YYYY
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "raw" / "tcmb_rates.parquet"

# --- Kapı eşikleri ---
RATE_MIN, RATE_MAX = 0.01, 1.50   # ondalık: 0.19 = %19. Bandın dışı -> birim hatası
MIN_POINTS = 12                   # en az 12 aylık nokta gelmeli


def _today_evds() -> str:
    """Bugünün tarihi EVDS'in beklediği GG-AA-YYYY formatında."""
    return date.today().strftime("%d-%m-%Y")


def _pick_column(df: pd.DataFrame, candidates: list[str], numeric: bool) -> str:
    """evdspy sürümüne göre kolon adı değişebiliyor (TP.APIFON4 / TP_APIFON4 ...).
    Adı tahmin etmek yerine bulalım: önce bilinen adları dene, sonra tipe göre ara."""
    for c in candidates:
        if c in df.columns:
            return c
    if numeric:
        for c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                return c
    raise ValueError(f"Beklenen kolon bulunamadı. Gelen kolonlar: {list(df.columns)}")


def fetch_policy_rate(start: str = START_DATE, end: "str | None" = None) -> pd.DataFrame:
    """EVDS'ten aylık seriyi çeker, ['date', 'rate'] şeklinde temiz tablo döndürür.
    `rate` ondalıktır (19.0 -> 0.19). Dosyaya YAZMAZ."""
    end = end or _today_evds()
    logger.info("EVDS %s çekiliyor: %s -> %s (aylık)", SERIES_CODE, start, end)

    raw = get_series(SERIES_CODE, start_date=start, end_date=end,
                     frequency="monthly", cache=False)
    if raw is None or len(raw) == 0:
        raise ValueError("EVDS boş sonuç döndürdü")

    date_col = _pick_column(raw, ["Tarih_string", "Tarih", "date", "month"], numeric=False)
    rate_col = _pick_column(raw, [SERIES_CODE, SERIES_CODE.replace(".", "_")], numeric=True)

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "rate": pd.to_numeric(raw[rate_col], errors="coerce") / 100.0,
    })
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    logger.info("Çekildi: %d aylık nokta | %s -> %s",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df


def check_rates(df: pd.DataFrame) -> list[str]:
    """Yazmadan önceki kapı. Boş liste dönerse temiz."""
    if df is None or df.empty:
        return ["boş tablo"]

    problems = []

    if len(df) < MIN_POINTS:
        problems.append(f"sadece {len(df)} nokta geldi (eşik {MIN_POINTS})")

    n_null = int(df["rate"].isna().sum())
    if n_null:
        problems.append(f"{n_null} satırda rate null")

    valid = df["rate"].dropna()
    out_of_band = valid[(valid < RATE_MIN) | (valid > RATE_MAX)]
    if len(out_of_band):
        problems.append(
            f"{len(out_of_band)} oran bandın dışında "
            f"[{RATE_MIN}, {RATE_MAX}] — örn. {out_of_band.iloc[0]:.4f} "
            f"(muhtemelen /100 iki kez ya da hiç uygulanmadı)")

    n_dup = int(df["date"].duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} mükerrer tarih")

    if df["date"].max() > pd.Timestamp.today().normalize():
        problems.append(f"gelecek tarihli nokta: {df['date'].max().date()}")

    return problems


def refresh_rates(out_path: Path = OUT_FILE) -> pd.DataFrame:
    """Çek -> kapı -> (küçülme kontrolü) -> atomik yaz. Sorun varsa ValueError
    fırlatır ve MEVCUT DOSYAYA DOKUNMAZ.

    Küçülme kontrolü fon verisindekiyle aynı mantık: aylık bir seri asla
    kısalmamalı. Kısaldıysa EVDS eksik dönmüştür, iyi tabloyu ezmeyelim.
    """
    df = fetch_policy_rate()

    problems = check_rates(df)
    if problems:
        raise ValueError("rf kapısı düştü: " + "; ".join(problems))

    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(df) < len(old):
            raise ValueError(
                f"rf kapısı düştü: nokta sayısı düştü ({len(old)} -> {len(df)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)                       # atomik: yarım dosya kalmaz

    logger.info("Yazıldı: %s | %d nokta, son: %s (%.2f%%)",
                out_path, len(df), df["date"].max().date(), df["rate"].iloc[-1] * 100)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    result = refresh_rates()
    print(result.tail(6).to_string(index=False))
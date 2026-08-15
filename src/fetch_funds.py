"""
fetch_funds.py — TEFAS ham veri çekme katmanı (pytefas).

Neden kütüphane değişti: TEFAS 2026'da eski BindHistory* uçlarını kapattı.
tefas-crawler yeni API'ye uyarlandı ama yeni backend fiyatları fon bazında
verdiği için, isim vermeden çağrıldığında fon başına bir HTTP isteğine dağılıyor
(900+ istek) ve pay/kişi sayısı kolonlarını artık döndürmüyor.

pytefas yeni resmi uçları kullanıyor: fund_code vermezsen tarihe göre TÜM fonları
tek çağrıda getiriyor, pay ve kişi sayısı dahil. TEFAS'ın dakikada 6 istek
sınırını ve tek istekteki ~1 ay limitini kendisi yönetiyor (28 günlük chunk).

TEK sorumluluk: tarih aralığı al, temiz DataFrame döndür. Dosya YAZMAZ,
birleştirme YAPMAZ — onlar ingest.py'nin işi.

DİKKAT — 5 yıl duvarı: yeni API sabit bir geriye bakış penceresi uyguluyor
(en fazla 5 yıl). 2020-2021 arası veri artık yeniden çekilemez. Elindeki eski
parquet o dönem için TEK kaynak — asla üzerine yazma, birleştir.
"""
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pytefas import Crawler, TefasAPIError

logger = logging.getLogger(__name__)

KIND = "YAT"          # yatırım fonları; EMK/BYF ayrı evren, karıştırma
MAX_RETRY = 5         # pytefas rate-limit/geçici hatalarda kendisi bekler

# pytefas kolonları -> bu projenin şeması.
# feature_engineering.py yalnızca date/code/title/price kullanıyor; diğerleri
# ileride net akım metriği için ham veride duruyor (pay sayısı × fiyat farkı).
RENAME = {
    "fund_code": "code",
    "fund_name": "title",
    "shares_outstanding": "number_of_shares",
    "investor_count": "number_of_investors",
    "portfolio_size": "market_cap",
}
DROP = ["exchange_bulletin_price"]   # YAT için hep None


class FetchError(RuntimeError):
    """Veri alınamadı. Boş DataFrame döndürmek yerine bunu fırlatıyoruz ki
    üstteki katman hatayı görsün ve iş kırmızı olsun."""


def recent_window(days: int = 10, today: "date | None" = None) -> tuple[str, str]:
    """Son `days` günlük kayan pencere.

    Neden sadece dünü değil: job bir gün sessizce çökerse ve yalnızca T-1
    çekiyorsan o delik kalıcı olur. Kayan pencere kendi kendini onarır. Ayrıca
    TEFAS bazen geç yayınlıyor ya da sonradan düzeltiyor; düzeltilmiş değer
    pencere içindeyse upsert onu otomatik yakalar.
    """
    end = today or date.today()
    return (end - timedelta(days=days)).isoformat(), end.isoformat()


def _to_project_schema(df: pd.DataFrame) -> pd.DataFrame:
    """pytefas çıktısını projenin kolon adlarına çevirir."""
    out = df.rename(columns=RENAME).drop(columns=DROP, errors="ignore")
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.strip()
    return out


def fetch_range(start: str, end: str, kind: str = KIND) -> pd.DataFrame:
    """[start, end] aralığındaki TÜM fonların günlük verisini çeker.

    Hata durumunda FetchError fırlatır — boş dönmez. Boş dönmek tehlikeli olurdu:
    alttaki katman onu "bugün hiç fon yok" diye yorumlayıp veri setini bozabilir.
    """
    try:
        raw = Crawler(max_retry=MAX_RETRY).fetch(start, end, kind=kind, columns="info")
    except TefasAPIError as exc:
        raise FetchError(f"{start} -> {end} çekilemedi: {exc}") from exc

    if raw is None or raw.empty:
        raise FetchError(f"{start} -> {end} için boş sonuç döndü")

    df = _to_project_schema(raw)
    logger.info("Çekildi: %s -> %s | %d satır, %d fon, %d gün",
                start, end, len(df), df["code"].nunique(), df["date"].nunique())
    return df


# --------------------------------------------------------------------------- #
# __main__ SADECE backfill içindir. Günlük güncelleme -> ingest.py
# Çıktıyı bilerek AYRI bir dosyaya yazıyor; mevcut funds_raw.parquet'e
# dokunmuyor. Birleştirmeyi kapılarıyla birlikte ingest.py --from-file yapacak.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    default_start = (date.today() - timedelta(days=365 * 5 - 5)).isoformat()
    parser = argparse.ArgumentParser(description="TEFAS backfill (tek seferlik)")
    parser.add_argument("--start", default=default_start,
                        help="varsayılan: bugünden 5 yıl öncesi (API duvarı)")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", default="data/raw/funds_backfill.parquet")
    args = parser.parse_args()

    logger.info("Backfill: %s -> %s  (5 yıl ~15 dakika sürebilir)", args.start, args.end)
    df = fetch_range(args.start, args.end)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nYazıldı: {out}")
    print(f"  {len(df):,} satır | {df['code'].nunique()} fon | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  kolonlar: {df.columns.tolist()}")
    print(f"\nŞimdi birleştir:  python src/ingest.py --from-file {out}")
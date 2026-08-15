"""
ingest.py — Günlük veri güncelleme orkestratörü.

Akış:
    çek -> HAM KAPI -> upsert -> atomik yaz -> rf tazelik -> feature_engineering.run()

feature_engineering.py'ye HİÇ dokunulmadı. Oradaki health_checks() zaten
türetilmiş metrikleri denetliyor ve gate düşerse yazmıyor. Buradaki kapı ondan
farklı bir soruyu soruyor: "bugün TEFAS'tan gelen ŞEY makul mü?" Mevcut gate
fiyat serilerine bakıyor, bu gate ise ingest'in kendisine bakıyor.

Çalıştır:
    python src/ingest.py               # son 10 gün
    python src/ingest.py --days 30     # daha geniş pencere
    python src/ingest.py --dry-run     # hiçbir şey yazma, sadece kapıyı test et

Çıkış kodu 0 = başarılı, 1 = kapı düştü / hata. GitHub Actions bunu okur.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from fetch_funds import fetch_range, recent_window, FetchError
from fetch_tcmb_rates import refresh_rates
import feature_engineering as fe

logger = logging.getLogger("ingest")

BASE = Path(__file__).parent.parent / "data"
RAW_FUNDS = BASE / "raw" / "funds_raw.parquet"
RAW_RF = BASE / "raw" / "tcmb_rates.parquet"
FEATURES = BASE / "processed" / "funds_features.parquet"

# --- Kapı eşikleri (hepsi tek yerde, "karar"lar burada) ---
REQUIRED_COLS = ["date", "code", "title", "price"]
MIN_FUNDS_LAST_DAY = 600     # bir iş gününde beklenen en az fon sayısı
MAX_GAP_DAYS = 5             # mevcut verinin sonu ile yeni partinin başı arası azami boşluk
# rf tablosu fon verisinden bu kadar geride kalabilir. 75 seçildi çünkü seri
# AYLIK ve ay başına damgalı; TCMB o ayın değerini ay bittikten sonra yayımlıyor.
# Ay sonlarında doğal gecikme 60 günü aşıyor — daha dar bir eşik her ay boş yere
# alarm çalardı. 75, gerçek donmayı yakalar ama normal yayın takvimini affeder.
MAX_RF_LAG_DAYS = 75


# --------------------------------------------------------------------------- #
# Normalizasyon — tip tutarlılığı yoksa merge sessizce mükerrer üretir
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """date'i datetime'a, code'u temiz string'e çevirir.

    Neden şart: donmuş parquet'te date string, crawler'dan gelen datetime olabilir.
    Tipleri eşitlemezsen drop_duplicates aynı günü iki farklı satır sanar ve
    veri setin sessizce şişer.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.strip()
    return out


# --------------------------------------------------------------------------- #
# HAM KAPI — yazmadan önce
# --------------------------------------------------------------------------- #
def check_batch(new: pd.DataFrame) -> list[str]:
    """Yeni gelen partiye bakar. Boş liste dönerse temiz, doluysa yazma."""
    if new is None or new.empty:
        return ["boş veri geldi"]

    missing = [c for c in REQUIRED_COLS if c not in new.columns]
    if missing:
        return [f"eksik kolon: {missing}"]      # devamını kontrol edemeyiz

    problems = []

    if new["date"].isna().any() or new["code"].isna().any():
        problems.append("date/code kolonunda null var")

    n_dup = int(new.duplicated(subset=["date", "code"]).sum())
    if n_dup:
        problems.append(f"{n_dup} mükerrer (date, code) satırı")

    last = new["date"].max()
    if last > pd.Timestamp.today().normalize():
        problems.append(f"gelecek tarihli veri: {last.date()}")

    n_last = int(new.loc[new["date"] == last, "code"].nunique())
    if n_last < MIN_FUNDS_LAST_DAY:
        problems.append(
            f"{last.date()} için sadece {n_last} fon geldi (eşik {MIN_FUNDS_LAST_DAY})")

    return problems


def check_continuity(existing: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Mevcut verinin bittiği yer ile yeni partinin başladığı yer arasında delik
    var mı?

    Neden ayrı bir kontrol: diğer kapıların hiçbiri bunu göremez. Deliğe rağmen
    satır sayısı artar, son gün dolu görünür, health gate geçer — ve aylık bir
    boşluk veri setine sessizce yerleşir. Sonra o boşluk getirileri, rejim
    metriklerini ve max drawdown'ı bozar, ama hiçbir yerde alarm çalmaz.
    """
    last_existing = existing["date"].max()
    first_new = new["date"].min()
    gap = int((first_new - last_existing).days)
    if gap > MAX_GAP_DAYS:
        return [f"veri deliği: mevcut veri {last_existing.date()} tarihinde bitiyor, "
                f"yeni parti {first_new.date()} tarihinde başlıyor ({gap} gün boşluk). "
                f"Kapatmak için: python src/ingest.py --days {gap + 10}"]
    return []


def check_merge(existing: pd.DataFrame, merged: pd.DataFrame) -> list[str]:
    """Birleştirme SONRASI kontrol. Upsert yalnızca büyütmeli ya da aynı bırakmalı.
    Küçüldüyse bir yerde ezme olmuştur — yazma.

    Bu iki satır, "TEFAS bir sabah boş liste döndü ve 5 yıllık geçmişi sildim"
    senaryosunun tek gerçek panzehiri.
    """
    problems = []
    if len(merged) < len(existing):
        problems.append(f"satır sayısı düştü: {len(existing):,} -> {len(merged):,}")
    if merged["code"].nunique() < existing["code"].nunique():
        problems.append(
            f"fon sayısı düştü: {existing['code'].nunique()} -> {merged['code'].nunique()}")
    return problems


# --------------------------------------------------------------------------- #
# Upsert + atomik yazma
# --------------------------------------------------------------------------- #
def upsert(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """(date, code) anahtarı üzerinden birleştirir; çakışmada YENİ kazanır.

    keep='last' önemli: TEFAS bir günün fiyatını sonradan düzeltirse düzeltilmiş
    değer eskisinin yerine geçer. Kör append yapsaydık iki satır kalırdı.

    İdempotent: aynı günü iki kez çalıştırınca satır sayısı değişmez.
    """
    return (pd.concat([existing, new], ignore_index=True)
              .drop_duplicates(subset=["date", "code"], keep="last")
              .sort_values(["code", "date"])
              .reset_index(drop=True))


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Önce .tmp'ye yaz, sonra yerine kaydır. Yazma ortasında süreç ölürse
    mevcut dosya bozulmadan kalır."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def rf_lag_days(funds: pd.DataFrame, rf_path: Path) -> int:
    """Risksiz faiz tablosu, fon verisinin kaç gün gerisinde kaldı?

    Neden önemli: metrics.risk_free_series() ffill kullanıyor. rf tablosu
    donduğunda hata fırlatmaz, NaN üretmez — son bilinen oranı sonsuza kadar
    taşır. Sharpe ve Sortino sessizce kaymaya başlar. Bu fonksiyon o sessizliği
    bozar.
    """
    rf = pd.read_parquet(rf_path)
    rf["date"] = pd.to_datetime(rf["date"])
    return int((funds["date"].max() - rf["date"].max()).days)


# --------------------------------------------------------------------------- #
# Orkestratör
# --------------------------------------------------------------------------- #
def main(days: int = 10, dry_run: bool = False, skip_features: bool = False,
         from_file: "str | None" = None) -> int:
    if not RAW_FUNDS.exists():
        logger.error("%s yok. Önce backfill: python src/fetch_funds.py", RAW_FUNDS)
        return 1

    if from_file:
        # Backfill dosyasını çekmek yerine diskten oku. Kapıların hepsi aynen
        # çalışır — tek seferlik büyük birleştirmeyi de günlük işle aynı
        # güvenlik ağından geçirmiş oluruz.
        logger.info("Kaynak: %s (çekim yok)", from_file)
        new = normalize(pd.read_parquet(from_file))
    else:
        start, end = recent_window(days)
        logger.info("Pencere: %s -> %s", start, end)
        try:
            new = normalize(fetch_range(start, end))
        except FetchError as exc:
            logger.error("Çekme başarısız: %s", exc)
            return 1

    problems = check_batch(new)
    if problems:
        for p in problems:
            logger.error("HAM KAPI düştü: %s", p)
        return 1
    logger.info("Ham kapı geçti: %d satır, %d fon", len(new), new["code"].nunique())

    existing = normalize(pd.read_parquet(RAW_FUNDS))

    # Kütüphane göçünden sonra şema kayması sessiz bir tehlike: eski dosyada olup
    # yenide olmayan bir kolon, birleşmede o satırlar için NaN olur ve kimse
    # fark etmez. Bloklamıyoruz (yeni kolon eklenmesi meşru), ama görünür kılıyoruz.
    only_old = sorted(set(existing.columns) - set(new.columns))
    only_new = sorted(set(new.columns) - set(existing.columns))
    if only_old:
        logger.warning("Kolon sadece ESKİ veride var: %s", only_old)
    if only_new:
        logger.warning("Kolon sadece YENİ veride var: %s", only_new)

    problems = check_continuity(existing, new)
    if problems:
        for p in problems:
            logger.error("SÜREKLİLİK KAPISI düştü: %s", p)
        return 1

    merged = upsert(existing, new)

    problems = check_merge(existing, merged)
    if problems:
        for p in problems:
            logger.error("BİRLEŞTİRME KAPISI düştü: %s", p)
        return 1

    added = len(merged) - len(existing)
    logger.info("Upsert: %d -> %d satır (+%d yeni)",
                len(existing), len(merged), added)

    if dry_run:
        logger.info("dry-run: hiçbir şey yazılmadı.")
        return 0

    write_atomic(merged, RAW_FUNDS)
    logger.info("Ham veri yazıldı: %s", RAW_FUNDS)

    # --- rf tazeleme: BEST-EFFORT ---
    # Bilerek ölümcül değil. EVDS çökerse fon pipeline'ı durmamalı; seri aylık,
    # bir gün kaçırmak zararsız. Asıl karar bir alttaki gecikme kontrolünde.
    try:
        refresh_rates(RAW_RF)
    except Exception as exc:                    # noqa: BLE001
        logger.warning("rf tazelenemedi (%s) — mevcut tabloyla devam", exc)

    # --- rf tazelik kapısı: ham veri güvende, ama metrikleri bununla üretmeyelim ---
    lag = rf_lag_days(merged, RAW_RF)
    if lag > MAX_RF_LAG_DAYS:
        logger.error(
            "Risksiz faiz tablosu %d gün geride (eşik %d). Ham veri yazıldı ama "
            "metrikler YENİDEN ÜRETİLMEDİ — eski funds_features.parquet servis "
            "edilmeye devam ediyor. fetch_tcmb_rates.py'yi çalıştır.",
            lag, MAX_RF_LAG_DAYS)
        return 1
    logger.info("rf gecikmesi: %d gün (eşik %d) — kabul", lag, MAX_RF_LAG_DAYS)

    if skip_features:
        logger.info("--skip-features: metrik üretimi atlandı.")
        return 0

    # Mevcut Layer-1 pipeline'ı olduğu gibi çağırıyoruz. İçindeki health gate
    # düşerse zaten funds_features.parquet'e yazmaz — eski dosya ayakta kalır.
    features, meta, report = fe.run(RAW_FUNDS, RAW_RF, FEATURES)
    if not report["passed"]:
        logger.error("Health gate düştü (%d imkânsız değer) — metrik dosyası "
                     "güncellenmedi.", len(report["hard_violations"]))
        return 1

    logger.info("Tamam: %d fon, veri son tarihi %s",
                len(features), merged["date"].max().date())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="TEFAS günlük güncelleme")
    parser.add_argument("--days", type=int, default=10, help="kayan pencere (gün)")
    parser.add_argument("--dry-run", action="store_true", help="yazma, sadece kapıyı dene")
    parser.add_argument("--skip-features", action="store_true",
                        help="ham veriyi güncelle, metrikleri yeniden üretme")
    parser.add_argument("--from-file", default=None,
                        help="çekim yerine yerel bir parquet'i birleştir (backfill)")
    args = parser.parse_args()
    sys.exit(main(days=args.days, dry_run=args.dry_run,
                  skip_features=args.skip_features, from_file=args.from_file))
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from tefas import Crawler

# Logging ayarı
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sabitler
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_FILE = DATA_DIR / "funds_raw.parquet"
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RATE_LIMIT_SECONDS = 1.5


def fetch_with_retry(crawler, start, end, retries=3):
    """Rate limiting ve retry ile veri çek."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            data = crawler.fetch(start=start, end=end)
            return data
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    logger.error(f"All {retries} attempts failed for {start} - {end}")
    return None


def fetch_all_funds():
    """Tüm fonları çek ve parquet'e kaydet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Cache kontrolü
    if CACHE_FILE.exists():
        logger.info(f"Cache bulundu: {CACHE_FILE}. Yeniden çekilmiyor.")
        df = pd.read_parquet(CACHE_FILE)
        logger.info(f"Cache'den yüklendi: {df.shape}")
        return df

    logger.info(f"Veri çekiliyor: {START_DATE} - {END_DATE}")
    crawler = Crawler(fund_limit=900)

    df = fetch_with_retry(crawler, start=START_DATE, end=END_DATE)

    if df is None or df.empty:
        logger.error("Veri çekilemedi.")
        return None

    df.to_parquet(CACHE_FILE, index=False)
    logger.info(f"Kaydedildi: {CACHE_FILE} | Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = fetch_all_funds()
    if df is not None:
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"\nSütunlar: {df.columns.tolist()}")
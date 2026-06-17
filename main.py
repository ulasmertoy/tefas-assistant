import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- Veri modeli ---
class Fon(BaseModel):
    code: str
    title: str
    return_1y: float | None = None
    return_2y: float | None = None
    return_4y: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None

# --- Veriyi sunucu acilirken bir kez yükle ---
df = pd.read_parquet("data/processed/funds_features.parquet")
df = df.reset_index()                              # 'code' index'ten sütuna dönsün
df = df.astype(object).where(df.notna(), None)     # bos hücreleri (NaN) -> None
FONLAR = [Fon(**r) for r in df.to_dict(orient="records")]

# --- Endpoint'ler ---
@app.get("/")
def merhaba():
    return {"durum": "calisiyor"}

@app.get("/funds", response_model=list[Fon])
def fonlari_getir(limit: int = 10):
    gecerli = [f for f in FONLAR if f.sharpe is not None]
    sirali = sorted(gecerli, key=lambda f: f.sharpe, reverse=True)
    return sirali[:limit]
# ---- Base: aynı Python sürümü, deploy ile tutarlı ----
    FROM python:3.12-slim

    ENV PYTHONUNBUFFERED=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PIP_NO_CACHE_DIR=1
    
    WORKDIR /app
    
    # ---- Healthcheck için curl ----
    RUN apt-get update \
        && apt-get install -y --no-install-recommends curl \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Önce sadece requirements (layer cache: kod değişince
    #      kütüphaneler yeniden kurulmaz) ----
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    # ---- Sonra kod ----
    COPY . .
    
    # ---- root yerine sınırlı kullanıcı (güvenlik best practice) ----
    RUN useradd --create-home appuser && chown -R appuser:appuser /app
    USER appuser
    
    EXPOSE 8501
    
    # ---- Container ayakta mı diye Streamlit'in health endpoint'i ----
    HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
        CMD curl --fail http://localhost:8501/_stcore/health || exit 1
    
    # ---- 0.0.0.0 kritik: olmadan container dışından erişilemez ----
    CMD ["streamlit", "run", "src/app.py", \
         "--server.port=8501", \
         "--server.address=0.0.0.0", \
         "--server.headless=true"]
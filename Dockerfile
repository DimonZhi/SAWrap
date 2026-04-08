FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAWRAP_SKIP_MISSING_RECALC=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

RUN useradd --create-home --uid 10001 appuser

RUN mkdir -p /app/survival_wrappers
COPY __init__.py /app/survival_wrappers/__init__.py
COPY metrics_sa.py /app/survival_wrappers/metrics_sa.py
COPY UI /app/survival_wrappers/UI

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "survival_wrappers.UI.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

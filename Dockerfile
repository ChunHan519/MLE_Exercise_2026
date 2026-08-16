FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

FROM python:3.10-slim AS runner

WORKDIR /app

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY model/ /model/

COPY src/product_classifier/api /app/src/product_classifier/api
COPY src/product_classifier/models/serve.py /app/src/product_classifier/models/serve.py

USER appuser

EXPOSE 8000

CMD ["uvicorn", "product_classifier.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
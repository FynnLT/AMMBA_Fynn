# Multi-stage build: base -> dependencies -> runtime
# (follows the GSY-DEX Dockerfile pattern)

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM dependencies AS runtime
COPY . .
EXPOSE 8082
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8082"]

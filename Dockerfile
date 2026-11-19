# Training and scoring studio (trainer/app.py). Build: docker compose build

# --- Builder: compiler for packages without a wheel on this platform ----------------
# ecos and scikit-survival have no Linux aarch64 wheels (Apple Silicon hosts), so
# they are built from source here; the compiler stays out of the final image.
FROM python:3.13-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
# Dependencies only, so code changes do not invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Runtime ------------------------------------------------------------------------
FROM python:3.13-slim

# libgomp1: OpenMP runtime needed by XGBoost and scikit-survival
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    RUNS_DIR=/app/runs \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
# Same base image and path as the builder, so the virtualenv works as copied
COPY --from=builder /app/.venv /app/.venv
COPY utils.py ./
COPY dashboard ./dashboard
COPY trainer ./trainer

# Unprivileged user; /app/runs is the volume with trained models and artifacts
RUN useradd --create-home --uid 1000 studio \
    && mkdir -p /app/runs \
    && chown -R studio:studio /app/runs
USER studio
VOLUME /app/runs

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["/app/.venv/bin/streamlit", "run", "trainer/app.py"]

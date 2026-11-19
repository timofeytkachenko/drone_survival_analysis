# One image, two services (see docker-compose.yml):
#   studio  trainer/app.py   training, release and scoring UI (default command)
#   api     trainer/api.py   scoring API for the production model
# Build: docker compose build

# --- Builder: compiler for packages without a wheel on this platform ----------------
# ecos and scikit-survival have no Linux aarch64 wheels (Apple Silicon hosts), so
# they are built from source here; the compiler stays out of the final image.
FROM python:3.13-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# Pinned so that a rebuild resolves uv.lock with the same tool version
RUN pip install --no-cache-dir uv==0.12.17

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

ARG VERSION=0.1.0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="drone-survival-studio" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# The containers run with a read-only root filesystem: everything a library wants
# to cache goes to /tmp (a tmpfs in docker-compose.yml), the data to /app/runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    RUNS_DIR=/app/runs \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=20 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none

WORKDIR /app
# Same base image and path as the builder, so the virtualenv works as copied
COPY --from=builder /app/.venv /app/.venv
COPY utils.py ./
COPY dashboard ./dashboard
COPY trainer ./trainer

# Unprivileged user; /app/runs is the volume with the runs, the production pointer
# and the scoring log
RUN useradd --create-home --uid 1000 studio \
    && mkdir -p /app/runs \
    && chown -R studio:studio /app/runs
USER studio
VOLUME /app/runs

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["/app/.venv/bin/streamlit", "run", "trainer/app.py"]

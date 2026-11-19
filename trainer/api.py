"""REST API: score drones with the production model.

Run from the project root (the Docker image does this)::

    uv run python -m trainer.api

Endpoints
---------
``GET /healthz``
    Liveness, no authentication. Reports whether a production model is set.
``GET /v1/model``
    The production version: run, kind, training data, release gate, test metrics.
``POST /v1/score``
    Forecast a batch. Body: ``text/csv`` (the file the studio accepts) or
    ``application/json`` (``{"drones": [{...}, ...]}`` or a bare list of rows).

Every ``/v1`` request needs the key from ``API_KEY`` in the ``X-API-Key`` header.
The model is looked up on every request, so a promotion or rollback made in the
studio or the CLI takes effect without a restart.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import time
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from dashboard import common
from trainer import registry, scoring
from trainer import settings as config

# Not __name__: the module runs as __main__ under ``python -m trainer.api``
logger = logging.getLogger("trainer.api")

_API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Key from the API_KEY variable"
)
_SCORE_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "text/csv": {"schema": {"type": "string"}},
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["drones"],
                    "properties": {
                        "drones": {"type": "array", "items": {"type": "object"}}
                    },
                }
            },
        },
    }
}


def _clean(value: object) -> object:
    """Replace NaN and infinity with ``None``: they are not valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


async def _read_body(request: Request, limit: int) -> bytes:
    """Request body, refusing more than ``limit`` bytes without reading the rest.

    Raises
    ------
    HTTPException
        413 if the declared or the actual size exceeds ``limit``.
    """
    too_large = HTTPException(
        status.HTTP_413_CONTENT_TOO_LARGE, f"Body larger than {limit} bytes"
    )
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def _parse(body: bytes, content_type: str, lang: str) -> pd.DataFrame:
    """Rows of a scoring request.

    Raises
    ------
    HTTPException
        415 for an unsupported content type, 422 for an unreadable body.
    """
    media = content_type.split(";")[0].strip().lower()
    if media == "text/csv":
        try:
            return common.read_csv_bytes(body, lang)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)
            ) from exc
    if media == "application/json":
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid JSON: {exc}"
            ) from exc
        rows = payload.get("drones") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                'Expected {"drones": [{...}, ...]} or a list of objects',
            )
        return pd.DataFrame.from_records(rows)
    raise HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "Content-Type must be text/csv or application/json",
    )


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    settings : trainer.settings.Settings, optional
        Configuration; read from the environment by default.

    Returns
    -------
    FastAPI
        The application.

    Raises
    ------
    trainer.settings.SettingsError
        If the configuration is invalid or ``API_KEY`` is unset while
        authentication is required.
    """
    settings = config.load() if settings is None else settings
    api_key = config.require(settings, "api_key")
    if api_key is None:
        logger.warning("REQUIRE_AUTH=false and no API_KEY: the API is open")

    app = FastAPI(
        title="Drone survival scoring",
        version="1",
        docs_url="/docs" if settings.api_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.api_docs else None,
    )

    def authenticate(key: str | None = Depends(_API_KEY_HEADER)) -> None:
        if api_key is None:
            return
        if key is None or not hmac.compare_digest(key.encode(), api_key.encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key"
            )

    def production() -> scoring.Bundle:
        try:
            return scoring.production_bundle()
        except scoring.NoProductionModelError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        # Path only: no query string, headers or body, so no data or key is logged.
        # The container health probe calls /healthz every 30 s: keep it out of INFO.
        logger.log(
            logging.DEBUG if request.url.path == "/healthz" else logging.INFO,
            "%s %s %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(1000 * (time.perf_counter() - started), 1),
            },
        )
        return response

    @app.get("/healthz")
    def healthz() -> dict:
        try:
            ready = registry.current() is not None
        except registry.RegistryError:
            ready = False
        return {"status": "ok", "production_model": ready}

    @app.get("/v1/model", dependencies=[Depends(authenticate)])
    def model() -> JSONResponse:
        bundle = production()
        gate = bundle.data.gate["kinds"][bundle.kind]
        return JSONResponse(
            _clean(
                {
                    **bundle.describe(),
                    "promoted": registry.current(),
                    "gate": gate,
                    "features": bundle.settings.features,
                }
            )
        )

    @app.post(
        "/v1/score", dependencies=[Depends(authenticate)], openapi_extra=_SCORE_BODY
    )
    async def score(
        request: Request, lang: Literal["en", "ru"] = Query("en")
    ) -> JSONResponse:
        body = await _read_body(request, settings.max_upload_bytes)
        frame = _parse(body, request.headers.get("content-type", ""), lang)
        if len(frame) > settings.max_score_rows:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"{len(frame)} rows; at most {settings.max_score_rows} per request",
            )
        bundle = production()
        try:
            result = await run_in_threadpool(scoring.score, frame, bundle, "api", lang)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)
            ) from exc
        return JSONResponse(
            _clean(
                {
                    "model": {
                        "run_id": bundle.run_id,
                        "kind": bundle.kind,
                        "data_sha256": bundle.data.card["data"]["sha256"],
                    },
                    "n": len(result.scored),
                    "warnings": result.warnings,
                    "drift": result.drift,
                    "results": json.loads(result.scored.to_json(orient="records")),
                }
            )
        )

    return app


def main() -> None:
    """Serve the API with uvicorn on ``API_HOST``:``API_PORT`` (127.0.0.1:8000)."""
    import uvicorn

    settings = config.load()
    config.configure_logging(settings)
    uvicorn.run(
        create_app(settings),
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
        # The root logger is already configured; uvicorn must not replace it
        log_config=None,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()

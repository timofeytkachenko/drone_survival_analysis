"""Configuration from environment variables and logging set-up.

Read once at start-up by the studio, the CLI and the API; a bad value stops the
process with a message naming the variable. Secrets are never logged.

Variables
---------
``RUNS_DIR``
    Directory of runs, the production pointer and the scoring log (read in
    ``trainer.runs``).
``STUDIO_PASSWORD``
    Password of the studio, at least ``MIN_PASSWORD_LENGTH`` characters.
``API_KEY``
    Key the API expects in ``X-API-Key``, at least ``MIN_API_KEY_LENGTH`` characters.
``REQUIRE_AUTH``
    ``true`` (default): the studio and the API refuse to work without their
    secret. ``false`` is for local development only.
``MAX_UPLOAD_MB``
    Largest accepted file or request body, megabytes (default 20).
``MAX_TRAIN_ROWS``
    Largest accepted training table (default 200000 rows).
``MAX_SCORE_ROWS``
    Largest accepted scoring batch (default and upper limit 10000 rows).
``LOG_LEVEL``, ``LOG_FORMAT``
    ``INFO`` and ``json`` by default; ``LOG_FORMAT=text`` for a terminal.
``API_DOCS``
    ``true`` exposes ``/docs`` and ``/openapi.json`` (default ``false``).
``GATE_*``
    Release-gate thresholds, see ``trainer.release.GateConfig``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dashboard.common import MAX_UPLOAD_ROWS
from trainer.release import GateConfig

MIN_PASSWORD_LENGTH = 12
MIN_API_KEY_LENGTH = 32
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# Attributes of every LogRecord; anything else was passed through ``extra``
_RECORD_FIELDS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class SettingsError(ValueError):
    """Raised when an environment variable has an unusable value."""


@dataclass(frozen=True)
class Settings:
    """Validated configuration; the secrets are excluded from ``repr``."""

    studio_password: str | None = field(repr=False)
    api_key: str | None = field(repr=False)
    require_auth: bool
    max_upload_mb: int
    max_train_rows: int
    max_score_rows: int
    log_level: str
    log_format: str
    api_docs: bool
    gate: GateConfig

    @property
    def max_upload_bytes(self) -> int:
        """``max_upload_mb`` in bytes."""
        return self.max_upload_mb * 1024 * 1024


def _flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise SettingsError(f"{name} must be true or false")


def _positive_int(
    environ: Mapping[str, str], name: str, default: int, upper: int | None = None
) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a whole number") from exc
    if value < 1 or (upper is not None and value > upper):
        limit = "" if upper is None else f" and at most {upper}"
        raise SettingsError(f"{name} must be at least 1{limit}")
    return value


def _secret(environ: Mapping[str, str], name: str, min_length: int) -> str | None:
    """A secret, or ``None`` if unset. The value never appears in an error."""
    value = environ.get(name, "")
    if not value:
        return None
    if len(value) < min_length:
        raise SettingsError(f"{name} must be at least {min_length} characters long")
    return value


def load(environ: Mapping[str, str] | None = None) -> Settings:
    """Read and validate the configuration.

    Parameters
    ----------
    environ : Mapping[str, str], optional
        Source of the variables; ``os.environ`` by default.

    Returns
    -------
    Settings
        Validated configuration.

    Raises
    ------
    SettingsError
        If a variable has an unusable value. A missing secret is not an error
        here: each service checks the one it needs (``require``).
    """
    environ = os.environ if environ is None else environ
    log_level = environ.get("LOG_LEVEL", "").strip().upper() or "INFO"
    if log_level not in _LOG_LEVELS:
        raise SettingsError(f"LOG_LEVEL must be one of {', '.join(_LOG_LEVELS)}")
    log_format = environ.get("LOG_FORMAT", "").strip().lower() or "json"
    if log_format not in ("json", "text"):
        raise SettingsError("LOG_FORMAT must be json or text")
    try:
        gate = GateConfig.from_env(environ)
    except ValueError as exc:
        raise SettingsError(str(exc)) from exc
    return Settings(
        studio_password=_secret(environ, "STUDIO_PASSWORD", MIN_PASSWORD_LENGTH),
        api_key=_secret(environ, "API_KEY", MIN_API_KEY_LENGTH),
        require_auth=_flag(environ, "REQUIRE_AUTH", True),
        max_upload_mb=_positive_int(environ, "MAX_UPLOAD_MB", 20),
        max_train_rows=_positive_int(environ, "MAX_TRAIN_ROWS", 200_000),
        max_score_rows=_positive_int(
            environ, "MAX_SCORE_ROWS", MAX_UPLOAD_ROWS, upper=MAX_UPLOAD_ROWS
        ),
        log_level=log_level,
        log_format=log_format,
        api_docs=_flag(environ, "API_DOCS", False),
        gate=gate,
    )


def require(settings: Settings, name: str) -> str | None:
    """The secret a service needs, enforcing ``REQUIRE_AUTH``.

    Parameters
    ----------
    settings : Settings
        Loaded configuration.
    name : str
        ``"studio_password"`` or ``"api_key"``.

    Returns
    -------
    str | None
        The secret; ``None`` only when ``REQUIRE_AUTH=false`` and it is unset,
        which means the service runs without authentication.

    Raises
    ------
    SettingsError
        If the secret is unset while authentication is required.
    """
    value = getattr(settings, name)
    if value is None and settings.require_auth:
        raise SettingsError(
            f"{name.upper()} is not set. Set it, or set REQUIRE_AUTH=false "
            "for local development without authentication"
        )
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per log line: time, level, logger, message and extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {k: v for k, v in vars(record).items() if k not in _RECORD_FIELDS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings) -> None:
    """Send the root logger to stderr in the configured format, replacing handlers."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    # Optuna reports every trial at INFO; the stage log already shows progress
    logging.getLogger("optuna").setLevel(logging.WARNING)

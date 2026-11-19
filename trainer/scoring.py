"""Single scoring entry point for the studio, the CLI and the API.

All three call ``score`` with a ``Bundle``; it validates and forecasts the batch
with ``dashboard.common.score_upload``, compares the batch with the training
profile and appends one line of aggregates to the scoring log.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from dashboard import common
from dashboard.data import DashboardData, load
from dashboard.labels import PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_MEDIUM
from trainer import monitoring, registry, runs

logger = logging.getLogger(__name__)

# XGBoost prediction is not documented as safe for concurrent calls on one booster
_PREDICT_LOCK = threading.Lock()


class NoProductionModelError(RuntimeError):
    """Raised when no version has been promoted to production."""


@dataclass(frozen=True)
class Bundle:
    """A model version ready to score.

    Attributes
    ----------
    run_id : str
        Run the models come from.
    kind : str
        Model kind that makes the forecasts.
    production : bool
        Whether this is the version the production pointer names.
    data : DashboardData
        Artifacts and models of the run, forecasting with ``kind``.
    settings : common.Settings
        Horizon, window and priority thresholds of the run.
    """

    run_id: str
    kind: str
    production: bool
    data: DashboardData
    settings: common.Settings

    def describe(self) -> dict:
        """What scored a batch: version, kind, training data, gate, test metrics."""
        card = self.data.card
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "production": self.production,
            "created": card["created"],
            "data_sha256": card["data"]["sha256"],
            "n_fit": card["data"]["n_fit"],
            "rounds": card["models"][self.kind]["rounds"],
            "gate_passed": card["gate"]["kinds"][self.kind]["passed"],
            "test_metrics": card["metrics"]["production"][self.kind]["test"],
            "rul_horizon_h": self.settings.horizon,
            "window_h": self.settings.window,
            "versions": card["versions"],
        }


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of ``score``.

    Attributes
    ----------
    scored : pd.DataFrame
        Output of ``common.score_upload``, riskiest first.
    warnings : list[str]
        Data-quality issues that did not prevent scoring.
    drift : dict
        ``monitoring.batch_report`` of the batch.
    """

    scored: pd.DataFrame
    warnings: list[str]
    drift: dict


@lru_cache(maxsize=4)
def _load(run_id: str, kind: str) -> tuple[DashboardData, common.Settings]:
    base = runs.run_path(run_id)
    data = dataclasses.replace(
        load(base / "artifacts", base / "models"), forecast_kind=kind
    )
    return data, common.Settings.from_data(data)


def load_bundle(run_id: str, kind: str | None = None) -> Bundle:
    """Load a finished run for scoring.

    Parameters
    ----------
    run_id : str
        Run to load.
    kind : str, optional
        Model kind; by default the run's forecasting kind.

    Returns
    -------
    Bundle
        The run with ``production`` set if it is what the pointer names.

    Raises
    ------
    FileNotFoundError, ValueError
        If the run's files are missing or of an older format.
    KeyError
        If the run has no model of ``kind``.
    """
    card = runs.read_card(run_id)
    if card is None:
        raise ValueError(f"Run {run_id} has no production models (evaluation only)")
    kind = kind or card["forecast_name"]
    if kind not in card["gate"]["kinds"]:
        raise KeyError(kind)
    data, settings = _load(run_id, kind)
    try:
        entry = registry.current()
    except registry.RegistryError:
        entry = None
    return Bundle(
        run_id=run_id,
        kind=kind,
        production=entry is not None
        and (entry["run_id"], entry["kind"]) == (run_id, kind),
        data=data,
        settings=settings,
    )


def production_bundle() -> Bundle:
    """The version the production pointer names, re-read on every call.

    The models are cached per version, so a call costs one small file read; after
    a promotion or rollback in another process the next call loads the new one.

    Raises
    ------
    NoProductionModelError
        If nothing has been promoted, or the promoted version is no longer
        loadable or did not pass its release gate (a hand-edited pointer).
    """
    try:
        entry = registry.current()
    except registry.RegistryError as exc:
        raise NoProductionModelError(str(exc)) from exc
    if entry is None:
        raise NoProductionModelError("No model has been promoted to production")
    try:
        registry.check_promotable(entry["run_id"], entry["kind"])
        return load_bundle(entry["run_id"], entry["kind"])
    except (registry.RegistryError, FileNotFoundError, ValueError, KeyError) as exc:
        raise NoProductionModelError(
            f"The production version {entry.get('run_id')} cannot be used: {exc}"
        ) from exc


def score(
    raw: pd.DataFrame,
    bundle: Bundle,
    source: str,
    lang: str = "en",
    log: bool = True,
) -> ScoreResult:
    """Validate and forecast a batch of drones, then log its aggregates.

    Parameters
    ----------
    raw : pd.DataFrame
        One row per drone (see ``common.score_upload``).
    bundle : Bundle
        Model version to score with.
    source : str
        Caller recorded in the log: ``"studio"``, ``"cli"`` or ``"api"``.
    lang : str
        Language of the error and warning texts.
    log : bool
        Append the batch to the scoring log; ``False`` for a drift-only check.

    Returns
    -------
    ScoreResult
        Forecasts, warnings and the drift report.

    Raises
    ------
    ValueError
        If the batch fails validation (nothing is logged).
    """
    with _PREDICT_LOCK:
        scored, warnings = common.score_upload(raw, bundle.data, bundle.settings, lang)
    features = bundle.settings.features
    drift = monitoring.batch_report(scored, bundle.data.card["profile"])
    counts = scored["priority"].value_counts()
    record = {
        "source": source,
        "run_id": bundle.run_id,
        "kind": bundle.kind,
        "production": bundle.production,
        "input_hash": monitoring.input_hash(scored, features),
        "high": int(counts.get(PRIORITY_HIGH, 0)),
        "medium": int(counts.get(PRIORITY_MEDIUM, 0)),
        "low": int(counts.get(PRIORITY_LOW, 0)),
        "mean_p_fail": round(float(scored["p_fail_window"].mean()), 4),
        "out_of_range_share": round(float(scored["out_of_range"].mean()), 4),
        "n_warnings": len(warnings),
        "n_unknown_categories": sum(drift["unknown_categories"].values()),
        **drift,
    }
    if log:
        try:
            monitoring.log_batch(runs.RUNS_DIR, record)
        except OSError:
            # The forecast is still valid; a full or read-only volume must not hide it
            logger.exception("Could not write the scoring log")
    logger.info(
        "Scored %d drones (source=%s, run=%s, kind=%s, production=%s, max_psi=%.3f)",
        len(scored),
        source,
        bundle.run_id,
        bundle.kind,
        bundle.production,
        drift["max_psi"],
    )
    return ScoreResult(scored=scored, warnings=warnings, drift=drift)

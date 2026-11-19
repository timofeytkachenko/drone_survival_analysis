"""Input-data monitoring: training profile, drift of scored batches, scoring log.

The profile of the rows a production model was trained on is stored in its model
card. Every scored batch is compared with it (population stability index per
feature, unknown categories, values outside the training range) and one line of
aggregates is appended to a JSONL log. The log holds no drone rows or identifiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trainer.fslock import locked

logger = logging.getLogger(__name__)

LOG_NAME = "scoring_log.jsonl"
N_BINS = 10
# Shares below this are floored so an empty bin does not make the index infinite
PSI_FLOOR = 1e-4
# Conventional reading of the index: below 0.1 stable, above 0.25 a clear shift
PSI_MODERATE, PSI_SIGNIFICANT = 0.1, 0.25
# With fewer rows the bin shares are too noisy for the index to mean anything
MIN_PSI_ROWS = 100
_MISSING = "(missing)"
_OTHER = "(unknown)"


def _level(value: object) -> str:
    if pd.isna(value):
        return _MISSING
    # A column with gaps is read as float: 3.0 is the level "3"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _levels(values: pd.Series) -> pd.Series:
    """Categorical values as text, missing ones as ``_MISSING``."""
    return values.astype("object").map(_level)


def training_profile(
    frame: pd.DataFrame, features: list[str], cat_features: list[str]
) -> dict:
    """Distribution of the model features in the training rows.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows the production model was fitted on.
    features, cat_features : list[str]
        Model features and the categorical ones among them.

    Returns
    -------
    dict
        ``n`` (rows), ``numeric`` (feature -> inner decile ``edges`` and the
        ``shares`` of the bins they form, the last share being missing values)
        and ``categorical`` (feature -> level ``shares``). JSON-serialisable.
    """
    numeric, categorical = {}, {}
    for feature in features:
        if feature in cat_features:
            shares = _levels(frame[feature]).value_counts(normalize=True)
            categorical[feature] = {
                "shares": {str(k): float(v) for k, v in shares.items()}
            }
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        edges = np.unique(
            np.quantile(values.dropna(), np.linspace(0, 1, N_BINS + 1)[1:-1])
        )
        numeric[feature] = {
            "edges": [float(e) for e in edges],
            "shares": [float(s) for s in _numeric_shares(values, edges)],
        }
    return {"n": len(frame), "numeric": numeric, "categorical": categorical}


def _numeric_shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Shares of ``values`` in the bins cut by ``edges``; missing values come last."""
    present = values.dropna().to_numpy(dtype=float)
    counts = np.bincount(
        np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1
    )
    return np.append(counts, values.isna().sum()) / max(len(values), 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index ``sum((a - e) * ln(a / e))`` over matching bins."""
    e = np.clip(np.asarray(expected, dtype=float), PSI_FLOOR, None)
    a = np.clip(np.asarray(actual, dtype=float), PSI_FLOOR, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_level(value: float | None) -> str:
    """``stable``, ``moderate`` or ``significant`` by the conventional cut-offs."""
    if value is None:
        return "n/a"
    if value >= PSI_SIGNIFICANT:
        return "significant"
    return "moderate" if value >= PSI_MODERATE else "stable"


def batch_report(frame: pd.DataFrame, profile: dict) -> dict:
    """Compare a batch of drones with the training profile.

    Parameters
    ----------
    frame : pd.DataFrame
        Batch with the model features already parsed (numbers as numbers).
    profile : dict
        Result of ``training_profile``.

    Returns
    -------
    dict
        ``n_rows``; ``psi`` per feature with ``max_psi``, ``max_psi_feature``
        and ``level``; ``psi_reliable`` (false below ``MIN_PSI_ROWS`` rows: the
        index is still reported but ``level`` is ``"n/a"``); ``unknown_categories``
        (feature -> rows with a level absent from training) and ``missing``
        (feature -> share of empty values, features without any are omitted).
    """
    scores, unknown, missing = {}, {}, {}
    for feature, spec in profile["numeric"].items():
        values = pd.to_numeric(frame[feature], errors="coerce")
        actual = _numeric_shares(values, np.asarray(spec["edges"]))
        scores[feature] = psi(spec["shares"], actual)
        if values.isna().any():
            missing[feature] = float(values.isna().mean())
    for feature, spec in profile["categorical"].items():
        levels = _levels(frame[feature])
        known = list(spec["shares"])
        is_unknown = ~levels.isin(known) & (levels != _MISSING)
        if is_unknown.any():
            unknown[feature] = int(is_unknown.sum())
        if (levels == _MISSING).any():
            missing[feature] = float((levels == _MISSING).mean())
        actual = levels.where(~is_unknown, _OTHER).value_counts(normalize=True)
        keys = sorted({*known, *actual.index})
        scores[feature] = psi(
            [spec["shares"].get(k, 0.0) for k in keys],
            [float(actual.get(k, 0.0)) for k in keys],
        )
    worst = max(scores, key=scores.get)
    reliable = len(frame) >= MIN_PSI_ROWS
    return {
        "n_rows": len(frame),
        "psi": {k: round(v, 4) for k, v in scores.items()},
        "max_psi": round(scores[worst], 4),
        "max_psi_feature": worst,
        "level": drift_level(scores[worst] if reliable else None),
        "psi_reliable": reliable,
        "unknown_categories": unknown,
        "missing": missing,
    }


def input_hash(frame: pd.DataFrame, features: list[str]) -> str:
    """Short SHA-256 of the feature values of a batch (order-sensitive)."""
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame[features], index=False).to_numpy().tobytes()
    )
    return digest.hexdigest()[:16]


def log_batch(directory: Path, record: dict) -> None:
    """Append one JSON line to the scoring log in ``directory``.

    Parameters
    ----------
    directory : Path
        Directory of the log (the runs directory shared by all services).
    record : dict
        JSON-serialisable aggregates of one scored batch; a UTC ``ts`` is added.
    """
    line = json.dumps(
        {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record},
        ensure_ascii=False,
    )
    path = directory / LOG_NAME
    with locked(directory / ".scoring_log.lock"), path.open("a") as handle:
        handle.write(line + "\n")


def read_log(directory: Path, last: int = 200) -> list[dict]:
    """The newest ``last`` records of the scoring log, oldest first.

    Lines that are not valid JSON (a write cut short by a crash) are skipped.
    """
    path = directory / LOG_NAME
    if not path.exists():
        return []
    with path.open() as handle:
        lines = deque(handle, maxlen=last)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping an unreadable line of %s", path.name)
    return records


def log_frame(records: list[dict]) -> pd.DataFrame:
    """Scoring log as a table, one row per batch (without the per-feature index)."""
    columns = [
        "ts",
        "source",
        "run_id",
        "kind",
        "production",
        "n_rows",
        "high",
        "medium",
        "low",
        "mean_p_fail",
        "out_of_range_share",
        "max_psi",
        "max_psi_feature",
        "level",
        "psi_reliable",
        "n_unknown_categories",
        "input_hash",
    ]
    return pd.DataFrame(
        [{c: r.get(c) for c in columns} for r in records], columns=columns
    )

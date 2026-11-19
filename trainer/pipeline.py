"""Training pipeline of ``drone_survival.ipynb`` as a function.

``train`` runs the same steps with the same parameters as the notebook and writes
the same files (``artifacts/*.parquet``, ``artifacts/meta.json``,
``artifacts/model_card.json``, ``models/best_{cox,aft}.json``,
``models/production_{cox,aft}.json``), so ``dashboard.data.load`` and the existing
dashboards can read the result. The notebook stays the reference implementation.

Two stages: the evaluation models are tuned on train, early-stopped on validation
and reported on test; the production models are refitted on train + validation
and must pass the release gate on test (``trainer.release``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trainer.artifacts import write_artifacts
from trainer.release import (
    MIN_FRAILTY_VARIANCE,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    RUL_WINDOW,
    GateConfig,
    backtest_groups,
    build_production,
    model_card,
)
from utils import (
    SCALAR_METRICS,
    Model,
    bootstrap_compare,
    calibration_curve,
    evaluate,
    evaluation_times,
    tune,
)

__all__ = [
    "CAT_FEATURES",
    "FEATURES",
    "PRIORITY_HIGH",
    "PRIORITY_MEDIUM",
    "REQUIRED_COLUMNS",
    "RUL_WINDOW",
    "Cancelled",
    "TrainConfig",
    "prepare_frame",
    "split_frame",
    "train",
]

logger = logging.getLogger(__name__)

FEATURES = [
    "model",
    "environment",
    "is_refurbished",
    "initial_battery_cycles",
    "firmware_major",
    "maintenance_interval_h",
    "operator_experience_yrs",
    "daily_utilization_h",
    "avg_flight_min",
    "avg_payload_kg",
    "payload_ratio",
    "mean_temp_c",
    "avg_humidity_pct",
    "avg_wind_ms",
    "hard_landings_per_100h",
    "mean_vibration_g",
]
CAT_FEATURES = ["model", "environment", "is_refurbished", "firmware_major"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]
REQUIRED_COLUMNS = ["drone_id", "commission_date", "duration_h", "event", *FEATURES]
OPTIONAL_COLUMNS = ["failure_mode"]

MIN_ROWS = 500
MIN_EVENTS = {"train": 30, "val": 10, "test": 10}

_BOOL_VALUES = {
    "true": True,
    "1": True,
    "yes": True,
    "false": False,
    "0": False,
    "no": False,
}

Progress = Callable[[str, float], None]


class Cancelled(Exception):
    """Raised through ``progress`` to stop a training run between stages."""


@dataclass(frozen=True)
class TrainConfig:
    """Training settings; defaults match the notebook."""

    n_trials: int = 60
    n_boot: int = 1000
    seed: int = 42
    gate: GateConfig = field(default_factory=GateConfig)


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate a training table and return a cleaned copy.

    Parameters
    ----------
    raw : pd.DataFrame
        One row per drone with ``REQUIRED_COLUMNS`` (``failure_mode`` is optional).

    Returns
    -------
    pd.DataFrame
        Copy with parsed dates, boolean ``is_refurbished``, numeric columns and an
        ``in_service`` flag, sorted by nothing in particular.

    Raises
    ------
    ValueError
        If a column is missing, a value cannot be parsed, or there are too few
        drones or failures to split and train.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if len(raw) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} drones are required, got {len(raw)}")
    df = raw.copy()

    df["commission_date"] = pd.to_datetime(df["commission_date"], errors="coerce")
    if df["commission_date"].isna().any():
        raise ValueError("`commission_date` has empty or unreadable dates")

    for column in ["duration_h", "event", "daily_utilization_h"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"`{column}` has empty or non-numeric values")
    if (df["duration_h"] <= 0).any():
        raise ValueError("`duration_h` must be positive")
    if (df["daily_utilization_h"] <= 0).any():
        raise ValueError("`daily_utilization_h` must be positive")
    if not df["event"].isin([0, 1]).all():
        raise ValueError("`event` must be 0 (censored) or 1 (failure)")
    df["event"] = df["event"].astype(int)

    for column in NUMERIC_FEATURES:
        if column == "daily_utilization_h":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().all():
            raise ValueError(f"`{column}` has no numeric values")

    flags = df["is_refurbished"].astype(str).str.strip().str.lower().map(_BOOL_VALUES)
    if flags.isna().any():
        raise ValueError("`is_refurbished` must be True/False, 1/0 or yes/no")
    df["is_refurbished"] = flags.astype(bool)

    if df["drone_id"].duplicated().any():
        raise ValueError("`drone_id` must be unique")
    df["drone_id"] = df["drone_id"].astype(str)
    if "failure_mode" not in df.columns:
        df["failure_mode"] = np.where(df["event"] == 1, "unknown", "censored")

    # Last day each drone was observed: for a drone still flying it is the data
    # cut-off; a censored drone last seen earlier has left the fleet.
    last_seen = df["commission_date"] + pd.to_timedelta(
        df["duration_h"] / df["daily_utilization_h"], unit="D"
    )
    cutoff = last_seen.max()
    df["in_service"] = (df["event"] == 0) & (last_seen > cutoff - pd.Timedelta(days=1))

    splits = split_frame(df)[:3]
    for name, part in zip(("train", "val", "test"), splits):
        if int(part["event"].sum()) < MIN_EVENTS[name]:
            raise ValueError(
                f"The {name} split has {int(part['event'].sum())} failures, "
                f"at least {MIN_EVENTS[name]} are required"
            )
    return df


def split_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Split by commissioning date: oldest 70% train, next 10% validation, newest 20% test."""
    t_val, t_test = df["commission_date"].quantile([0.70, 0.80])
    train = df[df["commission_date"] < t_val]
    val = df[(df["commission_date"] >= t_val) & (df["commission_date"] < t_test)]
    test = df[df["commission_date"] >= t_test]
    return train, val, test, t_val, t_test


def train(
    raw: pd.DataFrame,
    config: TrainConfig,
    out_dir: Path,
    progress: Progress | None = None,
) -> dict:
    """Train Cox and AFT, evaluate them, build the production models and write everything.

    Parameters
    ----------
    raw : pd.DataFrame
        Training table (see ``prepare_frame``).
    config : TrainConfig
        Number of Optuna trials and bootstrap replicates, release-gate thresholds.
    out_dir : Path
        Directory that receives ``models/`` and ``artifacts/``; created if missing.
    progress : callable, optional
        ``progress(stage, fraction)`` called between stages. It may raise
        ``Cancelled`` to stop the run.

    Returns
    -------
    dict
        The contents of ``meta.json``.

    Raises
    ------
    ValueError
        If the table fails validation or leaves no remaining-life horizon.
    Cancelled
        If ``progress`` raised it.
    """

    def step(stage: str, fraction: float) -> None:
        logger.info("%s (%.0f%%)", stage, 100 * fraction)
        if progress is not None:
            progress(stage, fraction)

    models_dir, artifacts_dir = out_dir / "models", out_dir / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    step("Validating data", 0.0)
    df = prepare_frame(raw)
    train_df, val, test, t_val, t_test = split_frame(df)

    step("Tuning Cox", 0.02)
    best_cox = tune("cox", train_df, val, config.n_trials, FEATURES, CAT_FEATURES)
    step("Tuning AFT", 0.25)
    best_aft = tune("aft", train_df, val, config.n_trials, FEATURES, CAT_FEATURES)

    step("Recalibrating for unobserved heterogeneity", 0.50)
    models = {"cox": best_cox, "aft": best_aft}
    for model in models.values():
        model.calibrate(train_df, val)
        # A variance of 0 means no heterogeneity was found: the model stays as trained
        if model.calibration["frailty_variance"] < MIN_FRAILTY_VARIANCE:
            model.calibration = None

    # Save and reload both models: checks the round trip, as the notebook does
    for name, model in models.items():
        model.save(models_dir / f"best_{name}.json")
    models = {name: Model.load(models_dir / f"best_{name}.json") for name in models}

    step("Evaluating", 0.62)
    splits = {"val": val, "test": test}
    eval_times = evaluation_times(val, test)
    results = {
        (name, split): evaluate(model, data, data, eval_times)
        for name, model in models.items()
        for split, data in splits.items()
    }
    metrics = pd.DataFrame(
        {key: {m: r[m] for m in SCALAR_METRICS} for key, r in results.items()}
    ).T.rename_axis(["model", "split"])
    best_name = str(metrics.xs("val", level="split")["ibs"].idxmin())
    best_model = models[best_name]

    horizons = np.quantile(test["duration_h"], [0.25, 0.5, 0.75])
    calibration = pd.concat(
        calibration_curve(model, test, h).assign(model=name, horizon=h)
        for name, model in models.items()
        for h in horizons
    )

    step("Bootstrap comparison", 0.66)
    boot_summary, boot_samples = bootstrap_compare(
        models, test, test, eval_times, n_boot=config.n_boot, seed=config.seed
    )

    step("SHAP", 0.78)
    shap_values = {name: model.explain(test) for name, model in models.items()}

    step("Backtest", 0.84)
    grid = np.linspace(1, np.quantile(train_df["duration_h"], 0.95), 200)
    landmark = float(np.quantile(test["duration_h"], 0.25))
    residual = test.loc[test["duration_h"] > landmark, "duration_h"] - landmark
    backtest_u = np.linspace(0, residual.quantile(0.9), 200)

    # "raw": the best model without the recalibration, to show what it changes
    backtest_models = dict(models)
    if best_model.calibration is not None:
        raw_model = Model.load(models_dir / f"best_{best_name}.json")
        raw_model.calibration = None
        backtest_models["raw"] = raw_model
    evaluation_backtest = pd.concat(
        backtest_groups(model, test, landmark, RUL_WINDOW).assign(source=name)
        for name, model in backtest_models.items()
    )[["source", "quintile", "n", "predicted", "observed", "ci_low", "ci_high"]]

    step("Production models (train + validation)", 0.88)
    train_val = pd.concat([train_df, val])
    production = build_production(
        models,
        best_name,
        train_val,
        test,
        eval_times=eval_times,
        horizons=horizons,
        landmark=landmark,
        evaluation_metrics=metrics,
        evaluation_calibration=calibration,
        models_dir=models_dir,
        config=config.gate,
        window=RUL_WINDOW,
    )
    split = {
        "t_val": str(t_val.date()),
        "t_test": str(t_test.date()),
        "n_train": len(train_df),
        "n_val": len(val),
        "n_test": len(test),
    }
    card = model_card(
        frame=df,
        split=split,
        models=models,
        best_name=best_name,
        evaluation_metrics=metrics,
        production=production,
        train_val=train_val,
        settings={
            "n_trials": config.n_trials,
            "n_boot": config.n_boot,
            "seed": config.seed,
        },
    )

    step("Writing artifacts", 0.96)
    meta = write_artifacts(
        artifacts_dir,
        test=test,
        models=models,
        best_name=best_name,
        metrics=metrics,
        results=results,
        eval_times=eval_times,
        horizons=horizons,
        calibration=calibration,
        boot_summary=boot_summary,
        boot_samples=boot_samples,
        shap_values=shap_values,
        grid=grid,
        landmark=landmark,
        backtest_u=backtest_u,
        backtest_groups=evaluation_backtest,
        production=production,
        card=card,
        split=split,
        n_trials=config.n_trials,
    )
    step("Done", 1.0)
    return meta

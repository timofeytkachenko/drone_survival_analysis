"""Training pipeline of ``drone_survival.ipynb`` as a function.

``train`` runs the same steps with the same parameters as the notebook and writes
the same files (``artifacts/*.parquet``, ``artifacts/meta.json``,
``models/best_{cox,aft}.json``), so ``dashboard.data.load`` and the existing
dashboards can read the result. The notebook stays the reference implementation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator

from utils import (
    SCALAR_METRICS,
    Model,
    bootstrap_compare,
    calibration_curve,
    evaluate,
    evaluation_times,
    fit_categories,
    remaining_life,
    tune,
)

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

RUL_WINDOW = 100  # look-ahead window of the failure probability, flight hours
PRIORITY_HIGH, PRIORITY_MEDIUM = 0.5, 0.2
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


def _save_table(table: pd.DataFrame, directory: Path, name: str) -> None:
    table.to_parquet(directory / f"{name}.parquet", index=False)


def train(
    raw: pd.DataFrame,
    config: TrainConfig,
    out_dir: Path,
    progress: Progress | None = None,
) -> dict:
    """Train Cox and AFT, evaluate them and write models and artifacts.

    Parameters
    ----------
    raw : pd.DataFrame
        Training table (see ``prepare_frame``).
    config : TrainConfig
        Number of Optuna trials and bootstrap replicates.
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
        If the table fails validation.
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
    cats = fit_categories(train_df, CAT_FEATURES)

    step("Tuning Cox", 0.02)
    best_cox = tune("cox", train_df, val, config.n_trials, FEATURES, CAT_FEATURES)
    step("Tuning AFT", 0.25)
    best_aft = tune("aft", train_df, val, config.n_trials, FEATURES, CAT_FEATURES)

    step("Recalibrating for unobserved heterogeneity", 0.50)
    models = {"cox": best_cox, "aft": best_aft}
    for model in models.values():
        model.calibrate(train_df, val)
        # A variance of 0 means no heterogeneity was found: the model stays as trained
        if model.calibration["frailty_variance"] < 1e-8:
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

    step("Bootstrap comparison", 0.68)
    boot_summary, boot_samples = bootstrap_compare(
        models, test, test, eval_times, n_boot=config.n_boot, seed=config.seed
    )

    step("SHAP", 0.80)
    shap_values = {name: model.explain(test) for name, model in models.items()}
    shap_mean = pd.DataFrame(
        {name: np.abs(e.values).mean(axis=0) for name, e in shap_values.items()},
        index=FEATURES,
    )

    step("Remaining life and backtest", 0.88)
    grid = np.linspace(1, np.quantile(train_df["duration_h"], 0.95), 200)
    curves = {name: model.survival(test, grid) for name, model in models.items()}
    km_test = kaplan_meier_estimator(
        test["event"].astype(bool), test["duration_h"], conf_type="log-log"
    )

    landmark = np.quantile(test["duration_h"], 0.25)
    at_risk = test[test["duration_h"] > landmark]
    residual = at_risk["duration_h"] - landmark
    at_risk_event = at_risk["event"].astype(bool)
    at_risk_age = np.full(len(at_risk), landmark)
    km_backtest = kaplan_meier_estimator(at_risk_event, residual, conf_type="log-log")
    u = np.linspace(0, residual.quantile(0.9), 200)

    # "raw": the best model without the recalibration, to show what it changes
    backtest_models = dict(models)
    if best_model.calibration is not None:
        raw_model = Model.load(models_dir / f"best_{best_name}.json")
        raw_model.calibration = None
        backtest_models["raw"] = raw_model
    group_rows = []
    for name, model in backtest_models.items():
        p_fail = (
            1
            - model.conditional_survival(at_risk, at_risk_age, np.array([RUL_WINDOW]))[
                :, 0
            ]
        )
        quintile = pd.qcut(p_fail, 5, labels=False)
        for q in np.unique(quintile):
            in_group = quintile == q
            group_t, group_s, group_ci = kaplan_meier_estimator(
                at_risk_event[in_group], residual[in_group], conf_type="log-log"
            )
            at_window = np.searchsorted(group_t, RUL_WINDOW, side="right") - 1
            group_rows.append(
                {
                    "source": name,
                    "quintile": q + 1,
                    "n": int(in_group.sum()),
                    "predicted": p_fail[in_group].mean(),
                    "observed": 1 - group_s[at_window],
                    "ci_low": 1 - group_ci[1][at_window],
                    "ci_high": 1 - group_ci[0][at_window],
                }
            )
    backtest_groups = pd.DataFrame(group_rows)

    # Drones in service at the data cut-off; current age = duration_h. Keep age +
    # horizon inside the Cox baseline support (flat after the last training event).
    active = test[test["in_service"]]
    last_train_event = train_df.loc[train_df["event"] == 1, "duration_h"].max()
    horizon = (
        float(np.floor(last_train_event - active["duration_h"].max()))
        if len(active)
        else 0.0
    )
    if len(active) and horizon <= 0:
        raise ValueError(
            "The oldest drone in service has flown longer than the last training "
            "failure, so no remaining-life horizon exists"
        )
    rul = (
        remaining_life(best_model, active, horizon=horizon, window=RUL_WINDOW)
        if len(active)
        else pd.DataFrame(
            columns=["drone_id", "rul_q10", "rul_median", "rmrl", "p_fail_window"]
        )
    )

    step("Writing artifacts", 0.95)
    _save_table(metrics.reset_index(), artifacts_dir, "metrics")
    _save_table(
        pd.concat(
            pd.DataFrame(
                {
                    "model": name,
                    "t": eval_times,
                    "brier": results[(name, "test")]["brier"],
                    "auc": results[(name, "test")]["auc"],
                }
            )
            for name in models
        ),
        artifacts_dir,
        "brier_auc",
    )
    _save_table(calibration, artifacts_dir, "calibration")
    _save_table(boot_summary, artifacts_dir, "bootstrap_summary")
    _save_table(
        pd.concat(
            pd.DataFrame(
                {
                    "iteration": boot_samples.index,
                    "model": model_name,
                    "metric": metric,
                    "value": boot_samples[(model_name, metric)].to_numpy(),
                }
            )
            for model_name, metric in boot_samples.columns
        ),
        artifacts_dir,
        "bootstrap_samples",
    )

    n_features = len(FEATURES)
    _save_table(
        pd.concat(
            pd.DataFrame(
                {
                    "drone_id": np.repeat(test["drone_id"].to_numpy(), n_features),
                    "model": name,
                    "feature": np.tile(e.feature_names, len(test)),
                    "shap": e.values.ravel(),
                    "value": e.data.ravel(),
                    "display": test[e.feature_names].astype(str).to_numpy().ravel(),
                }
            )
            for name, e in shap_values.items()
        ),
        artifacts_dir,
        "shap_values",
    )
    importance = shap_mean.reset_index(names="feature").melt(
        id_vars="feature", var_name="model", value_name="mean_abs_shap"
    )
    importance["share"] = importance["mean_abs_shap"] / importance.groupby("model")[
        "mean_abs_shap"
    ].transform("sum")
    _save_table(importance, artifacts_dir, "shap_importance")

    # risk_group: terciles of the best model's risk over all test drones; priority:
    # maintenance urgency of drones in service (failed or departed drones have none)
    predictions = test.copy()
    for name, model in models.items():
        predictions[f"risk_{name}"] = model.risk(test)
    predictions["risk_group"] = pd.qcut(
        predictions[f"risk_{best_name}"], 3, labels=["Low", "Medium", "High"]
    ).astype(str)
    predictions = predictions.join(rul.drop(columns=["drone_id"]))
    predictions["priority"] = np.select(
        [
            predictions["p_fail_window"] >= PRIORITY_HIGH,
            predictions["p_fail_window"] >= PRIORITY_MEDIUM,
            predictions["p_fail_window"].notna(),
            predictions["event"] == 1,
        ],
        ["High", "Medium", "Low", "None (failed)"],
        default="None (left the fleet)",
    )
    _save_table(predictions, artifacts_dir, "predictions")

    _save_table(
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "source": "km",
                        "t": km_test[0],
                        "s": km_test[1],
                        "ci_low": km_test[2][0],
                        "ci_high": km_test[2][1],
                    }
                ),
                *(
                    pd.DataFrame({"source": name, "t": grid, "s": s.mean(axis=0)})
                    for name, s in curves.items()
                ),
            ]
        ),
        artifacts_dir,
        "survival_mean",
    )
    _save_table(
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "source": "km",
                        "u": km_backtest[0],
                        "s": km_backtest[1],
                        "ci_low": km_backtest[2][0],
                        "ci_high": km_backtest[2][1],
                    }
                ),
                *(
                    pd.DataFrame(
                        {
                            "source": name,
                            "u": u,
                            "s": model.conditional_survival(
                                at_risk, at_risk_age, u
                            ).mean(axis=0),
                        }
                    )
                    for name, model in models.items()
                ),
            ]
        ),
        artifacts_dir,
        "backtest",
    )
    _save_table(backtest_groups, artifacts_dir, "backtest_groups")

    meta = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "best_name": best_name,
        "models": {name: f"best_{name}.json" for name in models},
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "categories": cats,
        "numeric_ranges": {
            f: [float(train_df[f].min()), float(train_df[f].max())]
            for f in NUMERIC_FEATURES
            if f != "is_refurbished"
        },
        "split": {"t_val": str(t_val.date()), "t_test": str(t_test.date())},
        "n_trials": config.n_trials,
        "n_boot": len(boot_samples),
        "eval_times": {name: eval_times.tolist() for name in splits},
        "calibration_horizons": [float(h) for h in horizons],
        "rul_horizon": horizon,
        "rul_window": RUL_WINDOW,
        "landmark": float(landmark),
        "priority_thresholds": {"high": PRIORITY_HIGH, "medium": PRIORITY_MEDIUM},
        "shap_base": {name: float(e.base_values[0]) for name, e in shap_values.items()},
    }
    (artifacts_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    step("Done", 1.0)
    return meta

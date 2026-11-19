"""Write the files the dashboards, the studio and the scoring services read.

``write_artifacts`` is the single writer of ``artifacts/``: both
``drone_survival.ipynb`` and ``trainer.pipeline.train`` call it, so the two cannot
drift apart. The model files are written earlier (``Model.save`` for the
evaluation models, ``trainer.release.build_production`` for the production ones).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sksurv.nonparametric import kaplan_meier_estimator

from trainer.release import (
    FORMAT_VERSION,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    Production,
)
from utils import Model


def _save(table: pd.DataFrame, directory: Path, name: str) -> None:
    table.to_parquet(directory / f"{name}.parquet", index=False)


def _shap_long(test: pd.DataFrame, values: dict[str, shap.Explanation]) -> pd.DataFrame:
    """SHAP in long format: one row per (drone, model, feature)."""
    return pd.concat(
        pd.DataFrame(
            {
                "drone_id": np.repeat(
                    test["drone_id"].to_numpy(), len(e.feature_names)
                ),
                "model": name,
                "feature": np.tile(e.feature_names, len(test)),
                "shap": e.values.ravel(),
                "value": e.data.ravel(),
                "display": test[e.feature_names].astype(str).to_numpy().ravel(),
            }
        )
        for name, e in values.items()
    )


def write_artifacts(
    artifacts_dir: Path,
    *,
    test: pd.DataFrame,
    models: dict[str, Model],
    best_name: str,
    metrics: pd.DataFrame,
    results: dict[tuple[str, str], dict],
    eval_times: np.ndarray,
    horizons: Sequence[float],
    calibration: pd.DataFrame,
    boot_summary: pd.DataFrame,
    boot_samples: pd.DataFrame,
    shap_values: dict[str, shap.Explanation],
    grid: np.ndarray,
    landmark: float,
    backtest_u: np.ndarray,
    backtest_groups: pd.DataFrame,
    production: Production,
    card: dict,
    split: dict,
    n_trials: int,
    priority_high: float = PRIORITY_HIGH,
    priority_medium: float = PRIORITY_MEDIUM,
) -> dict:
    """Write the parquet tables, ``meta.json`` and ``model_card.json``.

    Quality tables (metrics, calibration, bootstrap, SHAP importance, backtest)
    describe the evaluation models. Everything that is a forecast (remaining
    life, priority, risk group, ``shap_production``) comes from the production
    models, for which the test drones are out of sample; the kind that makes
    them is ``production.forecast_name``.

    Parameters
    ----------
    artifacts_dir : Path
        Output directory; created if missing.
    test : pd.DataFrame
        Test rows with ``drone_id`` and ``in_service``.
    models : dict[str, Model]
        Evaluation models keyed by kind, saved as ``best_<kind>.json``.
    best_name : str
        Kind selected on validation.
    metrics : pd.DataFrame
        Scalar metrics indexed by ``(model, split)``.
    results : dict[tuple[str, str], dict]
        ``utils.evaluate`` output per ``(kind, split)``; the test ``brier`` and
        ``auc`` curves are stored.
    eval_times : numpy.ndarray
        Evaluation grid of ``results``.
    horizons : Sequence[float]
        Calibration horizons in flight hours.
    calibration : pd.DataFrame
        Decile calibration of the evaluation models (``model``, ``horizon``).
    boot_summary, boot_samples : pd.DataFrame
        Output of ``utils.bootstrap_compare``.
    shap_values : dict[str, shap.Explanation]
        TreeSHAP of the evaluation models on ``test``.
    grid : numpy.ndarray
        Time grid of the mean survival curves.
    landmark : float
        Landmark of the backtest in flight hours.
    backtest_u : numpy.ndarray
        Grid of additional flight hours of the backtest curves.
    backtest_groups : pd.DataFrame
        Backtest by risk quintile of the evaluation models (``source`` column).
    production : Production
        Result of ``trainer.release.build_production``.
    card : dict
        Result of ``trainer.release.model_card``.
    split : dict
        Split dates and sizes (``t_val``, ``t_test``, ``n_train``, ``n_val``,
        ``n_test``).
    n_trials : int
        Optuna trials per model.
    priority_high, priority_medium : float
        Failure-probability thresholds of the maintenance priority.

    Returns
    -------
    dict
        The contents of ``meta.json``.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    features = models[best_name].features
    cat_features = models[best_name].cat_features

    _save(metrics.reset_index(), artifacts_dir, "metrics")
    _save(
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
    _save(calibration, artifacts_dir, "calibration")
    _save(boot_summary, artifacts_dir, "bootstrap_summary")
    _save(
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

    _save(_shap_long(test, shap_values), artifacts_dir, "shap_values")
    _save(_shap_long(test, production.shap), artifacts_dir, "shap_production")
    importance = (
        pd.DataFrame(
            {name: np.abs(e.values).mean(axis=0) for name, e in shap_values.items()},
            index=features,
        )
        .reset_index(names="feature")
        .melt(id_vars="feature", var_name="model", value_name="mean_abs_shap")
    )
    importance["share"] = importance["mean_abs_shap"] / importance.groupby("model")[
        "mean_abs_shap"
    ].transform("sum")
    _save(importance, artifacts_dir, "shap_importance")

    # risk_<kind>: evaluation models; forecast_risk_<kind>: production models.
    # risk_group: terciles of the production risk over all test drones; priority:
    # maintenance urgency of drones in service (failed or departed drones have none).
    predictions = test.copy()
    for name, model in models.items():
        predictions[f"risk_{name}"] = model.risk(test)
    for name, model in production.models.items():
        predictions[f"forecast_risk_{name}"] = model.risk(test)
    predictions["risk_group"] = pd.qcut(
        predictions[f"forecast_risk_{production.forecast_name}"],
        3,
        labels=["Low", "Medium", "High"],
    ).astype(str)
    predictions = predictions.join(production.rul.drop(columns=["drone_id"]))
    predictions["priority"] = np.select(
        [
            predictions["p_fail_window"] >= priority_high,
            predictions["p_fail_window"] >= priority_medium,
            predictions["p_fail_window"].notna(),
            predictions["event"] == 1,
        ],
        ["High", "Medium", "Low", "None (failed)"],
        default="None (left the fleet)",
    )
    _save(predictions, artifacts_dir, "predictions")

    km_test = kaplan_meier_estimator(
        test["event"].astype(bool), test["duration_h"], conf_type="log-log"
    )
    _save(
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
                    pd.DataFrame(
                        {
                            "source": name,
                            "t": grid,
                            "s": model.survival(test, grid).mean(axis=0),
                        }
                    )
                    for name, model in models.items()
                ),
            ]
        ),
        artifacts_dir,
        "survival_mean",
    )

    at_risk = test[test["duration_h"] > landmark]
    at_risk_age = np.full(len(at_risk), landmark)
    km_backtest = kaplan_meier_estimator(
        at_risk["event"].astype(bool),
        at_risk["duration_h"] - landmark,
        conf_type="log-log",
    )
    _save(
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
                            "u": backtest_u,
                            "s": model.conditional_survival(
                                at_risk, at_risk_age, backtest_u
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
    _save(backtest_groups, artifacts_dir, "backtest_groups")

    meta = {
        "format": FORMAT_VERSION,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "best_name": best_name,
        # Kind that makes the forecasts: best_name unless it failed the release gate
        "forecast_name": production.forecast_name,
        "models": {name: f"best_{name}.json" for name in models},
        "production_models": {
            name: f"production_{name}.json" for name in production.models
        },
        "features": features,
        "cat_features": cat_features,
        # Vocabulary, ranges and horizon of the production models (train + validation)
        "categories": production.categories,
        "numeric_ranges": production.numeric_ranges,
        "rul_horizon": production.horizon,
        "rul_window": production.window,
        "split": split,
        "n_trials": n_trials,
        "n_boot": len(boot_samples),
        "eval_times": {name: eval_times.tolist() for name in ("val", "test")},
        "calibration_horizons": [float(h) for h in horizons],
        "landmark": float(landmark),
        "priority_thresholds": {"high": priority_high, "medium": priority_medium},
        "shap_base": {name: float(e.base_values[0]) for name, e in shap_values.items()},
        "shap_base_production": {
            name: float(e.base_values[0]) for name, e in production.shap.items()
        },
    }
    (artifacts_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    (artifacts_dir / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2)
    )
    return meta

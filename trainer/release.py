"""Production stage: refit on train + validation, release gate and model card.

Evaluation models are tuned on train, early-stopped on validation and reported on
test. The production models reuse their parameters and number of boosting rounds
but are fitted on train + validation, so they have seen every drone except the
test split. ``release_gate`` compares them with the evaluation models on test; a
kind that fails any check cannot be promoted to production.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sksurv.nonparametric import kaplan_meier_estimator

from trainer.monitoring import training_profile
from utils import (
    SCALAR_METRICS,
    Model,
    calibration_curve,
    evaluate,
    fit_categories,
    remaining_life,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2  # layout of meta.json / model_card.json written by this package
RUL_WINDOW = 100  # look-ahead window of the failure probability, flight hours
PRIORITY_HIGH, PRIORITY_MEDIUM = 0.5, 0.2
# A fitted frailty variance below this means no heterogeneity was found
MIN_FRAILTY_VARIANCE = 1e-8
# Survival probabilities must survive Model.save / Model.load to this precision
ROUNDTRIP_TOLERANCE = 1e-6
N_BACKTEST_GROUPS = 5
_LIBRARIES = (
    "xgboost",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "scikit-survival",
    "shap",
    "optuna",
)


@dataclass(frozen=True)
class GateConfig:
    """Thresholds of the release gate; every check is made on the test split.

    Attributes
    ----------
    c_index_tolerance : float
        The production Uno C-index may be at most this much below the
        evaluation model's.
    ibs_tolerance : float
        The production integrated Brier score may be at most this much above
        the evaluation model's.
    max_calibration_error : float
        Upper limit of the mean absolute gap between predicted and observed
        survival over prediction deciles, averaged over the three calibration
        horizons.
    min_backtest_groups : int
        Minimum number of the five risk quintiles of the landmark backtest whose
        predicted failure probability lies inside the 95% interval of the
        observed one.
    """

    c_index_tolerance: float = 0.01
    ibs_tolerance: float = 0.005
    max_calibration_error: float = 0.05
    min_backtest_groups: int = 4

    def __post_init__(self) -> None:
        for name in ("c_index_tolerance", "ibs_tolerance", "max_calibration_error"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.min_backtest_groups <= N_BACKTEST_GROUPS:
            raise ValueError(
                f"min_backtest_groups must be between 0 and {N_BACKTEST_GROUPS}"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GateConfig:
        """Defaults overridden by ``GATE_<FIELD>`` variables, e.g. ``GATE_IBS_TOLERANCE``.

        Raises
        ------
        ValueError
            If a variable is not a number or is out of range.
        """
        environ = os.environ if environ is None else environ
        values = {}
        for field in fields(cls):
            raw = environ.get(f"GATE_{field.name.upper()}")
            if raw is None or raw == "":
                continue
            cast = int if field.name == "min_backtest_groups" else float
            try:
                values[field.name] = cast(raw)
            except ValueError as exc:
                raise ValueError(
                    f"GATE_{field.name.upper()} must be a number, got {raw!r}"
                ) from exc
        return cls(**values)


@dataclass(frozen=True)
class Production:
    """Production models with everything derived from them.

    Attributes
    ----------
    models : dict[str, Model]
        Models fitted on train + validation, reloaded from ``models_dir``.
    metrics : pd.DataFrame
        ``SCALAR_METRICS`` on test, indexed by model kind.
    calibration : pd.DataFrame
        Decile calibration on test per kind and horizon.
    backtest_groups : pd.DataFrame
        Landmark backtest by risk quintile per kind (``source`` column).
    shap : dict[str, shap.Explanation]
        TreeSHAP values on test per kind.
    horizon : float
        Remaining-life horizon in flight hours.
    forecast_name : str
        Kind that forecasts the fleet: the gate's ``release_kind``, or the kind
        selected on validation when no kind passed.
    rul : pd.DataFrame
        ``remaining_life`` of the test drones in service by ``forecast_name``.
    gate : dict
        Result of ``release_gate``.
    categories : dict
        Category vocabulary of train + validation.
    numeric_ranges : dict
        ``feature -> [min, max]`` over train + validation.
    rounds : dict[str, int]
        Boosting rounds per kind.
    n_train : int
        Rows the production models were fitted on.
    window : float
        Look-ahead window of the failure probability in flight hours.
    """

    models: dict[str, Model]
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    backtest_groups: pd.DataFrame
    shap: dict[str, shap.Explanation]
    horizon: float
    forecast_name: str
    rul: pd.DataFrame
    gate: dict
    categories: dict
    numeric_ranges: dict
    rounds: dict[str, int]
    n_train: int
    window: float


def frame_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of the values of a table (row order matters, the index does not)."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    ).hexdigest()


def library_versions() -> dict[str, str]:
    """Python and modelling library versions for the model card."""
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in _LIBRARIES},
    }


def refit_production(
    models: dict[str, Model], train_val: pd.DataFrame
) -> tuple[dict[str, Model], dict[str, str]]:
    """Refit the evaluation models on train + validation and recalibrate them.

    Every model keeps its tuned parameters and the number of rounds early
    stopping selected. The recalibration is refitted on out-of-fold predictions
    for ``train_val`` and kept only if its variance is above 0, the rule the
    evaluation models follow.

    Parameters
    ----------
    models : dict[str, Model]
        Fitted evaluation models keyed by kind.
    train_val : pd.DataFrame
        Train and validation rows together.

    Returns
    -------
    production : dict[str, Model]
        Refitted models keyed by kind.
    errors : dict[str, str]
        Kinds whose recalibration did not converge, with the message; such a
        model is returned without the recalibration and fails the release gate.
    """
    production, errors = {}, {}
    for name, model in models.items():
        refitted = model.refit(train_val)
        try:
            refitted.calibrate(train_val)
        except RuntimeError as exc:
            logger.warning("Recalibration of the production %s failed: %s", name, exc)
            errors[name] = str(exc)
            refitted.calibration = None
        else:
            if refitted.calibration["frailty_variance"] < MIN_FRAILTY_VARIANCE:
                refitted.calibration = None
        production[name] = refitted
    return production, errors


def fleet_horizon(train: pd.DataFrame, active: pd.DataFrame) -> float:
    """Remaining-life horizon in flight hours.

    Keeps age + horizon inside the Cox baseline support, which is flat after the
    last failure among the rows the model was fitted on. Not positive when the
    oldest drone in service has flown longer than that.
    """
    last_event = train.loc[train["event"] == 1, "duration_h"].max()
    oldest = active["duration_h"].max() if len(active) else 0.0
    return float(np.floor(last_event - oldest))


def backtest_groups(
    model: Model, test: pd.DataFrame, landmark: float, window: float = RUL_WINDOW
) -> pd.DataFrame:
    """Predicted vs observed failure within ``window`` hours by risk quintile.

    Uses the test drones still flying at ``landmark`` flight hours; the observed
    share is one minus Kaplan-Meier of their residual life at ``window``.

    Parameters
    ----------
    model : Model
        Fitted model.
    test : pd.DataFrame
        Rows with ``duration_h``, ``event`` and the model features.
    landmark : float
        Age in flight hours from which the forecast is made.
    window : float
        Look-ahead window in flight hours.

    Returns
    -------
    pd.DataFrame
        One row per quintile: ``quintile`` (1 = lowest risk), ``n``,
        ``predicted``, ``observed`` and its 95% interval ``ci_low``, ``ci_high``.
    """
    at_risk = test[test["duration_h"] > landmark]
    residual = at_risk["duration_h"] - landmark
    event = at_risk["event"].astype(bool)
    ages = np.full(len(at_risk), landmark)
    p_fail = 1 - model.conditional_survival(at_risk, ages, np.array([window]))[:, 0]
    quintile = pd.qcut(p_fail, N_BACKTEST_GROUPS, labels=False)
    rows = []
    for q in np.unique(quintile):
        in_group = quintile == q
        group_t, group_s, group_ci = kaplan_meier_estimator(
            event[in_group], residual[in_group], conf_type="log-log"
        )
        at_window = np.searchsorted(group_t, window, side="right") - 1
        rows.append(
            {
                "quintile": q + 1,
                "n": int(in_group.sum()),
                "predicted": p_fail[in_group].mean(),
                "observed": 1 - group_s[at_window],
                "ci_low": 1 - group_ci[1][at_window],
                "ci_high": 1 - group_ci[0][at_window],
            }
        )
    return pd.DataFrame(rows)


def calibration_error(calibration: pd.DataFrame) -> float:
    """Mean absolute gap between predicted and observed survival.

    Weighted by bin size within a horizon, then averaged over the horizons.
    """
    gap = (calibration["predicted"] - calibration["observed"]).abs() * calibration["n"]
    per_horizon = (
        gap.groupby(calibration["horizon"]).sum()
        / calibration.groupby("horizon")["n"].sum()
    )
    return float(per_horizon.mean())


def _check(
    name: str,
    value: float,
    threshold: float,
    passed: bool,
    reference: float | None = None,
) -> dict:
    return {
        "name": name,
        "value": float(value),
        "reference": None if reference is None else float(reference),
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def release_gate(
    *,
    best_name: str,
    order: Sequence[str],
    evaluation_metrics: pd.DataFrame,
    production_metrics: pd.DataFrame,
    evaluation_calibration: pd.DataFrame,
    calibration: pd.DataFrame,
    backtest: pd.DataFrame,
    roundtrip_error: Mapping[str, float],
    finite: Mapping[str, bool],
    recalibration_errors: Mapping[str, str],
    config: GateConfig,
) -> dict:
    """Decide per model kind whether the production model may be released.

    Checks, all on the test split:

    ``c_index_ipcw``
        Uno C-index not below the evaluation model's minus the tolerance.
    ``ibs``
        Integrated Brier score not above the evaluation model's plus the tolerance.
    ``calibration_error``
        ``calibration_error`` of the decile calibration not above the limit (the
        evaluation model's error is recorded as ``reference`` but not compared).
    ``backtest_groups``
        Enough risk quintiles with the prediction inside the observed 95% interval
        (a quintile without an interval does not count).
    ``roundtrip``
        Predictions unchanged by saving and loading the model.
    ``finite``
        No NaN or infinite predictions.
    ``recalibration``
        The recalibration converged.

    Parameters
    ----------
    best_name : str
        Kind selected on validation.
    order : Sequence[str]
        Kinds from best to worst validation score; the first one that passes
        becomes ``release_kind``.
    evaluation_metrics : pd.DataFrame
        Test metrics of the evaluation models, indexed by kind.
    production_metrics : pd.DataFrame
        Test metrics of the production models, indexed by kind.
    evaluation_calibration : pd.DataFrame
        Decile calibration of the evaluation models (``model``, ``horizon``).
    calibration : pd.DataFrame
        Decile calibration of the production models (``model``, ``horizon``).
    backtest : pd.DataFrame
        ``backtest_groups`` of the production models (``source`` column).
    roundtrip_error : Mapping[str, float]
        Largest change of a survival probability after save and load, per kind.
    finite : Mapping[str, bool]
        Whether every prediction of the kind is finite.
    recalibration_errors : Mapping[str, str]
        Kinds whose recalibration did not converge.
    config : GateConfig
        Thresholds.

    Returns
    -------
    dict
        ``release_kind`` (``None`` if no kind passed), ``passed`` (a kind can be
        released), ``best_name``, ``config`` and ``kinds``: kind -> ``passed``
        and the list of ``checks`` (``name``, ``value``, ``reference``,
        ``threshold``, ``passed``). JSON-serialisable.
    """
    kinds = {}
    for kind in production_metrics.index:
        reference, value = evaluation_metrics.loc[kind], production_metrics.loc[kind]
        c_min = reference["c_index_ipcw"] - config.c_index_tolerance
        ibs_max = reference["ibs"] + config.ibs_tolerance
        error = calibration_error(calibration[calibration["model"] == kind])
        groups = backtest[backtest["source"] == kind]
        # A comparison with a NaN interval bound is False: such a group is not counted
        inside = int(
            (
                (groups["predicted"] >= groups["ci_low"])
                & (groups["predicted"] <= groups["ci_high"])
            ).sum()
        )
        checks = [
            _check(
                "c_index_ipcw",
                value["c_index_ipcw"],
                c_min,
                value["c_index_ipcw"] >= c_min,
                reference["c_index_ipcw"],
            ),
            _check(
                "ibs", value["ibs"], ibs_max, value["ibs"] <= ibs_max, reference["ibs"]
            ),
            _check(
                "calibration_error",
                error,
                config.max_calibration_error,
                error <= config.max_calibration_error,
                calibration_error(
                    evaluation_calibration[evaluation_calibration["model"] == kind]
                ),
            ),
            _check(
                "backtest_groups",
                inside,
                config.min_backtest_groups,
                inside >= config.min_backtest_groups,
            ),
            _check(
                "roundtrip",
                roundtrip_error[kind],
                ROUNDTRIP_TOLERANCE,
                roundtrip_error[kind] <= ROUNDTRIP_TOLERANCE,
            ),
            _check("finite", float(finite[kind]), 1.0, finite[kind]),
            _check(
                "recalibration",
                float(kind not in recalibration_errors),
                1.0,
                kind not in recalibration_errors,
            ),
        ]
        kinds[kind] = {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
        }
    release_kind = next((kind for kind in order if kinds[kind]["passed"]), None)
    return {
        "passed": release_kind is not None,
        "release_kind": release_kind,
        "best_name": best_name,
        "config": asdict(config),
        "kinds": kinds,
    }


def build_production(
    models: dict[str, Model],
    best_name: str,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    eval_times: np.ndarray,
    horizons: Sequence[float],
    landmark: float,
    evaluation_metrics: pd.DataFrame,
    evaluation_calibration: pd.DataFrame,
    models_dir: Path,
    config: GateConfig | None = None,
    window: float = RUL_WINDOW,
) -> Production:
    """Fit, save, check and apply the production models.

    Parameters
    ----------
    models : dict[str, Model]
        Fitted evaluation models keyed by kind.
    best_name : str
        Kind selected on validation. It forecasts the drones in service unless it
        fails the gate and the other kind passes.
    train_val : pd.DataFrame
        Train and validation rows together.
    test : pd.DataFrame
        Test rows with an ``in_service`` column. Never used for fitting.
    eval_times : numpy.ndarray
        Evaluation grid of the test metrics (as for the evaluation models).
    horizons : Sequence[float]
        Calibration horizons in flight hours.
    landmark : float
        Landmark of the backtest in flight hours.
    evaluation_metrics : pd.DataFrame
        Metrics of the evaluation models indexed by ``(model, split)``.
    evaluation_calibration : pd.DataFrame
        Decile calibration of the evaluation models on test at ``horizons``
        (``model``, ``horizon`` columns); recorded in the gate for reference.
    models_dir : Path
        Receives ``production_<kind>.json``.
    config : GateConfig, optional
        Gate thresholds; ``GateConfig()`` by default.
    window : float
        Look-ahead window of the failure probability in flight hours.

    Returns
    -------
    Production
        Reloaded models, their test diagnostics, the fleet forecast and the gate
        result.

    Raises
    ------
    ValueError
        If the remaining-life horizon is shorter than ``window``: the oldest
        drone in service has flown almost as long as the last failure the models
        were fitted on, so no forecast can be made.
    """
    config = GateConfig() if config is None else config
    features, cat_features = models[best_name].features, models[best_name].cat_features
    fitted, recalibration_errors = refit_production(models, train_val)

    # Save and reload: the files are what gets released, so they are what is checked
    models_dir.mkdir(parents=True, exist_ok=True)
    production, roundtrip_error = {}, {}
    for name, model in fitted.items():
        path = models_dir / f"production_{name}.json"
        model.save(path)
        production[name] = Model.load(path)
        roundtrip_error[name] = float(
            np.abs(
                model.survival(test, eval_times)
                - production[name].survival(test, eval_times)
            ).max()
        )

    results = {
        name: evaluate(model, test, test, eval_times)
        for name, model in production.items()
    }
    metrics = pd.DataFrame(
        {name: {m: r[m] for m in SCALAR_METRICS} for name, r in results.items()}
    ).T.rename_axis("model")
    calibration = pd.concat(
        calibration_curve(model, test, h).assign(model=name, horizon=h)
        for name, model in production.items()
        for h in horizons
    )
    backtest = pd.concat(
        backtest_groups(model, test, landmark, window).assign(source=name)
        for name, model in production.items()
    )

    active = test[test["in_service"]]
    horizon = fleet_horizon(train_val, active)
    if horizon < window:
        raise ValueError(
            f"The remaining-life horizon is {horizon:.0f} h, shorter than the "
            f"{window:.0f} h window: the oldest drone in service has flown almost "
            "as long as the last failure in the training data"
        )
    # Every kind forecasts the fleet so that its forecast is checked before release
    forecasts = {
        name: (
            remaining_life(model, active, horizon, window)
            if len(active)
            else pd.DataFrame(
                columns=[
                    "drone_id",
                    "age_h",
                    "rul_q10",
                    "rul_median",
                    "rmrl",
                    "p_fail_window",
                ],
                dtype=float,
            )
        )
        for name, model in production.items()
    }
    finite = {
        name: bool(
            np.isfinite(model.survival(test, eval_times)).all()
            and np.isfinite(model.risk(test)).all()
            and np.isfinite(forecasts[name]["p_fail_window"]).all()
        )
        for name, model in production.items()
    }
    validation_ibs = evaluation_metrics.xs("val", level="split")["ibs"]
    gate = release_gate(
        best_name=best_name,
        order=list(validation_ibs.sort_values().index),
        evaluation_metrics=evaluation_metrics.xs("test", level="split"),
        production_metrics=metrics,
        evaluation_calibration=evaluation_calibration,
        calibration=calibration,
        backtest=backtest,
        roundtrip_error=roundtrip_error,
        finite=finite,
        recalibration_errors=recalibration_errors,
        config=config,
    )
    forecast_name = gate["release_kind"] or best_name
    numeric = [f for f in features if f not in cat_features]
    return Production(
        models=production,
        metrics=metrics,
        calibration=calibration,
        backtest_groups=backtest,
        shap={name: model.explain(test) for name, model in production.items()},
        horizon=horizon,
        forecast_name=forecast_name,
        rul=forecasts[forecast_name],
        gate=gate,
        categories=fit_categories(train_val, cat_features),
        numeric_ranges={
            f: [float(train_val[f].min()), float(train_val[f].max())] for f in numeric
        },
        rounds={
            name: int(model.booster.best_iteration + 1)
            for name, model in production.items()
        },
        n_train=len(train_val),
        window=float(window),
    )


def model_card(
    *,
    frame: pd.DataFrame,
    split: dict,
    models: dict[str, Model],
    best_name: str,
    evaluation_metrics: pd.DataFrame,
    production: Production,
    train_val: pd.DataFrame,
    settings: dict,
) -> dict:
    """Description of a trained version, stored next to its artifacts.

    Parameters
    ----------
    frame : pd.DataFrame
        The whole training table after validation (all three splits).
    split : dict
        Split dates and sizes (``t_val``, ``t_test``, ``n_train``, ``n_val``,
        ``n_test``).
    models : dict[str, Model]
        Evaluation models keyed by kind.
    best_name : str
        Kind selected on validation.
    evaluation_metrics : pd.DataFrame
        Metrics of the evaluation models indexed by ``(model, split)``.
    production : Production
        Result of ``build_production``.
    train_val : pd.DataFrame
        Rows the production models were fitted on; profiled for drift monitoring.
    settings : dict
        Run settings worth recording (number of trials, bootstrap replicates, seed).

    Returns
    -------
    dict
        JSON-serialisable card: data fingerprint, split, parameters and rounds
        per kind, library versions, metrics, gate result and the training profile.
    """
    features, cat_features = models[best_name].features, models[best_name].cat_features
    return {
        "format": FORMAT_VERSION,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "best_name": best_name,
        "forecast_name": production.forecast_name,
        "data": {
            "sha256": frame_hash(frame),
            "n_rows": len(frame),
            "n_events": int(frame["event"].sum()),
            "n_fit": production.n_train,
            "split": split,
        },
        "settings": settings,
        "models": {
            kind: {
                "params": model.params,
                "rounds": production.rounds[kind],
                "evaluation_calibration": model.calibration,
                "production_calibration": production.models[kind].calibration,
            }
            for kind, model in models.items()
        },
        "versions": library_versions(),
        "metrics": {
            "evaluation": {
                kind: {
                    split_name: evaluation_metrics.loc[(kind, split_name)].to_dict()
                    for split_name in ("val", "test")
                }
                for kind in models
            },
            "production": {
                kind: {"test": production.metrics.loc[kind].to_dict()}
                for kind in production.models
            },
        },
        "gate": production.gate,
        "rul": {"horizon": production.horizon, "window": production.window},
        "profile": training_profile(train_val, features, cat_features),
    }

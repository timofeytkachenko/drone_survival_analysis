import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import norm
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

logger = logging.getLogger(__name__)


def to_category(data: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Return a copy of ``data`` with ``cat_features`` cast to ``category`` dtype.

    Boolean columns are left as ``bool``: XGBoost rejects boolean category
    levels and already splits a binary feature optimally.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data.
    cat_features : list[str]
        Columns to treat as categorical.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with categorical dtype on the non-boolean ``cat_features``.
    """
    data = data.copy()
    cols = [c for c in cat_features if not pd.api.types.is_bool_dtype(data[c])]
    data[cols] = data[cols].astype("category")
    return data


def fit_categories(data: pd.DataFrame, cat_features: list[str]) -> dict[str, list]:
    """Learn the category vocabulary of each categorical column.

    Fit on the training split only. Only levels actually observed in ``data``
    are kept, so unused levels of a ``category`` dtype built on the full
    dataset do not leak into the vocabulary. Boolean columns are skipped, as
    in ``to_category``.

    Parameters
    ----------
    data : pd.DataFrame
        Training data.
    cat_features : list[str]
        Categorical columns.

    Returns
    -------
    dict[str, list]
        Mapping column -> sorted list of observed levels.
    """
    return {
        c: sorted(data[c].dropna().unique().tolist())
        for c in cat_features
        if not pd.api.types.is_bool_dtype(data[c])
    }


def encode_unknown_categories(
    data: pd.DataFrame, categories: dict[str, list]
) -> pd.DataFrame:
    """Cast categoricals to the training vocabulary; unseen levels become NaN.

    XGBoost >= 3 raises on categories absent from the training set, while NaN
    is routed along each split's default branch.

    Parameters
    ----------
    data : pd.DataFrame
        New data (validation, test or inference).
    categories : dict[str, list]
        Output of ``fit_categories``.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with categorical columns restricted to ``categories``.
    """
    data = data.copy()
    for col, levels in categories.items():
        values = data[col].astype("object")
        unknown = values.notna() & ~values.isin(levels)
        if unknown.any():
            logger.warning(
                "%s: %d rows with unseen categories %s set to NaN",
                col,
                int(unknown.sum()),
                sorted(values[unknown].unique().tolist()),
            )
        data[col] = pd.Categorical(values.where(~unknown), categories=levels)
    return data


def _features_frame(
    data: pd.DataFrame, features: list[str], categories: dict[str, list] | None
) -> pd.DataFrame:
    """Select features, apply the categorical vocabulary and validate dtypes."""
    X = data[features]
    if categories is not None:
        X = encode_unknown_categories(
            X, {c: lv for c, lv in categories.items() if c in features}
        )
    invalid = X.select_dtypes(exclude=["number", "bool", "category"]).columns
    if len(invalid):
        raise TypeError(
            f"Unsupported feature dtypes (drop or convert): "
            f"{X[invalid].dtypes.astype(str).to_dict()}"
        )
    # XGBoost category levels must be str or int.
    bool_cats = [
        c
        for c in X.select_dtypes("category").columns
        if pd.api.types.is_bool_dtype(X[c].cat.categories)
    ]
    if bool_cats:
        raise TypeError(
            f"Boolean categoricals are not supported by XGBoost, keep them as bool: "
            f"{bool_cats}"
        )
    return X


def make_aft_df(
    data: pd.DataFrame,
    features: list[str],
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
    """Features and AFT interval labels: [t, t] for events, [t, +inf) for censored.

    Parameters
    ----------
    data : pd.DataFrame
        Rows with ``duration_h`` and ``event`` plus the ``features`` columns.
    features : list[str]
        Feature columns to return.
    categories : dict[str, list] | None, default None
        Output of ``fit_categories``; unseen levels become NaN.

    Returns
    -------
    tuple[pd.DataFrame, tuple[numpy.ndarray, numpy.ndarray]]
        Feature frame ``X`` and the ``(lower, upper)`` label bounds ``y``.
    """
    X = _features_frame(data, features, categories)
    lower = data["duration_h"].to_numpy()
    upper = np.where(data["event"] == 1, lower, np.inf)
    return X, (lower, upper)


def make_cox_df(
    data: pd.DataFrame,
    features: list[str],
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and Cox labels: +t for events, -t for right-censored.

    Parameters
    ----------
    data : pd.DataFrame
        Rows with ``duration_h`` and ``event`` plus the ``features`` columns.
    features : list[str]
        Feature columns to return.
    categories : dict[str, list] | None, default None
        Output of ``fit_categories``; unseen levels become NaN.

    Returns
    -------
    tuple[pd.DataFrame, numpy.ndarray]
        Feature frame ``X`` and the signed survival-time label ``y``.
    """
    X = _features_frame(data, features, categories)
    t = data["duration_h"].to_numpy()
    y = np.where(data["event"] == 1, t, -t)
    return X, y


def _breslow_baseline(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimate of the cumulative baseline hazard.

    Parameters
    ----------
    time : numpy.ndarray
        Observed durations of the training rows.
    event : numpy.ndarray
        Boolean event indicators (True = failure).
    risk : numpy.ndarray
        ``exp(margin)`` of each training row under the fitted Cox model.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Distinct event times and the cumulative baseline hazard at each of them.
    """
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    # Risk-set sum for each row: sum of risk over rows with time >= own time.
    risk_sum = np.cumsum(risk[order][::-1])[::-1]
    ev_times = np.unique(time[event])
    at_risk = risk_sum[np.searchsorted(t_sorted, ev_times, side="left")]
    deaths = np.bincount(
        np.searchsorted(ev_times, time[event]), minlength=len(ev_times)
    )
    return ev_times, np.cumsum(deaths / at_risk)


def _epanechnikov(grid: np.ndarray, times: np.ndarray, bandwidth: float) -> np.ndarray:
    """Kernel weights ``K((grid - times) / b) / b``, shape ``(len(grid), len(times))``."""
    z = (np.asarray(grid, dtype=float)[:, None] - times[None, :]) / bandwidth
    return np.where(np.abs(z) < 1, 0.75 * (1 - z**2), 0.0) / bandwidth


def hazard_bandwidth(time: np.ndarray, event: np.ndarray) -> float:
    """Kernel bandwidth for hazard smoothing: Silverman's rule on the event times.

    Parameters
    ----------
    time : numpy.ndarray
        Observed durations.
    event : numpy.ndarray
        Boolean event indicators (True = failure).

    Returns
    -------
    float
        Bandwidth in the units of ``time``.

    Raises
    ------
    ValueError
        If there are fewer than two events.
    """
    t = np.asarray(time, dtype=float)[np.asarray(event, dtype=bool)]
    if len(t) < 2:
        raise ValueError("At least two events are needed to choose a bandwidth")
    iqr = np.subtract(*np.quantile(t, [0.75, 0.25]))
    spread = min(t.std(ddof=1), iqr / 1.34) if iqr > 0 else t.std(ddof=1)
    return float(0.9 * spread * len(t) ** -0.2)


def observed_hazard(
    time: np.ndarray, event: np.ndarray, grid: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel-smoothed Nelson-Aalen hazard with a pointwise 95% interval.

    Estimates are biased downwards within ``bandwidth`` of 0 and of the last
    event time (the kernel loses mass there), so keep ``grid`` away from them.

    Parameters
    ----------
    time : numpy.ndarray
        Observed durations.
    event : numpy.ndarray
        Boolean event indicators (True = failure).
    grid : numpy.ndarray
        Evaluation times.
    bandwidth : float
        Epanechnikov kernel half-width, in the units of ``time``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Hazard per unit of time, lower and upper 95% bounds (lower clipped at 0).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    ev_times, deaths = np.unique(time[event], return_counts=True)
    at_risk = len(time) - np.searchsorted(np.sort(time), ev_times, side="left")
    weights = _epanechnikov(grid, ev_times, bandwidth)
    hazard = weights @ (deaths / at_risk)
    se = np.sqrt(weights**2 @ (deaths / at_risk**2))
    return hazard, np.maximum(hazard - 1.96 * se, 0.0), hazard + 1.96 * se


def _frailty_link(
    cum_hazard: np.ndarray, frailty_variance: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-frailty link ``log(1 + v * scale * H) / v`` and its derivative in ``H``.

    It is the cumulative hazard of a group whose members have cumulative hazard
    ``Z * scale * H`` with an unobserved ``Z ~ Gamma(mean 1, variance v)``. The
    limit ``v -> 0`` is ``scale * H``.
    """
    x = scale * np.asarray(cum_hazard, dtype=float)
    if frailty_variance < 1e-8:
        return x, np.full_like(x, scale)
    return (
        np.log1p(frailty_variance * x) / frailty_variance,
        scale / (1.0 + frailty_variance * x),
    )


def _fit_frailty_link(cum_hazard: np.ndarray, event: np.ndarray) -> dict[str, float]:
    """Maximum-likelihood parameters of ``_frailty_link`` on right-censored data.

    Parameters
    ----------
    cum_hazard : numpy.ndarray
        Predicted cumulative hazard of each row at its observed duration.
    event : numpy.ndarray
        Boolean event indicators (True = failure).

    Returns
    -------
    dict[str, float]
        ``frailty_variance`` (>= 0) and ``scale`` (> 0).

    Raises
    ------
    RuntimeError
        If the optimiser does not converge.
    """

    def negative_log_likelihood(params: np.ndarray) -> float:
        linked, slope = _frailty_link(cum_hazard, params[0], np.exp(params[1]))
        return linked.sum() - np.log(slope[event]).sum()

    result = minimize(
        negative_log_likelihood,
        x0=[0.1, 0.0],
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
    )
    if not result.success:
        raise RuntimeError(f"Frailty link did not converge: {result.message}")
    return {
        "frailty_variance": float(result.x[0]),
        "scale": float(np.exp(result.x[1])),
    }


class Model:
    """XGBoost survival model: Cox (``survival:cox``) or AFT (``survival:aft``).

    Cox survival curves use the Breslow baseline hazard estimated on the
    training data; AFT curves assume the normal (log-normal time) distribution.
    After ``calibrate`` the curves are corrected for unobserved heterogeneity
    between drones (attribute ``calibration``).

    Parameters
    ----------
    kind : str
        ``"cox"`` or ``"aft"``.
    params : dict
        XGBoost booster parameters. For ``"aft"`` must contain
        ``aft_loss_distribution_scale``.
    features : list[str]
        Feature columns used by the model.
    cat_features : list[str]
        Subset of ``features`` treated as categorical.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"cox"`` or ``"aft"``.
    """

    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50

    def __init__(
        self,
        kind: str,
        params: dict,
        features: list[str],
        cat_features: list[str],
    ) -> None:
        if kind not in ("cox", "aft"):
            raise ValueError(f"kind must be 'cox' or 'aft', got {kind!r}")
        self.kind = kind
        self.params = dict(params)
        self.features = features
        self.cat_features = cat_features
        self.booster: xgb.Booster | None = None
        self.categories: dict[str, list] | None = None
        self.history: dict[str, dict[str, list[float]]] = {}
        self._breslow: tuple[np.ndarray, np.ndarray] | None = None
        self.calibration: dict[str, float] | None = None

    def _dmatrix(self, data: pd.DataFrame) -> xgb.DMatrix:
        """Build a labelled DMatrix for ``data``."""
        if self.kind == "cox":
            X, y = make_cox_df(data, self.features, self.categories)
            return xgb.DMatrix(X, label=y, enable_categorical=True)
        X, (lower, upper) = make_aft_df(data, self.features, self.categories)
        dm = xgb.DMatrix(X, enable_categorical=True)
        dm.set_float_info("label_lower_bound", lower)
        dm.set_float_info("label_upper_bound", upper)
        return dm

    def _margin(self, dm: xgb.DMatrix) -> np.ndarray:
        """Raw margin using only the trees up to the early-stopping best iteration."""
        return self.booster.predict(
            dm,
            output_margin=True,
            iteration_range=(0, self.booster.best_iteration + 1),
        )

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "Model":
        """Fit on ``train`` with early stopping on ``valid``.

        Parameters
        ----------
        train : pd.DataFrame
            Training rows with ``duration_h`` and ``event``.
        valid : pd.DataFrame
            Validation rows used for early stopping.

        Returns
        -------
        Model
            The fitted model (``self``).
        """
        self.calibration = None
        self.categories = fit_categories(train, self.cat_features)
        dtrain = self._dmatrix(train)
        dvalid = self._dmatrix(valid)

        if self.kind == "cox":
            objective, metric = "survival:cox", "cox-nloglik"
        else:
            objective, metric = "survival:aft", "aft-nloglik"
        params = {"objective": objective, "eval_metric": metric, **self.params}
        if self.kind == "aft":
            params["aft_loss_distribution"] = "normal"
        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.NUM_BOOST_ROUND,
            # Early stopping monitors the last entry, so "valid" must stay last.
            evals=[(dtrain, "train"), (dvalid, "valid")],
            evals_result=self.history,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        if self.kind == "cox":
            risk = np.exp(self._margin(dtrain))
            self._breslow = _breslow_baseline(
                train["duration_h"].to_numpy(),
                train["event"].to_numpy().astype(bool),
                risk,
            )
        return self

    def calibrate(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        n_folds: int = 5,
        seed: int = 0,
    ) -> "Model":
        """Correct the survival curves for unobserved heterogeneity between drones.

        Drones with the same features still differ in durability. The weaker ones
        fail first, so the survivors of a high-risk group are sturdier than their
        features suggest, and a model without this effect overstates their hazard
        at later ages. The correction replaces the predicted cumulative hazard
        ``H(t | x)`` with ``log(1 + v * scale * H) / v`` (gamma frailty with
        variance ``v``); ``v = 0`` with ``scale = 1`` leaves the model unchanged.
        It applies to ``survival``, ``conditional_survival`` and ``hazard``; the
        ranking of drones at a fixed time does not change.

        ``v`` and ``scale`` are fitted by maximum likelihood on out-of-fold
        predictions for ``train``: the model is refitted ``n_folds`` times with
        the same parameters and each drone is predicted by the fit that did not
        see it. In-sample predictions would hide the effect.

        Parameters
        ----------
        train : pd.DataFrame
            Training rows with ``duration_h`` and ``event``.
        valid : pd.DataFrame
            Validation rows, used for early stopping of the fold fits.
        n_folds : int, default 5
            Number of cross-fitting folds.
        seed : int, default 0
            Seed of the fold assignment.

        Returns
        -------
        Model
            The model (``self``) with ``calibration`` set.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or the fit does not converge.
        ValueError
            If ``n_folds`` is less than 2.
        """
        if self.booster is None:
            raise RuntimeError("Model.fit must be called before calibrate")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        fold = np.random.default_rng(seed).permutation(len(train)) % n_folds
        cum_hazard = np.empty(len(train))
        for k in range(n_folds):
            held_out = fold == k
            rows = train[held_out]
            fold_model = Model(
                self.kind, self.params, self.features, self.cat_features
            ).fit(train[~held_out], valid)
            cum_hazard[held_out] = fold_model._cum_hazard_at(
                fold_model._predict_margin(rows),
                rows["duration_h"].to_numpy()[:, None],
            )[:, 0]
        self.calibration = _fit_frailty_link(
            cum_hazard, train["event"].to_numpy().astype(bool)
        )
        return self

    def _predict_margin(self, data: pd.DataFrame) -> np.ndarray:
        """Best-iteration margin for raw rows; requires a fitted model."""
        if self.booster is None or self.categories is None:
            raise RuntimeError("Model.fit must be called before predicting")
        X = _features_frame(data, self.features, self.categories)
        return self._margin(xgb.DMatrix(X, enable_categorical=True))

    def risk(self, data: pd.DataFrame) -> np.ndarray:
        """Risk score for ranking metrics: higher means earlier expected failure.

        Log hazard ratio for Cox, negative predicted log-time for AFT.

        Parameters
        ----------
        data : pd.DataFrame
            Rows with the ``features`` columns.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(data),)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        margin = self._predict_margin(data)
        return margin if self.kind == "cox" else -margin

    def explain(self, data: pd.DataFrame) -> shap.Explanation:
        """TreeSHAP values of the ``risk`` score.

        Uses the trees up to the early-stopping best iteration, like ``risk``.
        AFT values are negated, so for both kinds a positive SHAP value pushes
        towards earlier failure. Categorical features are passed to ``shap`` as
        integer codes (NaN for unseen levels) so plots can colour by them.

        Parameters
        ----------
        data : pd.DataFrame
            Rows with the ``features`` columns.

        Returns
        -------
        shap.Explanation
            SHAP values of shape ``(len(data), len(features))``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.booster is None or self.categories is None:
            raise RuntimeError("Model.fit must be called before explain")
        X = _features_frame(data, self.features, self.categories)
        explainer = shap.TreeExplainer(self.booster)
        sign = 1.0 if self.kind == "cox" else -1.0
        numeric = X.apply(
            lambda col: (
                col.cat.codes.where(col.cat.codes >= 0)
                if isinstance(col.dtype, pd.CategoricalDtype)
                else col
            )
        ).astype(float)
        return shap.Explanation(
            values=sign * explainer.shap_values(X),
            base_values=np.full(len(X), sign * float(explainer.expected_value)),
            data=numeric.to_numpy(),
            feature_names=list(X.columns),
        )

    def survival(self, data: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Predicted survival probabilities ``S(t | x)``.

        Parameters
        ----------
        data : pd.DataFrame
            Rows with the ``features`` columns (durations are not used).
        times : numpy.ndarray
            Evaluation times in flight hours.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(data), len(times))``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        return self._survival_at(self._predict_margin(data), times)

    def _cum_hazard_at(self, margin: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Cumulative hazard ``-log S(t | x)`` before ``calibration``.

        ``times`` is ``(m,)`` shared or ``(n, m)`` per row.
        """
        times = np.atleast_2d(np.asarray(times, dtype=float))

        if self.kind == "cox":
            ev_times, h0 = self._breslow
            idx = np.searchsorted(ev_times, times, side="right")
            cum_hazard = np.where(idx > 0, h0[np.maximum(idx - 1, 0)], 0.0)
            return np.exp(margin)[:, None] * cum_hazard

        sigma = self.params["aft_loss_distribution_scale"]
        with np.errstate(divide="ignore"):
            log_times = np.log(times)
        return -norm.logsf((log_times - margin[:, None]) / sigma)

    def _survival_at(self, margin: np.ndarray, times: np.ndarray) -> np.ndarray:
        """``S(t | x)`` from margins; ``times`` is ``(m,)`` shared or ``(n, m)`` per row."""
        cum_hazard = self._cum_hazard_at(margin, times)
        if self.calibration is not None:
            cum_hazard, _ = _frailty_link(cum_hazard, **self.calibration)
        return np.exp(-cum_hazard)

    def conditional_survival(
        self, data: pd.DataFrame, ages: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Survival for ``times`` more hours given survival up to ``ages``.

        ``S(a + u | x) / S(a | x)``; valid because features are fixed at
        commissioning. For Cox the Breslow baseline is flat after the last
        training event time, so curves beyond it are optimistic (a warning is
        logged).

        Parameters
        ----------
        data : pd.DataFrame
            Rows with the ``features`` columns.
        ages : numpy.ndarray
            Flight hours already survived, one per row.
        times : numpy.ndarray
            Additional flight hours ``u >= 0``, shared by all rows.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(data), len(times))``; NaN where ``S(a) = 0``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        margin = self._predict_margin(data)
        ages = np.asarray(ages, dtype=float)[:, None]
        times = np.asarray(times, dtype=float)[None, :]
        if self.kind == "cox" and (ages + times).max() > self._breslow[0][-1]:
            logger.warning(
                "Cox baseline hazard is flat after %.0f h; conditional survival "
                "beyond it is optimistic",
                self._breslow[0][-1],
            )
        s_age = self._survival_at(margin, ages)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                s_age > 0, self._survival_at(margin, ages + times) / s_age, np.nan
            )

    def hazard(
        self, data: pd.DataFrame, times: np.ndarray, bandwidth: float | None = None
    ) -> np.ndarray:
        """Hazard ``h(t | x)``: failure rate per flight hour among drones alive at t.

        AFT uses the exact log-normal hazard. The Breslow baseline is a step
        function, so for Cox its jumps are smoothed with an Epanechnikov kernel
        and scaled by ``exp(f(x))``; that estimate is biased downwards within
        ``bandwidth`` of 0 and of the last training event time. With
        ``calibration`` the hazard is multiplied by the slope of the link.

        Parameters
        ----------
        data : pd.DataFrame
            Rows with the ``features`` columns.
        times : numpy.ndarray
            Evaluation times in flight hours.
        bandwidth : float, optional
            Kernel half-width in flight hours; required for Cox, ignored for AFT.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(data), len(times))``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If ``bandwidth`` is missing for a Cox model.
        """
        margin = self._predict_margin(data)
        times = np.asarray(times, dtype=float)
        if self.kind == "cox":
            if bandwidth is None:
                raise ValueError("bandwidth is required for the Cox hazard")
            ev_times, h0 = self._breslow
            baseline = _epanechnikov(times, ev_times, bandwidth) @ np.diff(h0, prepend=0.0)
            hazard = np.exp(margin)[:, None] * baseline[None, :]
        else:
            sigma = self.params["aft_loss_distribution_scale"]
            with np.errstate(divide="ignore"):
                z = (np.log(times)[None, :] - margin[:, None]) / sigma
            # pdf / sf in log space stays finite deep in the right tail.
            with np.errstate(divide="ignore", invalid="ignore"):
                hazard = np.where(
                    times > 0,
                    np.exp(norm.logpdf(z) - norm.logsf(z)) / (sigma * times),
                    0.0,
                )
        if self.calibration is None:
            return hazard
        _, slope = _frailty_link(self._cum_hazard_at(margin, times), **self.calibration)
        return slope * hazard

    def save(self, path: str | Path) -> None:
        """Save the fitted model, including the state ``survival`` needs, as JSON.

        ``booster.save_model`` alone drops the Breslow baseline and the category
        vocabulary, so a model restored from it cannot produce survival curves.

        Parameters
        ----------
        path : str | Path
            Output file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.booster is None or self.categories is None:
            raise RuntimeError("Model.fit must be called before save")
        state = {
            "kind": self.kind,
            "params": self.params,
            "features": self.features,
            "cat_features": self.cat_features,
            "categories": self.categories,
            "history": self.history,
            "breslow": None
            if self._breslow is None
            else [arr.tolist() for arr in self._breslow],
            "calibration": self.calibration,
            "booster": json.loads(self.booster.save_raw("json")),
        }
        Path(path).write_text(json.dumps(state))

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        """Load a model written by ``save``.

        Parameters
        ----------
        path : str | Path
            File produced by ``Model.save``.

        Returns
        -------
        Model
            Fitted model ready for ``survival``.
        """
        state = json.loads(Path(path).read_text())
        model = cls(
            state["kind"], state["params"], state["features"], state["cat_features"]
        )
        model.categories = state["categories"]
        model.history = state["history"]
        if state["breslow"] is not None:
            ev_times, h0 = state["breslow"]
            model._breslow = (np.asarray(ev_times), np.asarray(h0))
        # Files written before ``calibrate`` existed have no such key.
        model.calibration = state.get("calibration")
        model.booster = xgb.Booster()
        model.booster.load_model(bytearray(json.dumps(state["booster"]).encode()))
        return model


def _surv(data: pd.DataFrame) -> np.ndarray:
    """Structured ``(event, time)`` array expected by ``sksurv`` metrics."""
    return Surv.from_arrays(data["event"].astype(bool), data["duration_h"])


SCALAR_METRICS = ("c_index", "c_index_ipcw", "mean_auc", "ibs")
LOWER_IS_BETTER = {
    "c_index": False,
    "c_index_ipcw": False,
    "mean_auc": False,
    "ibs": True,
}


def evaluation_times(*splits: pd.DataFrame, n_times: int = 50) -> np.ndarray:
    """Evaluation grid inside the follow-up range of every split.

    ``sksurv`` needs the times inside the follow-up of the evaluated rows, and IPCW
    weights grow without bound where few of them are still observed, so the grid
    covers the part of the 5%-80% ``duration_h`` quantile range shared by all
    ``splits``. Metrics computed on one grid are comparable across the splits.

    Parameters
    ----------
    *splits : pd.DataFrame
        Row sets with ``duration_h`` that will be evaluated on the grid.
    n_times : int, default 50
        Number of grid points.

    Returns
    -------
    numpy.ndarray
        Increasing times in flight hours, shape ``(n_times,)``.
    """
    bounds = np.array([np.quantile(d["duration_h"], [0.05, 0.8]) for d in splits])
    return np.linspace(bounds[:, 0].max(), bounds[:, 1].min(), n_times)


def _scalar_metrics(
    y_reference: np.ndarray,
    y: np.ndarray,
    surv: np.ndarray,
    risk: np.ndarray,
    times: np.ndarray,
) -> dict[str, float]:
    """``SCALAR_METRICS`` from precomputed survival probabilities and risk scores."""
    _, mean_auc = cumulative_dynamic_auc(y_reference, y, 1 - surv, times)
    return {
        "c_index": concordance_index_censored(y["event"], y["time"], risk)[0],
        "c_index_ipcw": concordance_index_ipcw(y_reference, y, risk, tau=times[-1])[0],
        "mean_auc": mean_auc,
        "ibs": integrated_brier_score(y_reference, y, surv, times),
    }


def evaluate(
    model: Model, reference: pd.DataFrame, data: pd.DataFrame, times: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Discrimination and calibration metrics of ``model`` on ``data``.

    IPCW-based metrics estimate the censoring distribution on ``reference`` with
    Kaplan-Meier, i.e. they assume censoring independent of the features.

    Parameters
    ----------
    model : Model
        Fitted model.
    reference : pd.DataFrame
        Rows used only for the censoring distribution. They must be censored
        like ``data``. With a split by commissioning date newer drones are
        censored earlier, so pass ``data`` itself: the training rows would
        understate the weights and with them the Brier score.
    data : pd.DataFrame
        Rows to evaluate on.
    times : numpy.ndarray
        Increasing evaluation times inside the follow-up range of ``data``
        (``sksurv`` requires ``min <= t < max`` of its durations).

    Returns
    -------
    dict[str, float | numpy.ndarray]
        Scalars ``c_index`` (Harrell), ``c_index_ipcw`` (Uno, truncated at
        ``times[-1]``), ``mean_auc`` and ``ibs``; per-time arrays ``brier``
        and ``auc`` aligned with ``times``.
    """
    y_reference, y = _surv(reference), _surv(data)
    surv = model.survival(data, times)
    _, brier = brier_score(y_reference, y, surv, times)
    auc, _ = cumulative_dynamic_auc(y_reference, y, 1 - surv, times)
    return {
        **_scalar_metrics(y_reference, y, surv, model.risk(data), times),
        "brier": brier,
        "auc": auc,
    }


def bootstrap_compare(
    models: dict[str, Model],
    reference: pd.DataFrame,
    data: pd.DataFrame,
    times: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap comparison of fitted models on ``data``.

    Every replicate resamples rows of ``data`` with replacement and scores all
    models on the same rows, so metric differences account for the correlation
    between models. Models are fixed: the intervals reflect evaluation-set
    sampling noise only, not refitting variability. The censoring distribution
    is estimated once on ``reference`` and kept fixed across replicates.

    Parameters
    ----------
    models : dict[str, Model]
        Fitted models keyed by name; every pair is compared.
    reference : pd.DataFrame
        Rows used only for the IPCW censoring distribution, as in ``evaluate``
        (normally ``data`` itself).
    data : pd.DataFrame
        Evaluation rows (e.g. the test split).
    times : numpy.ndarray
        Evaluation times, as in ``evaluate``.
    n_boot : int, default 1000
        Number of bootstrap replicates.
    seed : int, default 0
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    summary : pd.DataFrame
        One row per model pair and metric: point estimates ``value_a`` and
        ``value_b``, ``diff = value_a - value_b``, 95% percentile interval of
        the difference (``ci_low``, ``ci_high``), two-sided bootstrap
        ``p_value`` (resolution ``2 / n_boot``; 0 means below it), the
        ``better`` model by point estimate and ``significant`` (interval
        excludes 0).
    samples : pd.DataFrame
        Bootstrap replicates, columns ``(model, metric)``.
    """
    y_reference, y = _surv(reference), _surv(data)
    preds = {
        name: (model.survival(data, times), model.risk(data))
        for name, model in models.items()
    }
    point = {
        name: _scalar_metrics(y_reference, y, surv, risk, times)
        for name, (surv, risk) in preds.items()
    }

    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        records.append(
            {
                (name, metric): value
                for name, (surv, risk) in preds.items()
                for metric, value in _scalar_metrics(
                    y_reference, y[idx], surv[idx], risk[idx], times
                ).items()
            }
        )
    samples = pd.DataFrame(records)
    samples.columns = pd.MultiIndex.from_tuples(
        samples.columns, names=["model", "metric"]
    )

    rows = []
    for a, b in combinations(models, 2):
        for metric in SCALAR_METRICS:
            diff_boot = samples[(a, metric)] - samples[(b, metric)]
            diff = point[a][metric] - point[b][metric]
            ci_low, ci_high = np.quantile(diff_boot, [0.025, 0.975])
            p_value = min(
                1.0, 2 * min((diff_boot <= 0).mean(), (diff_boot >= 0).mean())
            )
            a_better = (diff < 0) if LOWER_IS_BETTER[metric] else (diff > 0)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "value_a": point[a][metric],
                    "value_b": point[b][metric],
                    "diff": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "better": a if a_better else b,
                    "significant": not ci_low <= 0 <= ci_high,
                }
            )
    return pd.DataFrame(rows), samples


def calibration_curve(
    model: Model, data: pd.DataFrame, horizon: float, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed survival at ``horizon`` by quantile bins of the prediction.

    Observed survival in each bin is the Kaplan-Meier estimate at ``horizon``.
    If a bin has no follow-up up to ``horizon``, its last Kaplan-Meier value is
    used, so keep ``horizon`` well inside the follow-up range.

    Parameters
    ----------
    model : Model
        Fitted model.
    data : pd.DataFrame
        Rows with ``duration_h``, ``event`` and the model features.
    horizon : float
        Time point in flight hours.
    n_bins : int, default 10
        Number of quantile bins; fewer are used if predictions have ties.

    Returns
    -------
    pd.DataFrame
        Columns ``predicted`` (mean predicted S), ``observed`` (Kaplan-Meier S)
        and ``n`` (rows in the bin), one row per bin.
    """
    pred = model.survival(data, np.array([horizon]))[:, 0]
    bins = pd.qcut(pred, n_bins, labels=False, duplicates="drop")
    event = data["event"].to_numpy().astype(bool)
    time = data["duration_h"].to_numpy()
    rows = []
    for b in np.unique(bins):
        mask = bins == b
        km_time, km_surv = kaplan_meier_estimator(event[mask], time[mask])
        seen = km_time <= horizon
        rows.append(
            {
                "predicted": pred[mask].mean(),
                "observed": km_surv[seen][-1] if seen.any() else 1.0,
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def population_hazard(
    model: Model, data: pd.DataFrame, times: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Predicted hazard of the whole group, comparable with ``observed_hazard``.

    Drones that already failed leave the risk set, so the group hazard is the
    survival-weighted mean ``sum S_i h_i / sum S_i``, not the plain mean of ``h_i``.

    Parameters
    ----------
    model : Model
        Fitted model.
    data : pd.DataFrame
        Rows with the model features.
    times : numpy.ndarray
        Evaluation times in flight hours.
    bandwidth : float
        Kernel half-width passed to ``Model.hazard`` (used by Cox only).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(times),)``; NaN where every drone has ``S = 0``.
    """
    surv = model.survival(data, times)
    weighted = (surv * model.hazard(data, times, bandwidth)).sum(axis=0)
    total = surv.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, weighted / total, np.nan)


def remaining_life(
    model: Model,
    data: pd.DataFrame,
    horizon: float,
    window: float = 100.0,
    n_grid: int = 1001,
) -> pd.DataFrame:
    """Remaining useful life of drones still in service.

    Uses the conditional survival ``S(a + u | x) / S(a | x)`` with the current
    age ``a = duration_h`` (flight hours survived so far).

    Parameters
    ----------
    model : Model
        Fitted model.
    data : pd.DataFrame
        Drones in service: ``drone_id``, ``duration_h`` and the model features.
    horizon : float
        Maximum additional flight hours considered.
    window : float, default 100.0
        Look-ahead window for ``p_fail_window``.
    n_grid : int, default 1001
        Grid points on ``[0, horizon]``; RUL resolution is
        ``horizon / (n_grid - 1)``.

    Returns
    -------
    pd.DataFrame
        Indexed like ``data`` with ``drone_id``, ``age_h``, ``rul_q10`` (hours
        until failure probability reaches 10%), ``rul_median`` (50%),
        ``rmrl`` (mean residual life restricted to ``horizon``) and
        ``p_fail_window`` (failure probability within ``window`` hours).
        ``rul_*`` is ``inf`` when the level is not reached within ``horizon``.

    Raises
    ------
    ValueError
        If ``window`` is not in ``(0, horizon]``.
    """
    if not 0 < window <= horizon:
        raise ValueError("window must be in (0, horizon]")
    ages = data["duration_h"].to_numpy(dtype=float)
    grid = np.linspace(0.0, horizon, n_grid)
    surv = model.conditional_survival(data, ages, grid)

    def first_crossing(level: float) -> np.ndarray:
        reached = surv <= level
        return np.where(reached.any(axis=1), grid[reached.argmax(axis=1)], np.inf)

    p_survive_window = model.conditional_survival(data, ages, np.array([window]))[:, 0]
    return pd.DataFrame(
        {
            "drone_id": data["drone_id"].to_numpy(),
            "age_h": ages,
            "rul_q10": first_crossing(0.9),
            "rul_median": first_crossing(0.5),
            "rmrl": np.trapezoid(surv, grid, axis=1),
            "p_fail_window": 1.0 - p_survive_window,
        },
        index=data.index,
    )


def tune(
    kind: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    trials: int,
    features: list[str],
    cat_features: list[str],
) -> Model:
    """Tune hyper-parameters with Optuna (TPE) on the validation integrated Brier score.

    The score uses ``evaluation_times(valid)`` and the censoring distribution of
    ``valid``, so it stays inside the validation follow-up.

    Parameters
    ----------
    kind : str
        ``"cox"`` or ``"aft"``.
    train, valid : pd.DataFrame
        Training and validation rows with ``duration_h`` and ``event``.
    trials : int
        Number of Optuna trials.
    features, cat_features : list[str]
        Passed to ``Model``.

    Returns
    -------
    Model
        Model refitted with the best parameters.
    """
    y_valid = _surv(valid)
    times = evaluation_times(valid)

    def objective(trial):

        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "eta": trial.suggest_float("eta", 0.025, 0.15, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 30, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }

        if kind != "cox":
            params["aft_loss_distribution_scale"] = trial.suggest_float(
                "aft_loss_distribution_scale",
                0.2,
                2.0,
                log=True,
            )

        model = Model(kind, params, features, cat_features).fit(train, valid)

        survival = model.survival(valid, times)

        return integrated_brier_score(
            y_valid,
            y_valid,
            survival,
            times,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=trials,
    )

    return Model(kind, study.best_params, features, cat_features).fit(train, valid)

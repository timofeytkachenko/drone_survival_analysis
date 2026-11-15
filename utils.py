import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sksurv.metrics import integrated_brier_score
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


class Model:
    """XGBoost survival model: Cox (``survival:cox``) or AFT (``survival:aft``).

    Cox survival curves use the Breslow baseline hazard estimated on the
    training data; AFT curves assume the normal (log-normal time) distribution.

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
        if self.booster is None or self.categories is None:
            raise RuntimeError("Model.fit must be called before survival")
        X = _features_frame(data, self.features, self.categories)
        margin = self._margin(xgb.DMatrix(X, enable_categorical=True))
        times = np.asarray(times, dtype=float)

        if self.kind == "cox":
            ev_times, h0 = self._breslow
            idx = np.searchsorted(ev_times, times, side="right")
            cum_hazard = np.where(idx > 0, h0[np.maximum(idx - 1, 0)], 0.0)
            return np.exp(-np.outer(np.exp(margin), cum_hazard))

        sigma = self.params["aft_loss_distribution_scale"]
        return norm.sf((np.log(times)[None, :] - margin[:, None]) / sigma)

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
        model.booster = xgb.Booster()
        model.booster.load_model(bytearray(json.dumps(state["booster"]).encode()))
        return model


def tune(
    kind: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    trials: int,
    features: list[str],
    cat_features: list[str],
) -> Model:
    """Tune hyper-parameters with Optuna (TPE) on the validation integrated Brier score.

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
    y_train = Surv.from_arrays(train["event"].astype(bool), train["duration_h"])
    y_valid = Surv.from_arrays(valid["event"].astype(bool), valid["duration_h"])
    TIMES = np.quantile(train["duration_h"], np.linspace(0.1, 0.9, 9))

    def objective(trial):

        params = {
            "max_depth": trial.suggest_int(
                "max_depth", 2, 5
            ),
            "eta": trial.suggest_float(
                "eta", 0.025, 0.15, log=True
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 30, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 20, log=True
            ),
        }

        if kind != "cox":

            params["aft_loss_distribution_scale"] = (
                trial.suggest_float(
                    "aft_loss_distribution_scale",
                    0.2,
                    1.2,
                    log=True,
                )
            )

        model = Model(kind, params, features, cat_features).fit(train, valid)

        survival = model.survival(valid, TIMES)

        return integrated_brier_score(
            y_train,
            y_valid,
            survival,
            TIMES,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42
        ),
    )

    study.optimize(
        objective,
        n_trials=trials,
    )

    return Model(kind, study.best_params, features, cat_features).fit(train, valid)
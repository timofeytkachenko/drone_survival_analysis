"""Framework-agnostic view logic shared by the Dash and Streamlit dashboards."""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard import insights, texts
from dashboard.data import DashboardData
from dashboard.labels import (
    FAILURE_MODE_LABELS,
    FEATURE_LABELS,
    METRIC_LABELS,
    MODEL_LABELS,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_ORDER,
    RISK_GROUP_ORDER,
    SEGMENT_LABELS,
)
from utils import (
    Model,
    hazard_bandwidth,
    observed_hazard,
    population_hazard,
    remaining_life,
)

SEGMENT_ORDER = {"risk_group": RISK_GROUP_ORDER, "priority": PRIORITY_ORDER}
SEGMENTS = {**SEGMENT_LABELS, "priority": "Maintenance priority"}
PRIORITY_COLORS = {
    PRIORITY_HIGH: "#f8d7da",
    PRIORITY_MEDIUM: "#fff3cd",
    PRIORITY_LOW: "#d1e7dd",
}


@dataclass(frozen=True)
class Settings:
    """Run settings derived from ``meta.json`` and the test predictions."""

    best: str
    features: list[str]
    cat_features: list[str]
    horizon: float
    window: float
    p_high: float
    p_medium: float
    surv_grid: np.ndarray
    rul_grid: np.ndarray
    hazard_bandwidth: float
    hazard_grid: np.ndarray

    @classmethod
    def from_data(cls, data: DashboardData) -> "Settings":
        """Build settings from loaded dashboard data."""
        meta = data.meta
        pred = data.predictions
        bandwidth = hazard_bandwidth(pred["duration_h"], pred["event"])
        return cls(
            best=data.best_name,
            features=meta["features"],
            cat_features=meta["cat_features"],
            horizon=meta["rul_horizon"],
            window=meta["rul_window"],
            p_high=meta["priority_thresholds"]["high"],
            p_medium=meta["priority_thresholds"]["medium"],
            surv_grid=np.linspace(0.0, data.predictions["duration_h"].max(), 300),
            rul_grid=np.linspace(0.0, meta["rul_horizon"], 300),
            hazard_bandwidth=bandwidth,
            # The smoothed hazard is biased within one bandwidth of 0, and beyond the
            # 90th percentile the observed estimate rests on too few drones.
            hazard_grid=np.linspace(bandwidth, pred["duration_h"].quantile(0.9), 200),
        )


def active(data: DashboardData) -> pd.DataFrame:
    """Test drones in service at the data cut-off (censored and not left the fleet)."""
    return data.predictions[data.predictions["in_service"]]


def text_context(data: DashboardData, settings: Settings) -> dict:
    """Placeholders for the ``texts`` templates."""
    meta = data.meta
    return {
        "window": settings.window,
        "horizon": settings.horizon,
        "p_high": settings.p_high,
        "p_medium": settings.p_medium,
        "landmark": meta["landmark"],
        "hazard_bandwidth": settings.hazard_bandwidth,
        "n_boot": meta["n_boot"],
        "n_trials": meta["n_trials"],
        "n_features": len(settings.features),
        "n_test": len(data.predictions),
        "n_active": len(active(data)),
        "n_censored": int((data.predictions["event"] == 0).sum()),
        "t_val": meta["split"]["t_val"],
        "t_test": meta["split"]["t_test"],
        "cat_features": ", ".join(
            FEATURE_LABELS[f].lower() for f in settings.cat_features
        ),
        "best": MODEL_LABELS[settings.best],
        "generated_at": meta["generated_at"].replace("T", " "),
    }


def fmt_hours(x: float, horizon: float) -> str:
    """Remaining-life hours, ``> horizon`` when not reached."""
    return f"> {horizon:.0f} h" if not np.isfinite(x) else f"{x:.0f} h"


def priority(p_fail: float, settings: Settings) -> str:
    """Maintenance priority for a failure probability within the window."""
    if p_fail >= settings.p_high:
        return PRIORITY_HIGH
    return PRIORITY_MEDIUM if p_fail >= settings.p_medium else PRIORITY_LOW


# --- Overview -----------------------------------------------------------------


def kpis(data: DashboardData, settings: Settings) -> list[tuple[str, str, str, str]]:
    """``(title, value, subtitle, colour)`` of the overview cards."""
    pred, act = data.predictions, active(data)
    test_metrics = (
        data.tables["metrics"]
        .query("model == @settings.best and split == 'test'")
        .iloc[0]
    )
    n_fail = int(pred["event"].sum())
    finite_rul = act.loc[np.isfinite(act["rul_median"]), "rul_median"]
    return [
        (
            "Drones in the test set",
            f"{len(pred)}",
            f"commissioned from {data.meta['split']['t_test']}",
            "primary",
        ),
        ("Failures", f"{n_fail}", f"{n_fail / len(pred):.0%} of the fleet", "danger"),
        ("In service", f"{len(act)}", "flying at the data cut-off", "success"),
        (
            "High maintenance priority",
            f"{int((act['priority'] == PRIORITY_HIGH).sum())}",
            f"P(failure in {settings.window:.0f} h) ≥ {settings.p_high:.0%}",
            "danger",
        ),
        (
            "Median remaining life",
            f"{finite_rul.median():.0f} h",
            f"drones in service ({len(finite_rul)} with an estimate)",
            "info",
        ),
        (
            f"{MODEL_LABELS[settings.best]} model",
            f"C = {test_metrics['c_index_ipcw']:.3f}",
            f"IBS = {test_metrics['ibs']:.3f} on test",
            "secondary",
        ),
    ]


KPI_HELP = (
    '**How to read the cards.** "Failures": how many drones have already gone out '
    'of service for a critical reason. "In service": drones still flying at the '
    "data cut-off; drones that left the fleet earlier without a failure (sold, lost, "
    'retired) are in neither group. "High maintenance priority": drones that, '
    "according to the model, will fail in the next {window:.0f} flight hours with "
    'probability ≥ {p_high:.0%}. "Median remaining life": how many more flight hours '
    "a typical working drone flies before its failure risk reaches 50%. C (C-index, "
    "Uno): quality of ranking drones by risk (0.5 is random, 1 is perfect); IBS: "
    "probability forecast error (0.25 is a 50/50 forecast, lower is better)."
)


def key_findings(data: DashboardData, settings: Settings) -> list[str]:
    """Headline conclusions for the overview tab."""
    importance = data.tables["shap_importance"]
    top = importance[importance["model"] == settings.best].nlargest(3, "share")
    factors = ", ".join(
        f'"{FEATURE_LABELS[f]}" ({s:.0%})' for f, s in zip(top["feature"], top["share"])
    )
    findings = [
        (
            f"The best model is {MODEL_LABELS[settings.best]} (lowest Integrated Brier "
            "score on validation)."
        ),
        f"Main failure risk factors: {factors} of the total influence.",
    ]
    summary = data.tables["bootstrap_summary"]
    for row in summary[summary["significant"]].itertuples():
        findings.append(
            f"Bootstrap: {MODEL_LABELS[row.better]} is statistically better on "
            f'"{METRIC_LABELS[row.metric]}" (p = {row.p_value:.3f}, without correction '
            "for multiple comparisons)."
        )
    if not summary["significant"].any():
        findings.append("Bootstrap: no significant difference between the models.")
    act = active(data)
    n_high = int((act["priority"] == PRIORITY_HIGH).sum())
    n_medium = int((act["priority"] == PRIORITY_MEDIUM).sum())
    findings.append(
        f"Of {len(act)} drones in service, {n_high} will fail in the next "
        f"{settings.window:.0f} h with probability ≥ {settings.p_high:.0%} and another "
        f"{n_medium} with ≥ {settings.p_medium:.0%}: these are candidates for "
        'unscheduled maintenance ("Fleet" tab).'
    )
    return findings


# --- Methodology --------------------------------------------------------------------


def hyperparams_frame(models: dict[str, Model]) -> pd.DataFrame:
    """Tuned hyper-parameters with plain-language meaning."""
    rows = []
    for key, (name, meaning) in texts.HYPERPARAMS.items():
        row = {"Setting": name, "Meaning": meaning}
        present = False
        for kind, model in models.items():
            value = (
                model.booster.best_iteration
                if key == "best_iteration"
                else {**model.params, **(model.calibration or {})}.get(key)
            )
            present |= value is not None
            if value is None:
                row[MODEL_LABELS[kind]] = "—"
            else:
                row[MODEL_LABELS[kind]] = (
                    value if isinstance(value, str) else f"{value:.4g}"
                )
        if present:
            rows.append(row)
    return pd.DataFrame(rows)


def features_frame(meta: dict) -> pd.DataFrame:
    """Feature dictionary with training ranges."""
    rows = []
    for f in meta["features"]:
        if f == "is_refurbished":
            kind, values = "yes/no", "True / False"
        elif f in meta["cat_features"]:
            kind, values = "category", ", ".join(map(str, meta["categories"][f]))
        else:
            low, high = meta["numeric_ranges"][f]
            kind, values = "number", f"{low:.3g} – {high:.3g}"
        rows.append(
            {
                "Feature": FEATURE_LABELS[f],
                "Column": f,
                "Description": texts.FEATURE_DESCRIPTIONS[f],
                "Type": kind,
                "Values in training": values,
            }
        )
    return pd.DataFrame(rows)


# --- Segments -----------------------------------------------------------------------


def segment_levels(pred: pd.DataFrame, segment: str) -> list:
    """Levels of ``segment`` in display order."""
    levels = pred[segment].dropna().unique().tolist()
    order = SEGMENT_ORDER.get(segment)
    return [lv for lv in order if lv in levels] if order else sorted(levels, key=str)


def segment_stats(
    pred: pd.DataFrame, segment: str, levels: list, window: float
) -> pd.DataFrame:
    """Per-level counts, Kaplan-Meier median and mean predicted failure probability."""
    groups = pred.groupby(segment, observed=True)
    stats = pd.DataFrame(
        {
            "Drones": groups["drone_id"].count(),
            "Failures": groups["event"].sum(),
            "Failure share": groups["event"].mean().round(3),
            "Median to failure, h": groups[["event", "duration_h"]].apply(
                lambda g: insights.hours_text(insights.km_median(g))
            ),
            "Mean flight hours, h": groups["duration_h"].mean().round(0),
            "In service": groups["in_service"].sum(),
            f"Mean P(failure in {window:.0f} h)": groups["p_fail_window"]
            .mean()
            .round(3),
        }
    )
    return stats.reindex(levels).rename_axis(SEGMENTS[segment]).reset_index()


def segment_findings(
    pred: pd.DataFrame, segment: str, levels: list, window: float
) -> list[str]:
    """Takeaways for a segment; the priority split is defined by the outcome itself."""
    if segment == "priority":
        return [
            (
                "Maintenance priority is defined by the forecast for **working** "
                'drones, while "None (failed)" and "None (left the fleet)" hold the '
                "drones that are no longer in service. So the survival curves here "
                "differ by construction and **do not compare reliability**."
            ),
            (
                "Use this breakdown for the table below: how many drones are in each "
                "priority group, their mean flight hours and mean failure probability "
                f"in {window:.0f} h."
            ),
        ]
    return insights.segments(pred, segment, levels)


# --- Model quality ----------------------------------------------------------------


def hazard_table(data: DashboardData, settings: Settings) -> pd.DataFrame:
    """Observed (smoothed Nelson-Aalen) and predicted group hazard on test.

    Columns ``source`` (``"observed"`` or the model kind), ``t``, ``h`` (per flight
    hour) and, for the observed rows, the 95% interval ``ci_low`` / ``ci_high``.
    """
    pred, grid, bandwidth = data.predictions, settings.hazard_grid, settings.hazard_bandwidth
    h, low, high = observed_hazard(pred["duration_h"], pred["event"], grid, bandwidth)
    frames = [
        pd.DataFrame(
            {"source": "observed", "t": grid, "h": h, "ci_low": low, "ci_high": high}
        ),
        *(
            pd.DataFrame(
                {
                    "source": kind,
                    "t": grid,
                    "h": population_hazard(model, pred, grid, bandwidth),
                }
            )
            for kind, model in data.models.items()
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Metric table with readable labels."""
    frame = metrics.copy()
    frame["model"] = frame["model"].map(MODEL_LABELS)
    return frame.rename(
        columns={"model": "Model", "split": "Split", **METRIC_LABELS}
    ).round(4)


def bootstrap_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap summary with readable labels."""
    frame = summary.copy()
    frame["metric"] = frame["metric"].map(METRIC_LABELS)
    for col in ("model_a", "model_b", "better"):
        frame[col] = frame[col].map(MODEL_LABELS)
    return frame.round(4)


# --- Risk factors -------------------------------------------------------------------


def beeswarm_findings(data: DashboardData, settings: Settings, kind: str) -> list[str]:
    """Direction of effect for the three most important features of ``kind``."""
    shap_values, importance = data.tables["shap_values"], data.tables["shap_importance"]
    top = importance[importance["model"] == kind].nlargest(3, "share")["feature"]
    return [
        f'"{FEATURE_LABELS[f]}": '
        + insights.shap_direction(shap_values, kind, f, f in settings.cat_features)[0]
        for f in top
    ]


# --- Fleet --------------------------------------------------------------------------


def fleet_columns(settings: Settings) -> dict[str, str]:
    """Column -> header of the fleet table."""
    return {
        "drone_id": "Drone",
        "model": "Model",
        "environment": "Environment",
        "age_h": "Flight hours, h",
        "p_fail_window": f"P(failure in {settings.window:.0f} h)",
        "rul_q10": "Life until 10% risk, h",
        "rul_median": "Median remaining life, h",
        "rmrl": "Expected remaining life, h",
        "risk_group": "Risk group",
        "priority": "Maintenance priority",
    }


def fleet_tooltips(settings: Settings) -> dict[str, str]:
    """Column -> header tooltip of the fleet table."""
    return {
        "age_h": "How many flight hours the drone has already worked without failure",
        "p_fail_window": f"Probability of failure in the next {settings.window:.0f} flight hours",
        "rul_q10": "After how many hours the failure probability reaches 10%",
        "rul_median": "After how many hours the failure probability reaches 50%",
        "rmrl": f"Mean remaining time within {settings.horizon:.0f} h",
        "risk_group": "Third of the fleet by model risk score (regardless of flight hours)",
        "priority": f"High: P ≥ {settings.p_high:.0%}, medium: P ≥ {settings.p_medium:.0%}",
    }


def fleet_frame(data: DashboardData, settings: Settings) -> pd.DataFrame:
    """Drones in service, riskiest first; ``inf`` remaining life becomes NaN."""
    columns = list(fleet_columns(settings))
    frame = active(data).sort_values("p_fail_window", ascending=False)[columns]
    return frame.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


# --- Drone card ---------------------------------------------------------------------


def drone_status(row: pd.Series) -> str:
    """One-line status of a drone."""
    if row["event"] == 1:
        mode = FAILURE_MODE_LABELS.get(row["failure_mode"], row["failure_mode"])
        return f"failed ({mode}) at {row['duration_h']:.0f} h"
    if not row["in_service"]:
        return f"left the fleet without failure at {row['duration_h']:.0f} h"
    return f"in service, {row['duration_h']:.0f} h, priority: {row['priority']}"


def drone_options(pred: pd.DataFrame) -> dict[str, str]:
    """drone_id -> label; drones in service first, riskiest first."""
    ordered = pred.sort_values(["event", "p_fail_window"], ascending=[True, False])
    return {
        row["drone_id"]: f"{row['drone_id']} · {row['model']} · {drone_status(row)}"
        for _, row in ordered.iterrows()
    }


def drone_rows(pred: pd.DataFrame, drone_id: str) -> pd.DataFrame:
    """Single-row frame of a drone."""
    return pred[pred["drone_id"] == drone_id]


def drone_metrics(row: pd.Series, settings: Settings) -> list[tuple[str, str, str]]:
    """``(name, value, hint)`` remaining-life summary of a drone in service."""
    return [
        (
            f"P(failure in {settings.window:.0f} h)",
            f"{row['p_fail_window']:.1%}",
            "probability of failure in the next flight hours",
        ),
        (
            "Life until 10% risk",
            fmt_hours(row["rul_q10"], settings.horizon),
            "conservative deadline for scheduled maintenance",
        ),
        (
            "Median remaining life",
            fmt_hours(row["rul_median"], settings.horizon),
            "the drone works longer with 50% probability",
        ),
        (
            "Expected remaining life",
            f"{row['rmrl']:.0f} h",
            f"mean within {settings.horizon:.0f} h",
        ),
    ]


def drone_features(row: pd.Series, features: list[str]) -> pd.DataFrame:
    """Feature values of a drone with readable names."""
    return pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in features],
            "Value": [str(row[f]) for f in features],
        }
    )


def drone_contrib(data: DashboardData, drone_id: str) -> pd.DataFrame:
    """SHAP contributions of the best model for one drone."""
    shap_values = data.tables["shap_values"]
    return shap_values[
        (shap_values["drone_id"] == drone_id) & (shap_values["model"] == data.best_name)
    ]


def drone_hazard_grid(settings: Settings, age: float) -> np.ndarray:
    """Hazard grid of a drone card: the fleet range, extended to the drone's next window.

    Beyond the fleet range the smoothed Cox baseline rests on few training failures
    and oscillates, so it is shown only as far as the drone's forecast needs.
    """
    end = max(settings.hazard_grid[-1], age + settings.window)
    return np.linspace(settings.hazard_bandwidth, end, 300)


def drone_findings(data: DashboardData, settings: Settings, drone_id: str) -> list[str]:
    """Narrative explanation of one drone."""
    rows = drone_rows(data.predictions, drone_id)
    curve = data.best_model.survival(rows, settings.surv_grid)[0]
    below = curve <= 0.5
    predicted_median = settings.surv_grid[below.argmax()] if below.any() else np.inf
    findings = insights.drone(
        rows.iloc[0],
        drone_contrib(data, drone_id),
        data.predictions,
        settings.best,
        settings.window,
        predicted_median,
    )
    row = rows.iloc[0]
    age = float(row["duration_h"])
    if row["in_service"] and settings.hazard_grid[0] <= age <= settings.hazard_grid[-1]:
        at_age = np.array([age])
        rate = data.best_model.hazard(rows, at_age, settings.hazard_bandwidth)[0, 0]
        pred = data.predictions
        fleet_rate = observed_hazard(
            pred["duration_h"], pred["event"], at_age, settings.hazard_bandwidth
        )[0][0]
        findings.append(insights.drone_hazard(age, rate, fleet_rate, settings.best))
    return findings


def whatif_defaults(row: pd.Series, features: list[str]) -> list:
    """Current feature values as plain Python objects (``is_refurbished`` as str)."""
    values = [str(row[f]) if f == "is_refurbished" else row[f] for f in features]
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def whatif_base(pred: pd.DataFrame, drone_id: str) -> pd.DataFrame:
    """Drone rows for the what-if; drones no longer in service restart from age 0."""
    rows = drone_rows(pred, drone_id).copy()
    if not rows["in_service"].iloc[0]:
        rows["duration_h"] = 0.0
    return rows


def whatif_scenario(
    rows: pd.DataFrame, features: list[str], values: list
) -> pd.DataFrame:
    """Copy of ``rows`` with features replaced by ``values``."""
    scenario = rows.copy()
    for feature, value in zip(features, values):
        scenario[feature] = value == "True" if feature == "is_refurbished" else value
    return scenario


def whatif_changes(
    row: pd.Series, features: list[str], cat_features: list[str], values: list
) -> list[tuple[str, object, object]]:
    """``(feature, old, new)`` for every edited feature."""
    changes = []
    for feature, value in zip(features, values):
        old = str(row[feature]) if feature == "is_refurbished" else row[feature]
        if feature in cat_features:
            changed = str(old) != str(value)
        else:
            changed = not np.isclose(float(old), float(value))
        if changed:
            changes.append((feature, old, value))
    return changes


def whatif_compare(
    model: Model, rows: pd.DataFrame, scenario: pd.DataFrame, settings: Settings
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Remaining-life summaries of current and scenario conditions and a comparison table."""
    base = remaining_life(model, rows, settings.horizon, settings.window).iloc[0]
    new = remaining_life(model, scenario, settings.horizon, settings.window).iloc[0]
    table = pd.DataFrame(
        {
            "": [
                f"P(failure in {settings.window:.0f} h)",
                "Maintenance priority",
                "Median remaining life",
                "Expected remaining life",
            ],
            "Now": [
                f"{base['p_fail_window']:.1%}",
                priority(base["p_fail_window"], settings),
                fmt_hours(base["rul_median"], settings.horizon),
                f"{base['rmrl']:.0f} h",
            ],
            "Scenario": [
                f"{new['p_fail_window']:.1%}",
                priority(new["p_fail_window"], settings),
                fmt_hours(new["rul_median"], settings.horizon),
                f"{new['rmrl']:.0f} h",
            ],
        }
    )
    return base, new, table


# --- Scoring new drones -------------------------------------------------------------

ID_COLUMN, AGE_COLUMN = "drone_id", "age_h"
MAX_UPLOAD_ROWS = 10_000
_BOOL_VALUES = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV file.

    Parameters
    ----------
    content : bytes
        Raw file content.

    Returns
    -------
    pd.DataFrame
        Parsed table.

    Raises
    ------
    ValueError
        If the content is empty or not a readable CSV file.
    """
    try:
        return pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"The file could not be read as CSV: {exc}") from exc


def upload_template(data: DashboardData, n: int = 5) -> pd.DataFrame:
    """Example upload: the riskiest drones in service with their current flight hours."""
    rows = active(data).nlargest(n, "p_fail_window")
    template = rows[[ID_COLUMN, *data.meta["features"]]].copy()
    template.insert(1, AGE_COLUMN, rows["duration_h"])
    return template.reset_index(drop=True)


def _to_bool(values: pd.Series) -> pd.Series:
    """``True/False``, ``1/0`` or ``yes/no`` text to bool; unknown text stays NaN."""
    if pd.api.types.is_bool_dtype(values):
        return values
    return values.astype(str).str.strip().str.lower().map(_BOOL_VALUES)


def _risk_group(risk: np.ndarray, reference: pd.Series) -> np.ndarray:
    """Place risk scores into the terciles of the reference risk, like ``pd.qcut`` in the notebook."""
    low, high = np.quantile(reference, [1 / 3, 2 / 3])
    return np.select(
        [risk <= low, risk <= high], [PRIORITY_LOW, PRIORITY_MEDIUM], PRIORITY_HIGH
    )


def score_upload(
    raw: pd.DataFrame, data: DashboardData, settings: Settings
) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded table of drones and forecast them with the best model.

    Parameters
    ----------
    raw : pd.DataFrame
        One row per drone with all ``settings.features`` columns, optionally
        ``drone_id`` and ``age_h`` (flight hours already flown without failure,
        default 0). Other columns are copied into the result.
    data : DashboardData
        Loaded artifacts (model, category vocabulary, training ranges, test risk).
    settings : Settings
        Run settings (horizon, window, priority thresholds).

    Returns
    -------
    scored : pd.DataFrame
        Input columns plus ``risk_group``, ``p_fail_window``, ``rul_q10``,
        ``rul_median``, ``rmrl``, ``priority`` and ``out_of_range``, riskiest
        first. Remaining life beyond the horizon is NaN.
    warnings : list[str]
        Data-quality issues that do not prevent scoring.

    Raises
    ------
    ValueError
        If the table is empty or too large, required columns are missing, or
        values cannot be interpreted.
    """
    if raw.empty:
        raise ValueError("The file has no rows.")
    if len(raw) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"The file has {len(raw):,} rows; at most {MAX_UPLOAD_ROWS:,} are supported."
        )
    missing = [f for f in settings.features if f not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    frame = raw.copy()
    if ID_COLUMN not in frame.columns:
        frame.insert(0, ID_COLUMN, [f"NEW-{i:04d}" for i in range(1, len(frame) + 1)])
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    if AGE_COLUMN not in frame.columns:
        frame[AGE_COLUMN] = 0.0

    categories = data.meta["categories"]
    numeric = [f for f in [*settings.features, AGE_COLUMN] if f not in settings.cat_features]
    errors = []
    for col in numeric:
        values = pd.to_numeric(frame[col], errors="coerce")
        bad = values.isna() & frame[col].notna()
        if bad.any():
            errors.append(f"{col}: non-numeric values in {int(bad.sum())} rows")
        frame[col] = values
    for col in (c for c in settings.cat_features if c not in categories):
        values = _to_bool(frame[col])
        bad = values.isna()
        if bad.any():
            errors.append(f"{col}: expected True/False in {int(bad.sum())} rows")
        else:
            frame[col] = values.astype(bool)
    if (frame[AGE_COLUMN] < 0).any():
        errors.append(f"{AGE_COLUMN}: negative values in {int((frame[AGE_COLUMN] < 0).sum())} rows")
    if frame[AGE_COLUMN].isna().any():
        errors.append(f"{AGE_COLUMN}: empty values in {int(frame[AGE_COLUMN].isna().sum())} rows")
    if errors:
        raise ValueError("Invalid values: " + "; ".join(errors) + ".")

    warnings = []
    for col, levels in categories.items():
        values = frame[col].astype("object")
        unknown = values.notna() & ~values.isin(levels)
        if unknown.any():
            seen = ", ".join(map(str, sorted(values[unknown].astype(str).unique())))
            warnings.append(
                f"{FEATURE_LABELS[col]}: {int(unknown.sum())} rows with values not seen "
                f"in training ({seen}); treated as missing."
            )
    n_missing = int(frame[settings.features].isna().any(axis=1).sum())
    if n_missing:
        warnings.append(
            f"{n_missing} rows have empty feature values; the model treats them as missing."
        )
    out_of_range = pd.Series(False, index=frame.index)
    for col, (low, high) in data.meta["numeric_ranges"].items():
        outside = (frame[col] < low) | (frame[col] > high)
        if outside.any():
            out_of_range |= outside
            warnings.append(
                f"{FEATURE_LABELS[col]}: {int(outside.sum())} rows outside the training "
                f"range {low:.3g} – {high:.3g}."
            )
    max_age = active(data)["duration_h"].max()
    too_old = frame[AGE_COLUMN] > max_age
    if too_old.any():
        warnings.append(
            f"{int(too_old.sum())} drones have {AGE_COLUMN} above {max_age:.0f} h (the "
            "oldest drone in the test fleet); their remaining-life forecast is optimistic."
        )

    model = data.best_model
    rows = frame.assign(duration_h=frame[AGE_COLUMN])
    life = remaining_life(model, rows, settings.horizon, settings.window)
    scored = frame.assign(
        risk_group=_risk_group(
            model.risk(rows), data.predictions[f"risk_{settings.best}"]
        ),
        p_fail_window=life["p_fail_window"],
        rul_q10=life["rul_q10"],
        rul_median=life["rul_median"],
        rmrl=life["rmrl"],
        out_of_range=out_of_range,
    )
    scored["priority"] = [priority(p, settings) for p in scored["p_fail_window"]]
    scored = scored.replace([np.inf, -np.inf], np.nan)
    return scored.sort_values("p_fail_window", ascending=False, ignore_index=True), warnings


def score_columns(settings: Settings) -> dict[str, str]:
    """Column -> header of the scoring results table."""
    return {**fleet_columns(settings), "out_of_range": "Outside training range"}


def score_summary(scored: pd.DataFrame, settings: Settings) -> list[str]:
    """Headline numbers of a scored upload."""
    counts = scored["priority"].value_counts()
    return [
        (
            f"{len(scored)} drones scored: high maintenance priority "
            f"**{counts.get(PRIORITY_HIGH, 0)}**, medium **{counts.get(PRIORITY_MEDIUM, 0)}**, "
            f"low {counts.get(PRIORITY_LOW, 0)}."
        ),
        (
            f"Mean P(failure in {settings.window:.0f} h) is "
            f"**{scored['p_fail_window'].mean():.1%}**; expected number of failures in "
            f"this group over the next {settings.window:.0f} flight hours: "
            f"**≈ {scored['p_fail_window'].sum():.1f}**."
        ),
        (
            f"{int(scored['out_of_range'].sum())} drones have values outside the training "
            "range; treat their forecast with caution."
        ),
    ]

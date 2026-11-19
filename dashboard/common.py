"""Framework-agnostic view logic shared by the Dash and Streamlit dashboards."""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard import insights
from dashboard.data import DashboardData
from dashboard.labels import (
    PRIORITY_FAILED,
    PRIORITY_HIGH,
    PRIORITY_LEFT,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_ORDER,
    RISK_GROUP_ORDER,
)
from dashboard.locale import bundle, tr
from utils import (
    Model,
    hazard_bandwidth,
    observed_hazard,
    population_hazard,
    remaining_life,
)

SEGMENT_ORDER = {"risk_group": RISK_GROUP_ORDER, "priority": PRIORITY_ORDER}
# Levers a maintainer can change; everything else is fixed when the drone is commissioned.
LEVERS = (
    "maintenance_interval_h",
    "operator_experience_yrs",
    "daily_utilization_h",
    "avg_flight_min",
    "avg_payload_kg",
    "payload_ratio",
)
PRIORITY_COLORS = {
    PRIORITY_HIGH: "#f8d7da",
    PRIORITY_MEDIUM: "#fff3cd",
    PRIORITY_LOW: "#d1e7dd",
}


@dataclass(frozen=True)
class Settings:
    """Run settings derived from ``meta.json`` and the test predictions."""

    best: str
    forecast: str
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
            forecast=data.forecast_name,
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


def text_context(data: DashboardData, settings: Settings, lang: str = "en") -> dict:
    """Placeholders for the text templates."""
    meta, labels = data.meta, bundle(lang)
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
            labels["feature_labels"][f].lower() for f in settings.cat_features
        ),
        "best": labels["model_labels"][settings.best],
        "forecast": labels["model_labels"][settings.forecast],
        "n_fit": data.card["data"]["n_fit"],
        "generated_at": meta["generated_at"].replace("T", " "),
    }


def fmt_hours(x: float, horizon: float, lang: str = "en") -> str:
    """Remaining-life hours, beyond the horizon when the level is not reached."""
    if not np.isfinite(x):
        return tr(lang, "ui.above_horizon", horizon=horizon)
    return tr(lang, "ui.hours_value", hours=x)


def priority(p_fail: float, settings: Settings) -> str:
    """Maintenance priority for a failure probability within the window."""
    if p_fail >= settings.p_high:
        return PRIORITY_HIGH
    return PRIORITY_MEDIUM if p_fail >= settings.p_medium else PRIORITY_LOW


# --- Overview -----------------------------------------------------------------


def kpis(data: DashboardData, settings: Settings, lang: str = "en") -> list[tuple[str, str, str, str]]:
    """``(title, value, subtitle, colour)`` of the overview cards."""
    pred, act = data.predictions, active(data)
    test_metrics = (
        data.tables["metrics"]
        .query("model == @settings.best and split == 'test'")
        .iloc[0]
    )
    n_fail = int(pred["event"].sum())
    finite_rul = act.loc[np.isfinite(act["rul_median"]), "rul_median"]
    model = bundle(lang)["model_labels"][settings.best]
    return [
        (
            tr(lang, "ui.kpi_drones"),
            f"{len(pred)}",
            tr(lang, "ui.kpi_drones_sub", t_test=data.meta["split"]["t_test"]),
            "primary",
        ),
        (
            tr(lang, "ui.kpi_failures"),
            f"{n_fail}",
            tr(lang, "ui.kpi_failures_sub", share=n_fail / len(pred)),
            "danger",
        ),
        (tr(lang, "ui.kpi_in_service"), f"{len(act)}", tr(lang, "ui.kpi_in_service_sub"), "success"),
        (
            tr(lang, "ui.kpi_high"),
            f"{int((act['priority'] == PRIORITY_HIGH).sum())}",
            tr(lang, "ui.kpi_high_sub", window=settings.window, p_high=settings.p_high),
            "danger",
        ),
        (
            tr(lang, "ui.kpi_expected", window=settings.window),
            f"{act['p_fail_window'].sum():.0f}",
            tr(lang, "ui.kpi_expected_sub"),
            "danger",
        ),
        (
            tr(lang, "ui.kpi_median"),
            tr(lang, "ui.hours_value", hours=finite_rul.median()),
            tr(lang, "ui.kpi_median_sub", n=len(finite_rul)),
            "info",
        ),
        (
            tr(lang, "ui.kpi_model", model=model),
            f"C = {test_metrics['c_index_ipcw']:.3f}",
            tr(lang, "ui.kpi_model_sub", ibs=test_metrics["ibs"]),
            "secondary",
        ),
    ]


def key_findings(data: DashboardData, settings: Settings, lang: str = "en") -> list[str]:
    """Headline conclusions for the overview tab."""
    labels = bundle(lang)
    importance = data.tables["shap_importance"]
    top = importance[importance["model"] == settings.best].nlargest(3, "share")
    factors = ", ".join(
        tr(lang, "phrases.findings_factor", label=labels["feature_labels"][f], share=s)
        for f, s in zip(top["feature"], top["share"])
    )
    findings = [
        tr(lang, "phrases.findings_best", name=labels["model_labels"][settings.best]),
        tr(lang, "phrases.findings_factors", items=factors),
    ]
    summary = data.tables["bootstrap_summary"]
    for row in summary[summary["significant"]].itertuples():
        findings.append(
            tr(
                lang,
                "phrases.findings_boot",
                name=labels["model_labels"][row.better],
                metric=labels["metric_labels"][row.metric],
                p=row.p_value,
            )
        )
    if not summary["significant"].any():
        findings.append(tr(lang, "phrases.findings_boot_none"))
    act = active(data)
    findings.append(
        tr(
            lang,
            "phrases.findings_queue",
            n=len(act),
            high=int((act["priority"] == PRIORITY_HIGH).sum()),
            medium=int((act["priority"] == PRIORITY_MEDIUM).sum()),
            window=settings.window,
            p_high=settings.p_high,
            p_medium=settings.p_medium,
            expected=act["p_fail_window"].sum(),
        )
    )
    return findings


def priority_sensitivity(
    active_rows: pd.DataFrame, settings: Settings, lang: str = "en"
) -> pd.DataFrame:
    """Queue size at high-priority cutoffs 30/40/50%, medium cutoff unchanged."""
    rows = []
    probability = active_rows["p_fail_window"]
    for high in (0.3, 0.4, 0.5):
        rows.append(
            {
                tr(lang, "ui.col_threshold"): f"{high:.0%}",
                tr(lang, "ui.col_high_count"): int((probability >= high).sum()),
                tr(lang, "ui.col_medium_count"): int(
                    ((probability >= settings.p_medium) & (probability < high)).sum()
                ),
                tr(lang, "ui.col_current"): (
                    tr(lang, "ui.current_mark") if np.isclose(high, settings.p_high) else ""
                ),
            }
        )
    return pd.DataFrame(rows)


# --- Methodology --------------------------------------------------------------------


def hyperparams_frame(models: dict[str, Model], lang: str = "en") -> pd.DataFrame:
    """Tuned hyper-parameters with plain-language meaning."""
    labels = bundle(lang)
    rows = []
    for key, (name, meaning) in labels["hyperparams"].items():
        row = {tr(lang, "ui.setting"): name, tr(lang, "ui.meaning"): meaning}
        present = False
        for kind, model in models.items():
            value = (
                model.booster.best_iteration
                if key == "best_iteration"
                else {**model.params, **(model.calibration or {})}.get(key)
            )
            present |= value is not None
            row[labels["model_labels"][kind]] = (
                "—"
                if value is None
                else value if isinstance(value, str) else f"{value:.4g}"
            )
        if present:
            rows.append(row)
    return pd.DataFrame(rows)


def features_frame(meta: dict, lang: str = "en") -> pd.DataFrame:
    """Feature dictionary with training ranges."""
    labels = bundle(lang)
    rows = []
    for feature in meta["features"]:
        if feature == "is_refurbished":
            kind, values = tr(lang, "ui.kind_bool"), tr(lang, "ui.bool_values")
        elif feature in meta["cat_features"]:
            kind, values = tr(lang, "ui.kind_category"), ", ".join(
                map(str, meta["categories"][feature])
            )
        else:
            low, high = meta["numeric_ranges"][feature]
            kind, values = tr(lang, "ui.kind_number"), f"{low:.3g} – {high:.3g}"
        rows.append(
            {
                tr(lang, "ui.feature"): labels["feature_labels"][feature],
                tr(lang, "ui.column"): feature,
                tr(lang, "ui.description"): labels["feature_descriptions"][feature],
                tr(lang, "ui.type"): kind,
                tr(lang, "ui.values_in_training"): values,
            }
        )
    return pd.DataFrame(rows)


# --- Segments -----------------------------------------------------------------------


def segment_levels(pred: pd.DataFrame, segment: str) -> list:
    """Levels of ``segment`` in display order."""
    levels = pred[segment].dropna().unique().tolist()
    order = SEGMENT_ORDER.get(segment)
    return [lv for lv in order if lv in levels] if order else sorted(levels, key=str)


def numeric_segment_features(data: DashboardData, settings: Settings) -> list[str]:
    """Three numeric features with the largest SHAP share of the selected model."""
    importance = data.tables["shap_importance"]
    ranked = importance[importance["model"] == settings.best].sort_values(
        "share", ascending=False
    )
    numeric = [feature for feature in ranked["feature"] if feature not in settings.cat_features]
    return numeric[:3]


def segment_options(data: DashboardData, settings: Settings, lang: str = "en") -> dict[str, str]:
    """Segment key -> localized label, including quartile cuts of the top numeric features."""
    labels = bundle(lang)
    options = dict(labels["segment_labels"])
    for feature in numeric_segment_features(data, settings):
        options[f"q:{feature}"] = tr(
            lang,
            "ui.quantile_segment",
            feature=labels["feature_labels"][feature],
        )
    return options


def prepare_segment(pred: pd.DataFrame, segment: str) -> tuple[pd.DataFrame, str, list]:
    """Frame, column and ordered levels for a segment, binning numeric ``q:`` keys."""
    if not str(segment).startswith("q:"):
        return pred, segment, segment_levels(pred, segment)
    feature = segment[2:]
    out = pred.copy()
    bins = pd.qcut(out[feature], 4, duplicates="drop")
    labels = [f"{iv.left:.3g}–{iv.right:.3g}" for iv in bins.cat.categories]
    out["_bin"] = bins.map(dict(zip(bins.cat.categories, labels, strict=True)))
    return out, "_bin", labels


def segment_stats(
    pred: pd.DataFrame,
    segment: str,
    levels: list,
    window: float,
    lang: str = "en",
    axis_name: str = "",
) -> pd.DataFrame:
    """Per-level counts, Kaplan-Meier median and mean predicted failure probability."""
    groups = pred.groupby(segment, observed=True)
    stats = pd.DataFrame(
        {
            tr(lang, "ui.col_drones"): groups["drone_id"].count(),
            tr(lang, "ui.col_failures"): groups["event"].sum(),
            tr(lang, "ui.col_failure_share"): groups["event"].mean().round(3),
            tr(lang, "ui.col_median"): groups[["event", "duration_h"]].apply(
                lambda g: insights.hours_text(insights.km_median(g), lang)
            ),
            tr(lang, "ui.col_mean_hours"): groups["duration_h"].mean().round(0),
            tr(lang, "ui.col_in_service"): groups["in_service"].sum(),
            tr(lang, "ui.col_mean_p", window=window): groups["p_fail_window"].mean().round(3),
        }
    )
    stats = stats.reindex(levels)
    source = segment if not segment.startswith("q:") else ""
    if segment in ("priority", "risk_group"):
        stats.index = [insights.level_text(lang, segment, lv) for lv in stats.index]
    return stats.rename_axis(axis_name or source).reset_index()


def segment_findings(
    pred: pd.DataFrame, segment: str, levels: list, window: float, lang: str = "en"
) -> list[str]:
    """Takeaways for a segment; the priority split is defined by the outcome itself."""
    if segment == "priority":
        labels = bundle(lang)["priority_labels"]
        return [
            tr(
                lang,
                "phrases.priority_segment_1",
                failed=labels[PRIORITY_FAILED],
                left=labels[PRIORITY_LEFT],
            ),
            tr(lang, "phrases.priority_segment_2", window=window),
        ]
    return insights.segments(pred, segment, levels, lang)


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


def metrics_frame(metrics: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Metric table with readable labels."""
    labels = bundle(lang)
    frame = metrics.copy()
    frame["model"] = frame["model"].map(labels["model_labels"])
    frame["split"] = frame["split"].map(
        {"val": tr(lang, "ui.split_val"), "test": tr(lang, "ui.split_test")}
    )
    return frame.rename(
        columns={
            "model": tr(lang, "ui.col_model"),
            "split": tr(lang, "ui.col_split"),
            **labels["metric_labels"],
        }
    ).round(4)


def bootstrap_frame(summary: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Bootstrap summary with readable labels."""
    labels = bundle(lang)
    frame = summary.copy()
    frame["metric"] = frame["metric"].map(labels["metric_labels"])
    for col in ("model_a", "model_b", "better"):
        frame[col] = frame[col].map(labels["model_labels"])
    return frame.round(4)


def gate_frame(data: DashboardData, lang: str = "en") -> pd.DataFrame:
    """Release-gate checks of every production model, one row per model and check."""
    labels = bundle(lang)
    names = labels["gate_checks"]
    rows = []
    for kind, result in data.gate["kinds"].items():
        for check in result["checks"]:
            reference = check["reference"]
            rows.append(
                {
                    tr(lang, "ui.model"): labels["model_labels"][kind],
                    tr(lang, "ui.gate_check"): names[check["name"]],
                    tr(lang, "ui.gate_value"): f"{check['value']:.4g}",
                    tr(lang, "ui.gate_reference"): (
                        "—" if reference is None else f"{reference:.4g}"
                    ),
                    tr(lang, "ui.gate_threshold"): f"{check['threshold']:.4g}",
                    tr(lang, "ui.gate_result"): tr(
                        lang, "ui.gate_pass" if check["passed"] else "ui.gate_fail"
                    ),
                }
            )
    return pd.DataFrame(rows)


def gate_findings(data: DashboardData, lang: str = "en") -> list[str]:
    """What the release gate decided and which model makes the forecasts."""
    labels, gate = bundle(lang), data.gate
    names = labels["model_labels"]
    forecast, best = data.forecast_name, data.best_name
    out = [
        tr(
            lang,
            "phrases.gate_scheme",
            n_fit=data.card["data"]["n_fit"],
            n_test=len(data.predictions),
        )
    ]
    failed = [
        labels["gate_checks"][check["name"]]
        for check in gate["kinds"][best]["checks"]
        if not check["passed"]
    ]
    if gate["release_kind"] is None:
        out.append(
            tr(lang, "phrases.gate_none", name=names[forecast], checks=", ".join(failed))
        )
    elif forecast != best:
        out.append(
            tr(
                lang,
                "phrases.gate_switched",
                best=names[best],
                checks=", ".join(failed),
                name=names[forecast],
            )
        )
    else:
        out.append(tr(lang, "phrases.gate_passed", name=names[forecast]))
    return out


# --- Risk factors -------------------------------------------------------------------


def beeswarm_findings(
    data: DashboardData, settings: Settings, kind: str, lang: str = "en"
) -> list[str]:
    """Direction of effect for the three most important features of ``kind``."""
    shap_values, importance = data.tables["shap_values"], data.tables["shap_importance"]
    labels = bundle(lang)["feature_labels"]
    top = importance[importance["model"] == kind].nlargest(3, "share")["feature"]
    return [
        f'"{labels[feature]}": '
        + insights.shap_direction(
            shap_values, kind, feature, feature in settings.cat_features, lang
        )[0]
        for feature in top
    ]


# --- Fleet --------------------------------------------------------------------------


def fleet_columns(settings: Settings, lang: str = "en") -> dict[str, str]:
    """Column -> header of the fleet table."""
    return {
        "drone_id": tr(lang, "ui.fleet_drone"),
        "model": tr(lang, "ui.fleet_model"),
        "environment": tr(lang, "ui.fleet_environment"),
        "age_h": tr(lang, "ui.fleet_age"),
        "p_fail_window": tr(lang, "ui.fleet_p", window=settings.window),
        "rul_q10": tr(lang, "ui.fleet_q10"),
        "rul_median": tr(lang, "ui.fleet_median"),
        "rmrl": tr(lang, "ui.fleet_rmrl"),
        "main_factor": tr(lang, "ui.main_factor"),
        "risk_group": tr(lang, "ui.fleet_risk"),
        "priority": tr(lang, "ui.fleet_priority"),
    }


def fleet_tooltips(settings: Settings, lang: str = "en") -> dict[str, str]:
    """Column -> header tooltip of the fleet table."""
    return {
        "age_h": tr(lang, "ui.tip_age"),
        "p_fail_window": tr(lang, "ui.tip_p", window=settings.window),
        "rul_q10": tr(lang, "ui.tip_q10"),
        "rul_median": tr(lang, "ui.tip_median"),
        "rmrl": tr(lang, "ui.tip_rmrl", horizon=settings.horizon),
        "risk_group": tr(lang, "ui.tip_risk"),
        "priority": tr(
            lang, "ui.tip_priority", p_high=settings.p_high, p_medium=settings.p_medium
        ),
    }


def main_factor_labels(data: DashboardData, lang: str = "en") -> pd.Series:
    """drone_id -> feature that raises the forecasting model's risk the most."""
    shap = data.tables["shap_production"]
    part = shap[(shap["model"] == data.forecast_name) & (shap["shap"] > 0)]
    if part.empty:
        return pd.Series(dtype=str)
    top = part.loc[part.groupby("drone_id")["shap"].idxmax()]
    labels = bundle(lang)["feature_labels"]
    return pd.Series(
        [
            tr(lang, "phrases.main_factor_value", label=labels[feature], value=value)
            for feature, value in zip(top["feature"], top["display"])
        ],
        index=top["drone_id"].to_numpy(),
    )


def fleet_frame(data: DashboardData, settings: Settings, lang: str = "en") -> pd.DataFrame:
    """Drones in service, riskiest first; ``inf`` remaining life becomes NaN."""
    columns = [col for col in fleet_columns(settings, lang) if col != "main_factor"]
    frame = active(data).sort_values("p_fail_window", ascending=False)
    frame = frame[columns].copy()
    labels = main_factor_labels(data, lang)
    frame["main_factor"] = frame["drone_id"].map(labels).fillna(tr(lang, "ui.no_factor"))
    ordered = list(fleet_columns(settings, lang))
    return frame[ordered].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def disagreeing_drones(
    data: DashboardData, settings: Settings, n: int = 5
) -> pd.DataFrame:
    """Test drones with the largest |S_cox - S_aft| at the maintenance window."""
    pred = data.predictions
    at = np.array([float(settings.window)])
    gap = np.abs(
        data.models["cox"].survival(pred, at)[:, 0]
        - data.models["aft"].survival(pred, at)[:, 0]
    )
    return pred.iloc[np.argsort(-gap)[:n]]


# --- Drone card ---------------------------------------------------------------------


def drone_status(row: pd.Series, lang: str = "en") -> str:
    """One-line status of a drone."""
    if row["event"] == 1:
        mode = bundle(lang)["failure_mode_labels"].get(
            row["failure_mode"], row["failure_mode"]
        )
        return tr(lang, "phrases.status_failed", mode=mode, hours=row["duration_h"])
    if not row["in_service"]:
        return tr(lang, "phrases.status_left", hours=row["duration_h"])
    return tr(
        lang,
        "phrases.status_active",
        hours=row["duration_h"],
        priority=bundle(lang)["priority_labels"].get(row["priority"], row["priority"]),
    )


def drone_options(pred: pd.DataFrame, lang: str = "en") -> dict[str, str]:
    """drone_id -> label; drones in service first, riskiest first."""
    ordered = pred.sort_values(["event", "p_fail_window"], ascending=[True, False])
    return {
        row["drone_id"]: f"{row['drone_id']} · {row['model']} · {drone_status(row, lang)}"
        for _, row in ordered.iterrows()
    }


def drone_rows(pred: pd.DataFrame, drone_id: str) -> pd.DataFrame:
    """Single-row frame of a drone."""
    return pred[pred["drone_id"] == drone_id]


def drone_metrics(
    row: pd.Series, settings: Settings, lang: str = "en"
) -> list[tuple[str, str, str]]:
    """``(name, value, hint)`` remaining-life summary of a drone in service."""
    return [
        (
            tr(lang, "ui.fleet_p", window=settings.window),
            f"{row['p_fail_window']:.1%}",
            tr(lang, "ui.p_fail_hint"),
        ),
        (
            tr(lang, "ui.fleet_q10"),
            fmt_hours(row["rul_q10"], settings.horizon, lang),
            tr(lang, "ui.life_10_hint"),
        ),
        (
            tr(lang, "ui.fleet_median"),
            fmt_hours(row["rul_median"], settings.horizon, lang),
            tr(lang, "ui.median_life_hint"),
        ),
        (
            tr(lang, "ui.fleet_rmrl"),
            tr(lang, "ui.expected_life_value", hours=row["rmrl"]),
            tr(lang, "ui.expected_life_hint", horizon=settings.horizon),
        ),
    ]


def drone_features(row: pd.Series, features: list[str], lang: str = "en") -> pd.DataFrame:
    """Feature values of a drone with readable names."""
    labels = bundle(lang)["feature_labels"]
    return pd.DataFrame(
        {
            tr(lang, "ui.feature"): [labels[f] for f in features],
            tr(lang, "ui.value"): [str(row[f]) for f in features],
        }
    )


def drone_contrib(data: DashboardData, drone_id: str) -> pd.DataFrame:
    """SHAP contributions of the forecasting (production) model for one drone."""
    shap_values = data.tables["shap_production"]
    return shap_values[
        (shap_values["drone_id"] == drone_id)
        & (shap_values["model"] == data.forecast_name)
    ]


def drone_hazard_grid(settings: Settings, age: float) -> np.ndarray:
    """Hazard grid of a drone card: the fleet range, extended to the drone's next window.

    Beyond the fleet range the smoothed Cox baseline rests on few training failures
    and oscillates, so it is shown only as far as the drone's forecast needs.
    """
    end = max(settings.hazard_grid[-1], age + settings.window)
    return np.linspace(settings.hazard_bandwidth, end, 300)


def drone_findings(
    data: DashboardData, settings: Settings, drone_id: str, lang: str = "en"
) -> list[str]:
    """Narrative explanation of one drone."""
    rows = drone_rows(data.predictions, drone_id)
    curve = data.forecast_model.survival(rows, settings.surv_grid)[0]
    below = curve <= 0.5
    predicted_median = settings.surv_grid[below.argmax()] if below.any() else np.inf
    findings = insights.drone(
        rows.iloc[0],
        drone_contrib(data, drone_id),
        data.predictions,
        settings.forecast,
        settings.window,
        predicted_median,
        lang,
    )
    row = rows.iloc[0]
    age = float(row["duration_h"])
    if row["in_service"] and settings.hazard_grid[0] <= age <= settings.hazard_grid[-1]:
        at_age = np.array([age])
        rate = data.forecast_model.hazard(rows, at_age, settings.hazard_bandwidth)[0, 0]
        pred = data.predictions
        fleet_rate = observed_hazard(
            pred["duration_h"], pred["event"], at_age, settings.hazard_bandwidth
        )[0][0]
        findings.append(
            insights.drone_hazard(age, rate, fleet_rate, settings.forecast, lang)
        )
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


def whatif_order(features: list[str]) -> list[str]:
    """Editable features: operational levers first, then characteristics fixed at entry."""
    levers = [feature for feature in LEVERS if feature in features]
    return [*levers, *[feature for feature in features if feature not in levers]]


def whatif_compare(
    model: Model,
    rows: pd.DataFrame,
    scenario: pd.DataFrame,
    settings: Settings,
    lang: str = "en",
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Remaining-life summaries of current and scenario conditions and a comparison table."""
    base = remaining_life(model, rows, settings.horizon, settings.window).iloc[0]
    new = remaining_life(model, scenario, settings.horizon, settings.window).iloc[0]
    labels = bundle(lang)["priority_labels"]
    table = pd.DataFrame(
        {
            "": [
                tr(lang, "ui.fleet_p", window=settings.window),
                tr(lang, "ui.fleet_priority"),
                tr(lang, "ui.fleet_median"),
                tr(lang, "ui.fleet_rmrl"),
            ],
            tr(lang, "ui.now"): [
                f"{base['p_fail_window']:.1%}",
                labels[priority(base["p_fail_window"], settings)],
                fmt_hours(base["rul_median"], settings.horizon, lang),
                tr(lang, "ui.expected_life_value", hours=base["rmrl"]),
            ],
            tr(lang, "ui.scenario_col"): [
                f"{new['p_fail_window']:.1%}",
                labels[priority(new["p_fail_window"], settings)],
                fmt_hours(new["rul_median"], settings.horizon, lang),
                tr(lang, "ui.expected_life_value", hours=new["rmrl"]),
            ],
        }
    )
    return base, new, table


# --- Scoring new drones -------------------------------------------------------------

ID_COLUMN, AGE_COLUMN = "drone_id", "age_h"
MAX_UPLOAD_ROWS = 10_000
_BOOL_VALUES = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def read_csv_bytes(content: bytes, lang: str = "en") -> pd.DataFrame:
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
        raise ValueError(tr(lang, "phrases.empty_file")) from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(tr(lang, "phrases.bad_csv", error=exc)) from exc


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
    raw: pd.DataFrame, data: DashboardData, settings: Settings, lang: str = "en"
) -> tuple[pd.DataFrame, list[str]]:
    """Validate an uploaded table of drones and forecast them with the production model.

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
        raise ValueError(tr(lang, "phrases.no_rows"))
    if len(raw) > MAX_UPLOAD_ROWS:
        raise ValueError(tr(lang, "phrases.too_many", n=len(raw), limit=MAX_UPLOAD_ROWS))
    missing = [f for f in settings.features if f not in raw.columns]
    if missing:
        raise ValueError(tr(lang, "phrases.missing_columns", columns=", ".join(missing)))

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
            errors.append(tr(lang, "phrases.non_numeric", column=col, n=int(bad.sum())))
        frame[col] = values
    for col in (c for c in settings.cat_features if c not in categories):
        values = _to_bool(frame[col])
        bad = values.isna()
        if bad.any():
            errors.append(tr(lang, "phrases.expected_bool", column=col, n=int(bad.sum())))
        else:
            frame[col] = values.astype(bool)
    if (frame[AGE_COLUMN] < 0).any():
        errors.append(
            tr(lang, "phrases.negative_age", column=AGE_COLUMN, n=int((frame[AGE_COLUMN] < 0).sum()))
        )
    if frame[AGE_COLUMN].isna().any():
        errors.append(
            tr(lang, "phrases.empty_age", column=AGE_COLUMN, n=int(frame[AGE_COLUMN].isna().sum()))
        )
    if errors:
        raise ValueError(tr(lang, "phrases.invalid_values", details="; ".join(errors)))

    warnings = []
    for col, levels in categories.items():
        values = frame[col].astype("object")
        unknown = values.notna() & ~values.isin(levels)
        if unknown.any():
            seen = ", ".join(map(str, sorted(values[unknown].astype(str).unique())))
            warnings.append(
                tr(
                    lang,
                    "phrases.unknown_category",
                    label=bundle(lang)["feature_labels"][col],
                    n=int(unknown.sum()),
                    seen=seen,
                )
            )
    n_missing = int(frame[settings.features].isna().any(axis=1).sum())
    if n_missing:
        warnings.append(tr(lang, "phrases.empty_features", n=n_missing))
    out_of_range = pd.Series(False, index=frame.index)
    for col, (low, high) in data.meta["numeric_ranges"].items():
        outside = (frame[col] < low) | (frame[col] > high)
        if outside.any():
            out_of_range |= outside
            warnings.append(
                tr(
                    lang,
                    "phrases.outside_training",
                    label=bundle(lang)["feature_labels"][col],
                    n=int(outside.sum()),
                    low=low,
                    high=high,
                )
            )
    max_age = active(data)["duration_h"].max()
    too_old = frame[AGE_COLUMN] > max_age
    if too_old.any():
        warnings.append(
            tr(lang, "phrases.too_old", n=int(too_old.sum()), column=AGE_COLUMN, max_age=max_age)
        )

    model = data.forecast_model
    rows = frame.assign(duration_h=frame[AGE_COLUMN])
    life = remaining_life(model, rows, settings.horizon, settings.window)
    scored = frame.assign(
        risk_group=_risk_group(
            model.risk(rows), data.predictions[f"forecast_risk_{settings.forecast}"]
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


def score_columns(settings: Settings, lang: str = "en") -> dict[str, str]:
    """Column -> header of the scoring results table."""
    columns = {k: v for k, v in fleet_columns(settings, lang).items() if k != "main_factor"}
    return {**columns, "out_of_range": tr(lang, "ui.fleet_outside")}


def score_summary(scored: pd.DataFrame, settings: Settings, lang: str = "en") -> list[str]:
    """Headline numbers of a scored upload."""
    counts = scored["priority"].value_counts()
    return [
        tr(
            lang,
            "phrases.score_counts",
            n=len(scored),
            high=int(counts.get(PRIORITY_HIGH, 0)),
            medium=int(counts.get(PRIORITY_MEDIUM, 0)),
            low=int(counts.get(PRIORITY_LOW, 0)),
        ),
        tr(
            lang,
            "phrases.score_expected",
            window=settings.window,
            mean=scored["p_fail_window"].mean(),
            total=scored["p_fail_window"].sum(),
        ),
        tr(lang, "phrases.score_outside", n=int(scored["out_of_range"].sum())),
    ]

"""Plotly figure builders for the dashboard (pure functions of the loaded data)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sksurv.nonparametric import kaplan_meier_estimator

from dashboard.labels import (
    FAILURE_MODE_LABELS,
    FEATURE_LABELS,
    HAZARD_SCALE,
    HAZARD_UNIT,
    METRIC_LABELS,
    MODEL_LABELS,
)
from utils import Model

MODEL_COLORS = {"cox": "#1f77b4", "aft": "#ff7f0e"}
KM_COLOR = "#222222"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
FAILURE_COLORS = {
    "motor": "#8c564b",
    "battery": "#d62728",
    "electronics": "#9467bd",
    "crash": "#ff7f0e",
}
HOURS = "Flight hours"


def _style(
    fig: go.Figure, title: str, x_title: str = "", y_title: str = "", height: int = 400
) -> go.Figure:
    """Apply the shared layout."""
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "top", "y": -0.28, "x": 0},
        hovermode="closest",
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` to an ``rgba()`` string."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _step_with_band(
    t: np.ndarray,
    s: np.ndarray,
    ci_low: np.ndarray | None,
    ci_high: np.ndarray | None,
    name: str,
    color: str,
) -> list[go.Scatter]:
    """Kaplan-Meier step curve starting at S(0) = 1, with an optional CI band."""
    t, s = np.r_[0.0, t], np.r_[1.0, s]
    traces = []
    if ci_low is not None and ci_high is not None:
        lo, hi = np.r_[1.0, ci_low], np.r_[1.0, ci_high]
        traces += [
            go.Scatter(
                x=t,
                y=hi,
                mode="lines",
                line={"width": 0, "shape": "hv"},
                legendgroup=name,
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=t,
                y=lo,
                mode="lines",
                line={"width": 0, "shape": "hv"},
                fill="tonexty",
                fillcolor=_rgba(color, 0.15),
                legendgroup=name,
                showlegend=False,
                hoverinfo="skip",
            ),
        ]
    traces.append(
        go.Scatter(
            x=t,
            y=s,
            mode="lines",
            name=name,
            legendgroup=name,
            line={"color": color, "shape": "hv"},
            hovertemplate="%{x:.0f} h: S = %{y:.3f}<extra>" + name + "</extra>",
        )
    )
    return traces


def _km(data: pd.DataFrame, name: str, color: str, ci: bool = True) -> list[go.Scatter]:
    """Kaplan-Meier traces for ``data`` (``duration_h``, ``event``)."""
    event = data["event"].to_numpy().astype(bool)
    time = data["duration_h"].to_numpy()
    if ci and event.any():
        t, s, (lo, hi) = kaplan_meier_estimator(event, time, conf_type="log-log")
        return _step_with_band(t, s, lo, hi, name, color)
    t, s = kaplan_meier_estimator(event, time)
    return _step_with_band(t, s, None, None, name, color)


# --- Overview ---------------------------------------------------------------


def km_overall(pred: pd.DataFrame) -> go.Figure:
    """Fleet-level Kaplan-Meier survival on test."""
    fig = go.Figure(_km(pred, "Kaplan-Meier (all test drones)", KM_COLOR))
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "Share of drones without failure by flight hours", HOURS, "S(t)")


def hazard(table: pd.DataFrame, with_models: bool) -> go.Figure:
    """Smoothed observed hazard with its 95% band, optionally with model predictions."""
    obs = table[table["source"] == "observed"]
    t = obs["t"].to_numpy()
    fig = go.Figure(
        [
            go.Scatter(
                x=t,
                y=HAZARD_SCALE * obs["ci_high"],
                mode="lines",
                line={"width": 0},
                legendgroup="observed",
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=t,
                y=HAZARD_SCALE * obs["ci_low"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(KM_COLOR, 0.15),
                legendgroup="observed",
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=t,
                y=HAZARD_SCALE * obs["h"],
                mode="lines",
                name="Observed (smoothed Nelson-Aalen)",
                legendgroup="observed",
                line={"color": KM_COLOR},
                hovertemplate="%{x:.0f} h: %{y:.3f}<extra>observed</extra>",
            ),
        ]
    )
    if with_models:
        for kind, color in MODEL_COLORS.items():
            part = table[table["source"] == kind]
            fig.add_trace(
                go.Scatter(
                    x=part["t"],
                    y=HAZARD_SCALE * part["h"],
                    mode="lines",
                    name=f"{MODEL_LABELS[kind]}: prediction",
                    line={"color": color},
                    hovertemplate="%{x:.0f} h: %{y:.3f}<extra>"
                    + MODEL_LABELS[kind]
                    + "</extra>",
                )
            )
    fig.update_yaxes(rangemode="tozero")
    title = (
        "Failure rate: predicted vs observed"
        if with_models
        else "Failure rate among drones still flying"
    )
    return _style(fig, title, HOURS, HAZARD_UNIT)


def failure_modes(pred: pd.DataFrame) -> go.Figure:
    """Counts of observed failures by mechanism."""
    counts = pred.loc[pred["event"] == 1, "failure_mode"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.to_numpy(),
            y=[FAILURE_MODE_LABELS.get(m, m) for m in counts.index],
            orientation="h",
            marker_color=[FAILURE_COLORS.get(m, "#999") for m in counts.index],
            text=[f"{c} ({c / counts.sum():.0%})" for c in counts.to_numpy()],
            textposition="auto",
            hovertemplate="%{y}: %{x} failures<extra></extra>",
        )
    )
    return _style(fig, "Failure causes", "Number of failures", "", height=320)


# --- Segments ---------------------------------------------------------------


def _segment_levels(pred: pd.DataFrame, segment: str, order: list | None) -> list:
    levels = pred[segment].dropna().unique().tolist()
    if order:
        return [lv for lv in order if lv in levels]
    return sorted(levels, key=str)


def segment_km(
    pred: pd.DataFrame, segment: str, order: list | None = None
) -> go.Figure:
    """Kaplan-Meier curves per level of ``segment``."""
    fig = go.Figure()
    for k, level in enumerate(_segment_levels(pred, segment, order)):
        part = pred[pred[segment] == level]
        name = f"{level} (n={len(part)}, failures {int(part['event'].sum())})"
        for trace in _km(part, name, PALETTE[k % len(PALETTE)], ci=False):
            fig.add_trace(trace)
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "Survival by segment (Kaplan-Meier)", HOURS, "S(t)", 430)


def segment_failure_structure(
    pred: pd.DataFrame, segment: str, order: list | None = None
) -> go.Figure:
    """Share of drones in each segment that failed, split by failure mechanism."""
    levels = _segment_levels(pred, segment, order)
    size = pred.groupby(segment)["drone_id"].count()
    fig = go.Figure()
    for mode, color in FAILURE_COLORS.items():
        failed = pred[pred["failure_mode"] == mode].groupby(segment)["drone_id"].count()
        share = (failed.reindex(levels).fillna(0) / size.reindex(levels)).to_numpy()
        fig.add_trace(
            go.Bar(
                x=[str(lv) for lv in levels],
                y=share,
                name=FAILURE_MODE_LABELS[mode],
                marker_color=color,
                hovertemplate="%{x}: %{y:.1%}<extra>"
                + FAILURE_MODE_LABELS[mode]
                + "</extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%")
    return _style(
        fig, "Share of failed drones and causes", "", "Share of segment drones", 430
    )


# --- Model quality ------------------------------------------------------------


def learning_curves(models: dict[str, Model]) -> go.Figure:
    """Train/valid loss per boosting iteration with the best iteration marked."""
    fig = make_subplots(
        rows=1, cols=len(models), subplot_titles=[MODEL_LABELS[k] for k in models]
    )
    for col, (kind, model) in enumerate(models.items(), start=1):
        for k, (split, metrics) in enumerate(model.history.items()):
            for metric, values in metrics.items():
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode="lines",
                        name=f"{MODEL_LABELS[kind]} {split}",
                        line={"color": PALETTE[k + 2 * (col - 1)]},
                        hovertemplate="iteration %{x}: %{y:.4f}<extra>"
                        + metric
                        + "</extra>",
                    ),
                    row=1,
                    col=col,
                )
        fig.add_vline(
            x=model.booster.best_iteration,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=col,
        )
    return _style(
        fig, "Learning curves (dashed: best iteration)", "Boosting iteration", "Loss"
    )


def survival_mean(table: pd.DataFrame) -> go.Figure:
    """Mean predicted survival per model vs Kaplan-Meier on test."""
    km = table[table["source"] == "km"]
    fig = go.Figure(
        _step_with_band(
            km["t"].to_numpy(),
            km["s"].to_numpy(),
            km["ci_low"].to_numpy(),
            km["ci_high"].to_numpy(),
            "Kaplan-Meier (observed)",
            KM_COLOR,
        )
    )
    for kind, color in MODEL_COLORS.items():
        part = table[table["source"] == kind]
        fig.add_trace(
            go.Scatter(
                x=part["t"],
                y=part["s"],
                mode="lines",
                name=f"{MODEL_LABELS[kind]}: mean prediction",
                line={"color": color},
            )
        )
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "Predicted vs observed on test", HOURS, "S(t)")


def individual_curves(
    models: dict[str, Model], rows: pd.DataFrame, grid: np.ndarray
) -> go.Figure:
    """Survival curves of selected drones (solid: Cox, dashed: AFT)."""
    fig = go.Figure()
    curves = {kind: model.survival(rows, grid) for kind, model in models.items()}
    for k, drone_id in enumerate(rows["drone_id"]):
        for kind, curve in curves.items():
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=curve[k],
                    mode="lines",
                    name=f"{drone_id} {MODEL_LABELS[kind]}",
                    legendgroup=drone_id,
                    line={
                        "color": PALETTE[k % len(PALETTE)],
                        "dash": "solid" if kind == "cox" else "dash",
                    },
                )
            )
    fig.update_yaxes(range=[0, 1.02])
    return _style(
        fig, "Individual curves (solid: Cox, dashed: AFT)", HOURS, "S(t)"
    )


def brier_curve(brier_auc: pd.DataFrame, metrics: pd.DataFrame) -> go.Figure:
    """Time-dependent Brier score per model on test."""
    fig = go.Figure()
    for kind, part in brier_auc.groupby("model", sort=False):
        ibs = metrics.query("model == @kind and split == 'test'")["ibs"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=part["t"],
                y=part["brier"],
                mode="lines",
                name=f"{MODEL_LABELS[kind]} (IBS={ibs:.4f})",
                line={"color": MODEL_COLORS[kind]},
            )
        )
    return _style(fig, "Brier score over time (lower is better)", HOURS, "Brier score")


def auc_curve(brier_auc: pd.DataFrame, metrics: pd.DataFrame) -> go.Figure:
    """Cumulative/dynamic AUC per model on test."""
    fig = go.Figure()
    for kind, part in brier_auc.groupby("model", sort=False):
        mean_auc = metrics.query("model == @kind and split == 'test'")["mean_auc"].iloc[
            0
        ]
        fig.add_trace(
            go.Scatter(
                x=part["t"],
                y=part["auc"],
                mode="lines",
                name=f"{MODEL_LABELS[kind]} (mean AUC={mean_auc:.4f})",
                line={"color": MODEL_COLORS[kind]},
            )
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="random")
    return _style(
        fig, "AUC(t): separating drones failed by time t (higher is better)", HOURS, "AUC(t)"
    )


def calibration(table: pd.DataFrame, horizon: float) -> go.Figure:
    """Predicted vs observed survival at ``horizon`` by prediction decile."""
    part = table[np.isclose(table["horizon"], horizon)]
    fig = go.Figure(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "grey", "dash": "dot"},
        )
    )
    for kind, rows in part.groupby("model", sort=False):
        fig.add_trace(
            go.Scatter(
                x=rows["predicted"],
                y=rows["observed"],
                mode="lines+markers",
                name=MODEL_LABELS[kind],
                line={"color": MODEL_COLORS[kind]},
                customdata=rows["n"],
                hovertemplate="predicted %{x:.2f}, observed %{y:.2f}, n=%{customdata}<extra></extra>",
            )
        )
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", constrain="domain")
    return _style(
        fig,
        f"Calibration at {horizon:.0f} h",
        "Predicted S(t)",
        "Observed S(t) (Kaplan-Meier)",
        430,
    )


def bootstrap(samples: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """Bootstrap distributions of metric differences between the first model pair."""
    a, b = summary.iloc[0][["model_a", "model_b"]]
    metrics = summary["metric"].tolist()
    titles = [
        f"{METRIC_LABELS[m]}<br>p = {summary.set_index('metric').at[m, 'p_value']:.3f}"
        for m in metrics
    ]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=titles)
    wide = samples.pivot_table(
        index="iteration", columns=["model", "metric"], values="value"
    )
    for col, metric in enumerate(metrics, start=1):
        row = summary.set_index("metric").loc[metric]
        fig.add_trace(
            go.Histogram(
                x=wide[(a, metric)] - wide[(b, metric)],
                nbinsx=40,
                marker_color=MODEL_COLORS["cox"],
                opacity=0.75,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=0, line_dash="dot", line_color="black", row=1, col=col)
        for x in (row["ci_low"], row["ci_high"]):
            fig.add_vline(x=x, line_dash="dash", line_color="red", row=1, col=col)
    fig.update_annotations(font_size=12)
    _style(
        fig,
        f"Bootstrap: difference {MODEL_LABELS[a]} − {MODEL_LABELS[b]} "
        "(red: 95% CI, black: no difference)",
        height=380,
    )
    fig.update_layout(margin={"t": 100})
    return fig


def backtest(table: pd.DataFrame, landmark: float) -> go.Figure:
    """Conditional survival after the landmark: predicted vs Kaplan-Meier."""
    km = table[table["source"] == "km"]
    fig = go.Figure(
        _step_with_band(
            km["u"].to_numpy(),
            km["s"].to_numpy(),
            km["ci_low"].to_numpy(),
            km["ci_high"].to_numpy(),
            "Kaplan-Meier (observed)",
            KM_COLOR,
        )
    )
    u_max = table.loc[table["source"] != "km", "u"].max()
    for kind, color in MODEL_COLORS.items():
        part = table[table["source"] == kind]
        fig.add_trace(
            go.Scatter(
                x=part["u"],
                y=part["s"],
                mode="lines",
                name=f"{MODEL_LABELS[kind]}: mean prediction",
                line={"color": color},
            )
        )
    fig.update_xaxes(range=[0, u_max])
    fig.update_yaxes(range=[0, 1.02])
    return _style(
        fig,
        f"Backtest: remaining life after {landmark:.0f} h",
        "Additional flight hours",
        "S(L + u) / S(L)",
    )


# --- Risk factors -------------------------------------------------------------


def shap_importance(importance: pd.DataFrame, order_by: str) -> go.Figure:
    """Share of total mean |SHAP| per feature and model."""
    order = (
        importance[importance["model"] == order_by]
        .sort_values("share")["feature"]
        .tolist()
    )
    fig = go.Figure()
    for kind, color in MODEL_COLORS.items():
        part = importance[importance["model"] == kind].set_index("feature").loc[order]
        fig.add_trace(
            go.Bar(
                x=part["share"],
                y=[FEATURE_LABELS[f] for f in order],
                orientation="h",
                name=MODEL_LABELS[kind],
                marker_color=color,
                hovertemplate="%{y}: %{x:.1%}<extra>" + MODEL_LABELS[kind] + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%")
    return _style(fig, "Factor importance (share of total |SHAP|)", "", "", 560)


def shap_beeswarm(
    shap_values: pd.DataFrame, importance: pd.DataFrame, kind: str
) -> go.Figure:
    """Beeswarm-style scatter of SHAP values, coloured by within-feature value rank."""
    part = shap_values[shap_values["model"] == kind].copy()
    order = (
        importance[importance["model"] == kind]
        .sort_values("mean_abs_shap")["feature"]
        .tolist()
    )
    position = {f: i for i, f in enumerate(order)}
    rng = np.random.default_rng(0)
    part["y"] = part["feature"].map(position) + rng.uniform(-0.3, 0.3, len(part))
    part["rank"] = part.groupby("feature")["value"].rank(pct=True).fillna(0.5)
    fig = go.Figure(
        go.Scattergl(
            x=part["shap"],
            y=part["y"],
            mode="markers",
            marker={
                "size": 4,
                "color": part["rank"],
                "colorscale": "Bluered",
                "opacity": 0.6,
                "colorbar": {
                    "title": "Feature<br>value",
                    "tickvals": [0, 1],
                    "ticktext": ["low", "high"],
                },
            },
            customdata=np.c_[part["drone_id"], part["display"]],
            hovertemplate="%{customdata[0]}: value %{customdata[1]}, SHAP %{x:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_color="grey")
    fig.update_yaxes(
        tickvals=list(range(len(order))), ticktext=[FEATURE_LABELS[f] for f in order]
    )
    return _style(
        fig,
        f"{MODEL_LABELS[kind]}: factor impact on risk (right: earlier failure)",
        "SHAP (contribution to risk score)",
        "",
        560,
    )


def shap_dependence(
    shap_values: pd.DataFrame, kind: str, feature: str, categorical: bool
) -> go.Figure:
    """SHAP value of ``feature`` against its value."""
    part = shap_values[
        (shap_values["model"] == kind) & (shap_values["feature"] == feature)
    ]
    if categorical:
        fig = go.Figure(
            go.Box(
                x=part["display"],
                y=part["shap"],
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"size": 3, "color": MODEL_COLORS[kind]},
                line={"color": MODEL_COLORS[kind]},
                showlegend=False,
            )
        )
        fig.update_xaxes(categoryorder="category ascending")
    else:
        fig = go.Figure(
            go.Scattergl(
                x=part["value"],
                y=part["shap"],
                mode="markers",
                marker={"size": 4, "color": MODEL_COLORS[kind], "opacity": 0.6},
                customdata=part["drone_id"],
                hovertemplate="%{customdata}: %{x}, SHAP %{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line_color="grey")
    return _style(
        fig,
        f"{MODEL_LABELS[kind]}: risk vs \"{FEATURE_LABELS[feature]}\"",
        FEATURE_LABELS[feature],
        "SHAP",
        430,
    )


# --- Fleet and drone card -----------------------------------------------------


def rul_histogram(active: pd.DataFrame, horizon: float) -> go.Figure:
    """Distribution of median remaining useful life of drones in service."""
    finite = np.isfinite(active["rul_median"])
    fig = go.Figure(
        go.Histogram(
            x=active.loc[finite, "rul_median"],
            nbinsx=40,
            marker_color=MODEL_COLORS["cox"],
        )
    )
    return _style(
        fig,
        f"Median remaining life ({int((~finite).sum())} drones > {horizon:.0f} h)",
        HOURS,
        "Drones",
    )


def conditional_curves(
    model: Model,
    rows: pd.DataFrame,
    ages: np.ndarray,
    grid: np.ndarray,
    names: list[str],
    title: str,
    dashes: list[str] | None = None,
) -> go.Figure:
    """Conditional survival ``S(a + u) / S(a)`` for each row."""
    curves = model.conditional_survival(rows, ages, grid)
    fig = go.Figure()
    for k, (name, curve) in enumerate(zip(names, curves)):
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=curve,
                mode="lines",
                name=name,
                line={
                    "color": PALETTE[k % len(PALETTE)],
                    "dash": dashes[k] if dashes else "solid",
                },
            )
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="median")
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, title, "Additional flight hours", "S(a + u) / S(a)")


def drone_survival(
    models: dict[str, Model],
    row: pd.DataFrame,
    grid: np.ndarray,
    age: float,
    failed: bool,
) -> go.Figure:
    """Unconditional survival of one drone under both models with its status marked."""
    fig = go.Figure()
    for kind, model in models.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=model.survival(row, grid)[0],
                mode="lines",
                name=MODEL_LABELS[kind],
                line={"color": MODEL_COLORS[kind]},
            )
        )
    fig.add_vline(
        x=age,
        line_dash="dash",
        line_color="red" if failed else "green",
        annotation_text=f"failed at {age:.0f} h" if failed else f"flown {age:.0f} h",
    )
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, "Probability of operating without failure", HOURS, "S(t)")


def drone_hazard(
    models: dict[str, Model],
    row: pd.DataFrame,
    grid: np.ndarray,
    bandwidth: float,
    age: float,
    failed: bool,
) -> go.Figure:
    """Hazard of one drone under both models with its status marked."""
    fig = go.Figure()
    for kind, model in models.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=HAZARD_SCALE * model.hazard(row, grid, bandwidth)[0],
                mode="lines",
                name=MODEL_LABELS[kind],
                line={"color": MODEL_COLORS[kind]},
                hovertemplate="%{x:.0f} h: %{y:.3f}<extra>"
                + MODEL_LABELS[kind]
                + "</extra>",
            )
        )
    fig.add_vline(
        x=age,
        line_dash="dash",
        line_color="red" if failed else "green",
        annotation_text=f"failed at {age:.0f} h" if failed else f"flown {age:.0f} h",
    )
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, "Failure rate of this drone by flight hours", HOURS, HAZARD_UNIT)


def shap_waterfall(
    contrib: pd.DataFrame, base: float, kind: str, top: int = 12
) -> go.Figure:
    """Largest per-feature contributions to one drone's risk score."""
    rows = contrib.reindex(contrib["shap"].abs().sort_values().index).tail(top)
    fig = go.Figure(
        go.Bar(
            x=rows["shap"],
            y=[
                f"{FEATURE_LABELS[f]} = {v}"
                for f, v in zip(rows["feature"], rows["display"])
            ],
            orientation="h",
            marker_color=np.where(rows["shap"] > 0, "#d62728", "#2ca02c"),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    total = base + contrib["shap"].sum()
    return _style(
        fig,
        f"{MODEL_LABELS[kind]}: why this risk (base {base:.2f} → {total:.2f}; "
        "red raises risk)",
        "Contribution to risk score (SHAP)",
        "",
        460,
    )

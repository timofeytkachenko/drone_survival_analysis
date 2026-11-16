"""Plotly figure builders for the dashboard (pure functions of the loaded data)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sksurv.nonparametric import kaplan_meier_estimator

from dashboard.insights import level_text
from dashboard.labels import HAZARD_SCALE
from dashboard.locale import bundle, tr
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


def _hours(lang: str) -> str:
    return tr(lang, "ui.hours")


def _model(lang: str, kind: str) -> str:
    return bundle(lang)["model_labels"][kind]


def _feature(lang: str, feature: str) -> str:
    return bundle(lang)["feature_labels"][feature]


def _mode(lang: str, mode: str) -> str:
    return bundle(lang)["failure_mode_labels"].get(mode, mode)


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


def km_overall(pred: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Fleet-level Kaplan-Meier survival on test."""
    fig = go.Figure(_km(pred, tr(lang, "ui.km_all"), KM_COLOR))
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, tr(lang, "ui.km_overall"), _hours(lang), "S(t)")


def hazard(
    table: pd.DataFrame, kinds: list[str] | None = None, lang: str = "en"
) -> go.Figure:
    """Smoothed observed hazard with its 95% band and, optionally, model rates."""
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
                name=tr(lang, "ui.km_observed"),
                legendgroup="observed",
                line={"color": KM_COLOR},
                hovertemplate="%{x:.0f} h: %{y:.3f}<extra>" + tr(lang, "ui.observed") + "</extra>",
            ),
        ]
    )
    for kind in kinds or []:
        part = table[table["source"] == kind]
        name = _model(lang, kind)
        fig.add_trace(
            go.Scatter(
                x=part["t"],
                y=HAZARD_SCALE * part["h"],
                mode="lines",
                name=tr(lang, "ui.prediction", model=name),
                line={"color": MODEL_COLORS[kind]},
                hovertemplate="%{x:.0f} h: %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_yaxes(rangemode="tozero")
    title = tr(lang, "ui.hazard_models" if kinds else "ui.hazard_observed")
    return _style(fig, title, _hours(lang), bundle(lang)["hazard_unit"])


def failure_modes(pred: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Counts of observed failures by mechanism."""
    counts = pred.loc[pred["event"] == 1, "failure_mode"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.to_numpy(),
            y=[_mode(lang, m) for m in counts.index],
            orientation="h",
            marker_color=[FAILURE_COLORS.get(m, "#999") for m in counts.index],
            text=[f"{c} ({c / counts.sum():.0%})" for c in counts.to_numpy()],
            textposition="auto",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    return _style(fig, tr(lang, "ui.failure_causes"), tr(lang, "ui.number_of_failures"), "", height=320)


def failure_mode_timing(pred: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Cumulative count of observed failures of each cause by flight hour."""
    failed = pred[pred["event"] == 1]
    times = np.sort(failed["duration_h"].unique())
    fig = go.Figure()
    for mode, color in FAILURE_COLORS.items():
        event_times = np.sort(failed.loc[failed["failure_mode"] == mode, "duration_h"].to_numpy())
        if len(event_times) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.searchsorted(event_times, times, side="right"),
                mode="lines",
                line={"color": color, "shape": "hv"},
                name=_mode(lang, mode),
                hovertemplate="%{x:.0f} h: %{y}<extra>" + _mode(lang, mode) + "</extra>",
            )
        )
    return _style(
        fig,
        tr(lang, "ui.failure_timing"),
        _hours(lang),
        tr(lang, "ui.cumulative_failures"),
        320,
    )


# --- Segments ---------------------------------------------------------------


def _segment_levels(pred: pd.DataFrame, segment: str, order: list | None) -> list:
    levels = pred[segment].dropna().unique().tolist()
    if order:
        return [lv for lv in order if lv in levels]
    return sorted(levels, key=str)


def segment_km(
    pred: pd.DataFrame,
    segment: str,
    order: list | None = None,
    lang: str = "en",
    source_segment: str = "",
) -> go.Figure:
    """Kaplan-Meier curves per level of ``segment``."""
    fig = go.Figure()
    shown = source_segment or segment
    for k, level in enumerate(_segment_levels(pred, segment, order)):
        part = pred[pred[segment] == level]
        label = level_text(lang, shown, level) if shown in ("priority", "risk_group", "is_refurbished") else str(level)
        name = tr(
            lang,
            "ui.segment_legend",
            level=label,
            n=len(part),
            failures=int(part["event"].sum()),
        )
        for trace in _km(part, name, PALETTE[k % len(PALETTE)], ci=False):
            fig.add_trace(trace)
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, tr(lang, "ui.segment_km"), _hours(lang), "S(t)", 430)


def segment_failure_structure(
    pred: pd.DataFrame,
    segment: str,
    order: list | None = None,
    lang: str = "en",
    source_segment: str = "",
) -> go.Figure:
    """Share of drones in each segment that failed, split by failure mechanism."""
    levels = _segment_levels(pred, segment, order)
    shown = source_segment or segment
    labels = [
        level_text(lang, shown, lv) if shown in ("priority", "risk_group", "is_refurbished") else str(lv)
        for lv in levels
    ]
    size = pred.groupby(segment)["drone_id"].count()
    fig = go.Figure()
    for mode, color in FAILURE_COLORS.items():
        failed = pred[pred["failure_mode"] == mode].groupby(segment)["drone_id"].count()
        share = (failed.reindex(levels).fillna(0) / size.reindex(levels)).to_numpy()
        name = _mode(lang, mode)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=share,
                name=name,
                marker_color=color,
                hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, tr(lang, "ui.segment_structure"), "", tr(lang, "ui.share_of_segment"), 430)


# --- Model quality ------------------------------------------------------------


def learning_curves(models: dict[str, Model], lang: str = "en") -> go.Figure:
    """Train/valid loss per boosting iteration with the best iteration marked."""
    fig = make_subplots(
        rows=1, cols=len(models), subplot_titles=[_model(lang, k) for k in models]
    )
    for col, (kind, model) in enumerate(models.items(), start=1):
        for k, (split, metrics) in enumerate(model.history.items()):
            split_name = tr(lang, "ui.train" if split == "train" else "ui.valid")
            for metric, values in metrics.items():
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode="lines",
                        name=f"{_model(lang, kind)} {split_name}",
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
        fig, tr(lang, "ui.learning"), tr(lang, "ui.boosting_iteration"), tr(lang, "ui.loss")
    )


def survival_mean(table: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Mean predicted survival per model vs Kaplan-Meier on test."""
    km = table[table["source"] == "km"]
    fig = go.Figure(
        _step_with_band(
            km["t"].to_numpy(),
            km["s"].to_numpy(),
            km["ci_low"].to_numpy(),
            km["ci_high"].to_numpy(),
            tr(lang, "ui.km_observed"),
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
                name=tr(lang, "ui.mean_prediction", model=_model(lang, kind)),
                line={"color": color},
            )
        )
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, tr(lang, "ui.survival_mean"), _hours(lang), "S(t)")


def individual_curves(
    models: dict[str, Model], rows: pd.DataFrame, grid: np.ndarray, lang: str = "en"
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
                    name=f"{drone_id} {_model(lang, kind)}",
                    legendgroup=drone_id,
                    line={
                        "color": PALETTE[k % len(PALETTE)],
                        "dash": "solid" if kind == "cox" else "dash",
                    },
                )
            )
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, tr(lang, "ui.individual_title"), _hours(lang), "S(t)")


def brier_curve(brier_auc: pd.DataFrame, metrics: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Time-dependent Brier score per model on test."""
    fig = go.Figure()
    for kind, part in brier_auc.groupby("model", sort=False):
        ibs = metrics.query("model == @kind and split == 'test'")["ibs"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=part["t"],
                y=part["brier"],
                mode="lines",
                name=f"{_model(lang, kind)} (IBS={ibs:.4f})",
                line={"color": MODEL_COLORS[kind]},
            )
        )
    return _style(fig, tr(lang, "ui.brier_title"), _hours(lang), tr(lang, "ui.brier_score"))


def auc_curve(brier_auc: pd.DataFrame, metrics: pd.DataFrame, lang: str = "en") -> go.Figure:
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
                name=f"{_model(lang, kind)} (mean AUC={mean_auc:.4f})",
                line={"color": MODEL_COLORS[kind]},
            )
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text=tr(lang, "ui.random"))
    return _style(fig, tr(lang, "ui.auc_title"), _hours(lang), "AUC(t)")


def calibration(table: pd.DataFrame, horizon: float, lang: str = "en") -> go.Figure:
    """Predicted vs observed survival at ``horizon`` by prediction decile."""
    part = table[np.isclose(table["horizon"], horizon)]
    fig = go.Figure(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name=tr(lang, "ui.perfect_calibration"),
            line={"color": "grey", "dash": "dot"},
        )
    )
    for kind, rows in part.groupby("model", sort=False):
        fig.add_trace(
            go.Scatter(
                x=rows["predicted"],
                y=rows["observed"],
                mode="lines+markers",
                name=_model(lang, kind),
                line={"color": MODEL_COLORS[kind]},
                customdata=rows["n"],
                hovertemplate="predicted %{x:.2f}, observed %{y:.2f}, n=%{customdata}<extra></extra>",
            )
        )
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", constrain="domain")
    return _style(
        fig,
        tr(lang, "ui.calibration_title", horizon=horizon),
        tr(lang, "ui.predicted_s"),
        tr(lang, "ui.observed_s"),
        430,
    )


def bootstrap(samples: pd.DataFrame, summary: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Bootstrap distributions of metric differences between the first model pair."""
    a, b = summary.iloc[0][["model_a", "model_b"]]
    metrics = summary["metric"].tolist()
    titles = [
        f"{bundle(lang)['metric_labels'][m]}<br>p = {summary.set_index('metric').at[m, 'p_value']:.3f}"
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
        tr(lang, "ui.bootstrap_title", a=_model(lang, a), b=_model(lang, b)),
        height=380,
    )
    fig.update_layout(margin={"t": 100})
    return fig


def backtest(table: pd.DataFrame, landmark: float, lang: str = "en") -> go.Figure:
    """Conditional survival after the landmark: predicted vs Kaplan-Meier."""
    km = table[table["source"] == "km"]
    fig = go.Figure(
        _step_with_band(
            km["u"].to_numpy(),
            km["s"].to_numpy(),
            km["ci_low"].to_numpy(),
            km["ci_high"].to_numpy(),
            tr(lang, "ui.km_observed"),
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
                name=tr(lang, "ui.mean_prediction", model=_model(lang, kind)),
                line={"color": color},
            )
        )
    fig.update_xaxes(range=[0, u_max])
    fig.update_yaxes(range=[0, 1.02])
    return _style(
        fig,
        tr(lang, "ui.backtest_title", landmark=landmark),
        tr(lang, "ui.additional_hours"),
        tr(lang, "ui.conditional_survival"),
    )


def backtest_groups(
    table: pd.DataFrame, best: str, window: float, landmark: float, lang: str = "en"
) -> go.Figure:
    """Predicted vs observed failure probability within ``window`` by prediction quintile.

    ``table`` has one row per ``source`` and quintile; ``source`` is a model kind or
    ``"raw"`` for the best model before recalibration. The interval of the observed
    share is drawn for the best model only.
    """
    top = 1.1 * table[["predicted", "ci_high"]].to_numpy().max()
    fig = go.Figure(
        go.Scatter(
            x=[0, top],
            y=[0, top],
            mode="lines",
            name=tr(lang, "ui.perfect_calibration"),
            line={"color": "grey", "dash": "dot"},
        )
    )
    for source, rows in table.groupby("source", sort=False):
        raw = source == "raw"
        kind = best if raw else source
        fig.add_trace(
            go.Scatter(
                x=rows["predicted"],
                y=rows["observed"],
                mode="lines+markers",
                name=(
                    tr(lang, "ui.before_recalibration", model=_model(lang, kind))
                    if raw
                    else _model(lang, kind)
                ),
                line=(
                    {"color": "grey", "dash": "dash"}
                    if raw
                    else {"color": MODEL_COLORS[kind]}
                ),
                error_y=(
                    {
                        "type": "data",
                        "symmetric": False,
                        "array": rows["ci_high"] - rows["observed"],
                        "arrayminus": rows["observed"] - rows["ci_low"],
                        "thickness": 1,
                    }
                    if source == best
                    else None
                ),
                customdata=rows["n"],
                hovertemplate="predicted %{x:.1%}, observed %{y:.1%}, n=%{customdata}<extra></extra>",
            )
        )
    fig.update_xaxes(range=[0, top], tickformat=".0%")
    fig.update_yaxes(range=[0, top], tickformat=".0%")
    return _style(
        fig,
        tr(lang, "ui.backtest_groups", window=window, landmark=landmark),
        tr(lang, "ui.predicted_p"),
        tr(lang, "ui.observed_share"),
    )


# --- Risk factors -------------------------------------------------------------


def shap_importance(importance: pd.DataFrame, order_by: str, lang: str = "en") -> go.Figure:
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
                y=[_feature(lang, f) for f in order],
                orientation="h",
                name=_model(lang, kind),
                marker_color=color,
                hovertemplate="%{y}: %{x:.1%}<extra>" + _model(lang, kind) + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%")
    return _style(fig, tr(lang, "ui.shap_importance"), "", "", 560)


def shap_beeswarm(
    shap_values: pd.DataFrame, importance: pd.DataFrame, kind: str, lang: str = "en"
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
                    "title": tr(lang, "ui.feature_value"),
                    "tickvals": [0, 1],
                    "ticktext": [tr(lang, "ui.low"), tr(lang, "ui.high")],
                },
            },
            customdata=np.c_[part["drone_id"], part["display"]],
            hovertemplate="%{customdata[0]}: value %{customdata[1]}, SHAP %{x:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_color="grey")
    fig.update_yaxes(
        tickvals=list(range(len(order))), ticktext=[_feature(lang, f) for f in order]
    )
    return _style(
        fig,
        tr(lang, "ui.shap_beeswarm", model=_model(lang, kind)),
        tr(lang, "ui.contribution"),
        "",
        560,
    )


def shap_dependence(
    shap_values: pd.DataFrame, kind: str, feature: str, categorical: bool, lang: str = "en"
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
        tr(lang, "ui.shap_dependence", model=_model(lang, kind), feature=_feature(lang, feature)),
        _feature(lang, feature),
        "SHAP",
        430,
    )


# --- Fleet and drone card -----------------------------------------------------


def rul_histogram(active: pd.DataFrame, horizon: float, lang: str = "en") -> go.Figure:
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
        tr(lang, "ui.rul_hist", n=int((~finite).sum()), horizon=horizon),
        tr(lang, "ui.hours"),
        tr(lang, "ui.drones"),
    )


def conditional_curves(
    model: Model,
    rows: pd.DataFrame,
    ages: np.ndarray,
    grid: np.ndarray,
    names: list[str],
    title: str,
    dashes: list[str] | None = None,
    lang: str = "en",
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
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text=tr(lang, "ui.median_mark"))
    fig.update_yaxes(range=[0, 1.02])
    return _style(
        fig, title, tr(lang, "ui.additional_hours"), tr(lang, "ui.conditional_survival")
    )


def drone_survival(
    models: dict[str, Model],
    row: pd.DataFrame,
    grid: np.ndarray,
    age: float,
    failed: bool,
    lang: str = "en",
) -> go.Figure:
    """Unconditional survival of one drone under both models with its status marked."""
    fig = go.Figure()
    for kind, model in models.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=model.survival(row, grid)[0],
                mode="lines",
                name=_model(lang, kind),
                line={"color": MODEL_COLORS[kind]},
            )
        )
    fig.add_vline(
        x=age,
        line_dash="dash",
        line_color="red" if failed else "green",
        annotation_text=tr(lang, "ui.failed_at" if failed else "ui.flown", age=age),
    )
    fig.update_yaxes(range=[0, 1.02])
    return _style(fig, tr(lang, "ui.drone_survival"), tr(lang, "ui.hours"), "S(t)")


def drone_hazard(
    models: dict[str, Model],
    row: pd.DataFrame,
    grid: np.ndarray,
    bandwidth: float,
    age: float,
    failed: bool,
    lang: str = "en",
) -> go.Figure:
    """Hazard of one drone under both models with its status marked."""
    fig = go.Figure()
    for kind, model in models.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=HAZARD_SCALE * model.hazard(row, grid, bandwidth)[0],
                mode="lines",
                name=_model(lang, kind),
                line={"color": MODEL_COLORS[kind]},
                hovertemplate="%{x:.0f} h: %{y:.3f}<extra>" + _model(lang, kind) + "</extra>",
            )
        )
    fig.add_vline(
        x=age,
        line_dash="dash",
        line_color="red" if failed else "green",
        annotation_text=tr(lang, "ui.failed_at" if failed else "ui.flown", age=age),
    )
    fig.update_yaxes(rangemode="tozero")
    return _style(
        fig, tr(lang, "ui.drone_hazard"), tr(lang, "ui.hours"), bundle(lang)["hazard_unit"]
    )


def shap_waterfall(
    contrib: pd.DataFrame, base: float, kind: str, top: int = 12, lang: str = "en"
) -> go.Figure:
    """Largest per-feature contributions to one drone's risk score."""
    rows = contrib.reindex(contrib["shap"].abs().sort_values().index).tail(top)
    fig = go.Figure(
        go.Bar(
            x=rows["shap"],
            y=[
                f"{_feature(lang, f)} = {v}"
                for f, v in zip(rows["feature"], rows["display"])
            ],
            orientation="h",
            marker_color=np.where(rows["shap"] > 0, "#d62728", "#2ca02c"),
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    total = base + contrib["shap"].sum()
    title_key = "ui.waterfall_aft" if kind == "aft" else "ui.waterfall_cox"
    axis_key = "ui.contribution_aft" if kind == "aft" else "ui.contribution"
    return _style(
        fig,
        tr(lang, title_key, model=_model(lang, kind), base=base, total=total),
        tr(lang, axis_key),
        "",
        460,
    )


def age_risk(active: pd.DataFrame, window: float, lang: str = "en") -> go.Figure:
    """Hours already flown against the window failure probability, coloured by risk group."""
    labels = bundle(lang)["priority_labels"]
    colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
    fig = go.Figure()
    for group, color in colors.items():
        part = active[active["risk_group"] == group]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=part["duration_h"],
                y=part["p_fail_window"],
                mode="markers",
                name=labels.get(group, group),
                marker={"color": color, "size": 7, "opacity": 0.75},
                customdata=part["drone_id"],
                hovertemplate="%{customdata}: %{x:.0f} h, %{y:.1%}<extra></extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _style(
        fig,
        tr(lang, "ui.age_risk"),
        tr(lang, "ui.hours"),
        tr(lang, "ui.age_risk_y", window=window),
        420,
    )

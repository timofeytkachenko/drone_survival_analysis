"""Streamlit dashboard for drone failure analysis on the test split.

Run from the project root after executing ``drone_survival.ipynb``::

    uv run streamlit run dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

# ``streamlit run`` puts only the script directory on sys.path; ``dashboard`` and
# ``utils`` are imported from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import common, figures, insights, texts
from dashboard.data import DashboardData, load
from dashboard.labels import FEATURE_LABELS, MODEL_LABELS

TABS = {
    "overview": "Overview",
    "method": "Methodology",
    "segments": "Failure segments",
    "quality": "Model quality",
    "factors": "Risk factors",
    "fleet": "Fleet",
    "drone": "Drone card",
    "score": "Score new drones",
}
CARD_COLORS = {
    "primary": "blue",
    "danger": "red",
    "success": "green",
    "info": "blue",
    "secondary": "gray",
}
PLOT_CONFIG = {"displaylogo": False}
USAGE = "**How to use it.**"
# The shared fleet intro describes the Dash table controls.
INTRO_USAGE = {
    "fleet": f"{USAGE} Sort by clicking a column header; search and filters are above "
    "the table; hover over a header for the column explanation. Tick a drone's "
    "checkbox on the left and its card with an explanation of the forecast opens.",
}


@st.cache_resource(show_spinner="Loading artifacts and models…")
def _load() -> tuple[DashboardData, common.Settings, dict, pd.DataFrame]:
    data = load()
    settings = common.Settings.from_data(data)
    return (
        data,
        settings,
        common.text_context(data, settings),
        common.hazard_table(data, settings),
    )


st.set_page_config(page_title="Drone failures", layout="wide")
DATA, SETTINGS, TEXT_CONTEXT, HAZARD = _load()
PRED, TABLES, MODELS, META = DATA.predictions, DATA.tables, DATA.models, DATA.meta
BEST = SETTINGS.best


# --- Building blocks ----------------------------------------------------------------


def _fmt(text: str) -> str:
    return text.format(**TEXT_CONTEXT)


def _plot(fig) -> None:
    # The figure builders set their own colours for a light background.
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme=None)


def _table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def _findings(items: list[str]) -> None:
    st.info("**What the data shows**\n\n" + "\n".join(f"- {s}" for s in items))


def _explain(key: str, findings: list[str] | None = None) -> None:
    """Explanation card for a figure: what, how to read, findings, method details."""
    text = texts.FIGURES[key]
    with st.container(border=True):
        st.markdown(
            _fmt(f"**What is shown.** {text['what']}\n\n**How to read it.** {text['how']}")
        )
        if findings:
            _findings(findings)
        with st.expander("More about the method"):
            st.markdown(_fmt(text["details"]))


def _intro(key: str) -> None:
    text = texts.TAB_INTROS[key]
    if key in INTRO_USAGE:
        text = text.split(USAGE)[0] + INTRO_USAGE[key]
    with st.container(border=True):
        st.markdown(_fmt(text))


def _save(name: str) -> None:
    st.session_state[f"_{name}"] = st.session_state[name]


def _kept(name: str, default) -> dict:
    """Widget kwargs that keep the value while its tab is not rendered."""
    store = f"_{name}"
    st.session_state.setdefault(store, default)
    st.session_state[name] = st.session_state[store]
    return {"key": name, "on_change": _save, "args": (name,)}


# --- Tabs -----------------------------------------------------------------------------


def overview_tab() -> None:
    _intro("overview")
    cards = st.columns(3)
    for k, (title, value, subtitle, color) in enumerate(common.kpis(DATA, SETTINGS)):
        with cards[k % 3].container(border=True):
            st.caption(title)
            st.markdown(f"### :{CARD_COLORS[color]}[{value}]")
            st.caption(subtitle)
    st.caption(_fmt(common.KPI_HELP))
    with st.container(border=True):
        st.markdown(
            "##### Key findings\n\n"
            + "\n".join(f"- {f}" for f in common.key_findings(DATA, SETTINGS))
        )
    overview = insights.overview(PRED)
    left, right = st.columns([7, 5])
    with left:
        _plot(figures.km_overall(PRED))
        _explain("km_overall", overview[:2])
    with right:
        _plot(figures.failure_modes(PRED))
        _explain("failure_modes", overview[2:])
    left, _ = st.columns([7, 5])
    with left:
        _plot(figures.hazard(HAZARD, with_models=False))
        _explain("hazard_observed", insights.hazard_trend(HAZARD))


def method_tab() -> None:
    _intro("method")
    st.subheader("Analysis steps")
    for title, body in texts.METHOD_STEPS:
        with st.expander(title, expanded=True):
            st.markdown(_fmt(body))
    st.subheader("Tuned model settings")
    st.caption(
        _fmt(
            "Result of automatic tuning (step 6). Values were chosen by the lowest "
            "validation IBS out of {n_trials} trials per model."
        )
    )
    _table(common.hyperparams_frame(MODELS))
    st.subheader("Feature dictionary")
    st.caption(
        "What the model knows about each drone. The range is the values in the "
        "training set: outside it the forecast is less reliable, so \"what if\" "
        "scenarios are limited to it. Use the column names in uploaded CSV files."
    )
    _table(common.features_frame(META))
    st.subheader("Glossary")
    _table(
        pd.DataFrame(texts.GLOSSARY, columns=["Term", "Definition"]),
        column_config={
            "Term": st.column_config.TextColumn(width="small"),
            "Definition": st.column_config.TextColumn(width="large"),
        },
    )


def segments_tab() -> None:
    _intro("segments")
    segment = st.selectbox(
        "Breakdown",
        list(common.SEGMENTS),
        format_func=common.SEGMENTS.get,
        **_kept("segment", "environment"),
    )
    order = common.SEGMENT_ORDER.get(segment)
    levels = common.segment_levels(PRED, segment)
    _findings(common.segment_findings(PRED, segment, levels, SETTINGS.window))
    left, right = st.columns(2)
    with left:
        _plot(figures.segment_km(PRED, segment, order))
        _explain("segment_km")
    with right:
        _plot(figures.segment_failure_structure(PRED, segment, order))
        _explain("segment_structure")
    st.subheader("Statistics by segment")
    _table(common.segment_stats(PRED, segment, levels, SETTINGS.window))
    _explain("segment_table")


def quality_tab() -> None:
    brier_auc, metric_table = TABLES["brier_auc"], TABLES["metrics"]
    _intro("quality")
    st.subheader("Step 1. Quality metrics")
    _table(common.metrics_frame(metric_table))
    _explain("metrics", insights.metrics(metric_table, BEST))

    st.subheader("Step 2. How the models trained")
    _plot(figures.learning_curves(MODELS))
    _explain("learning_curves", insights.learning(MODELS))

    st.subheader("Step 3. Predicted vs observed")
    left, right = st.columns(2)
    with left:
        _plot(figures.survival_mean(TABLES["survival_mean"]))
        _explain(
            "survival_mean",
            insights.mean_vs_km(
                TABLES["survival_mean"], META["eval_times"]["test"][-1]
            ),
        )
    with right:
        sample = PRED.sample(5, random_state=0)
        _plot(figures.individual_curves(MODELS, sample, SETTINGS.surv_grid))
        _explain("individual_curves")
    left, right = st.columns(2)
    with left:
        _plot(figures.hazard(HAZARD, with_models=True))
    with right:
        _explain("hazard_mean", insights.hazard_fit(HAZARD))

    st.subheader("Step 4. Accuracy and discrimination over time")
    left, right = st.columns(2)
    with left:
        _plot(figures.brier_curve(brier_auc, metric_table))
        _explain("brier", insights.brier(brier_auc))
    with right:
        _plot(figures.auc_curve(brier_auc, metric_table))
        _explain("auc", insights.auc(brier_auc))

    st.subheader("Step 5. Calibration")
    horizons = META["calibration_horizons"]
    left, right = st.columns(2)
    with left:
        horizon = st.selectbox(
            "Calibration horizon",
            horizons,
            format_func=lambda h: f"{h:.0f} h",
            **_kept("calibration_horizon", horizons[0]),
        )
        _plot(figures.calibration(TABLES["calibration"], horizon))
    with right:
        _explain("calibration", insights.calibration(TABLES["calibration"], horizon))

    st.subheader("Step 6. Statistical model comparison (bootstrap)")
    _table(common.bootstrap_frame(TABLES["bootstrap_summary"]))
    _explain("bootstrap_table")
    _plot(figures.bootstrap(TABLES["bootstrap_samples"], TABLES["bootstrap_summary"]))
    _explain("bootstrap", insights.bootstrap(TABLES["bootstrap_summary"]))


def factors_tab() -> None:
    importance, shap_values = TABLES["shap_importance"], TABLES["shap_values"]
    top_feature = (
        importance[importance["model"] == BEST].nlargest(1, "share")["feature"].iloc[0]
    )
    _intro("factors")
    left, right = st.columns([5, 7])
    with left:
        st.subheader("Step 1. Which factors matter most")
        _plot(figures.shap_importance(importance, BEST))
        _explain("shap_importance", insights.shap_top(importance, BEST))
    with right:
        st.subheader("Step 2. Which way each factor acts")
        kind = st.radio(
            "Model",
            list(MODELS),
            format_func=MODEL_LABELS.get,
            horizontal=True,
            **_kept("shap_model", BEST),
        )
        _plot(figures.shap_beeswarm(shap_values, importance, kind))
        _explain("shap_beeswarm", common.beeswarm_findings(DATA, SETTINGS, kind))

    st.subheader("Step 3. How exactly risk depends on a factor")
    feature = st.selectbox(
        "Feature",
        SETTINGS.features,
        format_func=FEATURE_LABELS.get,
        **_kept("shap_feature", top_feature),
    )
    st.caption(texts.FEATURE_DESCRIPTIONS[feature])
    categorical = feature in SETTINGS.cat_features
    left, right = st.columns([7, 5])
    with left:
        _plot(
            figures.shap_dependence(shap_values, kind, feature, categorical=categorical)
        )
    with right:
        _explain(
            "shap_dependence",
            insights.shap_direction(shap_values, kind, feature, categorical),
        )


def _open_drone() -> None:
    """Fleet row selection opens the drone card."""
    rows = st.session_state["fleet"].selection.rows
    if rows:
        st.session_state["_drone"] = st.session_state["fleet_ids"][rows[0]]
        st.session_state["tab"] = TABS["drone"]


def _fleet_table() -> None:
    frame = common.fleet_frame(DATA, SETTINGS)
    filters = st.columns([2, 3, 3, 3])
    query = filters[0].text_input("Search by drone", **_kept("fleet_query", ""))
    for col, column in zip(filters[1:], ("priority", "environment", "model")):
        levels = sorted(frame[column].dropna().unique().tolist(), key=str)
        chosen = col.multiselect(
            common.fleet_columns(SETTINGS)[column],
            levels,
            placeholder="all",
            **_kept(f"fleet_{column}", []),
        )
        if chosen:
            frame = frame[frame[column].isin(chosen)]
    if query:
        frame = frame[frame["drone_id"].str.contains(query, case=False, regex=False)]
    frame = frame.reset_index(drop=True)
    st.session_state["fleet_ids"] = frame["drone_id"].tolist()
    st.caption(
        f"Drones shown: {len(frame)}. Select a row to open the drone card."
    )
    st.dataframe(
        _priority_style(frame),
        hide_index=True,
        width="stretch",
        height=560,
        column_config=_forecast_config(common.fleet_columns(SETTINGS)),
        key="fleet",
        on_select=_open_drone,
        selection_mode="single-row",
    )


def _forecast_config(names: dict[str, str]) -> dict:
    """Column headers, tooltips and number formats of a forecast table."""
    tooltips = common.fleet_tooltips(SETTINGS)
    formats = {
        "age_h": "%.0f",
        "p_fail_window": "%.3f",
        "rul_q10": "%.0f",
        "rul_median": "%.0f",
        "rmrl": "%.0f",
    }
    config = {}
    for col, name in names.items():
        if col in formats:
            config[col] = st.column_config.NumberColumn(
                name, help=tooltips.get(col), format=formats[col]
            )
        elif col == "out_of_range":
            config[col] = st.column_config.CheckboxColumn(name)
        else:
            config[col] = st.column_config.TextColumn(name, help=tooltips.get(col))
    return config


def _priority_style(frame: pd.DataFrame):
    """Row background by maintenance priority."""
    return frame.style.apply(
        lambda r: (
            [f"background-color: {common.PRIORITY_COLORS.get(r['priority'], '')}"]
            * len(r)
        ),
        axis=1,
    )


def fleet_tab() -> None:
    active = common.active(DATA)
    riskiest = active.nlargest(8, "p_fail_window")
    _intro("fleet")
    _findings(insights.fleet(active, SETTINGS.window, SETTINGS.horizon))
    st.subheader("Step 1. Drones in service")
    _fleet_table()
    _explain("fleet_table")
    st.subheader("Step 2. How much life is left")
    left, right = st.columns(2)
    with left:
        _plot(figures.rul_histogram(active, SETTINGS.horizon))
        _explain("rul_hist")
    with right:
        _plot(
            figures.conditional_curves(
                DATA.best_model,
                riskiest,
                riskiest["duration_h"].to_numpy(),
                SETTINGS.rul_grid,
                riskiest["drone_id"].tolist(),
                "Remaining life of the riskiest drones",
            )
        )
        _explain("riskiest")
    st.subheader("Step 3. Can the remaining-life forecast be trusted")
    left, right = st.columns(2)
    with left:
        _plot(figures.backtest(TABLES["backtest"], META["landmark"]))
        _explain("backtest", insights.backtest(TABLES["backtest"], META["landmark"]))
    with right:
        _plot(
            figures.backtest_groups(
                TABLES["backtest_groups"], BEST, SETTINGS.window, META["landmark"]
            )
        )
        _explain(
            "backtest_groups",
            insights.backtest_groups(TABLES["backtest_groups"], BEST),
        )


def _whatif_key(drone_id: str, feature: str) -> str:
    return f"whatif-{drone_id}-{feature}"


def _run_whatif(drone_id: str) -> None:
    """Store the submitted scenario so the figure above the form can use it."""
    st.session_state["whatif"] = {
        "drone": drone_id,
        "values": [
            st.session_state[_whatif_key(drone_id, f)] for f in SETTINGS.features
        ],
    }


def _whatif_form(drone_id: str, current: list, scenario: list | None) -> None:
    values = scenario or current
    with st.form(f"whatif-form-{drone_id}"):
        cols = st.columns(4)
        for k, (feature, value) in enumerate(zip(SETTINGS.features, values)):
            col, key, label = (
                cols[k % 4],
                _whatif_key(drone_id, feature),
                FEATURE_LABELS[feature],
            )
            if feature == "is_refurbished":
                col.selectbox(
                    label,
                    ["True", "False"],
                    index=["True", "False"].index(value),
                    format_func={"True": "Yes", "False": "No"}.get,
                    key=key,
                )
            elif feature in SETTINGS.cat_features:
                options = META["categories"][feature]
                col.selectbox(
                    label,
                    options,
                    index=options.index(value) if value in options else 0,
                    format_func=str,
                    key=key,
                )
            else:
                low, high = META["numeric_ranges"][feature]
                col.number_input(
                    label,
                    value=value,
                    key=key,
                    help=f"Training range: {low:.3g} – {high:.3g}",
                )
        st.form_submit_button(
            "Recalculate", type="primary", on_click=_run_whatif, args=(drone_id,)
        )


def _out_of_range(changes: list[tuple[str, object, object]]) -> list[str]:
    """Edited numeric features whose new value is outside the training range."""
    bad = []
    for feature, _old, new in changes:
        if feature in SETTINGS.cat_features:
            continue
        low, high = META["numeric_ranges"][feature]
        if new is None or not low <= float(new) <= high:
            bad.append(f'"{FEATURE_LABELS[feature]}" ({low:.3g} – {high:.3g})')
    return bad


def drone_tab() -> None:
    _intro("drone")
    options = common.drone_options(PRED)
    drone_id = st.selectbox(
        "Drone",
        list(options),
        format_func=options.get,
        **_kept("drone", next(iter(options))),
    )
    rows = common.drone_rows(PRED, drone_id)
    row = rows.iloc[0]
    failed = bool(row["event"])
    _findings(common.drone_findings(DATA, SETTINGS, drone_id))

    left, right = st.columns([4, 8])
    with left:
        st.markdown(f"#### {drone_id}\n\n**{common.drone_status(row)}**")
        if row["in_service"]:
            st.table(
                pd.DataFrame(
                    common.drone_metrics(row, SETTINGS),
                    columns=["Measure", "Value", "Explanation"],
                ).set_index("Measure")
            )
        st.markdown(f"Risk group: {row['risk_group']}")
        st.table(common.drone_features(row, SETTINGS.features).set_index("Feature"))
    with right:
        _plot(
            figures.drone_survival(
                MODELS, rows, SETTINGS.surv_grid, row["duration_h"], failed
            )
        )
        _explain("drone_survival")
        _plot(
            figures.drone_hazard(
                MODELS,
                rows,
                common.drone_hazard_grid(SETTINGS, row["duration_h"]),
                SETTINGS.hazard_bandwidth,
                row["duration_h"],
                failed,
            )
        )
        _explain("drone_hazard")

    current = common.whatif_defaults(row, SETTINGS.features)
    stored = st.session_state.get("whatif")
    scenario_values = (
        stored["values"] if stored and stored["drone"] == drone_id else None
    )
    changes = (
        common.whatif_changes(
            row, SETTINGS.features, SETTINGS.cat_features, scenario_values
        )
        if scenario_values
        else []
    )
    invalid = _out_of_range(changes)

    base_rows = common.whatif_base(PRED, drone_id)
    age = base_rows["duration_h"].to_numpy()
    model = DATA.best_model
    title = f"Remaining life from {age[0]:.0f} flight hours ({MODEL_LABELS[BEST]})"
    left, right = st.columns(2)
    with left:
        _plot(
            figures.shap_waterfall(
                common.drone_contrib(DATA, drone_id), META["shap_base"][BEST], BEST
            )
        )
        _explain("drone_shap")
    with right:
        if scenario_values and not invalid:
            scenario = common.whatif_scenario(
                base_rows, SETTINGS.features, scenario_values
            )
            fig = figures.conditional_curves(
                model,
                pd.concat([base_rows, scenario], ignore_index=True),
                np.r_[age, age],
                SETTINGS.rul_grid,
                ["Current conditions", "Scenario"],
                title,
                dashes=["solid", "dash"],
            )
        else:
            fig = figures.conditional_curves(
                model, base_rows, age, SETTINGS.rul_grid, ["Current conditions"], title
            )
        _plot(fig)
        _explain("drone_whatif")

    st.subheader('"What if" scenario')
    st.caption(texts.WHATIF_HELP)
    _whatif_form(drone_id, current, scenario_values)
    if invalid:
        st.warning(
            "Values outside the training range: " + ", ".join(invalid) + "."
        )
    elif scenario_values:
        base, new, table = common.whatif_compare(model, base_rows, scenario, SETTINGS)
        st.table(table.set_index(""))
        _findings(insights.whatif(base, new, changes, SETTINGS.window))


def score_tab() -> None:
    _intro("score")
    st.subheader("Step 1. Prepare the file")
    st.download_button(
        "Download template",
        common.upload_template(DATA).to_csv(index=False),
        file_name="drone_template.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.subheader("Step 2. Upload it")
    upload = st.file_uploader("CSV file with one row per drone", type="csv", key="score_file")
    if upload is None:
        return
    try:
        scored, warnings = common.score_upload(
            common.read_csv_bytes(upload.getvalue()), DATA, SETTINGS
        )
    except ValueError as exc:
        st.error(f"{upload.name}: {exc}")
        return
    _findings(common.score_summary(scored, SETTINGS))
    if warnings:
        st.warning("**Data warnings**\n\n" + "\n".join(f"- {w}" for w in warnings))

    st.subheader("Step 3. Review and download the forecast")
    columns = common.score_columns(SETTINGS)
    _table(
        _priority_style(scored[list(columns)]),
        height=min(560, 38 + 35 * len(scored)),
        column_config=_forecast_config(columns),
    )
    st.download_button(
        "Download results (CSV)",
        scored.to_csv(index=False),
        file_name="drone_forecasts.csv",
        mime="text/csv",
        type="primary",
        on_click="ignore",
    )
    _explain("score_table")


# --- App ----------------------------------------------------------------------------


RENDERERS = {
    "overview": overview_tab,
    "method": method_tab,
    "segments": segments_tab,
    "quality": quality_tab,
    "factors": factors_tab,
    "fleet": fleet_tab,
    "drone": drone_tab,
    "score": score_tab,
}

st.title("Drone failure analysis: test set")
for key, container in zip(
    TABS, st.tabs(list(TABS.values()), key="tab", on_change="rerun")
):
    if container.open:
        with container:
            RENDERERS[key]()

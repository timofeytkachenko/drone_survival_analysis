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

from dashboard import common, figures, insights
from dashboard.data import DashboardData, load
from dashboard.locale import bundle, tr

TAB_KEYS = [
    "overview",
    "method",
    "segments",
    "quality",
    "factors",
    "fleet",
    "drone",
    "score",
]
CARD_COLORS = {
    "primary": "blue",
    "danger": "red",
    "success": "green",
    "info": "blue",
    "secondary": "gray",
}
PLOT_CONFIG = {"displaylogo": False}
LANG = "en"


@st.cache_resource(show_spinner="Loading artifacts and models…")
def _load() -> tuple[DashboardData, common.Settings, pd.DataFrame, pd.DataFrame]:
    data = load()
    settings = common.Settings.from_data(data)
    return (
        data,
        settings,
        common.hazard_table(data, settings),
        common.disagreeing_drones(data, settings),
    )


st.set_page_config(page_title="Drone failures", layout="wide")
DATA, SETTINGS, HAZARD, DISAGREE = _load()
PRED, TABLES, MODELS, META = DATA.predictions, DATA.tables, DATA.models, DATA.meta
BEST = SETTINGS.best
# Forecasts come from the production models (train + validation), see trainer.release
FORECAST = SETTINGS.forecast
FORECAST_MODELS = DATA.production


def _fmt(text: str) -> str:
    return text.format(**common.text_context(DATA, SETTINGS, LANG))


def _plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme=None)


def _table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def _findings(items: list[str]) -> None:
    st.info("\n".join(f"- {item}" for item in items), title=tr(LANG, "ui.what_data_shows"))


def _explain(key: str, findings: list[str] | None = None) -> None:
    """Explanation card for a figure: what, how to read, findings, method details."""
    text = bundle(LANG)["figures"][key]
    with st.container(border=True):
        st.markdown(
            _fmt(
                f"**{tr(LANG, 'ui.what_shown')}** {text['what']}\n\n"
                f"**{tr(LANG, 'ui.how_to_read')}** {text['how']}"
            )
        )
        if findings:
            _findings(findings)
        with st.expander(tr(LANG, "ui.more_method")):
            st.markdown(_fmt(text["details"]))


def _intro(key: str) -> None:
    text = (
        tr(LANG, "ui.fleet_streamlit")
        if key == "fleet"
        else bundle(LANG)["tab_intros"][key]
    )
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


def overview_tab() -> None:
    _intro("overview")
    cards = st.columns(4)
    for k, (title, value, subtitle, color) in enumerate(common.kpis(DATA, SETTINGS, LANG)):
        with cards[k % 4].container(border=True):
            st.caption(title)
            st.markdown(f"### :{CARD_COLORS[color]}[{value}]")
            st.caption(subtitle)
    st.caption(_fmt(tr(LANG, "ui.priority_table", p_medium=SETTINGS.p_medium)))
    _table(common.priority_sensitivity(common.active(DATA), SETTINGS, LANG))
    with st.container(border=True):
        st.markdown(
            f"##### {tr(LANG, 'ui.key_findings')}\n\n"
            + "\n".join(f"- {item}" for item in common.key_findings(DATA, SETTINGS, LANG))
        )
    overview = insights.overview(PRED, LANG)
    left, right = st.columns(2)
    with left:
        _plot(figures.km_overall(PRED, LANG))
        _explain("km_overall", overview[:2])
    with right:
        _plot(figures.hazard(HAZARD, [BEST], LANG))
        _explain("hazard_overview", insights.hazard_trend(HAZARD, LANG))
    left, right = st.columns(2)
    with left:
        _plot(figures.failure_modes(PRED, LANG))
        _explain("failure_modes", overview[2:])
    with right:
        _plot(figures.failure_mode_timing(PRED, LANG))
        _explain("failure_timing")


def method_tab() -> None:
    catalog = bundle(LANG)
    _intro("method")
    st.subheader(tr(LANG, "ui.analysis_steps"))
    for title, body in catalog["method_steps"]:
        with st.expander(title, expanded=True):
            st.markdown(_fmt(body))
    st.subheader(tr(LANG, "ui.tuned_settings"))
    st.caption(_fmt(tr(LANG, "ui.tuned_caption", n_trials=META["n_trials"])))
    _table(common.hyperparams_frame(MODELS, LANG))
    st.subheader(tr(LANG, "ui.feature_dictionary"))
    st.caption(tr(LANG, "ui.feature_dictionary_caption"))
    _table(common.features_frame(META, LANG))
    st.subheader(tr(LANG, "ui.glossary"))
    _table(
        pd.DataFrame(
            catalog["glossary"],
            columns=[tr(LANG, "ui.term"), tr(LANG, "ui.definition")],
        ),
        column_config={
            tr(LANG, "ui.term"): st.column_config.TextColumn(width="small"),
            tr(LANG, "ui.definition"): st.column_config.TextColumn(width="large"),
        },
    )


def segments_tab() -> None:
    _intro("segments")
    options = common.segment_options(DATA, SETTINGS, LANG)
    segment = st.selectbox(
        tr(LANG, "ui.breakdown"),
        list(options),
        format_func=options.get,
        **_kept("segment", "environment"),
    )
    frame, column, levels = common.prepare_segment(PRED, segment)
    _findings(common.segment_findings(frame, column, levels, SETTINGS.window, LANG))
    left, right = st.columns(2)
    with left:
        _plot(figures.segment_km(frame, column, levels, LANG, segment))
        _explain("segment_km")
    with right:
        _plot(figures.segment_failure_structure(frame, column, levels, LANG, segment))
        _explain("segment_structure")
    st.subheader(tr(LANG, "ui.statistics_by_segment"))
    _table(
        common.segment_stats(
            frame, column, levels, SETTINGS.window, LANG, options[segment]
        )
    )
    _explain("segment_table")


def quality_tab() -> None:
    brier_auc, metric_table = TABLES["brier_auc"], TABLES["metrics"]
    _intro("quality")
    st.subheader(tr(LANG, "ui.step_metrics"))
    _table(common.metrics_frame(metric_table, LANG))
    _explain("metrics", insights.metrics(metric_table, BEST, LANG))

    st.subheader(tr(LANG, "ui.gate_title"))
    _findings(common.gate_findings(DATA, LANG))
    _table(common.gate_frame(DATA, LANG))

    st.subheader(tr(LANG, "ui.step_training"))
    _plot(figures.learning_curves(MODELS, LANG))
    _explain("learning_curves", insights.learning(MODELS, LANG))

    st.subheader(tr(LANG, "ui.step_predicted"))
    left, right = st.columns(2)
    with left:
        _plot(figures.survival_mean(TABLES["survival_mean"], LANG))
        _explain(
            "survival_mean",
            insights.mean_vs_km(
                TABLES["survival_mean"], META["eval_times"]["test"][-1], LANG
            ),
        )
    with right:
        _plot(figures.individual_curves(MODELS, DISAGREE, SETTINGS.surv_grid, LANG))
        _explain("individual_curves")
    left, right = st.columns(2)
    with left:
        _plot(figures.hazard(HAZARD, list(MODELS), LANG))
    with right:
        _explain("hazard_mean", insights.hazard_fit(HAZARD, LANG))

    st.subheader(tr(LANG, "ui.step_accuracy"))
    left, right = st.columns(2)
    with left:
        _plot(figures.brier_curve(brier_auc, metric_table, LANG))
        _explain("brier", insights.brier(brier_auc, LANG))
    with right:
        _plot(figures.auc_curve(brier_auc, metric_table, LANG))
        _explain("auc", insights.auc(brier_auc, LANG))

    st.subheader(tr(LANG, "ui.step_calibration"))
    horizons = META["calibration_horizons"]
    left, right = st.columns(2)
    with left:
        horizon = st.selectbox(
            tr(LANG, "ui.calibration_horizon"),
            horizons,
            format_func=lambda hours: tr(LANG, "ui.horizon_hours", h=hours),
            **_kept("calibration_horizon", horizons[0]),
        )
        _plot(figures.calibration(TABLES["calibration"], horizon, LANG))
    with right:
        _explain(
            "calibration", insights.calibration(TABLES["calibration"], horizon, LANG)
        )

    st.subheader(tr(LANG, "ui.step_bootstrap"))
    _table(common.bootstrap_frame(TABLES["bootstrap_summary"], LANG))
    _explain("bootstrap_table")
    _plot(figures.bootstrap(TABLES["bootstrap_samples"], TABLES["bootstrap_summary"], LANG))
    _explain("bootstrap", insights.bootstrap(TABLES["bootstrap_summary"], LANG))


def factors_tab() -> None:
    importance, shap_values = TABLES["shap_importance"], TABLES["shap_values"]
    labels = bundle(LANG)
    top_feature = (
        importance[importance["model"] == BEST].nlargest(1, "share")["feature"].iloc[0]
    )
    _intro("factors")
    left, right = st.columns([5, 7])
    with left:
        st.subheader(tr(LANG, "ui.step_importance"))
        _plot(figures.shap_importance(importance, BEST, LANG))
        _explain("shap_importance", insights.shap_top(importance, BEST, lang=LANG))
    with right:
        st.subheader(tr(LANG, "ui.step_direction"))
        kind = st.radio(
            tr(LANG, "ui.model"),
            list(MODELS),
            format_func=labels["model_labels"].get,
            horizontal=True,
            **_kept("shap_model", BEST),
        )
        _plot(figures.shap_beeswarm(shap_values, importance, kind, LANG))
        _explain("shap_beeswarm", common.beeswarm_findings(DATA, SETTINGS, kind, LANG))

    st.subheader(tr(LANG, "ui.step_dependence"))
    feature = st.selectbox(
        tr(LANG, "ui.feature"),
        SETTINGS.features,
        format_func=labels["feature_labels"].get,
        **_kept("shap_feature", top_feature),
    )
    st.caption(labels["feature_descriptions"][feature])
    categorical = feature in SETTINGS.cat_features
    left, right = st.columns([7, 5])
    with left:
        _plot(figures.shap_dependence(shap_values, kind, feature, categorical, LANG))
    with right:
        _explain(
            "shap_dependence",
            insights.shap_direction(shap_values, kind, feature, categorical, LANG),
        )


def _open_drone() -> None:
    """Fleet row selection opens the drone card."""
    rows = st.session_state["fleet"].selection.rows
    if rows:
        st.session_state["_drone"] = st.session_state["fleet_ids"][rows[0]]
        st.session_state["tab_id"] = "drone"
        st.session_state["tab"] = tr(LANG, "ui.tabs.drone")


def _fleet_table() -> None:
    frame = common.fleet_frame(DATA, SETTINGS, LANG)
    headers = common.fleet_columns(SETTINGS, LANG)
    filters = st.columns([2, 3, 3, 3])
    query = filters[0].text_input(tr(LANG, "ui.search_drone"), **_kept("fleet_query", ""))
    for col, column in zip(filters[1:], ("priority", "environment", "model")):
        levels = sorted(frame[column].dropna().unique().tolist(), key=str)
        chosen = col.multiselect(
            headers[column],
            levels,
            placeholder=tr(LANG, "ui.all"),
            **_kept(f"fleet_{column}", []),
        )
        if chosen:
            frame = frame[frame[column].isin(chosen)]
    if query:
        frame = frame[frame["drone_id"].str.contains(query, case=False, regex=False)]
    frame = frame.reset_index(drop=True)
    st.session_state["fleet_ids"] = frame["drone_id"].tolist()
    st.caption(tr(LANG, "ui.drones_shown", n=len(frame)))
    st.download_button(
        tr(LANG, "ui.download_fleet"),
        frame.to_csv(index=False),
        file_name="fleet.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.dataframe(
        _priority_style(frame),
        hide_index=True,
        width="stretch",
        height=560,
        column_config=_forecast_config(headers),
        key="fleet",
        on_select=_open_drone,
        selection_mode="single-row",
    )


def _forecast_config(names: dict[str, str]) -> dict:
    """Column headers, tooltips and number formats of a forecast table."""
    tooltips = common.fleet_tooltips(SETTINGS, LANG)
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
    """Row background by the stored maintenance-priority code."""
    return frame.style.apply(
        lambda row: (
            [f"background-color: {common.PRIORITY_COLORS.get(row['priority'], '')}"]
            * len(row)
        ),
        axis=1,
    )


def fleet_tab() -> None:
    active = common.active(DATA)
    riskiest = active.nlargest(8, "p_fail_window")
    _intro("fleet")
    _findings(insights.fleet(active, SETTINGS.window, SETTINGS.horizon, LANG))
    st.subheader(tr(LANG, "ui.step_fleet"))
    _fleet_table()
    _explain("fleet_table")
    _plot(figures.age_risk(active, SETTINGS.window, LANG))
    _explain("age_risk")
    st.subheader(tr(LANG, "ui.step_life"))
    left, right = st.columns(2)
    with left:
        _plot(figures.rul_histogram(active, SETTINGS.horizon, LANG))
        _explain("rul_hist")
    with right:
        _plot(
            figures.conditional_curves(
                DATA.forecast_model,
                riskiest,
                riskiest["duration_h"].to_numpy(),
                SETTINGS.rul_grid,
                riskiest["drone_id"].tolist(),
                tr(LANG, "ui.riskiest_title"),
                lang=LANG,
            )
        )
        _explain("riskiest")
    st.subheader(tr(LANG, "ui.step_trust"))
    left, right = st.columns(2)
    with left:
        _plot(figures.backtest(TABLES["backtest"], META["landmark"], LANG))
        _explain(
            "backtest", insights.backtest(TABLES["backtest"], META["landmark"], LANG)
        )
    with right:
        _plot(
            figures.backtest_groups(
                TABLES["backtest_groups"], BEST, SETTINGS.window, META["landmark"], LANG
            )
        )
        _explain(
            "backtest_groups",
            insights.backtest_groups(TABLES["backtest_groups"], BEST, LANG),
        )


def _whatif_key(drone_id: str, feature: str) -> str:
    return f"whatif-{drone_id}-{feature}"


def _run_whatif(drone_id: str, order: list[str]) -> None:
    """Store the submitted scenario so the figure above the form can use it."""
    st.session_state["whatif"] = {
        "drone": drone_id,
        "values": [st.session_state[_whatif_key(drone_id, feature)] for feature in order],
    }


def _whatif_fields(drone_id: str, features: list[str], values: list) -> None:
    labels = bundle(LANG)["feature_labels"]
    yes_no = {"True": tr(LANG, "ui.yes"), "False": tr(LANG, "ui.no")}
    cols = st.columns(4)
    for k, (feature, value) in enumerate(zip(features, values)):
        col, key, label = cols[k % 4], _whatif_key(drone_id, feature), labels[feature]
        if feature == "is_refurbished":
            col.selectbox(
                label,
                ["True", "False"],
                index=["True", "False"].index(value),
                format_func=yes_no.get,
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
            shown = float(np.clip(float(value), low, high))
            col.number_input(
                label,
                min_value=float(low),
                max_value=float(high),
                value=shown,
                key=key,
                help=tr(LANG, "ui.training_range", low=low, high=high),
            )


def _whatif_form(drone_id: str, order: list[str], current: list, scenario: list | None) -> None:
    values = scenario or current
    levers = [feature for feature in order if feature in common.LEVERS]
    fixed = [feature for feature in order if feature not in common.LEVERS]
    lever_values = values[: len(levers)]
    fixed_values = values[len(levers) :]
    with st.form(f"whatif-form-{drone_id}"):
        st.markdown(f"**{tr(LANG, 'ui.whatif_levers')}**")
        _whatif_fields(drone_id, levers, lever_values)
        st.markdown(f"**{tr(LANG, 'ui.whatif_fixed')}**")
        _whatif_fields(drone_id, fixed, fixed_values)
        st.form_submit_button(
            tr(LANG, "ui.recalculate"),
            type="primary",
            on_click=_run_whatif,
            args=(drone_id, order),
        )


def drone_tab() -> None:
    labels = bundle(LANG)
    _intro("drone")
    options = common.drone_options(PRED, LANG)
    drone_id = st.selectbox(
        tr(LANG, "ui.drone"),
        list(options),
        format_func=options.get,
        **_kept("drone", next(iter(options))),
    )
    rows = common.drone_rows(PRED, drone_id)
    row = rows.iloc[0]
    failed = bool(row["event"])
    _findings(common.drone_findings(DATA, SETTINGS, drone_id, LANG))

    left, right = st.columns([4, 8])
    with left:
        st.markdown(f"#### {drone_id}\n\n**{common.drone_status(row, LANG)}**")
        if row["in_service"]:
            st.table(
                pd.DataFrame(
                    common.drone_metrics(row, SETTINGS, LANG),
                    columns=[
                        tr(LANG, "ui.measure"),
                        tr(LANG, "ui.value"),
                        tr(LANG, "ui.explanation"),
                    ],
                ).set_index(tr(LANG, "ui.measure"))
            )
        group = labels["priority_labels"].get(row["risk_group"], row["risk_group"])
        st.markdown(tr(LANG, "ui.risk_group_line", group=group))
        features = common.drone_features(row, SETTINGS.features, LANG)
        st.table(features.set_index(tr(LANG, "ui.feature")))
    with right:
        _plot(
            figures.drone_survival(
                FORECAST_MODELS,
                rows,
                SETTINGS.surv_grid,
                row["duration_h"],
                failed,
                LANG,
            )
        )
        _explain("drone_survival")
        _plot(
            figures.drone_hazard(
                FORECAST_MODELS,
                rows,
                common.drone_hazard_grid(SETTINGS, row["duration_h"]),
                SETTINGS.hazard_bandwidth,
                row["duration_h"],
                failed,
                LANG,
            )
        )
        _explain("drone_hazard")

    order = common.whatif_order(SETTINGS.features)
    current = common.whatif_defaults(row, order)
    stored = st.session_state.get("whatif")
    scenario_values = stored["values"] if stored and stored["drone"] == drone_id else None
    changes = (
        common.whatif_changes(row, order, SETTINGS.cat_features, scenario_values)
        if scenario_values
        else []
    )
    base_rows = common.whatif_base(PRED, drone_id)
    age = base_rows["duration_h"].to_numpy()
    model = DATA.forecast_model
    title = tr(
        LANG,
        "ui.remaining_life_title",
        age=age[0],
        model=labels["model_labels"][FORECAST],
    )
    names = [tr(LANG, "ui.current_conditions"), tr(LANG, "ui.scenario")]
    left, right = st.columns(2)
    with left:
        _plot(
            figures.shap_waterfall(
                common.drone_contrib(DATA, drone_id),
                META["shap_base_production"][FORECAST],
                FORECAST,
                lang=LANG,
            )
        )
        _explain("drone_shap")
    with right:
        if scenario_values:
            scenario = common.whatif_scenario(base_rows, order, scenario_values)
            fig = figures.conditional_curves(
                model,
                pd.concat([base_rows, scenario], ignore_index=True),
                np.r_[age, age],
                SETTINGS.rul_grid,
                names,
                title,
                dashes=["solid", "dash"],
                lang=LANG,
            )
        else:
            scenario = None
            fig = figures.conditional_curves(
                model,
                base_rows,
                age,
                SETTINGS.rul_grid,
                names[:1],
                title,
                lang=LANG,
            )
        _plot(fig)
        _explain("drone_whatif")

    st.subheader(tr(LANG, "ui.whatif_title"))
    st.caption(bundle(LANG)["whatif_help"])
    _whatif_form(drone_id, order, current, scenario_values)
    if scenario_values and scenario is not None:
        base, new, table = common.whatif_compare(model, base_rows, scenario, SETTINGS, LANG)
        st.table(table.set_index(""))
        _findings(insights.whatif(base, new, changes, SETTINGS.window, LANG))


def score_tab() -> None:
    _intro("score")
    st.subheader(tr(LANG, "ui.prepare_file"))
    st.download_button(
        tr(LANG, "ui.download_template"),
        common.upload_template(DATA).to_csv(index=False),
        file_name="drone_template.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.subheader(tr(LANG, "ui.upload"))
    upload = st.file_uploader(tr(LANG, "ui.csv_file"), type="csv", key="score_file")
    if upload is None:
        return
    try:
        scored, warnings = common.score_upload(
            common.read_csv_bytes(upload.getvalue(), LANG), DATA, SETTINGS, LANG
        )
    except ValueError as exc:
        st.error(f"{upload.name}: {exc}")
        return
    _findings(common.score_summary(scored, SETTINGS, LANG))
    if warnings:
        st.warning("\n".join(f"- {item}" for item in warnings), title=tr(LANG, "ui.data_warnings"))

    st.subheader(tr(LANG, "ui.review_download"))
    columns = common.score_columns(SETTINGS, LANG)
    _table(
        _priority_style(scored[list(columns)]),
        height=min(560, 38 + 35 * len(scored)),
        column_config=_forecast_config(columns),
    )
    st.download_button(
        tr(LANG, "ui.download_results"),
        scored.to_csv(index=False),
        file_name="drone_forecasts.csv",
        mime="text/csv",
        type="primary",
        on_click="ignore",
    )
    _explain("score_table")


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

st.session_state.setdefault("lang", "en")
st.session_state.setdefault("tab_id", "overview")
with st.sidebar:
    st.radio(
        tr(st.session_state["lang"], "ui.language"),
        ["en", "ru"],
        format_func=lambda code: "English" if code == "en" else "Русский",
        key="lang",
    )
LANG = st.session_state["lang"]
tab_labels = [tr(LANG, f"ui.tabs.{key}") for key in TAB_KEYS]
if st.session_state.get("tab") not in tab_labels:
    st.session_state["tab"] = tr(LANG, f"ui.tabs.{st.session_state['tab_id']}")

st.title(tr(LANG, "ui.brand"))
st.caption(_fmt(tr(LANG, "ui.generated")))
for key, container in zip(
    TAB_KEYS, st.tabs(tab_labels, key="tab", on_change="rerun")
):
    if container.open:
        st.session_state["tab_id"] = key
        with container:
            RENDERERS[key]()

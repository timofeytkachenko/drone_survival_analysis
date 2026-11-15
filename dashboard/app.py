"""Dash dashboard for drone failure analysis on the test split.

Run from the project root after executing ``drone_survival.ipynb``::

    uv run python -m dashboard.app
"""

import argparse
import base64
import logging

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from dashboard import common, figures, insights, texts
from dashboard.data import load
from dashboard.labels import FEATURE_LABELS, MODEL_LABELS

logger = logging.getLogger(__name__)

DATA = load()
SETTINGS = common.Settings.from_data(DATA)
META = DATA.meta
PRED = DATA.predictions
TABLES = DATA.tables
MODELS = DATA.models
BEST = SETTINGS.best
FEATURES = SETTINGS.features
CAT_FEATURES = SETTINGS.cat_features
WINDOW = SETTINGS.window
HORIZON = SETTINGS.horizon
ACTIVE = common.active(DATA)
HAZARD = common.hazard_table(DATA, SETTINGS)
TEXT_CONTEXT = common.text_context(DATA, SETTINGS)
GRAPH_CONFIG = {"displaylogo": False}


# --- Building blocks ----------------------------------------------------------------


def _graph(figure=None, graph_id: str | None = None) -> dcc.Graph:
    kwargs = {"id": graph_id} if graph_id else {}
    return dcc.Graph(figure=figure, config=GRAPH_CONFIG, **kwargs)


def _md(text: str, **kwargs) -> dcc.Markdown:
    return dcc.Markdown(text.format(**TEXT_CONTEXT), mathjax=True, **kwargs)


def _findings(items: list[str]) -> dbc.Alert:
    return dbc.Alert(
        [
            html.Strong("What the data shows"),
            dcc.Markdown("\n".join(f"- {s}" for s in items), className="mb-0 mt-1"),
        ],
        color="info",
        className="py-2 mb-2",
    )


def _explain(
    key: str, findings: list[str] | None = None, findings_id: str | None = None
):
    """Explanation card for a figure: what, how to read, findings, method details."""
    text = texts.FIGURES[key]
    children = [
        _md(f"**What is shown.** {text['what']}\n\n**How to read it.** {text['how']}"),
    ]
    if findings_id:
        children.append(html.Div(id=findings_id))
    elif findings:
        children.append(_findings(findings))
    children.append(
        dbc.Accordion(
            dbc.AccordionItem(_md(text["details"]), title="More about the method"),
            start_collapsed=True,
            flush=True,
        )
    )
    return dbc.Card(
        dbc.CardBody(children, className="small"), className="mb-4 bg-light border-0"
    )


def _intro(key: str) -> dbc.Alert:
    return dbc.Alert(
        _md(texts.TAB_INTROS[key], className="mb-0"),
        color="light",
        className="mt-3 border",
    )


def _card(title: str, value: str, subtitle: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.H3(value, className=f"text-{color} mb-0"),
                html.Div(subtitle, className="small"),
            ]
        ),
        className="h-100 shadow-sm",
    )


def _datatable(frame: pd.DataFrame, table_id: str, **kwargs) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dict("records"),
        columns=[{"name": c, "id": c} for c in frame.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "inherit",
            "fontSize": 13,
            "padding": "4px 8px",
            "whiteSpace": "normal",
            "height": "auto",
            "textAlign": "left",
        },
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
        **kwargs,
    )


# --- Tabs -----------------------------------------------------------------------------


def overview_tab() -> dbc.Container:
    overview = insights.overview(PRED)
    return dbc.Container(
        [
            _intro("overview"),
            dbc.Row(
                [
                    dbc.Col(_card(*kpi), sm=6, lg=4, xxl=2, className="mb-3")
                    for kpi in common.kpis(DATA, SETTINGS)
                ],
            ),
            _md(common.KPI_HELP, className="small text-muted"),
            dbc.Alert(
                [
                    html.H5("Key findings"),
                    html.Ul([html.Li(f) for f in common.key_findings(DATA, SETTINGS)]),
                ],
                color="light",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.km_overall(PRED)),
                            _explain("km_overall", overview[:2]),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            _graph(figures.failure_modes(PRED)),
                            _explain("failure_modes", overview[2:]),
                        ],
                        md=5,
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    [
                        _graph(figures.hazard(HAZARD, with_models=False)),
                        _explain("hazard_observed", insights.hazard_trend(HAZARD)),
                    ],
                    md=7,
                )
            ),
        ],
        fluid=True,
    )


def method_tab() -> dbc.Container:
    steps = [
        dbc.AccordionItem(_md(body), title=title, item_id=f"step-{k}")
        for k, (title, body) in enumerate(texts.METHOD_STEPS)
    ]
    return dbc.Container(
        [
            _intro("method"),
            html.H5("Analysis steps"),
            dbc.Accordion(
                steps,
                always_open=True,
                active_item=[f"step-{k}" for k in range(len(steps))],
            ),
            html.H5("Tuned model settings", className="mt-4"),
            _md(
                "Result of automatic tuning (step 6). Values were chosen by the lowest "
                "validation IBS out of {n_trials} trials per model.",
                className="small text-muted",
            ),
            _datatable(common.hyperparams_frame(MODELS), "hyperparams-table"),
            html.H5("Feature dictionary", className="mt-4"),
            _md(
                "What the model knows about each drone. The range is the values in the "
                "training set: outside it the forecast is less reliable, so \"what if\" "
                "scenarios are limited to it. Use the column names in uploaded CSV files.",
                className="small text-muted",
            ),
            _datatable(common.features_frame(META), "features-table"),
            html.H5("Glossary", className="mt-4"),
            dbc.Table(
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(html.Strong(term), style={"width": "22%"}),
                                html.Td(desc),
                            ]
                        )
                        for term, desc in texts.GLOSSARY
                    ]
                ),
                bordered=True,
                size="sm",
                className="small mb-5",
            ),
        ],
        fluid=True,
    )


def segments_tab() -> dbc.Container:
    return dbc.Container(
        [
            _intro("segments"),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Breakdown"),
                        dcc.Dropdown(
                            id="segment",
                            options=[
                                {"label": v, "value": k}
                                for k, v in common.SEGMENTS.items()
                            ],
                            value="environment",
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
            ),
            html.Div(id="segment-findings", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [_graph(graph_id="segment-km"), _explain("segment_km")], md=6
                    ),
                    dbc.Col(
                        [
                            _graph(graph_id="segment-structure"),
                            _explain("segment_structure"),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5("Statistics by segment"),
            html.Div(id="segment-table"),
            html.Div(_explain("segment_table"), className="mt-2"),
        ],
        fluid=True,
    )


def quality_tab() -> dbc.Container:
    sample = PRED.sample(5, random_state=0)
    horizons = META["calibration_horizons"]
    brier_auc, metric_table = TABLES["brier_auc"], TABLES["metrics"]
    return dbc.Container(
        [
            _intro("quality"),
            html.H5("Step 1. Quality metrics"),
            _datatable(common.metrics_frame(metric_table), "metrics-table"),
            html.Div(
                _explain("metrics", insights.metrics(metric_table, BEST)),
                className="mt-2",
            ),
            html.H5("Step 2. How the models trained"),
            _graph(figures.learning_curves(MODELS)),
            _explain("learning_curves", insights.learning(MODELS)),
            html.H5("Step 3. Predicted vs observed"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.survival_mean(TABLES["survival_mean"])),
                            _explain(
                                "survival_mean",
                                insights.mean_vs_km(
                                    TABLES["survival_mean"],
                                    META["eval_times"]["test"][-1],
                                ),
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(
                                figures.individual_curves(
                                    MODELS, sample, SETTINGS.surv_grid
                                )
                            ),
                            _explain("individual_curves"),
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(_graph(figures.hazard(HAZARD, with_models=True)), md=6),
                    dbc.Col(
                        _explain("hazard_mean", insights.hazard_fit(HAZARD)),
                        md=6,
                        className="mt-4",
                    ),
                ]
            ),
            html.H5("Step 4. Accuracy and discrimination over time"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.brier_curve(brier_auc, metric_table)),
                            _explain("brier", insights.brier(brier_auc)),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(figures.auc_curve(brier_auc, metric_table)),
                            _explain("auc", insights.auc(brier_auc)),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5("Step 5. Calibration"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Calibration horizon"),
                            dcc.Dropdown(
                                id="calibration-horizon",
                                options=[
                                    {"label": f"{h:.0f} h", "value": h}
                                    for h in horizons
                                ],
                                value=horizons[0],
                                clearable=False,
                            ),
                            _graph(graph_id="calibration"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        _explain("calibration", findings_id="calibration-findings"),
                        md=6,
                        className="mt-4",
                    ),
                ]
            ),
            html.H5("Step 6. Statistical model comparison (bootstrap)"),
            _datatable(
                common.bootstrap_frame(TABLES["bootstrap_summary"]), "bootstrap-table"
            ),
            html.Div(_explain("bootstrap_table"), className="mt-2"),
            _graph(
                figures.bootstrap(
                    TABLES["bootstrap_samples"], TABLES["bootstrap_summary"]
                )
            ),
            _explain("bootstrap", insights.bootstrap(TABLES["bootstrap_summary"])),
        ],
        fluid=True,
    )


def factors_tab() -> dbc.Container:
    importance = TABLES["shap_importance"]
    top_feature = (
        importance[importance["model"] == BEST].nlargest(1, "share")["feature"].iloc[0]
    )
    return dbc.Container(
        [
            _intro("factors"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Step 1. Which factors matter most"),
                            _graph(figures.shap_importance(importance, BEST)),
                            _explain(
                                "shap_importance", insights.shap_top(importance, BEST)
                            ),
                        ],
                        lg=12,
                        xl=5,
                    ),
                    dbc.Col(
                        [
                            html.H5("Step 2. Which way each factor acts"),
                            dbc.RadioItems(
                                id="shap-model",
                                options=[
                                    {"label": MODEL_LABELS[k], "value": k}
                                    for k in MODELS
                                ],
                                value=BEST,
                                inline=True,
                            ),
                            _graph(graph_id="shap-beeswarm"),
                            _explain("shap_beeswarm", findings_id="beeswarm-findings"),
                        ],
                        lg=12,
                        xl=7,
                    ),
                ],
            ),
            html.H5("Step 3. How exactly risk depends on a factor"),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Feature"),
                        dcc.Dropdown(
                            id="shap-feature",
                            options=[
                                {"label": FEATURE_LABELS[f], "value": f}
                                for f in FEATURES
                            ],
                            value=top_feature,
                            clearable=False,
                        ),
                        html.Div(
                            id="feature-description", className="small text-muted mt-1"
                        ),
                    ],
                    md=6,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(_graph(graph_id="shap-dependence"), md=7),
                    dbc.Col(
                        _explain("shap_dependence", findings_id="dependence-findings"),
                        md=5,
                        className="mt-3",
                    ),
                ]
            ),
        ],
        fluid=True,
    )


def _forecast_table(
    frame: pd.DataFrame, table_id: str, headers: dict[str, str], **kwargs
) -> dash_table.DataTable:
    """Sortable, filterable forecast table coloured by maintenance priority."""
    numeric = {"age_h", "p_fail_window", "rul_q10", "rul_median", "rmrl"}
    columns = [
        {
            "name": name,
            "id": col,
            **(
                {
                    "type": "numeric",
                    "format": Format(
                        precision=3 if col == "p_fail_window" else 0,
                        scheme=Scheme.fixed,
                    ),
                }
                if col in numeric
                else {}
            ),
        }
        for col, name in headers.items()
    ]
    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dict("records"),
        columns=columns,
        filter_action="native",
        sort_action="native",
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "4px 8px"},
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f1f3f5",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_data_conditional=[
            {"if": {"filter_query": f'{{priority}} = "{p}"'}, "backgroundColor": color}
            for p, color in common.PRIORITY_COLORS.items()
        ],
        tooltip_header=common.fleet_tooltips(SETTINGS),
        **kwargs,
    )


def _fleet_table() -> dash_table.DataTable:
    frame = common.fleet_frame(DATA, SETTINGS)
    frame["id"] = frame["drone_id"]
    return _forecast_table(
        frame, "fleet-table", common.fleet_columns(SETTINGS), row_selectable="single"
    )


def fleet_tab() -> dbc.Container:
    riskiest = ACTIVE.nlargest(8, "p_fail_window")
    return dbc.Container(
        [
            _intro("fleet"),
            _findings(insights.fleet(ACTIVE, WINDOW, HORIZON)),
            html.H5("Step 1. Drones in service", className="mt-3"),
            _fleet_table(),
            html.Div(_explain("fleet_table"), className="mt-2"),
            html.H5("Step 2. How much life is left"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.rul_histogram(ACTIVE, HORIZON)),
                            _explain("rul_hist"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(
                                figures.conditional_curves(
                                    DATA.best_model,
                                    riskiest,
                                    riskiest["duration_h"].to_numpy(),
                                    SETTINGS.rul_grid,
                                    riskiest["drone_id"].tolist(),
                                    "Remaining life of the riskiest drones",
                                )
                            ),
                            _explain("riskiest"),
                        ],
                        md=6,
                    ),
                ],
            ),
            html.H5("Step 3. Can the remaining-life forecast be trusted"),
            _graph(figures.backtest(TABLES["backtest"], META["landmark"])),
            _explain(
                "backtest", insights.backtest(TABLES["backtest"], META["landmark"])
            ),
        ],
        fluid=True,
    )


def _whatif_input(feature: str) -> dbc.Col:
    if feature == "is_refurbished":
        control = dcc.Dropdown(
            id={"type": "whatif", "feature": feature},
            options=[
                {"label": "Yes", "value": "True"},
                {"label": "No", "value": "False"},
            ],
            clearable=False,
        )
    elif feature in CAT_FEATURES:
        control = dcc.Dropdown(
            id={"type": "whatif", "feature": feature},
            options=[
                {"label": str(c), "value": c} for c in META["categories"][feature]
            ],
            clearable=False,
        )
    else:
        low, high = META["numeric_ranges"][feature]
        control = dbc.Input(
            id={"type": "whatif", "feature": feature},
            type="number",
            min=low,
            max=high,
            step="any",
            debounce=True,
        )
    return dbc.Col(
        [dbc.Label(FEATURE_LABELS[feature], className="small mb-0"), control],
        md=3,
        className="mb-2",
    )


def drone_tab() -> dbc.Container:
    options = [
        {"label": label, "value": drone_id}
        for drone_id, label in common.drone_options(PRED).items()
    ]
    return dbc.Container(
        [
            _intro("drone"),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Drone"),
                        dcc.Dropdown(
                            id="drone-select",
                            options=options,
                            value=options[0]["value"],
                            clearable=False,
                        ),
                    ],
                    md=6,
                ),
            ),
            html.Div(id="drone-findings", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="drone-info"), md=4),
                    dbc.Col(
                        [
                            _graph(graph_id="drone-survival"),
                            _explain("drone_survival"),
                            _graph(graph_id="drone-hazard"),
                            _explain("drone_hazard"),
                        ],
                        md=8,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [_graph(graph_id="drone-shap"), _explain("drone_shap")], md=6
                    ),
                    dbc.Col(
                        [
                            _graph(graph_id="drone-whatif-graph"),
                            _explain("drone_whatif"),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5("\"What if\" scenario"),
            _md(texts.WHATIF_HELP, className="small text-muted"),
            dbc.Row([_whatif_input(f) for f in FEATURES]),
            dbc.Button(
                "Recalculate", id="whatif-run", color="primary", className="my-2"
            ),
            html.Div(id="whatif-summary", className="mb-5"),
        ],
        fluid=True,
    )


def score_tab() -> dbc.Container:
    return dbc.Container(
        [
            _intro("score"),
            html.H5("Step 1. Prepare the file"),
            dbc.Button(
                "Download template",
                id="score-template-button",
                color="secondary",
                outline=True,
            ),
            dcc.Download(id="score-template-download"),
            html.H5("Step 2. Upload it", className="mt-4"),
            dcc.Upload(
                id="score-upload",
                children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
                accept=".csv",
                multiple=False,
                className="border border-2 rounded text-center py-4 bg-light",
                style={"borderStyle": "dashed"},
            ),
            html.Div(id="score-status", className="mt-3"),
            html.H5("Step 3. Review and download the forecast", className="mt-2"),
            html.Div(id="score-results"),
            dbc.Button(
                "Download results (CSV)",
                id="score-download-button",
                color="primary",
                className="my-2",
                disabled=True,
            ),
            dcc.Download(id="score-download"),
            dcc.Store(id="score-store"),
            html.Div(_explain("score_table"), className="mt-2 mb-5"),
        ],
        fluid=True,
    )


# --- App ----------------------------------------------------------------------------


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Drone failures")
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Drone failure analysis: test set",
            color="primary",
            dark=True,
            fluid=True,
            className="mb-2",
        ),
        dbc.Tabs(
            id="tabs",
            active_tab="overview",
            children=[
                dbc.Tab(overview_tab(), label="Overview", tab_id="overview"),
                dbc.Tab(method_tab(), label="Methodology", tab_id="method"),
                dbc.Tab(segments_tab(), label="Failure segments", tab_id="segments"),
                dbc.Tab(quality_tab(), label="Model quality", tab_id="quality"),
                dbc.Tab(factors_tab(), label="Risk factors", tab_id="factors"),
                dbc.Tab(fleet_tab(), label="Fleet", tab_id="fleet"),
                dbc.Tab(drone_tab(), label="Drone card", tab_id="drone"),
                dbc.Tab(score_tab(), label="Score new drones", tab_id="score"),
            ],
        ),
    ],
    fluid=True,
    className="px-0",
)


@app.callback(
    Output("segment-km", "figure"),
    Output("segment-structure", "figure"),
    Output("segment-table", "children"),
    Output("segment-findings", "children"),
    Input("segment", "value"),
)
def update_segments(segment: str):
    order = common.SEGMENT_ORDER.get(segment)
    levels = common.segment_levels(PRED, segment)
    return (
        figures.segment_km(PRED, segment, order),
        figures.segment_failure_structure(PRED, segment, order),
        _datatable(
            common.segment_stats(PRED, segment, levels, WINDOW), "segment-stats"
        ),
        _findings(common.segment_findings(PRED, segment, levels, WINDOW)),
    )


@app.callback(
    Output("calibration", "figure"),
    Output("calibration-findings", "children"),
    Input("calibration-horizon", "value"),
)
def update_calibration(horizon: float):
    return (
        figures.calibration(TABLES["calibration"], horizon),
        _findings(insights.calibration(TABLES["calibration"], horizon)),
    )


@app.callback(
    Output("shap-beeswarm", "figure"),
    Output("beeswarm-findings", "children"),
    Input("shap-model", "value"),
)
def update_beeswarm(kind: str):
    figure = figures.shap_beeswarm(
        TABLES["shap_values"], TABLES["shap_importance"], kind
    )
    return figure, _findings(common.beeswarm_findings(DATA, SETTINGS, kind))


@app.callback(
    Output("shap-dependence", "figure"),
    Output("dependence-findings", "children"),
    Output("feature-description", "children"),
    Input("shap-model", "value"),
    Input("shap-feature", "value"),
)
def update_dependence(kind: str, feature: str):
    categorical = feature in CAT_FEATURES
    shap_values = TABLES["shap_values"]
    return (
        figures.shap_dependence(shap_values, kind, feature, categorical=categorical),
        _findings(insights.shap_direction(shap_values, kind, feature, categorical)),
        texts.FEATURE_DESCRIPTIONS[feature],
    )


@app.callback(
    Output("drone-select", "value"),
    Output("tabs", "active_tab"),
    Input("fleet-table", "selected_row_ids"),
    prevent_initial_call=True,
)
def open_drone(selected: list[str] | None):
    if not selected:
        raise PreventUpdate
    return selected[0], "drone"


@app.callback(
    Output("drone-info", "children"),
    Output("drone-survival", "figure"),
    Output("drone-hazard", "figure"),
    Output("drone-shap", "figure"),
    Output("drone-findings", "children"),
    Output({"type": "whatif", "feature": ALL}, "value"),
    Input("drone-select", "value"),
)
def update_drone(drone_id: str):
    rows = common.drone_rows(PRED, drone_id)
    row = rows.iloc[0]
    failed = bool(row["event"])
    info = [html.H4(drone_id), html.P(common.drone_status(row), className="fw-bold")]
    if not failed:
        info.append(
            dbc.Table(
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(name),
                                html.Td(html.Strong(value)),
                                html.Td(hint, className="text-muted small"),
                            ]
                        )
                        for name, value, hint in common.drone_metrics(row, SETTINGS)
                    ]
                ),
                size="sm",
                bordered=True,
            )
        )
    info.append(html.P(f"Risk group: {row['risk_group']}", className="mb-1"))
    info.append(
        dbc.Table(
            html.Tbody(
                [
                    html.Tr([html.Td(name), html.Td(value)])
                    for name, value in common.drone_features(row, FEATURES).itertuples(
                        index=False
                    )
                ]
            ),
            size="sm",
            striped=True,
            className="small",
        )
    )
    return (
        info,
        figures.drone_survival(
            MODELS, rows, SETTINGS.surv_grid, row["duration_h"], failed
        ),
        figures.drone_hazard(
            MODELS,
            rows,
            common.drone_hazard_grid(SETTINGS, row["duration_h"]),
            SETTINGS.hazard_bandwidth,
            row["duration_h"],
            failed,
        ),
        figures.shap_waterfall(
            common.drone_contrib(DATA, drone_id), META["shap_base"][BEST], BEST
        ),
        _findings(common.drone_findings(DATA, SETTINGS, drone_id)),
        common.whatif_defaults(row, FEATURES),
    )


@app.callback(
    Output("drone-whatif-graph", "figure"),
    Output("whatif-summary", "children"),
    Input("drone-select", "value"),
    Input("whatif-run", "n_clicks"),
    State({"type": "whatif", "feature": ALL}, "value"),
)
def update_whatif(drone_id: str, _clicks: int | None, values: list):
    rows = common.whatif_base(PRED, drone_id)
    age = rows["duration_h"].to_numpy()
    model = DATA.best_model
    title = f"Remaining life from {age[0]:.0f} flight hours ({MODEL_LABELS[BEST]})"
    baseline = figures.conditional_curves(
        model, rows, age, SETTINGS.rul_grid, ["Current conditions"], title
    )
    if ctx.triggered_id != "whatif-run":
        return baseline, None
    if any(v is None or v == "" for v in values):
        return baseline, dbc.Alert(
            "Fill in all fields with values from the training range.",
            color="warning",
        )

    scenario = common.whatif_scenario(rows, FEATURES, values)
    fig = figures.conditional_curves(
        model,
        pd.concat([rows, scenario], ignore_index=True),
        np.r_[age, age],
        SETTINGS.rul_grid,
        ["Current conditions", "Scenario"],
        title,
        dashes=["solid", "dash"],
    )
    base, new, table = common.whatif_compare(model, rows, scenario, SETTINGS)
    changes = common.whatif_changes(rows.iloc[0], FEATURES, CAT_FEATURES, values)
    comparison = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(c) for c in table.columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(v) for v in r])
                    for r in table.itertuples(index=False)
                ]
            ),
        ],
        bordered=True,
        size="sm",
        style={"maxWidth": "600px"},
    )
    return fig, [comparison, _findings(insights.whatif(base, new, changes, WINDOW))]


@app.callback(
    Output("score-template-download", "data"),
    Input("score-template-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_template(_clicks: int):
    return dcc.send_data_frame(
        common.upload_template(DATA).to_csv, "drone_template.csv", index=False
    )


@app.callback(
    Output("score-status", "children"),
    Output("score-results", "children"),
    Output("score-store", "data"),
    Output("score-download-button", "disabled"),
    Input("score-upload", "contents"),
    State("score-upload", "filename"),
    prevent_initial_call=True,
)
def score_file(contents: str | None, filename: str | None):
    if contents is None:
        raise PreventUpdate
    try:
        content = base64.b64decode(contents.split(",", 1)[1])
        scored, warnings = common.score_upload(
            common.read_csv_bytes(content), DATA, SETTINGS
        )
    except ValueError as exc:
        alert = dbc.Alert([html.Strong(f"{filename}: "), str(exc)], color="danger")
        return alert, None, None, True
    status = [_findings(common.score_summary(scored, SETTINGS))]
    if warnings:
        status.append(
            dbc.Alert(
                [
                    html.Strong("Data warnings"),
                    html.Ul([html.Li(w) for w in warnings], className="mb-0"),
                ],
                color="warning",
                className="small",
            )
        )
    display = scored.assign(out_of_range=scored["out_of_range"].map({True: "yes", False: ""}))
    table = _forecast_table(display, "score-table", common.score_columns(SETTINGS))
    return status, table, scored.to_csv(index=False), False


@app.callback(
    Output("score-download", "data"),
    Input("score-download-button", "n_clicks"),
    State("score-store", "data"),
    prevent_initial_call=True,
)
def download_scores(_clicks: int, csv: str | None):
    if not csv:
        raise PreventUpdate
    return dcc.send_string(csv, "drone_forecasts.csv")


def main() -> None:
    """Run the dashboard server."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

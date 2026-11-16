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

from dashboard import common, figures, insights
from dashboard.data import load
from dashboard.locale import bundle, tr

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
ACTIVE = common.active(DATA)
HAZARD = common.hazard_table(DATA, SETTINGS)
DISAGREE = common.disagreeing_drones(DATA, SETTINGS)
GRAPH_CONFIG = {"displaylogo": False}
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


def _md(text: str, lang: str, **kwargs) -> dcc.Markdown:
    return dcc.Markdown(
        text.format(**common.text_context(DATA, SETTINGS, lang)), mathjax=True, **kwargs
    )


def _findings(items: list[str], lang: str) -> dbc.Alert:
    return dbc.Alert(
        [
            html.Strong(tr(lang, "ui.what_data_shows")),
            dcc.Markdown("\n".join(f"- {item}" for item in items), className="mb-0 mt-1"),
        ],
        color="info",
        className="py-2 mb-2",
    )


def _explain(
    key: str,
    lang: str,
    findings: list[str] | None = None,
    findings_id: str | None = None,
):
    """Explanation card for a figure: what, how to read, findings, method details."""
    text = bundle(lang)["figures"][key]
    children = [
        _md(
            f"**{tr(lang, 'ui.what_shown')}** {text['what']}\n\n"
            f"**{tr(lang, 'ui.how_to_read')}** {text['how']}",
            lang,
        ),
    ]
    if findings_id:
        children.append(html.Div(id=findings_id))
    elif findings:
        children.append(_findings(findings, lang))
    children.append(
        dbc.Accordion(
            dbc.AccordionItem(
                _md(text["details"], lang), title=tr(lang, "ui.more_method")
            ),
            start_collapsed=True,
            flush=True,
        )
    )
    return dbc.Card(
        dbc.CardBody(children, className="small"), className="mb-4 bg-light border-0"
    )


def _intro(key: str, lang: str) -> dbc.Alert:
    return dbc.Alert(
        _md(bundle(lang)["tab_intros"][key], lang, className="mb-0"),
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


def _graph(figure=None, graph_id: str | None = None) -> dcc.Graph:
    kwargs = {"id": graph_id} if graph_id else {}
    return dcc.Graph(figure=figure, config=GRAPH_CONFIG, **kwargs)


def overview_tab(lang: str) -> dbc.Container:
    overview = insights.overview(PRED, lang)
    return dbc.Container(
        [
            _intro("overview", lang),
            dbc.Row(
                [
                    dbc.Col(_card(*kpi), sm=6, lg=3, className="mb-3")
                    for kpi in common.kpis(DATA, SETTINGS, lang)
                ],
            ),
            _md(tr(lang, "ui.priority_table", p_medium=SETTINGS.p_medium), lang, className="small text-muted"),
            _datatable(
                common.priority_sensitivity(ACTIVE, SETTINGS, lang), "priority-table"
            ),
            dbc.Alert(
                [
                    html.H5(tr(lang, "ui.key_findings"), className="mt-3"),
                    html.Ul(
                        [html.Li(item) for item in common.key_findings(DATA, SETTINGS, lang)]
                    ),
                ],
                color="light",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.km_overall(PRED, lang)),
                            _explain("km_overall", lang, overview[:2]),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(figures.hazard(HAZARD, [BEST], lang)),
                            _explain(
                                "hazard_overview", lang, insights.hazard_trend(HAZARD, lang)
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.failure_modes(PRED, lang)),
                            _explain("failure_modes", lang, overview[2:]),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(figures.failure_mode_timing(PRED, lang)),
                            _explain("failure_timing", lang),
                        ],
                        md=6,
                    ),
                ]
            ),
        ],
        fluid=True,
    )


def method_tab(lang: str) -> dbc.Container:
    catalog = bundle(lang)
    steps = [
        dbc.AccordionItem(_md(body, lang), title=title, item_id=f"step-{k}")
        for k, (title, body) in enumerate(catalog["method_steps"])
    ]
    return dbc.Container(
        [
            _intro("method", lang),
            html.H5(tr(lang, "ui.analysis_steps")),
            dbc.Accordion(
                steps,
                always_open=True,
                active_item=[f"step-{k}" for k in range(len(steps))],
            ),
            html.H5(tr(lang, "ui.tuned_settings"), className="mt-4"),
            _md(tr(lang, "ui.tuned_caption", n_trials=META["n_trials"]), lang, className="small text-muted"),
            _datatable(common.hyperparams_frame(MODELS, lang), "hyperparams-table"),
            html.H5(tr(lang, "ui.feature_dictionary"), className="mt-4"),
            _md(tr(lang, "ui.feature_dictionary_caption"), lang, className="small text-muted"),
            _datatable(common.features_frame(META, lang), "features-table"),
            html.H5(tr(lang, "ui.glossary"), className="mt-4"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(tr(lang, "ui.term")),
                                html.Th(tr(lang, "ui.definition")),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(html.Strong(term), style={"width": "22%"}),
                                    html.Td(desc),
                                ]
                            )
                            for term, desc in catalog["glossary"]
                        ]
                    ),
                ],
                bordered=True,
                size="sm",
                className="small mb-5",
            ),
        ],
        fluid=True,
    )


def segments_tab(lang: str) -> dbc.Container:
    options = common.segment_options(DATA, SETTINGS, lang)
    return dbc.Container(
        [
            _intro("segments", lang),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label(tr(lang, "ui.breakdown")),
                        dcc.Dropdown(
                            id="segment",
                            options=[{"label": label, "value": key} for key, label in options.items()],
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
                        [_graph(graph_id="segment-km"), _explain("segment_km", lang)], md=6
                    ),
                    dbc.Col(
                        [
                            _graph(graph_id="segment-structure"),
                            _explain("segment_structure", lang),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5(tr(lang, "ui.statistics_by_segment")),
            html.Div(id="segment-table"),
            html.Div(_explain("segment_table", lang), className="mt-2"),
        ],
        fluid=True,
    )


def quality_tab(lang: str) -> dbc.Container:
    horizons = META["calibration_horizons"]
    brier_auc, metric_table = TABLES["brier_auc"], TABLES["metrics"]
    return dbc.Container(
        [
            _intro("quality", lang),
            html.H5(tr(lang, "ui.step_metrics")),
            _datatable(common.metrics_frame(metric_table, lang), "metrics-table"),
            html.Div(
                _explain("metrics", lang, insights.metrics(metric_table, BEST, lang)),
                className="mt-2",
            ),
            html.H5(tr(lang, "ui.step_training")),
            _graph(figures.learning_curves(MODELS, lang)),
            _explain("learning_curves", lang, insights.learning(MODELS, lang)),
            html.H5(tr(lang, "ui.step_predicted")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.survival_mean(TABLES["survival_mean"], lang)),
                            _explain(
                                "survival_mean",
                                lang,
                                insights.mean_vs_km(
                                    TABLES["survival_mean"],
                                    META["eval_times"]["test"][-1],
                                    lang,
                                ),
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(
                                figures.individual_curves(
                                    MODELS, DISAGREE, SETTINGS.surv_grid, lang
                                )
                            ),
                            _explain("individual_curves", lang),
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _graph(figures.hazard(HAZARD, list(MODELS), lang)), md=6
                    ),
                    dbc.Col(
                        _explain("hazard_mean", lang, insights.hazard_fit(HAZARD, lang)),
                        md=6,
                        className="mt-4",
                    ),
                ]
            ),
            html.H5(tr(lang, "ui.step_accuracy")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.brier_curve(brier_auc, metric_table, lang)),
                            _explain("brier", lang, insights.brier(brier_auc, lang)),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(figures.auc_curve(brier_auc, metric_table, lang)),
                            _explain("auc", lang, insights.auc(brier_auc, lang)),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5(tr(lang, "ui.step_calibration")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(tr(lang, "ui.calibration_horizon")),
                            dcc.Dropdown(
                                id="calibration-horizon",
                                options=[
                                    {
                                        "label": tr(lang, "ui.horizon_hours", h=hours),
                                        "value": hours,
                                    }
                                    for hours in horizons
                                ],
                                value=horizons[0],
                                clearable=False,
                            ),
                            _graph(graph_id="calibration"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        _explain("calibration", lang, findings_id="calibration-findings"),
                        md=6,
                        className="mt-4",
                    ),
                ]
            ),
            html.H5(tr(lang, "ui.step_bootstrap")),
            _datatable(
                common.bootstrap_frame(TABLES["bootstrap_summary"], lang),
                "bootstrap-table",
            ),
            html.Div(_explain("bootstrap_table", lang), className="mt-2"),
            _graph(
                figures.bootstrap(
                    TABLES["bootstrap_samples"], TABLES["bootstrap_summary"], lang
                )
            ),
            _explain(
                "bootstrap", lang, insights.bootstrap(TABLES["bootstrap_summary"], lang)
            ),
        ],
        fluid=True,
    )


def factors_tab(lang: str) -> dbc.Container:
    importance = TABLES["shap_importance"]
    labels = bundle(lang)
    top_feature = (
        importance[importance["model"] == BEST].nlargest(1, "share")["feature"].iloc[0]
    )
    return dbc.Container(
        [
            _intro("factors", lang),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5(tr(lang, "ui.step_importance")),
                            _graph(figures.shap_importance(importance, BEST, lang)),
                            _explain(
                                "shap_importance",
                                lang,
                                insights.shap_top(importance, BEST, lang=lang),
                            ),
                        ],
                        lg=12,
                        xl=5,
                    ),
                    dbc.Col(
                        [
                            html.H5(tr(lang, "ui.step_direction")),
                            dbc.RadioItems(
                                id="shap-model",
                                options=[
                                    {"label": labels["model_labels"][kind], "value": kind}
                                    for kind in MODELS
                                ],
                                value=BEST,
                                inline=True,
                            ),
                            _graph(graph_id="shap-beeswarm"),
                            _explain("shap_beeswarm", lang, findings_id="beeswarm-findings"),
                        ],
                        lg=12,
                        xl=7,
                    ),
                ],
            ),
            html.H5(tr(lang, "ui.step_dependence")),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label(tr(lang, "ui.feature")),
                        dcc.Dropdown(
                            id="shap-feature",
                            options=[
                                {"label": labels["feature_labels"][feature], "value": feature}
                                for feature in FEATURES
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
                        _explain(
                            "shap_dependence", lang, findings_id="dependence-findings"
                        ),
                        md=5,
                        className="mt-3",
                    ),
                ]
            ),
        ],
        fluid=True,
    )


def _forecast_table(
    frame: pd.DataFrame, table_id: str, headers: dict[str, str], lang: str, **kwargs
) -> dash_table.DataTable:
    """Sortable, filterable forecast table coloured by the stored priority code."""
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
            {"if": {"filter_query": f'{{priority}} = "{code}"'}, "backgroundColor": color}
            for code, color in common.PRIORITY_COLORS.items()
        ],
        tooltip_header=common.fleet_tooltips(SETTINGS, lang),
        **kwargs,
    )


def fleet_tab(lang: str) -> dbc.Container:
    riskiest = ACTIVE.nlargest(8, "p_fail_window")
    frame = common.fleet_frame(DATA, SETTINGS, lang)
    frame["id"] = frame["drone_id"]
    return dbc.Container(
        [
            _intro("fleet", lang),
            _findings(insights.fleet(ACTIVE, WINDOW, SETTINGS.horizon, lang), lang),
            html.H5(tr(lang, "ui.step_fleet"), className="mt-3"),
            _forecast_table(
                frame,
                "fleet-table",
                common.fleet_columns(SETTINGS, lang),
                lang,
                row_selectable="single",
            ),
            dbc.Button(
                tr(lang, "ui.download_fleet"),
                id="fleet-download-button",
                color="secondary",
                outline=True,
                className="mt-2",
            ),
            dcc.Download(id="fleet-download"),
            html.Div(_explain("fleet_table", lang), className="mt-2"),
            _graph(figures.age_risk(ACTIVE, WINDOW, lang)),
            _explain("age_risk", lang),
            html.H5(tr(lang, "ui.step_life")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.rul_histogram(ACTIVE, SETTINGS.horizon, lang)),
                            _explain("rul_hist", lang),
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
                                    tr(lang, "ui.riskiest_title"),
                                    lang=lang,
                                )
                            ),
                            _explain("riskiest", lang),
                        ],
                        md=6,
                    ),
                ],
            ),
            html.H5(tr(lang, "ui.step_trust")),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            _graph(figures.backtest(TABLES["backtest"], META["landmark"], lang)),
                            _explain(
                                "backtest",
                                lang,
                                insights.backtest(TABLES["backtest"], META["landmark"], lang),
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            _graph(
                                figures.backtest_groups(
                                    TABLES["backtest_groups"],
                                    BEST,
                                    WINDOW,
                                    META["landmark"],
                                    lang,
                                )
                            ),
                            _explain(
                                "backtest_groups",
                                lang,
                                insights.backtest_groups(TABLES["backtest_groups"], BEST, lang),
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
        ],
        fluid=True,
    )


def _whatif_input(feature: str, lang: str) -> dbc.Col:
    if feature == "is_refurbished":
        control = dcc.Dropdown(
            id={"type": "whatif", "feature": feature},
            options=[
                {"label": tr(lang, "ui.yes"), "value": "True"},
                {"label": tr(lang, "ui.no"), "value": "False"},
            ],
            clearable=False,
        )
    elif feature in CAT_FEATURES:
        control = dcc.Dropdown(
            id={"type": "whatif", "feature": feature},
            options=[{"label": str(c), "value": c} for c in META["categories"][feature]],
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
        [
            dbc.Label(bundle(lang)["feature_labels"][feature], className="small mb-0"),
            control,
        ],
        md=3,
        className="mb-2",
    )


def drone_tab(lang: str) -> dbc.Container:
    options = [
        {"label": label, "value": drone_id}
        for drone_id, label in common.drone_options(PRED, lang).items()
    ]
    order = common.whatif_order(FEATURES)
    levers = [feature for feature in order if feature in common.LEVERS]
    fixed = [feature for feature in order if feature not in common.LEVERS]
    return dbc.Container(
        [
            _intro("drone", lang),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label(tr(lang, "ui.drone")),
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
                            _explain("drone_survival", lang),
                            _graph(graph_id="drone-hazard"),
                            _explain("drone_hazard", lang),
                        ],
                        md=8,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [_graph(graph_id="drone-shap"), _explain("drone_shap", lang)], md=6
                    ),
                    dbc.Col(
                        [
                            _graph(graph_id="drone-whatif-graph"),
                            _explain("drone_whatif", lang),
                        ],
                        md=6,
                    ),
                ]
            ),
            html.H5(tr(lang, "ui.whatif_title")),
            _md(bundle(lang)["whatif_help"], lang, className="small text-muted"),
            html.H6(tr(lang, "ui.whatif_levers"), className="mt-2"),
            dbc.Row([_whatif_input(feature, lang) for feature in levers]),
            html.H6(tr(lang, "ui.whatif_fixed"), className="mt-2"),
            dbc.Row([_whatif_input(feature, lang) for feature in fixed]),
            dbc.Button(
                tr(lang, "ui.recalculate"), id="whatif-run", color="primary", className="my-2"
            ),
            html.Div(id="whatif-summary", className="mb-5"),
        ],
        fluid=True,
    )


def score_tab(lang: str) -> dbc.Container:
    return dbc.Container(
        [
            _intro("score", lang),
            html.H5(tr(lang, "ui.prepare_file")),
            dbc.Button(
                tr(lang, "ui.download_template"),
                id="score-template-button",
                color="secondary",
                outline=True,
            ),
            dcc.Download(id="score-template-download"),
            html.H5(tr(lang, "ui.upload"), className="mt-4"),
            dcc.Upload(
                id="score-upload",
                children=html.Div(tr(lang, "ui.upload_hint")),
                accept=".csv",
                multiple=False,
                className="border border-2 rounded text-center py-4 bg-light",
                style={"borderStyle": "dashed"},
            ),
            html.Div(id="score-status", className="mt-3"),
            html.H5(tr(lang, "ui.review_download"), className="mt-2"),
            html.Div(id="score-results"),
            dbc.Button(
                tr(lang, "ui.download_results"),
                id="score-download-button",
                color="primary",
                className="my-2",
                disabled=True,
            ),
            dcc.Download(id="score-download"),
            dcc.Store(id="score-store"),
            html.Div(_explain("score_table", lang), className="mt-2 mb-5"),
        ],
        fluid=True,
    )


def _page(lang: str, tab: str | None) -> dbc.Tabs:
    builders = {
        "overview": overview_tab,
        "method": method_tab,
        "segments": segments_tab,
        "quality": quality_tab,
        "factors": factors_tab,
        "fleet": fleet_tab,
        "drone": drone_tab,
        "score": score_tab,
    }
    active = tab if tab in builders else "overview"
    return dbc.Tabs(
        id="tabs",
        active_tab=active,
        children=[
            dbc.Tab(builders[key](lang), label=tr(lang, f"ui.tabs.{key}"), tab_id=key)
            for key in TAB_KEYS
        ],
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Drone failures",
    suppress_callback_exceptions=True,
)
app.layout = dbc.Container(
    [
        dcc.Store(id="lang", data="en"),
        dcc.Store(id="active-tab", data="overview"),
        dbc.Navbar(
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Span(
                                tr("en", "ui.brand"),
                                id="brand",
                                className="navbar-brand mb-0 h1",
                            ),
                            html.Div(
                                tr(
                                    "en",
                                    "ui.generated",
                                    generated_at=META["generated_at"].replace("T", " "),
                                ),
                                id="generated",
                                className="small",
                            ),
                        ]
                    ),
                    dcc.RadioItems(
                        id="lang-radio",
                        options=[
                            {"label": "English", "value": "en"},
                            {"label": "Русский", "value": "ru"},
                        ],
                        value="en",
                        inline=True,
                        labelStyle={"color": "white", "marginLeft": "12px"},
                        inputStyle={"marginRight": "6px"},
                    ),
                ],
                fluid=True,
                className="d-flex justify-content-between align-items-center",
            ),
            color="primary",
            dark=True,
            className="mb-2",
        ),
        html.Div(id="page"),
    ],
    fluid=True,
    className="px-0",
)


@app.callback(
    Output("lang", "data"),
    Output("brand", "children"),
    Output("generated", "children"),
    Output("page", "children"),
    Input("lang-radio", "value"),
    State("active-tab", "data"),
)
def render_page(lang: str, tab: str | None):
    """Rebuild the translated shell. The active tab comes from the store."""
    return (
        lang,
        tr(lang, "ui.brand"),
        tr(lang, "ui.generated", generated_at=META["generated_at"].replace("T", " ")),
        _page(lang, tab),
    )


@app.callback(
    Output("active-tab", "data"),
    Input("tabs", "active_tab"),
    prevent_initial_call=True,
)
def remember_tab(tab: str | None):
    """Keep the open tab when the language callback rebuilds the page."""
    if not tab:
        raise PreventUpdate
    return tab


@app.callback(
    Output("segment-km", "figure"),
    Output("segment-structure", "figure"),
    Output("segment-table", "children"),
    Output("segment-findings", "children"),
    Input("segment", "value"),
    State("lang", "data"),
)
def update_segments(segment: str, lang: str):
    frame, column, levels = common.prepare_segment(PRED, segment)
    axis = common.segment_options(DATA, SETTINGS, lang)[segment]
    return (
        figures.segment_km(frame, column, levels, lang, segment),
        figures.segment_failure_structure(frame, column, levels, lang, segment),
        _datatable(
            common.segment_stats(frame, column, levels, WINDOW, lang, axis),
            "segment-stats",
        ),
        _findings(common.segment_findings(frame, column, levels, WINDOW, lang), lang),
    )


@app.callback(
    Output("calibration", "figure"),
    Output("calibration-findings", "children"),
    Input("calibration-horizon", "value"),
    State("lang", "data"),
)
def update_calibration(horizon: float, lang: str):
    return (
        figures.calibration(TABLES["calibration"], horizon, lang),
        _findings(insights.calibration(TABLES["calibration"], horizon, lang), lang),
    )


@app.callback(
    Output("shap-beeswarm", "figure"),
    Output("beeswarm-findings", "children"),
    Input("shap-model", "value"),
    State("lang", "data"),
)
def update_beeswarm(kind: str, lang: str):
    figure = figures.shap_beeswarm(
        TABLES["shap_values"], TABLES["shap_importance"], kind, lang
    )
    return figure, _findings(common.beeswarm_findings(DATA, SETTINGS, kind, lang), lang)


@app.callback(
    Output("shap-dependence", "figure"),
    Output("dependence-findings", "children"),
    Output("feature-description", "children"),
    Input("shap-model", "value"),
    Input("shap-feature", "value"),
    State("lang", "data"),
)
def update_dependence(kind: str, feature: str, lang: str):
    categorical = feature in CAT_FEATURES
    shap_values = TABLES["shap_values"]
    return (
        figures.shap_dependence(shap_values, kind, feature, categorical, lang),
        _findings(
            insights.shap_direction(shap_values, kind, feature, categorical, lang), lang
        ),
        bundle(lang)["feature_descriptions"][feature],
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
    State("lang", "data"),
)
def update_drone(drone_id: str, lang: str):
    rows = common.drone_rows(PRED, drone_id)
    row = rows.iloc[0]
    failed = bool(row["event"])
    labels = bundle(lang)
    info = [html.H4(drone_id), html.P(common.drone_status(row, lang), className="fw-bold")]
    if row["in_service"]:
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
                        for name, value, hint in common.drone_metrics(row, SETTINGS, lang)
                    ]
                ),
                size="sm",
                bordered=True,
            )
        )
    group = labels["priority_labels"].get(row["risk_group"], row["risk_group"])
    info.append(html.P(tr(lang, "ui.risk_group_line", group=group), className="mb-1"))
    info.append(
        dbc.Table(
            html.Tbody(
                [
                    html.Tr([html.Td(name), html.Td(value)])
                    for name, value in common.drone_features(
                        row, FEATURES, lang
                    ).itertuples(index=False)
                ]
            ),
            size="sm",
            striped=True,
            className="small",
        )
    )
    order = common.whatif_order(FEATURES)
    return (
        info,
        figures.drone_survival(
            MODELS, rows, SETTINGS.surv_grid, row["duration_h"], failed, lang
        ),
        figures.drone_hazard(
            MODELS,
            rows,
            common.drone_hazard_grid(SETTINGS, row["duration_h"]),
            SETTINGS.hazard_bandwidth,
            row["duration_h"],
            failed,
            lang,
        ),
        figures.shap_waterfall(
            common.drone_contrib(DATA, drone_id),
            META["shap_base"][BEST],
            BEST,
            lang=lang,
        ),
        _findings(common.drone_findings(DATA, SETTINGS, drone_id, lang), lang),
        common.whatif_defaults(row, order),
    )


@app.callback(
    Output("drone-whatif-graph", "figure"),
    Output("whatif-summary", "children"),
    Input("drone-select", "value"),
    Input("whatif-run", "n_clicks"),
    State({"type": "whatif", "feature": ALL}, "value"),
    State("lang", "data"),
)
def update_whatif(drone_id: str, _clicks: int | None, values: list, lang: str):
    order = common.whatif_order(FEATURES)
    rows = common.whatif_base(PRED, drone_id)
    age = rows["duration_h"].to_numpy()
    model = DATA.best_model
    labels = bundle(lang)["model_labels"]
    title = tr(lang, "ui.remaining_life_title", age=age[0], model=labels[BEST])
    current = tr(lang, "ui.current_conditions")
    baseline = figures.conditional_curves(
        model, rows, age, SETTINGS.rul_grid, [current], title, lang=lang
    )
    if ctx.triggered_id != "whatif-run":
        return baseline, None
    if any(value is None or value == "" for value in values):
        return baseline, dbc.Alert(tr(lang, "ui.fill_fields"), color="warning")

    scenario = common.whatif_scenario(rows, order, values)
    fig = figures.conditional_curves(
        model,
        pd.concat([rows, scenario], ignore_index=True),
        np.r_[age, age],
        SETTINGS.rul_grid,
        [current, tr(lang, "ui.scenario")],
        title,
        dashes=["solid", "dash"],
        lang=lang,
    )
    base, new, table = common.whatif_compare(model, rows, scenario, SETTINGS, lang)
    changes = common.whatif_changes(rows.iloc[0], order, CAT_FEATURES, values)
    comparison = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(column) for column in table.columns])),
            html.Tbody(
                [html.Tr([html.Td(value) for value in row]) for row in table.itertuples(index=False)]
            ),
        ],
        bordered=True,
        size="sm",
        style={"maxWidth": "600px"},
    )
    return fig, [comparison, _findings(insights.whatif(base, new, changes, WINDOW, lang), lang)]


@app.callback(
    Output("fleet-download", "data"),
    Input("fleet-download-button", "n_clicks"),
    State("fleet-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def download_fleet(_clicks: int, rows: list[dict] | None):
    """CSV of the fleet rows after the on-screen filter and sort."""
    if not rows:
        raise PreventUpdate
    frame = pd.DataFrame(rows).drop(columns=["id"], errors="ignore")
    return dcc.send_data_frame(frame.to_csv, "fleet.csv", index=False)


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
    State("lang", "data"),
    prevent_initial_call=True,
)
def score_file(contents: str | None, filename: str | None, lang: str):
    if contents is None:
        raise PreventUpdate
    try:
        content = base64.b64decode(contents.split(",", 1)[1])
        scored, warnings = common.score_upload(
            common.read_csv_bytes(content, lang), DATA, SETTINGS, lang
        )
    except ValueError as exc:
        alert = dbc.Alert([html.Strong(f"{filename}: "), str(exc)], color="danger")
        return alert, None, None, True
    status = [_findings(common.score_summary(scored, SETTINGS, lang), lang)]
    if warnings:
        status.append(
            dbc.Alert(
                [
                    html.Strong(tr(lang, "ui.data_warnings")),
                    html.Ul([html.Li(item) for item in warnings], className="mb-0"),
                ],
                color="warning",
                className="small",
            )
        )
    shown = scored.assign(
        out_of_range=scored["out_of_range"].map(
            {True: tr(lang, "ui.yes"), False: ""}
        )
    )
    table = _forecast_table(shown, "score-table", common.score_columns(SETTINGS, lang), lang)
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

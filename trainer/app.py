"""Streamlit studio: train Cox and AFT on an uploaded CSV, then score other files.

Run from the project root (the Docker image does this)::

    uv run streamlit run trainer/app.py
"""

import dataclasses
import sys
from pathlib import Path

# ``streamlit run`` puts only the script directory on sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard import common, figures, insights
from dashboard.data import DashboardData, load
from dashboard.labels import MODEL_LABELS
from dashboard.locale import tr
from trainer import runs
from trainer.pipeline import (
    MIN_ROWS,
    REQUIRED_COLUMNS,
    TrainConfig,
    prepare_frame,
    split_frame,
)

PLOT_CONFIG = {"displaylogo": False}
MAX_PREVIEW_ROWS = 20

STRINGS = {
    "en": {
        "title": "Drone survival studio",
        "caption": "Train Cox and AFT models on your data, then score any other file with them.",
        "tabs": ["Data", "Train", "Runs", "Results", "Score"],
        "data_intro": (
            "Upload a CSV with one row per drone. Required columns: {cols}. "
            f"At least {MIN_ROWS} drones are needed. `failure_mode` is optional."
        ),
        "download_header": "Download an empty template (header only)",
        "upload_train": "Training CSV",
        "read_error": "Could not read {name}: {error}",
        "invalid": "{name}: {error}",
        "rows": "Drones",
        "failures": "Failures",
        "censored": "Censored",
        "in_service": "In service",
        "split": "Split by commissioning date: train {train}, validation {val}, test {test}",
        "preview": "Preview",
        "train_need_data": "Upload and validate a training CSV on the Data tab first.",
        "trials": "Optuna trials per model",
        "boot": "Bootstrap replicates",
        "seed_note": "Defaults match drone_survival.ipynb. Training runs in the background; you can close the page and come back.",
        "start": "Start training",
        "cancel": "Cancel",
        "busy": "A training job is already running.",
        "started": "Started run {run_id}",
        "status": "Run {run_id}: {state}",
        "no_runs": "No runs yet.",
        "runs_cols": {
            "id": "Run",
            "created": "Started",
            "source": "File",
            "n_rows": "Drones",
            "state": "State",
            "best_name": "Best model",
            "c_index": "C-index (Uno), test",
            "ibs": "IBS, test",
        },
        "use_run": "Use this run",
        "delete_run": "Delete this run",
        "select_run": "Run",
        "active": "Active run: {run_id}",
        "no_active": "No finished run yet. Train a model first.",
        "results_note": "Results for the test split of the active run ({run_id}, {source}).",
        "score_intro": "Scores a CSV with one row per drone using the active run. Required columns: the 16 features; optional `drone_id` and `age_h`.",
        "model": "Model",
        "template": "Download a template with five drones from the test split",
        "score_csv": "CSV to score",
        "score_error": "{name}: {error}",
        "warnings": "Data warnings",
        "download": "Download results",
        "loading": "Loading models and artifacts…",
        "language": "Language",
    },
    "ru": {
        "title": "Студия анализа выживаемости дронов",
        "caption": "Обучайте модели Cox и AFT на своих данных и скорьте ими любые другие файлы.",
        "tabs": ["Данные", "Обучение", "Запуски", "Результаты", "Скоринг"],
        "data_intro": (
            "Загрузите CSV, одна строка — один дрон. Обязательные столбцы: {cols}. "
            f"Нужно не менее {MIN_ROWS} дронов. `failure_mode` необязателен."
        ),
        "download_header": "Скачать пустой шаблон (только заголовок)",
        "upload_train": "CSV для обучения",
        "read_error": "Не удалось прочитать {name}: {error}",
        "invalid": "{name}: {error}",
        "rows": "Дронов",
        "failures": "Отказов",
        "censored": "Цензурировано",
        "in_service": "В эксплуатации",
        "split": "Разбиение по дате ввода: train {train}, validation {val}, test {test}",
        "preview": "Предпросмотр",
        "train_need_data": "Сначала загрузите и проверьте CSV для обучения на вкладке «Данные».",
        "trials": "Попыток Optuna на модель",
        "boot": "Повторов bootstrap",
        "seed_note": "Значения по умолчанию совпадают с drone_survival.ipynb. Обучение идёт в фоне: страницу можно закрыть и вернуться.",
        "start": "Начать обучение",
        "cancel": "Отменить",
        "busy": "Обучение уже идёт.",
        "started": "Запущен прогон {run_id}",
        "status": "Прогон {run_id}: {state}",
        "no_runs": "Запусков пока нет.",
        "runs_cols": {
            "id": "Прогон",
            "created": "Начат",
            "source": "Файл",
            "n_rows": "Дронов",
            "state": "Состояние",
            "best_name": "Лучшая модель",
            "c_index": "C-index (Уно), test",
            "ibs": "IBS, test",
        },
        "use_run": "Использовать этот прогон",
        "delete_run": "Удалить этот прогон",
        "select_run": "Прогон",
        "active": "Активный прогон: {run_id}",
        "no_active": "Завершённых прогонов пока нет. Сначала обучите модель.",
        "results_note": "Результаты на тестовой выборке активного прогона ({run_id}, {source}).",
        "score_intro": "Скорит CSV (одна строка — один дрон) моделями активного прогона. Обязательные столбцы: 16 признаков; необязательные `drone_id` и `age_h`.",
        "model": "Модель",
        "template": "Скачать шаблон с пятью дронами из тестовой выборки",
        "score_csv": "CSV для скоринга",
        "score_error": "{name}: {error}",
        "warnings": "Предупреждения по данным",
        "download": "Скачать результаты",
        "loading": "Загрузка моделей и артефактов…",
        "language": "Язык",
    },
}
STATE_LABELS = {
    "en": {
        "running": "running",
        "done": "done",
        "failed": "failed",
        "cancelled": "cancelled",
    },
    "ru": {
        "running": "идёт",
        "done": "готов",
        "failed": "ошибка",
        "cancelled": "отменён",
    },
}


def _t(key: str, **kwargs):
    value = STRINGS[LANG][key]
    return value.format(**kwargs) if kwargs and isinstance(value, str) else value


def _plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme=None)


@st.cache_resource
def _startup() -> int:
    """Mark runs interrupted by a container stop as failed (once per process)."""
    return runs.mark_stale_running()


@st.cache_resource(show_spinner=False)
def _load_run(run_id: str, model: str | None) -> tuple[DashboardData, common.Settings]:
    """Artifacts of a run, with ``model`` (default: the run's best) used for scoring."""
    base = runs.run_path(run_id)
    data = load(base / "artifacts", base / "models")
    if model is not None and model != data.best_name:
        data = dataclasses.replace(data, meta={**data.meta, "best_name": model})
    return data, common.Settings.from_data(data)


def data_tab() -> None:
    st.markdown(_t("data_intro", cols=", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)))
    st.download_button(
        _t("download_header"),
        ",".join(REQUIRED_COLUMNS) + "\n",
        file_name="training_template.csv",
        mime="text/csv",
        on_click="ignore",
    )
    upload = st.file_uploader(_t("upload_train"), type="csv", key="train_file")
    if upload is None:
        st.session_state.pop("train_df", None)
        return
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(_t("read_error", name=upload.name, error=exc))
        st.session_state.pop("train_df", None)
        return
    try:
        df = prepare_frame(raw)
    except ValueError as exc:
        st.error(_t("invalid", name=upload.name, error=exc))
        st.session_state.pop("train_df", None)
        return
    st.session_state["train_df"] = raw
    st.session_state["train_name"] = upload.name
    cards = st.columns(4)
    cards[0].metric(_t("rows"), f"{len(df):,}")
    cards[1].metric(
        _t("failures"), f"{int(df['event'].sum()):,} ({df['event'].mean():.0%})"
    )
    cards[2].metric(_t("censored"), f"{int((df['event'] == 0).sum()):,}")
    cards[3].metric(_t("in_service"), f"{int(df['in_service'].sum()):,}")
    train, val, test, *_ = split_frame(df)
    st.caption(_t("split", train=len(train), val=len(val), test=len(test)))
    st.subheader(_t("preview"))
    st.dataframe(raw.head(MAX_PREVIEW_ROWS), hide_index=True, width="stretch")


@st.fragment(run_every=2)
def _job_status() -> None:
    """Progress of the newest run; refreshes itself while a job is running."""
    all_runs = runs.list_runs()
    if not all_runs:
        return
    status = all_runs[0]
    state = STATE_LABELS[LANG].get(status["state"], status["state"])
    st.markdown(f"**{_t('status', run_id=status['id'], state=state)}**")
    st.progress(float(status.get("fraction", 0.0)), text=status.get("stage", ""))
    if status["state"] in ("failed", "cancelled") and status.get("error"):
        st.error(status["error"])
    if status.get("log"):
        st.code("\n".join(status["log"][-12:]), language=None)
    if status["state"] == "running":
        if st.button(_t("cancel"), key="cancel_job"):
            runs.cancel_training()
    elif st.session_state.get("_was_running"):
        st.session_state["_was_running"] = False
        st.rerun()
    st.session_state["_was_running"] = status["state"] == "running"


def train_tab() -> None:
    raw = st.session_state.get("train_df")
    if raw is None:
        st.info(_t("train_need_data"))
    else:
        st.caption(_t("seed_note"))
        left, right = st.columns(2)
        trials = left.number_input(_t("trials"), 1, 200, TrainConfig.n_trials)
        boot = right.number_input(_t("boot"), 20, 5000, TrainConfig.n_boot, step=20)
        if st.button(_t("start"), type="primary", disabled=runs.job_running()):
            try:
                run_id = runs.start_training(
                    raw,
                    TrainConfig(n_trials=int(trials), n_boot=int(boot)),
                    st.session_state.get("train_name", "upload.csv"),
                )
            except runs.JobBusyError:
                st.warning(_t("busy"))
            else:
                st.success(_t("started", run_id=run_id))
    _job_status()


def runs_tab() -> None:
    all_runs = runs.list_runs()
    if not all_runs:
        st.info(_t("no_runs"))
        return
    labels = _t("runs_cols")
    rows = []
    for status in all_runs:
        metrics = (status.get("test_metrics") or {}).get(status.get("best_name"), {})
        rows.append(
            {
                labels["id"]: status["id"],
                labels["created"]: status["created"].replace("T", " "),
                labels["source"]: status["source"],
                labels["n_rows"]: status["n_rows"],
                labels["state"]: STATE_LABELS[LANG].get(
                    status["state"], status["state"]
                ),
                labels["best_name"]: MODEL_LABELS.get(status.get("best_name"), ""),
                labels["c_index"]: metrics.get("c_index_ipcw"),
                labels["ibs"]: metrics.get("ibs"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    active_id = runs.get_active_id()
    if active_id:
        st.caption(_t("active", run_id=active_id))
    chosen = st.selectbox(_t("select_run"), [s["id"] for s in all_runs])
    left, right, _ = st.columns([1, 1, 3])
    done = runs.usable(runs.read_status(chosen))
    if left.button(_t("use_run"), disabled=not done):
        runs.set_active(chosen)
        st.rerun()
    if right.button(_t("delete_run")):
        try:
            runs.delete_run(chosen)
        except runs.JobBusyError as exc:
            st.warning(str(exc))
        else:
            st.rerun()


def results_tab() -> None:
    run_id = runs.get_active_id()
    if run_id is None:
        st.info(_t("no_active"))
        return
    data, settings = _load_run(run_id, None)
    status = runs.read_status(run_id)
    st.caption(_t("results_note", run_id=run_id, source=status["source"]))
    cards = st.columns(4)
    for k, (title, value, subtitle, _color) in enumerate(
        common.kpis(data, settings, LANG)
    ):
        with cards[k % 4].container(border=True):
            st.caption(title)
            st.markdown(f"### {value}")
            st.caption(subtitle)
    st.dataframe(
        common.priority_sensitivity(common.active(data), settings, LANG),
        hide_index=True,
        width="stretch",
    )
    tables = data.tables
    st.subheader(tr(LANG, "ui.step_metrics"))
    st.dataframe(
        common.metrics_frame(tables["metrics"], LANG), hide_index=True, width="stretch"
    )
    st.info(
        "\n".join(
            f"- {item}"
            for item in insights.metrics(tables["metrics"], settings.best, LANG)
        )
    )
    left, right = st.columns(2)
    with left:
        _plot(figures.survival_mean(tables["survival_mean"], LANG))
    with right:
        horizons = data.meta["calibration_horizons"]
        horizon = st.selectbox(
            tr(LANG, "ui.calibration_horizon"),
            horizons,
            format_func=lambda hours: tr(LANG, "ui.horizon_hours", h=hours),
        )
        _plot(figures.calibration(tables["calibration"], horizon, LANG))
    st.subheader(tr(LANG, "ui.step_bootstrap"))
    st.dataframe(
        common.bootstrap_frame(tables["bootstrap_summary"], LANG),
        hide_index=True,
        width="stretch",
    )
    st.subheader(tr(LANG, "ui.step_importance"))
    _plot(figures.shap_importance(tables["shap_importance"], settings.best, LANG))
    st.subheader(tr(LANG, "ui.step_fleet"))
    fleet = common.fleet_frame(data, settings, LANG)
    st.dataframe(fleet, hide_index=True, width="stretch", height=420)
    st.download_button(
        tr(LANG, "ui.download_fleet"),
        fleet.to_csv(index=False),
        file_name="fleet.csv",
        mime="text/csv",
        on_click="ignore",
    )


def score_tab() -> None:
    run_id = runs.get_active_id()
    if run_id is None:
        st.info(_t("no_active"))
        return
    best = runs.read_status(run_id)["best_name"]
    st.caption(_t("active", run_id=run_id))
    st.markdown(_t("score_intro"))
    model = st.radio(
        _t("model"),
        ["cox", "aft"],
        index=["cox", "aft"].index(best),
        format_func=lambda kind: MODEL_LABELS[kind] + (" *" if kind == best else ""),
        horizontal=True,
    )
    data, settings = _load_run(run_id, model)
    st.download_button(
        _t("template"),
        common.upload_template(data).to_csv(index=False),
        file_name="drone_template.csv",
        mime="text/csv",
        on_click="ignore",
    )
    upload = st.file_uploader(_t("score_csv"), type="csv", key="score_file")
    if upload is None:
        return
    try:
        scored, warnings = common.score_upload(
            common.read_csv_bytes(upload.getvalue(), LANG), data, settings, LANG
        )
    except ValueError as exc:
        st.error(_t("score_error", name=upload.name, error=exc))
        return
    st.info(
        "\n".join(f"- {item}" for item in common.score_summary(scored, settings, LANG))
    )
    if warnings:
        st.warning("\n".join(f"- {item}" for item in warnings), title=_t("warnings"))
    columns = common.score_columns(settings, LANG)
    shown = scored[list(columns)].rename(columns=columns)
    st.dataframe(
        shown, hide_index=True, width="stretch", height=min(560, 38 + 35 * len(shown))
    )
    st.download_button(
        _t("download"),
        scored.to_csv(index=False),
        file_name="drone_forecasts.csv",
        mime="text/csv",
        type="primary",
        on_click="ignore",
    )


st.set_page_config(page_title="Drone survival studio", layout="wide")
_startup()
st.session_state.setdefault("lang", "en")
with st.sidebar:
    st.radio(
        STRINGS[st.session_state["lang"]]["language"],
        ["en", "ru"],
        format_func=lambda code: "English" if code == "en" else "Русский",
        key="lang",
    )
LANG = st.session_state["lang"]
st.title(_t("title"))
st.caption(_t("caption"))
RENDERERS = [data_tab, train_tab, runs_tab, results_tab, score_tab]
for renderer, container in zip(RENDERERS, st.tabs(_t("tabs"))):
    with container:
        renderer()

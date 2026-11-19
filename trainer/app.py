"""Streamlit studio: train on an uploaded CSV, release a version, score other files.

Run from the project root (the Docker image does this)::

    STUDIO_PASSWORD=... uv run streamlit run trainer/app.py

Without a password the studio refuses to work unless ``REQUIRE_AUTH=false``.
"""

import hashlib
import hmac
import logging
import sys
import time
from pathlib import Path

# ``streamlit run`` puts only the script directory on sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard import common, figures, insights
from dashboard.labels import MODEL_LABELS
from dashboard.locale import bundle as locale_bundle
from dashboard.locale import tr
from trainer import monitoring, registry, runs, scoring
from trainer import settings as config
from trainer.pipeline import (
    MIN_ROWS,
    REQUIRED_COLUMNS,
    TrainConfig,
    prepare_frame,
    split_frame,
)

logger = logging.getLogger(__name__)

PLOT_CONFIG = {"displaylogo": False}
MAX_PREVIEW_ROWS = 20
FAILED_SIGN_IN_DELAY_S = 1.0  # slows down password guessing

STRINGS = {
    "en": {
        "title": "Drone survival studio",
        "caption": "Train Cox and AFT models on your data, release a checked version, then score new drones with it.",
        "tabs": ["Data", "Train", "Runs", "Results", "Release", "Score", "Monitoring"],
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
            "gate": "Release gate",
        },
        "gate_passed_kinds": "passed: {kinds}",
        "gate_failed_all": "failed",
        "legacy": "evaluation only (old format)",
        "legacy_note": "Runs marked “evaluation only” were trained before production models existed. They cannot be opened, scored with or promoted; retrain on the same file.",
        "in_production": "in production",
        "use_run": "Use this run",
        "delete_run": "Delete this run",
        "select_run": "Run",
        "active": "Active run: {run_id}",
        "no_active": "No finished run yet. Train a model first.",
        "results_note": "Results for the test split of the active run ({run_id}, {source}). Quality tables describe the evaluation models (train only); the fleet forecast comes from the production model (train + validation).",
        "score_intro": "Scores a CSV with one row per drone. Required columns: the 16 features; optional `drone_id` and `age_h`.",
        "score_source": "Score with",
        "source_production": "Production model",
        "source_candidate": "Active run (candidate)",
        "scoring_with": "Scoring with run {run_id}, {kind}.",
        "not_production": "This is not the production model. Use the result for comparison only, not for maintenance decisions.",
        "no_production": "No model is in production yet. Promote a version on the Release tab.",
        "drift_line": "Input drift against the training data: largest PSI {psi:.3f} ({feature}), {level}.",
        "drift_unreliable": "Input drift: the batch has fewer than {n} drones, too few for a reliable index (largest PSI {psi:.3f}, {feature}).",
        "drift_levels": {
            "stable": "stable",
            "moderate": "moderate shift",
            "significant": "significant shift",
            "n/a": "not assessed",
        },
        "model": "Model",
        "template": "Download a template with five drones from the test split",
        "score_csv": "CSV to score",
        "score_error": "{name}: {error}",
        "warnings": "Data warnings",
        "download": "Download results",
        "loading": "Loading models and artifacts…",
        "language": "Language",
        "password": "Password",
        "sign_in": "Sign in",
        "sign_out": "Sign out",
        "wrong_password": "Wrong password.",
        "config_error": "Configuration error: {error}",
        "too_large": "{name} is larger than {limit} MB.",
        "too_many_rows": "{name} has {n} rows; at most {limit} are accepted.",
        "production_title": "In production",
        "production_none": "Nothing is in production: new drones cannot be scored by the API or the CLI yet.",
        "production_entry": "Run **{run_id}**, model **{kind}**, promoted {at}.",
        "candidate_title": "Candidate",
        "gate_intro": "The production models (train + validation) are checked on the test split against the evaluation models. A model that fails any check cannot be promoted; the limits are set by `GATE_*` variables.",
        "gate_kind": "{kind}: {result}",
        "gate_ok": "passed",
        "gate_bad": "failed",
        "promote_kind": "Model to promote",
        "promote_note": "Note (who, why)",
        "promote": "Promote to production",
        "promoted": "Run {run_id} ({kind}) is now in production.",
        "nothing_to_promote": "No model of this run passed the release gate.",
        "rollback": "Roll back to the previous version",
        "rolled_back": "Production rolled back to run {run_id} ({kind}).",
        "history": "History",
        "history_cols": {
            "at": "When",
            "action": "Action",
            "run_id": "Run",
            "kind": "Model",
            "replaced": "Replaced",
            "note": "Note",
        },
        "actions": {"promote": "promote", "rollback": "rollback"},
        "monitor_intro": "Every scored batch is compared with the data the model was trained on. PSI below {moderate} is stable, above {significant} a clear shift: check the data source and consider retraining. Batches under {n} drones are logged but not assessed. The log keeps aggregates only, no drone rows.",
        "monitor_empty": "Nothing has been scored yet.",
        "monitor_max_psi": "Largest PSI per batch",
        "monitor_last": "Last batch: PSI per feature",
        "monitor_log": "Scoring log (newest first)",
        "monitor_cols": {
            "ts": "Time (UTC)",
            "source": "Source",
            "run_id": "Run",
            "kind": "Model",
            "production": "Production",
            "n_rows": "Drones",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
            "mean_p_fail": "Mean P(failure)",
            "out_of_range_share": "Outside range",
            "max_psi": "Largest PSI",
            "max_psi_feature": "Feature",
            "level": "Drift",
            "psi_reliable": "Assessed",
            "n_unknown_categories": "Unknown categories",
            "input_hash": "Input hash",
        },
    },
    "ru": {
        "title": "Студия анализа выживаемости дронов",
        "caption": "Обучайте модели Cox и AFT на своих данных, выпускайте проверенную версию и скорьте ею новые дроны.",
        "tabs": [
            "Данные",
            "Обучение",
            "Запуски",
            "Результаты",
            "Выпуск",
            "Скоринг",
            "Мониторинг",
        ],
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
            "gate": "Проверка перед выпуском",
        },
        "gate_passed_kinds": "пройдена: {kinds}",
        "gate_failed_all": "не пройдена",
        "legacy": "только оценка (старый формат)",
        "legacy_note": "Прогоны с пометкой «только оценка» обучены до появления боевых моделей. Их нельзя открыть, использовать для скоринга или продвинуть; переобучите на том же файле.",
        "in_production": "в бою",
        "use_run": "Использовать этот прогон",
        "delete_run": "Удалить этот прогон",
        "select_run": "Прогон",
        "active": "Активный прогон: {run_id}",
        "no_active": "Завершённых прогонов пока нет. Сначала обучите модель.",
        "results_note": "Результаты на тестовой выборке активного прогона ({run_id}, {source}). Таблицы качества относятся к оценочным моделям (только train); прогноз по парку даёт боевая модель (train + validation).",
        "score_intro": "Скорит CSV, одна строка — один дрон. Обязательные столбцы: 16 признаков; необязательные `drone_id` и `age_h`.",
        "score_source": "Чем скорить",
        "source_production": "Боевая модель",
        "source_candidate": "Активный прогон (кандидат)",
        "scoring_with": "Скоринг прогоном {run_id}, {kind}.",
        "not_production": "Это не боевая модель. Результат годится только для сравнения, не для решений об обслуживании.",
        "no_production": "Боевой модели пока нет. Продвиньте версию на вкладке «Выпуск».",
        "drift_line": "Дрейф входных данных относительно обучающих: наибольший PSI {psi:.3f} ({feature}), {level}.",
        "drift_unreliable": "Дрейф входных данных: в партии меньше {n} дронов, для надёжного индекса этого мало (наибольший PSI {psi:.3f}, {feature}).",
        "drift_levels": {
            "stable": "стабильно",
            "moderate": "умеренный сдвиг",
            "significant": "существенный сдвиг",
            "n/a": "не оценивается",
        },
        "model": "Модель",
        "template": "Скачать шаблон с пятью дронами из тестовой выборки",
        "score_csv": "CSV для скоринга",
        "score_error": "{name}: {error}",
        "warnings": "Предупреждения по данным",
        "download": "Скачать результаты",
        "loading": "Загрузка моделей и артефактов…",
        "language": "Язык",
        "password": "Пароль",
        "sign_in": "Войти",
        "sign_out": "Выйти",
        "wrong_password": "Неверный пароль.",
        "config_error": "Ошибка конфигурации: {error}",
        "too_large": "{name} больше {limit} МБ.",
        "too_many_rows": "В {name} {n} строк; принимается не более {limit}.",
        "production_title": "В бою",
        "production_none": "Боевой модели нет: API и команда пока не могут скорить новые дроны.",
        "production_entry": "Прогон **{run_id}**, модель **{kind}**, продвинут {at}.",
        "candidate_title": "Кандидат",
        "gate_intro": "Боевые модели (train + validation) проверяются на тестовой выборке относительно оценочных. Модель, не прошедшую хотя бы одну проверку, продвинуть нельзя; пороги задаются переменными `GATE_*`.",
        "gate_kind": "{kind}: {result}",
        "gate_ok": "проверка пройдена",
        "gate_bad": "проверка не пройдена",
        "promote_kind": "Какую модель продвинуть",
        "promote_note": "Заметка (кто, зачем)",
        "promote": "Продвинуть в боевые",
        "promoted": "Прогон {run_id} ({kind}) теперь боевой.",
        "nothing_to_promote": "Ни одна модель этого прогона не прошла проверку перед выпуском.",
        "rollback": "Откатить к предыдущей версии",
        "rolled_back": "Боевая модель откачена к прогону {run_id} ({kind}).",
        "history": "История",
        "history_cols": {
            "at": "Когда",
            "action": "Действие",
            "run_id": "Прогон",
            "kind": "Модель",
            "replaced": "Заменён",
            "note": "Заметка",
        },
        "actions": {"promote": "продвижение", "rollback": "откат"},
        "monitor_intro": "Каждая оценённая партия сравнивается с данными, на которых обучена модель. PSI ниже {moderate} — стабильно, выше {significant} — явный сдвиг: проверьте источник данных и подумайте о переобучении. Партии меньше {n} дронов записываются, но не оцениваются. В журнале только агрегаты, строк дронов нет.",
        "monitor_empty": "Пока ничего не оценивалось.",
        "monitor_max_psi": "Наибольший PSI по партиям",
        "monitor_last": "Последняя партия: PSI по признакам",
        "monitor_log": "Журнал скоринга (сначала новые)",
        "monitor_cols": {
            "ts": "Время (UTC)",
            "source": "Источник",
            "run_id": "Прогон",
            "kind": "Модель",
            "production": "Боевая",
            "n_rows": "Дронов",
            "high": "Высокий",
            "medium": "Средний",
            "low": "Низкий",
            "mean_p_fail": "Средняя P(отказ)",
            "out_of_range_share": "Вне диапазона",
            "max_psi": "Наибольший PSI",
            "max_psi_feature": "Признак",
            "level": "Дрейф",
            "psi_reliable": "Оценён",
            "n_unknown_categories": "Неизвестные категории",
            "input_hash": "Хеш входа",
        },
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
def _settings() -> config.Settings:
    """Configuration from the environment; also sets up logging (once per process)."""
    settings = config.load()
    config.configure_logging(settings)
    return settings


@st.cache_resource
def _startup() -> int:
    """Mark runs interrupted by a container stop as failed (once per process)."""
    return runs.mark_stale_running()


def _sign_in(password: str) -> None:
    """Stop the script behind a password form until the session has signed in."""
    if st.session_state.get("signed_in"):
        return
    with st.form("sign_in"):
        entered = st.text_input(_t("password"), type="password")
        submitted = st.form_submit_button(_t("sign_in"), type="primary")
    if submitted:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["signed_in"] = True
            logger.info("Studio sign-in")
            st.rerun()
        logger.warning("Failed studio sign-in")
        time.sleep(FAILED_SIGN_IN_DELAY_S)
        st.error(_t("wrong_password"))
    st.stop()


def _run_bundle(run_id: str, kind: str | None = None) -> scoring.Bundle | None:
    """Bundle of a run, or ``None`` with an error shown if its files are unusable."""
    try:
        with st.spinner(_t("loading")):
            return scoring.load_bundle(run_id, kind)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return None


def _gate_text(status: dict) -> str:
    """Release-gate cell of the runs table."""
    if runs.legacy(status):
        return _t("legacy")
    gate = status.get("gate")
    if gate is None:
        return ""
    passed = [MODEL_LABELS[k] for k, ok in gate["kinds"].items() if ok]
    if not passed:
        return _t("gate_failed_all")
    return _t("gate_passed_kinds", kinds=", ".join(passed))


def _gate_table(result: dict) -> pd.DataFrame:
    """Checks of one model kind with localized names."""
    names = locale_bundle(LANG)["gate_checks"]
    return pd.DataFrame(
        [
            {
                tr(LANG, "ui.gate_check"): names[check["name"]],
                tr(LANG, "ui.gate_value"): check["value"],
                tr(LANG, "ui.gate_reference"): check["reference"],
                tr(LANG, "ui.gate_threshold"): check["threshold"],
                tr(LANG, "ui.gate_result"): tr(
                    LANG, "ui.gate_pass" if check["passed"] else "ui.gate_fail"
                ),
            }
            for check in result["checks"]
        ]
    )


def _gate_block(card: dict) -> None:
    """Gate result of every model kind of a run."""
    for kind, result in card["gate"]["kinds"].items():
        text = _t(
            "gate_kind",
            kind=MODEL_LABELS[kind],
            result=_t("gate_ok" if result["passed"] else "gate_bad"),
        )
        (st.success if result["passed"] else st.error)(text)
        st.dataframe(_gate_table(result), hide_index=True, width="stretch", alt=text)


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
    st.session_state.pop("train_df", None)
    if upload.size > SETTINGS.max_upload_bytes:
        st.error(_t("too_large", name=upload.name, limit=SETTINGS.max_upload_mb))
        return
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(_t("read_error", name=upload.name, error=exc))
        return
    if len(raw) > SETTINGS.max_train_rows:
        st.error(
            _t(
                "too_many_rows",
                name=upload.name,
                n=len(raw),
                limit=SETTINGS.max_train_rows,
            )
        )
        return
    try:
        df = prepare_frame(raw)
    except ValueError as exc:
        st.error(_t("invalid", name=upload.name, error=exc))
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
                    TrainConfig(
                        n_trials=int(trials), n_boot=int(boot), gate=SETTINGS.gate
                    ),
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
    try:
        production = registry.current()
    except registry.RegistryError:
        production = None  # the Release tab reports the unreadable pointer
    rows = []
    for status in all_runs:
        metrics = (status.get("test_metrics") or {}).get(status.get("best_name"), {})
        state = STATE_LABELS[LANG].get(status["state"], status["state"])
        if production is not None and production["run_id"] == status["id"]:
            state = f"{state}, {_t('in_production')}"
        rows.append(
            {
                labels["id"]: status["id"],
                labels["created"]: status["created"].replace("T", " "),
                labels["source"]: status["source"],
                labels["n_rows"]: status["n_rows"],
                labels["state"]: state,
                labels["best_name"]: MODEL_LABELS.get(status.get("best_name"), ""),
                labels["c_index"]: metrics.get("c_index_ipcw"),
                labels["ibs"]: metrics.get("ibs"),
                labels["gate"]: _gate_text(status),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if any(runs.legacy(status) for status in all_runs):
        st.caption(_t("legacy_note"))
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
            registry.delete_run(chosen)
        except (runs.JobBusyError, registry.RegistryError) as exc:
            st.warning(str(exc))
        else:
            st.rerun()


def results_tab() -> None:
    run_id = runs.get_active_id()
    if run_id is None:
        st.info(_t("no_active"))
        return
    loaded = _run_bundle(run_id)
    if loaded is None:
        return
    data, settings = loaded.data, loaded.settings
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
    st.subheader(tr(LANG, "ui.gate_title"))
    st.info("\n".join(f"- {item}" for item in common.gate_findings(data, LANG)))
    st.dataframe(common.gate_frame(data, LANG), hide_index=True, width="stretch")
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


def release_tab() -> None:
    st.subheader(_t("production_title"))
    try:
        entry, events = registry.current(), registry.history()
    except registry.RegistryError as exc:
        st.error(str(exc))
        return
    if entry is None:
        st.warning(_t("production_none"))
    else:
        st.success(
            _t(
                "production_entry",
                run_id=entry["run_id"],
                kind=MODEL_LABELS.get(entry["kind"], entry["kind"]),
                at=entry["promoted_at"].replace("T", " "),
            )
        )
    if st.button(_t("rollback"), disabled=entry is None, key="rollback"):
        try:
            restored = registry.rollback()
        except registry.RegistryError as exc:
            st.warning(str(exc))
        else:
            st.session_state["release_notice"] = _t(
                "rolled_back",
                run_id=restored["run_id"],
                kind=MODEL_LABELS[restored["kind"]],
            )
            st.rerun()
    if notice := st.session_state.pop("release_notice", None):
        st.toast(notice)

    st.subheader(_t("candidate_title"))
    candidates = [s["id"] for s in runs.list_runs() if runs.usable(s)]
    if not candidates:
        st.info(_t("no_active"))
    else:
        active_id = runs.get_active_id()
        run_id = st.selectbox(
            _t("select_run"),
            candidates,
            index=candidates.index(active_id) if active_id in candidates else 0,
            key="release_run",
        )
        card = runs.read_card(run_id)
        st.caption(_t("gate_intro"))
        _gate_block(card)
        passed = registry.passed_kinds(card)
        if not passed:
            st.warning(_t("nothing_to_promote"))
        else:
            kind = st.segmented_control(
                _t("promote_kind"),
                passed,
                default=card["gate"]["release_kind"],
                format_func=MODEL_LABELS.get,
                required=True,
                key=f"promote_kind_{run_id}",
            )
            note = st.text_input(_t("promote_note"), max_chars=200, key="promote_note")
            if st.button(_t("promote"), type="primary", key="promote"):
                try:
                    promoted = registry.promote(run_id, kind, note)
                except registry.RegistryError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state["release_notice"] = _t(
                        "promoted",
                        run_id=promoted["run_id"],
                        kind=MODEL_LABELS[promoted["kind"]],
                    )
                    st.rerun()

    if events:
        st.subheader(_t("history"))
        labels, actions = _t("history_cols"), _t("actions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        labels["at"]: event["at"].replace("T", " "),
                        labels["action"]: actions.get(event["action"], event["action"]),
                        labels["run_id"]: event["run_id"],
                        labels["kind"]: MODEL_LABELS.get(event["kind"], event["kind"]),
                        labels["replaced"]: (event["replaced"] or {}).get("run_id", ""),
                        labels["note"]: event["note"],
                    }
                    for event in events
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _score_bundle() -> scoring.Bundle | None:
    """Model chosen on the Score tab: production by default, else the active run."""
    active_id = runs.get_active_id()
    try:
        production = scoring.production_bundle()
    except scoring.NoProductionModelError:
        production = None
        st.warning(_t("no_production"))
    sources = [
        name
        for name, ok in (("production", production), ("candidate", active_id))
        if ok
    ]
    if not sources:
        return None
    source = st.segmented_control(
        _t("score_source"),
        sources,
        default=sources[0],
        format_func=lambda name: _t(f"source_{name}"),
        required=True,
        key="score_source",
    )
    if source == "production":
        return production
    card = runs.read_card(active_id)
    kind = st.segmented_control(
        _t("model"),
        list(card["gate"]["kinds"]),
        default=card["forecast_name"],
        format_func=MODEL_LABELS.get,
        required=True,
        key=f"score_kind_{active_id}",
    )
    return _run_bundle(active_id, kind)


def score_tab() -> None:
    st.markdown(_t("score_intro"))
    chosen = _score_bundle()
    if chosen is None:
        return
    st.caption(_t("scoring_with", run_id=chosen.run_id, kind=MODEL_LABELS[chosen.kind]))
    if not chosen.production:
        st.warning(_t("not_production"))
    data, settings = chosen.data, chosen.settings
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
    if upload.size > SETTINGS.max_upload_bytes:
        st.error(_t("too_large", name=upload.name, limit=SETTINGS.max_upload_mb))
        return
    content = upload.getvalue()
    # Streamlit reruns the script on every interaction: score and log a file once
    # per model and language, then reuse the result.
    request = (hashlib.sha256(content).hexdigest(), chosen.run_id, chosen.kind, LANG)
    cached = st.session_state.get("score_result")
    if cached is None or cached[0] != request:
        try:
            frame = common.read_csv_bytes(content, LANG)
            if len(frame) > SETTINGS.max_score_rows:
                raise ValueError(
                    _t(
                        "too_many_rows",
                        name=upload.name,
                        n=len(frame),
                        limit=SETTINGS.max_score_rows,
                    )
                )
            result = scoring.score(frame, chosen, "studio", LANG)
        except ValueError as exc:
            st.session_state.pop("score_result", None)
            st.error(_t("score_error", name=upload.name, error=exc))
            return
        cached = st.session_state["score_result"] = (request, result)
    result = cached[1]
    scored, drift = result.scored, result.drift
    st.info(
        "\n".join(f"- {item}" for item in common.score_summary(scored, settings, LANG))
    )
    if result.warnings:
        st.warning(
            "\n".join(f"- {item}" for item in result.warnings), title=_t("warnings")
        )
    if drift["psi_reliable"]:
        show = st.error if drift["level"] == "significant" else st.caption
        show(
            _t(
                "drift_line",
                psi=drift["max_psi"],
                feature=drift["max_psi_feature"],
                level=_t("drift_levels")[drift["level"]],
            )
        )
    else:
        st.caption(
            _t(
                "drift_unreliable",
                n=monitoring.MIN_PSI_ROWS,
                psi=drift["max_psi"],
                feature=drift["max_psi_feature"],
            )
        )
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


def monitoring_tab() -> None:
    st.markdown(
        _t(
            "monitor_intro",
            moderate=monitoring.PSI_MODERATE,
            significant=monitoring.PSI_SIGNIFICANT,
            n=monitoring.MIN_PSI_ROWS,
        )
    )
    records = monitoring.read_log(runs.RUNS_DIR)
    if not records:
        st.info(_t("monitor_empty"))
        return
    labels, levels = _t("monitor_cols"), _t("drift_levels")
    frame = monitoring.log_frame(records)
    left, right = st.columns(2)
    with left:
        st.subheader(_t("monitor_max_psi"))
        assessed = frame[frame["psi_reliable"].fillna(False).astype(bool)]
        st.line_chart(
            assessed.rename(columns=labels),
            x=labels["ts"],
            y=labels["max_psi"],
            alt=_t("monitor_max_psi"),
        )
    with right:
        st.subheader(_t("monitor_last"))
        last = pd.Series(records[-1]["psi"], name=labels["max_psi"])
        st.bar_chart(last, horizontal=True, alt=_t("monitor_last"))
    st.subheader(_t("monitor_log"))
    table = frame.iloc[::-1].copy()
    table["level"] = table["level"].map(levels).fillna(table["level"])
    st.dataframe(table.rename(columns=labels), hide_index=True, width="stretch")


st.set_page_config(page_title="Drone survival studio", layout="wide")
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
try:
    SETTINGS = _settings()
    PASSWORD = config.require(SETTINGS, "studio_password")
except config.SettingsError as exc:
    st.error(_t("config_error", error=exc))
    st.stop()
if PASSWORD is not None:
    _sign_in(PASSWORD)
    with st.sidebar:
        if st.button(_t("sign_out")):
            st.session_state.clear()
            st.rerun()
_startup()
RENDERERS = [
    data_tab,
    train_tab,
    runs_tab,
    results_tab,
    release_tab,
    score_tab,
    monitoring_tab,
]
for renderer, container in zip(RENDERERS, st.tabs(_t("tabs"))):
    with container:
        renderer()

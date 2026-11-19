"""Run registry and background training job.

A run is a directory ``RUNS_DIR/<run_id>/`` with ``status.json`` and, once training
succeeded, ``artifacts/`` and ``models/`` in the layout ``dashboard.data.load`` reads.
Only one training job runs at a time, in a thread of the app process; the state
lives in ``status.json`` so a page reload or a restart does not lose it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from trainer.pipeline import Cancelled, TrainConfig, train

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
_RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_LOCK = threading.Lock()
_JOB: dict[str, object] = {"thread": None, "cancel": None, "run_id": None}


class JobBusyError(RuntimeError):
    """Raised when a training job is already running."""


def run_path(run_id: str) -> Path:
    """Directory of ``run_id``, checked to be a generated id inside ``RUNS_DIR``.

    Raises
    ------
    ValueError
        If ``run_id`` is not of the form produced by ``new_run_id``.
    """
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    path = (RUNS_DIR / run_id).resolve()
    if path.parent != RUNS_DIR.resolve():
        raise ValueError(f"Invalid run id: {run_id!r}")
    return path


def new_run_id() -> str:
    """Time-ordered run id with a random suffix."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _write_status(run_id: str, status: dict) -> None:
    path = run_path(run_id) / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read_status(run_id: str) -> dict:
    """Contents of the run's ``status.json``."""
    return json.loads((run_path(run_id) / "status.json").read_text())


def _update(run_id: str, **changes: object) -> dict:
    status = read_status(run_id)
    status.update(changes)
    _write_status(run_id, status)
    return status


def list_runs() -> list[dict]:
    """Status of every run, newest first. Unreadable directories are skipped."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not path.is_dir() or not _RUN_ID.fullmatch(path.name):
            continue
        try:
            runs.append(json.loads((path / "status.json").read_text()))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping run %s: unreadable status.json", path.name)
    return runs


def usable(status: dict) -> bool:
    """True if the run finished and its files are in place."""
    base = run_path(status["id"])
    return status.get("state") == "done" and (base / "artifacts" / "meta.json").exists()


def mark_stale_running() -> int:
    """Mark runs left ``running`` by a killed process as failed; return their count."""
    count = 0
    with _LOCK:
        live = _JOB["run_id"] if _is_alive() else None
        for status in list_runs():
            if status.get("state") == "running" and status["id"] != live:
                _update(
                    status["id"],
                    state="failed",
                    error="Interrupted: the container stopped during training",
                )
                count += 1
    return count


def get_active_id() -> str | None:
    """Id of the selected run; falls back to the newest usable run."""
    marker = RUNS_DIR / "active.json"
    if marker.exists():
        try:
            run_id = json.loads(marker.read_text()).get("run_id")
            if run_id and _RUN_ID.fullmatch(run_id):
                status = read_status(run_id)
                if usable(status):
                    return run_id
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("Ignoring unreadable active run marker")
    for status in list_runs():
        if usable(status):
            return status["id"]
    return None


def set_active(run_id: str) -> None:
    """Select ``run_id`` for results and scoring."""
    if not usable(read_status(run_id)):
        raise ValueError("Only finished runs can be selected")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "active.json").write_text(json.dumps({"run_id": run_id}))


def delete_run(run_id: str) -> None:
    """Delete a run directory.

    Raises
    ------
    JobBusyError
        If the run is being trained.
    """
    with _LOCK:
        if _is_alive() and _JOB["run_id"] == run_id:
            raise JobBusyError("The run is still training; cancel it first")
        shutil.rmtree(run_path(run_id))


def _is_alive() -> bool:
    thread = _JOB["thread"]
    return isinstance(thread, threading.Thread) and thread.is_alive()


def job_running() -> bool:
    """True while a training thread is alive."""
    return _is_alive()


def start_training(raw: pd.DataFrame, config: TrainConfig, source_name: str) -> str:
    """Start a background training run.

    Parameters
    ----------
    raw : pd.DataFrame
        Training table; validated in the job so the error lands in ``status.json``.
    config : TrainConfig
        Training settings.
    source_name : str
        Uploaded file name, kept for display only.

    Returns
    -------
    str
        The new run id.

    Raises
    ------
    JobBusyError
        If another job is running.
    """
    with _LOCK:
        if _is_alive():
            raise JobBusyError("A training job is already running")
        run_id = new_run_id()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_path(run_id).mkdir()
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(raw, index=False).to_numpy().tobytes()
        ).hexdigest()[:12]
        _write_status(
            run_id,
            {
                "id": run_id,
                "state": "running",
                "stage": "Queued",
                "fraction": 0.0,
                "created": datetime.now().isoformat(timespec="seconds"),
                "finished": None,
                "source": os.path.basename(source_name),
                "data_hash": digest,
                "n_rows": len(raw),
                "config": asdict(config),
                "log": [],
                "error": None,
            },
        )
        cancel = threading.Event()
        thread = threading.Thread(
            target=_run_job, args=(run_id, raw.copy(), config, cancel), daemon=True
        )
        _JOB.update(thread=thread, cancel=cancel, run_id=run_id)
        thread.start()
    return run_id


def cancel_training() -> None:
    """Ask the running job to stop at the next stage boundary."""
    cancel = _JOB["cancel"]
    if _is_alive() and isinstance(cancel, threading.Event):
        cancel.set()


def _run_job(
    run_id: str, raw: pd.DataFrame, config: TrainConfig, cancel: threading.Event
) -> None:
    base = run_path(run_id)
    work = base / "work"

    def progress(stage: str, fraction: float) -> None:
        if cancel.is_set():
            raise Cancelled
        status = read_status(run_id)
        status["log"].append(f"{datetime.now():%H:%M:%S} {stage}")
        _update(run_id, stage=stage, fraction=fraction, log=status["log"])

    try:
        meta = train(raw, config, work, progress)
        (work / "artifacts").replace(base / "artifacts")
        (work / "models").replace(base / "models")
        work.rmdir()
        metrics = pd.read_parquet(base / "artifacts" / "metrics.parquet")
        test = metrics[metrics["split"] == "test"].set_index("model")
        _update(
            run_id,
            state="done",
            stage="Done",
            fraction=1.0,
            finished=datetime.now().isoformat(timespec="seconds"),
            best_name=meta["best_name"],
            test_metrics=test[["c_index_ipcw", "ibs"]].round(4).to_dict(orient="index"),
        )
    except Cancelled:
        _fail(run_id, work, "cancelled", "Cancelled by the user")
    except ValueError as exc:
        _fail(run_id, work, "failed", str(exc))
    except Exception as exc:
        logger.exception("Training run %s failed", run_id)
        _fail(run_id, work, "failed", f"{type(exc).__name__}: {exc}")


def _fail(run_id: str, work: Path, state: str, message: str) -> None:
    shutil.rmtree(work, ignore_errors=True)
    _update(
        run_id,
        state=state,
        error=message,
        finished=datetime.now().isoformat(timespec="seconds"),
    )

"""Model registry: which trained version is in production.

Every finished run with a model card is a candidate version. ``production.json``
in ``RUNS_DIR`` points at the version and model kind that score new drones and
keeps the versions it replaced, so a promotion can be rolled back. A kind can be
promoted only if it passed the release gate of its run; there is no override.

``promote`` and ``rollback`` hold a file lock and replace the pointer atomically,
so the studio, the CLI and the API (separate processes on one volume) never see a
half-written pointer.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from trainer import runs
from trainer.fslock import locked

logger = logging.getLogger(__name__)

POINTER_NAME = "production.json"
_LOCK_NAME = ".registry.lock"
MAX_LOG = 200  # promotions and rollbacks kept in the pointer file


class RegistryError(RuntimeError):
    """Raised when a version cannot be promoted, rolled back to or deleted."""


def pointer_path() -> Path:
    """Path of the production pointer."""
    return runs.RUNS_DIR / POINTER_NAME


def _read() -> dict:
    """Pointer contents; an empty registry if the file does not exist.

    Raises
    ------
    RegistryError
        If the file exists but is not valid JSON.
    """
    path = pointer_path()
    if not path.exists():
        return {"current": None, "previous": [], "log": []}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{path.name} is unreadable: {exc}") from exc


def _write(pointer: dict) -> None:
    path = pointer_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(pointer, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def current() -> dict | None:
    """The production entry (``run_id``, ``kind``, ``promoted_at``, ...) or ``None``."""
    return _read()["current"]


def history() -> list[dict]:
    """Promotions and rollbacks, newest first."""
    return list(reversed(_read()["log"]))


def passed_kinds(card: dict) -> list[str]:
    """Model kinds of a card that passed the release gate."""
    return [kind for kind, result in card["gate"]["kinds"].items() if result["passed"]]


def check_promotable(run_id: str, kind: str | None = None) -> tuple[dict, str]:
    """Validate that a version can go to production.

    Parameters
    ----------
    run_id : str
        Run to promote.
    kind : str, optional
        ``"cox"`` or ``"aft"``; by default the gate's release kind: the best on
        validation among the kinds that passed.

    Returns
    -------
    card : dict
        Model card of the run.
    kind : str
        The kind to promote.

    Raises
    ------
    RegistryError
        If the run is unfinished, has no model card (trained by an older
        version), or the kind did not pass the release gate.
    ValueError
        If ``run_id`` is malformed.
    """
    try:
        status = runs.read_status(run_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Run {run_id} not found") from exc
    if status.get("state") != "done":
        raise RegistryError(f"Run {run_id} is not finished ({status.get('state')})")
    card = runs.read_card(run_id)
    if card is None:
        raise RegistryError(
            f"Run {run_id} has no production models (evaluation only); retrain it"
        )
    passed = passed_kinds(card)
    if kind is None:
        kind = card["gate"]["release_kind"]
        if kind is None:
            raise RegistryError(f"Run {run_id}: no model passed the release gate")
    elif kind not in card["gate"]["kinds"]:
        raise RegistryError(f"Run {run_id} has no model of kind {kind!r}")
    elif kind not in passed:
        failed = [
            check["name"]
            for check in card["gate"]["kinds"][kind]["checks"]
            if not check["passed"]
        ]
        raise RegistryError(
            f"Run {run_id}: {kind} failed the release gate ({', '.join(failed)})"
        )
    return card, kind


def _entry(run_id: str, kind: str, card: dict, note: str) -> dict:
    return {
        "run_id": run_id,
        "kind": kind,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "data_sha256": card["data"]["sha256"],
        "note": note,
    }


def _log(pointer: dict, action: str, entry: dict, replaced: dict | None) -> None:
    pointer["log"].append(
        {
            "at": entry["promoted_at"],
            "action": action,
            "run_id": entry["run_id"],
            "kind": entry["kind"],
            "replaced": None
            if replaced is None
            else {"run_id": replaced["run_id"], "kind": replaced["kind"]},
            "note": entry["note"],
        }
    )
    del pointer["log"][:-MAX_LOG]


def promote(run_id: str, kind: str | None = None, note: str = "") -> dict:
    """Make a version the production model.

    Parameters
    ----------
    run_id : str
        Run to promote.
    kind : str, optional
        Model kind; by default the gate's release kind.
    note : str
        Free text stored with the promotion (who, why).

    Returns
    -------
    dict
        The new production entry.

    Raises
    ------
    RegistryError
        If the version cannot be promoted (see ``check_promotable``) or already
        is the production model.
    """
    with locked(runs.RUNS_DIR / _LOCK_NAME):
        card, kind = check_promotable(run_id, kind)
        pointer = _read()
        replaced = pointer["current"]
        if replaced and (replaced["run_id"], replaced["kind"]) == (run_id, kind):
            raise RegistryError(f"Run {run_id} ({kind}) is already in production")
        entry = _entry(run_id, kind, card, note)
        if replaced is not None:
            pointer["previous"].append(replaced)
        pointer["current"] = entry
        _log(pointer, "promote", entry, replaced)
        _write(pointer)
    logger.info("Promoted run %s (%s) to production", run_id, kind)
    return entry


def rollback(note: str = "") -> dict:
    """Return to the version that was in production before the current one.

    Versions whose run was deleted or no longer passes ``check_promotable`` are
    skipped.

    Returns
    -------
    dict
        The restored production entry.

    Raises
    ------
    RegistryError
        If there is no earlier version to return to.
    """
    with locked(runs.RUNS_DIR / _LOCK_NAME):
        pointer = _read()
        replaced = pointer["current"]
        while pointer["previous"]:
            candidate = pointer["previous"].pop()
            try:
                card, kind = check_promotable(candidate["run_id"], candidate["kind"])
            except (RegistryError, ValueError) as exc:
                logger.warning("Skipping rollback target: %s", exc)
                continue
            entry = _entry(candidate["run_id"], kind, card, note)
            pointer["current"] = entry
            _log(pointer, "rollback", entry, replaced)
            _write(pointer)
            logger.info("Rolled production back to run %s (%s)", entry["run_id"], kind)
            return entry
        raise RegistryError("No earlier production version to roll back to")


def delete_run(run_id: str) -> None:
    """Delete a run unless it is the production version.

    Raises
    ------
    RegistryError
        If the run is in production.
    runs.JobBusyError
        If the run is being trained.
    """
    with locked(runs.RUNS_DIR / _LOCK_NAME):
        entry = _read()["current"]
        if entry is not None and entry["run_id"] == run_id:
            raise RegistryError(
                f"Run {run_id} is in production; promote another version first"
            )
        runs.delete_run(run_id)

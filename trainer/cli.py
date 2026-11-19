"""Command line: train, promote, roll back and score in batch.

Run from the project root::

    uv run python -m trainer.cli train data/drone_survival.csv
    uv run python -m trainer.cli status
    uv run python -m trainer.cli promote <run_id>
    uv run python -m trainer.cli score new_drones.csv forecasts.csv
    uv run python -m trainer.cli drift
    uv run python -m trainer.cli rollback

Exit codes: 0 success; 1 bad input (file, data, arguments); 2 the release gate
failed or the version cannot be promoted or rolled back; 3 no production model;
4 another training job is running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from dashboard import common
from trainer import monitoring, registry, runs, scoring
from trainer import settings as config
from trainer.pipeline import TrainConfig

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_INPUT, EXIT_GATE, EXIT_NO_MODEL, EXIT_BUSY = 0, 1, 2, 3, 4


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")


def _read_csv(path: Path, max_bytes: int | None = None) -> pd.DataFrame:
    """Read a CSV file.

    Raises
    ------
    ValueError
        If the file is missing, too large or not a readable CSV.
    """
    if not path.is_file():
        raise ValueError(f"{path}: no such file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"{path}: larger than {max_bytes} bytes (MAX_UPLOAD_MB)")
    return common.read_csv_bytes(path.read_bytes())


def _gate_lines(card: dict) -> list[str]:
    lines = []
    for kind, result in card["gate"]["kinds"].items():
        lines.append(f"  {kind}: {'passed' if result['passed'] else 'FAILED'}")
        for check in result["checks"]:
            reference = (
                ""
                if check["reference"] is None
                else f", evaluation {check['reference']:.4g}"
            )
            lines.append(
                f"    {'ok  ' if check['passed'] else 'FAIL'} {check['name']}: "
                f"{check['value']:.4g} (limit {check['threshold']:.4g}{reference})"
            )
    return lines


def _train(args: argparse.Namespace, settings: config.Settings) -> int:
    raw = _read_csv(args.csv)
    if len(raw) > settings.max_train_rows:
        raise ValueError(
            f"{args.csv}: {len(raw)} rows, at most {settings.max_train_rows} "
            "(MAX_TRAIN_ROWS)"
        )
    train_config = TrainConfig(
        n_trials=args.trials, n_boot=args.boot, gate=settings.gate
    )
    try:
        status = runs.run_training(
            raw,
            train_config,
            args.csv.name,
            lambda stage, fraction: _err(f"[{fraction:4.0%}] {stage}"),
        )
    except runs.JobBusyError as exc:
        _err(str(exc))
        return EXIT_BUSY
    if status["state"] != "done":
        _err(f"Run {status['id']} {status['state']}: {status['error']}")
        return EXIT_INPUT
    card = runs.read_card(status["id"])
    _out(f"Run {status['id']}: best on validation {card['best_name']}")
    _out("Release gate:")
    for line in _gate_lines(card):
        _out(line)
    release_kind = card["gate"]["release_kind"]
    if release_kind is None:
        _out("No model passed the release gate; nothing can be promoted.")
        return EXIT_GATE
    _out(
        f"Can be promoted: {release_kind} (python -m trainer.cli promote {status['id']})"
    )
    if args.promote:
        entry = registry.promote(status["id"], note=args.note)
        _out(f"Promoted {entry['run_id']} ({entry['kind']}) to production.")
    return EXIT_OK


def _promote(args: argparse.Namespace, settings: config.Settings) -> int:
    entry = registry.promote(args.run_id, args.kind, args.note)
    _out(f"Promoted {entry['run_id']} ({entry['kind']}) to production.")
    return EXIT_OK


def _rollback(args: argparse.Namespace, settings: config.Settings) -> int:
    entry = registry.rollback(args.note)
    _out(f"Production rolled back to {entry['run_id']} ({entry['kind']}).")
    return EXIT_OK


def _status(args: argparse.Namespace, settings: config.Settings) -> int:
    entry = registry.current()
    if entry is None:
        _out("Production: none")
    else:
        _out(
            f"Production: {entry['run_id']} ({entry['kind']}), "
            f"promoted {entry['promoted_at']}"
        )
    _out("Runs:")
    for status in runs.list_runs():
        gate = status.get("gate")
        if status.get("state") != "done":
            detail = status.get("state", "?")
        elif gate is None:
            detail = "evaluation only (old format)"
        else:
            passed = [k for k, ok in gate["kinds"].items() if ok]
            detail = f"gate passed: {', '.join(passed) or 'none'}"
        _out(f"  {status['id']}  {status.get('source', '')}  {detail}")
    if args.history:
        _out("History:")
        for event in registry.history():
            _out(
                f"  {event['at']}  {event['action']}  {event['run_id']} ({event['kind']})"
            )
    return EXIT_OK


def _score(args: argparse.Namespace, settings: config.Settings) -> int:
    raw = _read_csv(args.input, settings.max_upload_bytes)
    if len(raw) > settings.max_score_rows:
        raise ValueError(
            f"{args.input}: {len(raw)} rows, at most {settings.max_score_rows} "
            "(MAX_SCORE_ROWS)"
        )
    bundle = (
        scoring.load_bundle(args.run, args.kind)
        if args.run
        else scoring.production_bundle()
    )
    result = scoring.score(raw, bundle, "cli", args.lang)
    result.scored.to_csv(args.output, index=False)
    _out(
        f"Scored {len(result.scored)} drones with {bundle.run_id} ({bundle.kind}"
        f"{'' if bundle.production else ', NOT the production model'}) -> {args.output}"
    )
    for line in common.score_summary(result.scored, bundle.settings, args.lang):
        _out(line.replace("**", ""))
    for warning in result.warnings:
        _err(f"warning: {warning}")
    _drift_lines(result.drift)
    return EXIT_OK


def _drift_lines(report: dict) -> None:
    note = "" if report["psi_reliable"] else " (too few rows for a reliable index)"
    _out(
        f"Drift: max PSI {report['max_psi']:.3f} on {report['max_psi_feature']} "
        f"-> {report['level']}{note}"
    )
    if report["unknown_categories"]:
        _out(f"Unknown categories: {json.dumps(report['unknown_categories'])}")


def _drift(args: argparse.Namespace, settings: config.Settings) -> int:
    if args.input is not None:
        raw = _read_csv(args.input, settings.max_upload_bytes)
        result = scoring.score(raw, scoring.production_bundle(), "cli", log=False)
        _drift_lines(result.drift)
        table = pd.Series(result.drift["psi"], name="psi").sort_values(ascending=False)
        _out(table.to_string())
        return EXIT_OK
    records = monitoring.read_log(runs.RUNS_DIR, args.last)
    if not records:
        _out("The scoring log is empty.")
        return EXIT_OK
    _out(
        monitoring.log_frame(records)
        .drop(columns=["input_hash"])
        .to_string(index=False)
    )
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trainer.cli", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a new version from a CSV file")
    train.add_argument("csv", type=Path)
    train.add_argument("--trials", type=int, default=TrainConfig.n_trials)
    train.add_argument("--boot", type=int, default=TrainConfig.n_boot)
    train.add_argument(
        "--promote", action="store_true", help="promote if the release gate passes"
    )
    train.add_argument("--note", default="", help="text stored with the promotion")
    train.set_defaults(handler=_train)

    promote = commands.add_parser("promote", help="make a version the production model")
    promote.add_argument("run_id")
    promote.add_argument("--kind", choices=["cox", "aft"])
    promote.add_argument("--note", default="")
    promote.set_defaults(handler=_promote)

    rollback = commands.add_parser("rollback", help="return to the previous version")
    rollback.add_argument("--note", default="")
    rollback.set_defaults(handler=_rollback)

    status = commands.add_parser("status", help="production version and runs")
    status.add_argument("--history", action="store_true")
    status.set_defaults(handler=_status)

    score = commands.add_parser("score", help="score a CSV file of drones")
    score.add_argument("input", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--run", help="score with this run instead of production")
    score.add_argument("--kind", choices=["cox", "aft"])
    score.add_argument("--lang", choices=["en", "ru"], default="en")
    score.set_defaults(handler=_score)

    drift = commands.add_parser(
        "drift", help="drift of a CSV file, or of the logged batches without one"
    )
    drift.add_argument("input", type=Path, nargs="?")
    drift.add_argument("--last", type=int, default=20, help="logged batches to show")
    drift.set_defaults(handler=_drift)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its exit code."""
    args = _parser().parse_args(argv)
    try:
        settings = config.load()
    except config.SettingsError as exc:
        _err(f"Configuration error: {exc}")
        return EXIT_INPUT
    config.configure_logging(settings)
    try:
        return args.handler(args, settings)
    except scoring.NoProductionModelError as exc:
        _err(str(exc))
        return EXIT_NO_MODEL
    except registry.RegistryError as exc:
        _err(str(exc))
        return EXIT_GATE
    except (ValueError, KeyError, FileNotFoundError) as exc:
        _err(f"Error: {exc}")
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())

"""Load the notebook artifacts and fitted models used by the dashboard."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dashboard.labels import PRIORITY_ORDER
from trainer.release import FORMAT_VERSION
from utils import Model

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

TABLES = (
    "metrics",
    "brier_auc",
    "calibration",
    "bootstrap_summary",
    "bootstrap_samples",
    "shap_values",
    "shap_production",
    "shap_importance",
    "predictions",
    "survival_mean",
    "backtest",
    "backtest_groups",
)


@dataclass(frozen=True)
class DashboardData:
    """Everything the dashboard renders.

    Attributes
    ----------
    meta : dict
        Run metadata written by the notebook (``meta.json``).
    tables : dict[str, pd.DataFrame]
        Parquet artifacts keyed by file stem (see ``TABLES``).
    models : dict[str, Model]
        Evaluation models keyed by kind (``"cox"``, ``"aft"``): tuned on train,
        early-stopped on validation. The quality tabs describe them.
    production : dict[str, Model]
        Production models keyed by kind: the same parameters refitted on train +
        validation. Every forecast comes from one of them.
    card : dict
        Model card (``model_card.json``) with the release-gate result.
    forecast_kind : str, optional
        Overrides the kind that makes the forecasts (``meta["forecast_name"]``).
    """

    meta: dict
    tables: dict[str, pd.DataFrame]
    models: dict[str, Model]
    production: dict[str, Model]
    card: dict
    forecast_kind: str | None = None

    @property
    def predictions(self) -> pd.DataFrame:
        """Per-drone test predictions."""
        return self.tables["predictions"]

    @property
    def best_name(self) -> str:
        """Kind of the model selected in the notebook."""
        return self.meta["best_name"]

    @property
    def best_model(self) -> Model:
        """Evaluation model selected in the notebook."""
        return self.models[self.best_name]

    @property
    def forecast_name(self) -> str:
        """Kind that makes the forecasts.

        The kind selected on validation, unless its production model failed the
        release gate and the other kind passed.
        """
        return self.forecast_kind or self.meta["forecast_name"]

    @property
    def forecast_model(self) -> Model:
        """Production model that makes the forecasts."""
        return self.production[self.forecast_name]

    @property
    def gate(self) -> dict:
        """Release-gate result of the production models."""
        return self.card["gate"]


def load(
    artifacts_dir: Path = ARTIFACTS_DIR, models_dir: Path = MODELS_DIR
) -> DashboardData:
    """Read artifacts and models written by ``drone_survival.ipynb``.

    Parameters
    ----------
    artifacts_dir : Path
        Directory with ``meta.json`` and the parquet tables.
    models_dir : Path
        Directory with the ``Model.save`` JSON files.

    Returns
    -------
    DashboardData
        Loaded artifacts and models.

    Raises
    ------
    FileNotFoundError
        If any artifact or model file is missing.
    ValueError
        If the artifacts were written by an older notebook version: no production
        models (``format`` below ``FORMAT_VERSION``), no ``in_service`` column or
        priority labels other than ``PRIORITY_ORDER``.
    """
    hint = "run drone_survival.ipynb end to end first"
    meta_path = artifacts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; {hint}")
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{meta_path} has format {meta.get('format', 1)}, expected "
            f"{FORMAT_VERSION} (artifacts without production models); "
            "re-run drone_survival.ipynb"
        )

    card_path = artifacts_dir / "model_card.json"
    table_paths = {name: artifacts_dir / f"{name}.parquet" for name in TABLES}
    model_paths = {kind: models_dir / name for kind, name in meta["models"].items()}
    production_paths = {
        kind: models_dir / name for kind, name in meta["production_models"].items()
    }
    missing = [
        str(p)
        for p in [
            card_path,
            *table_paths.values(),
            *model_paths.values(),
            *production_paths.values(),
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing dashboard inputs ({hint}): {', '.join(missing)}"
        )

    tables = {name: pd.read_parquet(path) for name, path in table_paths.items()}
    if "in_service" not in tables["predictions"]:
        raise ValueError(
            "predictions.parquet has no in_service column; re-run drone_survival.ipynb"
        )
    unknown = set(tables["predictions"]["priority"]) - set(PRIORITY_ORDER)
    if unknown:
        raise ValueError(
            f"Unexpected priority labels {sorted(unknown)} in predictions.parquet "
            f"(expected {PRIORITY_ORDER}); re-run drone_survival.ipynb"
        )
    logger.info(
        "Loaded artifacts from %s (generated %s)", artifacts_dir, meta["generated_at"]
    )
    return DashboardData(
        meta=meta,
        tables=tables,
        models={kind: Model.load(path) for kind, path in model_paths.items()},
        production={kind: Model.load(path) for kind, path in production_paths.items()},
        card=json.loads(card_path.read_text()),
    )

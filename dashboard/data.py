"""Load the notebook artifacts and fitted models used by the dashboard."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dashboard.labels import PRIORITY_ORDER
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
    "shap_importance",
    "predictions",
    "survival_mean",
    "backtest",
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
        Fitted models keyed by kind (``"cox"``, ``"aft"``).
    """

    meta: dict
    tables: dict[str, pd.DataFrame]
    models: dict[str, Model]

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
        """Model selected in the notebook."""
        return self.models[self.best_name]


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
        If the predictions use priority labels other than ``PRIORITY_ORDER``
        (artifacts written by an older notebook version).
    """
    hint = "run drone_survival.ipynb end to end first"
    meta_path = artifacts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; {hint}")
    meta = json.loads(meta_path.read_text())

    table_paths = {name: artifacts_dir / f"{name}.parquet" for name in TABLES}
    model_paths = {kind: models_dir / name for kind, name in meta["models"].items()}
    missing = [
        str(p) for p in [*table_paths.values(), *model_paths.values()] if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing dashboard inputs ({hint}): {', '.join(missing)}"
        )

    tables = {name: pd.read_parquet(path) for name, path in table_paths.items()}
    unknown = set(tables["predictions"]["priority"]) - set(PRIORITY_ORDER)
    if unknown:
        raise ValueError(
            f"Unexpected priority labels {sorted(unknown)} in predictions.parquet "
            f"(expected {PRIORITY_ORDER}); re-run drone_survival.ipynb"
        )
    models = {kind: Model.load(path) for kind, path in model_paths.items()}
    logger.info(
        "Loaded artifacts from %s (generated %s)", artifacts_dir, meta["generated_at"]
    )
    return DashboardData(meta=meta, tables=tables, models=models)

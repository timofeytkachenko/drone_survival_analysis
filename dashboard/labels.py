"""Labels and the priority codes stored in the data.

Display names come from the English catalog. Priority codes stay the values
written in the parquet files (``High``, ``Medium``, ``Low``); only their
captions are translated.
"""

from dashboard.locale import bundle

_EN = bundle("en")
MODEL_LABELS = _EN["model_labels"]
FEATURE_LABELS = _EN["feature_labels"]
SEGMENT_LABELS = _EN["segment_labels"]
FAILURE_MODE_LABELS = _EN["failure_mode_labels"]
METRIC_LABELS = _EN["metric_labels"]
HAZARD_UNIT = _EN["hazard_unit"]

PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW = "High", "Medium", "Low"
PRIORITY_FAILED = "None (failed)"
PRIORITY_LEFT = "None (left the fleet)"
PRIORITY_ORDER = [
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    PRIORITY_FAILED,
    PRIORITY_LEFT,
]
RISK_GROUP_ORDER = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]

# Hazard is per flight hour in the models; the charts show it per 100 flight hours.
HAZARD_SCALE = 100.0

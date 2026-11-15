"""Human-readable labels for the business-facing dashboard."""

MODEL_LABELS = {"cox": "Cox", "aft": "AFT"}

FEATURE_LABELS = {
    "model": "Drone model",
    "environment": "Operating environment",
    "is_refurbished": "Refurbished",
    "initial_battery_cycles": "Battery cycles at start",
    "firmware_major": "Firmware version",
    "maintenance_interval_h": "Maintenance interval, h",
    "operator_experience_yrs": "Operator experience, yrs",
    "daily_utilization_h": "Flight hours per day",
    "avg_flight_min": "Average flight, min",
    "avg_payload_kg": "Average payload, kg",
    "payload_ratio": "Payload (share of max)",
    "mean_temp_c": "Average temperature, °C",
    "avg_humidity_pct": "Average humidity, %",
    "avg_wind_ms": "Average wind, m/s",
    "hard_landings_per_100h": "Hard landings per 100 h",
    "mean_vibration_g": "Vibration, g",
}

SEGMENT_LABELS = {
    "model": "Drone model",
    "environment": "Operating environment",
    "firmware_major": "Firmware version",
    "is_refurbished": "Refurbished",
    "risk_group": "Risk group (model)",
}

FAILURE_MODE_LABELS = {
    "motor": "Motor",
    "battery": "Battery",
    "electronics": "Electronics",
    "crash": "Crash",
    "censored": "No failure",
}

METRIC_LABELS = {
    "c_index": "C-index (Harrell)",
    "c_index_ipcw": "C-index (Uno, IPCW)",
    "mean_auc": "Mean AUC(t)",
    "ibs": "Integrated Brier score",
}

PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW = "High", "Medium", "Low"
PRIORITY_FAILED = "None (failed)"
PRIORITY_ORDER = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW, PRIORITY_FAILED]
RISK_GROUP_ORDER = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]

# Hazard is per flight hour in the models; it is shown per 100 flight hours.
HAZARD_SCALE = 100.0
HAZARD_UNIT = "Failures per 100 flight hours"

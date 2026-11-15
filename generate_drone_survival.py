"""Generate a synthetic survival-analysis dataset for drone failure prediction.

One row per drone. Time is measured in flight hours from commissioning until
the first critical failure (``event == 1``) or right-censoring (``event == 0``).

Data-generating process
-----------------------
* Baseline (time-fixed) covariates are sampled with realistic dependencies:
  operating environment drives climate, wind drives hard landings, hard
  landings and airframe type drive vibration, and so on.
* Four competing failure mechanisms (motor, battery, electronics, crash) each
  follow a Weibull proportional-hazards model
  ``h_k(t | x) = (k_k / s_k) (t / s_k)^(k_k - 1) exp(eta_k(x))``
  with partly non-linear log-hazards ``eta_k`` (thresholds, U-shapes).
* A shared gamma frailty (mean 1) adds unobserved per-drone heterogeneity.
* The observed failure is the earliest latent failure time across mechanisms.
* Censoring: administrative (staggered fleet entry, fixed study end, converted
  to flight hours via daily utilisation) plus random dropout (drone sold,
  lost, or retired for non-failure reasons).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS = ("quad_light", "quad_heavy", "hexa_heavy", "fixed_wing_vtol")
MODEL_PROBS = (0.40, 0.25, 0.20, 0.15)
MODEL_MAX_PAYLOAD_KG = {
    "quad_light": 1.0,
    "quad_heavy": 5.0,
    "hexa_heavy": 10.0,
    "fixed_wing_vtol": 3.0,
}
MODEL_BASE_VIBRATION_G = {
    "quad_light": 0.35,
    "quad_heavy": 0.50,
    "hexa_heavy": 0.45,
    "fixed_wing_vtol": 0.30,
}
MODEL_FLIGHT_MIN = {
    "quad_light": (25.0, 5.0),
    "quad_heavy": (30.0, 6.0),
    "hexa_heavy": (35.0, 7.0),
    "fixed_wing_vtol": (70.0, 15.0),
}
MODEL_DAILY_HOURS = {
    "quad_light": 1.2,
    "quad_heavy": 2.0,
    "hexa_heavy": 2.5,
    "fixed_wing_vtol": 3.0,
}

ENVIRONMENTS = ("urban", "desert", "coastal", "forest")
ENV_PROBS = (0.35, 0.20, 0.25, 0.20)
ENV_TEMP_C = {
    "urban": (15.0, 8.0),
    "desert": (32.0, 6.0),
    "coastal": (18.0, 5.0),
    "forest": (12.0, 7.0),
}
ENV_WIND_MS = {"urban": 4.0, "desert": 5.5, "coastal": 7.5, "forest": 3.5}
ENV_HUMIDITY_BETA = {
    "urban": (5.0, 5.0),
    "desert": (2.0, 8.0),
    "coastal": (8.0, 3.0),
    "forest": (6.0, 4.0),
}

FAILURE_MODES = ("motor", "battery", "electronics", "crash")
# Weibull baseline (shape k, scale s in flight hours) per failure mechanism:
# motor/battery wear out (k > 1), electronics show infant mortality (k < 1),
# crashes are roughly memoryless (k ~ 1).
WEIBULL_PARAMS = {
    "motor": (2.2, 2000.0),
    "battery": (1.6, 1700.0),
    "electronics": (0.8, 12000.0),
    "crash": (1.0, 5000.0),
}


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the simulation.

    Parameters
    ----------
    n_drones : int
        Number of drones (rows).
    study_days : int
        Length of the observation window in calendar days.
    effect_strength : float
        Multiplier applied to all covariate log-hazard ratios; controls how
        much signal the features carry (0 makes covariates uninformative).
    frailty_variance : float
        Variance of the shared gamma frailty; 0 disables it.
    dropout_mean_hours : float
        Mean of the exponential non-failure dropout time, in flight hours.
    seed : int | None
        Seed for ``numpy.random.default_rng``.
    study_start : str
        Calendar date of the start of the observation window.
    """

    n_drones: int = 5000
    study_days: int = 730
    effect_strength: float = 2.0
    frailty_variance: float = 0.3
    dropout_mean_hours: float = 5000.0
    seed: int | None = 42
    study_start: str = "2024-01-01"


def _relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ``max(x, 0)``."""
    return np.maximum(x, 0.0)


def _sample_covariates(rng: np.random.Generator, cfg: SimulationConfig) -> pd.DataFrame:
    """Sample baseline drone covariates with realistic dependencies."""
    n = cfg.n_drones
    model = rng.choice(MODELS, size=n, p=MODEL_PROBS)
    env = rng.choice(ENVIRONMENTS, size=n, p=ENV_PROBS)

    temp_mu, temp_sd = (np.array([ENV_TEMP_C[e][i] for e in env]) for i in (0, 1))
    mean_temp_c = rng.normal(temp_mu, temp_sd)

    wind_mean = np.array([ENV_WIND_MS[e] for e in env])
    avg_wind_ms = rng.gamma(shape=4.0, scale=wind_mean / 4.0)

    hum_a, hum_b = (np.array([ENV_HUMIDITY_BETA[e][i] for e in env]) for i in (0, 1))
    avg_humidity_pct = 100.0 * rng.beta(hum_a, hum_b)

    max_payload = np.array([MODEL_MAX_PAYLOAD_KG[m] for m in model])
    payload_ratio = np.clip(rng.beta(2.5, 3.0, size=n) * 1.2, 0.0, 1.2)
    avg_payload_kg = payload_ratio * max_payload

    fm_mu, fm_sd = (np.array([MODEL_FLIGHT_MIN[m][i] for m in model]) for i in (0, 1))
    avg_flight_min = np.clip(rng.normal(fm_mu, fm_sd), 5.0, None)

    operator_experience_yrs = rng.gamma(shape=2.0, scale=1.5, size=n)

    # Hard landings: more frequent with inexperienced pilots and strong wind.
    hl_rate = 2.0 * np.exp(-0.25 * operator_experience_yrs + 0.08 * avg_wind_ms)
    hard_landings_per_100h = rng.gamma(shape=3.0, scale=hl_rate / 3.0)

    base_vib = np.array([MODEL_BASE_VIBRATION_G[m] for m in model])
    mean_vibration_g = (base_vib + 0.03 * hard_landings_per_100h) * rng.lognormal(
        0.0, 0.15, size=n
    )

    is_refurbished = rng.random(n) < 0.15
    initial_battery_cycles = np.where(
        is_refurbished,
        rng.integers(80, 400, size=n),
        rng.integers(0, 30, size=n),
    )

    maintenance_interval_h = rng.choice(
        [25, 50, 100, 200], size=n, p=[0.2, 0.4, 0.3, 0.1]
    )
    firmware_major = rng.choice([3, 4, 5], size=n, p=[0.25, 0.45, 0.30])

    util_mean = np.array([MODEL_DAILY_HOURS[m] for m in model])
    daily_utilization_h = np.clip(
        rng.gamma(shape=5.0, scale=util_mean / 5.0), 0.1, 12.0
    )

    # Staggered fleet entry over the first 80% of the window.
    entry_day = rng.integers(0, int(0.8 * cfg.study_days), size=n)
    commission_date = pd.Timestamp(cfg.study_start) + pd.to_timedelta(
        entry_day, unit="D"
    )

    return pd.DataFrame(
        {
            "drone_id": [f"DR-{i:06d}" for i in range(n)],
            "model": pd.Categorical(model, categories=MODELS),
            "environment": pd.Categorical(env, categories=ENVIRONMENTS),
            "commission_date": commission_date,
            "is_refurbished": is_refurbished,
            "initial_battery_cycles": initial_battery_cycles,
            "firmware_major": firmware_major,
            "maintenance_interval_h": maintenance_interval_h,
            "operator_experience_yrs": operator_experience_yrs.round(2),
            "daily_utilization_h": daily_utilization_h.round(2),
            "avg_flight_min": avg_flight_min.round(1),
            "avg_payload_kg": avg_payload_kg.round(3),
            "payload_ratio": payload_ratio.round(3),
            "mean_temp_c": mean_temp_c.round(1),
            "avg_humidity_pct": avg_humidity_pct.round(1),
            "avg_wind_ms": avg_wind_ms.round(2),
            "hard_landings_per_100h": hard_landings_per_100h.round(2),
            "mean_vibration_g": mean_vibration_g.round(3),
        }
    )


def _log_hazards(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cause-specific log-hazard ratios ``eta_k(x)``, centred near zero for a typical drone."""
    model = df["model"].to_numpy()
    env = df["environment"].to_numpy()
    payload = df["payload_ratio"].to_numpy()
    temp = df["mean_temp_c"].to_numpy()
    hum = df["avg_humidity_pct"].to_numpy()
    wind = df["avg_wind_ms"].to_numpy()
    vib = df["mean_vibration_g"].to_numpy()
    exp_yrs = df["operator_experience_yrs"].to_numpy()
    hard = df["hard_landings_per_100h"].to_numpy()
    maint = df["maintenance_interval_h"].to_numpy()
    fw = df["firmware_major"].to_numpy()
    cycles = df["initial_battery_cycles"].to_numpy()
    flight_min = df["avg_flight_min"].to_numpy()
    refurb = df["is_refurbished"].to_numpy().astype(float)

    motor = (
        1.0 * (payload - 0.5)
        + 2.5 * _relu(payload - 0.9)  # sharp penalty for overloading
        + 2.0 * (vib - 0.45)
        + 0.5 * (env == "desert")  # dust ingress into bearings
        + 0.005 * (maint - 50)
        + 0.03 * _relu(temp - 30.0)
    )
    battery = (
        0.004 * (temp - 22.0) ** 2  # U-shape: both cold and heat degrade cells
        - 0.4
        + 0.003 * cycles
        + 0.6 * (payload - 0.5)
        + 0.015 * _relu(flight_min - 30.0)  # deep discharge on long sorties
        + 0.3 * refurb
        - 0.4 * (model == "fixed_wing_vtol")  # gliding reduces current draw
    )
    electronics = (
        0.025 * (hum - 50.0)
        + 0.6 * (env == "coastal")  # salt corrosion
        - 0.45 * (fw - 4)
        + 0.06 * _relu(temp - 35.0)
    )
    crash = (
        -0.8 * np.log1p(exp_yrs)
        + 0.8
        + 0.12 * (wind - 5.0)
        + 0.15 * (hard - 2.0)
        + 0.5 * (env == "forest")  # obstacles
        + 0.3 * (env == "urban")
        - 0.3 * (model == "fixed_wing_vtol")
        + 0.2 * (fw == 3)  # older flight-control firmware
    )
    return {
        "motor": motor,
        "battery": battery,
        "electronics": electronics,
        "crash": crash,
    }


def _sample_weibull_ph(
    rng: np.random.Generator, eta: np.ndarray, shape: float, scale: float
) -> np.ndarray:
    """Inverse-transform sampling from a Weibull PH model with log-HR ``eta``."""
    u = rng.random(eta.shape[0])
    return scale * (-np.log(u) * np.exp(-eta)) ** (1.0 / shape)


def generate_dataset(
    cfg: SimulationConfig, with_ground_truth: bool = False
) -> pd.DataFrame:
    """Generate the drone survival dataset.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    with_ground_truth : bool, default False
        If True, append latent quantities (true failure time and mode,
        frailty, per-cause log-hazards after ``effect_strength`` scaling)
        prefixed with ``true_``. These must not
        be used as features; they are for evaluating models (e.g. oracle C-index).

    Returns
    -------
    pandas.DataFrame
        One row per drone with covariates, ``duration_h`` (observed flight
        hours), ``event`` (1 = failure, 0 = censored) and ``failure_mode``
        (failure mechanism, or ``"censored"``).

    Raises
    ------
    ValueError
        If configuration values are out of range.
    """
    if cfg.n_drones <= 0:
        raise ValueError("n_drones must be positive")
    if cfg.study_days <= 1:
        raise ValueError("study_days must be > 1")
    if cfg.effect_strength < 0:
        raise ValueError("effect_strength must be non-negative")
    if cfg.frailty_variance < 0:
        raise ValueError("frailty_variance must be non-negative")
    if cfg.dropout_mean_hours <= 0:
        raise ValueError("dropout_mean_hours must be positive")

    rng = np.random.default_rng(cfg.seed)
    df = _sample_covariates(rng, cfg)
    n = len(df)

    if cfg.frailty_variance > 0:
        theta = cfg.frailty_variance
        frailty = rng.gamma(shape=1.0 / theta, scale=theta, size=n)
    else:
        frailty = np.ones(n)
    log_frailty = np.log(frailty)

    etas = {mode: cfg.effect_strength * eta for mode, eta in _log_hazards(df).items()}
    latent = np.column_stack(
        [
            _sample_weibull_ph(rng, etas[mode] + log_frailty, *WEIBULL_PARAMS[mode])
            for mode in FAILURE_MODES
        ]
    )
    cause_idx = latent.argmin(axis=1)
    failure_time = latent[np.arange(n), cause_idx]

    entry_day = (
        df["commission_date"] - pd.Timestamp(cfg.study_start)
    ).dt.days.to_numpy()
    admin_censor = (cfg.study_days - entry_day) * df["daily_utilization_h"].to_numpy()
    dropout_censor = rng.exponential(cfg.dropout_mean_hours, size=n)
    censor_time = np.minimum(admin_censor, dropout_censor)

    event = failure_time <= censor_time
    df["duration_h"] = np.minimum(failure_time, censor_time).round(2)
    df["event"] = event.astype(np.int8)
    df["failure_mode"] = pd.Categorical(
        np.where(event, np.asarray(FAILURE_MODES)[cause_idx], "censored"),
        categories=(*FAILURE_MODES, "censored"),
    )

    if with_ground_truth:
        df["true_failure_time_h"] = failure_time.round(2)
        df["true_failure_mode"] = pd.Categorical(
            np.asarray(FAILURE_MODES)[cause_idx], categories=FAILURE_MODES
        )
        df["true_frailty"] = frailty.round(4)
        for mode in FAILURE_MODES:
            df[f"true_eta_{mode}"] = etas[mode].round(4)

    return df


def _summarise(df: pd.DataFrame) -> None:
    """Log basic dataset statistics."""
    logger.info("Drones: %d, event rate: %.1f%%", len(df), 100 * df["event"].mean())
    logger.info(
        "Observed duration (h): median %.0f, p90 %.0f, max %.0f",
        df["duration_h"].median(),
        df["duration_h"].quantile(0.9),
        df["duration_h"].max(),
    )
    logger.info("Failure modes:\n%s", df["failure_mode"].value_counts().to_string())


def _parse_args() -> argparse.Namespace:
    defaults = SimulationConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--n-drones", type=int, default=defaults.n_drones)
    p.add_argument("--study-days", type=int, default=defaults.study_days)
    p.add_argument("--effect-strength", type=float, default=defaults.effect_strength)
    p.add_argument("--frailty-variance", type=float, default=defaults.frailty_variance)
    p.add_argument(
        "--dropout-mean-hours", type=float, default=defaults.dropout_mean_hours
    )
    p.add_argument("--study-start", default=defaults.study_start)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("-o", "--output", type=Path, default=Path("data/drone_survival.csv"))
    p.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="append latent true_* columns (not for use as features)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cfg = SimulationConfig(
        n_drones=args.n_drones,
        study_days=args.study_days,
        effect_strength=args.effect_strength,
        frailty_variance=args.frailty_variance,
        dropout_mean_hours=args.dropout_mean_hours,
        seed=args.seed,
        study_start=args.study_start,
    )
    df = generate_dataset(cfg, with_ground_truth=args.with_ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %s", args.output)
    _summarise(df)


if __name__ == "__main__":
    main()

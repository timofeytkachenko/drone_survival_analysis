"""Data-driven takeaways (Markdown sentences) shown under the figures."""

import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from dashboard.labels import (
    FAILURE_MODE_LABELS,
    FEATURE_LABELS,
    HAZARD_SCALE,
    METRIC_LABELS,
    MODEL_LABELS,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
)

USELESS_BRIER = 0.25
BONFERRONI_ALPHA = 0.05 / 4


def _km(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return kaplan_meier_estimator(
        data["event"].to_numpy().astype(bool), data["duration_h"].to_numpy()
    )


def km_median(data: pd.DataFrame) -> float:
    """Kaplan-Meier median time to failure; ``inf`` if S never drops to 0.5."""
    t, s = _km(data)
    below = s <= 0.5
    return float(t[below.argmax()]) if below.any() else np.inf


def km_at(data: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival at ``times`` (last value carried forward)."""
    t, s = _km(data)
    idx = np.searchsorted(t, times, side="right") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)


def hours_text(x: float) -> str:
    return "not reached" if not np.isfinite(x) else f"{x:.0f} h"


def _top_share(share: float) -> str:
    """``0.004`` -> ``"1%"``: never claim a share smaller than one percent."""
    return f"{max(share, 0.01):.0%}"


def _pp(x: float) -> str:
    """Signed difference of probabilities in percentage points."""
    return f"{x * 100:+.1f} pp"


def _ordinal(n: int) -> str:
    """``1`` -> ``"1st"``, ``12`` -> ``"12th"``, ``22`` -> ``"22nd"``."""
    suffix = (
        "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


# --- Overview -------------------------------------------------------------------


def overview(pred: pd.DataFrame) -> list[str]:
    """Fleet-level survival and failure-mode facts."""
    median = km_median(pred)
    checkpoints = [t for t in (100, 250, 500) if t < pred["duration_h"].max()]
    surv = km_at(pred, np.array(checkpoints, dtype=float))
    out = [
        f"Half of the test drones fail by **{hours_text(median)}** of flight "
        "(Kaplan-Meier median)."
        if np.isfinite(median)
        else "Fewer than half of the test drones failed during the observation period.",
        "Drones still without failure: "
        + ", ".join(f"at {t} h, **{s:.0%}**" for t, s in zip(checkpoints, surv))
        + ".",
    ]
    modes = pred.loc[pred["event"] == 1, "failure_mode"].value_counts(normalize=True)
    if len(modes) >= 2:
        out.append(
            f"The most frequent failure is **{FAILURE_MODE_LABELS[modes.index[0]].lower()}** "
            f"({modes.iloc[0]:.0%} of failures), followed by "
            f"{FAILURE_MODE_LABELS[modes.index[1]].lower()} ({modes.iloc[1]:.0%})."
        )
    return out


def _rate(h: float) -> str:
    """Hazard per flight hour as a rate per 100 flight hours."""
    return f"{HAZARD_SCALE * h:.2f} per 100 flight hours"


def hazard_trend(table: pd.DataFrame) -> list[str]:
    """Level of the observed hazard and whether it rises or falls with age.

    The trend counts only when the first-third mean lies outside the mean 95%
    interval of the last third, so a wide right tail does not read as a trend.
    """
    obs = table[table["source"] == "observed"]
    h, t = obs["h"].to_numpy(), obs["t"].to_numpy()
    third = max(len(h) // 3, 1)
    early, late = h[:third].mean(), h[-third:].mean()
    late_low, late_high = obs["ci_low"].iloc[-third:].mean(), obs["ci_high"].iloc[-third:].mean()
    share = 1 - np.exp(-HAZARD_SCALE * h.mean())
    out = [
        (
            f"Between {t[0]:.0f} and {t[-1]:.0f} h the observed failure rate averages "
            f"**{_rate(h.mean())}**: about {share:.0%} of the drones still flying fail "
            "within the next 100 flight hours."
        )
    ]
    change = f"from {_rate(early)} early on to {_rate(late)} at the end of this range"
    if late > early and early < late_low:
        out.append(
            f"The rate **rises** with age ({change}): wear-out. Older drones fail more "
            "often, so maintenance or replacement by flight hours pays off."
        )
    elif late < early and early > late_high:
        out.append(
            f"The rate **falls** with age ({change}): early failures dominate. "
            "Acceptance checks and burn-in of new drones pay off more than replacement "
            "by age."
        )
    else:
        out.append(
            f"There is **no clear trend** with age ({change}, within the uncertainty "
            "band): failures look random with respect to flight hours. Replacing a drone "
            "just because it is old buys little; target maintenance by the drone's risk "
            "factors instead (\"Fleet\" tab)."
        )
    return out


# --- Segments -------------------------------------------------------------------


def segments(pred: pd.DataFrame, segment: str, levels: list) -> list[str]:
    """Log-rank test, survival at the fleet median and failure profile per level."""
    data = pred[pred[segment].isin(levels)]
    out = []
    if len(levels) >= 2:
        y = Surv.from_arrays(
            data["event"].to_numpy().astype(bool), data["duration_h"].to_numpy()
        )
        chisq, p_value = compare_survival(y, data[segment].astype(str).to_numpy())
        verdict = (
            "**statistically significant**" if p_value < 0.05 else "**not significant**"
        )
        out.append(
            f"Survival differences between segments are {verdict}: log-rank "
            f"χ² = {chisq:.1f}, p = {p_value:.2g}."
        )
    t_ref = km_median(pred)
    if np.isfinite(t_ref):
        surv = {
            lv: km_at(data[data[segment] == lv], np.array([t_ref]))[0] for lv in levels
        }
        worst, best = min(surv, key=surv.get), max(surv, key=surv.get)
        out.append(
            f"Still without failure at {t_ref:.0f} h (fleet median): "
            + ", ".join(f"{lv}: {s:.0%}" for lv, s in surv.items())
            + f". Worst: **{worst}**, best: **{best}**."
        )
    profile = []
    for lv in levels:
        modes = data.loc[(data[segment] == lv) & (data["event"] == 1), "failure_mode"]
        if len(modes):
            share = modes.value_counts(normalize=True)
            profile.append(
                f"{lv}: {FAILURE_MODE_LABELS[share.index[0]].lower()} ({share.iloc[0]:.0%})"
            )
    if profile:
        out.append("Main failure cause: " + "; ".join(profile) + ".")
    return out


# --- Model quality ----------------------------------------------------------------


def metrics(table: pd.DataFrame, best: str) -> list[str]:
    """Interpretation of the metric table."""
    m = table.set_index(["model", "split"])
    test, val = m.loc[(best, "test")], m.loc[(best, "val")]
    name = MODEL_LABELS[best]
    out = [
        (
            f"{name} on test: C-index (Uno) = **{test['c_index_ipcw']:.3f}**: in "
            f"{test['c_index_ipcw']:.0%} of comparable drone pairs the model correctly "
            "identifies which one fails first."
        ),
        (
            f"IBS = **{test['ibs']:.3f}** versus {USELESS_BRIER} for a 50/50 forecast: "
            f"the probability error is {1 - test['ibs'] / USELESS_BRIER:.0%} lower."
        ),
        f"From validation to test, C-index (Uno) changed from {val['c_index_ipcw']:.3f} "
        f"to {test['c_index_ipcw']:.3f} and IBS from {val['ibs']:.3f} to "
        f"{test['ibs']:.3f}. Test drones are newer and have fewer flight hours, so some "
        "loss of discrimination on them is expected; IBS stays at the validation level, "
        "so the model estimates probabilities just as accurately."
        if abs(test["ibs"] - val["ibs"]) < 0.01
        else f"From validation to test, C-index (Uno) changed from "
        f"{val['c_index_ipcw']:.3f} to {test['c_index_ipcw']:.3f} and IBS from "
        f"{val['ibs']:.3f} to {test['ibs']:.3f}.",
    ]
    others = [k for k in m.index.get_level_values("model").unique() if k != best]
    for other in others:
        o = m.loc[(other, "test")]
        wins = [
            METRIC_LABELS[k]
            for k in METRIC_LABELS
            if (test[k] < o[k] if k == "ibs" else test[k] > o[k])
        ]
        losses = [
            METRIC_LABELS[k] for k in METRIC_LABELS if METRIC_LABELS[k] not in wins
        ]
        out.append(
            f"On test, {name} beats {MODEL_LABELS[other]} on: "
            f"{', '.join(wins) or '—'}; loses on: {', '.join(losses) or '—'}. "
            "See the bootstrap comparison below for significance."
        )
    return out


def learning(models: dict) -> list[str]:
    """Best iteration and validation-loss improvement per model."""
    out = []
    for kind, model in models.items():
        ((metric, values),) = model.history["valid"].items()
        best = model.booster.best_iteration
        start, end = values[0], values[best]
        out.append(
            f"{MODEL_LABELS[kind]}: best iteration **{best}** of {len(values)} "
            f"trained; validation loss ({metric}) went from {start:.3f} to "
            f"{end:.3f} ({(end - start) / abs(start):+.0%}). The next "
            f"{len(values) - best - 1} iterations brought no improvement, so "
            "training stopped."
        )
    return out


def mean_vs_km(table: pd.DataFrame, t_max: float) -> list[str]:
    """Largest gap between mean predicted survival and Kaplan-Meier up to ``t_max``.

    The Kaplan-Meier tail rests on few drones, so the comparison is cut at the
    evaluation range used for the metrics.
    """
    km = table[table["source"] == "km"]
    out = []
    for kind in ("cox", "aft"):
        part = table[(table["source"] == kind) & (table["t"] <= t_max)]
        idx = (
            np.searchsorted(km["t"].to_numpy(), part["t"].to_numpy(), side="right") - 1
        )
        valid = idx >= 0
        s_km = km["s"].to_numpy()[idx[valid]]
        lo, hi = (
            km["ci_low"].to_numpy()[idx[valid]],
            km["ci_high"].to_numpy()[idx[valid]],
        )
        s = part["s"].to_numpy()[valid]
        gap = s - s_km
        worst = np.abs(gap).argmax()
        inside = ((s >= lo) & (s <= hi)).mean()
        tendency = "more optimistic" if gap.mean() > 0 else "more pessimistic"
        out.append(
            f"{MODEL_LABELS[kind]} (up to {t_max:.0f} h): the mean curve lies inside "
            f"the 95% interval of the observed curve on **{inside:.0%}** of this range; "
            f"largest gap {_pp(gap[worst])} at "
            f"{part['t'].to_numpy()[valid][worst]:.0f} h; on average the model is "
            f"{tendency} than the data."
        )
    out.append(
        f"Beyond {t_max:.0f} h the observed curve rests on few drones (wide band), so "
        "gaps there carry little information."
    )
    return out


def hazard_fit(table: pd.DataFrame) -> list[str]:
    """How closely each model's group hazard follows the observed one."""
    obs = table[table["source"] == "observed"]
    out = []
    for kind in ("cox", "aft"):
        h = table.loc[table["source"] == kind, "h"].to_numpy()
        inside = ((h >= obs["ci_low"].to_numpy()) & (h <= obs["ci_high"].to_numpy())).mean()
        bias = h.mean() / obs["h"].mean() - 1
        tendency = "higher" if bias > 0 else "lower"
        out.append(
            f"{MODEL_LABELS[kind]}: the predicted rate lies inside the 95% band of the "
            f"observed one on **{inside:.0%}** of the range and is on average "
            f"{abs(bias):.0%} {tendency} than observed."
        )
    return out


def brier(table: pd.DataFrame) -> list[str]:
    """Peak Brier score per model."""
    out = []
    for kind, part in table.groupby("model", sort=False):
        peak = part.loc[part["brier"].idxmax()]
        out.append(
            f"{MODEL_LABELS[kind]}: peak error {peak['brier']:.3f} at "
            f"{peak['t']:.0f} h, the most uncertain period when most drones fail; "
            "below the 0.25 benchmark everywhere."
            if part["brier"].max() < USELESS_BRIER
            else f"{MODEL_LABELS[kind]}: at {peak['t']:.0f} h the error {peak['brier']:.3f} "
            "reaches the level of a useless forecast."
        )
    return out


def auc(table: pd.DataFrame) -> list[str]:
    """AUC(t) range per model."""
    out = []
    for kind, part in table.groupby("model", sort=False):
        lo, hi = part.loc[part["auc"].idxmin()], part.loc[part["auc"].idxmax()]
        out.append(
            f"{MODEL_LABELS[kind]}: AUC from {lo['auc']:.3f} (at {lo['t']:.0f} h) to "
            f"{hi['auc']:.3f} (at {hi['t']:.0f} h); the model is consistently better "
            "than random at every horizon."
            if lo["auc"] > 0.55
            else f"{MODEL_LABELS[kind]}: at {lo['t']:.0f} h AUC drops to "
            f"{lo['auc']:.3f}; at this horizon the model barely separates drones."
        )
    return out


def calibration(table: pd.DataFrame, horizon: float) -> list[str]:
    """Weighted absolute and signed calibration error per model."""
    part = table[np.isclose(table["horizon"], horizon)]
    out = []
    for kind, rows in part.groupby("model", sort=False):
        w = rows["n"] / rows["n"].sum()
        gap = rows["predicted"] - rows["observed"]
        mae, bias = float((w * gap.abs()).sum()), float((w * gap).sum())
        direction = (
            "overestimates reliability (predicted survival above observed)"
            if bias > 0
            else "underestimates reliability (predicted survival below observed)"
        )
        out.append(
            f"{MODEL_LABELS[kind]}: mean calibration error at {horizon:.0f} h is "
            f"**{mae * 100:.1f} pp**; on average the model {direction} by "
            f"{abs(bias) * 100:.1f} pp."
        )
    return out


def bootstrap(summary: pd.DataFrame) -> list[str]:
    """Plain-language verdict per metric."""
    out = []
    for row in summary.itertuples():
        a, b = MODEL_LABELS[row.model_a], MODEL_LABELS[row.model_b]
        if row.significant:
            strict = (
                " and passes the strict threshold corrected for multiple comparisons"
                if row.p_value < BONFERRONI_ALPHA
                else ", but does not pass the strict Bonferroni threshold (0.0125)"
            )
            verdict = (
                f"**{MODEL_LABELS[row.better]} is significantly better** "
                f"(p = {row.p_value:.3f}){strict}"
            )
        else:
            verdict = f"the difference is **not significant** (p = {row.p_value:.3f})"
        out.append(
            f"{METRIC_LABELS[row.metric]}: {a} − {b} = {row.diff:+.4f}, 95% CI "
            f"[{row.ci_low:+.4f}; {row.ci_high:+.4f}]: {verdict}."
        )
    out.append(
        "Bottom line: the models are close in quality; the best one is chosen by "
        "validation IBS, not by test, so the test stays an independent check."
    )
    return out


# --- Risk factors ------------------------------------------------------------------


def shap_top(importance: pd.DataFrame, kind: str, n: int = 5) -> list[str]:
    """Top features of ``kind`` and their agreement with the other model."""
    ranked = importance[importance["model"] == kind].nlargest(n, "share")
    others = {
        k: set(g.nlargest(n, "share")["feature"])
        for k, g in importance.groupby("model")
        if k != kind
    }
    out = [
        f"Top {n} factors of {MODEL_LABELS[kind]}: "
        + ", ".join(
            f'"{FEATURE_LABELS[f]}" ({s:.0%})'
            for f, s in zip(ranked["feature"], ranked["share"])
        )
        + f"; together **{ranked['share'].sum():.0%}** of the influence."
    ]
    for other, top in others.items():
        common = [FEATURE_LABELS[f] for f in ranked["feature"] if f in top]
        out.append(
            f"{len(common)} of the top {n} factors match the {MODEL_LABELS[other]} "
            f"model ({', '.join(common) or '—'}): the larger the overlap, the more "
            "reliable the conclusion."
        )
    return out


def shap_direction(
    shap_values: pd.DataFrame, kind: str, feature: str, categorical: bool
) -> list[str]:
    """Where the feature raises and lowers risk."""
    part = shap_values[
        (shap_values["model"] == kind) & (shap_values["feature"] == feature)
    ]
    label = FEATURE_LABELS[feature]
    spread = (
        f"The contribution to risk ranges from {part['shap'].min():+.3f} "
        f"to {part['shap'].max():+.3f}."
    )
    if categorical:
        means = (
            part.groupby("display")["shap"].agg(["mean", "size"]).sort_values("mean")
        )
        return [
            (
                f'Highest risk at "{label}" = **{means.index[-1]}** (mean contribution '
                f"{means['mean'].iloc[-1]:+.3f}, {int(means['size'].iloc[-1])} drones), "
                f"lowest at **{means.index[0]}** ({means['mean'].iloc[0]:+.3f}, "
                f"{int(means['size'].iloc[0])} drones)."
            ),
            spread,
        ]
    data = part.dropna(subset=["value"])
    bins = pd.qcut(data["value"], 5, duplicates="drop")
    means = data.groupby(bins, observed=True)["shap"].mean()
    rho = data["value"].corr(data["shap"], method="spearman")
    if rho > 0.5:
        shape = f'Monotonic relationship: **the higher "{label}", the higher the risk** (ρ = {rho:.2f}).'
    elif rho < -0.5:
        shape = f'Monotonic relationship: **the higher "{label}", the lower the risk** (ρ = {rho:.2f}).'
    else:
        shape = (
            f"**Non-monotonic** relationship (ρ = {rho:.2f}): risk depends on the range "
            "of values rather than simply rising or falling; look at the shape of the cloud."
        )

    lowest = data["value"].min()

    def interval(iv: pd.Interval) -> str:
        return f"{max(iv.left, lowest):.3g}–{iv.right:.3g}"

    return [
        shape,
        (
            f"Highest mean contribution ({means.max():+.3f}) at values "
            f"**{interval(means.idxmax())}**, lowest ({means.min():+.3f}) at "
            f"**{interval(means.idxmin())}**."
        ),
        spread,
    ]


# --- Fleet and drone card -------------------------------------------------------------


def fleet(active: pd.DataFrame, window: float, horizon: float) -> list[str]:
    """Operational summary of drones in service."""
    counts = active["priority"].value_counts()
    finite = active.loc[np.isfinite(active["rul_median"]), "rul_median"]
    expected = active["p_fail_window"].sum()
    above_10 = (active["p_fail_window"] > 0.1).mean()
    return [
        (
            f"{len(active)} drones in service: high maintenance priority "
            f"**{counts.get(PRIORITY_HIGH, 0)}**, medium **{counts.get(PRIORITY_MEDIUM, 0)}**, "
            f"low {counts.get(PRIORITY_LOW, 0)}."
        ),
        (
            f"Expected number of failures in the fleet over the next {window:.0f} flight "
            f"hours (sum of probabilities): **≈ {expected:.0f}** drones. Use it as a "
            "benchmark for spare parts and backup drones."
        ),
        (
            f"{above_10:.0%} of drones in service have a failure probability above 10% "
            f"in the next {window:.0f} h: failures are spread across the whole fleet "
            'rather than concentrated in a few "bad" drones.'
        ),
        (
            f"Fleet median remaining life is {finite.median():.0f} h; drones with a "
            f"median beyond the {horizon:.0f} h horizon: {len(active) - len(finite)}."
        ),
    ]


def backtest(table: pd.DataFrame, landmark: float) -> list[str]:
    """Largest gap between predicted conditional survival and Kaplan-Meier."""
    km = table[table["source"] == "km"]
    out = []
    for kind in ("cox", "aft"):
        part = table[(table["source"] == kind) & (table["u"] <= km["u"].max())]
        idx = (
            np.searchsorted(km["u"].to_numpy(), part["u"].to_numpy(), side="right") - 1
        )
        s_km = np.where(idx >= 0, km["s"].to_numpy()[np.clip(idx, 0, None)], 1.0)
        gap = part["s"].to_numpy() - s_km
        worst = np.abs(gap).argmax()
        out.append(
            f"{MODEL_LABELS[kind]}: largest gap between predicted and observed remaining "
            f"life after {landmark:.0f} h is {_pp(gap[worst])} (after "
            f"{part['u'].to_numpy()[worst]:.0f} h), mean gap "
            f"{np.abs(gap).mean() * 100:.1f} pp."
        )
    return out


def drone(
    row: pd.Series,
    contrib: pd.DataFrame,
    pred: pd.DataFrame,
    best: str,
    window: float,
    predicted_median: float,
) -> list[str]:
    """Narrative explanation of one drone's prediction."""
    out = []
    top = (pred[f"risk_{best}"] >= row[f"risk_{best}"]).mean()
    if row["event"] == 1:
        mode = FAILURE_MODE_LABELS.get(row["failure_mode"], row["failure_mode"])
        timing = (
            ("before" if row["duration_h"] < predicted_median else "after")
            if np.isfinite(predicted_median)
            else "—"
        )
        out.append(
            f"The drone **failed** at {row['duration_h']:.0f} flight hours (cause: "
            f"{mode.lower()}). For a drone with these characteristics the model expected "
            f"a median time to failure of {hours_text(predicted_median)}; the failure "
            f"happened {timing} the median."
        )
    else:
        active = pred[pred["event"] == 0]
        rank = int((active["p_fail_window"] > row["p_fail_window"]).sum()) + 1
        out.append(
            f"The drone is **in service** with {row['duration_h']:.0f} h flown without "
            f"failure. Failure probability in the next {window:.0f} h is "
            f"**{row['p_fail_window']:.0%}**: {_ordinal(rank)} of {len(active)} drones "
            f"in service on this measure; maintenance priority **{row['priority']}**."
        )
    bottom = (pred[f"risk_{best}"] <= row[f"risk_{best}"]).mean()
    position = (
        f"the **{_top_share(top)}** riskiest"
        if top <= 0.5
        else f"the **{_top_share(bottom)}** least risky"
    )
    out.append(
        f"By the model's risk score the drone is among {position} test drones (risk "
        f"group: {row['risk_group']})."
    )
    up = contrib[contrib["shap"] > 0].nlargest(3, "shap")
    down = contrib[contrib["shap"] < 0].nsmallest(3, "shap")
    if len(up):
        out.append(
            "Factors that **raise** risk the most: "
            + "; ".join(
                f"{FEATURE_LABELS[f]} = {v} ({s:+.3f})"
                for f, v, s in zip(up["feature"], up["display"], up["shap"])
            )
            + "."
        )
    if len(down):
        out.append(
            "Factors that **lower** risk the most: "
            + "; ".join(
                f"{FEATURE_LABELS[f]} = {v} ({s:+.3f})"
                for f, v, s in zip(down["feature"], down["display"], down["shap"])
            )
            + "."
        )
    return out


def drone_hazard(age: float, rate: float, fleet_rate: float, best: str) -> str:
    """Current failure rate of a drone in service, relative to the fleet at that age."""
    text = (
        f"At its current {age:.0f} flight hours the drone's failure rate is "
        f"**{_rate(rate)}** ({MODEL_LABELS[best]})"
    )
    if fleet_rate <= 0:
        return text + "."
    return (
        text + f": **{rate / fleet_rate:.1f}×** the observed rate of test drones at the "
        f"same age ({_rate(fleet_rate)})."
    )


def whatif(
    base: pd.Series,
    new: pd.Series,
    changes: list[tuple[str, object, object]],
    window: float,
) -> list[str]:
    """Narrative comparison of the current and scenario remaining-life summaries."""
    if not changes:
        return ["No inputs changed: the scenario matches the current conditions."]
    out = [
        "Changed: "
        + "; ".join(f"{FEATURE_LABELS[f]}: {old} → {new_}" for f, old, new_ in changes)
        + "."
    ]
    dp = new["p_fail_window"] - base["p_fail_window"]
    effect = "lowers" if dp < 0 else "raises"
    out.append(
        f"The scenario {effect} the failure probability in the next {window:.0f} h from "
        f"{base['p_fail_window']:.1%} to **{new['p_fail_window']:.1%}** ({_pp(dp)})."
    )
    out.append(
        f"Expected remaining life changes from {base['rmrl']:.0f} to "
        f"**{new['rmrl']:.0f} h** ({new['rmrl'] - base['rmrl']:+.0f} h); median from "
        f"{hours_text(base['rul_median'])} to {hours_text(new['rul_median'])}."
    )
    out.append(
        "This is the model's forecast with everything else held equal, not a guaranteed "
        "effect: the model learned associations from historical data, not causal mechanisms."
    )
    return out

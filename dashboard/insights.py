"""Data-driven takeaways (Markdown sentences) shown under the figures."""

import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from dashboard.labels import HAZARD_SCALE, PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_MEDIUM
from dashboard.locale import bundle, tr

USELESS_BRIER = 0.25
BONFERRONI_ALPHA = 0.05 / 4


def _lab(lang: str, group: str, key: str) -> str:
    labels = bundle(lang)[group]
    return labels.get(key, str(key))


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


def hours_text(x: float, lang: str = "en") -> str:
    """Hours, or the localized 'not reached' when the level is never crossed."""
    if not np.isfinite(x):
        return tr(lang, "phrases.not_reached")
    return tr(lang, "ui.hours_value", hours=x)


def _top_share(share: float) -> str:
    """``0.004`` -> ``"1%"``: never claim a share smaller than one percent."""
    return f"{max(share, 0.01):.0%}"


def _pp(x: float, lang: str) -> str:
    """Signed difference of probabilities in percentage points."""
    return tr(lang, "phrases.pp", value=x * 100)


def _ordinal(n: int, lang: str) -> str:
    """English ordinal; Russian templates add their own suffix to the number."""
    if lang == "ru":
        return str(n)
    suffix = (
        "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def _rate(h: float, lang: str) -> str:
    """Hazard per flight hour as a rate per 100 flight hours."""
    return tr(lang, "phrases.per_100h", rate=HAZARD_SCALE * h)


def level_text(lang: str, segment: str, level: object) -> str:
    """Display label of a segment level. Data codes other than priority stay as stored."""
    if segment in ("priority", "risk_group"):
        return _lab(lang, "priority_labels", str(level))
    if segment == "is_refurbished":
        flag = level if isinstance(level, bool) else str(level).lower() in {"true", "1", "yes"}
        return tr(lang, "ui.yes" if flag else "ui.no")
    return str(level)


# --- Overview -------------------------------------------------------------------


def overview(pred: pd.DataFrame, lang: str = "en") -> list[str]:
    """Fleet-level survival and failure-mode facts."""
    median = km_median(pred)
    checkpoints = [t for t in (100, 250, 500) if t < pred["duration_h"].max()]
    surv = km_at(pred, np.array(checkpoints, dtype=float))
    out = [
        tr(lang, "phrases.overview_median", hours=hours_text(median, lang))
        if np.isfinite(median)
        else tr(lang, "phrases.overview_no_median"),
        tr(
            lang,
            "phrases.overview_still",
            items=", ".join(
                tr(lang, "phrases.overview_still_item", t=t, share=s)
                for t, s in zip(checkpoints, surv)
            ),
        ),
    ]
    modes = pred.loc[pred["event"] == 1, "failure_mode"].value_counts(normalize=True)
    if len(modes) >= 2:
        out.append(
            tr(
                lang,
                "phrases.overview_modes",
                first=_lab(lang, "failure_mode_labels", modes.index[0]).lower(),
                share0=modes.iloc[0],
                second=_lab(lang, "failure_mode_labels", modes.index[1]).lower(),
                share1=modes.iloc[1],
            )
        )
    return out


def hazard_trend(table: pd.DataFrame, lang: str = "en") -> list[str]:
    """Level of the observed hazard and whether it rises or falls with age.

    The trend counts only when the first-third mean lies outside the mean 95%
    interval of the last third, so a wide right tail does not read as a trend.
    """
    obs = table[table["source"] == "observed"]
    h, t = obs["h"].to_numpy(), obs["t"].to_numpy()
    third = max(len(h) // 3, 1)
    early, late = h[:third].mean(), h[-third:].mean()
    late_low = obs["ci_low"].iloc[-third:].mean()
    late_high = obs["ci_high"].iloc[-third:].mean()
    share = 1 - np.exp(-HAZARD_SCALE * h.mean())
    out = [
        tr(
            lang,
            "phrases.hazard_level",
            t0=t[0],
            t1=t[-1],
            rate=_rate(h.mean(), lang),
            share=share,
        )
    ]
    change = tr(
        lang,
        "phrases.hazard_change",
        early=_rate(early, lang),
        late=_rate(late, lang),
    )
    if late > early and early < late_low:
        out.append(tr(lang, "phrases.hazard_rises", change=change))
    elif late < early and early > late_high:
        out.append(tr(lang, "phrases.hazard_falls", change=change))
    else:
        out.append(tr(lang, "phrases.hazard_flat", change=change))
    return out


# --- Segments -------------------------------------------------------------------


def segments(pred: pd.DataFrame, segment: str, levels: list, lang: str = "en") -> list[str]:
    """Log-rank test, survival at the fleet median and failure profile per level."""
    data = pred[pred[segment].isin(levels)]
    out = []
    if len(levels) >= 2:
        y = Surv.from_arrays(
            data["event"].to_numpy().astype(bool), data["duration_h"].to_numpy()
        )
        chisq, p_value = compare_survival(y, data[segment].astype(str).to_numpy())
        verdict = tr(
            lang, "phrases.verdict_sig" if p_value < 0.05 else "phrases.verdict_ns"
        )
        out.append(
            tr(lang, "phrases.segments_sig", verdict=verdict, chi=chisq, p=p_value)
        )
    t_ref = km_median(pred)
    if np.isfinite(t_ref):
        surv = {
            lv: km_at(data[data[segment] == lv], np.array([t_ref]))[0] for lv in levels
        }
        worst, best = min(surv, key=surv.get), max(surv, key=surv.get)
        out.append(
            tr(
                lang,
                "phrases.segments_at",
                t=t_ref,
                items=", ".join(
                    f"{level_text(lang, segment, lv)}: {s:.0%}" for lv, s in surv.items()
                ),
                worst=level_text(lang, segment, worst),
                best=level_text(lang, segment, best),
            )
        )
    profile = []
    for lv in levels:
        modes = data.loc[(data[segment] == lv) & (data["event"] == 1), "failure_mode"]
        if len(modes):
            share = modes.value_counts(normalize=True)
            profile.append(
                f"{level_text(lang, segment, lv)}: "
                f"{_lab(lang, 'failure_mode_labels', share.index[0]).lower()} "
                f"({share.iloc[0]:.0%})"
            )
    if profile:
        out.append(tr(lang, "phrases.segments_causes", items="; ".join(profile)))
    return out


# --- Model quality ----------------------------------------------------------------


def metrics(table: pd.DataFrame, best: str, lang: str = "en") -> list[str]:
    """Interpretation of the metric table."""
    m = table.set_index(["model", "split"])
    test, val = m.loc[(best, "test")], m.loc[(best, "val")]
    name = _lab(lang, "model_labels", best)
    out = [
        tr(lang, "phrases.metrics_c", name=name, c=test["c_index_ipcw"], share=test["c_index_ipcw"]),
        tr(
            lang,
            "phrases.metrics_ibs",
            ibs=test["ibs"],
            useless=USELESS_BRIER,
            drop=1 - test["ibs"] / USELESS_BRIER,
        ),
        tr(
            lang,
            "phrases.metrics_shift_flat"
            if abs(test["ibs"] - val["ibs"]) < 0.01
            else "phrases.metrics_shift",
            c0=val["c_index_ipcw"],
            c1=test["c_index_ipcw"],
            i0=val["ibs"],
            i1=test["ibs"],
        ),
    ]
    others = [k for k in m.index.get_level_values("model").unique() if k != best]
    for other in others:
        o = m.loc[(other, "test")]
        wins = [
            _lab(lang, "metric_labels", k)
            for k in bundle(lang)["metric_labels"]
            if (test[k] < o[k] if k == "ibs" else test[k] > o[k])
        ]
        losses = [
            _lab(lang, "metric_labels", k)
            for k in bundle(lang)["metric_labels"]
            if _lab(lang, "metric_labels", k) not in wins
        ]
        out.append(
            tr(
                lang,
                "phrases.metrics_beats",
                name=name,
                other=_lab(lang, "model_labels", other),
                wins=", ".join(wins) or "—",
                losses=", ".join(losses) or "—",
            )
        )
    return out


def learning(models: dict, lang: str = "en") -> list[str]:
    """Best iteration and validation-loss improvement per model."""
    out = []
    for kind, model in models.items():
        ((metric, values),) = model.history["valid"].items()
        best = model.booster.best_iteration
        start, end = values[0], values[best]
        out.append(
            tr(
                lang,
                "phrases.learning",
                name=_lab(lang, "model_labels", kind),
                best=best,
                n=len(values),
                metric=metric,
                start=start,
                end=end,
                change=(end - start) / abs(start),
                rest=len(values) - best - 1,
            )
        )
    return out


def mean_vs_km(table: pd.DataFrame, t_max: float, lang: str = "en") -> list[str]:
    """Largest gap between mean predicted survival and Kaplan-Meier up to ``t_max``.

    The Kaplan-Meier tail rests on few drones, so the comparison is cut at the
    evaluation range used for the metrics.
    """
    km = table[table["source"] == "km"]
    out = []
    for kind in ("cox", "aft"):
        part = table[(table["source"] == kind) & (table["t"] <= t_max)]
        idx = np.searchsorted(km["t"].to_numpy(), part["t"].to_numpy(), side="right") - 1
        valid = idx >= 0
        s_km = km["s"].to_numpy()[idx[valid]]
        lo, hi = km["ci_low"].to_numpy()[idx[valid]], km["ci_high"].to_numpy()[idx[valid]]
        s = part["s"].to_numpy()[valid]
        gap = s - s_km
        worst = np.abs(gap).argmax()
        inside = ((s >= lo) & (s <= hi)).mean()
        tendency = tr(
            lang, "phrases.optimistic" if gap.mean() > 0 else "phrases.pessimistic"
        )
        out.append(
            tr(
                lang,
                "phrases.mean_gap",
                name=_lab(lang, "model_labels", kind),
                t=t_max,
                inside=inside,
                gap=_pp(gap[worst], lang),
                at=part["t"].to_numpy()[valid][worst],
                tendency=tendency,
            )
        )
    out.append(tr(lang, "phrases.mean_tail", t=t_max))
    return out


def hazard_fit(table: pd.DataFrame, lang: str = "en") -> list[str]:
    """How closely each model's group hazard follows the observed one."""
    obs = table[table["source"] == "observed"]
    out = []
    for kind in ("cox", "aft"):
        h = table.loc[table["source"] == kind, "h"].to_numpy()
        inside = ((h >= obs["ci_low"].to_numpy()) & (h <= obs["ci_high"].to_numpy())).mean()
        bias = h.mean() / obs["h"].mean() - 1
        out.append(
            tr(
                lang,
                "phrases.hazard_fit",
                name=_lab(lang, "model_labels", kind),
                inside=inside,
                bias=abs(bias),
                tendency=tr(lang, "phrases.higher" if bias > 0 else "phrases.lower"),
            )
        )
    return out


def brier(table: pd.DataFrame, lang: str = "en") -> list[str]:
    """Peak Brier score per model."""
    out = []
    for kind, part in table.groupby("model", sort=False):
        peak = part.loc[part["brier"].idxmax()]
        key = "phrases.brier_ok" if part["brier"].max() < USELESS_BRIER else "phrases.brier_bad"
        out.append(
            tr(
                lang,
                key,
                name=_lab(lang, "model_labels", kind),
                brier=peak["brier"],
                t=peak["t"],
            )
        )
    return out


def auc(table: pd.DataFrame, lang: str = "en") -> list[str]:
    """AUC(t) range per model."""
    out = []
    for kind, part in table.groupby("model", sort=False):
        lo, hi = part.loc[part["auc"].idxmin()], part.loc[part["auc"].idxmax()]
        if lo["auc"] > 0.55:
            out.append(
                tr(
                    lang,
                    "phrases.auc_ok",
                    name=_lab(lang, "model_labels", kind),
                    lo=lo["auc"],
                    t0=lo["t"],
                    hi=hi["auc"],
                    t1=hi["t"],
                )
            )
        else:
            out.append(
                tr(
                    lang,
                    "phrases.auc_bad",
                    name=_lab(lang, "model_labels", kind),
                    t=lo["t"],
                    auc=lo["auc"],
                )
            )
    return out


def calibration(table: pd.DataFrame, horizon: float, lang: str = "en") -> list[str]:
    """Weighted absolute and signed calibration error per model."""
    part = table[np.isclose(table["horizon"], horizon)]
    out = []
    for kind, rows in part.groupby("model", sort=False):
        w = rows["n"] / rows["n"].sum()
        gap = rows["predicted"] - rows["observed"]
        mae, bias = float((w * gap.abs()).sum()), float((w * gap).sum())
        out.append(
            tr(
                lang,
                "phrases.calibration",
                name=_lab(lang, "model_labels", kind),
                h=horizon,
                mae=mae * 100,
                direction=tr(
                    lang, "phrases.calib_over" if bias > 0 else "phrases.calib_under"
                ),
                bias=abs(bias) * 100,
            )
        )
    return out


def bootstrap(summary: pd.DataFrame, lang: str = "en") -> list[str]:
    """Plain-language verdict per metric."""
    out = []
    for row in summary.itertuples():
        if row.significant:
            strict = tr(
                lang,
                "phrases.boot_strict_yes"
                if row.p_value < BONFERRONI_ALPHA
                else "phrases.boot_strict_no",
            )
            verdict = tr(
                lang,
                "phrases.boot_better",
                name=_lab(lang, "model_labels", row.better),
                p=row.p_value,
                strict=strict,
            )
        else:
            verdict = tr(lang, "phrases.boot_ns", p=row.p_value)
        out.append(
            tr(
                lang,
                "phrases.boot_line",
                metric=_lab(lang, "metric_labels", row.metric),
                a=_lab(lang, "model_labels", row.model_a),
                b=_lab(lang, "model_labels", row.model_b),
                diff=row.diff,
                lo=row.ci_low,
                hi=row.ci_high,
                verdict=verdict,
            )
        )
    out.append(tr(lang, "phrases.boot_bottom"))
    return out


# --- Risk factors ------------------------------------------------------------------


def shap_top(importance: pd.DataFrame, kind: str, n: int = 5, lang: str = "en") -> list[str]:
    """Top features of ``kind`` and their agreement with the other model."""
    ranked = importance[importance["model"] == kind].nlargest(n, "share")
    others = {
        k: set(g.nlargest(n, "share")["feature"])
        for k, g in importance.groupby("model")
        if k != kind
    }
    out = [
        tr(
            lang,
            "phrases.shap_top",
            n=n,
            name=_lab(lang, "model_labels", kind),
            items=", ".join(
                tr(
                    lang,
                    "phrases.shap_item",
                    label=_lab(lang, "feature_labels", feature),
                    share=share,
                )
                for feature, share in zip(ranked["feature"], ranked["share"])
            ),
            share=ranked["share"].sum(),
        )
    ]
    for other, top in others.items():
        common = [_lab(lang, "feature_labels", f) for f in ranked["feature"] if f in top]
        out.append(
            tr(
                lang,
                "phrases.shap_overlap",
                k=len(common),
                n=n,
                other=_lab(lang, "model_labels", other),
                items=", ".join(common) or "—",
            )
        )
    return out


def shap_direction(
    shap_values: pd.DataFrame,
    kind: str,
    feature: str,
    categorical: bool,
    lang: str = "en",
) -> list[str]:
    """Where the feature raises and lowers risk."""
    part = shap_values[(shap_values["model"] == kind) & (shap_values["feature"] == feature)]
    label = _lab(lang, "feature_labels", feature)
    spread = tr(lang, "phrases.shap_spread", lo=part["shap"].min(), hi=part["shap"].max())
    if categorical:
        means = part.groupby("display")["shap"].agg(["mean", "size"]).sort_values("mean")
        return [
            tr(
                lang,
                "phrases.shap_cat",
                label=label,
                high=means.index[-1],
                high_v=means["mean"].iloc[-1],
                high_n=int(means["size"].iloc[-1]),
                low=means.index[0],
                low_v=means["mean"].iloc[0],
                low_n=int(means["size"].iloc[0]),
            ),
            spread,
        ]
    data = part.dropna(subset=["value"])
    bins = pd.qcut(data["value"], 5, duplicates="drop")
    means = data.groupby(bins, observed=True)["shap"].mean()
    rho = data["value"].corr(data["shap"], method="spearman")
    if rho > 0.5:
        shape = tr(lang, "phrases.shap_up", label=label, rho=rho)
    elif rho < -0.5:
        shape = tr(lang, "phrases.shap_down", label=label, rho=rho)
    else:
        shape = tr(lang, "phrases.shap_nonmono", rho=rho)
    lowest = data["value"].min()

    def interval(iv: pd.Interval) -> str:
        return f"{max(iv.left, lowest):.3g}–{iv.right:.3g}"

    return [
        shape,
        tr(
            lang,
            "phrases.shap_bins",
            hi=means.max(),
            high=interval(means.idxmax()),
            lo=means.min(),
            low=interval(means.idxmin()),
        ),
        spread,
    ]


# --- Fleet and drone card -------------------------------------------------------------


def fleet(active: pd.DataFrame, window: float, horizon: float, lang: str = "en") -> list[str]:
    """Operational summary of drones in service."""
    counts = active["priority"].value_counts()
    finite = active.loc[np.isfinite(active["rul_median"]), "rul_median"]
    return [
        tr(
            lang,
            "phrases.fleet_counts",
            n=len(active),
            high=int(counts.get(PRIORITY_HIGH, 0)),
            medium=int(counts.get(PRIORITY_MEDIUM, 0)),
            low=int(counts.get(PRIORITY_LOW, 0)),
        ),
        tr(
            lang,
            "phrases.fleet_expected",
            window=window,
            expected=active["p_fail_window"].sum(),
        ),
        tr(
            lang,
            "phrases.fleet_spread",
            share=(active["p_fail_window"] > 0.1).mean(),
            window=window,
        ),
        tr(
            lang,
            "phrases.fleet_median",
            median=finite.median(),
            horizon=horizon,
            beyond=len(active) - len(finite),
        ),
    ]


def backtest(table: pd.DataFrame, landmark: float, lang: str = "en") -> list[str]:
    """Largest gap between predicted conditional survival and Kaplan-Meier."""
    km = table[table["source"] == "km"]
    out = []
    for kind in ("cox", "aft"):
        part = table[(table["source"] == kind) & (table["u"] <= km["u"].max())]
        idx = np.searchsorted(km["u"].to_numpy(), part["u"].to_numpy(), side="right") - 1
        s_km = np.where(idx >= 0, km["s"].to_numpy()[np.clip(idx, 0, None)], 1.0)
        gap = part["s"].to_numpy() - s_km
        worst = np.abs(gap).argmax()
        out.append(
            tr(
                lang,
                "phrases.backtest_gap",
                name=_lab(lang, "model_labels", kind),
                landmark=landmark,
                gap=_pp(gap[worst], lang),
                u=part["u"].to_numpy()[worst],
                mean=np.abs(gap).mean() * 100,
            )
        )
    return out


def backtest_groups(table: pd.DataFrame, best: str, lang: str = "en") -> list[str]:
    """Agreement of predicted and observed failure shares across prediction quintiles."""
    out = []
    riskiest = table[table["quintile"] == table["quintile"].max()].set_index("source")
    for kind, rows in table[table["source"] != "raw"].groupby("source", sort=False):
        inside = rows["predicted"].between(rows["ci_low"], rows["ci_high"]).sum()
        gap = (rows["predicted"] - rows["observed"]).abs()
        top = riskiest.loc[kind]
        out.append(
            tr(
                lang,
                "phrases.backtest_group",
                name=_lab(lang, "model_labels", kind),
                inside=int(inside),
                n=len(rows),
                gap=np.average(gap, weights=rows["n"]) * 100,
                pred=top["predicted"],
                obs=top["observed"],
                lo=top["ci_low"],
                hi=top["ci_high"],
            )
        )
    if "raw" in riskiest.index:
        out.append(
            tr(
                lang,
                "phrases.backtest_raw",
                name=_lab(lang, "model_labels", best),
                before=riskiest.loc["raw", "predicted"],
                after=riskiest.loc[best, "predicted"],
            )
        )
    return out


def drone(
    row: pd.Series,
    contrib: pd.DataFrame,
    pred: pd.DataFrame,
    best: str,
    window: float,
    predicted_median: float,
    lang: str = "en",
) -> list[str]:
    """Narrative explanation of one drone's prediction."""
    out = []
    # ``best`` is the forecasting kind: the position uses the production model's risk
    risk = f"forecast_risk_{best}"
    top = (pred[risk] >= row[risk]).mean()
    if row["event"] == 1:
        timing = (
            (tr(lang, "phrases.before") if row["duration_h"] < predicted_median else tr(lang, "phrases.after"))
            if np.isfinite(predicted_median)
            else "—"
        )
        out.append(
            tr(
                lang,
                "phrases.drone_failed",
                hours=row["duration_h"],
                mode=_lab(lang, "failure_mode_labels", row["failure_mode"]).lower(),
                median=hours_text(predicted_median, lang),
                when=timing,
            )
        )
    elif not row["in_service"]:
        out.append(tr(lang, "phrases.drone_left", hours=row["duration_h"]))
    else:
        active = pred[pred["in_service"]]
        rank = int((active["p_fail_window"] > row["p_fail_window"]).sum()) + 1
        out.append(
            tr(
                lang,
                "phrases.drone_in_service",
                hours=row["duration_h"],
                window=window,
                p=row["p_fail_window"],
                rank=_ordinal(rank, lang),
                n=len(active),
                priority=_lab(lang, "priority_labels", row["priority"]),
            )
        )
    bottom = (pred[risk] <= row[risk]).mean()
    position = tr(
        lang,
        "phrases.riskiest_share" if top <= 0.5 else "phrases.least_share",
        share=_top_share(top if top <= 0.5 else bottom),
    )
    out.append(
        tr(
            lang,
            "phrases.drone_rank",
            position=position,
            group=_lab(lang, "priority_labels", row["risk_group"]),
        )
    )

    def bits(frame: pd.DataFrame) -> str:
        return "; ".join(
            tr(
                lang,
                "phrases.factor_bits",
                label=_lab(lang, "feature_labels", feature),
                value=value,
                shap=shap,
            )
            for feature, value, shap in zip(frame["feature"], frame["display"], frame["shap"])
        )

    up = contrib[contrib["shap"] > 0].nlargest(3, "shap")
    down = contrib[contrib["shap"] < 0].nsmallest(3, "shap")
    if len(up):
        out.append(tr(lang, "phrases.raises", items=bits(up)))
    if len(down):
        out.append(tr(lang, "phrases.lowers", items=bits(down)))
    return out


def drone_hazard(
    age: float, rate: float, fleet_rate: float, best: str, lang: str = "en"
) -> str:
    """Current failure rate of a drone in service, relative to the fleet at that age."""
    if fleet_rate <= 0:
        return tr(
            lang,
            "phrases.drone_rate_only",
            age=age,
            rate=_rate(rate, lang),
            model=_lab(lang, "model_labels", best),
        )
    return tr(
        lang,
        "phrases.drone_rate",
        age=age,
        rate=_rate(rate, lang),
        model=_lab(lang, "model_labels", best),
        ratio=rate / fleet_rate,
        fleet=_rate(fleet_rate, lang),
    )


def whatif(
    base: pd.Series,
    new: pd.Series,
    changes: list[tuple[str, object, object]],
    window: float,
    lang: str = "en",
) -> list[str]:
    """Narrative comparison of the current and scenario remaining-life summaries."""
    if not changes:
        return [tr(lang, "phrases.whatif_same")]
    out = [
        tr(
            lang,
            "phrases.whatif_changed",
            items="; ".join(
                tr(
                    lang,
                    "phrases.whatif_change",
                    label=_lab(lang, "feature_labels", feature),
                    old=old,
                    new=new_,
                )
                for feature, old, new_ in changes
            ),
        )
    ]
    dp = new["p_fail_window"] - base["p_fail_window"]
    out.append(
        tr(
            lang,
            "phrases.whatif_p_down" if dp < 0 else "phrases.whatif_p_up",
            window=window,
            before=base["p_fail_window"],
            after=new["p_fail_window"],
            delta=_pp(dp, lang),
        )
    )
    out.append(
        tr(
            lang,
            "phrases.whatif_life",
            before=base["rmrl"],
            after=new["rmrl"],
            delta=new["rmrl"] - base["rmrl"],
            med0=hours_text(base["rul_median"], lang),
            med1=hours_text(new["rul_median"], lang),
        )
    )
    out.append(tr(lang, "phrases.whatif_caveat"))
    return out

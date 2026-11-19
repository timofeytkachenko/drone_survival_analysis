# Methodology: drone failure forecasting with survival analysis

Technical reference for everything implemented in this repository: the data-generating process, the survival-analysis theory behind each step, every chart in the notebook and the dashboards, and the modelling decisions and their caveats. The business-level overview is in [README.md](README.md).

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Data](#2-data)
3. [Survival-analysis foundations](#3-survival-analysis-foundations)
4. [Time split and preprocessing](#4-time-split-and-preprocessing)
5. [Models](#5-models)
6. [Hyper-parameter tuning](#6-hyper-parameter-tuning)
7. [Recalibration for unobserved heterogeneity](#7-recalibration-for-unobserved-heterogeneity)
8. [Predicted hazard](#8-predicted-hazard)
9. [Evaluation](#9-evaluation)
10. [Explanations (SHAP)](#10-explanations-shap)
11. [Remaining useful life and maintenance priority](#11-remaining-useful-life-and-maintenance-priority)
12. [Artifacts](#12-artifacts)
13. [Dashboards](#13-dashboards)
    - [13a. Training and scoring studio](#13a-training-and-scoring-studio)
14. [Modelling caveats](#14-modelling-caveats)
15. [Reference results](#15-reference-results)

---

## 1. Pipeline overview

```
generate_drone_survival.py ─► data/drone_survival.csv (5000 drones, seed 42)
                                   │
drone_survival.ipynb               ▼
  1 setup ─► 2 split by commissioning date (70/10/20), in-service status
          ─► 3 Optuna tuning ─► XGBoost Cox + AFT ─► frailty recalibration ─► save/load
          ─► 4 survival and hazard vs Kaplan-Meier / Nelson-Aalen
          ─► 5 metrics, model choice, Brier/AUC(t), calibration, paired bootstrap
          ─► 6 SHAP
          ─► 7 conditional survival, landmark backtest, RUL of drones in service
          ─► 8 artifacts/*.parquet + meta.json, models/best_{cox,aft}.json
                                   │
                                   ▼
          dashboard/app.py (Dash)  ·  dashboard/streamlit_app.py (Streamlit)
```

Code layout:

- `generate_drone_survival.py`: synthetic data with a known ground truth.
- `utils.py`: modelling library (`Model`, `tune`, metrics, bootstrap, calibration, hazard, RUL).
- `drone_survival.ipynb`: the end-to-end analysis; the only producer of models and artifacts.
- `predict_drone.ipynb`: scoring one drone with the saved model. `debug_predictions.ipynb` (Russian: `debug_predictions_RU.ipynb`): recomputes one drone's forecast by hand, step by step, and compares each intermediate with the library; use it when a number or a curve looks wrong.
- `trainer/`: the training and scoring studio (section 13a): `pipeline.py`, `runs.py`, `app.py`; with `Dockerfile` and `docker-compose.yml` it runs in a container.
- `dashboard/`: `data.py` (artifact loading and validation), `common.py` (shared logic of both apps: KPIs, fleet table, drone card, what-if, CSV scoring), `figures.py` (Plotly charts), `insights.py` (auto-generated text findings), `locale/` (English and Russian catalogs with all display strings; loading fails if the two differ in structure or placeholders), `texts.py` and `labels.py` (English re-exports of the catalog), `app.py` and `streamlit_app.py` (the two UIs).

---

## 2. Data

### 2.1 Data-generating process

Each row is one drone; time is **flight hours** from commissioning to the first critical failure or to censoring.

**Covariates** are sampled with realistic dependencies: the environment drives temperature, humidity and wind; wind and operator experience drive hard landings; hard landings and the airframe drive vibration; the airframe drives payload capacity, sortie length and daily utilisation; 15% of drones are refurbished and start with 80–400 battery cycles.

**Failure mechanisms.** Four competing causes, each a Weibull proportional-hazards model

$$
h_m(t \mid x) = \frac{k_m}{s_m}\left(\frac{t}{s_m}\right)^{k_m - 1} \exp\big(\eta_m(x)\big) \cdot Z,
$$

| Mechanism | Shape $k_m$ | Scale $s_m$, h | Hazard over age |
|---|---|---|---|
| motor | 2.2 | 2000 | rising (wear-out) |
| battery | 1.6 | 1700 | rising |
| electronics | 0.8 | 12000 | falling (infant mortality) |
| crash | 1.0 | 5000 | constant |

The log-hazards $\eta_m(x)$ are partly non-linear: an overload penalty above 90% payload, a U-shape in temperature for batteries, thresholds in temperature and sortie length, $\log(1 + \text{experience})$ for crashes. All are multiplied by `effect_strength = 2.0`.

**Shared gamma frailty** $Z$, gamma-distributed with mean 1 and variance $\theta = 0.3$, multiplies all four hazards of a drone. It is unobserved: two drones with identical features differ in durability. This is what section 7 corrects for.

The observed failure is the earliest of the four latent times; its cause is stored in `failure_mode`.

**Censoring** is the minimum of two mechanisms:

- administrative: fleet entry is staggered over the first 80% of a 730-day window starting 2024-01-01, and the study ends at day 730; calendar days are converted to flight hours via `daily_utilization_h`;
- random dropout (sold, lost, retired): exponential with mean 5000 flight hours.

With `--with-ground-truth` the generator also returns `true_failure_time_h`, `true_failure_mode`, `true_frailty` and `true_eta_*`. They are never used as features; they allow oracle checks of the models (section 15).

### 2.2 Columns

- Identifiers and dates: `drone_id`, `commission_date`.
- Target: `duration_h` (observed flight hours), `event` (1 = failure, 0 = censored).
- `failure_mode`: the cause, or `censored`. Not a feature (it is known only after the outcome); used in the dashboards to describe failure structure.
- 16 features (`FEATURES` in the notebook), 4 of them categorical (`model`, `environment`, `is_refurbished`, `firmware_major`):
`model`, `environment`, `is_refurbished`, `initial_battery_cycles`, `firmware_major`, `maintenance_interval_h`, `operator_experience_yrs`, `daily_utilization_h`, `avg_flight_min`, `avg_payload_kg`, `payload_ratio`, `mean_temp_c`, `avg_humidity_pct`, `avg_wind_ms`, `hard_landings_per_100h`, `mean_vibration_g`.

All features are fixed at commissioning. This is what makes conditional survival $S(a+u)/S(a)$ valid with the same feature vector at any age (section 3.4).

### 2.3 In-service status

A censored drone is either still flying at the data cut-off or has left the fleet earlier. The notebook reconstructs the last day each drone was observed:

$$
D_i = c_i + \frac{T_i}{u_i}, \qquad D_{\max} = \max_i D_i,
$$

where $c_i$ is `commission_date`, $T_i$ is `duration_h` (flight hours) and $u_i$ is `daily_utilization_h` (flight hours per day), so $T_i / u_i$ is in days and $D_{\max}$ is the data cut-off. It then sets `in_service = (event == 0) & (last_seen > cutoff − 1 day)`. Only drones in service get a remaining-life forecast and a maintenance priority. Drones that left the fleet stay in all estimates as censored observations.

---

## 3. Survival-analysis foundations

### 3.1 Censoring

For each drone we observe $T = \min(T^{\ast}, C)$ and $\delta = \mathbf{1}[T^{\ast} \le C]$, where $T^{\ast}$ is the failure time and $C$ the censoring time. A censored drone carries the information "survived at least $T$ hours". Dropping censored drones or treating $T$ as the lifetime biases lifetimes downwards; treating them as non-failures biases the failure probability downwards. All estimators below use censored rows correctly under the assumption of **non-informative censoring** (censoring independent of failure given the features).

### 3.2 Core quantities

- Survival function: $S(t) = P(T^{\ast} > t)$.
- Hazard: $h(t) = \lim_{\Delta \to 0} P(t \le T^{\ast} < t + \Delta \mid T^{\ast} \ge t) / \Delta$, the failure rate among drones still flying at $t$. Reported as failures per 100 flight hours.
- Cumulative hazard: $H(t) = \int_0^t h(s)\thinspace ds$, with $S(t) = \exp(-H(t))$.
- Density: $f(t) = h(t) S(t)$.

### 3.3 Non-parametric estimators

**Kaplan-Meier** (`sksurv.nonparametric.kaplan_meier_estimator`):

$$
\hat S(t) = \prod_{t_j \le t}\left(1 - \frac{d_j}{Y_j}\right),
$$

with $d_j$ failures and $Y_j$ drones at risk at event time $t_j$. Confidence bands use the **log-log** transform (`conf_type="log-log"`), which keeps them inside $[0, 1]$.

**Kernel-smoothed Nelson-Aalen hazard** (`utils.observed_hazard`). The Nelson-Aalen increments $d_j / Y_j$ estimate the jumps of $H$; smoothing them with a kernel gives a hazard estimate:

$$
\hat h(t) = \sum_j K_b(t - t_j)\thinspace \frac{d_j}{Y_j}, \qquad \widehat{\mathrm{Var}}\thinspace \hat h(t) = \sum_j K_b(t - t_j)^2\thinspace \frac{d_j}{Y_j^2},
$$

with the Epanechnikov kernel $K_b(z) = \frac{3}{4b}(1 - (z/b)^2)$ for $|z| < b$, and a pointwise 95% band $\hat h \pm 1.96\thinspace \widehat{\mathrm{se}}$ (lower bound clipped at 0). The bandwidth (`utils.hazard_bandwidth`) is Silverman's rule on the event times: $b = 0.9 \min(\text{sd}, \text{IQR}/1.34)\thinspace n^{-1/5}$. Within $b$ of 0 and of the last event the kernel loses mass and the estimate is biased downwards, so hazard grids start at $b$.

### 3.4 Conditional survival

For a drone that has already flown $a$ hours without failure:

$$
S(a + u \mid T^{\ast} > a) = \frac{S(a+u)}{S(a)} = \exp\left(-\int_a^{a+u} h(t)\thinspace dt\right).
$$

A flat hazard gives the same curve at every age; a rising hazard makes older drones' curves fall faster. At fleet level the notebook compares the Kaplan-Meier ratio $\hat S(a+u)/\hat S(a)$ with the model's $\sum_i S_i(a+u) / \sum_i S_i(a)$. The sums are over all test drones, so drones that failed before $a$ drop out through their small $S_i(a)$.

### 3.5 Remaining-life summaries

From the conditional curve $S_a(u) = S(a+u)/S(a)$ on $u \in [0, \text{horizon}]$ (`utils.remaining_life`):

- `p_fail_window` $= 1 - S_a(100)$: probability of failure within the next 100 flight hours (`RUL_WINDOW`);
- `rul_q10`, `rul_median`: first $u$ where $S_a(u) \le 0.9$ and $\le 0.5$; `inf` if not reached within the horizon;
- `rmrl` $= \int_0^{\text{horizon}} S_a(u)\thinspace du$: mean residual life restricted to the horizon (trapezoid rule on 1001 points).

---

## 4. Time split and preprocessing

**Split by commissioning date** (`commission_date` quantiles 0.70 and 0.80): the oldest 70% of drones train, the next 10% validation, the newest 20% test (3499 / 500 / 1001 drones). This imitates deployment: the model is trained on earlier cohorts and applied to newer ones.

Consequence: later cohorts have shorter follow-up, so their observed failure share is lower (train 66%, validation 55%, test 45%). This is censoring, not a drift in reliability, and it is why every evaluation estimates its censoring distribution on the evaluated split itself (section 9.1).

**Cohort vs calendar split.** Training drones are followed up to the data cut-off, i.e. after the test drones were commissioned. Nothing in these data depends on calendar time, so this does not leak. With seasonality or fleet-wide events (a firmware rollout, a bad batch of batteries) it would.

**Categoricals.** `fit_categories` learns the vocabulary on train only. In validation, test and uploads, unseen levels become NaN with a logged warning (`encode_unknown_categories`). XGBoost routes NaN along each split's default branch, whereas XGBoost ≥ 3 would raise on an unknown category. Boolean `is_refurbished` stays boolean (XGBoost rejects boolean category levels and splits a binary feature optimally anyway).

---

## 5. Models

Both models are XGBoost gradient-boosted trees with survival losses; trees capture non-linearities (overload thresholds, temperature U-shape) and interactions without feature engineering. Training (`Model.fit`): up to 2000 rounds, early stopping after 50 rounds without improvement of the validation loss; all predictions use the trees up to `best_iteration`.

### 5.1 Cox proportional hazards (`survival:cox`)

$$
h(t \mid x) = h_0(t)\thinspace \exp(f(x)),
$$

where $f$ is the boosted margin (log hazard ratio). XGBoost maximises the Cox partial likelihood $\prod_{i:\delta_i=1} \exp(f(x_i)) / \sum_{j: T_j \ge T_i} \exp(f(x_j))$. Labels are signed: $+T$ for failures, $-T$ for censored rows. The partial likelihood does not identify $h_0$, so after training the **Breslow estimator** is computed on the training set (`_breslow_baseline`):

$$
\hat H_0(t) = \sum_{t_j \le t} \frac{d_j}{\sum_{i: T_i \ge t_j} \exp(f(x_i))}, \qquad S(t \mid x) = \exp\big(-\exp(f(x))\thinspace \hat H_0(t)\big).
$$

Properties and caveats:

- no assumption on the shape of $h_0$; curves of different drones never cross (proportionality);
- $\hat H_0$ is a step function and is **flat after the last training failure**, so any forecast that needs $a + u$ beyond that point is optimistic. `conditional_survival` logs a warning in that case and the RUL horizon is capped (section 11);
- the baseline $\hat H_0$ uses in-sample margins of the training drones.

### 5.2 Accelerated failure time (`survival:aft`)

$$
\log T^{\ast} = f(x) + \sigma \varepsilon,
$$

where the trees predict the location $f(x)$ and $\varepsilon$ is noise with a fixed distribution: `aft_loss_distribution` selects it, $\sigma$ = `aft_loss_distribution_scale`, and both are tuned (section 6). Labels are intervals: $[T, T]$ for failures, $[T, +\infty)$ for censored rows. Then

$$
S(t \mid x) = e^{-H(z)}, \qquad h(t \mid x) = \frac{H'(z)}{\sigma t}, \qquad z = \frac{\log t - f(x)}{\sigma},
$$

where the cumulative hazard $H$ of the noise depends on the distribution (`utils._aft_noise`):

| `aft_loss_distribution` | Lifetime | $H(z)$ | $H'(z)$ | Hazard shape |
|---|---|---|---|---|
| `normal` | log-normal | $-\log(1 - \Phi(z))$ | $\varphi(z) / (1 - \Phi(z))$ | rises, peaks, then falls |
| `logistic` | log-logistic | $\log(1 + e^{z})$ | $e^{z} / (1 + e^{z})$ | one peak for $\sigma < 1$, otherwise falls |
| `extreme` | Weibull | $e^{z}$ | $e^{z}$ | monotone: rises for $\sigma < 1$, falls for $\sigma > 1$ |

For `normal`, $H'$ is computed as `exp(logpdf − logsf)` to stay finite deep in the right tail. Whichever distribution is chosen, the shape of the hazard is imposed by it and not learned from the data. Parameters without `aft_loss_distribution` (models saved before it was tuned) mean `normal`.

### 5.3 Common interface

- `risk(data)`: score for ranking metrics, higher = earlier failure (Cox margin, negated AFT margin).
- `survival`, `conditional_survival`, `hazard`: curves on a time grid; all apply the recalibration of section 7 when it is set.
- `save` / `load`: one JSON with the booster, parameters, category vocabulary, training history, Breslow baseline and calibration. `booster.save_model` alone would lose the baseline and vocabulary. The notebook saves and reloads both models before using them, which checks the round trip.

---

## 6. Hyper-parameter tuning

`utils.tune`: Optuna, TPE sampler (seed 42), 60 trials per model (`N_TRIALS` in the notebook). Search space:

- `max_depth` 2–5; `eta` 0.025–0.15 (log); `min_child_weight` 1–30 (log);
- `subsample`, `colsample_bytree` 0.7–1.0; `reg_lambda` 0.1–20 (log);
- AFT only: `aft_loss_distribution` in `normal`, `logistic`, `extreme`; `aft_loss_distribution_scale` 0.2–2.0 (log).

**Objective: validation Integrated Brier score** on `evaluation_times(valid)`, with the censoring distribution of the validation set. IBS measures probability accuracy, not just ranking, which matters because the forecasts are used as probabilities (spare-parts planning, maintenance thresholds). The best parameters are refitted with early stopping on validation.

The validation set is therefore used three times: early stopping, tuning and model selection. Its metrics are optimistic; quote test.

**Training-curves chart** (notebook section 3, Dash/Streamlit "Model quality"): train and validation loss (`cox-nloglik`, `aft-nloglik`) per boosting iteration, with the best iteration marked. A widening gap with a rising validation curve is overfitting; early stopping cuts there.

---

## 7. Recalibration for unobserved heterogeneity

### 7.1 Why

Drones with identical features still differ in durability (the shared frailty in section 2.1). The weaker drones of a high-risk group fail first, so the survivors become, on average, sturdier than their features suggest. A Cox model assumes a constant hazard ratio between drones at every age, so it **overstates the hazard of high-risk drones at later ages**, exactly the drones that drive maintenance priority. Before the correction, the landmark backtest showed this: in the riskiest fifth of drones Cox predicted 35.1% failures in 100 h against an observed 28.4%.

### 7.2 Gamma-frailty link

If each drone's cumulative hazard is $Z \cdot k \cdot H(t \mid x)$ with an unobserved $Z$, gamma-distributed with mean 1 and variance $v$, the survival of drones with features $x$ is

$$
S(t \mid x) = \mathbb{E}\left[e^{-Z k H}\right] = (1 + v k H)^{-1/v}, \qquad \tilde H = g(H) = \frac{\log(1 + v k H)}{v}.
$$

`_frailty_link` returns $g(H)$ and its slope $g'(H) = k / (1 + v k H)$; the limit $v \to 0$ is $kH$, and $v = 0, k = 1$ changes nothing. The corrected hazard is $\tilde h = g'(H)\thinspace h$: it bends down where $H$ is large (risky or old drones) and stays close to $k h$ where $H$ is small.

Properties:

- $g$ is strictly increasing, so at any fixed $t$ the order of drones by $S(t \mid x)$ is unchanged. Harrell's and Uno's C-index (computed on `risk`) and AUC(t) (computed on $1 - S(t)$) are unaffected; IBS, calibration, conditional survival, RUL and priorities change;
- survival still starts at $S(0) = 1$, curves stay monotone, conditional survival is still the ratio of survivals.

### 7.3 Estimation (`Model.calibrate`)

Parameters maximise the right-censored log-likelihood of the linked model. The model's own $\log h$ term does not depend on $(v, k)$, so the objective reduces to

$$
\ell(v, k) = \sum_i \Big[\delta_i \log g'(H_i) - g(H_i)\Big], \qquad H_i = H(T_i \mid x_i),
$$

optimised with L-BFGS-B over $v \ge 0$ and $\log k$ (start $v = 0.1$, $k = 1$; a non-converged fit raises).

$H_i$ must not be in-sample: a model evaluated on the drones it was trained on has already absorbed part of the effect. So $H_i$ are **out-of-fold**: the training set is split into 5 random folds (seed 0); for each fold a model with the same tuned parameters is refitted on the other four (early stopping on validation) and predicts the held-out drones, including its own Breslow baseline. Validation and test are not used to fit the link. Fitting on validation alone was tried during development and was too noisy (bootstrap 5–95% range of $v$ roughly 0–0.4).

### 7.4 Rule: keep only a real frailty effect

The notebook keeps the correction only if the fitted $v > 0$. With $v = 0$ the link is a plain rescaling $kH$, which is not what this step is for.

- **Cox**: $v = 0.289$, $k = 1.052$ (the generator's true variance is 0.3). Applied.
- **AFT**: $v = 0$, $k = 1.033$. Not applied. Its tuned log-logistic hazard already falls with age, which mimics the frailty effect; the pure rescaling made its test IBS slightly worse (0.1754 → 0.1760).

A landmark Poisson recalibration of the 100-hour failure probability alone was also considered and rejected: against the generator's true probabilities it was less accurate (RMSE 6.9 pp vs 5.5 pp, measured in an earlier run with the same generator and a different tuning; not repeated) and it would have made `p_fail_window` inconsistent with the rest of the curve (median RUL, RMRL).

---

## 8. Predicted hazard

`Model.hazard(data, times, bandwidth)`:

- **AFT**: the exact hazard of its distribution (section 5.2).
- **Cox**: the Breslow baseline is a step function with no density, so its jumps are smoothed with the same Epanechnikov kernel as the observed hazard, $\hat h_0(t) = \sum_j K_b(t - t_j)\thinspace \Delta \hat H_0(t_j)$, and multiplied by $\exp(f(x))$. Same edge bias near 0 and near the last training event.
- With calibration, both are multiplied by $g'(H(t \mid x))$.

**Fleet (population) hazard** (`population_hazard`) is comparable with the Nelson-Aalen estimate only if drones that already failed leave the risk set:

$$
h_{\text{fleet}}(t) = \frac{\sum_i S_i(t)\thinspace h_i(t)}{\sum_i S_i(t)},
$$

not the plain mean of $h_i(t)$.

**Hazard chart** (notebook 4.2; dashboards "Overview" and "Model quality"): left, the kernel-smoothed Nelson-Aalen hazard of the test set with its 95% band against each model's fleet hazard, per 100 flight hours, on a grid from $b$ to the 90% quantile of test durations; right, individual hazards of five random drones (solid Cox, dashed AFT). The hazard is a stricter check than the survival curve: $S(t)$ accumulates errors, the hazard shows them where they happen.

---

## 9. Evaluation

All metrics are on the held-out test set unless stated; validation is shown for reference.

### 9.1 Time grid and IPCW

`evaluation_times(val, test)` returns 50 points over the part of the 5%–80% `duration_h` quantile range shared by both splits, so validation and test metrics are on one grid and comparable. The upper bound stays below the tail, where IPCW weights explode because few drones are still observed.

Several metrics reweight observed rows by the inverse probability of remaining uncensored (IPCW), $\hat G(t)$, the Kaplan-Meier estimate of the **censoring** distribution. The reference set for $\hat G$ is **the evaluated split itself**: newer drones are censored earlier, so the training $\hat G$ would understate the weights and with them the Brier score. IPCW assumes censoring independent of the features; here administrative censoring time is days in service × `daily_utilization_h`, a feature, so the assumption holds only approximately.

### 9.2 Discrimination

- **Harrell's C-index**: share of comparable pairs (the earlier time is a failure) where the drone with the higher `risk` fails first. Ties in risk count as ½. Depends on the censoring pattern.
- **Uno's C-index** (`concordance_index_ipcw`, truncated at the last grid time $\tau$): the IPCW version, consistent for the concordance probability under independent censoring.
- **Cumulative/dynamic AUC(t)** (`cumulative_dynamic_auc` on $1 - S(t)$): at each $t$, how well the model separates drones that failed by $t$ (cases) from those still flying (controls). `mean_auc` is sksurv's weighted mean over the grid.

### 9.3 Probability accuracy

**Brier score** at $t$ with IPCW:

$$
\text{BS}(t) = \frac{1}{n}\sum_i\left[\frac{\hat S_i(t)^2\thinspace \mathbf{1}(T_i \le t, \delta_i = 1)}{\hat G(T_i)} + \frac{(1 - \hat S_i(t))^2\thinspace \mathbf{1}(T_i > t)}{\hat G(t)}\right].
$$

**IBS** is its integral over the grid divided by the grid length. A constant 50% forecast scores 0.25.

**Model selection: lowest validation IBS** (the tuning objective). The test set plays no part in the choice and remains an independent check.

**Charts** (notebook 5.2; "Model quality" tab): Brier score over time and AUC(t) over time on test, with IBS / mean AUC in the legend and the 0.5 random-AUC line.

### 9.4 Calibration

**By decile** (`calibration_curve`; notebook 5.3; "Model quality" tab with a horizon selector): at each horizon (25/50/75% quantiles of test durations) drones are split into 10 quantile bins of predicted $S(h)$; in each bin the mean prediction is plotted against the bin's Kaplan-Meier $\hat S(h)$. Points on the diagonal mean calibrated probabilities. If a bin has no follow-up to $h$, its last Kaplan-Meier value is used, so horizons are kept well inside follow-up.

**Mean curve vs Kaplan-Meier** (notebook 4.1 left; "Model quality" tab): the mean predicted $S(t)$ over test drones against the test Kaplan-Meier curve with its 95% band. This is fleet-level calibration over the whole time range: what spare-parts planning uses. The right panel shows five random drones (solid Cox, dashed AFT) to illustrate individual spread and the shape constraints (Cox curves never cross; AFT curves can).

### 9.5 Paired bootstrap Cox vs AFT

`bootstrap_compare` (notebook 5.4; "Model quality" tab), 1000 replicates, seed 42:

- each replicate resamples test rows with replacement and scores **both models on the same rows**, so the differences account for the correlation between models;
- the censoring distribution $\hat G$ is estimated once and kept fixed; models are not refitted, so intervals reflect evaluation-set noise only;
- 95% percentile interval of the difference; two-sided p-value $2\min(P(d \le 0), P(d \ge 0))$, capped at 1, resolution $2/n_{\text{boot}}$;
- four metrics are tested, so the dashboards also report the Bonferroni threshold $0.05/4 = 0.0125$.

**Chart**: histograms of the bootstrap differences per metric with 0 and the 95% interval marked.

### 9.6 Conditional survival check

Notebook 7.1; "Fleet" tab. Ages $a \in \lbrace 0, Q_{25}, Q_{50}, Q_{75} \rbrace$ of test durations. Left: Kaplan-Meier $\hat S(a+u)/\hat S(a)$ (solid) vs the best model's fleet conditional survival (dashed). Right: one drone's conditional curves at the same ages. The table reports the observed and predicted probability of failing within the next 100 h given survival to $a$. With a flat fleet hazard the four observed curves nearly coincide.

### 9.7 Landmark backtest

Notebook 7.2; "Fleet" tab. Landmark $L = Q_{25}$ of test durations. Drones with $T > L$ form the at-risk group; their residual life $T - L$ (with the same event flags) is the observed outcome, and every one of them is predicted with $S(L + u \mid x)/S(L \mid x)$.

- **Whole group** (left chart): mean predicted conditional survival vs Kaplan-Meier of residual life with its band.
- **By risk group** (right chart, table `backtest_groups`): drones are split into quintiles of the predicted probability of failing within 100 h after $L$, the quantity behind the maintenance priority. In each quintile the mean prediction is compared with the Kaplan-Meier failure share at 100 h and its log-log 95% interval. Points below the diagonal mean the risk is overstated. The best model is drawn with error bars; if it was recalibrated, a grey dashed line ("raw") shows the same model before the recalibration of section 7. The line is drawn only for a recalibrated best model, so it is absent when AFT is the best model (the current run); `backtest_groups` then holds the Cox and AFT rows.

This checks the conditional forecasts on drones that actually survived to $L$. It is not an out-of-time backtest: features are fixed and the outcomes come from the same test cohort, observed after $L$.

---

## 10. Explanations (SHAP)

`Model.explain`: TreeSHAP (`shap.TreeExplainer` on the booster) of the `risk` score, using the trees up to the best iteration. AFT values are negated, so for both models **positive SHAP = earlier failure**. Cox values are in log-hazard-ratio units, AFT values in log-time units, so the models are compared by **shares** of total mean |SHAP|, not by raw values. Categorical features are passed to `shap` as integer codes (NaN for unseen levels) so plots can colour by them.

Charts (notebook section 6; "Risk factors" tab, model selector in the dashboards):

- **Importance**: share of total mean |SHAP| per feature for both models.
- **Beeswarm**: one point per drone and feature, x = SHAP, colour = feature value; shows direction and spread of each effect.
- **Dependence**: SHAP vs feature value for the top three features of the best model; reveals thresholds and U-shapes (overload above 90% payload, temperature).
- **Waterfall** (dashboard "Drone card"): how each feature moves one drone's risk away from the fleet baseline.

SHAP explains the model, not the causal mechanism: correlated features (environment, temperature, humidity, wind) share credit in ways that do not map one-to-one onto the generator's $\eta_m$.

---

## 11. Remaining useful life and maintenance priority

Notebook 7.3 ("Drones in service"):

- population: test drones with `in_service`; current age $a$ = `duration_h`;
- **horizon** = floor(last training failure time − oldest in-service age), so that $a + u$ stays inside the Cox baseline support for every drone. The far end of that support rests on few training drones (the notebook prints how many are still at risk at 50/75/100% of it), so long-horizon figures for the oldest drones are noisy;
- `remaining_life` gives `p_fail_window`, `rul_q10`, `rul_median`, `rmrl` (section 3.5) with the best model, recalibration included.

**Charts**: conditional survival curves of the five riskiest drones with the 0.5 line (median RUL); histogram of median RUL with the count of drones beyond the horizon. Dashboards: the same on the "Fleet" tab plus the per-drone card.

**Maintenance priority** (from `p_fail_window`): High ≥ 0.5, Medium ≥ 0.2, Low otherwise; drones not in service get `None (failed)` or `None (left the fleet)`. Thresholds are stored in `meta.json` and are a business choice, not a statistical one.

**Risk group** (Low / Medium / High): terciles of the best model's `risk` over all test drones; a time-free segmentation used in the "Failure segments" tab. Uploaded drones are placed into the same terciles.

---

## 12. Artifacts

Written by notebook section 8 into `artifacts/` (parquet unless stated); the dashboards read only these and `models/`:

- `metrics`: scalar metrics per model and split;
- `brier_auc`: Brier score and AUC per test grid time and model;
- `calibration`: decile calibration per model and horizon;
- `bootstrap_summary`, `bootstrap_samples`: paired bootstrap results and replicates (long format);
- `shap_values` (long format: drone × model × feature, with the displayed value), `shap_importance` (mean |SHAP| and share);
- `predictions`: test drones with features, outcome, `in_service`, `risk_cox`, `risk_aft`, `risk_group`, RUL columns and `priority`;
- `survival_mean`: test Kaplan-Meier with band and mean predicted curves;
- `backtest`: Kaplan-Meier of residual life after $L$ and mean predicted conditional curves;
- `backtest_groups`: the quintile backtest (section 9.7);
- `meta.json`: best model, features, category vocabulary, numeric training ranges, split dates, evaluation grids, calibration horizons, RUL horizon and window, landmark, priority thresholds, SHAP base values.

`dashboard/data.py` refuses to start if a table is missing or the predictions come from an older notebook version (no `in_service`, different priority labels).

---

## 13. Dashboards

Dash (`dashboard/app.py`, port 8050) and Streamlit (`dashboard/streamlit_app.py`) share all logic in `common.py`, `figures.py`, `insights.py` and the `locale` catalogs, so they show the same numbers. Both have an English/Russian switch that keeps the open tab; data values (priority codes, feature names, failure modes) stay in English and only their captions are translated. Every chart has a "what it shows / how to read it / details" explanation and auto-generated findings computed from the artifacts.

| Tab | Content |
|---|---|
| Overview | KPIs, queue size at high-priority thresholds of 30/40/50%, key findings, Kaplan-Meier of the test fleet, fleet hazard, failure-mode breakdown and timing |
| Methodology | survival-analysis concepts, pipeline steps, features, hyper-parameters (including the recalibration parameters) |
| Failure segments | Kaplan-Meier and failure-mode structure by segment (drone model, environment, firmware, refurbished, risk group, maintenance priority) |
| Model quality | training curves, mean curve vs Kaplan-Meier, individual curves (the five drones where Cox and AFT disagree most), hazard fit, metrics table, Brier(t), AUC(t), calibration with horizon selector, bootstrap |
| Risk factors | SHAP importance, beeswarm and dependence with model selector |
| Fleet | drones in service with priority, filters and CSV download; hours flown against failure probability; landmark backtest overall and by risk group; conditional curves of the riskiest drones; median-RUL histogram |
| Drone card | status, forecast metrics, conditional survival and hazard, SHAP waterfall, "what if" scenario |
| Score new drones | CSV upload, template download, forecast table |

**Drone card hazard.** The grid covers the fleet range and extends to the drone's age + 100 h only as far as its forecast needs: beyond the fleet range the smoothed Cox baseline rests on few training failures and oscillates.

**What-if.** The user edits a drone's features. `whatif_compare` recomputes `remaining_life` for the current and edited features at the drone's current age and reports P(failure in 100 h), priority, median and expected remaining life side by side, with both conditional curves. This is a model counterfactual, not a causal estimate: changing one feature holds correlated features fixed.

**CSV scoring** (`score_upload`):

- required columns: the 16 features; optional `drone_id` (generated if absent) and `age_h` (flight hours already flown without failure, default 0); other columns are passed through; at most 10 000 rows;
- hard errors: missing columns, non-numeric values, unreadable booleans, negative or empty `age_h`;
- warnings: unseen category levels (treated as missing), empty feature values, numeric values outside the training range (flagged per row in `out_of_range`), ages above the oldest in-service test drone (remaining life optimistic);
- output: risk group (test terciles), `p_fail_window`, `rul_q10`, `rul_median`, `rmrl`, priority, sorted by risk. Scoring the template reproduces the notebook's predictions exactly.

---

## 13a. Training and scoring studio

`trainer/` is a second entry point that runs the notebook pipeline on an uploaded CSV and keeps the result as a *run*.

- `trainer/pipeline.py`: `prepare_frame` validates the table (required columns, parseable dates, `event` in {0, 1}, positive `duration_h` and `daily_utilization_h`, unique `drone_id`, at least 500 drones and 30 / 10 / 10 failures in train / validation / test, a positive remaining-life horizon) and derives `in_service` as in section 2.3; `train` repeats notebook sections 2-8 with the same parameters (split quantiles 0.70 and 0.80, `tune` with 60 trials by default, `Model.calibrate` and the $v > 0$ rule, evaluation grid, paired bootstrap with seed 42, SHAP, landmark backtest, RUL, priorities) and writes the artifacts of section 12 plus `meta.json`.
- `trainer/runs.py`: `RUNS_DIR/<run_id>/` holds `status.json` (state, stage, progress log, settings, source file name, row count, a hash of the data, test metrics) and, after success, `artifacts/` and `models/`. Training happens in `work/` and is moved into place only on success. Run ids are generated and checked to stay inside `RUNS_DIR`. One job at a time; a stale `running` state after a container stop becomes `failed`.
- `trainer/app.py`: Streamlit UI (Data, Train, Runs, Results, Score; English and Russian) built on `dashboard.data.load`, `common.score_upload`, `figures` and `insights`, so scoring validation, `out_of_range` flags and priorities are identical to the dashboards. The Score tab can use either model of the run; the risk groups are terciles of the chosen model's test risk.
- Docker: `Dockerfile` (python:3.13-slim, `uv sync --frozen --no-dev`, non-root user) and `docker-compose.yml` (port 8501, volume `studio_runs` at `/app/runs`).

Check: on `data/drone_survival.csv` with default settings the studio reproduces `artifacts/metrics.parquet` exactly and scoring the template reproduces the notebook's `p_fail_window`. The Docker image was built and run on Linux aarch64: health check passed, a run trained inside the container and was still listed after a container restart. On this platform `ecos` and `scikit-survival` have no wheels and are compiled in the builder stage (about 35 s with the layer cache cold, apart from downloads); the final image carries no compiler. Not tested: linux/amd64 builds and the browser UI inside the container.

Caveats: the schema is fixed; the model choice, thresholds and horizons follow the notebook and cannot be changed in the UI; cancellation takes effect between stages, not inside a tuning stage; the app has no authentication.

---

## 14. Modelling caveats

1. **Synthetic data.** All conclusions hold for the generator's assumptions: fixed covariates, Weibull mechanisms, shared gamma frailty, independent censoring. On a real fleet, check them before trusting probabilities.
2. **Time-fixed features.** Wear, firmware updates and changes in environment are not modelled; conditional survival assumes today's features are those at commissioning.
3. **Competing risks are pooled.** The models forecast "any critical failure". Cause-specific forecasts (cause-specific hazards or Fine-Gray) are not implemented; `failure_mode` is descriptive only.
4. **Cox baseline support.** $\hat H_0$ is flat after the last training failure; the RUL horizon is capped, and uploaded ages above the test fleet range are flagged.
5. **Residual overstatement for the riskiest drones.** Against the generator's truth (section 15), the two high-priority drones of the selected AFT model are forecast at 61% vs a true 53%, and AFT's expected remaining life is 621 h vs a true 536 h. Cox's recalibration removes most of its overstatement (riskiest fifth 37.5% → 29.7% vs a true 28.2%) but then overshoots the expected remaining life (557 h). Use the high-priority list as a ranking and its probabilities as an upper estimate.
6. **IPCW independence.** Administrative censoring depends on `daily_utilization_h`, so IPCW-based metrics are approximate.
7. **Optimistic validation.** Validation was used for early stopping, tuning, model choice and the fold refits' early stopping; quote test metrics.
8. **Bootstrap scope.** Intervals reflect test-sample noise with fixed models, not retraining variability.
9. **Kernel edge bias.** Smoothed hazards (observed and Cox) are biased downwards within one bandwidth of 0 and of the last event.
10. **Calibration bins.** Decile calibration uses the bin's last Kaplan-Meier value when follow-up ends before the horizon.
11. **Cohort split.** Harmless here because nothing depends on calendar time; with calendar effects it would leak.
12. **Small groups.** The quintile backtest has 150 drones per group; its 95% intervals are wide (top group: 22–37% around an observed 28.4% for Cox, 18–33% around 24.9% for AFT).

---

## 15. Reference results

From the last full run of the current code (5000 drones, seed 42, `generated_at` 2026-11-15 20:41 in `meta.json`). A re-run reproduces them up to library-level non-determinism.

**Tuning**: 60 trials per model. Cox: depth 3, eta 0.102, best iteration 117. AFT: log-logistic, scale 0.738, depth 5, eta 0.032, best iteration 345.

**Recalibration**: Cox $v = 0.289$, $k = 1.052$ (applied); AFT $v = 0$, $k = 1.033$ (not applied).

**Metrics** (C-index Harrell / Uno / mean AUC / IBS):

- Cox: validation 0.730 / 0.718 / 0.788 / 0.152; test 0.672 / 0.665 / 0.700 / 0.171.
- AFT: validation 0.733 / 0.719 / 0.797 / 0.152; test 0.666 / 0.658 / 0.687 / 0.175.
- Best model by validation IBS: AFT (0.1519 against 0.1524 for Cox, a near tie). On test Cox is ahead on every metric.

**Bootstrap (Cox − AFT, test)**: ranking metrics do not differ (p = 0.27, 0.24, 0.08); IBS −0.0045, 95% CI [−0.0084, −0.0008], p = 0.018: significant on its own, not after Bonferroni.

**Fit checks (test)**: fleet hazard inside the observed 95% band on 98% of the range for Cox (AFT 72%), on average 9% below the observed rate (AFT 7%); mean curve inside the Kaplan-Meier band on 100% of the informative range up to 385 h for Cox (AFT 88%); mean decile calibration error at 119 h 3.7 pp for Cox (AFT 4.4 pp).

**Conditional P(fail in 100 h)** at ages 0 / 119 / 219 / 348 h: observed 17.5% / 15.2% / 17.5% / 17.1%, Cox 17.3% / 14.4% / 15.9% / 15.5%, AFT 18.3% / 16.8% / 14.9% / 13.0%. The fleet hazard is roughly flat at about 0.18 failures per 100 flight hours: wear-out of motors and batteries and infant mortality of electronics offset each other at fleet level. AFT's falling hazard is its imposed shape, not a feature of the data.

**Quintile backtest ($L$ = 119 h)**, predicted vs observed failure share in 100 h:

- Cox: 6.2 / 8.6 / 11.5 / 17.0 / 30.9% vs 3.9 / 10.4 / 16.1 / 17.1 / 28.4%, all inside the 95% intervals; before the recalibration the top quintile was 35.1%.
- AFT: 6.2 / 10.4 / 14.8 / 22.5 / 35.1% vs 5.5 / 11.1 / 10.7 / 23.4 / 24.9%; the top quintile lies above its interval (18.4–33.1%).

**SHAP shares of the total mean |SHAP|**: Cox, temperature 21%, humidity 11%, battery cycles 10%, firmware 9%, environment 8%; AFT, hard landings 57%, temperature 8%, humidity 6%, firmware 4%. The two models rank the factors differently.

**Drones in service (test)**: 500 (53 of the 553 censored test drones left the fleet earlier); priority High 2, Medium 92, Low 406 with AFT; at a high-priority threshold of 40% / 30% the High count is 4 / 25. RUL horizon 1111 h; 48 drones have a median remaining life beyond it.

**Oracle check against the generator** (true conditional failure probability with the known Weibull hazards and frailty variance 0.3, drones in service; true mean P(failure in 100 h) 14.7%, expected failures 74, expected remaining life 536 h):

| | mean P(100 h) | RMSE, pp | riskiest fifth (true) | expected failures | expected remaining life, h |
|---|---|---|---|---|---|
| Cox, no recalibration | 17.4% | 7.6 | 37.5% (28.2%) | 87 | 494 |
| Cox, recalibrated | 15.0% | 5.3 | 29.7% (28.2%) | 75 | 557 |
| AFT (selected) | 14.1% | 6.3 | 26.9% (26.5%) | 70 | 621 |

The riskiest fifth is taken by each model's own ranking, so its true value differs slightly between rows. The two high-priority drones of AFT: forecast 61.3% against a true 53.2%. Rows were recomputed for this run with a scratch script that is not kept in the repository.

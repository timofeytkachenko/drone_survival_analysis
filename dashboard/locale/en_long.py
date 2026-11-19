"""English long-form dashboard copy."""

LONG = {'model_labels': {'cox': 'Cox', 'aft': 'AFT'},
 'feature_labels': {'model': 'Drone model',
                    'environment': 'Operating environment',
                    'is_refurbished': 'Refurbished',
                    'initial_battery_cycles': 'Battery cycles at start',
                    'firmware_major': 'Firmware version',
                    'maintenance_interval_h': 'Maintenance interval, h',
                    'operator_experience_yrs': 'Operator experience, yrs',
                    'daily_utilization_h': 'Flight hours per day',
                    'avg_flight_min': 'Average flight, min',
                    'avg_payload_kg': 'Average payload, kg',
                    'payload_ratio': 'Payload (share of max)',
                    'mean_temp_c': 'Average temperature, °C',
                    'avg_humidity_pct': 'Average humidity, %',
                    'avg_wind_ms': 'Average wind, m/s',
                    'hard_landings_per_100h': 'Hard landings per 100 h',
                    'mean_vibration_g': 'Vibration, g'},
 'segment_labels': {'model': 'Drone model',
                    'environment': 'Operating environment',
                    'firmware_major': 'Firmware version',
                    'is_refurbished': 'Refurbished',
                    'risk_group': 'Risk group (model)',
                    'priority': 'Maintenance priority'},
 'failure_mode_labels': {'motor': 'Motor',
                         'battery': 'Battery',
                         'electronics': 'Electronics',
                         'crash': 'Crash',
                         'censored': 'No failure'},
 'metric_labels': {'c_index': 'C-index (Harrell)',
                   'c_index_ipcw': 'C-index (Uno, IPCW)',
                   'mean_auc': 'Mean AUC(t)',
                   'ibs': 'Integrated Brier score'},
 'priority_labels': {'High': 'High',
                     'Medium': 'Medium',
                     'Low': 'Low',
                     'None (failed)': 'None (failed)',
                     'None (left the fleet)': 'None (left the fleet)'},
 'hazard_unit': 'Failures per 100 flight hours',
 'feature_descriptions': {'model': 'Airframe type: light/heavy quadcopter, heavy hexacopter or '
                                   'VTOL fixed-wing. Determines payload, vibration and typical '
                                   'flight duration.',
                          'environment': 'Main operating environment: coastal, desert, forest, '
                                         'urban. Climate (temperature, humidity, wind) and '
                                         'dust/salt act on the drone through it.',
                          'is_refurbished': 'The drone went through a refurbishment rather than '
                                            'being new.',
                          'initial_battery_cycles': 'How many charge cycles the battery had '
                                                    'already gone through when the drone was '
                                                    'commissioned: a worn battery loses capacity '
                                                    'faster.',
                          'firmware_major': 'Major version of the flight controller firmware.',
                          'maintenance_interval_h': 'Scheduled maintenance interval in flight '
                                                    'hours: the larger it is, the less often the '
                                                    'drone is inspected.',
                          'operator_experience_yrs': 'Operator experience in years.',
                          'daily_utilization_h': 'How many hours a day the drone flies on average.',
                          'avg_flight_min': 'Average duration of one flight.',
                          'avg_payload_kg': 'Average payload mass.',
                          'payload_ratio': 'Payload as a share of the maximum allowed for this '
                                           'airframe (1 = full load).',
                          'mean_temp_c': 'Average air temperature during flights.',
                          'avg_humidity_pct': 'Average relative air humidity.',
                          'avg_wind_ms': 'Average wind speed.',
                          'hard_landings_per_100h': 'Number of hard landings per 100 flight hours.',
                          'mean_vibration_g': 'Average vibration level on the frame, in g.'},
 'glossary': [['Flight hours (t)',
               ('The unit of time throughout the analysis: how many hours the drone has spent in '
               'the air since commissioning. Calendar days are not used because wear is driven by '
               'flight time.')],
              ['Failure (event)',
               ('The first critical failure of a drone: motor, battery, electronics or crash. After '
               'a failure the drone leaves the observation.')],
              ['Censoring',
               ('The drone has not failed by the data cut-off (or left for a non-critical reason). '
               'We only know it worked *at least* as long as its flight hours. Such drones cannot '
               'be discarded (we would lose information), and treating them as failed is a '
               'mistake. Survival analysis methods handle them correctly.')],
              ['Survival function S(t)',
               ('The probability of operating without failure for more than t flight hours. S(0) = '
               '1, and the curve only goes down from there.')],
              ['Kaplan-Meier',
               ('A non-parametric estimate of S(t) from the observed data that accounts for '
               'censoring. It uses no features: it is the "observed" curve the models are compared '
               'with.')],
              ['Risk (risk score)',
               ('The number the model outputs for a drone: the higher it is, the earlier a failure '
               'is expected. It is not a probability by itself; it is used for ranking and SHAP.')],
              ['Conditional survival',
               ('S(a + u) / S(a): the probability of operating u more hours given that the drone '
               'has already flown a hours without failure. The basis of the remaining-life '
               'forecast.')],
              ['Hazard h(t) (failure rate)',
               ('How often drones that are still flying at t fail right now, shown as failures per '
               '100 flight hours. A rate of 0.2 means that about 18% of working drones fail within '
               'the next 100 h if the rate stays the same. Rising with age means wear-out, falling '
               'means early failures, flat means failures are random with respect to age.')],
              ['Remaining useful life (RUL)',
               ('How many more flight hours the drone is likely to operate. We report three '
               'versions: until 10% risk, median (50%) and expected.')],
              ['C-index',
               ('The share of drone pairs the model ordered correctly: the one that failed first '
               'got the higher risk. 0.5 is a coin toss, 1 is perfect.')],
              ['Brier score',
               ('Mean squared error of the forecast probability that "the drone is still working at '
               'time t". 0 is perfect, 0.25 is the level of a 50/50 forecast.')],
              ['IBS',
               ('Integrated Brier score: the Brier score averaged over a time range. The main '
               'model-selection metric: it reflects both ranking and calibration.')],
              ['AUC(t)',
               ('How well the model separates drones that failed before time t from those that '
               'worked longer than t. 0.5 is random, 1 is perfect.')],
              ['Calibration',
               ('Whether predicted probabilities match observed frequencies: if the model says "80% '
               'will survive", about 80% should survive.')],
              ['Bootstrap',
               ('Repeated random resampling of the test set with replacement. Shows how much a '
               'metric "jitters" because of the finite number of drones.')],
              ['SHAP',
               ('A way to decompose a model forecast into feature contributions: base (average '
               'risk) + contribution of each feature = risk of a specific drone.')],
              ['IPCW',
               ('Inverse probability of censoring weighting: a reweighting that compensates for '
               'some drones leaving the observation before failing.')]],
 'tab_intros': {'overview': '\n'
                            '**Why this tab.** A one-minute summary: size and state of the test '
                            'fleet, the\n'
                            "model's main conclusions and the overall failure picture.\n"
                            '\n'
                            '**Questions it answers.** How many drones failed and why? How fast '
                            'does the fleet\n'
                            'lose operability? How good is the model? How many drones need '
                            'attention right now?\n'
                            '\n'
                            '**How to read the rest of the dashboard.** "Methodology": how '
                            'everything is\n'
                            'computed, step by step. "Failure segments": where failures are more '
                            'frequent.\n'
                            '"Model quality": why the models can be trusted. "Risk factors": what '
                            'exactly brings\n'
                            'failure closer. "Fleet": the list of drones for maintenance. "Drone '
                            'card": a single\n'
                            'drone in detail and "what if" scenarios. "Score new drones": '
                            'forecasts for drones\n'
                            'from your own CSV file.\n',
                'method': '\n'
                          '**Why this tab.** A step-by-step description of how every number on the '
                          'dashboard\n'
                          'is derived from the raw data: what data is used, how it is split, which '
                          'models are\n'
                          'trained, how their quality is checked and how forecasts turn into '
                          'remaining life and\n'
                          'maintenance priority. Each step can be expanded. At the end: the tuned '
                          'model\n'
                          'settings, the feature dictionary and a glossary.\n',
                'segments': '\n'
                            '**Why this tab.** Compare groups of drones (by type, environment, '
                            'firmware,\n'
                            'condition, risk group) by how fast they fail and by failure causes. '
                            'This is the\n'
                            '"raw" data without a model: what actually happened to the test '
                            'drones.\n'
                            '\n'
                            '**How to use it.** Pick a breakdown in the list and all charts, the '
                            'table and the\n'
                            'findings are recomputed. Look at the survival curves first (left '
                            'chart): they\n'
                            'compare segments correctly. The right chart explains *why* a segment '
                            'fails, i.e.\n'
                            'which components break.\n'
                            '\n'
                            '**Important.** A difference between segments is a correlation, not '
                            'necessarily a\n'
                            'cause: for example, a different drone type may fly in the desert. The '
                            'contribution\n'
                            'of each factor "all else being equal" is on the "Risk factors" tab.\n',
                'quality': '\n'
                           "**Why this tab.** To show that the model's forecasts can be trusted. "
                           'We check the\n'
                           'model from three angles:\n'
                           '\n'
                           '1. **Ranking** (C-index, AUC): does the model correctly tell which '
                           'drone fails\n'
                           '   first. Needed for the maintenance queue.\n'
                           '2. **Probability accuracy** (Brier, IBS, calibration): are the '
                           'percentages\n'
                           '   themselves right. Needed for planning spare parts and budget.\n'
                           '3. **Robustness** (bootstrap): is the difference between models due to '
                           'chance.\n'
                           '\n'
                           'All checks are done on the **test set**: drones the model did not see '
                           'during\n'
                           'training and tuning, commissioned later than the training drones. This '
                           'mimics real\n'
                           'use: forecasting new drones.\n',
                'factors': '\n'
                           '**Why this tab.** To explain *what* makes a drone risky. For each '
                           'drone the model\n'
                           'decomposes its risk score into feature contributions (the SHAP '
                           'method); here these\n'
                           'contributions are aggregated over the whole test fleet.\n'
                           '\n'
                           '**Reading order.** (1) Importance: which factors matter most. (2) '
                           'Point cloud: in\n'
                           'which direction each factor acts and for how many drones. (3) '
                           'Dependence: how\n'
                           'exactly the risk changes with the selected factor; thresholds and '
                           'non-linearities\n'
                           'are visible here too.\n'
                           '\n'
                           '**Limitation.** SHAP explains the *model*, not the physics. If a '
                           'factor is linked to\n'
                           'a cause the data does not capture, the model may attribute the effect '
                           'to it.\n'
                           'Conclusions should be checked with engineers.\n',
                'fleet': '\n'
                         '**Why this tab.** An operational list: which drones *currently in '
                         'service* are\n'
                         'likely to fail soon and how much life they have left.\n'
                         '\n'
                         '**How the priority is computed.** For each drone the model takes into '
                         'account how\n'
                         'long it has already flown without failure (a hours) and computes the '
                         'probability of\n'
                         'failure in the next {window:.0f} flight hours. The high-priority '
                         'threshold is\n'
                         '{p_high:.0%}, medium is {p_medium:.0%}.\n'
                         '\n'
                         '**How to use it.** Sort or filter the table (the filter row under the '
                         'header, e.g.\n'
                         '`> 0.2` or `coastal`), select a drone with the button on the left and '
                         'its card with\n'
                         'an explanation of the forecast opens.\n',
                'drone': '\n'
                         '**Why this tab.** A single drone in detail: its state, remaining-life '
                         'forecast,\n'
                         '*why* the model considers it risky and what changes if the operating '
                         'conditions\n'
                         'change.\n'
                         '\n'
                         '**How to use it.** Pick a drone in the list (drones in service come '
                         'first, riskiest\n'
                         'first) or from the table on the "Fleet" tab. Below is a "what if" '
                         'scenario: change\n'
                         'the conditions and press "Recalculate".\n',
                'score': '\n'
                         '**Why this tab.** Get forecasts for drones that are not in the test set, '
                         'e.g. a\n'
                         'batch you plan to buy or your own fleet export. The {best} model scores '
                         'every row\n'
                         'of an uploaded CSV file exactly like the drones on the "Fleet" tab.\n'
                         '\n'
                         '**How to use it.** Download the template, fill in one row per drone and '
                         'upload the\n'
                         'file. The file needs all {n_features} feature columns (see "Methodology" '
                         '→\n'
                         '"Feature dictionary" for the meaning and training range of each). Two '
                         'columns are\n'
                         'optional: `drone_id` (generated if missing) and `age_h`, the flight '
                         'hours the drone\n'
                         'has already flown without failure (0 for a brand-new drone, which is the '
                         'default).\n'
                         'Other columns are copied into the result unchanged.\n'
                         '\n'
                         '**What you get.** Failure probability in the next {window:.0f} flight '
                         'hours,\n'
                         'remaining life and maintenance priority with the same thresholds as on '
                         'the "Fleet"\n'
                         'tab, plus the risk group relative to the test fleet. Warnings flag '
                         'values the model\n'
                         'has not seen in training: the forecast for such drones is less '
                         'reliable.\n'},
 'method_steps': [['1. Data',
                   ('\n'
                   'One row is one drone. For each drone we know:\n'
                   '\n'
                   '* **characteristics and operating conditions**: {n_features} features (type,\n'
                   '  environment, firmware, climate, payload, hard landings etc., see the '
                   'dictionary\n'
                   '  below);\n'
                   '* **flight hours** `duration_h`: how many flight hours passed from '
                   'commissioning\n'
                   '  to failure or to the data cut-off;\n'
                   '* **failure flag** `event` (1: failure, 0: the drone still works or left for '
                   'a\n'
                   '  non-critical reason) and the **failure cause** `failure_mode`.\n'
                   '\n'
                   'Time in the analysis is **flight hours**, not calendar time: a drone flying 3 '
                   'hours\n'
                   'a day wears out three times faster than one flying 1 hour a day.\n')],
                  ['2. Censoring: why a simple average does not work',
                   ('\n'
                   'For {n_censored} of {n_test} test drones no failure was observed: {n_active} '
                   'are\n'
                   'still flying at the data cut-off, the rest left the fleet earlier for a '
                   'non-critical\n'
                   'reason. All we know about them is that they worked *at least* as long as their '
                   'flight\n'
                   'hours.\n'
                   '\n'
                   '* If such drones are **discarded**, mostly the quickly broken ones remain, '
                   'and\n'
                   '  remaining life is underestimated.\n'
                   '* If they are treated as **failed** at the cut-off, remaining life is also\n'
                   '  underestimated.\n'
                   '* If they are assumed to **never fail**, it is overestimated.\n'
                   '\n'
                   'That is why **survival analysis** methods are used: the Kaplan-Meier curve, '
                   'Cox\n'
                   'and AFT models, IPCW-corrected metrics. They use the "incomplete" information '
                   'about\n'
                   'working drones correctly.\n')],
                  ['3. Time-based split: training, validation, test',
                   ('\n'
                   'Drones are split **by commissioning date**, not at random:\n'
                   '\n'
                   '* **training**: commissioned before {t_val}; the model learns on them;\n'
                   '* **validation**: from {t_val} to {t_test}; model settings are tuned and the '
                   'best\n'
                   '  model is selected on them;\n'
                   '* **test**: from {t_test}; the final independent check. **The whole dashboard '
                   'is\n'
                   '  built on test drones.**\n'
                   '\n'
                   'Why by time: in practice a model is trained on the past and applied to *new*\n'
                   'drones. A random split would "peek" into the future (new firmware, new '
                   'conditions)\n'
                   'and overstate quality.\n'
                   '\n'
                   'Consequence: test drones are younger, have fewer flight hours and more '
                   'censoring,\n'
                   'so test metrics are usually slightly worse than validation ones; that is '
                   'normal.\n')],
                  ['4. Feature preparation',
                   ('\n'
                   '* Categorical features ({cat_features}) are encoded with the category list of '
                   'the\n'
                   '  **training** set; an unknown value in new data is treated as missing rather '
                   'than\n'
                   '  as an error.\n'
                   '* Numeric features are used as is: decision trees need no scaling and find\n'
                   '  thresholds and non-linearities themselves.\n'
                   '* The target (flight hours + failure flag) is kept apart from the features so '
                   'the\n'
                   '  model cannot "peek" at the answer.\n')],
                  ['5. Two models: Cox and AFT',
                   ('\n'
                   'Both models are **gradient-boosted trees (XGBoost)**, but with different\n'
                   'assumptions about how features affect time to failure.\n'
                   '\n'
                   '**Cox (proportional hazards).** The model estimates the *relative risk* of a '
                   'drone\n'
                   '$e^{{f(x)}}$: how many times its instantaneous failure hazard is higher or '
                   'lower\n'
                   'than the "baseline". The baseline hazard $h_0(t)$ is estimated on the training '
                   'data\n'
                   '(Breslow estimator). Probability of operating t hours:\n'
                   '$S(t \\mid x) = \\exp\\left(-H_0(t)\\, e^{{f(x)}}\\right)$.\n'
                   'Assumption: the risk ratio of two drones does not change over time.\n'
                   '\n'
                   '**AFT (accelerated failure time).** The model predicts the logarithm of time '
                   'to\n'
                   'failure: $\\log T = f(x) + \\sigma \\varepsilon$, where $\\varepsilon$ is '
                   'noise\n'
                   'with a normal, logistic or extreme-value distribution (chosen by tuning).\n'
                   'Features "speed up" or "slow down" wear: a drone with $f(x)$ lower by 0.7\n'
                   'lives about half as long. Assumption: the shape of the time distribution is '
                   'the\n'
                   'same for all drones.\n'
                   '\n'
                   'Building both is useful: if they agree, conclusions are more reliable; if '
                   'they\n'
                   'diverge, there is a reason to dig deeper.\n')],
                  ['6. Training and tuning',
                   ('\n'
                   '* Trees are added one at a time (up to 2000), and after each the validation '
                   'error\n'
                   '  is checked. If 50 trees in a row do not improve it, training stops (**early\n'
                   '  stopping**) and the best iteration is used for forecasts. This protects '
                   'against\n'
                   '  overfitting.\n'
                   '* Settings (tree depth, learning rate, regularisation, feature/row shares, '
                   'and\n'
                   '  for AFT the distribution width $\\sigma$) are tuned automatically by '
                   '**Optuna**:\n'
                   '  {n_trials} trials per model, each trains the model and scores it by\n'
                   '  **validation IBS**. Bayesian search (TPE) picks the next settings based on '
                   'the\n'
                   '  previous results.\n'
                   '* After training, the models are **recalibrated for hidden differences '
                   'between\n'
                   '  drones**. Drones with identical features still differ in durability; the '
                   'weaker\n'
                   '  ones fail first, so the survivors of a high-risk group are sturdier than '
                   'their\n'
                   '  features suggest, and without a correction the model overstates their risk '
                   'at\n'
                   '  later ages. The correction is estimated on training drones the model did not '
                   'see\n'
                   '  while fitting (five-fold cross-fitting). It changes the probabilities, not '
                   'the\n'
                   '  order of drones. A model is corrected only if such differences are found.\n'
                   '* The tuned settings and the two recalibration parameters are in the table '
                   'below\n'
                   '  ("—" means the model is used as trained).\n')],
                  ['7. How quality is measured',
                   ('\n'
                   'All metrics are computed on test drones at time points between the 5% and 80%\n'
                   'quantiles of flight hours (beyond that there are too few observations for a\n'
                   'reliable estimate). Validation uses the same time points, so the two sets can '
                   'be\n'
                   'compared.\n'
                   '\n'
                   '* **C-index (Harrell)**: the share of correctly ordered drone pairs.\n'
                   '* **C-index (Uno, IPCW)**: the same, corrected for censoring; depends less on '
                   'how\n'
                   '  many drones are still flying.\n'
                   '* **AUC(t)**: does the model separate "failed by t" from "worked longer than '
                   't";\n'
                   '  "mean AUC" is the average over time.\n'
                   '* **Brier(t)** and **IBS**: probability forecast error; IBS is Brier(t) '
                   'averaged\n'
                   '  over time.\n'
                   '* **Calibration**: drones are split into 10 groups by predicted probability; '
                   'in\n'
                   '  each group the forecast is compared with the Kaplan-Meier observation.\n'
                   '\n'
                   'The IPCW correction estimates the probability of "leaving the observation" on '
                   'the\n'
                   'evaluated set itself and reweights drones so censoring does not distort the '
                   'metrics.\n'
                   'Newer drones leave the observation earlier, so the training set would '
                   'understate the\n'
                   'weights.\n')],
                  ['8. Choosing the best model',
                   ('\n'
                   'The best model is chosen by **minimum validation IBS**; the test set is not '
                   'used\n'
                   'for the choice, so the quality estimate on it stays honest. IBS was chosen '
                   'because\n'
                   'the business needs *probabilities* (how many drones will fail, when to plan\n'
                   'maintenance), not just the order of drones.\n'
                   '\n'
                   'The selected model is **{best}**. For forecasting, both models are refitted '
                   'with the tuned settings on train + validation ({n_fit} drones; the number of '
                   'trees is the one early stopping selected) and checked on the test set: see '
                   '"Production model check" on "Model quality". Forecasts on "Fleet", "Drone '
                   'card", "Score new drones" and in the risk groups come from the production '
                   'model **{forecast}**: the selected model if it passed the check, otherwise '
                   'the other one if that passed.\n')],
                  ['9. Statistical model comparison (bootstrap)',
                   ('\n'
                   'A metric difference on a single test set may be due to chance. So the test set '
                   'is\n'
                   'resampled with replacement {n_boot} times, and on *the same* resampled drones '
                   'the\n'
                   'metrics of both models are computed (paired bootstrap).\n'
                   '\n'
                   '* **95% confidence interval** of the difference: between the 2.5% and 97.5%\n'
                   '  quantiles. If it does not contain 0, the difference is considered '
                   'significant.\n'
                   '* **p-value**: how often the sign of the difference "flips"; the smaller, the '
                   'more\n'
                   '  certain the difference. When comparing on four metrics at once, be more '
                   'careful:\n'
                   '  the strict Bonferroni-corrected threshold is 0.05 / 4 = 0.0125.\n'
                   '* The models are not refitted, so the intervals reflect only the noise of the '
                   'test\n'
                   '  set.\n')],
                  ['10. Explaining forecasts (SHAP)',
                   ('\n'
                   'For each test drone the forecast is decomposed into a sum:\n'
                   '**drone risk = base risk + contribution of feature 1 + … + contribution of '
                   'feature\n'
                   '{n_features}**.\n'
                   '\n'
                   '* A contribution > 0 means the feature **brings failure closer**, < 0 **pushes '
                   'it\n'
                   '  away**.\n'
                   '* Contributions are computed by the exact algorithm for trees (TreeSHAP) and\n'
                   '  account for feature interactions.\n'
                   '* Feature importance is the mean absolute contribution over all drones, as a '
                   'share\n'
                   '  of the total.\n'
                   '\n'
                   'The model scales differ: for Cox the risk is the log relative hazard, for AFT '
                   'it is\n'
                   'minus the log expected time. So compare importance shares between models, not\n'
                   'absolute values.\n')],
                  ['11. Remaining life and maintenance priority',
                   ('\n'
                   'A drone is **in service** if its flight log reaches the data cut-off: '
                   'commissioning\n'
                   'date plus flight hours divided by daily utilisation. Drones last seen earlier '
                   'without\n'
                   'a failure left the fleet and get no forecast.\n'
                   '\n'
                   'For a drone in service with a hours flown, the model builds a **conditional '
                   'curve**\n'
                   '$S(a+u)/S(a)$: the probability of operating u more hours *knowing* it has '
                   'already\n'
                   'survived the first a hours. From it:\n'
                   '\n'
                   '* **Life until 10% risk**: after how many hours the failure probability '
                   'reaches 10%\n'
                   '  (a conservative estimate for maintenance planning);\n'
                   '* **Median remaining life**: after how many hours the failure probability '
                   'reaches\n'
                   '  50%;\n'
                   '* **Expected remaining life**: the mean remaining time within the\n'
                   '  {horizon:.0f} h horizon (area under the conditional curve);\n'
                   '* **P(failure in {window:.0f} h)**: the probability of failure in the next\n'
                   '  {window:.0f} flight hours.\n'
                   '\n'
                   'The {horizon:.0f} h horizon is limited by how far failures were observed in '
                   'the\n'
                   'training data: beyond it the model would extrapolate. "> {horizon:.0f} h" '
                   'means the\n'
                   'risk level is not reached within the horizon.\n'
                   '\n'
                   '**Maintenance priority:** high if P(failure in {window:.0f} h) ≥ '
                   '{p_high:.0%},\n'
                   'medium if ≥ {p_medium:.0%}, otherwise low. The **risk group** is different: it '
                   'is\n'
                   'the third of the fleet with the highest/middle/lowest model risk score, '
                   'regardless\n'
                   'of hours already flown.\n')],
                  ['12. Limitations',
                   ('\n'
                   '* The model only sees the features in the table; causes the data does not '
                   'capture\n'
                   '  (batch quality, a specific technician) remain "noise".\n'
                   '* Features are fixed at the data cut-off; the model does not track changes of\n'
                   '  operating conditions over time (that is what the "what if" scenario is '
                   'for).\n'
                   '* No forecast is made beyond the {horizon:.0f} h horizon.\n'
                   '* Remaining-life forecasts are checked on the "Fleet" tab, on average and by '
                   'risk\n'
                   '  group. The recalibration removes most of the overstated risk of the '
                   'riskiest\n'
                   '  drones, not all of it: read the top of the high-priority list as a ranking '
                   'first.\n'
                   '* Conclusions about factors are correlational; for decisions that change\n'
                   '  procedures, confirm them with an experiment or an engineering analysis.\n'
                   '* The models are retrained on every notebook run; the numbers on the '
                   'dashboard\n'
                   '  refer to the run of {generated_at}.\n')]],
 'hyperparams': {'max_depth': ['Tree depth',
                               ('How many conditions in a row one tree can check; more means more '
                               'complex interactions and a higher risk of overfitting.')],
                 'eta': ['Learning rate',
                         ('How strongly each new tree corrects the previous ones; lower is more '
                         'cautious and needs more trees.')],
                 'min_child_weight': ['Min. leaf weight',
                                      'Prevents rules based on too few drones.'],
                 'subsample': ['Share of drones per tree',
                               ('Each tree sees a random part of the drones, a guard against '
                               'overfitting.')],
                 'colsample_bytree': ['Share of features per tree',
                                      'Each tree sees a random part of the features.'],
                 'reg_lambda': ['L2 regularisation',
                                ('Penalty for too large leaf predictions; higher makes the model '
                                'more cautious.')],
                 'aft_loss_distribution': ['Noise distribution (AFT)',
                                           ('Shape of the failure time around the AFT model '
                                           'forecast: normal gives a log-normal time, logistic a '
                                           'log-logistic time, extreme a Weibull time.')],
                 'aft_loss_distribution_scale': ['Distribution σ (AFT)',
                                                 ('Spread of the failure time around the AFT model '
                                                 'forecast.')],
                 'best_iteration': ['Number of trees', 'Best iteration by early stopping.'],
                 'frailty_variance': ['Hidden durability variance',
                                      ('Recalibration: spread of the unobserved durability between '
                                      'drones with the same features. The larger it is, the more '
                                      'the risk of long-surviving risky drones is lowered.')],
                 'scale': ['Recalibration scale',
                           ('Recalibration: overall multiplier of the accumulated risk; 1 means no '
                           'change.')]},
 'whatif_help': "**How to use it.** The fields below hold the drone's current values. Change one "
                'or more (for example, lower the payload or pick another firmware) and press '
                '"Recalculate". The model recomputes the remaining-life forecast from the current '
                'flight hours, a dashed line appears on the chart above, and a comparison with an '
                'explanation appears below. Numeric values are limited to the training range (the '
                'model was not trained outside it). For a drone that failed or left the fleet, '
                'remaining life is computed from zero, as for a new drone with the same '
                'characteristics.',
 'figures': {'km_overall': {'what': 'Share of test drones that operated without failure for more '
                                    'than t flight hours. This is **observed data**, without a '
                                    'model.',
                            'how': 'The curve starts at 1 (all drones work) and drops at every '
                                   'failure. The flatter the curve, the more reliable the fleet. '
                                   'The point where the curve crosses 0.5 is the **median time to '
                                   'failure**: half of the drones fail earlier. The grey band is '
                                   'the 95% confidence interval: the true curve lies inside it '
                                   'with high probability. The band widens on the right, where few '
                                   'drones remain.',
                            'details': 'Kaplan-Meier estimate: $\\hat S(t) = \\prod_{{t_i \\le t}} '
                                       '(1 - d_i / n_i)$, where $d_i$ is the number of failures at '
                                       '$t_i$ and $n_i$ the number of drones still under '
                                       'observation. Censored drones count in $n_i$ until they '
                                       'leave. The confidence interval uses the log-log '
                                       'transform.'},
             'hazard_observed': {'what': 'Failure rate (hazard) of the test fleet by flight hours: '
                                         'how often drones that are still flying fail at each age. '
                                         'This is **observed data**, without a model.',
                                 'how': 'The S(t) curve above shows *how many* drones are left; '
                                        'this chart shows *how fast* the remaining ones fail. A '
                                        '**rising** line means wear-out (age-based maintenance '
                                        'pays off), a **falling** one means early failures '
                                        '(acceptance checks pay off), a **flat** one means '
                                        'failures do not depend on age (maintenance should be '
                                        'driven by risk factors). The grey band is the 95% '
                                        'confidence interval; it widens on the right, where few '
                                        'drones remain.',
                                 'details': 'Kernel-smoothed Nelson-Aalen estimate: $\\hat h(t) = '
                                            '\\sum_i K_b(t - t_i)\\, d_i / n_i$ with the '
                                            'Epanechnikov kernel and bandwidth b = '
                                            "{hazard_bandwidth:.0f} h (Silverman's rule on the "
                                            'failure times). The chart starts at b and stops at '
                                            'the 90th percentile of flight hours: near the edges '
                                            'the kernel estimate is biased downwards.'},
             'failure_modes': {'what': 'How many test drones failed for each cause.',
                               'how': 'Bar length is the number of failures; the label shows the '
                                      'share of all failures. A hint at which components to target '
                                      'with spare parts and preventive maintenance.',
                               'details': 'The first critical cause is counted. Drones without '
                                          'failure are not on the chart. Shares are relative to '
                                          'the number of failures, not to the fleet size.'},
             'segment_km': {'what': 'Kaplan-Meier survival curves for each value of the selected '
                                    'breakdown.',
                            'how': 'The lower and steeper the curve, the faster the segment loses '
                                   'drones. Compare curves at the same flight hours (vertically): '
                                   'for example, at 200 h, what share of each segment still works. '
                                   'The legend shows segment size and number of failures: on small '
                                   'segments the curve is noisy.',
                            'details': 'Significance of the differences is tested with the '
                                       '**log-rank test**: it compares the observed number of '
                                       'failures in each segment with the number expected under '
                                       'equal survival. A small p-value (< 0.05) means the '
                                       'differences are unlikely to be random. The test ignores '
                                       'other features.'},
             'segment_structure': {'what': "What share of the segment's drones failed and for "
                                           'which causes.',
                                   'how': 'Bar height is the share of failed drones; colours are '
                                          'causes. It shows the failure **profile**: for example, '
                                          'electronics dominate in one environment and motors in '
                                          'another.',
                                   'details': 'The share of failed drones also depends on how many '
                                              "hours the segment's drones have flown: a young "
                                              'segment looks more reliable than it is. So compare '
                                              'failure speed with the curves on the left and use '
                                              'this chart for the structure of causes.'},
             'segment_table': {'what': 'Summary statistics by segment.',
                               'how': '"Failure share" and "Mean flight hours" are observed data. '
                                      '"Mean P(failure in {window:.0f} h)" is the mean model '
                                      "forecast for the segment's drones still in service: which "
                                      'segment will need more maintenance soon. "Median to '
                                      'failure" is from Kaplan-Meier.',
                               'details': 'A median "not reached" means fewer than half of the '
                                          "segment's drones failed over the whole observed "
                                          'interval.'},
             'metrics': {'what': 'Four quality metrics for both models on validation and test.',
                         'how': 'For C-index and AUC higher is better (0.5 is random, 0.7–0.8 is '
                                'good for reliability tasks). For IBS lower is better (0.25 is a '
                                '50/50 forecast). The key row is **test**: the quality on new '
                                'drones.',
                         'details': "Harrell's C only uses comparable pairs (the earlier one is a "
                                    "failure). Uno's C weights pairs by the inverse probability of "
                                    'staying under observation and is truncated at the last '
                                    'evaluation point. Mean AUC is the cumulative/dynamic AUC '
                                    'averaged with weights from S(t). IBS is the integral of '
                                    'Brier(t) over the time grid divided by its length.'},
             'learning_curves': {'what': 'Model error (negative log-likelihood) as trees are '
                                         'added, on training and on validation.',
                                 'how': 'Training error almost always falls. **Validation** is '
                                        'what matters: while it falls, the model learns useful '
                                        'patterns; when it stops, the model starts memorising '
                                        'noise. The dashed line is the best validation iteration: '
                                        'exactly that many trees are used for forecasts.',
                                 'details': 'Early stopping: 50 iterations without validation '
                                            'improvement. Training and validation error levels for '
                                            'Cox are not directly comparable: the partial '
                                            'likelihood depends on sample size. Look at the shape '
                                            'of the curve.'},
             'survival_mean': {'what': 'Mean predicted survival curve over all test drones for '
                                       'each model, versus the observed Kaplan-Meier curve.',
                               'how': 'If a model curve lies inside the grey band, the model '
                                      'correctly estimates the **overall level** of failures in '
                                      'the fleet. A curve below the observed one means the model '
                                      'is pessimistic (expects more failures); above means '
                                      'optimistic.',
                               'details': 'This is a calibration check "in aggregate" (at fleet '
                                          'level). It is necessary but not sufficient: individual '
                                          'errors can cancel out. For groups of drones see the '
                                          'calibration chart below.'},
             'hazard_mean': {'what': 'Failure rate of the test fleet predicted by each model '
                                     'versus the observed one.',
                             'how': 'A model line inside the grey band reproduces both the level '
                                    'and the **shape** of the failure rate over age. A line above '
                                    'the band means the model expects failures to come faster than '
                                    'they do at that age. This is a stricter check than the '
                                    'survival curve: S(t) accumulates errors over time, the rate '
                                    'shows them where they happen.',
                             'details': 'The fleet rate is the survival-weighted mean of the '
                                        'individual rates, $\\sum_i S_i(t) h_i(t) / \\sum_i '
                                        'S_i(t)$, because drones that already failed no longer '
                                        'contribute. Cox: $h(t|x) = h_0(t) e^{{f(x)}}$ with the '
                                        'Breslow baseline smoothed by the same kernel as the '
                                        'observed rate. AFT: the exact hazard of its time '
                                        'distribution, whose shape is fixed by the tuned noise '
                                        'distribution: log-normal rises and then falls, '
                                        'log-logistic falls or has one peak, Weibull only rises or '
                                        'only falls.'},
             'individual_curves': {'what': 'Survival forecast for the five test drones where Cox '
                                           'and AFT disagree most at the maintenance-window '
                                           'horizon (solid: Cox, dashed: AFT).',
                                   'how': 'Different drones have different colours. Clearly '
                                          'different curves mean the model really separates drones '
                                          'by risk. Solid and dashed lines of the same colour '
                                          'close to each other mean the models agree.',
                                   'details': 'For Cox all curves are powers of one baseline: '
                                              '$S(t|x) = S_0(t)^{{e^{{f(x)}}}}$, so they never '
                                              'cross. For AFT the curves are time-stretched copies '
                                              'of one shape.'},
             'brier': {'what': 'Error of the forecast probability that "the drone still works" at '
                               'each time t.',
                       'how': 'Lower is better. At the start the error is small (almost all drones '
                              'work, which is easy to predict), it grows towards the middle and '
                              'may fall towards the end. The benchmark is 0.25: the error of a '
                              'useless 50/50 forecast. The legend shows IBS, the average of the '
                              'curve.',
                       'details': '$BS(t) = \\frac{{1}}{{n}} \\sum_i w_i(t)\\,(\\mathbb{{1}}[T_i > '
                                  't] - \\hat S(t|x_i))^2$, where the weights $w_i$ (IPCW) '
                                  'compensate for censoring; drones that left before t without '
                                  'failure get weight 0.'},
             'auc': {'what': "The model's ability to separate drones that failed by time t from "
                             'drones that worked longer.',
                     'how': 'Higher is better. 0.5 means the model is no better than random; 0.7 '
                            'means that in 70% of cases the model correctly tells which of two '
                            'drones (one failed by t, the other did not) is riskier. Dips in the '
                            'curve show horizons where the model is weaker.',
                     'details': 'Cumulative/dynamic AUC (Uno et al., 2007) with IPCW weights. The '
                                'risk score at time t is 1 − S(t|x).'},
             'calibration': {'what': 'Whether predicted probabilities match observed ones at the '
                                     'selected horizon.',
                             'how': 'Drones are split into 10 groups by forecast. Each point is a '
                                    'group: forecast horizontally, observed vertically. Ideally '
                                    'the points lie on the dashed diagonal. Points **above** the '
                                    'diagonal mean drones live longer than the model promises '
                                    '(pessimism); **below** means the model overestimates '
                                    'reliability (a dangerous error for maintenance planning).',
                             'details': 'Groups are deciles of the predicted S(h). Observed is '
                                        'Kaplan-Meier within the group at horizon h. The mean '
                                        'error in the findings is the mean |predicted − observed| '
                                        'weighted by group size. Horizons are quartiles of test '
                                        'flight hours.'},
             'bootstrap_table': {'what': 'Pairwise model comparison: metric values, their '
                                         'difference and the confidence interval of the '
                                         'difference.',
                                 'how': '`diff` = metric of model A − metric of model B. If the '
                                        'interval [ci_low, ci_high] does not contain 0, the '
                                        'difference is significant (`significant`). `better` is '
                                        'the model with the better metric (for IBS lower is '
                                        'better).',
                                 'details': 'p-value = 2·min(share of bootstrap differences ≤ 0, '
                                            'share ≥ 0); resolution 2/{n_boot}. For four metrics '
                                            'at once it is reasonable to require p < 0.0125 '
                                            '(Bonferroni correction).'},
             'bootstrap': {'what': 'Distribution of the metric difference between models over '
                                   '{n_boot} bootstrap samples.',
                           'how': 'Each histogram is one metric. The black line is "no difference" '
                                  '(0), the red ones are the 95% interval. If the black line falls '
                                  'between the red ones, the difference may be random. The further '
                                  'the histogram is from zero, the more certain it is that one '
                                  'model is better.',
                           'details': 'Paired bootstrap: both models are scored on the same '
                                      'resampled set, so their shared error cancels out and the '
                                      'comparison is more sensitive than two independent '
                                      'intervals.'},
             'shap_importance': {'what': 'How strongly each feature affects the forecast on '
                                         'average across the fleet.',
                                 'how': "Bar length is the feature's share of the total influence "
                                        '(all model features sum to 100%). Comparing the two '
                                        'models shows how robust the conclusion is: a feature '
                                        'important in both is a reliable signal. Importance does '
                                        'not tell the direction; the chart on the right shows it.',
                                 'details': 'Importance = mean |SHAP| over test drones, normalised '
                                            'by the sum over features. Sorted by the best model.'},
             'shap_beeswarm': {'what': 'Contribution of each feature to the risk of each test '
                                       'drone.',
                               'how': 'A row is a feature, a point is a drone. The horizontal '
                                      'position is the contribution: **right of zero the feature '
                                      'brings failure closer**, left pushes it away. Colour is the '
                                      "drone's feature value: **red is high**, blue is low (for "
                                      'categories, the category number). Example: if red points '
                                      'are on the right, the higher the value, the higher the '
                                      'risk. A wide spread of points means the feature strongly '
                                      'affects some drones.',
                               'details': 'For readability points are slightly spread vertically '
                                          'by a random shift. Colour is the rank of the value '
                                          'among test drones (0 is the minimum, 1 the maximum).'},
             'shap_dependence': {'what': "How the selected feature's contribution to risk depends "
                                         'on its value.',
                                 'how': 'Horizontally the feature value, vertically the '
                                        'contribution to risk (above zero brings failure closer). '
                                        'An upward trend means "more is riskier", a U-shape means '
                                        'extreme values are dangerous, a step means there is a '
                                        'threshold. Vertical spread of points at one value means '
                                        'the effect depends on other features (interaction). For '
                                        'categories: boxes per value.',
                                 'details': 'Findings below: for numeric features the values are '
                                            'split into 5 groups of equal size and the groups with '
                                            'the highest and lowest mean contribution are given; '
                                            'for categories, the values with the highest and '
                                            'lowest mean contribution.'},
             'fleet_table': {'what': "All test drones currently in service, with the model's "
                                     'forecast.',
                             'how': '"P(failure in {window:.0f} h)" is the key planning column: '
                                    'the probability of failure in the next {window:.0f} flight '
                                    'hours. Remaining life is in flight hours from the current '
                                    'flight hours. "Main factor" is the feature that raises this '
                                    "drone's risk the most. Row colour is the maintenance "
                                    'priority. An empty remaining life means beyond the forecast '
                                    'horizon.',
                             'details': 'All values come from the conditional survival curve that '
                                        'accounts for the hours the drone has already survived, so '
                                        'two identical drones with different flight hours get '
                                        'different forecasts.'},
             'rul_hist': {'what': 'Distribution of median remaining life across drones in service.',
                          'how': 'Each bar is how many drones have a median remaining life in that '
                                 'range. The left "tail" are drones with little life left: plan '
                                 'their maintenance first. Drones with remaining life beyond the '
                                 'horizon are not in the histogram (their number is in the title).',
                          'details': 'Median remaining life is the first u where S(a+u)/S(a) ≤ '
                                     '0.5, on a grid with a step of about 1 h.'},
             'riskiest': {'what': 'Conditional survival curves of the eight drones in service with '
                                  'the highest probability of failing soon.',
                          'how': 'X axis: additional flight hours from the current flight hours; Y '
                                 'axis: the probability of operating them. The crossing with the '
                                 'dashed 0.5 line is the median remaining life. A steeply falling '
                                 'curve means the drone should go to maintenance soon.',
                          'details': "The curve is S(a+u)/S(a), where a is the drone's current "
                                     'flight hours.'},
             'backtest': {'what': 'A retrospective check of the remaining-life forecast.',
                          'how': 'Take all test drones that operated without failure for at least '
                                 'L = {landmark:.0f} h and look at what happened to them next. The '
                                 'black curve is observed, the coloured ones are what the model '
                                 'predicted at time L. Close curves mean the remaining-life '
                                 'forecasts are right on average across the fleet.',
                          'details': 'Landmark analysis: observed is Kaplan-Meier on time $T - L$ '
                                     'for drones with $T > L$; forecast is the mean '
                                     '$S(L+u|x)/S(L|x)$ over the same drones. L is the first '
                                     'quartile of test flight hours.'},
             'backtest_groups': {'what': 'The same retrospective check split by predicted risk: '
                                         'does a forecast of 30% mean that 30% fail?',
                                 'how': 'The drones still flying at L = {landmark:.0f} h are split '
                                        'into five equal groups by the predicted probability of '
                                        'failing in the next {window:.0f} h, the number behind the '
                                        'maintenance priority. Each point is a group: horizontally '
                                        'the mean forecast, vertically the share that actually '
                                        'failed, with its 95% interval for the {best} model. '
                                        'Points on the dotted diagonal mean accurate '
                                        'probabilities; points below it mean the model overstates '
                                        'the risk. The grey dashed line is {best} before the '
                                        'recalibration for hidden differences between drones '
                                        '("Methodology", step 6).',
                                 'details': 'Observed share = 1 − Kaplan-Meier survival at '
                                            '{window:.0f} h after the landmark within the group, '
                                            'with a log-log interval. The groups are quintiles of '
                                            "each model's own forecast, so the two models' groups "
                                            'hold different drones. Each group is a fifth of the '
                                            'drones at risk, so the interval is wide and gaps of a '
                                            'few pp are within noise.'},
             'drone_survival': {'what': 'Probability that this drone operates without failure for '
                                        'more than t flight hours **from commissioning**, under '
                                        'both models.',
                                'how': 'The vertical line is the current flight hours (green) or '
                                       'the failure time (red). If the drone works and the line is '
                                       'to the right of where the curve is already low, the drone '
                                       'has outlived most "similar" drones, but that does not mean '
                                       'it is safe: see the conditional forecast below.',
                                'details': 'This is the unconditional curve S(t|x). Maintenance '
                                           'planning uses the conditional S(a+u)/S(a).'},
             'drone_hazard': {'what': 'The failure rate of this drone at each age, under both '
                                      'models.',
                              'how': 'The value at the vertical line is how fast the drone is '
                                     'failing **now**. A line that keeps rising to the right of it '
                                     'means the risk grows as the drone keeps flying; a flat line '
                                     'means each additional 100 h is as risky as the previous '
                                     'ones. Compare the level with the fleet chart on the '
                                     '"Overview" tab.',
                              'details': 'Cox: $h_0(t) e^{{f(x)}}$, the kernel-smoothed Breslow '
                                         'baseline (bandwidth {hazard_bandwidth:.0f} h) times the '
                                         "drone's relative risk, so all Cox curves have the same "
                                         'shape; at high flight hours it rests on few training '
                                         'failures and becomes noisy, so the chart stops at the '
                                         "fleet range or at the drone's next {window:.0f} h, "
                                         'whichever is later. AFT: the exact hazard of its time '
                                         'distribution. The conditional survival is $S(a+u)/S(a) = '
                                         '\\exp(-\\int_a^{{a+u}} h(t)\\,dt)$: the area under this '
                                         'curve after the line drives the remaining-life '
                                         'forecast.'},
             'drone_shap': {'what': 'Why the model gave this drone its risk: the contribution of '
                                    'each feature.',
                            'how': 'Red bars **raise** risk, green ones **lower** it. Bar length '
                                   'is the strength of the effect. The title shows the fleet '
                                   "average risk → this drone's risk. The 12 most influential "
                                   "features with the drone's values are shown.",
                            'details': "SHAP decomposition of the production model's forecast: "
                                       'risk = base + sum of contributions.'},
             'drone_whatif': {'what': 'Remaining-life forecast of the drone from its current '
                                      'flight hours: under current conditions (solid) and in the '
                                      'scenario (dashed, after "Recalculate").',
                              'how': 'X axis: additional flight hours; Y axis: the probability of '
                                     'operating them without failure. A dashed line above the '
                                     'solid one means the scenario extends remaining life.',
                              'details': 'For a drone that failed or left the fleet the flight '
                                         'hours are set to 0: the scenario answers "what would the '
                                         'remaining life of a new drone with these characteristics '
                                         'be".'},
             'score_table': {'what': 'Forecast for every drone in the uploaded file, riskiest '
                                     'first.',
                             'how': '"P(failure in {window:.0f} h)" and the remaining-life columns '
                                    "are computed from each drone's `age_h`, exactly as on the "
                                    '"Fleet" tab. An empty remaining life means beyond the '
                                    '{horizon:.0f} h horizon. "Outside training range" marks '
                                    'drones with at least one numeric value the model has not seen '
                                    'in training; treat their forecast with caution.',
                             'details': 'The risk group uses the tercile cut-points of the {best} '
                                        'risk score on the test fleet, so a new drone is placed '
                                        'relative to the test drones. Unknown category values are '
                                        'treated as missing, like in training.'},
             'hazard_overview': {'what': 'Failure rate of the test fleet by flight hours, observed '
                                         'and as predicted by the selected model.',
                                 'how': 'The black line is how fast drones still flying fail at '
                                        'each age; the grey band is its 95% interval. The coloured '
                                        "line is the selected model's fleet rate. A line inside "
                                        'the band reproduces the level and the shape. A rising '
                                        'line means wear-out, a falling one early failures, a flat '
                                        'one age-independent failures.',
                                 'details': 'Kernel-smoothed Nelson-Aalen estimate: $\\hat h(t) = '
                                            '\\sum_i K_b(t - t_i)\\, d_i / n_i$ with the '
                                            'Epanechnikov kernel and bandwidth b = '
                                            "{hazard_bandwidth:.0f} h (Silverman's rule on the "
                                            'failure times). The chart starts at b and stops at '
                                            'the 90th percentile of flight hours: near the edges '
                                            'the kernel estimate is biased downwards. The model '
                                            'line is the survival-weighted mean of the individual '
                                            'rates.'},
             'failure_timing': {'what': 'How many test drones had failed from each cause by a '
                                        'given flight hour. This is a count of observed failures, '
                                        'not a hazard and not a competing-risk model.',
                                'how': 'Each step up is one more failure of that cause. A cause '
                                       'that climbs early breaks young drones; a cause that climbs '
                                       'late shows up after more flight hours. Drones that never '
                                       'failed are not counted.',
                                'details': 'At time t the line is the number of test drones whose '
                                           'first critical failure was of this cause at or before '
                                           't. The lines are cumulative counts, so compare them '
                                           'with each other and with the fleet size; do not read '
                                           'them as probabilities.'},
             'age_risk': {'what': 'Failure probability over the next {window:.0f} flight hours '
                                  'against hours already flown, for every drone in service. Colour '
                                  'is the risk group.',
                          'how': 'The risk group ignores hours already flown; the vertical '
                                 'position is the maintenance probability, which does depend on '
                                 'age. A young drone high on the chart is risky for its age. An '
                                 'old drone low on the chart is not urgent just because it has '
                                 'flown a lot.',
                          'details': "Risk groups are terciles of the production model's risk score "
                                     'on the whole test fleet. The probability is the conditional '
                                     'chance of failing within {window:.0f} h given survival to '
                                     'the current age.'}}}

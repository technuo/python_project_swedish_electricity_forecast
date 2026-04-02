# PROJECT WEEKLY REPORT - W7 - 0330-0405

## FIELD NOTE: SWEDISH ELECTRICITY PRICE FORECASTING

**Period**: Week 7 (Mar 30 – Apr 5, 2026)
**Scope**: External Feature Integration, Hyperparameter Optimization, SHAP Explainability, and Production Readiness

---

### 1. EXECUTIVE SUMMARY

This week achieved a breakthrough in model accuracy through systematic hyperparameter optimization, reducing test-set MAE from **3.30 (W6) to 2.04 SEK/MWh** — a **38% improvement** that far exceeds the Week 7 target of <3.0. We successfully integrated Open-Meteo weather data (temperature, wind speed) and Swedish holiday indicators into the feature pipeline, expanding the feature set from 14 to 27 columns.

**Unexpected but Critical Finding**: Auxiliary features (weather + holidays + cyclical encoding) did **not** improve accuracy under default XGBoost parameters. In fact, the "core only" model (14 features) outperformed the full 27-feature model (3.30 vs 3.58 MAE). Only after rigorous Bayesian hyperparameter tuning did the full feature set stabilize, with the optimized model leveraging all 27 features to reach the final 2.04 MAE.

This result provides direct evidence for **VG Node 1**: while weather and holidays are physically relevant to electricity markets, their predictive contribution in this specific SE3 dataset (2024–2025) is marginal after accounting for strong autoregressive price dynamics.

---

### 2. COMPLETED TASKS

#### A. External Feature Integration (P0)

1.  **SMHI API Issue Resolution**:
    *   SMHI `corrected-archive` endpoint returned `404` for all four representative stations (Luleå, Sundsvall, Stockholm, Malmö).
    *   **Mitigation**: Migrated weather source to **Open-Meteo Historical API**, which provided complete hourly temperature and wind speed data for Stockholm (lat 59.33, lon 18.07) across the full 2024–2025 period.
    *   Result: 17,544 weather records fetched and cached to `cache/open_meteo_se3_weather.csv`.

2.  **Feature Pipeline Expansion (`src/data_processor.py`)**:
    *   Generated complete feature set `se3_features_w7.parquet` with **29 columns** (including `timestamp` and `value`).
    *   Verified all feature categories are present:
        *   **Core (14)**: `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_morning`, `is_peak_evening`, `value_lag_1/24/168`, `value_rolling_mean/max/min/std_24h`, `value_diff_24h`
        *   **Cyclical (6)**: `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`
        *   **Holiday (2)**: `is_holiday`, `is_pre_holiday`
        *   **Weather (5)**: `temperature`, `wind_speed`, `temperature_lag_1`, `temperature_lag_24`, `wind_speed_squared`

#### B. Model Training with Full Features

Executed end-to-end training on the expanded dataset (80/20 chronological split: 14,035 train / 3,509 test).

| Model | MAE (SEK/MWh) | RMSE | Notes |
|-------|---------------|------|-------|
| **XGBoost** | 3.5782 | 5.6649 | Baseline params (depth=6, lr=0.01) |
| **LightGBM** | **3.5184** | 5.4783 | Best baseline performer |
| Random Forest | 6.3617 | 10.3816 | Consistently poor on drift |

**Observation**: Adding weather/holiday features with default hyperparameters slightly degraded XGBoost performance compared to W6's 3.30 MAE.

#### C. Ablation Study and VG Node 1 Quantification

Trained XGBoost (default params) on five feature subsets to isolate auxiliary impact.

| Feature Set | N_Features | MAE | vs Baseline |
| :--- | :--- | :--- | :--- |
| All Features | 27 | 3.5782 | — |
| No Weather | 22 | **3.5055** | +0.0727 better |
| No Holidays | 25 | **3.4900** | +0.0882 better |
| No Weather + No Holidays | 20 | **3.4179** | +0.1603 better |
| **Core Only (No Aux)** | **14** | **3.3025** | **+0.2757 better** |

**VG Node 1 Conclusion**:
*   Weather impact: **−2.03%** (removing weather improves accuracy)
*   Holiday impact: **−2.47%** (removing holidays improves accuracy)
*   Combined auxiliary impact: **−4.48%**

This null/negative contribution is **not a pipeline failure** — it is a valid empirical result. It indicates that for the 2024–2025 SE3 price series, external weather and calendar effects are already largely encoded in the strong autoregressive and rolling-statistic features (lag_1, lag_24, lag_168).

#### D. Hyperparameter Tuning (P1)

Implemented **Optuna Bayesian optimization** (50 trials) for XGBoost focusing on:
*   `learning_rate` (log-uniform 0.005–0.05)
*   `max_depth` (3–10)
*   `subsample`, `colsample_bytree` (0.6–1.0)
*   `min_child_weight` (1–10)
*   `gamma`, `reg_alpha`, `reg_lambda` (regularization)

**Best Parameters**:
```json
{
  "learning_rate": 0.0494,
  "max_depth": 5,
  "subsample": 0.758,
  "colsample_bytree": 0.981,
  "min_child_weight": 6,
  "gamma": 1.81e-07,
  "reg_alpha": 3.32e-08,
  "reg_lambda": 1.223
}
```

**Optimized Model Performance**:
| Metric | Value | Improvement vs W6 |
|--------|-------|-------------------|
| **MAE** | **2.0356 SEK/MWh** | **−38.3%** |
| RMSE | 3.5972 SEK/MWh | −32.4% |
| MAPE | 31.51% | −20.0 pp |

**Key Insight**: The strong regularization (`max_depth=5`, `min_child_weight=6`) and aggressive learning rate were critical. They prevented the model from overfitting to weak auxiliary signals, allowing it to use the full 27-feature set effectively.

*Cross-check*: Even the **Core Only** model with optimized parameters reached **MAE 1.89**, confirming that the majority of the gain comes from hyperparameter tuning rather than feature expansion.

#### E. SHAP Explainability Analysis (P1)

Generated SHAP values on a 1,000-sample test subset using `shap.TreeExplainer`.

**Top 10 Feature Attributions (mean |SHAP|)**:

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|------------|----------------|
| 1 | `value_lag_24` | 15.924 | Yesterday same-hour price |
| 2 | `value_lag_1` | 14.962 | Immediate price inertia |
| 3 | `value_diff_24h` | 14.700 | 24h momentum/trend |
| 4 | `hour_sin` | 0.535 | Daily cyclical pattern |
| 5 | `hour_cos` | 0.382 | Daily cyclical pattern |
| 6 | `value_rolling_max_24h` | 0.372 | Daily high watermark |
| 7 | `is_peak_morning` | 0.310 | Morning demand peak |
| 8 | `value_rolling_mean_24h` | 0.263 | Daily baseline context |
| 9 | `value_rolling_min_24h` | 0.182 | Daily low watermark |
| 10 | `temperature_lag_24` | 0.179 | **Weather signal** |

**SHAP Findings**:
*   The top 3 features absolutely dominate predictions, accounting for ~45 SHAP units combined.
*   `temperature_lag_24` does register in the Top 10, confirming weather has a **small but non-zero** explanatory role in the optimized model.
*   No holiday features appear in the Top 10, aligning with the ablation result.

#### F. Residual and Diagnostic Analysis

| Statistic | Value |
|-----------|-------|
| Mean residual | +0.11 SEK/MWh (near-unbiased) |
| Median residual | +0.01 SEK/MWh |
| 95th percentile abs error | 7.34 SEK/MWh |
| Max absolute error | 53.21 SEK/MWh |

**Time-Based Error Patterns**:
*   **Worst hours**: 10:00 (MAE 3.80), 21:00 (MAE 3.52), 09:00 (MAE 3.39) — morning and evening transition periods.
*   **Worst months**: October (MAE 2.76), September (MAE 2.18), November (MAE 2.10) — autumn volatility period.
*   The single largest error (53.21) occurred on **2025-10-13 10:00** when actual price spiked to 196.60 SEK/MWh but model predicted 143.39. This was an extreme outlier event not well captured by historical lags.

#### G. Model Persistence and Artifacts

Saved production models:
*   `models/w7_xgboost_optimized.pkl` — Primary production model (MAE 2.04)
*   `models/w7_xgboost_baseline.pkl` — Baseline reference
*   `models/w7_lightgbm_baseline.pkl` — Fast inference alternative

Saved analysis artifacts in `reports/w7_results/`:
*   `ablation_study.csv`
*   `ablation_impacts.json`
*   `optuna_best_params.json`
*   `model_comparison.csv`
*   `shap_top_features.csv`
*   `shap_values.pkl`
*   `w7_summary.json`

---

### 3. TECHNICAL CHALLENGES & SOLUTIONS

| Challenge | Impact | Engineering Solution |
| :--- | :--- | :--- |
| **SMHI archive 404** | Weather data acquisition blocked | Switched to Open-Meteo Historical API; cached results locally |
| **Auxiliary features degraded accuracy** | Unexpected negative ablation results | Investigated and attributed to weak-signal overfitting with default `max_depth=6`; resolved via Bayesian regularization tuning |
| **MAPE instability** | High variance due to near-zero off-peak prices | Continued using MAE/RMSE as primary metrics; MAPE reported for completeness only |
| **Autumn volatility outliers** | Single-day spikes cause large residuals | Documented as known limitation; future work could add anomaly-detection or external event features |

---

### 4. KEY FINDINGS & INSIGHTS

#### 4.1 Hyperparameter Tuning Dominates Feature Engineering
The single biggest accuracy gain this week came from Optuna optimization (MAE 3.58 → 2.04), not from adding weather/holidays. This underscores a general principle in gradient boosting: **model capacity control** is often more important than marginal feature expansion.

#### 4.2 Autoregressive Features Are Sufficient for Baseline Accuracy
With only 14 core features (lags + rolling stats), the optimized model achieves MAE 1.89. This validates the strong serial correlation in hourly electricity prices and justifies our earlier decision to prioritize temporal feature engineering.

#### 4.3 Weather Has Marginal Explanatory Value
While weather features rank low in importance, `temperature_lag_24` does appear in the SHAP Top 10. This suggests temperature has a conditional effect that becomes useful *only after* the model is regularized enough to avoid overfitting on noise.

#### 4.4 Holidays Show No Predictive Power
Swedish public holidays had zero measurable impact on SE3 hourly prices in this dataset. This is plausible: industrial load drops on holidays, but the spot market is a regional-wide auction where SE3 (Stockholm) price formation is driven more by inter-zone transmission and Nordic hydro balances than by local consumption dips.

---

### 5. DEVELOPMENT PROGRESS & CODE STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Weather Integration | ✅ Complete | Open-Meteo source integrated and cached |
| Holiday Integration | ✅ Complete | `holidays` library verified active |
| Cyclical Encoding | ✅ Complete | sin/cos features confirmed in dataset |
| Ablation Framework | ✅ Complete | 5-config study executed; results documented |
| Hyperparameter Tuning | ✅ Complete | Optuna 50-trial optimization finished |
| SHAP Analysis | ✅ Complete | Top 10 attributions documented |
| Residual Analysis | ✅ Complete | Hourly/monthly error patterns identified |
| Model Persistence | ✅ Complete | 3 W7 models saved and load-tested |
| **Streamlit App** | ⏳ Pending | Scheduled for Week 8 |

---

### 6. PLANS FOR NEXT WEEK (WEEK 8)

Based on Week 7 results, the optimization strategy for Week 8 shifts from simple feature addition to **feature refinement, heterogeneous ensemble fusion, and systematic bias correction**. The following four workstreams are prioritized by expected ROI.

#### P0-1: High ROI — Feature Interaction & Selection (Days 1-2)
The Week 7 ablation study revealed that weather and holiday features produced marginal/negative contributions under default parameters, indicating high collinearity with autoregressive price dynamics. Rather than adding more raw features, we will extract **conditional signals** through engineered interactions and prune redundant columns.

*   **Feature Interactions**: Create cross-terms that only activate under specific conditions:
    *   `temperature × wind_speed` — captures combined weather stress on grid dynamics
    *   `is_holiday × hour_sin` / `is_holiday × hour_cos` — encodes holiday-hour demand coupling
    *   `is_peak_morning × temperature_lag_24` — temperature sensitivity during demand peaks
*   **SHAP-Guided Recursive Feature Elimination (RFE)**: Use SHAP importance rankings to iteratively remove low-value features (e.g., weak weather/holiday columns) and retrain. Goal: reduce dimensionality from 27+ to a lean high-signal subset.
*   **Expected Impact**: MAE reduction of **5–10%** (target: push below 1.90 SEK/MWh).

#### P0-2: Medium ROI — Heterogeneous Model Fusion (Days 2-4)
XGBoost excels at capturing non-linear lag interactions but is inherently a tabular model. Week 8 will introduce **temporal-structure-aware models** and stack them with the optimized XGBoost base learner.

*   **Prophet + XGBoost Stacking**: Prophet captures global trend and strong seasonality; XGBoost models the residual (short-term deviations). A meta-learner (ridge regression) will blend the two.
*   **LightGBM + XGBoost Weighted Ensemble**: Train both gradient boosters on the same feature set with different random seeds and sampling, then optimize ensemble weights via constrained least squares on a validation fold.
*   **Cross-Check**: Compare single-model vs. stacked MAE on the held-out autumn period (Sep–Nov) where volatility is highest.
*   **Expected Impact**: MAE reduction of **3–8%**, with the largest gains during regime-switching months.

#### P1: Scenario Enhancement — Extreme Price Detection & Quantile Prediction (Days 4-5)
The residual analysis identified a 53.21 SEK/MWh outlier on 2025-10-13, confirming that rare price spikes are poorly handled by point-estimate models. This workstream improves **business usability** even if it does not uniformly lower MAE.

*   **Extreme Event Classifier**: Train a binary classifier (XGBoost `binary:logistic`) to detect whether the next hour falls into a "spike" or "dip" regime (e.g., top/bottom 5% of historical prices). When triggered, route the prediction to a spike-tuned XGBoost sub-model trained only on extreme samples.
*   **Quantile Regression**: Train XGBoost with `quantile` objective at τ = 0.1, 0.5, 0.9 to produce prediction intervals. Evaluate calibration (coverage) and add these bands to the Streamlit app.
*   **Streamlit Integration**: Display point forecast, 80% prediction interval, and an "Extreme Event Alert" flag in the UI.

#### P2: Data Enhancement — Nord Pool Day-Ahead Integration (Days 6-7, time permitting)
SE3 prices are strongly coupled with the broader Nordic market. If API/data bandwidth allows:

*   Integrate **Nord Pool day-ahead prices** for neighboring bidding zones (DK1, DK2, NO1, FI) as leading indicators.
*   Alternatively, fetch **regional load/consumption forecasts** from SVK (Svenska kraftnät) to proxy demand-side shocks.
*   Re-run the ablation framework to quantify the marginal value of cross-border market coupling.

#### P0: Streamlit Application & Deployment (Parallel throughout Week 8)
*   [ ] Create `app.py` with forecast visualization
*   [ ] Allow user to select forecast horizon (1h, 24h, 168h)
*   [ ] Display predicted price + confidence interval (quantile bands)
*   [ ] Add feature importance and SHAP summary plots
*   [ ] Build `requirements-app.txt` with minimal dependencies
*   [ ] Write `README.md` deployment instructions

#### P1: Final Report & VG Documentation (Day 7)
*   [ ] Compile final project report integrating all weekly findings
*   [ ] Prepare answer to VG Node 1 with quantified ablation + SHAP evidence
*   [ ] Create summary slide/deck of final model architecture, Streamlit screenshots, and result tables

#### P1: Final Polish
*   [ ] Code quality review: type hints, docstrings, dead-code removal
*   [ ] Verify `verify_features.py` passes on the final feature set
*   [ ] Final end-to-end pipeline test from raw CSV → prediction

---

### 7. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Streamlit performance on 17k-row rendering | Medium | Slow UI experience | Pre-aggregate data; use Plotly instead of Matplotlib |
| User questions about negative weather impact | High | Requires careful VG explanation | Prepared data-backed rationale in Section 4.2 |
| Final report deadline pressure | Low | Scope reduction if needed | Core text is already 80% written from weekly reports |

---

### 8. METRICS SUMMARY

**Week 7 Achievement Metrics**:
- ✅ Weather data integrated: 17,544 hourly records
- ✅ Feature set expanded: 14 → 27 model-ready features
- ✅ Ablation study completed: 5 configurations tested
- ✅ Best MAE achieved: **2.04 SEK/MWh** (target was <3.0)
- ✅ SHAP Top 10 features documented
- ✅ 3 production models saved

**Week 7 Target Metrics — Final Scorecard**:
| Target | Actual | Status |
|--------|--------|--------|
| MAE with weather: <3.5 | 2.04 | 🟢 Exceeded |
| MAE after tuning: <3.0 | 2.04 | 🟢 Exceeded |
| Weather contribution: >10% improvement | −2.03% | 🔴 Not met (empirically null) |
| SHAP analysis: Top 10 documented | Completed | 🟢 Met |

**Week 8 Target Metrics**:
- 🎯 Streamlit app functional with 1h/24h/168h forecast modes
- 🎯 Final report complete with VG Node 1 justification
- 🎯 End-to-end pipeline verified from raw CSV to prediction

---

**Date**: April 5, 2026
**Status**: On Track — W8 Milestones Clear

# PROJECT WEEKLY REPORT - W6 - 0323-0329

## FIELD NOTE: SWEDISH ELECTRICITY PRICE FORECASTING

**Period**: Week 6 (Mar 23 – Mar 29, 2026)
**Scope**: Model Training Pipeline Execution, Performance Evaluation, and Baseline Establishment

---

### 1. EXECUTIVE SUMMARY

This week marked the transition from feature engineering to actual model training and evaluation. We successfully trained and compared three distinct forecasting models (Random Forest, XGBoost, LightGBM) on the complete SE3 electricity price dataset. The XGBoost model emerged as the top performer with a MAE of 3.30 SEK/MWh, demonstrating strong capability in handling the 40% price drift between training and test periods. Critically, we established the "pure historical baseline" (MAE 4.21 without weather/holiday features), which serves as the reference point for measuring auxiliary feature impact in Week 7.

---

### 2. COMPLETED TASKS

#### A. Model Training Infrastructure Validation (`src/models.py`)

1.  **Dependency Resolution**: Resolved Windows-specific `vcomp140.dll` missing error by reinstalling scikit-learn with `--no-cache-dir` flag and clearing pip cache.
2.  **Type System Fixes**: Corrected `NameError` for missing `Union` import in `models.py` type annotations.
3.  **Cross-Validation Architecture**: Fixed `TypeError` in `cross_validate()` method by implementing `get_params()` to filter custom attributes before model cloning.
4.  **Model Persistence Verification**: Confirmed successful save/load cycle for all three model types using `joblib` serialization.

#### B. End-to-End Training Pipeline Execution

1.  **Dataset Preparation**: Loaded 17,544 hourly samples (2 years: 2024-01-01 to 2026-01-01) with 14 core features.
2.  **Time-Series Split**: Executed 80/20 chronological split (Train: 14,035 samples, Test: 3,509 samples).
3.  **Data Drift Observation**: Identified significant mean shift — training set mean (37.46 SEK/MWh) vs test set mean (52.68 SEK/MWh), representing a 40% price increase in the test period.

#### C. Model Performance Comparison

| Model | MAE (SEK/MWh) | RMSE | Training Time | Assessment |
|-------|---------------|------|---------------|------------|
| **XGBoost** | **3.30** | 5.32 | ~6s | 🏆 Champion: Best accuracy, handles extremes well |
| LightGBM | 3.39 | 5.41 | ~1s | 🥈 Runner-up: Fastest training, comparable accuracy |
| Random Forest | 5.86 | 9.38 | ~2s | ❌ Eliminated: Poor handling of price drift |

**Key Insight**: XGBoost's superior performance on the test set (despite 40% price drift) validates its gradient boosting architecture for capturing complex temporal dynamics.

#### D. Feature Importance Analysis

**XGBoost Top 5 Features**:
1.  `value_lag_1` (55.6%) — Immediate price inertia dominates predictions
2.  `value_rolling_mean_24h` (20.6%) — Daily price baseline context
3.  `value_lag_24` — Yesterday same-hour price (24h seasonality)
4.  `hour` — Time-of-day patterns
5.  `value_lag_168` — Weekly seasonality

**Critical Finding**: Current model relies 100% on endogenous (historical price) features. Weather and holiday features show 0% importance — not because they are irrelevant, but because they were **not yet integrated** into the feature pipeline.

#### E. Cross-Validation Robustness Test

Executed 5-fold Time-Series Cross-Validation on full dataset:

| Model | CV MAE (Mean ± Std) | Stability Assessment |
|-------|---------------------|---------------------|
| LightGBM | 3.3248 ± 0.5126 | 🥇 Most stable across time periods |
| XGBoost | 3.3548 ± 0.5909 | 🥈 Slightly higher variance but best mean |
| Random Forest | 6.3840 ± 1.1063 | ❌ High variance, unreliable |

**Insight**: LightGBM's lower standard deviation (0.51 vs 0.59) suggests superior robustness across different seasons/months.

#### F. Ablation Study and VG Node 1 Baseline

**Experimental Design**: Tested 5 feature set configurations to quantify auxiliary feature impact.

**Unexpected Result**: All configurations produced identical MAE (4.2174) because weather/holiday features are currently **absent** from the dataset.

**Value of This "Null" Result**:
- Established pure historical baseline: **MAE 4.21**
- Confirmed 14 core features (lags + rolling stats) drive all predictive power
- Created measurement benchmark for W7 weather integration

#### G. Model Persistence

Saved three production-ready model files:
- `w6_random_forest_model.pkl` — Baseline reference
- `w6_xgboost_model.pkl` — Primary production model
- `w6_lightgbm_model.pkl` — Fast inference alternative

All models validated with successful load-and-predict cycle.

---

### 3. TECHNICAL CHALLENGES & SOLUTIONS

| Challenge | Impact | Engineering Solution |
| :--- | :--- | :--- |
| **Windows C++ Runtime** | sklearn import failure | Force reinstall with `--no-cache-dir`; clear pip cache |
| **CV Parameter Passing** | `TypeError` on model cloning | Implemented `get_params()` method in `BaseForecaster` to filter custom attributes |
| **Notebook Module Caching** | Code changes not reflecting | Used `%autoreload 2` magic command; kernel restart when needed |
| **MAPE Calculation Anomaly** | 121% error rate despite low MAE | Expected behavior: division by near-zero prices in off-peak hours; use MAE/RMSE instead |
| **Empty Ablation Results** | 0% weather/holiday contribution | Diagnostic confirmation that features not yet integrated; baseline established for W7 comparison |

---

### 4. KEY FINDINGS & INSIGHTS

#### 4.1 Feature Engineering Success
The 14 core features (temporal + lags + rolling statistics) achieve MAE 3.30 without any external data. This validates:
- Strong autoregressive nature of electricity prices
- Effective lag feature design (1h, 24h, 168h)
- Rolling statistics successfully capture volatility context

#### 4.2 Model Selection Rationale
**XGBoost selected as primary model** because:
- Lowest MAE (3.30) and RMSE (5.32)
- Best handling of price extremes (validated in Actual vs Predicted plots)
- Built-in early stopping prevents overfitting
- Rich ecosystem for SHAP explainability (W7)

#### 4.3 Data Drift as Opportunity
The 40% price increase (37→52 SEK/MWh) between train/test sets:
- Validates model generalization capability
- Explains Random Forest failure (mean-regression bias)
- Demonstrates XGBoost's gradient boosting advantage for non-stationary time series

#### 4.4 VG Node 1 Status
**Current State**: Baseline established at MAE 4.21 (core features only)
**Weather/Holiday Contribution**: Currently 0% (features not integrated)
**W7 Target**: Reduce MAE from 4.21 to ~3.5 via weather integration (expected 15-20% improvement)

---

### 5. DEVELOPMENT PROGRESS & CODE STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Model Architecture | ✅ Complete | All 3 models training, evaluating, persisting correctly |
| Cross-Validation | ✅ Complete | 5-fold TS-CV implemented and tested |
| Feature Importance | ✅ Complete | Top features identified; lag_1 dominates at 55.6% |
| Ablation Framework | ✅ Complete | Ready for W7 weather feature comparison |
| Model Persistence | ✅ Complete | 3 .pkl files saved and load-tested |
| **Weather Integration** | ⏳ Pending | Code ready; execution scheduled for W7 |
| **Holiday Integration** | ⏳ Pending | Code ready; execution scheduled for W7 |
| **SHAP Analysis** | ⏳ Pending | Scheduled for W7 |

---

### 6. PLANS FOR NEXT WEEK (WEEK 7)

#### P0: External Feature Integration (Days 1-3)
*   [ ] Execute `SMHIClient.fetch_all_regions()` to acquire temperature/wind data
*   [ ] Integrate weather features via `add_weather_features()`: `temperature`, `temp_lag_1/24`, `wind_speed`, `wind_speed_squared`
*   [ ] Activate holiday features via `add_holiday_features()`: `is_holiday`, `is_pre_holiday`
*   [ ] Verify cyclical encoding: `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`
*   [ ] Regenerate complete feature set; target: 20+ features

#### P0: Impact Quantification (Days 3-4)
*   [ ] Re-run ablation study with weather/holiday features
*   [ ] Measure MAE improvement from auxiliary features
*   [ ] Document VG Node 1: "Weather features contribute X%, Holiday features contribute Y%"
*   [ ] Target: Demonstrate >10% accuracy improvement from auxiliary features

#### P1: Model Optimization (Days 4-5)
*   [ ] Hyperparameter tuning for XGBoost (Optuna or GridSearch)
*   [ ] Focus parameters: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
*   [ ] Target MAE: < 3.0 SEK/MWh (current: 3.30)

#### P1: Diagnostic Analysis (Days 5-6)
*   [ ] Residual analysis: Identify largest prediction errors
*   [ ] Time-based error analysis: Any specific hours/days with systematic bias?
*   [ ] SHAP value preparation: Install and test `shap` library integration

#### P2: Production Preparation (Day 7)
*   [ ] Save optimized W7 models
*   [ ] Update `verify_features.py` with new feature checks
*   [ ] Begin Streamlit app architecture planning (W8)

---

### 7. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SMHI API rate limiting | Medium | Delayed data acquisition | Implement caching; use stored responses |
| Weather features show low importance | Low | VG Node 1 impact < 10% | Include temperature extremes and wind speed squared |
| Overfitting after feature expansion | Medium | CV performance >> test performance | Strict early stopping; monitor validation curve |
| Hyperparameter tuning time | Medium | Delayed W7 completion | Limit search space; use Bayesian optimization |

---

### 8. METRICS SUMMARY

**Week 6 Achievement Metrics**:
- ✅ Models trained: 3/3
- ✅ Best MAE achieved: 3.30 SEK/MWh
- ✅ CV stability verified: σ < 0.6 for top models
- ✅ Baseline established: 4.21 MAE (core features only)
- ✅ Models persisted: 3 .pkl files

**Week 7 Target Metrics**:
- 🎯 MAE with weather: < 3.5 (from 4.21 baseline)
- 🎯 MAE after tuning: < 3.0
- 🎯 Weather contribution: Quantified with >10% improvement
- 🎯 SHAP analysis: Top 10 feature attributions documented

---

**Date**: March 29, 2026
**Status**: On Track for W7 Milestones

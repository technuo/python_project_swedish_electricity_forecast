"""
Week 7 Pipeline: External Feature Integration, Impact Quantification,
Hyperparameter Tuning, and SHAP Analysis.

This script:
1. Fetches/caches Open-Meteo weather data (alternative to SMHI archive)
2. Generates complete feature set with cyclical encoding, holidays, and weather
3. Trains and compares models
4. Performs ablation study to quantify auxiliary feature impact
5. Runs Optuna hyperparameter tuning for XGBoost
6. Conducts SHAP analysis
7. Saves optimized models and artifacts
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
import json
import requests

from src.utils import logger, DATA_PATHS, convert_to_swedish_time
from src.data_processor import DataProcessor
from src.models import (
    XGBoostForecaster, LightGBMForecaster, RandomForestForecaster,
    compare_models, create_train_test_split
)

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_AREA = 'SE3'
WEATHER_CACHE = DATA_PATHS['cache'] / 'open_meteo_se3_weather.csv'
RESULTS_DIR = DATA_PATHS['reports'] / 'w7_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_open_meteo_weather() -> pd.DataFrame:
    """Fetch historical weather from Open-Meteo and cache locally."""
    if WEATHER_CACHE.exists():
        logger.info(f"Loading cached weather data from {WEATHER_CACHE}")
        df = pd.read_csv(WEATHER_CACHE, parse_dates=['timestamp'])
        return df

    logger.info("Fetching weather data from Open-Meteo...")
    lat, lon = 59.3293, 18.0686  # Stockholm
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': '2024-01-01',
        'end_date': '2025-12-31',
        'hourly': ['temperature_2m', 'windspeed_10m'],
        'timezone': 'Europe/Stockholm'
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m'],
        'wind_speed': data['hourly']['windspeed_10m']
    })

    df = convert_to_swedish_time(df, 'timestamp')
    WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEATHER_CACHE, index=False)
    logger.info(f"Cached {len(df)} weather records to {WEATHER_CACHE}")
    return df


def build_features() -> pd.DataFrame:
    """Run full feature engineering pipeline with weather integration."""
    processor = DataProcessor(target_area=TARGET_AREA)

    # Base features (time, lags, rolling, holidays)
    df = processor.prepare_features()
    logger.info(f"Base features shape: {df.shape}, columns: {list(df.columns)}")

    # Weather features
    weather_df = fetch_open_meteo_weather()
    df = processor.add_weather_features(df, smhi_df=weather_df)
    logger.info(f"With weather features shape: {df.shape}, columns: {list(df.columns)}")

    # Save
    processor.save_processed_data(df, 'se3_features_w7.parquet')
    return df


def run_ablation(df: pd.DataFrame) -> dict:
    """
    Train XGBoost with different feature subsets and measure impact.
    Returns a dictionary of results.
    """
    from src.models import create_train_test_split

    X_train, y_train, X_test, y_test = create_train_test_split(
        df, target_col='value', train_ratio=0.8, drop_cols=[]
    )

    all_features = list(X_train.columns)
    weather_features = [f for f in all_features if any(x in f.lower() for x in ['temp', 'wind'])]
    holiday_features = [f for f in all_features if 'holiday' in f.lower()]
    cyclical_features = [f for f in all_features if any(x in f for x in ['_sin', '_cos'])]

    logger.info(f"Weather features: {weather_features}")
    logger.info(f"Holiday features: {holiday_features}")
    logger.info(f"Cyclical features: {cyclical_features}")

    feature_sets = {
        'All Features': all_features,
        'No Weather': [f for f in all_features if f not in weather_features],
        'No Holidays': [f for f in all_features if f not in holiday_features],
        'No Weather + No Holidays': [f for f in all_features if f not in weather_features + holiday_features],
        'Core Only (No Aux)': [f for f in all_features if f not in weather_features + holiday_features + cyclical_features]
    }

    ablation_results = []
    for set_name, cols in feature_sets.items():
        model = XGBoostForecaster(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8)
        model.fit(X_train[cols], y_train)
        metrics = model.evaluate(X_test[cols], y_test, verbose=False)
        ablation_results.append({
            'Feature Set': set_name,
            'N_Features': len(cols),
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
        })
        logger.info(f"{set_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(RESULTS_DIR / 'ablation_study.csv', index=False)

    # Compute impacts
    baseline_mae = ablation_df[ablation_df['Feature Set'] == 'All Features']['MAE'].values[0]
    no_weather_mae = ablation_df[ablation_df['Feature Set'] == 'No Weather']['MAE'].values[0]
    no_holidays_mae = ablation_df[ablation_df['Feature Set'] == 'No Holidays']['MAE'].values[0]
    no_aux_mae = ablation_df[ablation_df['Feature Set'] == 'No Weather + No Holidays']['MAE'].values[0]
    core_only_mae = ablation_df[ablation_df['Feature Set'] == 'Core Only (No Aux)']['MAE'].values[0]

    impacts = {
        'baseline_mae': float(baseline_mae),
        'weather_impact_mae': float(no_weather_mae - baseline_mae),
        'weather_impact_pct': float((no_weather_mae - baseline_mae) / baseline_mae * 100),
        'holiday_impact_mae': float(no_holidays_mae - baseline_mae),
        'holiday_impact_pct': float((no_holidays_mae - baseline_mae) / baseline_mae * 100),
        'combined_aux_impact_mae': float(no_aux_mae - baseline_mae),
        'combined_aux_impact_pct': float((no_aux_mae - baseline_mae) / baseline_mae * 100),
        'core_only_mae': float(core_only_mae),
    }

    with open(RESULTS_DIR / 'ablation_impacts.json', 'w') as f:
        json.dump(impacts, f, indent=2)

    logger.info("=" * 50)
    logger.info("ABLATION IMPACT SUMMARY")
    logger.info("=" * 50)
    for k, v in impacts.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return impacts


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """Run Optuna Bayesian optimization for XGBoost."""
    import optuna

    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = XGBoostForecaster(**params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_val)
        mae = np.mean(np.abs(y_val.values - preds))
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_mae = study.best_value

    logger.info(f"Best Optuna trial MAE: {best_mae:.4f}")
    logger.info(f"Best params: {best}")

    with open(RESULTS_DIR / 'optuna_best_params.json', 'w') as f:
        json.dump({**best, 'best_val_mae': best_mae}, f, indent=2)

    return best, best_mae, study


def run_shap_analysis(model, X_test: pd.DataFrame, sample_size: int = 500):
    """Generate SHAP summary and save top feature attributions."""
    import shap

    explainer = shap.TreeExplainer(model.model)
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values
    mean_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    mean_shap.to_csv(RESULTS_DIR / 'shap_top_features.csv', index=False)
    logger.info("Top 10 SHAP features:")
    for _, row in mean_shap.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Save SHAP values for potential plotting in notebook
    joblib.dump({'shap_values': shap_values, 'sample': X_sample}, RESULTS_DIR / 'shap_values.pkl')
    return mean_shap


def main():
    logger.info("=" * 60)
    logger.info("WEEK 7 PIPELINE START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Feature Engineering with Weather
    # ------------------------------------------------------------------
    df = build_features()

    # ------------------------------------------------------------------
    # Step 2: Train-Test Split
    # ------------------------------------------------------------------
    X_train, y_train, X_test, y_test = create_train_test_split(
        df, target_col='value', train_ratio=0.8, drop_cols=[]
    )

    # ------------------------------------------------------------------
    # Step 3: Baseline Model Comparison
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BASELINE MODEL COMPARISON (Full Features)")
    logger.info("=" * 60)

    models = {
        'XGBoost': XGBoostForecaster(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8),
        'LightGBM': LightGBMForecaster(n_estimators=1000, learning_rate=0.01, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8),
        'Random Forest': RandomForestForecaster(n_estimators=100, max_depth=10),
    }

    trained_models = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        mdl.evaluate(X_test, y_test)
        trained_models[name] = mdl

    comparison_df = compare_models(trained_models, X_test, y_test)
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)

    # ------------------------------------------------------------------
    # Step 4: Ablation Study
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("=" * 60)
    impacts = run_ablation(df)

    # ------------------------------------------------------------------
    # Step 5: Hyperparameter Tuning (XGBoost)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("XGBOOST HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    # Use a validation set from the end of training data (~10%)
    val_size = int(len(X_train) * 0.1)
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    best_params, best_mae, study = tune_xgboost(X_tr, y_tr, X_val, y_val, n_trials=50)

    # Train final optimized model
    final_params = {'n_estimators': 2000, **best_params}  # More trees for final model
    xgb_optimized = XGBoostForecaster(name='XGBoost_Optimized', **final_params)
    xgb_optimized.fit(X_train, y_train)
    opt_metrics = xgb_optimized.evaluate(X_test, y_test)

    # ------------------------------------------------------------------
    # Step 6: SHAP Analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SHAP ANALYSIS")
    logger.info("=" * 60)
    shap_df = run_shap_analysis(xgb_optimized, X_test, sample_size=1000)

    # ------------------------------------------------------------------
    # Step 7: Save Models
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SAVING MODELS")
    logger.info("=" * 60)

    xgb_optimized.save('w7_xgboost_optimized.pkl')
    trained_models['XGBoost'].save('w7_xgboost_baseline.pkl')
    trained_models['LightGBM'].save('w7_lightgbm_baseline.pkl')

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    summary = {
        'baseline_xgboost_mae': float(comparison_df[comparison_df['Model'] == 'XGBoost']['MAE'].values[0]),
        'optimized_xgboost_mae': float(opt_metrics['mae']),
        'weather_impact_pct': impacts['weather_impact_pct'],
        'holiday_impact_pct': impacts['holiday_impact_pct'],
        'best_optuna_params': best_params,
    }

    with open(RESULTS_DIR / 'w7_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("WEEK 7 PIPELINE COMPLETE")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


if __name__ == '__main__':
    main()

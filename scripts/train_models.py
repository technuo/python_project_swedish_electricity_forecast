#!/usr/bin/env python3
"""
scripts/train_models.py
=======================
End-to-end model training pipeline for Week 6.

Runs training for all three models, generates evaluation metrics,
performs ablation study, and saves models.

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.features import DataProcessor
from src.models import (
    RandomForestForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
    compare_models,
    create_train_test_split
)
from src.utils import logger, load_data, DATA_PATHS


def main():
    logger.info("=" * 70)
    logger.info("W6 MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load Data
    logger.info("\n[1/5] Loading features...")
    try:
        df = load_data('processed', 'se3_features_v1.parquet')
        logger.info(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
    except FileNotFoundError:
        logger.info("Features not found, generating...")
        processor = DataProcessor(target_area='SE3')
        df = processor.prepare_features()
        processor.save_processed_data(df, 'se3_features_v1.parquet')

    # 2. Train/Test Split
    logger.info("\n[2/5] Creating train/test split...")
    X_train, y_train, X_test, y_test = create_train_test_split(
        df, target_col='value', train_ratio=0.8
    )

    # 3. Train Models
    logger.info("\n[3/5] Training models...")
    models = {
        'Random Forest': RandomForestForecaster(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'XGBoost': XGBoostForecaster(
            n_estimators=1000, learning_rate=0.01, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        'LightGBM': LightGBMForecaster(
            n_estimators=1000, learning_rate=0.01, max_depth=6,
            num_leaves=31, subsample=0.8, random_state=42
        )
    }

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train, X_test, y_test)

    # 4. Evaluation
    logger.info("\n" + "=" * 70)
    logger.info("[4/5] MODEL EVALUATION")
    logger.info("=" * 70)
    comparison = compare_models(models, X_test, y_test)

    # 5. Feature Importance (XGBoost)
    logger.info("\n" + "=" * 70)
    logger.info("[5/5] FEATURE IMPORTANCE (XGBoost Top 15)")
    logger.info("=" * 70)
    xgb_model = models['XGBoost']
    importance = xgb_model.feature_importance()
    for rank, (feat, imp) in enumerate(importance.head(15).items(), 1):
        marker = ""
        if any(x in feat.lower() for x in ['wind', 'temp']):
            marker = " [WEATHER]"
        elif 'holiday' in feat.lower():
            marker = " [HOLIDAY]"
        elif any(x in feat for x in ['_sin', '_cos']):
            marker = " [CYCLICAL]"
        logger.info(f"{rank:2d}. {feat:30s} {imp:.4f}{marker}")

    # 6. Ablation Study
    logger.info("\n" + "=" * 70)
    logger.info("VG NODE 1: AUXILIARY FEATURE IMPACT")
    logger.info("=" * 70)

    all_features = list(X_train.columns)
    weather_features = [f for f in all_features
                        if any(x in f.lower() for x in ['temp', 'wind'])]
    holiday_features = [f for f in all_features if 'holiday' in f.lower()]

    feature_sets = {
        'All Features': all_features,
        'No Weather': [f for f in all_features if f not in weather_features],
        'No Holidays': [f for f in all_features if f not in holiday_features],
        'No Weather + No Holidays': [f for f in all_features
                                      if f not in weather_features + holiday_features]
    }

    baseline_mae = None
    for set_name, features in feature_sets.items():
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]

        model = XGBoostForecaster(n_estimators=500, learning_rate=0.01, random_state=42)
        model.fit(X_train_sub, y_train)
        metrics = model.evaluate(X_test_sub, y_test, verbose=False)

        if set_name == 'All Features':
            baseline_mae = metrics['mae']

        logger.info(f"{set_name:25s}: MAE = {metrics['mae']:.4f}")

    # 7. Save Models
    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODELS")
    logger.info("=" * 70)
    for name, model in models.items():
        filename = f"w6_{name.lower().replace(' ', '_')}_model.pkl"
        path = model.save(filename)
        logger.info(f"Saved: {path}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)

    return models, comparison


if __name__ == "__main__":
    main()

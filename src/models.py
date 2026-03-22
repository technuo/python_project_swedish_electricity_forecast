"""
src/models.py
=============
Electricity price forecasting models with unified interface.

Implements:
- Random Forest (baseline)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting, faster training)

Design: Abstract base class ensures consistent interface across all models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Add project root to path for utils
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import logger, DATA_PATHS


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    Provides unified interface for fit, predict, evaluate, save, and load.
    """

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.target_col: str = 'value'
        self.cv_scores: Dict[str, List[float]] = {'mae': [], 'rmse': [], 'mape': []}

    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the underlying model instance."""
        pass

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaseForecaster':
        """
        Fit the model to training data.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation target

        Returns:
            self for method chaining
        """
        logger.info(f"[{self.name}] Starting training...")
        self.feature_names = list(X_train.columns)
        self.target_col = y_train.name if hasattr(y_train, 'name') else 'value'

        self.model = self._create_model()

        # Handle validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # Fit with appropriate arguments
        fit_kwargs = {}
        if hasattr(self.model, 'fit'):
            # Check if model supports eval_set (XGB, LGBM)
            import inspect
            fit_params = inspect.signature(self.model.fit).parameters
            if 'eval_set' in fit_params and eval_set is not None:
                fit_kwargs['eval_set'] = eval_set
                if 'early_stopping_rounds' in fit_params:
                    fit_kwargs['early_stopping_rounds'] = 50
                    fit_kwargs['verbose'] = False

            self.model.fit(X_train, y_train, **fit_kwargs)

        self.is_fitted = True
        logger.info(f"[{self.name}] Training complete.")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} has not been fitted yet.")

        # Ensure columns match training
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        return self.model.predict(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Whether to log results

        Returns:
            Dictionary with MAE, RMSE, MAPE metrics
        """
        y_pred = self.predict(X_test)
        y_true = y_test.values if hasattr(y_test, 'values') else y_test

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)

        metrics = {'mae': mae, 'rmse': rmse, 'mape': mape}

        if verbose:
            logger.info(f"[{self.name}] Evaluation Results:")
            logger.info(f"  MAE:  {mae:.4f} SEK/MWh")
            logger.info(f"  RMSE: {rmse:.4f} SEK/MWh")
            logger.info(f"  MAPE: {mape:.2f}%")

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this forecaster."""
        params = {'random_state': self.random_state}
        # Add attributes from subclasses (n_estimators, etc.)
        for attr in ['n_estimators', 'max_depth', 'min_samples_split', 
                    'learning_rate', 'subsample', 'colsample_bytree', 
                    'num_leaves', 'early_stopping_rounds', 'extra_params']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if attr == 'extra_params' and isinstance(val, dict):
                    params.update(val)
                else:
                    params[attr] = val
        return params

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Time-series cross-validation.

        Args:
            X: Full feature set
            y: Full target
            n_splits: Number of CV folds

        Returns:
            Dictionary with mean/std of CV metrics
        """
        logger.info(f"[{self.name}] Performing {n_splits}-fold time-series CV...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'mae': [], 'rmse': [], 'mape': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Create fresh model for each fold with same parameters
            params = self.get_params()
            params['name'] = f"{self.name}_fold{fold}"
            model = self.__class__(**params)
            
            model.fit(X_train_fold, y_train_fold)
            metrics = model.evaluate(X_val_fold, y_val_fold, verbose=False)

            cv_scores['mae'].append(metrics['mae'])
            cv_scores['rmse'].append(metrics['rmse'])
            cv_scores['mape'].append(metrics['mape'])

        # Store and report
        self.cv_scores = cv_scores
        summary = {
            'cv_mae_mean': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae']),
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_mape_mean': np.mean(cv_scores['mape']),
            'cv_mape_std': np.std(cv_scores['mape']),
        }

        logger.info(f"[{self.name}] CV Results (mean ± std):")
        logger.info(f"  MAE:  {summary['cv_mae_mean']:.4f} ± {summary['cv_mae_std']:.4f}")
        logger.info(f"  RMSE: {summary['cv_rmse_mean']:.4f} ± {summary['cv_rmse_std']:.4f}")
        logger.info(f"  MAPE: {summary['cv_mape_mean']:.2f}% ± {summary['cv_mape_std']:.2f}%")

        return summary

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save model to disk.

        Args:
            filename: Optional filename, defaults to {name}_model.pkl

        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_model.pkl"

        save_path = DATA_PATHS['models'] / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'target_col': self.target_col,
            'is_fitted': self.is_fitted,
            'cv_scores': self.cv_scores,
        }, save_path)

        logger.info(f"[{self.name}] Model saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseForecaster':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded forecaster instance
        """
        data = joblib.load(filepath)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.model = data['model']
        instance.name = data['name']
        instance.feature_names = data['feature_names']
        instance.target_col = data['target_col']
        instance.is_fitted = data['is_fitted']
        instance.cv_scores = data['cv_scores']
        instance.random_state = 42  # Default

        logger.info(f"[{instance.name}] Model loaded from {filepath}")
        return instance

    def feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.

        Returns:
            Series mapping feature names to importance scores, or None
        """
        if not self.is_fitted or self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        return None


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest baseline model.

    Pros: Robust, no scaling needed, handles non-linearities well
    Cons: Can overfit, slower than gradient boosting for large datasets
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        random_state: int = 42,
        name: str = "RandomForest",
        **kwargs
    ):
        super().__init__(name, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.extra_params = kwargs

    def _create_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,
            **self.extra_params
        )


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost gradient boosting model.

    Pros: Fast, accurate, built-in regularization, feature importance
    Cons: Requires careful hyperparameter tuning
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.01,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        name: str = "XGBoost",
        **kwargs
    ):
        super().__init__(name, random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs

    def _create_model(self):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                **self.extra_params
            )
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM gradient boosting model.

    Pros: Very fast training, memory efficient, handles large datasets well
    Cons: May need different hyperparameters than XGBoost
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.01,
        max_depth: int = 6,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        name: str = "LightGBM",
        **kwargs
    ):
        super().__init__(name, random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs

    def _create_model(self):
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1,  # Suppress training output
                **self.extra_params
            )
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")


def compare_models(
    models: Dict[str, BaseForecaster],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models on test set.

    Args:
        models: Dictionary mapping model names to forecaster instances
        X_test: Test features
        y_test: Test targets

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test, verbose=False)
        results.append({
            'Model': name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'MAPE (%)': metrics['mape'],
        })

    df = pd.DataFrame(results)

    # Highlight best (lowest) in each metric
    best_mae = df['MAE'].min()
    best_rmse = df['RMSE'].min()
    best_mape = df['MAPE (%)'].min()

    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 50)
    for _, row in df.iterrows():
        mae_str = f"{row['MAE']:.4f}" + (" *" if row['MAE'] == best_mae else "")
        rmse_str = f"{row['RMSE']:.4f}" + (" *" if row['RMSE'] == best_rmse else "")
        mape_str = f"{row['MAPE (%)']:.2f}" + (" *" if row['MAPE (%)'] == best_mape else "")
        logger.info(f"{row['Model']:<15} MAE: {mae_str:<12} RMSE: {rmse_str:<12} MAPE: {mape_str}")
    logger.info("* = best in category")
    logger.info("=" * 50)

    return df


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'value',
    train_ratio: float = 0.8,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function to create train/test split.

    Args:
        df: Full dataset with features and target
        target_col: Name of target column
        train_ratio: Proportion for training
        drop_cols: Additional columns to drop from features

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Sort by time if timestamp column exists
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Determine columns to drop
    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    if 'timestamp' in df.columns:
        cols_to_drop.append('timestamp')

    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_df[target_col]

    logger.info(f"Train split: {len(X_train)} samples")
    logger.info(f"Test split:  {len(X_test)} samples")
    logger.info(f"Features:    {len(X_train.columns)} columns")

    return X_train, y_train, X_test, y_test


__all__ = [
    'BaseForecaster',
    'RandomForestForecaster',
    'XGBoostForecaster',
    'LightGBMForecaster',
    'compare_models',
    'create_train_test_split',
    'calculate_mape',
]

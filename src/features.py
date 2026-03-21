"""
src/features.py
===============
Feature engineering utilities and re-exports for backward compatibility.

This module provides:
1. Re-export of DataProcessor (canonical implementation in data_processor.py)
2. Standalone feature transformation utilities for flexible use

Design Principles:
- Modularity: Each transformation is an independent function
- Reproducibility: Consistent handling of timezones and missing values
- Pipeline-ready: Designed to support scikit-learn or custom training loops
"""

import pandas as pd
import numpy as np
from typing import List, Optional

# Re-export DataProcessor for backward compatibility
# The canonical implementation lives in src.data_processor
from src.data_processor import DataProcessor


def add_cyclical_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    Generic cyclical encoder for any periodic column.

    Converts discrete periodic features (e.g., hour 0-23) into
    continuous sine/cosine pairs that preserve cyclic relationships.

    Args:
        df: Input DataFrame
        col: Column name to encode (e.g., 'hour', 'month', 'day_of_week')
        period: Period length (e.g., 24 for hours, 12 for months, 7 for days)

    Returns:
        DataFrame with added {col}_sin and {col}_cos columns

    Example:
        >>> df = pd.DataFrame({'hour': [0, 6, 12, 18, 23]})
        >>> add_cyclical_features(df, 'hour', 24)
        # hour_sin values: 0.0, 0.5, 0.0, -0.5, ~-0.26
        # hour_cos values: 1.0, ~0.87, -1.0, ~0.87, ~0.97
    """
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df


def add_holiday_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Adds Swedish public holiday indicators.

    Standalone version for use outside DataProcessor class.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of the datetime column

    Returns:
        DataFrame with 'is_holiday' and 'is_pre_holiday' columns
    """
    import holidays as hols

    swedish_holidays = hols.Sweden()
    ts = pd.to_datetime(df[timestamp_col])

    df['is_holiday'] = ts.dt.date.map(lambda d: int(d in swedish_holidays))
    df['is_pre_holiday'] = ts.apply(
        lambda x: int((pd.Timestamp(x) + pd.Timedelta(days=1)).date() in swedish_holidays)
    )

    return df


def calculate_rolling_features(
    df: pd.DataFrame,
    value_col: str = 'value',
    window: int = 24
) -> pd.DataFrame:
    """
    Calculate rolling window statistics for any value column.

    Args:
        df: Input DataFrame
        value_col: Column name to calculate statistics for
        window: Rolling window size in hours

    Returns:
        DataFrame with added columns:
        - {value_col}_rolling_mean_{window}h
        - {value_col}_rolling_max_{window}h
        - {value_col}_rolling_min_{window}h
        - {value_col}_rolling_std_{window}h
    """
    mean_col = f'{value_col}_rolling_mean_{window}h'
    max_col = f'{value_col}_rolling_max_{window}h'
    min_col = f'{value_col}_rolling_min_{window}h'
    std_col = f'{value_col}_rolling_std_{window}h'

    df[mean_col] = df[value_col].rolling(window=window).mean()
    df[max_col] = df[value_col].rolling(window=window).max()
    df[min_col] = df[value_col].rolling(window=window).min()
    df[std_col] = df[value_col].rolling(window=window).std()

    return df


def add_volatility_features(df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    """
    Adds price volatility indicators.

    Args:
        df: Input DataFrame with value column
        value_col: Column name (typically 'value' for price)

    Returns:
        DataFrame with added columns:
        - value_diff_1h: Change from previous hour
        - value_diff_24h: Change from same time yesterday
        - value_pct_change_1h: Percentage change from previous hour
    """
    df['value_diff_1h'] = df[value_col].diff(1)
    df['value_diff_24h'] = df[value_col].diff(24)
    df['value_pct_change_1h'] = df[value_col].pct_change(1) * 100

    return df


__all__ = [
    'DataProcessor',  # Re-export
    'add_cyclical_features',
    'add_holiday_features',
    'calculate_rolling_features',
    'add_volatility_features',
]

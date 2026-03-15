"""
src/data_processor.py
=====================
Core logic for electricity data processing and feature engineering.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Add project root to path for internal imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    logger, 
    convert_to_swedish_time, 
    create_lag_features,
    get_data_path,
    DATA_PATHS
)

class DataProcessor:
    """
    High-level class to transform raw electricity data into model-ready features.
    """
    
    def __init__(self, target_area: str = 'SE3', frequency: str = 'H'):
        self.target_area = target_area
        self.frequency = frequency
        self.logger = logger
        self.df = None

    def load_and_consolidate(self, data_type: str = 'price') -> pd.DataFrame:
        """
        Loads all relevant CSV files from data/raw and consolidates them.
        """
        self.logger.info(f"Consolidating raw {data_type} data for {self.target_area}...")
        
        raw_dir = DATA_PATHS['raw']
        pattern = f"{self.target_area}_PRICES_*.csv" if data_type == 'price' else "*.csv"
        files = list(raw_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No files found for {data_type} and area {self.target_area}")
            return pd.DataFrame()
            
        dfs = []
        for f in files:
            temp_df = pd.read_csv(f)
            
            # Identify columns
            mtu_col = [c for c in temp_df.columns if 'MTU' in c]
            if mtu_col:
                col = mtu_col[0]
                temp_df['timestamp'] = temp_df[col].str.split(' - ').str[0]
                temp_df['timestamp'] = temp_df['timestamp'].str.replace(r'\(CE[S]?T\)', '', regex=True).str.strip()
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], dayfirst=True, format='mixed')
            
            val_col = [c for c in temp_df.columns if 'Price' in c or 'Load' in c]
            if val_col:
                temp_df = temp_df.rename(columns={val_col[0]: 'value'})
                
            dfs.append(temp_df[['timestamp', 'value']])
            
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        return self.df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs deep data cleaning including timezone unification, reindexing, 
        interpolation and outlier capping.
        """
        self.logger.info("Starting deep data cleaning...")
        
        # 1. Unify Timezone
        df = convert_to_swedish_time(df, 'timestamp')
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        df = df.set_index('timestamp')
        
        # 2. Ensure continuous hourly index
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
        df = df.reindex(full_range)
        
        # 3. Handle Missing Values
        df['value'] = df['value'].interpolate(method='linear')
        
        # 4. Outlier detection and capping (3.0 * IQR)
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)
        
        self.logger.info(f"Cleaning complete. Data range: {df.index.min()} to {df.index.max()}")
        return df.reset_index().rename(columns={'index': 'timestamp'})

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal characteristics to capture seasonality.
        """
        self.logger.info("Adding temporal features...")
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['is_peak_morning'] = df['hour'].between(7, 10).astype(int)
        df['is_peak_evening'] = df['hour'].between(17, 20).astype(int)
        
        return df

    def add_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Creates historical look-back features.
        Target: Capture inertia (e.g., if it rose an hour ago, it might keep rising).
        """
        self.logger.info(f"Engineering lag features: {lags}")
        # We use create_lag_features from utils
        return create_lag_features(df, 'value', lags)

    def add_weather_features(self, df: pd.DataFrame, smhi_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Integrates SMHI weather data (temperature, wind speed) into the dataset.
        1. Timezone Alignment: Ensure Europe/Stockholm and align with df index.
        2. Missing values: Linear interpolation for temperature and wind speed.
        3. Feature engineering: Lags for temperature, squared wind speed.
        """
        self.logger.info("Adding external weather features (temperature, wind speed)...")
        
        if smhi_df is None or smhi_df.empty:
            self.logger.warning("No SMHI data provided. Skipping weather features.")
            return df
            
        # 1. Timezone Alignment and indexing
        if 'timestamp' in smhi_df.columns:
            smhi_df = convert_to_swedish_time(smhi_df, 'timestamp')
            smhi_df = smhi_df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
        elif isinstance(smhi_df.index, pd.DatetimeIndex):
            smhi_df = smhi_df[~smhi_df.index.duplicated(keep='first')]
        else:
            self.logger.warning("SMHI dataframe missing timestamp. Skipping weather features.")
            return df

        cols_to_join = []
        if 'temperature' in smhi_df.columns: cols_to_join.append('temperature')
        if 'wind_speed' in smhi_df.columns: cols_to_join.append('wind_speed')
        
        if not cols_to_join:
            self.logger.warning("No temperature or wind_speed columns in SMHI data. Skipping.")
            return df

        df = df.set_index('timestamp')
        df = df.join(smhi_df[cols_to_join], how='left')
        
        # 2. Handle missing weather data
        self.logger.info("Interpolating missing weather data...")
        for col in cols_to_join:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].bfill().ffill()

        df = df.reset_index()
        
        # 3. Feature Transformation
        self.logger.info("Applying weather feature transformations...")
        
        if 'temperature' in df.columns:
            df = create_lag_features(df, 'temperature', [1, 24])
            
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
            
        return df

    def add_rolling_features(self, df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
        """
        Calculates rolling window statistics (mean, max, min, std) for the past N hours.
        Target: Capture market volatility and trends.
        """
        self.logger.info(f"Adding {window}h rolling window statistics (mean, max, min, std)...")
        
        # Define columns
        mean_col = f'value_rolling_mean_{window}h'
        max_col = f'value_rolling_max_{window}h'
        min_col = f'value_rolling_min_{window}h'
        std_col = f'value_rolling_std_{window}h'
        
        # Calculate statistics
        df[mean_col] = df['value'].rolling(window=window).mean()
        df[max_col] = df['value'].rolling(window=window).max()
        df[min_col] = df['value'].rolling(window=window).min()
        df[std_col] = df['value'].rolling(window=window).std()
        
        # Momentum: Change compared to 24h ago
        df['value_diff_24h'] = df['value'].diff(24)
        
        # Fill NaNs created by rolling windows
        df = df.bfill()
        
        return df

    def filter_features(self, df: pd.DataFrame, target_col: str = 'value') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature selection using correlation analysis.
        """
        self.logger.info("Performing feature correlation screening...")
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        return df, correlations

    def prepare_features(self) -> pd.DataFrame:
        """
        Executes the entire feature engineering pipeline.
        """
        self.logger.info("Building feature engineering pipeline...")
        
        # 1. Load & Consolidate
        df = self.load_and_consolidate(data_type='price')
        if df.empty:
            self.logger.error("No data loaded. Pipeline aborted.")
            return df
            
        # 2. Deep Clean
        df = self.clean_data(df)
        
        # 3. Add Time Features
        df = self.add_time_features(df)
        
        # 4. Add Lags (1h, 24h, 168h)
        df = self.add_lag_features(df, [1, 24, 168])
        
        # 5. Add Rolling Statistics (24h)
        df = self.add_rolling_features(df, window=24)
        
        # Note: weather features should be called directly when SMHI data is available,
        # e.g.: df = processor.add_weather_features(df, smhi_df=weather_data)
        
        self.df = df
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Saves the final feature set to data/processed."""
        from src.utils import save_data
        save_data(df, 'processed', filename)

if __name__ == "__main__":
    processor = DataProcessor(target_area='SE3')
    print("DataProcessor initialized and ready for feature engineering.")

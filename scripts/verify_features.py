import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import DataProcessor
from src.utils import logger

def verify_pipeline():
    logger.info("--- Starting Feature Engineering Verification ---")
    
    # Initialize processor for SE3
    processor = DataProcessor(target_area='SE3')
    
    try:
        # 1. Test Base Pipeline
        logger.info("Running prepare_features()...")
        df = processor.prepare_features()

        if df.empty:
            logger.error("Pipeline returned empty DataFrame. Check if raw data exists.")
            return

        # 1b. Integrate Weather Features
        logger.info("Integrating weather features...")
        from src.utils import DATA_PATHS, convert_to_swedish_time
        import requests

        weather_cache = DATA_PATHS['cache'] / 'open_meteo_se3_weather.csv'
        if weather_cache.exists():
            weather_df = pd.read_csv(weather_cache, parse_dates=['timestamp'])
        else:
            lat, lon = 59.3293, 18.0686
            url = 'https://archive-api.open-meteo.com/v1/archive'
            params = {
                'latitude': lat, 'longitude': lon,
                'start_date': '2024-01-01', 'end_date': '2025-12-31',
                'hourly': ['temperature_2m', 'windspeed_10m'],
                'timezone': 'Europe/Stockholm'
            }
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            weather_df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'wind_speed': data['hourly']['windspeed_10m']
            })
            weather_df = convert_to_swedish_time(weather_df, 'timestamp')
            weather_cache.parent.mkdir(parents=True, exist_ok=True)
            weather_df.to_csv(weather_cache, index=False)

        df = processor.add_weather_features(df, smhi_df=weather_df)
        logger.info(f"Generated DataFrame shape: {df.shape}")
        
        # 2. Check for NaNs
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Columns with NaNs: {nan_cols}")
        else:
            logger.info("Verification Passed: No missing values in final feature set.")
            
        # 3. Check for Outliers
        max_val = df['value'].max()
        logger.info(f"Maximum price in processed data: {max_val:.2f}")
        
        # 4. Check temporal features
        expected_cols = ['hour', 'day_of_week', 'is_peak_morning', 'value_rolling_mean_24h']
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns: {missing_cols}")
        else:
            logger.info("Verification Passed: All expected features are present.")

        # 4b. Check W5 new features
        new_w5_features = ['hour_sin', 'hour_cos', 'is_holiday', 'is_pre_holiday']
        missing_w5 = [c for c in new_w5_features if c not in df.columns]
        if missing_w5:
            logger.error(f"Missing W5 features: {missing_w5}")
        else:
            logger.info("Verification Passed: All W5 features present (sin/cos, holidays).")

        # 4c. Check W7 weather features
        weather_features = ['temperature', 'wind_speed', 'temperature_lag_1', 'temperature_lag_24', 'wind_speed_squared']
        missing_weather = [c for c in weather_features if c not in df.columns]
        if missing_weather:
            logger.error(f"Missing W7 weather features: {missing_weather}")
        else:
            logger.info("Verification Passed: All W7 weather features present.")

        # 5. Feature Filtering
        logger.info("Testing feature filtering...")
        _, corr = processor.filter_features(df)
        logger.info("Top 5 correlated features with 'value':")
        print(corr.head(6)) # 1st is 'value' itself

        # 6. Save Data
        processor.save_processed_data(df, 'se3_features_w7.parquet')
        logger.info("--- Verification Complete ---\n")
        
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()

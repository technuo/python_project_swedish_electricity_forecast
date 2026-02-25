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
        # 1. Test Pipeline
        logger.info("Running prepare_features()...")
        df = processor.prepare_features()
        
        if df.empty:
            logger.error("Pipeline returned empty DataFrame. Check if raw data exists.")
            return
            
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
            
        # 5. Feature Filtering
        logger.info("Testing feature filtering...")
        _, corr = processor.filter_features(df)
        logger.info("Top 5 correlated features with 'value':")
        print(corr.head(6)) # 1st is 'value' itself
        
        # 6. Save Data
        processor.save_processed_data(df, 'se3_features_v1.parquet')
        logger.info("--- Verification Complete ---\n")
        
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()

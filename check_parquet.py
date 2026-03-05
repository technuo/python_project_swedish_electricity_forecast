import pandas as pd
import pathlib

# Hardcoded absolute path
file_path = r'd:\2026\python_project_swedish_electricity_forecast\data\processed\se3_features_v1.parquet'

if not pathlib.Path(file_path).exists():
    print(f"ERROR: File {file_path} does not exist!")
else:
    df = pd.read_parquet(file_path)
    print("Columns found in se3_features_v1.parquet:")
    for col in df.columns:
        print(f" - {col}")
    
    # Check specifically for requested features
    features_to_check = ['value_lag_1', 'value_lag_24', 'value_lag_168', 
                         'value_rolling_mean_24h', 'value_rolling_max_24h', 
                         'value_rolling_min_24h', 'value_rolling_std_24h']
    
    print("\nFeature check:")
    for feat in features_to_check:
        status = "✅ Found" if feat in df.columns else "❌ Missing"
        print(f" {feat}: {status}")

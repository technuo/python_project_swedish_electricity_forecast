Walkthrough: Core Feature Engineering
Completed the core feature engineering task for the Swedish Electricity Forecast project.

Changes Made
1. Data Cleaning (
src/features.py
)
Timezone Handling: Automatically localizes UTC/naive data to Europe/Stockholm using src.utils.convert_to_swedish_time.
Missing Values: Implemented linear interpolation to fill gaps in the hourly sequence.
Outliers: Applied a 1.5 * IQR (using 3.0 factor for "extreme") capping strategy to stabilize prices without data loss.
2. Core Feature Development (
add_power_features
)
Temporal Features: Hour, day of week, month, and peak hour indicators (morning/evening).
Lag Features: 24h, 48h, and 168h (1 week) lags to capture historical patterns.
Rolling Features: 24h and 7d moving averages.
Volatility: 24h rolling standard deviation to track market nervousness.
Momentum: Difference between current price and 24h ago.
3. Verification & Analysis
Verification Script: Created 
scripts/verify_features.py
 which validates the pipeline output, ensures no NaNs, and checks feature correlations.
EDA Notebook: Created 
notebooks/03_Feature_Engineering_Analysis.ipynb
 for visual verification of feature distributions and target relationships.
Verification Results
The verification script confirmed:

Shape: Correctly expanded feature set.
Data Integrity: 0 NaNs after interpolation and backfilling.
Correlation: value_rolling_mean_24h showed expected strong correlation (~0.75) with the target.
Example Correlation Ranking:
value (Target): 1.00
value_rolling_mean_24h: 0.75
value_lag_24: 0.71 ... etc.
Next Steps
Implement external data integration (SMHI weather data).
Begin model training selection (XGBoost/LightGBM).
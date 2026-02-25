Implementation Plan: Core Feature Engineering
Implement deep data cleaning and core feature development for the Swedish Electricity Forecast project.

Proposed Changes
[src]
[MODIFY] 
features.py
Implement 
clean_data()
:
Handle missing values using linear interpolation for numeric columns.
Implement outlier detection and capping for electricity prices (using IQR or Z-score).
Ensure timezone consistency (localized to Stockholm time, then converted to naive).
Implement add_power_features():
Add moving averages (24h, 7d).
Add volatility features (24h rolling standard deviation).
Add momentum features (price change vs 24h ago).
Add peak hour indicators (morning/evening peaks).
Implement filter_features():
Calculate correlation with target.
Identify high collinearity features (VIF analysis).
Rank features by importance.
[MODIFY] 
utils.py
Ensure all helper functions (like 
create_lag_features
) are robust for the new features.
[notebooks]
[NEW] 
03_Feature_Engineering_Analysis.ipynb
Interactive exploration of the newly created features.
Visualization of feature distributions and correlations.
Heatmap of feature relationships.
Verification Plan
Automated Tests
Create a test script scripts/verify_features.py to:
Load a sample of data.
Run the cleaning and feature engineering pipeline.
Assert no NaN values remain in core features.
Check the shape and types of the output DataFrame.
Verify correlation calculations run without error.
Command:
powershell
python -m scripts.verify_features
Manual Verification
Review the visualizations in the new EDA notebook to ensure features look sensible (e.g., peak indicators align with expected hours).
Inspect the saved feature importance ranking for validity.
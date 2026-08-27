"""
Run this once (from your modelling notebook or as a standalone script)
to produce the three artifacts app.py needs:

  - sales_forecast_xgb_model.pkl
  - feature_cols.json
  - historical_sales_for_inference.csv

Assumes `ts_m` (the full engineered feature DataFrame, with a 'Date' and
'Sales' column plus all engineered feature columns) is already built,
exactly as in the feature engineering notebook.
"""

import json

import joblib
from xgboost import XGBRegressor

# Must match the engineered feature columns used during training/evaluation.
feature_cols_clean = [
    'Year', 'Month', 'Quarter', 'Month_sin', 'Month_cos',
    'DayOfWeek', 'IsWeekend',
    'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3', 'Sales_lag_12',
    'Rolling_mean_3', 'Rolling_std_3', 'Rolling_mean_6',
]

# --- Retrain on the full dataset (train + test) for deployment ---
final_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
)
final_model.fit(ts_m[feature_cols_clean], ts_m['Sales'])

# --- Save artifacts ---
joblib.dump(final_model, 'sales_forecast_xgb_model.pkl')

with open('feature_cols.json', 'w') as f:
    json.dump(feature_cols_clean, f)

ts_m[['Date', 'Sales']].to_csv('historical_sales_for_inference.csv', index=False)

print("Saved: sales_forecast_xgb_model.pkl, feature_cols.json, historical_sales_for_inference.csv")

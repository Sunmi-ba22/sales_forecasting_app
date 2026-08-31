"""
Run this once (from your modelling notebook or as a standalone script)
to produce the artifacts app.py needs:

  - sales_forecast_xgb_model.pkl
  - feature_cols.json
  - historical_sales_for_inference.csv
  - model_meta.json           (residual std, for prediction intervals)

Assumes `ts_m` (the full engineered feature DataFrame, with 'Date',
'Sales', and all engineered feature columns including 'IsHoliday')
and your train/test split + fitted xgb_model / xgb_pred already exist,
exactly as in your modelling notebook.
"""

import json

import joblib
import numpy as np
from xgboost import XGBRegressor

# Must match the engineered feature columns used during training/evaluation.
feature_cols_clean = [
    'Year', 'Month', 'Quarter', 'Month_sin', 'Month_cos',
    'DayOfWeek', 'IsWeekend', 'IsHoliday',
    'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3', 'Sales_lag_12',
    'Rolling_mean_3', 'Rolling_std_3', 'Rolling_mean_6',
]

# --- Compute residual std from your existing test-set evaluation ---
# (xgb_pred and y_test should already exist from your modelling notebook)
residual_std = float(np.std(y_test - xgb_pred))

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

with open('model_meta.json', 'w') as f:
    json.dump({'residual_std': residual_std}, f)

print("Saved: sales_forecast_xgb_model.pkl, feature_cols.json, "
      "historical_sales_for_inference.csv, model_meta.json")
print(f"Residual std used for 90% prediction intervals: {residual_std:.2f}")

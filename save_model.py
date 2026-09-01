"""
Run this once (from your modelling notebook, after the tuned XGBoost cell)
to produce the artifacts app.py needs:

  - sales_forecast_xgb_model.pkl
  - feature_cols.json
  - model_meta.json           <- previously failed: `residual_std` was
                                  never assigned before being saved
  - historical_sales_for_inference.csv

Assumes `ts_m`, `feature_cols`, `y_test`, and `xgb_pred_tuned` already
exist from the modelling notebook.
"""

import json

import joblib
from xgboost import XGBRegressor

# --- Fix for the model_meta.json bug ---
# This was the missing line: residual_std was referenced when saving
# model_meta.json but never actually created as a variable.
residuals = y_test - xgb_pred_tuned
residual_std = residuals.std()

# --- Retrain the FINAL model on all available data, using the TUNED
#     hyperparameters (this is the model that actually won: MAE 10,881.89
#     vs 11,209.91 untuned) ---
final_model = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=1.0,
    min_child_weight=1,
    random_state=42,
)
final_model.fit(ts_m[feature_cols], ts_m['Sales'])

# --- Save artifacts ---
joblib.dump(final_model, 'sales_forecast_xgb_model.pkl')

with open('feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

with open('model_meta.json', 'w') as f:
    json.dump({'residual_std': float(residual_std)}, f)

ts_m[['Date', 'Sales']].to_csv('historical_sales_for_inference.csv', index=False)

print("Saved: sales_forecast_xgb_model.pkl, feature_cols.json, model_meta.json, historical_sales_for_inference.csv")
print(f"residual_std = {residual_std:.2f}  (used for the 90% prediction interval in the app)")

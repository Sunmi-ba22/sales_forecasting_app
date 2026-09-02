import csv
import json
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from model_utils import forecast_n_months

# --- Logging setup (Monitoring & Logging requirement) ---
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
FORECAST_LOG = "forecast_log.csv"


def log_forecast_request(n_months, forecast_df):
    """Append this forecast request to a simple audit-trail CSV, so
    forecast accuracy can be tracked/reviewed over time against actuals
    as they come in."""
    is_new = not os.path.exists(FORECAST_LOG)
    with open(FORECAST_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["requested_at", "horizon_months", "target_month", "predicted_sales"])
        for _, row in forecast_df.iterrows():
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                n_months,
                row["Date"].date(),
                round(row["Predicted_Sales"], 2),
            ])
    logging.info(f"Forecast generated: horizon={n_months} months")


st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="centered")

st.title(" Sales Forecasting")
st.caption(
    "Tuned XGBoost model trained on monthly Superstore sales (Jan 2016-Dec 2018). "
    "Forecasts are generated recursively month-by-month, so accuracy is highest "
    "for the next 1-2 months and should be read as directional beyond that."
)

# --- Load artifacts with error handling (Security requirement) ---
try:
    model = joblib.load("sales_forecast_xgb_model.pkl")
except FileNotFoundError:
    st.error("Model file 'sales_forecast_xgb_model.pkl' not found. Run save_model.py first.")
    st.stop()

try:
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
except FileNotFoundError:
    st.error("'feature_cols.json' not found. Run save_model.py first.")
    st.stop()

residual_std = None
try:
    with open("model_meta.json") as f:
        residual_std = json.load(f).get("residual_std")
except FileNotFoundError:
    st.warning(
        "'model_meta.json' not found — forecasts will be shown without a "
        "prediction interval. Run save_model.py to generate it."
    )

try:
    history = pd.read_csv("historical_sales_for_inference.csv", parse_dates=["Date"])
    history = history.sort_values("Date")
except FileNotFoundError:
    st.error("'historical_sales_for_inference.csv' not found. Run save_model.py first.")
    st.stop()

if history.empty or len(history) < 12:
    st.error("Historical data has fewer than 12 months — cannot compute lag/rolling features.")
    st.stop()

st.subheader("Historical Monthly Sales")
st.line_chart(history.set_index("Date")["Sales"])

st.subheader("Model Comparison")
st.caption("Evaluated on a chronological train/test split (Jan 2016-Apr 2018 train, May-Dec 2018 test).")
comparison_df = pd.DataFrame({
    "Model": ["XGBoost (tuned)", "XGBoost (untuned)", "Random Forest", "SARIMA", "Naive", "ARIMA", "Prophet"],
    "MAE": [10881.89, 11209.91, 12499.05, 15126.15, 17680.60, 24641.50, 23095.66],
    "MAPE (%)": [13.58, 14.34, 16.44, 20.18, 22.35, 28.20, 35.77],
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)
st.caption(
    "Linear Regression is excluded from this comparison: it produced a near-zero "
    "error on this dataset, which reflects overfitting"
    "rather than genuine forecasting skill."
)

st.subheader("Generate a Forecast")
n_months = st.slider("Months to forecast ahead", min_value=1, max_value=12, value=3)

if st.button("Generate Forecast", type="primary"):
    try:
        with st.spinner("Forecasting..."):
            forecast_df = forecast_n_months(model, history, feature_cols, n_months, residual_std)
            st.info(interpret_forecast(history, forecast_df))

    except ValueError as e:
        st.error(f"Could not generate forecast: {e}")
        st.stop()

    log_forecast_request(n_months, forecast_df)
    
    st.success(f"Forecast generated for {n_months} month(s) ahead.")

    chart_df = history[["Date", "Sales"]].tail(12).rename(columns={"Sales": "Actual"})
    forecast_chart = forecast_df[["Date", "Predicted_Sales"]].rename(columns={"Predicted_Sales": "Forecast"})
    combined = pd.concat([chart_df, forecast_chart]).set_index("Date")
    st.line_chart(combined)

    if "Lower_90" in forecast_df.columns:
        st.caption(
            "90% prediction interval (empirical coverage on test data: 87.5%, "
            "close to the nominal 90% target)."
        )

    st.dataframe(forecast_df, use_container_width=True)

    if n_months > 2:
        st.warning(
            "Forecasts beyond ~1-2 months are generated recursively from prior "
            "predictions rather than actual data, so error compounds — treat "
            "longer-horizon values as directional, not precise."
        )

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv_bytes, "sales_forecast.csv", "text/csv")


def interpret_forecast(history, forecast_df):
    last_actual = history['Sales'].iloc[-1]
    first_pred = forecast_df['Predicted_Sales'].iloc[0]
    last_pred = forecast_df['Predicted_Sales'].iloc[-1]
    pct_change = ((last_pred - last_actual) / last_actual) * 100
    direction = "rising" if last_pred > first_pred else "falling" if last_pred < first_pred else "flat"
    return f"Sales are trending **{direction}** over this forecast, moving from **{last_actual:,.0f}** (last known month) to **{last_pred:,.0f}** by the end of the horizon — a **{pct_change:+.1f}%** change."

st.divider()
st.caption(
    "Model: XGBoost Regressor (tuned: n_estimators=500, max_depth=3, learning_rate=0.05, "
    "subsample=0.6, colsample_bytree=1.0, min_child_weight=1). Selected for the lowest "
    "MAE/MAPE among all validated models. See README for full comparison and business insights."
)

import json
import logging
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from model_utils import forecast_n_days

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="centered")

# --- Logging setup: tracks every forecast request for monitoring/audit ---
logging.basicConfig(
    filename="forecast_log.csv",
    level=logging.INFO,
    format="%(asctime)s,%(message)s",
)

st.title("📈 Sales Forecasting")
st.caption(
    "XGBoost model trained on the Superstore Sales dataset. "
    "Forecasts are generated recursively, so accuracy is highest "
    "in the first ~7-10 days and degrades further out."
)

# --- Model comparison results, from the modelling notebook ---
COMPARISON_RESULTS = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest", "SARIMA", "Naive", "Prophet", "ARIMA"],
    "MAE": [11209.91, 12499.05, 15126.15, 17680.60, 23095.66, 24641.50],
    "RMSE": [15126.54, 15868.44, 20098.40, 22081.73, 27067.54, 33408.70],
    "MAPE (%)": [14.34, 16.44, 20.18, 22.35, 35.77, 28.20],
})


@st.cache_resource
def load_model():
    try:
        return joblib.load("sales_forecast_xgb_model.pkl")
    except FileNotFoundError:
        st.error(
            "Model file 'sales_forecast_xgb_model.pkl' not found. "
            "Run save_model.py first to generate it."
        )
        st.stop()


@st.cache_resource
def load_feature_cols():
    try:
        with open("feature_cols.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("feature_cols.json not found. Run save_model.py first.")
        st.stop()


@st.cache_resource
def load_meta():
    try:
        with open("model_meta.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(
            "model_meta.json not found — forecasts will be shown without "
            "confidence intervals. Run the updated save_model.py to enable them."
        )
        return {"residual_std": None}


@st.cache_data
def load_history():
    try:
        df = pd.read_csv("historical_sales_for_inference.csv", parse_dates=["Date"])
    except FileNotFoundError:
        st.error(
            "historical_sales_for_inference.csv not found. Run save_model.py first."
        )
        st.stop()
    if df.empty or "Sales" not in df.columns:
        st.error("Historical sales data is empty or malformed.")
        st.stop()
    return df.sort_values("Date")


model = load_model()
feature_cols = load_feature_cols()
meta = load_meta()
history = load_history()

st.subheader("Historical Sales")
st.line_chart(history.set_index("Date")["Sales"])

st.subheader("Model Comparison")
st.caption(
    "XGBoost was selected for deployment based on the best test-set metrics "
    "and the most consistent performance across cross-validation folds."
)
st.dataframe(
    COMPARISON_RESULTS.sort_values("MAE").reset_index(drop=True),
    use_container_width=True,
)

st.subheader("Generate a Forecast")
n_days = st.slider("Days to forecast ahead", min_value=1, max_value=30, value=7)

if st.button("Generate Forecast", type="primary"):
    try:
        with st.spinner("Forecasting..."):
            forecast_df = forecast_n_days(
                model, history, feature_cols, n_days,
                residual_std=meta.get("residual_std"),
            )
    except ValueError as e:
        st.error(f"Could not generate forecast: {e}")
        st.stop()

    st.success(f"Forecast generated for {n_days} day(s) ahead.")

    logging.info(
        f"forecast_request,n_days={n_days},"
        f"first_date={forecast_df['Date'].min()},"
        f"last_date={forecast_df['Date'].max()},"
        f"mean_predicted={forecast_df['Predicted_Sales'].mean():.2f}"
    )

    chart_df = history[["Date", "Sales"]].tail(30).rename(columns={"Sales": "Actual"})
    forecast_plot = forecast_df.rename(columns={"Predicted_Sales": "Forecast"})
    combined = pd.concat([chart_df, forecast_plot[["Date", "Forecast"]]]).set_index("Date")
    st.line_chart(combined)

    if "Lower_90" in forecast_df.columns:
        st.caption(
            "Forecast table includes a 90% prediction interval "
            "(Lower_90 / Upper_90), based on residual variance from the "
            "test-set evaluation. This is a simple approximation, not a "
            "rigorously calibrated interval — treat it as a rough range."
        )

    st.dataframe(forecast_df, use_container_width=True)

    if n_days > 10:
        st.warning(
            "Forecasts beyond ~10 days are generated recursively from prior "
            "predictions rather than actual data, so error compounds — treat "
            "longer-horizon values as directional, not precise."
        )

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv, "sales_forecast.csv", "text/csv")

st.divider()
st.caption(
    "Model: XGBoost Regressor (n_estimators=200, max_depth=3, learning_rate=0.05). "
    "Selected for its consistency across cross-validation folds rather than "
    "single-split performance alone. See README for full model comparison."
)

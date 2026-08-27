import json

import joblib
import pandas as pd
import streamlit as st

from model_utils import forecast_n_days

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="centered")

st.title(" Sales Forecasting")
st.caption(
    "XGBoost model trained on the Superstore Sales dataset. "
    "Forecasts are generated recursively, so accuracy is highest "
    "in the first ~7-10 days and degrades further out."
)


@st.cache_resource
def load_model():
    return joblib.load("sales_forecast_xgb_model.pkl")


@st.cache_resource
def load_feature_cols():
    with open("feature_cols.json") as f:
        return json.load(f)


@st.cache_data
def load_history():
    df = pd.read_csv("historical_sales_for_inference.csv", parse_dates=["Date"])
    return df.sort_values("Date")


model = load_model()
feature_cols = load_feature_cols()
history = load_history()

st.subheader("Historical Sales")
st.line_chart(history.set_index("Date")["Sales"])

st.subheader("Generate a Forecast")
n_days = st.slider("Days to forecast ahead", min_value=1, max_value=30, value=7)

if st.button("Generate Forecast", type="primary"):
    with st.spinner("Forecasting..."):
        forecast_df = forecast_n_days(model, history, feature_cols, n_days)

    st.success(f"Forecast generated for {n_days} day(s) ahead.")

    combined = pd.concat(
        [
            history[["Date", "Sales"]].tail(30).rename(columns={"Sales": "Actual"}),
            forecast_df.rename(columns={"Predicted_Sales": "Forecast"}),
        ]
    ).set_index("Date")

    st.line_chart(combined)
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

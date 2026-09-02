"""
Sales Forecasting App
Streamlit deployment for the tuned XGBoost monthly sales forecasting model.

Requires (in the same directory):
  - sales_forecast_xgb_model.pkl
  - feature_cols.json
  - model_meta.json                (residual_std, used for prediction intervals)
  - historical_sales_for_inference.csv
"""

import json
import os
from datetime import datetime

import holidays
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title=" Forecasting", page_icon="📈", layout="centered")

MODEL_PATH = "sales_forecast_xgb_model.pkl"
FEATURES_PATH = "feature_cols.json"
META_PATH = "model_meta.json"
HISTORY_PATH = "historical_sales_for_inference.csv"
LOG_PATH = "forecast_log.csv"

# Test-set metrics for the deployed model, for the comparison dashboard.
# Keep this in sync with the modelling notebook's final results table.
MODEL_COMPARISON = pd.DataFrame({
    "Model": ["XGBoost (Tuned)", "XGBoost (Untuned)", "Random Forest",
              "SARIMA (Untuned)", "Naive", "ARIMA", "Prophet"],
    "MAE": [10790.08, 11209.91, 12549.99, 15126.15, 17680.60, 24641.50, 23095.66],
    "RMSE": [14825.21, 15126.54, 15912.46, 20098.40, 22081.73, 33408.70, 27067.54],
    "MAPE (%)": [13.65, 14.34, 16.60, 20.18, 22.35, 28.20, 35.77],
})


# --------------------------------------------------------------------------
# Loading (cached, with error handling so a missing/corrupt artifact
# fails with a clear message instead of a raw traceback)
# --------------------------------------------------------------------------

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}. Run save_model.py first.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()


@st.cache_resource
def load_feature_cols():
    try:
        with open(FEATURES_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Feature column list not found: {FEATURES_PATH}. Run save_model.py first.")
        st.stop()


@st.cache_resource
def load_meta():
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(
            f"{META_PATH} not found — prediction intervals will be disabled. "
            "Run save_model.py to generate it."
        )
        return {"residual_std": None}


@st.cache_data
def load_history():
    try:
        df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    except FileNotFoundError:
        st.error(f"Historical data not found: {HISTORY_PATH}. Run save_model.py first.")
        st.stop()
    return df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature engineering for a future month — mirrors the training notebook,
# with Diff_1 / Pct_change_1 redefined to use only already-known months
# (see note above the app about why this differs from the training-time
# definition).
# --------------------------------------------------------------------------

def build_features_for_month(target_date, history_df, feature_cols):
    """
    target_date : pd.Timestamp, month-end date being forecast
    history_df  : DataFrame['Date','Sales'] containing all known/forecast
                  months strictly before target_date
    """
    hist = history_df.sort_values("Date").set_index("Date")["Sales"]

    if len(hist) < 12:
        raise ValueError(
            "Need at least 12 prior months of history to build Sales_lag_12 "
            "and rolling features."
        )

    row = {}
    row["Year"] = target_date.year
    row["Month"] = target_date.month
    row["Quarter"] = (target_date.month - 1) // 3 + 1
    row["Month_sin"] = np.sin(2 * np.pi * target_date.month / 12)
    row["Month_cos"] = np.cos(2 * np.pi * target_date.month / 12)

    row["Sales_lag_1"] = hist.iloc[-1]
    row["Sales_lag_2"] = hist.iloc[-2]
    row["Sales_lag_3"] = hist.iloc[-3]
    row["Sales_lag_12"] = hist.iloc[-12]
    row["Rolling_mean_3"] = hist.iloc[-3:].mean()
    row["Rolling_std_3"] = hist.iloc[-3:].std()
    row["Rolling_mean_6"] = hist.iloc[-6:].mean()

    # Diff_1 / Pct_change_1: computed from the two most recent
    # KNOWN months, not from the target month itself (which doesn't exist
    # yet at forecast time). This differs from the training-time definition
    # (Sales.diff(1) / Sales.pct_change(1) evaluated on the target month) —
    # see the module docstring note above.
    lag1, lag2 = hist.iloc[-1], hist.iloc[-2]
    row["Diff_1"] = lag1 - lag2
    row["Pct_change_1"] = (lag1 - lag2) / lag2 if lag2 != 0 else 0.0

    # Holiday flag — dynamically pulls US holidays for the target year,
    # so this works for any future year, not just the training years.
    us_holidays = holidays.US(years=[target_date.year])
    holiday_months = {d.month for d in us_holidays if d.year == target_date.year}
    row["IsHoliday"] = int(target_date.month in holiday_months)

    features = pd.DataFrame([row])
    missing = set(feature_cols) - set(features.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return features[feature_cols]


def forecast_n_months(model, history_df, feature_cols, n_months, residual_std=None):
    """
    Recursively forecasts n_months ahead of the last month in history_df.
    Each predicted month is appended back into working history so the next
    month's lag/rolling features are computed from it — meaning forecast
    error compounds with horizon length (flagged in the app UI beyond 3 months).
    """
    hist = history_df[["Date", "Sales"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    last_date = hist["Date"].max()

    forecasts = []
    for i in range(1, n_months + 1):
        target_date = (last_date + pd.DateOffset(months=i)) + pd.offsets.MonthEnd(0)
        X_new = build_features_for_month(target_date, hist, feature_cols)
        pred = float(model.predict(X_new)[0])

        row = {"Date": target_date, "Predicted_Sales": pred}
        if residual_std is not None:
            row["Lower_90"] = pred - 1.645 * residual_std
            row["Upper_90"] = pred + 1.645 * residual_std

        forecasts.append(row)
        hist = pd.concat(
            [hist, pd.DataFrame([{"Date": target_date, "Sales": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


# --------------------------------------------------------------------------
# Automatic plain-language interpretation of each forecast
# --------------------------------------------------------------------------

def interpret_forecast(forecast_df, history_df):
    """
    Builds a plain-language interpretation for each forecasted month plus
    an overall summary, based on the forecast values, recent history, and
    the holiday/seasonal context.
    """
    hist_sorted = history_df.sort_values("Date")
    last_actual_value = hist_sorted["Sales"].iloc[-1]
    last_actual_date = hist_sorted["Date"].iloc[-1]
    recent_avg = hist_sorted["Sales"].tail(6).mean()

    us_holidays_all = holidays.US(years=sorted(forecast_df["Date"].dt.year.unique()))
    holiday_months = {d.month for d in us_holidays_all}

    lines = []
    prev_value = last_actual_value
    prev_label = last_actual_date.strftime("%B %Y")

    for _, r in forecast_df.iterrows():
        month_label = r["Date"].strftime("%B %Y")
        pred = r["Predicted_Sales"]
        pct_change = ((pred - prev_value) / prev_value * 100) if prev_value else 0.0
        direction = "up" if pct_change > 0 else ("down" if pct_change < 0 else "flat versus")
        is_holiday = r["Date"].month in holiday_months

        sentence = (
            f"**{month_label}: ${pred:,.0f}** — {direction} {abs(pct_change):.1f}% "
            f"versus {prev_label} (${prev_value:,.0f})."
        )

        if "Lower_90" in r and "Upper_90" in r:
            sentence += f" 90% expected range: ${r['Lower_90']:,.0f} – ${r['Upper_90']:,.0f}."

        if is_holiday:
            sentence += " This month contains a US public holiday — historically associated with higher sales for Nova Mart."

        if pred > recent_avg * 1.15:
            sentence += " This is notably above the recent 6-month average, consistent with a seasonal peak."
        elif pred < recent_avg * 0.85:
            sentence += " This is notably below the recent 6-month average — worth reviewing before committing inventory."

        lines.append(sentence)
        prev_value = pred
        prev_label = month_label

    # Overall summary across the whole forecast horizon
    total_change_pct = (
        (forecast_df["Predicted_Sales"].iloc[-1] - last_actual_value) / last_actual_value * 100
        if last_actual_value else 0.0
    )
    trend_word = "growth" if total_change_pct > 0 else "decline"
    summary = (
        f"Over the {len(forecast_df)}-month forecast horizon, predicted sales show an overall "
        f"{trend_word} of {abs(total_change_pct):.1f}% from the last known month "
        f"({last_actual_date.strftime('%B %Y')}, ${last_actual_value:,.0f}) to "
        f"{forecast_df['Date'].iloc[-1].strftime('%B %Y')} "
        f"(${forecast_df['Predicted_Sales'].iloc[-1]:,.0f})."
    )
    if len(forecast_df) > 3:
        summary += (
            " Note: forecasts more than 3 months out are generated recursively from prior "
            "predictions rather than actual data, so treat later months as directional rather than precise."
        )

    return summary, lines


# --------------------------------------------------------------------------
# Forecast logging
# --------------------------------------------------------------------------

def log_forecast(forecast_df, n_months):
    log_entry = forecast_df.copy()
    log_entry["Generated_At"] = datetime.now().isoformat(timespec="seconds")
    log_entry["Horizon_Months"] = n_months

    try:
        if os.path.exists(LOG_PATH):
            log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            log_entry.to_csv(LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Could not write to forecast log ({e}) — forecast still shown below.")


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------

st.title(" Nova Mart Ltd — Sales Forecasting")
st.caption(
    "Tuned XGBoost model (MAE 10,790 · MAPE 13.65% on held-out test months). "
    "Forecasts are generated recursively month-by-month, so accuracy is highest "
    "in the first 1-3 months and becomes more directional further out."
)

model = load_model()
feature_cols = load_feature_cols()
meta = load_meta()
residual_std = meta.get("residual_std")
history = load_history()

st.subheader("Historical Monthly Sales")
st.line_chart(history.set_index("Date")["Sales"])

with st.expander("Model comparison (why XGBoost was chosen)"):
    st.dataframe(MODEL_COMPARISON.sort_values("MAE").reset_index(drop=True), use_container_width=True)
    st.caption(
        "XGBoost was selected for its strongest test-set and cross-validation performance. "
        "Linear Regression is excluded — it showed a data-leakage artifact rather than genuine skill."
    )

st.subheader("Generate a Forecast")
n_months = st.slider("Months to forecast ahead", min_value=1, max_value=12, value=3)

if st.button("Generate Forecast", type="primary"):
    try:
        with st.spinner("Forecasting..."):
            forecast_df = forecast_n_months(model, history, feature_cols, n_months, residual_std)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

    st.success(f"Forecast generated for {n_months} month(s) ahead.")

    plot_df = pd.concat([
        history[["Date", "Sales"]].tail(24).rename(columns={"Sales": "Actual"}),
        forecast_df[["Date", "Predicted_Sales"]].rename(columns={"Predicted_Sales": "Forecast"}),
    ]).set_index("Date")
    st.line_chart(plot_df)

    if residual_std is not None:
        display_df = forecast_df.style.format(
            {"Predicted_Sales": "${:,.0f}", "Lower_90": "${:,.0f}", "Upper_90": "${:,.0f}"}
        )
    else:
        display_df = forecast_df.style.format({"Predicted_Sales": "${:,.0f}"})
    st.dataframe(display_df, use_container_width=True)

    # --- Automatic interpretation ---
    st.subheader("Interpretation")
    summary, monthly_lines = interpret_forecast(forecast_df, history)
    st.markdown(summary)
    for line in monthly_lines:
        st.markdown(f"- {line}")

    if n_months > 3:
        st.warning(
            "Forecasts beyond 3 months are generated recursively from prior predictions "
            "rather than actual data, so error compounds — treat longer-horizon values as "
            "directional, not precise."
        )

    log_forecast(forecast_df, n_months)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv, "nova_mart_sales_forecast.csv", "text/csv")

st.divider()
st.caption(
    "Model: Tuned XGBoost Regressor (n_estimators=500, max_depth=3, learning_rate=0.05, "
    "subsample=0.6, colsample_bytree=1.0, min_child_weight=1). "
    "Selected for its consistency across cross-validation folds. See README for full model comparison."
)

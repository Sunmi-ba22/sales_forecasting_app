"""
Feature-generation utilities used at inference time by app.py.

Must exactly mirror the MONTHLY feature engineering used during training
(ts_m has one row per month, Jan 2016-Dec 2018). Feature columns:
  Year, Month, Quarter, Month_sin, Month_cos,
  Sales_lag_1, Sales_lag_2, Sales_lag_3, Sales_lag_12,
  Rolling_mean_3, Rolling_std_3, Rolling_mean_6,
  Pct_change_1, Diff_1, IsHoliday

NOTE on IsHoliday: in the training data this feature was constant 0 for
every row (a bug in how it was computed upstream), so the model was never
able to learn anything from it — its feature importance is exactly 0.
The fix below computes it correctly (whether the target month contains a
US federal holiday) so at least new forecasts are consistent, but this
will NOT change predictions until IsHoliday is recomputed correctly in
the feature engineering notebook and the model is retrained on it.
"""

import numpy as np
import pandas as pd
import holidays

_US_HOLIDAYS_CACHE = {}


def _month_has_holiday(target_date):
    """True if any US federal holiday falls within target_date's month."""
    year = target_date.year
    if year not in _US_HOLIDAYS_CACHE:
        _US_HOLIDAYS_CACHE[year] = holidays.US(years=[year])
    return int(any(
        d.year == target_date.year and d.month == target_date.month
        for d in _US_HOLIDAYS_CACHE[year]
    ))


def build_features_for_date(target_date, history_df, feature_cols):
    """
    Build a single-row feature DataFrame for `target_date` (a month-end
    timestamp), using only sales data from prior months.

    Parameters
    ----------
    target_date : pd.Timestamp
        Month-end date to generate a forecast for.
    history_df : pd.DataFrame
        Columns ['Date', 'Sales'] — monthly sales, actual or previously
        forecast, for all months before target_date.
    feature_cols : list[str]
        Exact ordered feature columns the model was trained on.

    Returns
    -------
    pd.DataFrame, single row, columns matching feature_cols.
    """
    row = {}
    row['Year'] = target_date.year
    row['Month'] = target_date.month
    row['Quarter'] = (target_date.month - 1) // 3 + 1
    row['Month_sin'] = np.sin(2 * np.pi * target_date.month / 12)
    row['Month_cos'] = np.cos(2 * np.pi * target_date.month / 12)
    row['IsHoliday'] = _month_has_holiday(target_date)

    hist = history_df.sort_values('Date').set_index('Date')['Sales']

    if len(hist) < 12:
        raise ValueError(
            "Need at least 12 prior months of history to build "
            "Sales_lag_12 / rolling features. Provide more historical data."
        )

    row['Sales_lag_1'] = hist.iloc[-1]
    row['Sales_lag_2'] = hist.iloc[-2]
    row['Sales_lag_3'] = hist.iloc[-3]
    row['Sales_lag_12'] = hist.iloc[-12]
    row['Rolling_mean_3'] = hist.iloc[-3:].mean()
    row['Rolling_std_3'] = hist.iloc[-3:].std()
    row['Rolling_mean_6'] = hist.iloc[-6:].mean()

    # Month-over-month growth features (mirrors Sales.diff(1) /
    # Sales.pct_change(1) from the feature engineering notebook).
    prev_1 = hist.iloc[-1]
    prev_2 = hist.iloc[-2]
    row['Diff_1'] = prev_1 - prev_2
    row['Pct_change_1'] = (prev_1 - prev_2) / prev_2 if prev_2 != 0 else 0.0

    features = pd.DataFrame([row])

    missing = set(feature_cols) - set(features.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return features[feature_cols]


def forecast_n_months(model, history_df, feature_cols, n_months, residual_std=None):
    """
    Recursively forecast `n_months` ahead of the last date in history_df.

    Each predicted month is appended back into the working history so the
    next month's lag/rolling features are computed from it — forecast
    error compounds with horizon length, so accuracy is strongest for the
    next 1-2 months and should be treated as directional beyond that.

    Parameters
    ----------
    residual_std : float, optional
        Std dev of the model's test-set residuals (loaded from
        model_meta.json). If given, adds a 90% prediction interval
        (+/- 1.645 * residual_std) to every forecasted row. This is a
        simple residual-based interval; its empirical coverage was 87.5%
        on held-out test data (close to the nominal 90% target).

    Returns
    -------
    pd.DataFrame with ['Date', 'Predicted_Sales'] and, if residual_std is
    given, ['Lower_90', 'Upper_90'].
    """
    hist = history_df[['Date', 'Sales']].copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    last_date = hist['Date'].max()

    forecasts = []
    for i in range(1, n_months + 1):
        target_date = last_date + pd.DateOffset(months=i)
        # Snap to month-end, matching the training data's resample('M') grain
        target_date = target_date + pd.offsets.MonthEnd(0)

        X_new = build_features_for_date(target_date, hist, feature_cols)
        pred = float(model.predict(X_new)[0])

        row = {'Date': target_date, 'Predicted_Sales': pred}
        if residual_std is not None:
            row['Lower_90'] = pred - 1.645 * residual_std
            row['Upper_90'] = pred + 1.645 * residual_std
        forecasts.append(row)

        hist = pd.concat(
            [hist, pd.DataFrame([{'Date': target_date, 'Sales': pred}])],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)

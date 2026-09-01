"""
Feature-generation utilities used at inference time by app.py.

These functions must exactly mirror the feature engineering logic used
during training (see the feature engineering notebook) so the model
receives inputs in the same shape/format it was trained on.
"""

import numpy as np
import pandas as pd
import holidays

_US_HOLIDAYS_CACHE = {}


def _is_holiday(target_date):
    """Cache the holidays.US() lookup per year to avoid rebuilding it on every call."""
    year = target_date.year
    if year not in _US_HOLIDAYS_CACHE:
        _US_HOLIDAYS_CACHE[year] = holidays.US(years=[year])
    return int(target_date in _US_HOLIDAYS_CACHE[year])


def build_features_for_date(target_date, history_df, feature_cols):
    """
    Build a single-row feature DataFrame for `target_date`, using only
    sales data that occurred before that date.

    Parameters
    ----------
    target_date : pd.Timestamp
        The date to generate a forecast for.
    history_df : pd.DataFrame
        Columns ['Date', 'Sales'], containing all known/actual or
        previously-forecast sales up to (but not including) target_date.
    feature_cols : list[str]
        The exact ordered list of feature columns the model was trained on.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns matching feature_cols, ready
        to pass to model.predict().
    """
    row = {}
    row['Year'] = target_date.year
    row['Month'] = target_date.month
    row['Quarter'] = (target_date.month - 1) // 3 + 1
    row['Month_sin'] = np.sin(2 * np.pi * target_date.month / 12)
    row['Month_cos'] = np.cos(2 * np.pi * target_date.month / 12)
    row['DayOfWeek'] = target_date.dayofweek
    row['IsWeekend'] = int(target_date.dayofweek in [5, 6])
    row['IsHoliday'] = _is_holiday(target_date)

    hist = history_df.sort_values('Date').set_index('Date')['Sales']

    if len(hist) < 12:
        raise ValueError(
            "Need at least 12 prior days of history to build lag_12 / "
            "rolling features. Provide more historical data."
        )

    row['Sales_lag_1'] = hist.iloc[-1]
    row['Sales_lag_2'] = hist.iloc[-2]
    row['Sales_lag_3'] = hist.iloc[-3]
    row['Sales_lag_12'] = hist.iloc[-12]
    row['Rolling_mean_3'] = hist.iloc[-3:].mean()
    row['Rolling_std_3'] = hist.iloc[-3:].std()
    row['Rolling_mean_6'] = hist.iloc[-6:].mean()

    # Growth features: day-over-day change, based on the two most recent
    # known/prior-forecast values (mirrors Sales.diff(1) / Sales.pct_change(1)
    # from the feature engineering notebook).
    prev_1 = hist.iloc[-1]
    prev_2 = hist.iloc[-2]
    row['Diff_1'] = prev_1 - prev_2
    row['Pct_change_1'] = (prev_1 - prev_2) / prev_2 if prev_2 != 0 else 0.0

    features = pd.DataFrame([row])

    # Guard against any mismatch between what's built here and what the
    # model expects — fail loudly rather than silently misaligning columns.
    missing = set(feature_cols) - set(features.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return features[feature_cols]


def forecast_n_days(model, history_df, feature_cols, n_days, residual_std=None):
    """
    Recursively forecast `n_days` ahead of the last date in history_df.

    Each predicted day is appended back into the working history so the
    next day's lag/rolling features are computed from it. This means
    forecast error compounds with horizon length — accuracy degrades the
    further out the forecast goes (see README limitations).

    Parameters
    ----------
    residual_std : float, optional
        Standard deviation of the model's test-set residuals (computed
        once at training time and saved — see save_model.py). If given,
        a 90% prediction interval (+/- 1.645 * residual_std) is added to
        every forecasted row. This is a simple residual-based interval,
        not a rigorously calibrated one — its actual coverage should be
        checked against held-out data (see README).

    Returns
    -------
    pd.DataFrame with columns ['Date', 'Predicted_Sales'] and, if
    residual_std is provided, ['Lower_90', 'Upper_90'].
    """
    hist = history_df[['Date', 'Sales']].copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    last_date = hist['Date'].max()

    forecasts = []
    for i in range(1, n_days + 1):
        target_date = last_date + pd.Timedelta(days=i)
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

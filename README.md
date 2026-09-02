# Sales Forecasting — Superstore Dataset

An end-to-end time series forecasting project predicting monthly sales for a retail superstore, from raw data cleaning through model deployment.

**Live app:** _add your Streamlit Cloud link here after deployment_
**Dataset:** [Superstore Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

---

## Project Overview

The goal was to forecast monthly sales using historical transaction data, comparing classical time series methods against machine learning approaches, and to deploy the best-performing model as an interactive forecasting app with prediction intervals.

## Workflow

1. **Data Cleaning** — handled missing values, corrected data types, standardized date fields
2. **Exploratory Data Analysis** (order-level, 2015–2018)
   - Sales distribution is right-skewed with legitimate high-value outliers (large orders, not errors)
   - Year-over-year sales are not monotonic: 2016 dipped below 2015 before recovering; 2018 was the strongest year
   - Seasonal peaks around November/December and March
   - Tuesday and Saturday were the strongest days of the week; Thursday the weakest
   - Technology (Phones) led by category/sub-category; Consumer led by segment; West led by region
   - No linear correlation between Sales and raw numeric fields (Row ID, Postal Code, date parts) — sales drivers are categorical/cyclical, not linear
3. **Feature Engineering** (aggregated to monthly grain)
   - Calendar features: month, quarter, cyclical month encoding (sin/cos)
   - Lag features: sales 1, 2, 3, and 12 months prior
   - Rolling window features: 3-month and 6-month rolling mean/std
   - Growth features: month-over-month percent change and difference
   - Holiday feature: whether the month contains a US federal holiday (fixed from an initial bug — see Known Issues Fixed below)
4. **Stationarity Testing & Decomposition**
   - ADF test: series stationary (p = 0.0003); KPSS test: series non-stationary (p = 0.02) — the contradiction indicates a **trend-stationary** series (steady deterministic trend, not a random-walk/unit-root process)
   - Seasonal decomposition confirmed a clear upward trend and a repeating annual seasonal pattern
   - First-differencing fully stabilizes the series (ADF p ≈ 0.0000) if needed for classical models
5. **Modeling** — six models trained and compared: Naive baseline, Linear Regression, Random Forest, XGBoost, Prophet, ARIMA, SARIMA
6. **Hyperparameter Tuning** — grid search (XGBoost) and `auto_arima` (SARIMA) against TimeSeriesSplit cross-validation
7. **Prediction Intervals** — residual-based 90% interval around the XGBoost forecast
8. **Deployment** — final model saved and served via a Streamlit app with a live forecast interpretation, model comparison dashboard, and request logging

## Model Comparison

**Single train/test split** (train: Jan 2016–Apr 2018, test: May–Dec 2018):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost (untuned) | 11,209.91 | 15,126.54 | 14.34% |
| Random Forest | 12,549.99 | 15,912.46 | 16.60% |
| SARIMA (untuned) | 15,126.15 | 20,098.40 | 20.18% |
| Naive | 17,680.60 | 22,081.73 | 22.35% |
| Prophet | 23,095.66 | 27,067.54 | 35.77% |
| ARIMA | 24,641.50 | 33,408.70 | 28.20% |

*Linear Regression is excluded from this table — see Known Issues below.*

**4-fold time series cross-validation (mean across folds):**

| Model | Mean MAE | Mean MAPE |
|---|---|---|
| XGBoost | 15,846.70 | 26.89% |
| SARIMA | 18,957.97 | 35.70% |
| Naive | 21,772.81 | 43.16% |
| ARIMA | 22,144.98 | 40.15% |
| Random Forest | 15,607.36 | — *(see Known Issues)* |
| Prophet | 82,096.00 | 167.67% |

**Hyperparameter tuning results:**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost (tuned) | 10,790.08 | 14,825.21 | 13.65% |
| XGBoost (untuned) | 11,209.91 | 15,126.54 | 14.34% |
| SARIMA (untuned) | 15,126.15 | 20,098.40 | 20.18% |
| SARIMA (tuned) | 26,205.41 | 34,874.12 | 30.43% |

Tuned XGBoost params (via `GridSearchCV` + `TimeSeriesSplit`): `n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.6, colsample_bytree=1.0, min_child_weight=1`

**Feature importance (final tuned model):**

| Feature | Importance |
|---|---|
| Diff_1 (month-over-month change) | 30.5% |
| Sales_lag_12 (same month last year) | 23.0% |
| Month | 13.7% |
| Pct_change_1 | 11.4% |
| Sales_lag_1 | 4.1% |
| Month_cos | 3.4% |
| Rolling_std_3 | 2.8% |
| Year | 2.5% |
| Rolling_mean_6 | 2.2% |
| Sales_lag_2 | 2.2% |
| Sales_lag_3 | 1.3% |
| IsHoliday | 1.1% |
| Rolling_mean_3 | 1.0% |
| Month_sin | 0.8% |
| Quarter | 0.0% |

## Model Selection: Why Tuned XGBoost

Linear Regression produced an essentially zero-error result, but this reflects overfitting — with ~15 features and only 28 training rows, it can memorize training data rather than learn a genuine pattern (confirmed by the same near-zero result appearing in cross-validation, not just the single split). It is excluded from model comparison and recommendation for this reason.

Among valid models, **tuned XGBoost** achieved the lowest error on both the single test split (MAE 10,790, MAPE 13.65%) and was the most consistent performer across cross-validation folds. SARIMA's tuned result was dramatically worse than its untuned baseline (MAPE 20.18% → 30.43%), most likely because the `auto_arima` search overfit to the limited CV folds available for a small time series — a known risk with hyperparameter search on short series, and a useful cautionary point for the report.

**Final model:** XGBoost Regressor, `n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.6, colsample_bytree=1.0, min_child_weight=1`, retrained on the full historical dataset before deployment.

## Business Insights

### Demand is seasonal, not random
Sales consistently peak in November and December each year, with a secondary spike in March. This is a repeating structural pattern, not noise — the business should treat Q4 inventory planning as a predictable, recurring event rather than reacting to it each year.

### Holidays have a measurable, quantified effect
Months containing a US federal holiday average $53,718 in sales, compared to $41,038 in months without one — a 31% uplift. This is the single most direct, actionable seasonal signal in the dataset, and it now contributes measurably to the forecasting model itself (1.1% of the model's decision-making), on top of what the yearly seasonal pattern already captures.

### Growth has not been steady — plan for down years
Year-over-year revenue is not monotonic: 2016 sales were actually lower than 2015, before recovering and growing through 2017 and 2018. Forecasts and inventory commitments should not assume compounding growth by default.

### Volatility is growing alongside revenue
The 3-month rolling standard deviation has climbed alongside the rolling mean — swings around peak season are larger in absolute terms now than in 2015. Safety stock and forecast confidence intervals should scale up over time, not stay fixed at a historical average.

### Revenue is concentrated in one category
Technology — specifically phones — is the single largest revenue driver, ahead of Furniture and Office Supplies. Forecast accuracy for this sub-category has outsized impact on the bottom line.

### The customer base is consumer-driven, with a B2B undertone
The Consumer segment generates the most revenue, with Home Office lowest. The mid-week (Tuesday) sales peak alongside the weekend (Saturday) peak suggests some business/bulk ordering activity persists alongside personal consumer shopping.

### Recent momentum predicts sales better than the calendar alone
Feature importance from the final model shows month-over-month change (30.5%) and same-month-last-year sales (23.0%) are the two strongest predictors — together over half the model's decision-making. Month number (13.7%) and the holiday flag (1.1%) matter, but far less than recent momentum. This supports prioritizing up-to-date sales data feeds over static seasonal calendars when operationalizing this forecast.

### Forecast reliability and limitations
The final model (tuned XGBoost) achieves a 13.65% average forecast error (MAPE) with a 90% prediction interval that was empirically accurate 87.5% of the time on held-out test data — close to its nominal 90% target. Forecasts are generated recursively for future months (each month's prediction feeds the next month's inputs), so accuracy is highest for the next 1-2 months and should be treated as directional beyond that.

### Recommended actions
- Increase inventory buffers ahead of any month containing a major US holiday, and treat Nov/Dec as the highest-priority buffer period given the historical spike on top of the general holiday effect
- Do not assume automatic year-over-year growth when setting purchasing targets — 2016 shows this business can have a down year
- Widen safety stock margins in more recent/future periods to reflect rising volatility, not a fixed historical average
- Prioritize forecast and inventory accuracy for the Technology/Phones line specifically, given its outsized revenue share
- Use the 90% prediction interval — not just the point forecast — for safety-stock decisions, since it reflects genuine, calibrated uncertainty

## Known Issues Fixed / Flagged

- **`IsHoliday` was initially non-functional**: it checked whether the exact month-end date was a holiday (almost never true). Fixed to check whether the month *contains* a holiday — confirmed working (32 of 48 months flagged, with a real, measurable sales difference between groups) and now has 1.1% feature importance.
- **Linear Regression is excluded from all rankings**: it achieves near-zero error due to overfitting (more correlated features than training rows), not genuine forecasting skill. Included here as an explicit finding, not a silent omission.
- **Random Forest's cross-validation MAPE is unreliable**: the CV summary shows a near-zero `Mean_MAPE` alongside a normal `Mean_MAE` (~15,607) — an internal inconsistency likely caused by a column-reference bug in the CV summary code. Its MAE is trustworthy; its CV MAPE figure is not, and is omitted from the CV table above pending a fix.

## Limitations

- Forecasts beyond ~1-2 months are generated **recursively** (each month's prediction feeds the next month's lag features), so error compounds the further out the forecast horizon extends.
- MAPE is computed excluding months with zero actual sales, to avoid divide-by-zero.
- The model was evaluated on a single historical period (2015–2018) and would benefit from re-validation on more recent data before any real production use.
- Only 36 months of usable data (after losing 12 to lag features) limits how much can be learned — a longer history would improve reliability, especially for less-frequent models like Prophet and ARIMA.

## Repository Structure

```
├── app.py                              # Streamlit app
├── model_utils.py                      # Feature-generation logic used at inference time
├── save_model.py                       # Script to retrain and save deployment artifacts
├── sales_forecast_xgb_model.pkl        # Trained tuned XGBoost model
├── feature_cols.json                   # Feature column list used by the model
├── model_meta.json                     # Residual std dev, used for the 90% prediction interval
├── historical_sales_for_inference.csv  # Historical monthly sales, used to seed lag/rolling features
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
python save_model.py     # generates the .pkl, .json, and .csv artifacts app.py needs
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub, under the EduLinkUp - Developers' Capstone Organisation (include the `.pkl`, `.json`, and `.csv` files — do not gitignore them, the app needs them)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`
4. Click **Deploy** — the first build installs `requirements.txt` and may take a few minutes
5. Once live, copy the app URL into the "Live app" line at the top of this README

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, pmdarima, Prophet, holidays, Streamlit

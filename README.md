# Sales Forecasting : Superstore Dataset

An end-to-end time series forecasting project predicting daily sales for a retail superstore, from raw data cleaning through model deployment.

**Live app:** https://salesforecastingapp-9ist53tckbbaplk7vbs688.streamlit.app/
**Dataset:** [Superstore Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

---

## Project Overview

The goal was to forecast daily sales using historical transaction data, comparing classical time series methods against machine learning approaches, and to deploy the best-performing model as an interactive forecasting app.

## Workflow

1. **Data Cleaning** — handled missing values, corrected data types, standardized date fields
2. **Exploratory Data Analysis**
   - Sales distribution is right-skewed with legitimate high-value outliers (large orders, not errors)
   - Clear upward trend from 2015–2018 (with a dip in 2016)
   - Seasonal peaks around November/December and March
   - Tuesday and Saturday were the strongest days of the week
   - Technology (Phones) led by category/sub-category; Consumer led by segment; West led by region
3. **Feature Engineering**
   - Calendar features: month, quarter, cyclical month encoding (sin/cos), day-of-week, weekend flag
   - Lag features: sales 1, 2, 3, and 12 days prior
   - Rolling window features: 3-day and 6-day rolling mean/std
   - Growth features: day-over-day percent change and difference
4. **Stationarity Testing & Decomposition**
   - ADF test: series stationary (p = 0.0003); KPSS test: series non-stationary (p = 0.02) — the contradiction indicates a **trend-stationary** series (steady deterministic trend, not a random-walk/unit-root process)
   - Seasonal decomposition confirmed a clear upward trend and a repeating annual seasonal pattern
   - First-differencing fully stabilizes the series (ADF p ≈ 0.0000) if needed for classical models
5. **Modeling** — six models trained and compared: Naive baseline, Linear Regression, XGBoost, Prophet, ARIMA, SARIMA
6. **Hyperparameter Tuning** — tuning attempted on the top CV performers (XGBoost, SARIMA)
7. **Deployment** — final model saved and served via a Streamlit app

## Model Comparison

**Single train/test split:**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | 10,977.77 | 14,289.98 | 15.78% |
| XGBoost | 11,929.81 | 15,652.41 | 16.08% |
| Naive | 17,680.60 | — | — |
| SARIMA | 20,368.67 | — | — |
| Prophet | 23,095.66 | — | 35.77% |
| ARIMA | 24,641.50 | 33,408.70 | — |

**5-fold time series cross-validation (mean across folds):**

| Model | Mean MAE | Mean MAPE |
|---|---|---|
| XGBoost | 16,063.31 | 27.30% |
| SARIMA | 18,957.97 | 35.70% |
| Naive | 21,772.81 | 43.16% |
| ARIMA | 22,144.98 | 40.15% |
| Linear Regression | 41,533.86 | 93.52% (high std) |
| Prophet | 81,095.99 | 167.67% (high std) |

**Hyperparameter tuning:**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost (untuned) | 11,929.81 | 15,652.41 | 16.08% |
| XGBoost (tuned) | 11,974.61 | 15,698.90 | 16.87% |
| SARIMA (untuned) | 15,126.15 | 20,098.40 | 20.18% |
| SARIMA (tuned) | 26,205.41 | 34,874.12 | 30.43% |

## Model Selection: Why XGBoost (Untuned)

Linear Regression scored best on the single test split, but performed worst and most erratically across 5-fold cross-validation a strong signal it overfit to that particular train/test boundary rather than generalizing. XGBoost was the most consistent performer across every CV fold, and its performance barely changed under hyperparameter tuning, indicating a stable, low-variance model rather than one whose apparent accuracy depends on a lucky split.

SARIMA's tuned result was dramatically worse than its untuned baseline, most likely because the hyperparameter search overfit to the limited CV folds available for a time series problem.

**Final model:** XGBoost Regressor, `n_estimators=200, max_depth=3, learning_rate=0.05`, retrained on the full historical dataset before deployment.

## Limitations

- Forecasts beyond ~7–10 days are generated **recursively** (each day's prediction feeds the next day's lag features), so error compounds the further out the forecast horizon extends. Short-horizon forecasts are considerably more reliable than long-horizon ones.
- MAPE is computed excluding days with zero actual sales (to avoid divide-by-zero); this is a limitation of MAPE as a metric on sparse/intermittent sales data, not of the model itself.
- The model was evaluated on a single historical period (2015–2018) and would benefit from re-validation on more recent data before any real production use.

## Repository Structure

```
├── app.py                              # Streamlit app
├── model_utils.py                      # Feature-generation logic used at inference time
├── sales_forecast_xgb_model.pkl        # Trained XGBoost model
├── feature_cols.json                   # Feature column list used by the model
├── historical_sales_for_inference.csv  # Historical daily sales, used to seed lag/rolling features
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (include the `.pkl`, `.json`, and `.csv` files — do not gitignore them, the app needs them)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select your repository, branch, and set the main file path to `app.py`
4. Click **Deploy**  the first build installs `requirements.txt` and may take a few minutes
5. Once live, copy the app URL into the "Live app" line at the top of this README

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, Prophet, Streamlit

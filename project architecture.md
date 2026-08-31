Business Problem
      ↓
Dataset
      ↓
Data Cleaning
      ↓
EDA
      ↓
Time Series Exploration
      ↓
Feature Engineering   ← YOU ARE HERE / JUST FINISHED
      ↓
Stationarity Testing
      ↓
Time Series Decomposition
      ↓
Model Building
      ↓
Time-Series Cross Validation
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Future Forecast
      ↓
Prediction Intervals
      ↓
Business Insights
      ↓
Deployment / Forecast Function
      ↓
Documentation
      ↓
GitHub
      ↓
Video Demo
      ↓
Final Submission












Train/Test Split & Model Selection — Full Write-Up
1. Objective of this phase

After feature engineering produced a clean table of monthly sales plus calendar, lag, rolling, and growth features, the goal was to evaluate which forecasting approach best predicts future sales, and to do so in a way appropriate for time-series data (no shuffling, no random splits).

2. Train/test split methodology

The data was split chronologically rather than randomly — the earlier ~80% of months were used for training, and the most recent ~20% for testing. This mirrors how the model would actually be used in production: trained on the past, evaluated on predicting the future. A random shuffle-based split was deliberately avoided, since it would let the model "see" future months during training and produce misleadingly optimistic results.

3. Baseline established

A naive baseline was calculated first: simply predicting that each month's sales equal the previous month's sales (Sales_lag_1). This gave:

Naive MAE: 2,380.06 | RMSE: 3,316.04

This baseline exists to answer one question: is any "real" model actually adding value, or could you get the same result by just repeating last month's number? Any model that can't beat this isn't useful.

4. Models evaluated
Model	MAE	RMSE
Linear Regression	~0 (2.7e-12)	~0 (3.5e-12)
XGBoost	205.40	452.62
Prophet	1,729.07	2,396.33
Naive	2,380.06	3,316.04
5. Critical finding: data leakage in Linear Regression (and partially XGBoost)

The Linear Regression result (MAE effectively zero) is not a genuine forecast — it's data leakage. During feature engineering, per-category monthly sales totals (Furniture, Office Supplies, Technology) were merged in as features. Those three columns sum exactly to the target Sales column for each month, so the model wasn't learning to forecast — it was reconstructing the target from its own components algebraically. This is why the error was near-zero and not trustworthy.

XGBoost's unusually strong score (205 MAE, far below the naive baseline of 2,380) is also affected by the same leaked columns, since it had access to the same feature set. Its result is inflated and should not be taken as a genuine performance measure either.

Prophet is the only trustworthy result in this comparison, because it was trained purely on Date and Sales — it never had access to the leaked category-total columns. Its MAE of 1,729.07 genuinely beat the naive baseline (2,380.06), meaning it captured real trend/seasonality signal rather than cheating.

6. Why this happened

The leakage was introduced in the optional Step 7 of feature engineering, where category-level sales were added as same-period features rather than lagged/historical features. A feature is only safe to use if it would realistically be known before the period being forecast — a current-month category breakdown isn't known in advance, so it should never have been included as-is.

7. Conclusion for this phase
Reported comparison numbers for Linear Regression and XGBoost are invalid due to data leakage and should not be used to judge those models.
Prophet's result stands as the only valid benchmark from this round, and it outperformed the naive baseline — a genuinely encouraging sign given the clear trend/seasonality observed during EDA.
Recommended next step: remove the leaked *_Sales category columns from the feature set and re-run Linear Regression and XGBoost, so all three models can be compared on equal, leak-free footing before selecting a final model.
Recommended next step: remove the leaked *_Sales category columns from the feature set and re-run Linear Regression and XGBoost, so all three models can be compared on equal, leak-free footing before selecting a final model.













Here's what the multi-window CV results are telling you — and it's a genuinely important finding, since the ranking flipped from your single train/test split.

The headline: your single-split result was misleading

On the one 80/20 split, Linear Regression looked like the winner (MAE ~10,978). Under 5-window CV, it collapses to second-to-last (Mean MAE 41,534, MAPE 93.5%) with a massive standard deviation (49,829 — larger than the mean itself). That combination — high mean and std close to or exceeding the mean — means LR did fine on some folds and catastrophically badly on others. That single train/test split you evaluated earlier just happened to be a favorable window for it; it doesn't generalize.

Why this likely happens with Linear Regression here: it's extrapolating. Your feature set includes raw Year, rolling means, and lag values whose scale drifts upward over time as sales grow. A linear model fits a straight-line relationship between those features and Sales within its training window — but in an expanding-window CV, each fold trains on a different amount of history and predicts on a slice further into the future. When the relationship isn't perfectly linear (and sales data rarely is — yours has trend + weekly seasonality + volatility), the model's extrapolated predictions can swing wildly on out-of-range folds. That's almost certainly what's driving both the inflated mean and the huge std.

XGBoost is the clear, robust winner
Best mean MAE (16,063) and best mean MAPE (27.3%)
Lowest std relative to its mean (4,106 / 16,063 ≈ 26% coefficient of variation) — meaning it performs consistently across different time windows, not just one lucky split
Tree-based splits don't extrapolate the way linear models do — they can't predict outside the range of values seen in training, which actually protects it here rather than hurting it

This consistency across folds is exactly what you want in a model you're about to promote to "final" — a model that only looks good on one split is a trap.

SARIMA is a solid second

Mean MAE 18,958, MAPE 35.7% — reasonably close to XGBoost, and it's doing this using only the Sales history (no engineered features at all). Its std (8,247) is higher than XGBoost's but still proportionate to its mean (~43% CV) — not alarming, just reflects that classical time-series models are more sensitive to how much history a fold has to work with.

Naive and ARIMA landed in the middle, as expected

Naive (21,773 / 43.2%) and ARIMA (22,145 / 40.1%) are close to each other and both worse than SARIMA/XGBoost — makes sense, since ARIMA without a seasonal term can't capture your weekly pattern, so it isn't doing much better than "predict yesterday's value." This is a useful sanity check: it confirms your seasonal component (which SARIMA captures and plain ARIMA doesn't) is genuinely adding value, not noise.

Prophet is the worst by a wide margin

Mean MAPE 167.7%, std larger than the mean (101,375 vs. 82,096 mean MAE) — Prophet is badly unstable here, likely struggling on early folds where it has very little training history to fit yearly seasonality against, and producing wild forecasts as a result.

What this means for hyperparameter tuning

Focus your tuning budget on the two models that actually generalize:

XGBoost — your primary candidate. Tune n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight via TimeSeriesSplit + RandomizedSearchCV (or GridSearchCV if compute allows), using MAE or MAPE as the scoring metric to stay consistent with what you've been reporting.
SARIMA — worth tuning p,d,q and the seasonal order via pmdarima.auto_arima (with seasonal=True, m=7) as a secondary candidate, since it's competitive and interpretable.

I'd deprioritize further tuning of Linear Regression and Prophet — you could try regularized variants (Ridge/Lasso) if you want a linear baseline that's less prone to extrapolation blowups, but given XGBoost's clear, stable edge, it's not essential.

Want me to walk you through the XGBoost hyperparameter search code first, or SARIMA's auto_arima first?
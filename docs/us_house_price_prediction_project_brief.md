# US House Price Prediction Using Random Forest

## Project Overview

This project predicts the **quarter-over-quarter direction of US house prices** (increase or decrease) across metro regions using a random forest classifier. It combines macroeconomic indicators from the Federal Reserve (FRED) and financial market data with granular housing metrics from Zillow to build a binary classification model.

The model is evaluated using a custom walk-forward backtesting engine, and iteratively improved by engineering and testing new predictors. This project deliberately scopes its prediction target to **house prices only** — inflation (CPI) is used strictly as an input feature to provide economic context, not as something the model attempts to forecast.

**Key Result:** A backtested random forest model that predicts whether house prices will rise or fall next quarter, with measurable improvement across iteration cycles.

---

## Data Sources & Acquisition

### Federal Reserve Economic Data (FRED)

| Dataset | Description | Expected Frequency |
|---|---|---|
| CPI (Consumer Price Index) | Broad inflation measure — used as a macro feature, not a prediction target | Monthly |
| Rental Vacancy Rate | Percentage of rental housing inventory that is vacant | Quarterly |
| Mortgage Interest Rates | Average rates on residential mortgages (e.g., 30-year fixed) | Weekly/Monthly |

### Zillow Housing Data

| Dataset | Description | Expected Frequency |
|---|---|---|
| ZHVI (Zillow Home Value Index) | Smoothed, seasonally adjusted measure of typical home value by region | Monthly |
| Median Sale Price | Actual median transaction price by metro region | Monthly |

### Yahoo Finance (via yfinance)

| Dataset | Description | Expected Frequency |
|---|---|---|
| Treasury Yields (10-Year, 2-Year, etc.) | US Treasury bond yields — proxy for long-term rate expectations and investor sentiment | Daily |

### Notes on Data Scope
- Zillow data covers multiple US metro regions, enabling cross-regional analysis and prediction
- FRED data is national-level, applied uniformly across all regions as macro context
- Treasury yield data is national-level, sourced via yfinance historical pulls

---

## Data Pipeline

### Step 1: Data Ingestion
- Pull FRED datasets using pandas (CSV download or FRED API)
- Pull Zillow ZHVI and median sale price datasets (CSV download from Zillow Research)
- Pull treasury yield history via yfinance

### Step 2: Frequency Alignment
- Resample all datasets to **quarterly frequency** to match the prediction horizon
- For monthly/weekly/daily data, aggregate to quarterly using appropriate methods (mean, end-of-quarter value, etc.)
- Align all datasets to a common date index

### Step 3: Merge Strategy
- Join FRED macro features and treasury yields to Zillow housing data on the quarterly date index
- FRED and treasury data are national-level and will be broadcast across all metro regions
- Zillow data is region-specific — the merged dataset will have one row per region per quarter

### Step 4: Cleaning
- Handle missing values (forward fill, interpolation, or drop depending on gap size)
- Verify date alignment across sources
- Remove any regions with insufficient historical coverage

---

## Feature Engineering

### Base Features (Raw)
- CPI (quarterly value)
- Rental vacancy rate
- Mortgage interest rate (quarterly average)
- Treasury yields (10-year, 2-year; quarterly average)
- ZHVI (current quarter)
- Median sale price (current quarter)

### Derived Features (Candidates for Iterative Improvement)
- **Lag variables:** Prior quarter values for all base features (t-1, t-2, etc.)
- **Rolling averages:** 2-quarter, 4-quarter moving averages for price and rate features
- **Rate of change:** Quarter-over-quarter percentage change for CPI, mortgage rates, yields
- **Yield spread:** 10-year minus 2-year treasury yield (yield curve indicator)
- **Price-to-rent ratio:** ZHVI divided by implied annual rent (derived from vacancy/price data if available)
- **Mortgage rate momentum:** Direction and magnitude of rate change over trailing quarters

---

## Target Variable

**Binary classification:**
- `1` = House price increased quarter-over-quarter (based on ZHVI or median sale price)
- `0` = House price decreased or stayed flat quarter-over-quarter

The specific price metric used to derive the target (ZHVI vs. median sale price) should be evaluated during development — ZHVI is smoother and less noisy, while median sale price reflects actual transactions.

---

## Modeling Approach

### Why Random Forest
- Handles non-linear relationships between macro indicators and housing prices
- Robust to feature scale differences (no normalization required)
- Built-in feature importance ranking supports iterative feature engineering
- Less prone to overfitting than single decision trees
- Well-supported in scikit-learn with straightforward hyperparameter tuning

### Model Configuration (Starting Point)
- `RandomForestClassifier` from scikit-learn
- Initial hyperparameters: default, with tuning via grid search or randomized search in later iterations
- Class balance check — if price increases vastly outnumber decreases, consider class weighting or resampling

---

## Backtesting & Evaluation

### Walk-Forward Validation
A custom backtesting loop that simulates real-world deployment:

1. Define an initial training window (e.g., first N quarters of data)
2. Train the model on the training window
3. Predict the next quarter (one step ahead)
4. Record the prediction and actual outcome
5. Expand the training window by one quarter and repeat
6. Continue until all available quarters are exhausted

This approach prevents data leakage and reflects how the model would perform if deployed in production — always predicting forward using only historically available data.

### Evaluation Metrics
- **Accuracy:** Overall correct predictions (up/down)
- **Precision / Recall:** Especially important if class distribution is imbalanced
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visual breakdown of prediction vs. actual across all backtested quarters
- **Baseline Comparison:** Compare against a naive model (e.g., "prices always go up") to demonstrate value

---

## Model Improvement

### Iteration Strategy
1. **Baseline model** — Train with raw base features only, establish backtested performance
2. **Add lag features** — Introduce t-1, t-2 quarter lags and re-evaluate
3. **Add rolling averages** — 2Q and 4Q rolling means for price and rate features
4. **Add derived features** — Yield spread, rate of change, momentum indicators
5. **Hyperparameter tuning** — Grid search over n_estimators, max_depth, min_samples_split
6. **Feature selection** — Use feature importance scores to prune low-value predictors

### Tracking
Each iteration should log:
- Features used
- Hyperparameters
- Backtested accuracy, precision, recall, F1
- Notable observations (e.g., "adding yield spread improved recall on downturns by X%")

---

## Technical Stack

| Component | Tool |
|---|---|
| Environment | Google Colab |
| Data manipulation | pandas |
| Financial data | yfinance |
| Modeling & evaluation | scikit-learn |
| Visualization | matplotlib / seaborn (as needed within Colab) |
| Version control | GitHub |

---

## Next Steps / Extensions

- **Regional deep-dive:** Train region-specific models vs. a single national model and compare performance
- **Additional features:** Explore housing starts, building permits, population migration data
- **Alternative models:** Test gradient boosting (XGBoost/LightGBM) or logistic regression as comparison baselines
- **Regression variant:** Predict the magnitude of price change rather than just direction
- **Deployment:** Package as a lightweight API or interactive dashboard for portfolio presentation

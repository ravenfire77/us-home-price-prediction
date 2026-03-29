# US Residential Home Price Direction Prediction

## A Machine Learning Portfolio Project

**Author:** Heath  
**Date:** March 2026  
**Tools:** Python, scikit-learn, pandas, yfinance, Google Colab  
**Model:** Random Forest Classifier with Walk-Forward Backtesting

---

## Executive Summary

This project predicts whether US residential home prices will rise or fall in the next quarter across 493 metropolitan areas, using publicly available macroeconomic data and a random forest classifier. The model was evaluated using a strict walk-forward backtest spanning 73 quarters (2008–2025) to simulate real-world deployment conditions where only historically available data is used for each prediction.

**Final model accuracy: 67.4%** — a 7.4 percentage point improvement over the naive baseline of "always predict up" (60.0%), achieved through iterative feature engineering, hyperparameter tuning, and feature selection. The model correctly identifies 67% of quarterly price declines, providing actionable signal in the direction that matters most for risk management — downside detection.

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Understanding the Data](#understanding-the-data)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [The Prediction Target](#the-prediction-target)
6. [Modeling Approach](#modeling-approach)
7. [Walk-Forward Backtesting](#walk-forward-backtesting)
8. [Iteration Results](#iteration-results)
9. [Final Model Performance](#final-model-performance)
10. [What We Learned](#what-we-learned)
11. [Limitations & Future Work](#limitations--future-work)
12. [Technical Stack](#technical-stack)

---

## Project Motivation

Housing is the largest asset class most Americans hold, and price direction has cascading effects on mortgage risk, consumer spending, and bank lending portfolios. The question this project answers is practical: given the macroeconomic data available today, can we predict whether home prices will go up or down next quarter?

This project deliberately scopes the prediction target to **direction only** (up vs. down), not magnitude. Direction prediction is a binary classification problem — simpler to evaluate, cleaner to backtest, and directly useful for risk flagging. Predicting the exact percentage change is a harder regression problem better suited as a future extension.

---

## Understanding the Data

This section walks through every data source, what it measures, why it matters for housing prices, and what it looks like in practice. Understanding the inputs is essential to interpreting what the model learns.

### Zillow Home Value Index (ZHVI)

**What it is:** ZHVI is Zillow's estimate of the typical home value in a given metro area. It is not a simple average of sale prices — it uses a statistical model that accounts for the mix of homes that sell in any given period, producing a smoother and more representative estimate of the overall market level. Think of it as the answer to "what is a typical home worth in this metro right now?"

**Frequency and coverage:** Monthly values for 894 metropolitan statistical areas (MSAs) across the United States, starting as early as February 1996 for the largest metros. After filtering for data quality (requiring coverage by 2000 and minimum home values above $75,000), we retained 493 metros spanning 47 states.

**What the numbers look like:**

| Metric | Value |
|---|---|
| Minimum ZHVI in dataset | $35,263 |
| Maximum ZHVI in dataset | $1,635,960 |
| Median ZHVI | $145,976 |
| Mean ZHVI | $187,740 |

The gap between mean and median tells you the distribution is right-skewed — a handful of expensive coastal metros (San Jose at $1.57M, San Francisco at $1.11M) pull the average well above the typical metro. At the other end, metros like Forrest City, AR ($79K) and Selma, AL ($85K) represent small, affordable markets.

**Why it matters for prediction:** ZHVI is both the source of our prediction target (did it go up or down this quarter?) and a set of input features (what is the current level, and what has the recent trajectory looked like?). Housing prices are strongly autocorrelated — if prices went up last quarter, they are more likely to go up this quarter than down. This "momentum" effect turns out to be the single most powerful predictor in our model.

**Format:** Zillow distributes this as a wide-format CSV where each row is a metro and each column is a monthly date. We melt this into long format (one row per metro per month) and resample to quarterly frequency using the end-of-quarter value.

### Consumer Price Index (CPI)

**What it is:** The CPI measures the average change in prices paid by urban consumers for a basket of goods and services. It is the US government's primary measure of inflation. A CPI of 326 (the most recent value in our data) means that the basket of goods costs 3.26 times what it cost in the base period (1982–1984 = 100).

**Frequency:** Monthly, from January 1947 to February 2026 (950 observations). We resample to quarterly using the end-of-quarter value.

**Range in our dataset:** 161.80 to 326.03 (mean: 229.05)

**Why it matters for prediction:** Inflation and housing prices are intertwined. Moderate inflation tends to support home values because replacement construction costs rise, and homeowners with fixed-rate mortgages benefit from paying back cheaper dollars. However, high inflation triggers Federal Reserve rate hikes, which raise mortgage rates and cool demand. The CPI gives the model a read on where we are in this cycle.

**Data quality:** One null value was found (October 2025) and filled via linear interpolation between the September and November values. No other issues.

### 30-Year Fixed Mortgage Rate

**What it is:** The average interest rate on a 30-year fixed-rate mortgage in the United States, as reported by Freddie Mac's Primary Mortgage Market Survey. This is the rate that directly determines monthly payments for the most common mortgage product in the US.

**Frequency:** Weekly, from April 1971 to March 2026 (~2,870 observations). We resample to quarterly using the average rate across all weeks in each quarter.

**Range in our dataset:** 2.76% to 8.32% (mean: 5.35%)

**Why it matters for prediction:** Mortgage rates are arguably the most direct lever on housing demand. When rates drop, buyers can afford more house with the same monthly payment, pushing prices up. When rates rise, the opposite happens. The rate journey from 2.65% (January 2021) to 7.79% (October 2023) was the sharpest tightening cycle in modern history — and it's a major reason our model's accuracy dipped in 2022–2025.

**What the model uses:** Not just the raw rate, but also the 4-quarter rolling average (trend), the momentum (current vs. two quarters ago), and the quarter-over-quarter change. These derived features help the model distinguish between "rates are high" and "rates are rising" — two different signals.

### Rental Vacancy Rate

**What it is:** The percentage of rental housing units in the US that are vacant and available for rent, as reported by the Census Bureau's Housing Vacancy Survey. A higher vacancy rate means more available rental supply relative to demand.

**Frequency:** Quarterly, from January 1956 to Q4 2025 (280 observations). Already at our target frequency — no resampling needed, just date alignment.

**Range in our dataset:** 5.60% to 11.10% (mean: 8.17%)

**Why it matters for prediction:** Rental vacancy is a proxy for overall housing supply/demand balance. When vacancy is low, it signals tight housing markets — people who can't find rentals are more likely to buy, supporting home prices. When vacancy is high, there's less pressure to buy, and prices may soften. This feature ranked consistently in the top 5 most important across all model iterations.

**Important nuance:** This is a national-level statistic applied uniformly across all 493 metros. In reality, vacancy rates vary enormously by city. This is one of the limitations of our approach — the model cannot distinguish between a metro with 2% vacancy and one with 15% because they both get the same national number.

### 10-Year US Treasury Yield

**What it is:** The yield (annualized return) on a 10-year US Treasury bond. This is one of the most important benchmarks in global finance — it reflects the market's expectations for future economic growth and inflation over the next decade.

**Frequency:** Daily, from January 1996 to December 2025 (~7,500 observations). Pulled via the Yahoo Finance API (yfinance) using the ticker `^TNX`. Resampled to quarterly using the average yield across all trading days in each quarter.

**Range in our dataset:** 0.64% to 6.46% (mean: 3.47%)

**Why it matters for prediction:** The 10Y yield influences mortgage rates (30-year mortgages are loosely benchmarked to it), reflects investor sentiment about the economy, and competes with housing as an investment. When treasury yields are high, bonds offer an attractive alternative to real estate; when yields are low, investors are pushed toward assets like housing. Three 10Y treasury features ranked in the top 15 most important in our model.

**We also tested the 2-Year Treasury yield and yield spread (10Y minus 2Y):** The yield spread is a well-known recession indicator — when it goes negative (inverted yield curve), a recession has historically followed within 12–18 months. We obtained full 2Y history from FRED (back to 1976) and tested these features. However, they did not improve model accuracy and were removed during feature selection. The information they carry is largely redundant with the 10Y yield and mortgage rate features already in the model.

### Data Not Used (and Why)

**Median Sale Price (Zillow):** We had weekly median sale price data for 203 metros (2008–2026). We chose ZHVI over median sale price for the prediction target because ZHVI covers 4x more metros, has 12 more years of history (critical for walk-forward backtesting), and its smoothing reduces label noise. Median sale price could be added as a feature in future iterations.

**2-Year Treasury Yield and Yield Spread:** Tested and removed. Adding these features diluted model performance — the yield spread information is already captured by the 10Y yield and mortgage rate combination. See the Iteration Results section for details.

---

## Data Pipeline

### How Raw Data Becomes a Modeling-Ready Panel

The raw data arrives in different formats, frequencies, and granularities. The pipeline converts everything into a single "panel" dataset: one row per metro per quarter, with aligned features and a binary prediction target.

**Step 1 — Ingestion and Cleaning:**
All CSV files are loaded and parsed with proper date handling. CPI's single null value (October 2025) is interpolated. ZHVI metros are filtered from 894 to 493 based on data coverage (must have data by 2000) and minimum home value ($75K) to exclude ultra-small markets with noisy price signals.

**Step 2 — Reshaping:**
Zillow's wide-format data (metros as rows, dates as columns) is melted into long format (one row per metro per date) for easier manipulation.

**Step 3 — Frequency Alignment:**
All datasets are resampled to quarterly frequency. Monthly data (CPI, ZHVI) uses end-of-quarter values. Weekly data (mortgage rates) uses the quarterly average. Daily data (treasury yields) uses the quarterly average. Rental vacancy rate is already quarterly and only needs date alignment.

**Step 4 — Merging:**
National-level macro features (CPI, mortgage rate, rental vacancy, treasury yields) are joined to the metro-level ZHVI data on the quarterly date. This means every metro in a given quarter gets the same macro values — the variation across rows within a quarter comes from the different ZHVI levels and trajectories of each metro.

**Step 5 — Feature Engineering:**
Lag features, rolling averages, rate-of-change measures, and momentum indicators are computed within each metro's time series. See the Feature Engineering section for details.

**Step 6 — Target Creation:**
The binary target is created: 1 if ZHVI increased quarter-over-quarter, 0 if it decreased or stayed flat.

**Final Panel:** 55,203 rows (493 metros × ~113 quarters), 35 columns, zero nulls.

---

## Feature Engineering

Feature engineering transforms raw data into signals the model can learn from. Starting with 4 raw data sources, we engineered 25 candidate features organized into categories.

### Base Features (5)
The raw quarterly values: ZHVI, CPI, mortgage rate, rental vacancy rate, and 10-year treasury yield. These represent "what is the current state of the world?"

### Lag Features (8)
Prior quarter (t-1) and two-quarters-ago (t-2) values for ZHVI, CPI, mortgage rate, rental vacancy rate, and 10Y treasury yield. These let the model see "what happened recently?" A key design decision: lagged ZHVI change (last quarter's price movement) is the single most important feature because housing prices exhibit strong momentum.

### Rolling Averages (4)
2-quarter and 4-quarter moving averages for ZHVI and mortgage rates. These smooth out noise and represent "what is the trend?" The 4-quarter rolling mortgage rate captures whether we're in a sustained high-rate or low-rate environment, as opposed to a single-quarter spike.

### Rate of Change (3)
Quarter-over-quarter percentage change for CPI, mortgage rates, and treasury yields. These capture "how fast are things changing?" The distinction matters: a mortgage rate of 7% that is falling sends a different signal than 7% that is rising.

### Momentum (1)
Mortgage rate momentum: the difference between the current rate and the rate two quarters ago. This feature captures the direction and speed of rate movement over a longer window than single-quarter change.

### Feature Selection Results

After testing all 25 features, 8 were removed for the final model because they added noise:

**Removed:** `mortgage_rate` (raw), `mortgage_rate_qoq_pct`, `rental_vacancy_rate_lag1`, `cpi_qoq_pct`, `treasury_10y_roll4q`, `treasury_10y_lag2`, `mortgage_rate_roll2q`, `mortgage_rate_lag2`

**Retained (17 features):** The model performs better with 17 features (67.4% accuracy) than with all 25 (65.4%). This is a common finding in machine learning — redundant or low-signal features give the model more ways to overfit without adding predictive power.

---

## The Prediction Target

**Binary classification:** For each metro in each quarter, the target is:
- `1` = ZHVI increased compared to the prior quarter (price went up)
- `0` = ZHVI decreased or stayed flat (price went down or was unchanged)

### Class Balance

The target is imbalanced — prices go up more often than they go down:

| Period | % Quarters with Price Increases | Context |
|---|---|---|
| 1998–2006 | 74.4% | Pre-crisis boom |
| 2007–2011 | 34.9% | Financial crisis and aftermath |
| 2012–2019 | 68.8% | Recovery and sustained growth |
| 2020–2025 | 67.4% | Pandemic boom, then rate shock |
| **Overall** | **64.0%** | |

This imbalance is not a data problem — it reflects reality. US housing prices have a long-term upward bias driven by population growth, inflation, and constrained land supply. The model uses `class_weight='balanced'` to prevent it from simply learning to always predict "up." The naive strategy of always predicting "up" achieves 60.0% accuracy in our backtest window — this is the floor the model must beat to demonstrate value.

---

## Modeling Approach

### Why Random Forest

Random forest was chosen for several reasons aligned with this problem's characteristics. It handles non-linear relationships between macro indicators and housing prices without requiring the analyst to specify the functional form. It is robust to features measured on different scales (ZHVI in hundreds of thousands vs. vacancy rates in single digits) without normalization. It provides built-in feature importance rankings that directly support the iterative improvement strategy. And it is less prone to overfitting than a single decision tree because it averages predictions across many trees.

### Final Model Configuration

| Parameter | Value | What It Controls |
|---|---|---|
| n_estimators | 200 | Number of decision trees in the forest |
| max_depth | None (unconstrained) | How deep each tree can grow |
| min_samples_split | 10 | Minimum samples needed to create a new branch |
| min_samples_leaf | 1 | Minimum samples in any leaf node |
| class_weight | balanced | Adjusts for the 64/36 class imbalance |
| n_jobs | -1 | Uses all available CPU cores |

The hyperparameter tuning process tested 24 configurations. The key finding was that **unconstrained tree depth wins** — every top-performing configuration used `max_depth=None`. With 493 metros and 17 features, the trees need deep splits to capture the interactions between macro conditions and regional price dynamics.

---

## Walk-Forward Backtesting

### Why Not a Simple Train/Test Split?

In a standard machine learning project, you might randomly split data into training and test sets. For time-series problems, this creates a fatal flaw: the model trains on future data and predicts the past. If the model sees 2020 CPI values during training and then "predicts" 2018 prices, it has access to information that would not have been available in 2018. The results would be artificially inflated.

### How Walk-Forward Validation Works

Walk-forward validation simulates real-world deployment by enforcing strict temporal ordering:

1. **Train on the first 40 quarters** (Q4 1997 through Q3 2007) — approximately 10 years of history
2. **Predict the next quarter** (Q4 2007) across all 493 metros — record predictions and actual outcomes
3. **Expand the training window** to include Q4 2007
4. **Predict Q1 2008** — repeat
5. **Continue until Q4 2025** — 73 prediction quarters total

At every step, the model only sees data that would have been historically available. This is identical to how the model would perform if deployed in production — making forward predictions using only the past.

### Backtest Scale

- **Training window:** Expanding from 40 quarters (~19,700 rows) to 112 quarters (~55,200 rows)
- **Prediction quarters:** 73 (Q4 2007 through Q4 2025)
- **Total predictions:** 35,989 (73 quarters × ~493 metros per quarter)
- **Compute time:** ~55 minutes for a full grid search run on Google Colab Pro (8 CPU cores)

---

## Iteration Results

The project followed a structured six-step iteration strategy. Each step's contribution is measured against the walk-forward backtest.

### Step 1: Baseline Model (20 features, default params)
- **Accuracy:** 62.4% | **F1:** 0.678
- Established the floor. Used base features plus lags, rolling averages, and rate-of-change features for CPI and mortgage rates.

### Step 2: Add Treasury Yields (25 features)
- **Accuracy:** 62.7% (+0.3pp) | **F1:** 0.679
- Three 10Y treasury features ranked in the top 15 by importance. Modest but positive lift.

### Step 3: Test Yield Spread (30 features)
- **Accuracy:** 62.3% (-0.1pp) | **F1:** 0.675
- Adding 2Y yield and yield spread features *hurt* performance. The information was redundant with the 10Y yield and mortgage rate features already in the model. These features were dropped.

### Step 4: Hyperparameter Tuning (25 features, tuned params)
- **Accuracy:** 65.4% (+2.6pp over V2) | **F1:** 0.712
- Tested 24 parameter combinations. The key finding: removing the `max_depth=8` constraint and allowing unconstrained tree growth yielded the largest single improvement.

### Step 5: Feature Selection (17 features, tuned params)
- **Accuracy:** 67.4% (+2.0pp over tuned, +4.7pp over V2) | **F1:** 0.723
- Removing the 8 least important features improved accuracy. The model performs better with fewer, more informative features.

### Cumulative Improvement

| Model | Accuracy | vs. Naive (+/-) | vs. Baseline (+/-) |
|---|---|---|---|
| Naive (always up) | 60.0% | — | — |
| Baseline (Step 1) | 62.4% | +2.4pp | — |
| + Treasury (Step 2) | 62.7% | +2.7pp | +0.3pp |
| + Tuning (Step 4) | 65.4% | +5.4pp | +3.0pp |
| **+ Pruning (Step 5)** | **67.4%** | **+7.4pp** | **+5.0pp** |

---

## Final Model Performance

### Overall Metrics

| Metric | Final Model | Naive Baseline |
|---|---|---|
| **Accuracy** | 67.4% | 60.0% |
| **Precision** | 72.3% | 60.0% |
| **Recall** | 72.0% | 100% |
| **F1 Score** | 0.723 | 0.750 |

### Accuracy by Period

| Period | Accuracy | Quarters | Context |
|---|---|---|---|
| 2007–2011 | ~68% | 20 | Financial crisis — model navigates the downturn |
| 2012–2016 | ~65% | 20 | Slow recovery, mixed regional signals |
| 2017–2021 | ~77% | 20 | Strong trend environment — model's best period |
| 2022–2025 | ~56% | 16 | Rate shock, unprecedented macro conditions |

### Feature Importances (Final 17-Feature Model)

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | zhvi_change_pct_lag1 | 0.134 | Price momentum |
| 2 | zhvi | 0.072 | Current price level |
| 3 | rental_vacancy_rate | 0.065 | Housing supply |
| 4 | zhvi_lag2 | 0.061 | Price history |
| 5 | zhvi_lag1 | 0.061 | Price history |
| 6 | zhvi_roll2q | 0.060 | Price trend |
| 7 | zhvi_roll4q | 0.058 | Price trend |
| 8 | treasury_10y_lag1 | 0.044 | Rate environment |
| 9 | cpi | 0.037 | Inflation |
| 10 | mortgage_rate_roll4q | 0.032 | Rate trend |
| 11 | treasury_10y_qoq_pct | 0.031 | Rate momentum |
| 12 | cpi_lag2 | 0.031 | Inflation history |
| 13 | rental_vacancy_rate_lag2 | 0.031 | Supply history |
| 14 | mortgage_rate_momentum_2q | 0.030 | Rate momentum |
| 15 | treasury_10y | 0.031 | Rate level |
| 16 | mortgage_rate_lag1 | 0.030 | Rate history |
| 17 | cpi_lag1 | 0.029 | Inflation history |

The importance distribution reveals that the model relies on three pillars: price momentum (features 1–7, ~52% of total importance), rate environment (features 8, 10–11, 14–16, ~20%), and macro context — inflation and housing supply (features 9, 12–13, 17, ~13%).

---

## What We Learned

### 1. Housing prices are predictable — within limits

A model using only publicly available macro data can beat both coin-flip and naive baselines by a meaningful margin. The 67.4% accuracy represents genuine predictive signal, not luck — it was validated across 73 quarters and 493 metros with strict temporal separation between training and test data.

### 2. Momentum is the dominant signal

Last quarter's price change direction is the single best predictor of next quarter's direction. This is consistent with well-documented housing market dynamics: transaction friction (it takes months to buy or sell), mortgage rate lock-in effects, and buyer/seller psychology all create persistence in price trends. The top 7 features by importance are all ZHVI-derived.

### 3. What you leave out matters as much as what you include

The model improved from 65.4% to 67.4% accuracy by removing 8 low-importance features. This is a counter-intuitive but well-established finding in machine learning: redundant features give the model more opportunities to fit noise in the training data, which hurts generalization. The yield spread, raw mortgage rate, and several lag/change features were all dropped.

### 4. The model fails when history stops being a guide

Accuracy dropped to ~56% in 2022–2025, the period of the Fed's most aggressive rate hiking cycle in 40 years. The model had never seen training data where rates moved this fast from this low a starting point. This is not a bug — it is a fundamental limitation of any model that learns from historical patterns. Structural breaks in the economy will always degrade statistical models.

### 5. National macro features have a ceiling

All macro features in this model (CPI, mortgage rates, vacancy rates, treasury yields) are national-level data broadcast across every metro. In reality, San Jose and Selma respond very differently to the same interest rate change. The model cannot capture this regional heterogeneity with national-level inputs alone. Adding metro-specific features (local employment, building permits, population flows) would likely push accuracy higher.

---

## Limitations & Future Work

### Current Limitations
- **National-level macro features only:** The model cannot distinguish regional economic conditions. A metro with booming tech employment and one with a declining manufacturing base receive identical feature values.
- **Quarterly granularity:** Monthly or even weekly prediction horizons might capture faster-moving dynamics, but would require more granular features.
- **No regime detection:** The model has no mechanism to detect when it has entered an unprecedented macro environment and should reduce confidence in its predictions.
- **Single model architecture:** Only random forest was tested. The brief's "Next Steps" section identifies gradient boosting and logistic regression as comparison baselines.

### Planned Extensions
- **Regional deep-dive:** Train metro-specific models vs. the national pooled model and compare performance. Do large metros benefit from their own model?
- **Alternative models:** Test XGBoost/LightGBM (gradient boosting) and logistic regression as comparison baselines. The hyperparameter tuning infrastructure built for this project transfers directly.
- **Additional features:** Housing starts, building permits, and population migration data would add supply-side and demand-side signals the current model lacks.
- **Regression variant:** Predict the magnitude of price change (what percentage will prices move?) rather than just direction.
- **Deployment:** Package the final model as an interactive dashboard or lightweight API for portfolio demonstration.

---

## Technical Stack

| Component | Tool | Purpose |
|---|---|---|
| Environment | Google Colab Pro | Cloud-based Python execution with 8 CPU cores |
| Data manipulation | pandas | Data cleaning, reshaping, feature engineering |
| Financial data | yfinance | Treasury yield data from Yahoo Finance |
| FRED data | CSV download | CPI, mortgage rates, rental vacancy |
| Zillow data | CSV download | ZHVI, median sale price |
| Modeling | scikit-learn | RandomForestClassifier, metrics, evaluation |
| Visualization | matplotlib | Charts, confusion matrices, importance plots |
| Version control | GitHub | Repository and project documentation |

---

## Repository Structure

```
us-home-price-prediction/
├── README.md                    # Project overview (this document)
├── data/
│   ├── raw/                     # Original CSVs (FRED, Zillow, DGS2)
│   └── processed/               # panel_features_v2.csv
├── notebooks/
│   └── full_pipeline.py         # Complete Colab pipeline script
├── results/
│   ├── grid_search_results.csv  # All 24 hyperparameter configs
│   ├── tuning_results.png       # Visualization of tuning/selection
│   └── backtest_report.png      # Baseline backtest visualization
└── docs/
    ├── project_brief.md         # Original project specification
    └── final_report.md          # This report
```

---

## How to Reproduce

1. Clone the repository
2. Download source data:
   - ZHVI: [Zillow Research](https://www.zillow.com/research/data/) → Home Values → Metro
   - CPI: [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL)
   - Mortgage Rate: [FRED MORTGAGE30US](https://fred.stlouisfed.org/series/MORTGAGE30US)
   - Rental Vacancy: [FRED RRVRUSQ156N](https://fred.stlouisfed.org/series/RRVRUSQ156N)
   - 2Y Treasury: [FRED DGS2](https://fred.stlouisfed.org/series/DGS2) (set date range to Max before downloading)
3. Open `notebooks/full_pipeline.py` in Google Colab
4. Set runtime to CPU + High-RAM
5. Upload `panel_features.csv` and `DGS2.csv` when prompted
6. Run — results will auto-download on completion

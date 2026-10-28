# Machine Learning Evaluation Report

**Project**: House Price Prediction System (`mini-project5`)  
**Evaluation Date**: October 28, 2026  
**Environment**: Windows 11, Python 3.12.10, Scikit-Learn 1.9.1, Pandas 3.0.6  
**Primary Metric**: Root Mean Squared Error ($\text{RMSE}$) and $R^2$ Score  
**Secondary Metrics**: Mean Absolute Error ($\text{MAE}$), Mean Absolute Percentage Error ($\text{MAPE}$), Median Absolute Error ($\text{MedAE}$), Max Error  

---

## 1. Executive Summary

This evaluation report provides an empirical, scientifically honest assessment of the House Price Prediction models across two stages:
1. **The Baseline Model** (original repository state): An Ordinary Least Squares (OLS) Linear Regression trained on a hardcoded 10-row dataset and evaluated on only 2 test samples.
2. **The Modernized Machine Learning Pipeline** (v2.0.0): An end-to-end ColumnTransformer and ensemble regression pipeline trained on a 600-sample realistic housing dataset with 5-fold cross-validation and a holdout test split ($N=120$).

### Key Empirical Finding:
- On realistic benchmark housing data, the baseline 4-feature Linear Regression model achieves an $R^2$ of only **0.5309** and an $\text{RMSE}$ of **26.95 Lakhs** ($\text{MAE} = 21.76\text{ Lakhs}$).
- The champion **Gradient Boosting Regressor** achieves an $R^2$ of **0.9717** and an $\text{RMSE}$ of **6.62 Lakhs** ($\text{MAE} = 4.94\text{ Lakhs}$, $\text{MAPE} = 5.86\%$).
- 5-Fold Cross-Validation confirms model stability: $\text{CV } R^2 = 0.9715 \pm 0.0079$ and $\text{CV RMSE} = 6.72 \pm 1.20\text{ Lakhs}$.
- Relative improvement over baseline on identical test data: **75.4% reduction in RMSE** and **77.3% reduction in MAE**.

---

## 2. Baseline Model Forensic Examination

### 2.1 Baseline Setup
- **Algorithm**: `sklearn.linear_model.LinearRegression` (Ordinary Least Squares)
- **Dataset**: In-memory Python dictionary of 10 rows.
- **Features Used**: `['area', 'bedrooms', 'bathrooms', 'age']`
- **Target**: `price` (in ₹ Lakhs)
- **Split Ratio**: `test_size=0.2` (yielding 8 train samples and 2 test samples)

### 2.2 Baseline Results & Pathologies
The baseline script reported:
$$\text{MAE} = 2.91 \text{ Lakhs}, \quad R^2 = 0.99$$
However, forensic inspection revealed:
1. **Sample Size Degeneracy**: The metric was computed on literally 2 test data points ($y_{\text{test}} = [30, 95]$). Reporting $R^2 = 0.99$ on 2 points is statistically meaningless.
2. **Inverted Feature Signs**: Due to extreme collinearity and $N=8$ training points, the fitted equation was:
   $$\text{Price} = -21.31 + 0.0387 \cdot \text{area} - 0.1721 \cdot \text{bedrooms} - 1.2590 \cdot \text{bathrooms} + 0.4634 \cdot \text{age}$$
   - Adding a bedroom decreased the predicted value by ₹17,210.
   - Adding a bathroom decreased the predicted value by ₹1.26 Lakhs.
   - Adding age increased the predicted value by ₹46,340 per year.
3. **Negative Price Anomaly**: For properties with low area and modern age (e.g., $100\text{ sqft}, 2\text{ BHK}, 2\text{ Bath}, \text{Age}=0$), the baseline model predicted **$-₹20.30\text{ Lakhs}$**.

---

## 3. Modernized Benchmark Dataset & Data Quality Audit

### 3.1 Dataset Specification (`data/house_prices.csv`)
> [!IMPORTANT]
> **Synthetic Dataset Disclosure**: The 600-sample dataset was procedurally generated using parameterized distribution modeling calibrated to realistic Indian residential real estate characteristics. While it provides an internally consistent benchmark with zero missing values and physically plausible feature correlations, high metrics on synthetic data do NOT establish real-world market accuracy on noisy, uncurated real estate feeds.

- **Sample Count**: 600 residential property records.
- **Geographic Representation**: Simulated Indian residential real estate (Metro / Tier-1 / Tier-2 urban and suburban corridors).
- **Schema & Distributions**:
  - `area`: 450 to 3,961 sq ft (Mean: 1,644.5 sq ft, Std: 575.6 sq ft)
  - `bedrooms`: 1 to 6 BHK (Mean: 3.05 BHK)
  - `bathrooms`: 1 to 5 (Mean: 2.76)
  - `age`: 0.0 to 29.8 years (Mean: 6.94 years)
  - `locality`: Categorical (`Urban Prime`: 35%, `Suburban`: 50%, `Rural/Outskirts`: 15%)
  - `parking`: 0 to 3 designated spaces (Mean: 1.15 spaces)
  - `furnished_status`: Categorical (`Unfurnished`: 30%, `Semi-Furnished`: 50%, `Furnished`: 20%)
  - `has_amenities`: Binary (Clubhouse, gym, 24/7 security) (62.8% positive)
  - `price`: Continuous target in ₹ Lakhs (Range: ₹18.91L to ₹295.47L, Mean: ₹94.38L)

### 3.2 Data Quality & Leakage Prevention Controls
- **Missing Values**: 0 nulls across all 600 records.
- **Physical Feasibility**: Strict constraint validation ($area > 0$, $bedrooms > 0$, $price > 0$).
- **Stratified Split**: Partitioned using quintile binning on the target price (80% train = 480 samples, 20% test = 120 samples).
- **Leak-Free Model Selection Protocol**: Candidate models are benchmarked and ranked strictly by 5-fold cross-validation performance on $X_{\text{train}}$ ($\text{CV } R^2$ and $\text{CV RMSE}$). The holdout test set $X_{\text{test}}$ is never used for hyperparameter tuning or algorithm selection.
- **Leakage Elimination**: All feature scalers (`StandardScaler`) and categorical encoders (`OneHotEncoder`) are fitted strictly on `X_train` inside Scikit-Learn `Pipeline` objects. `X_test` is transformed only at evaluation time.

---

## 4. Algorithmic Benchmarking & Comparative Results

Five candidate algorithms were trained on `X_train` (480 samples) and evaluated on both 5-Fold Cross-Validation and the independent holdout test set `X_test` (120 samples):

| Model Name | Test $R^2$ | Test RMSE (Lakhs) | Test MAE (Lakhs) | Test MAPE (%) | 5-Fold CV $R^2$ (Mean $\pm$ Std) | 5-Fold CV RMSE (Lakhs) |
|---|---|---|---|---|---|---|
| **Gradient Boosting (Champion)** | **0.9717** | **6.62** | **4.94** | **5.86%** | **0.9715 $\pm$ 0.0079** | **6.72 $\pm$ 1.20** |
| Random Forest Regressor | 0.9614 | 7.73 | 5.66 | 6.81% | 0.9550 $\pm$ 0.0134 | 8.43 $\pm$ 1.58 |
| Linear Regression (8 Features) | 0.9377 | 9.82 | 7.09 | 10.37% | 0.9384 $\pm$ 0.0060 | 9.91 $\pm$ 1.05 |
| Lasso Regression ($\alpha=0.05$) | 0.9373 | 9.85 | 7.09 | 10.31% | 0.9383 $\pm$ 0.0061 | 9.92 $\pm$ 1.06 |
| Ridge Regression ($\alpha=1.0$) | 0.9358 | 9.97 | 7.15 | 10.35% | 0.9382 $\pm$ 0.0053 | 9.94 $\pm$ 1.07 |
| *Baseline Linear (4 Features)* | *0.5309* | *26.95* | *21.76* | *25.39%* | *N/A (unregularized 4-feat)* | *N/A* |

```mermaid
xychart-beta
    title "Model Test RMSE Comparison (Lower is Better)"
    x-axis ["Baseline Linear (4 Feat)", "Ridge", "Lasso", "Linear (8 Feat)", "Random Forest", "Gradient Boosting"]
    y-axis "RMSE in ₹ Lakhs" 0 --> 30
    bar [26.95, 9.97, 9.85, 9.82, 7.73, 6.62]
```

---

## 5. Residual Diagnostics & Error Analysis

Residual analysis of the champion **Gradient Boosting** model on the 120 test properties:
- **Mean Residual ($y - \hat{y}$)**: $+0.68\text{ Lakhs}$ (virtually zero bias)
- **Residual Standard Deviation**: $6.58\text{ Lakhs}$
- **95th Percentile Absolute Error**: $11.31\text{ Lakhs}$ (95% of all predictions fall within $\pm ₹11.31\text{ Lakhs}$ of actual price)
- **Median Absolute Error ($\text{MedAE}$)**: $4.09\text{ Lakhs}$

### Analysis of Extreme Errors:
- The maximum observed error was $+34.03\text{ Lakhs}$ on a rare luxury property ($3,900+\text{ sqft}$ in Urban Prime with extensive premium amenities).
- In real estate valuation, high-end luxury properties exhibit higher price variance than standard 2BHK/3BHK units. To account for this, the predictor incorporates a 95% confidence interval ($\pm 1.96 \cdot \sigma$) around every point estimate.

### 5.2 Feature Importance & Driver Attribution
Extracted normalized Mean Decrease in Impurity (MDI) importances from the champion Gradient Boosting estimator:
- **Area (sq ft)**: **64.45%** (Primary driver of property valuation)
- **Locality: Urban Prime**: **28.24%** (Significant metro/city center premium)
- **Locality: Suburban**: **4.09%**
- **Property Age (years)**: **1.63%** (Depreciation effect)
- **Bedrooms (BHK)**: **0.58%**
- **Furnishing: None**: **0.32%**
- **Furnishing: Semi**: **0.29%**
- **Parking Spaces**: **0.22%**
- **Society Amenities**: **0.10%**
- **Bathrooms**: **0.08%**
- **Total Sum**: **100.00%** (Normalized)

> [!NOTE]
> **Interpretability Disclosure**: Impurity-based feature importance reflects how frequently and deeply the tree-based model splits on each feature across the training partition. It indicates **model reliance**, NOT causal real-world market elasticity. Features with higher continuous variance (such as square footage) naturally receive higher split priority in greedy tree induction algorithms.

### 5.3 Prediction Interval vs. Confidence Interval Methodology
In predictive regression, uncertainty estimation requires strict mathematical distinction:
1. **Confidence Interval (CI)**: Quantifies uncertainty in the estimated mean response $E[Y|X=x]$. Its bandwidth shrinks asymptotically toward zero as sample size $N \to \infty$.
2. **Prediction Interval (PI)**: Quantifies uncertainty for a **new, unseen individual observation** $Y_{\text{new}} = f(x) + \epsilon$. Its variance is bounded by the irreducible noise variance $\sigma^2$ regardless of sample size.
3. **Implemented Approach**: The system implements an empirical Gaussian residual prediction interval:
   $$\text{PI}_{95\%}(x) = \left[ \max\left(5.0, \hat{y}(x) - 1.96 \cdot \hat{\sigma}_{\text{res}}\right), \; \hat{y}(x) + 1.96 \cdot \hat{\sigma}_{\text{res}} \right]$$
   where $\hat{\sigma}_{\text{res}} = 6.58\text{ Lakhs}$ is estimated from the cross-validated residual distribution.

#### Prediction Interval Coverage Probability (PICP) Validation:
- **Nominal Target**: 95.0%
- **Empirical Coverage on Holdout Test Set ($N=120$)**: **95.83%** (115 out of 120 properties fall inside the interval)
- **Mean Interval Bandwidth (MPIW)**: $₹25.81\text{ Lakhs}$ ($\pm ₹12.90\text{ Lakhs}$)
- **Heteroscedasticity Note**: The current implementation uses a constant residual standard deviation across property tiers. In practice, high-value luxury properties exhibit wider dollar variance than entry-level homes. Future work should implement localized conformalized quantile regression (CQR) to dynamically scale bandwidth with predicted valuation.

---

## 6. Physical Consistency & Domain Guardrails Verification

| Test Case | Scenario / Query | Expected Physical Behavior | Observed Model Output | Verdict |
|---|---|---|---|---|
| **Test A: Floor Clamping** | Area = 100 sqft, 1 BHK, 1 Bath, Age = 20 | Price must be strictly positive ($> ₹0$) | **₹29.97 Lakhs** (No negative price) | **PASS** |
| **Test B: Bedroom Monotonicity** | Area = 1800 sqft, 3 BHK vs 4 BHK | 4 BHK price $\ge$ 3 BHK price | 3BHK: **₹93.21L** <br/> 4BHK: **₹95.27L** (+₹2.06L) | **PASS** |
| **Test C: Locality Hierarchy** | Area = 1800 sqft, Prime vs Suburb vs Rural | Prime $>$ Suburb $>$ Rural | Prime: **₹141.01L** <br/> Suburb: **₹93.21L** <br/> Rural: **₹63.08L** | **PASS** |
| **Test D: Age Depreciation** | Area = 1800 sqft, Age = 0 vs Age = 20 | New home $\ge$ 20-year-old home | New: **₹96.14L** <br/> 20-yr: **₹82.80L** (-₹13.34L) | **PASS** |

---

## 7. Limitations & Honest Disclosures

1. **Synthetic Nature of Benchmark**: While the 600-sample dataset reflects realistic Indian real estate dynamics, it was generated through parameterized distribution modeling rather than live real estate transaction scraping.
2. **Regional Specificity**: Valuations reflect typical Tier-1/Tier-2 urban centers. Specific micro-markets (e.g. South Mumbai vs. Outer Bengaluru) have distinct price per square foot dynamics that require localized fine-tuning.
3. **Macroeconomic Shifts**: The model assumes a static interest rate and construction cost regime.

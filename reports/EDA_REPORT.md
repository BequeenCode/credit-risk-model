# Exploratory Data Analysis Report
## Credit Risk Model Dataset

**Date:** May 30, 2026  
**Dataset:** Credit Risk Dataset  
**Total Records:** 5,000  
**Total Features:** 14 (11 numerical, 3 categorical)

---

## Executive Summary

This report presents a comprehensive exploratory data analysis of the credit risk dataset. The analysis reveals key data characteristics, patterns, and quality issues that will inform feature engineering and model development strategies.

---

## 1. Dataset Overview

### Structure
- **Rows:** 5,000 loan records
- **Columns:** 14 features
- **Numerical Features (11):** 
  - loan_id, age, income, loan_amount, loan_term, interest_rate
  - employment_years, num_accounts, num_delinquencies, credit_score, default
- **Categorical Features (3):** 
  - employment_type, home_ownership, loan_purpose

### Data Types
All features have been properly typed with appropriate data types for analysis.

---

## 2. Summary Statistics

### Key Findings

| Feature | Mean | Std Dev | Min | Max | Skew |
|---------|------|---------|-----|-----|------|
| **Age** | 44.83 | 14.43 | 18 | 103 | 0.19 |
| **Income** | $59,971 | $24,742 | $10,000 | $148,226 | 0.15 |
| **Loan Amount** | $151,632 | $72,934 | $5,000 | $407,168 | 0.15 |
| **Interest Rate** | 5.53% | 2.48% | 0.50% | 15.02% | 0.15 |
| **Credit Score** | 647.35 | 98.73 | 300 | 850 | -0.12 |
| **Employment Years** | 7.68 | 4.70 | 0 | 25 | 0.29 |
| **Delinquencies** | 0.86 | 1.13 | 0 | 3 | 0.87 |

### Observations
- Age distribution is relatively normal with slight right skew
- Income and loan amounts show consistent distributions
- Credit scores are reasonably spread across the spectrum
- Employment experience has moderate skewness
- Number of delinquencies is right-skewed (more borrowers with 0-1 delinquencies)

---

## 3. Distribution Analysis

### Numerical Features
- Most features exhibit approximately **normal distributions**
- Low to moderate skewness across the board (only num_delinquencies shows notable skew at 0.87)
- No extreme bimodal distributions observed

### Categorical Features
- **Employment Type:** Balanced distribution across Employed, Self-Employed, Unemployed
- **Home Ownership:** Well-distributed among Rent, Own, Mortgage
- **Loan Purpose:** Diverse purposes including Debt Consolidation, Home Improvement, Business, Auto, Education

---

## 4. Correlation Analysis

### Top Feature Correlations with Default

| Feature | Correlation |
|---------|-------------|
| num_delinquencies | 0.047 |
| interest_rate | 0.037 |
| income | 0.016 |
| loan_term | 0.006 |
| credit_score | -0.011 |
| loan_amount | -0.018 |

### Key Observations
- **Weak correlations overall** indicate complex relationships requiring sophisticated models
- **Delinquency history** is the strongest predictor (positive correlation with default)
- **Interest rates** show positive correlation (higher rates → higher default risk)
- **Credit score** shows slight negative correlation (higher scores → lower default risk)
- **Income and loan amount** show minimal direct correlation with default

---

## 5. Data Quality Issues

### Missing Values

| Column | Count | Percentage |
|--------|-------|-----------|
| employment_years | 30 | 0.6% |
| num_delinquencies | 20 | 0.4% |
| **Total** | **50** | **0.2%** |

**Impact:** Minimal missing data (< 1% of total entries)  
**Recommended Action:** Mean/median imputation for employment_years, mode imputation for num_delinquencies

---

## 6. Outlier Analysis

### Outliers Detected

| Feature | Count | Percentage | Notes |
|---------|-------|-----------|-------|
| default | 534 | 10.68% | Class itself (treated separately) |
| age | 28 | 0.56% | Very young/old borrowers |
| loan_amount | 19 | 0.38% | Very large/small loans |
| income | 17 | 0.34% | Extreme income values |
| interest_rate | 17 | 0.34% | Very high/low rates |
| credit_score | 16 | 0.32% | Boundary scores |
| employment_years | 12 | 0.24% | Very short/long tenure |

**Overall:** 643 outlier instances across all numerical features (~2% of data)

**Recommendations:**
- Investigate domain context before removing outliers
- Consider robust scaling or transformation
- May represent legitimate high-risk borrowers

---

## 7. Target Variable Analysis

### Class Distribution
- **Non-Default (0):** 4,466 records (89.3%)
- **Default (1):** 534 records (10.68%)

### Class Imbalance
**Severity:** **SIGNIFICANT** (10.68% default rate)

**Implications:**
- Baseline accuracy if predicting all non-default: 89.3%
- Model must focus on identifying the minority class
- Standard accuracy metrics insufficient

**Recommended Solutions:**
1. **SMOTE (Synthetic Minority Oversampling)** - Generate synthetic default samples
2. **Class Weights** - Increase penalty for misclassifying defaults
3. **Stratified Sampling** - Maintain class proportions in train/test splits
4. **Ensemble Methods** - Use balanced bagging or boosting techniques
5. **Adjusted Threshold** - Lower decision threshold for default class

---

## Key Insights & Recommendations

### 🎯 **Insight 1: Weak Direct Feature Correlations**
The relatively weak correlations between individual features and default suggest that default risk is determined by complex interactions between multiple factors. This indicates:
- **Recommendation:** Employ advanced models (XGBoost, LightGBM, Neural Networks) that can capture non-linear relationships
- **Action:** Create interaction features and polynomial terms during feature engineering

### 📊 **Insight 2: Significant Class Imbalance**
With only 10.68% default rate, standard models will struggle to identify defaulters:
- **Recommendation:** Implement SMOTE or adjust class weights during model training
- **Action:** Use F1-score, Precision-Recall curves, and ROC-AUC instead of accuracy

### 🔍 **Insight 3: Data Quality is Excellent**
Less than 1% missing values and minimal data issues:
- **Recommendation:** Straightforward imputation strategies sufficient
- **Action:** Use mean imputation for employment_years, proceed with analysis

### 📈 **Insight 4: Presence of Outliers Requires Investigation**
~2% of records contain outliers, particularly in loan amounts and interest rates:
- **Recommendation:** Investigate outliers for data entry errors vs. legitimate high-risk profiles
- **Action:** Consider separate models for different risk segments

### ✏️ **Insight 5: Limited Transformation Needed**
Only 1 feature (num_delinquencies) shows high skewness; most distributions are near-normal:
- **Recommendation:** Apply log transformation only to highly skewed features
- **Action:** Prioritize feature engineering over mathematical transformations

---

## Feature Engineering Recommendations

Based on the EDA, the following feature engineering approaches are recommended:

### 1. **Interaction Features**
- `credit_score × employment_years` - Stability premium
- `income / loan_amount` - Debt-to-income ratio
- `num_delinquencies × credit_score` - Risk multiplication

### 2. **Aggregate Features**
- `accounts_per_year = num_accounts / employment_years` - Account activity rate
- `age_group` - Categorical binning of age
- `income_category` - Income brackets

### 3. **Trend Features**
- `recent_delinquency` - Boolean flag if has delinquencies
- `loan_utilization = loan_amount / income`

### 4. **Encoded Features**
- One-hot encode employment_type, home_ownership, loan_purpose
- Consider target encoding for categorical features

---

## Next Steps

### Phase 1: Data Preprocessing
- [ ] Impute missing values (employment_years: mean, num_delinquencies: mode)
- [ ] Apply log transformation to num_delinquencies if using linear models
- [ ] Normalize/standardize features for distance-based algorithms

### Phase 2: Feature Engineering
- [ ] Create interaction features
- [ ] Engineer aggregate and trend features
- [ ] Perform feature selection (correlation, mutual information, model-based)

### Phase 3: Model Development
- [ ] Implement baseline logistic regression
- [ ] Build XGBoost and LightGBM models
- [ ] Apply SMOTE and class weights
- [ ] Perform cross-validation with stratified k-folds

### Phase 4: Model Evaluation
- [ ] Evaluate using F1-score, Precision-Recall, ROC-AUC
- [ ] Conduct threshold optimization for default class
- [ ] Perform feature importance analysis
- [ ] Create SHAP interpretability plots

---

## Deliverables Summary

✅ **Completed:**
1. Dataset loading and overview (5,000 × 14 records)
2. Summary statistics for all numerical and categorical features
3. Distribution analysis with 6 visualization plots
4. Correlation heatmap and target correlations analysis
5. Missing value detection and visualization
6. Outlier detection using IQR method
7. Target variable class imbalance analysis
8. Key insights and recommendations document
9. Feature engineering recommendations
10. Next steps roadmap

📊 **Visualizations Generated:**
- numerical_distributions.png
- categorical_distributions.png
- correlation_matrix.png
- target_correlations.png
- missing_values.png
- outliers_boxplots.png
- target_distribution.png

---

## Conclusion

The credit risk dataset is well-structured with minimal data quality issues. The primary challenge is significant class imbalance in the target variable (10.68% default rate), which will require sophisticated handling techniques. The weak direct correlations between features and default suggest that advanced models capable of capturing non-linear relationships will be most effective. The excellent data quality enables proceeding directly to feature engineering without extensive preprocessing.

**Ready for:** Feature Engineering & Model Development Phase

---

*Report Generated: May 30, 2026*  
*Analysis Tool: Python (pandas, matplotlib, seaborn, scikit-learn)*  
*All analysis code and visualizations available in the repository*

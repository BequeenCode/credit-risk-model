# Credit Risk Model Development

A comprehensive project for building interpretable and defensible credit risk models in a regulated financial context.

## Project Objective

Develop a credit scoring model that balances interpretability with predictive performance while adhering to Basel II regulatory requirements and best practices in financial risk modeling.

---

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Interpretability

**Context**: The Basel II Accord fundamentally reshaped banking regulation by emphasizing the measurement and quantification of credit risk. This regulatory framework has profound implications for model development.

**Why Interpretability Matters**:
- **Regulatory Requirement**: Regulators demand that financial institutions document and justify their credit risk models. A "black box" model fails this requirement.
- **Capital Allocation**: Under Basel II, credit risk models directly influence regulatory capital requirements. A model that cannot be explained cannot be defended to auditors or supervisors.
- **Model Governance**: Financial institutions must demonstrate that models are appropriate, validated, and operating as intended. Interpretability enables ongoing monitoring.
- **Stakeholder Confidence**: Board members, risk committees, and clients need to trust the model's logic. Complex models may be efficient but create reputational and operational risk if they fail.

**Key Implication**: In a regulated context, interpretability is not optional—it is a business and compliance requirement that often outweighs marginal improvements in prediction accuracy.

---

### 2. Proxy Variables for Default Prediction

**The Challenge**: Many credit datasets lack a direct "default" label, particularly for:
- Performing portfolios where no defaults have occurred
- Early-stage data collection before sufficient default events accumulate
- Proprietary or confidential default records that are not available for modeling

**Why Proxy Variables Are Necessary**:
- They allow model development when true default labels are unavailable
- They enable feature engineering and model testing on historical data
- They accelerate the development cycle without waiting for rare default events

**Business Risks of Proxy-Based Prediction**:
1. **Conceptual Mismatch**: A proxy (e.g., missed payment, increased delinquency) may not perfectly correlate with actual default. A borrower may miss a payment due to administrative error but never default.
2. **False Positive Risk**: The model may identify borrowers as high-risk when the proxy signals risk but default probability is low. This leads to excessive loan rejections and lost business.
3. **False Negative Risk**: The proxy may fail to capture true default drivers (e.g., macroeconomic shocks, industry disruption), resulting in underprovided for risks.
4. **Model Bias**: If proxy events are unevenly distributed across demographic groups, the model may encode unfair lending practices.
5. **Implementation Gap**: A model trained on a proxy must be continuously validated against actual defaults once they occur, requiring ongoing monitoring and recalibration.

**Mitigation Strategies**:
- Use multiple proxies to triangulate true risk
- Validate proxy-based models against actual defaults as soon as data becomes available
- Regularly backtest model performance
- Document proxy assumptions transparently in regulatory filings

---

### 3. Trade-offs: Interpretable vs. High-Performance Models

#### Simple, Interpretable Models (e.g., Logistic Regression with Weight of Evidence)

**Advantages**:
- **Explainability**: Coefficient signs and magnitudes are directly interpretable. Business users and regulators understand exactly how variables influence risk.
- **Regulatory Alignment**: Widely accepted by financial regulators; meets governance requirements with minimal documentation burden.
- **Stability**: Less prone to overfitting; model behavior is stable across time and segments.
- **Auditability**: Variables and transformations are transparent, making compliance audits simpler.
- **Operational Efficiency**: Easier to implement, monitor, and maintain in production environments.

**Disadvantages**:
- **Predictive Performance**: May leave predictive performance on the table compared to ensemble methods.
- **Feature Engineering Burden**: Requires manual binning (e.g., WoE transformation) and non-linear relationships must be explicitly encoded.
- **Limited Interaction Capture**: Cannot easily capture complex feature interactions unless explicitly specified.

#### High-Performance Models (e.g., Gradient Boosting)

**Advantages**:
- **Superior Accuracy**: Ensemble methods often achieve higher AUC, KS, and predictive power.
- **Automatic Feature Engineering**: Can capture non-linear relationships and interactions without manual binning.
- **Reduced Bias**: Especially for imbalanced data, boosting can achieve better calibration across risk segments.
- **Flexibility**: Handles mixed data types, outliers, and missing values more robustly.

**Disadvantages**:
- **Black Box Problem**: Feature importance rankings exist but do not explain individual predictions; regulators may reject them.
- **Model Risk**: Complex models are harder to validate, debug, and monitor. A failure is harder to diagnose.
- **Regulatory Friction**: Financial regulators often require additional documentation and justification. Some jurisdictions do not accept tree-based models without extensive bias testing.
- **Governance Overhead**: Requires more sophisticated monitoring infrastructure and expert staff.

#### Recommended Balanced Approach

In regulated financial contexts, the industry best practice is:

1. **Develop interpretable models first** (Logistic Regression with WoE) as the baseline model
2. **Use ensemble methods as secondary models** for internal benchmarking and risk measurement
3. **Adopt a **hybrid strategy**: Deploy the interpretable model in production (meeting regulatory requirements) while using ensemble models to identify additional risk factors and validate model assumptions
4. **Layer in interpretability**: Use SHAP values or other post-hoc explainability methods on ensemble models to bridge the gap between performance and interpretability
5. **Prioritize governance**: Regardless of model complexity, rigorous documentation, backtesting, and monitoring are mandatory

**Conclusion**: The "best" model depends on regulatory appetite, organizational risk tolerance, and the regulatory environment. In most regulated financial institutions, model interpretability and regulatory acceptance ultimately outweigh marginal accuracy gains.

---

## Project Structure

```
credit-risk-model/
├── README.md                  # Project documentation
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis
│   ├── feature_engineering.ipynb
│   ├── model_development.ipynb
│   └── model_validation.ipynb
├── data/
│   ├── raw/                  # Original, immutable data
│   ├── processed/            # Cleaned and transformed data
│   └── external/             # External data sources
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── models/
│   └── model_registry.pkl   # Trained model artifacts
├── reports/
│   └── model_report.md       # Model documentation
└── requirements.txt          # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/credit-risk-model.git
cd credit-risk-model

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Feature Engineering (Task 3)

`src/data_processing.py` implements the full, reproducible feature-engineering
flow as a single `sklearn.pipeline.Pipeline` object that maps raw input to a
model-ready `DataFrame`.

> **Data-grain note.** The brief is written for a *transaction-level* dataset
> (per-customer aggregations, transaction timestamps). This project's data is
> *loan-level* — one row per loan, with a real `default` label and no
> transaction or datetime columns. The engineering intent is adapted faithfully
> to the available data: per-loan financial ratios and bands stand in for the
> per-customer transaction aggregates and datetime parts.

### Pipeline stages

| Stage | Implementation | Brief requirement |
|-------|----------------|-------------------|
| Derived / "aggregate" features | `FeatureEngineer` — `debt_to_income`, `monthly_installment`, `installment_to_income`, `accounts_per_year_employed`, `total_interest_ratio` | Create aggregate features |
| Extracted bands / flags | `FeatureEngineer` — `has_delinquency`, `credit_score_band` | Extract features |
| Missing-value handling | `SimpleImputer` (median for numeric, most-frequent for categorical) | Handle missing values |
| Categorical encoding | `OneHotEncoder(handle_unknown="ignore")` | Encode categorical variables |
| Scaling | `StandardScaler` (mean 0, std 1) | Normalize / standardize |
| WoE / IV | `WoEEncoder` (custom) with Information Value reporting | Feature engineering with WoE & IV |

All stages are composed in `build_pipeline()` and fit together. The fitted
pipeline is pickled to `models/feature_pipeline.pkl` for reuse at inference time.

### Run it

```bash
# Fit the pipeline, write the model-ready dataset, and persist the pipeline
python src/data_processing.py
#   -> data/processed/credit_data_processed.csv
#   -> models/feature_pipeline.pkl

# Run the unit tests
pytest tests/ -v
```

```python
# Reuse the fitted pipeline programmatically
import joblib
from src.data_processing import load_raw_data

pipe = joblib.load("models/feature_pipeline.pkl")
X = load_raw_data().drop(columns=["default"])
X_model = pipe.transform(X)   # model-ready DataFrame, no NaNs
```

A note on **Weight of Evidence / Information Value**: rather than depend on
`xverse`/`woe` (which lag recent scikit-learn/pandas releases), WoE is
implemented natively in `WoEEncoder` with Laplace smoothing to avoid `ln(0)`,
quantile binning for numeric features, and a `MISSING` bin so missing values
receive their own evidence weight. `WoEEncoder.get_iv_table()` returns features
ranked by Information Value — the standard interpretability artifact for a
logistic-regression scorecard.

---

## Proxy Target Engineering (Task 4)

Since the brief assumes a dataset with no `default` label, it asks us to
*construct* a credit-risk target from behavioural engagement using **RFM +
K-Means**. `build_proxy_target()` in `src/data_processing.py` does this and adds
a binary `is_high_risk` column to the processed dataset.

> **Data-grain note.** RFM (Recency, Frequency, Monetary) is normally computed
> from transaction history — `CustomerId`, transaction dates, and amounts —
> relative to a snapshot date. This loan-level dataset has none of those
> columns (and already carries a real `default` label). The method is kept
> exactly as prescribed — profile → scale → `KMeans(3, random_state=42)` →
> flag the least-engaged cluster — while each RFM axis is mapped onto the
> behavioural proxy that exists per loan:
>
> | RFM axis | Proxy column | Interpretation |
> |----------|--------------|----------------|
> | Recency | `employment_years` | longer stable tenure = more engaged |
> | Frequency | `num_accounts` | more credit accounts = more active |
> | Monetary | `income` | higher income = greater capacity |

### Method
1. **Build the engagement profile** from the three proxy columns (median-imputed).
2. **Scale** with `StandardScaler` so no axis dominates the distance metric.
3. **Cluster** into 3 segments with `KMeans(random_state=42, n_init=10)` for
   reproducibility.
4. **Identify the high-risk cluster** as the one ranking lowest across all three
   engagement axes (low tenure, few accounts, low income).
5. **Assign `is_high_risk`** = 1 for that cluster, 0 otherwise, and **merge** it
   into `data/processed/credit_data_processed.csv` alongside the actual label.

The script also prints an **audit cross-tab** of the engagement proxy against
the real `default` label. On this synthetic data the two do **not** align
closely (default here is driven by credit score and delinquencies, not
engagement) — a concrete illustration of the *proxy risk* discussed in the
Business Understanding section: a proxy can diverge from true default, so any
proxy-trained model must be revalidated against actual defaults.

### Run it

```bash
python src/data_processing.py   # adds the is_high_risk column to the processed CSV
```

```python
from src.data_processing import build_proxy_target, load_raw_data

labels, info = build_proxy_target(load_raw_data())
print(info["high_risk_cluster"], labels.mean())  # cluster id, high-risk rate
```

---

## Model Training & Tracking (Task 5)

`src/train.py` trains and compares two classifiers on the processed dataset,
with hyperparameter tuning, MLflow experiment tracking, full evaluation, and
Model Registry registration.

| Step | Implementation |
|------|----------------|
| Reproducible split | `split_data()` — stratified 80/20, `random_state=42`; drops **both** target columns to prevent leakage |
| Models (≥2) | Logistic Regression (interpretable baseline) + Random Forest (ensemble), both `class_weight="balanced"` for the imbalanced target |
| Hyperparameter tuning | `GridSearchCV` (Logistic Regression) and `RandomizedSearchCV` (Random Forest), 5-fold CV, scored on ROC-AUC |
| Experiment tracking | MLflow logs params, metrics, and model artifacts per run |
| Evaluation | accuracy, precision, recall, F1, ROC-AUC (`evaluate_model()`) |
| Model selection & registry | Best model by ROC-AUC saved to `models/best_model.pkl` **and** registered as `credit-risk-best-model` in the MLflow Model Registry |

### Run it

```bash
python src/train.py                      # train on the actual `default` label
python src/train.py --target is_high_risk  # train on the Task 4 proxy instead

mlflow ui                                # browse/compare runs at http://localhost:5000
```

> MLflow is an optional dependency: if it is not installed, training still runs
> and the best model is saved to disk — only the tracking/registry steps are
> skipped.

**Note on metrics.** This is synthetic data with a weak, mostly-noise signal and
a ~11% default rate, so ROC-AUC sits around 0.63 and precision is low — honest
for this dataset. `class_weight="balanced"` is used so the models actually
identify defaulters (non-zero recall) instead of collapsing to the majority
class.

---

## References

- Basel Committee on Banking Supervision. (2004). *International Convergence of Capital Measurement and Capital Standards*.
- World Bank. *Credit Scoring Approaches Guidelines*.
- HKMA. *Alternative Credit Scoring Methods*.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*.

---

## License

This project is provided for educational purposes.
#   c r e d i t - r i s k - m o d e l  
 
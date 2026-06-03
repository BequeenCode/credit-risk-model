"""
Feature Engineering & Proxy Target Pipeline - Credit Risk Model (Tasks 3 & 4)
============================================================================

Builds a single, reproducible ``sklearn.pipeline.Pipeline`` that transforms the
raw credit dataset into a model-ready ``DataFrame`` (Task 3), and engineers a
behavioural ``is_high_risk`` proxy target via RFM + K-Means (Task 4).

Note on data grain
-------------------
The Week's instructions describe a *transaction-level* dataset (one row per
transaction) and ask for per-customer aggregations and datetime extraction.
This project's dataset (``data/raw/credit_data.csv``) is *loan-level*: one row
per loan, with a real ``default`` label and no transaction-grain or timestamp
columns. The same engineering intent is therefore adapted faithfully to the
data we actually have:

  * "Aggregate features"  -> domain financial ratios / interaction features
                             derived per loan (DTI, instalment burden, etc.).
  * "Extract features"    -> derived bands / flags (credit-score band,
                             delinquency flag) in place of datetime parts.
  * Encoding, imputation, scaling, WoE/IV are implemented exactly as required.

The whole flow is exposed as one fitted ``Pipeline`` object via
``build_pipeline()`` so it is reproducible and can be pickled for inference.

Usage
-----
    python src/data_processing.py            # fit, transform, save artifacts

    from src.data_processing import build_pipeline, load_raw_data
    df = load_raw_data("data/raw/credit_data.csv")
    pipe = build_pipeline(target="default")
    X = pipe.fit_transform(df.drop(columns=["default"]), df["default"])
"""

import os

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/credit_data.csv"
PROCESSED_DATA_PATH = "data/processed/credit_data_processed.csv"
PIPELINE_PATH = "models/feature_pipeline.pkl"

TARGET = "default"
ID_COLUMN = "loan_id"

# Raw categorical columns expected in the dataset.
CATEGORICAL_FEATURES = ["employment_type", "home_ownership", "loan_purpose"]

# --- Task 4: proxy target configuration -----------------------------------
# Name of the engineered proxy target column.
PROXY_TARGET = "is_high_risk"

# RFM analogues (see ``build_proxy_target`` docstring). The brief's RFM is
# defined on transaction history (CustomerId, transaction dates, amounts),
# which this loan-level dataset does not have. We map the engagement intent
# of each RFM axis onto the behavioural columns that do exist:
#   Recency   -> employment_years   (stability / recency of stable income)
#   Frequency -> num_accounts       (breadth of credit activity)
#   Monetary  -> income             (financial capacity)
RFM_RECENCY = "employment_years"
RFM_FREQUENCY = "num_accounts"
RFM_MONETARY = "income"
RFM_RANDOM_STATE = 42
RFM_N_CLUSTERS = 3


# ---------------------------------------------------------------------------
# Step 1 & 2 - Aggregate / Derived feature creation
# ---------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-driven derived features from the raw loan record.

    Because the data is loan-level (not transaction-level) the "aggregate"
    features are financial ratios and interaction terms that summarise a
    borrower's risk profile, analogous to the per-customer aggregates the
    brief describes. Drops the ID column, which carries no signal.
    """

    def __init__(self, id_column=ID_COLUMN):
        self.id_column = id_column

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        # Drop the identifier - it is not a predictor.
        if self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])

        eps = 1e-9  # guard against divide-by-zero

        # --- Ratio / aggregate features ------------------------------------
        # Debt-to-income: loan size relative to annual income.
        X["debt_to_income"] = X["loan_amount"] / (X["income"] + eps)

        # Approximate monthly instalment (simple amortisation) and the share
        # of monthly income it consumes - a classic affordability measure.
        monthly_rate = (X["interest_rate"] / 100.0) / 12.0
        n = X["loan_term"].clip(lower=1)
        factor = (1 + monthly_rate) ** n
        X["monthly_installment"] = np.where(
            monthly_rate.abs() < eps,
            X["loan_amount"] / n,
            X["loan_amount"] * monthly_rate * factor / (factor - 1 + eps),
        )
        X["installment_to_income"] = X["monthly_installment"] / (
            X["income"] / 12.0 + eps
        )

        # Credit utilisation proxy: accounts vs. tenure of employment.
        X["accounts_per_year_employed"] = X["num_accounts"] / (
            X["employment_years"] + 1.0
        )

        # Interest paid over the full term relative to principal.
        X["total_interest_ratio"] = (
            X["monthly_installment"] * n - X["loan_amount"]
        ) / (X["loan_amount"] + eps)

        # --- Extracted bands / flags (datetime-part analogue) --------------
        # Binary flag: any prior delinquency is a strong risk signal.
        X["has_delinquency"] = (X["num_delinquencies"].fillna(0) > 0).astype(int)

        # Coarse credit-score band, like extracting a calendar part.
        X["credit_score_band"] = pd.cut(
            X["credit_score"],
            bins=[0, 580, 670, 740, 800, 1000],
            labels=["poor", "fair", "good", "very_good", "excellent"],
        ).astype(object)

        return X


# ---------------------------------------------------------------------------
# Step 6 - Weight of Evidence (WoE) / Information Value (IV)
# ---------------------------------------------------------------------------
class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight-of-Evidence encoder with Information Value reporting.

    For each selected feature the values are binned (quantiles for numeric,
    categories for object) and replaced by the bin's WoE:

        WoE = ln( (% of non-events in bin) / (% of events in bin) )

    where an "event" is the positive target class (default == 1). The
    Information Value of a feature is::

        IV = sum_bins (%non-event - %event) * WoE

    WoE encoding linearises monotonic risk relationships and is the standard
    input transform for interpretable logistic-regression scorecards.

    Parameters
    ----------
    columns : list[str] or None
        Columns to WoE-encode. If None, every column seen in ``fit`` is used.
    n_bins : int
        Number of quantile bins for numeric features.
    """

    def __init__(self, columns=None, n_bins=10):
        self.columns = columns
        self.n_bins = n_bins

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)

        self.columns_ = list(X.columns) if self.columns is None else self.columns
        self.woe_maps_ = {}       # column -> {bin_label: woe_value}
        self.bin_edges_ = {}      # numeric column -> array of edges
        self.iv_ = {}             # column -> information value

        total_events = float(y.sum())          # defaults
        total_non_events = float((1 - y).sum())  # non-defaults

        for col in self.columns_:
            binned, edges = self._bin_column(X[col], fit=True)
            if edges is not None:
                self.bin_edges_[col] = edges

            woe_map, iv = self._compute_woe(binned, y, total_events, total_non_events)
            self.woe_maps_[col] = woe_map
            self.iv_[col] = iv

        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        out = pd.DataFrame(index=X.index)

        for col in self.columns_:
            binned, _ = self._bin_column(X[col], fit=False, col_name=col)
            woe_map = self.woe_maps_[col]
            # Unseen bins fall back to 0 (neutral evidence).
            out[f"{col}_woe"] = binned.map(woe_map).fillna(0.0).astype(float)

        return out

    # -- helpers -----------------------------------------------------------
    def _bin_column(self, s, fit, col_name=None):
        """Return a categorical Series of bin labels (+ edges when numeric)."""
        if pd.api.types.is_numeric_dtype(s):
            if fit:
                # Quantile edges; deduplicate for low-cardinality columns.
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                edges = np.unique(np.nanquantile(s, quantiles))
                edges[0], edges[-1] = -np.inf, np.inf
            else:
                edges = self.bin_edges_[col_name]
            binned = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")
            # Represent missing as its own bin so it gets a WoE.
            return binned.astype(object).where(s.notna(), other="MISSING"), edges

        # Categorical: each value is its own bin; NaN -> "MISSING".
        return s.astype(object).where(s.notna(), other="MISSING"), None

    @staticmethod
    def _compute_woe(binned, y, total_events, total_non_events):
        df = pd.DataFrame({"bin": binned.values, "y": y.values})
        grouped = df.groupby("bin", observed=True)["y"].agg(["sum", "count"])
        grouped["events"] = grouped["sum"]
        grouped["non_events"] = grouped["count"] - grouped["sum"]

        # Laplace smoothing avoids ln(0) / division by zero in empty bins.
        pct_events = (grouped["events"] + 0.5) / (total_events + 0.5)
        pct_non_events = (grouped["non_events"] + 0.5) / (total_non_events + 0.5)

        woe = np.log(pct_non_events / pct_events)
        iv = float(((pct_non_events - pct_events) * woe).sum())

        return woe.to_dict(), iv

    def get_iv_table(self):
        """Return features ranked by Information Value (call after ``fit``)."""
        return (
            pd.DataFrame({"feature": list(self.iv_.keys()),
                          "iv": list(self.iv_.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Steps 3, 4, 5 - Encoding, imputation, scaling (ColumnTransformer)
# ---------------------------------------------------------------------------
class PreprocessingBuilder(BaseEstimator, TransformerMixin):
    """Resolve column groups *after* feature engineering, then impute/scale/encode.

    Column lists must be discovered at fit time because ``FeatureEngineer``
    adds new columns. This wrapper inspects the engineered frame and builds a
    ``ColumnTransformer`` accordingly:

      * numeric    -> median imputation + standardisation (mean 0, std 1)
      * categorical-> most-frequent imputation + one-hot encoding
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.column_transformer_ = ColumnTransformer([
            ("num", numeric_pipe, self.numeric_cols_),
            ("cat", categorical_pipe, self.categorical_cols_),
        ], remainder="drop")

        self.column_transformer_.fit(X, y)
        self.feature_names_ = self.column_transformer_.get_feature_names_out()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        arr = self.column_transformer_.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
class _DataFrameUnion(BaseEstimator, TransformerMixin):
    """Run two transformers on the same input and concat their output columns.

    Used to keep the WoE features alongside the scaled/encoded features in a
    single model-ready frame.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def fit(self, X, y=None):
        self.left.fit(X, y)
        self.right.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        left_out = self.left.transform(X).reset_index(drop=True)
        right_out = self.right.transform(X).reset_index(drop=True)
        return pd.concat([left_out, right_out], axis=1)


def build_pipeline(target=TARGET, woe_n_bins=10):
    """Construct the full feature-engineering ``Pipeline``.

    Returns an unfitted ``sklearn.pipeline.Pipeline`` that maps a raw feature
    ``DataFrame`` (without the target) to a model-ready ``DataFrame`` combining
    standardised/one-hot-encoded features with WoE-encoded features.
    """
    engineer = FeatureEngineer()

    # After engineering, branch into (a) impute/scale/encode and (b) WoE,
    # then union the two column sets back together.
    preprocessing = PreprocessingBuilder()
    woe = WoEEncoder(n_bins=woe_n_bins)

    pipeline = Pipeline([
        ("feature_engineering", engineer),
        ("transform", _DataFrameUnion(left=preprocessing, right=woe)),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Task 4 - Proxy target via RFM + K-Means
# ---------------------------------------------------------------------------
def build_proxy_target(
    df,
    recency=RFM_RECENCY,
    frequency=RFM_FREQUENCY,
    monetary=RFM_MONETARY,
    n_clusters=RFM_N_CLUSTERS,
    random_state=RFM_RANDOM_STATE,
):
    """Engineer a binary ``is_high_risk`` proxy target via RFM + K-Means.

    Note on data grain
    ------------------
    The brief defines RFM on *transaction history* (``CustomerId``, transaction
    dates, transaction amounts) and asks for a snapshot date to compute
    Recency. This project's data is loan-level and has none of those columns,
    but it has a real ``default`` label. We therefore keep the method exactly as
    prescribed - build an engagement profile, scale it, cluster with K-Means
    into ``n_clusters`` groups (fixed ``random_state`` for reproducibility),
    and flag the least-engaged cluster as high-risk - while mapping each RFM
    axis onto the behavioural proxy that exists per loan:

        Recency   -> ``employment_years``  (more tenure  = more engaged)
        Frequency -> ``num_accounts``      (more accounts = more engaged)
        Monetary  -> ``income``            (more income   = more engaged)

    The least-engaged cluster (low on all three) is labelled ``is_high_risk = 1``.

    Parameters
    ----------
    df : DataFrame
        Raw loan data containing the three proxy columns.
    n_clusters : int
        Number of K-Means segments (3 per the brief).
    random_state : int
        Seed for reproducible clustering.

    Returns
    -------
    (labels, info) : (Series, dict)
        ``labels`` is an int ``is_high_risk`` Series aligned to ``df``'s index.
        ``info`` carries the cluster profile table and the chosen high-risk
        cluster id for reporting / auditing.
    """
    rfm_cols = [recency, frequency, monetary]
    missing = [c for c in rfm_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Proxy-target columns not found in data: {missing}")

    # Median-impute the proxy features so clustering sees no NaNs.
    rfm = df[rfm_cols].copy()
    rfm = rfm.fillna(rfm.median(numeric_only=True))

    # Scale before K-Means so no single axis dominates the distance metric.
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(rfm_scaled)

    # Profile each cluster on the original (unscaled) RFM means.
    profile = (
        rfm.assign(cluster=cluster_ids)
        .groupby("cluster")
        .mean()
    )
    profile["size"] = pd.Series(cluster_ids).value_counts().sort_index()

    # Least-engaged = lowest combined rank across all three engagement axes.
    # Rank each axis ascending (low value -> low rank); the cluster with the
    # smallest summed rank is the most disengaged / highest risk.
    engagement_rank = profile[rfm_cols].rank().sum(axis=1)
    high_risk_cluster = int(engagement_rank.idxmin())

    labels = pd.Series(
        (cluster_ids == high_risk_cluster).astype(int),
        index=df.index,
        name=PROXY_TARGET,
    )

    info = {
        "profile": profile,
        "high_risk_cluster": high_risk_cluster,
        "rfm_columns": rfm_cols,
    }
    return labels, info


# ---------------------------------------------------------------------------
# I/O helpers + entry point
# ---------------------------------------------------------------------------
def load_raw_data(path=RAW_DATA_PATH):
    """Load the raw credit dataset into a ``DataFrame``."""
    return pd.read_csv(path)


def main():
    print("=" * 80)
    print("FEATURE ENGINEERING PIPELINE - CREDIT RISK MODEL")
    print("=" * 80)

    print(f"\n[1/5] Loading raw data from {RAW_DATA_PATH} ...")
    df = load_raw_data()
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("\n[2/5] Fitting feature pipeline ...")
    pipeline = build_pipeline(target=TARGET)
    X_model = pipeline.fit_transform(X, y)
    print(f"  Model-ready matrix: {X_model.shape[0]} rows, {X_model.shape[1]} features")

    # Report Information Value of the WoE features (interpretability artifact).
    woe_encoder = pipeline.named_steps["transform"].right
    iv_table = woe_encoder.get_iv_table()
    print("\n  Information Value (predictive strength) ranking:")
    print(iv_table.to_string(index=False))

    print("\n[3/5] Building proxy target (RFM + K-Means) ...")
    proxy, info = build_proxy_target(df)
    print(f"  RFM proxy columns: {info['rfm_columns']}")
    print(f"  High-risk cluster: {info['high_risk_cluster']} "
          f"(least-engaged segment)")
    print("\n  Cluster engagement profile (original RFM means):")
    print(info["profile"].round(2).to_string())
    rate = proxy.mean()
    print(f"\n  '{PROXY_TARGET}' rate: {rate:.1%} "
          f"({int(proxy.sum())} of {len(proxy)} loans)")

    # Audit only: how the proxy lines up with the real default label. The proxy
    # is engagement-based, so this is a sanity check, not a fit metric.
    overlap = pd.crosstab(proxy, y, normalize="index")
    print("\n  Proxy vs. actual default (row-normalised, audit only):")
    print(overlap.round(3).to_string())

    print("\n[4/5] Writing processed dataset ...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed = X_model.copy()
    # Integrate both the actual label and the engineered proxy target.
    processed[TARGET] = y.reset_index(drop=True)
    processed[PROXY_TARGET] = proxy.reset_index(drop=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Saved -> {PROCESSED_DATA_PATH} "
          f"(includes '{PROXY_TARGET}' target column)")

    print("\n[5/5] Persisting fitted pipeline ...")
    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"  Saved -> {PIPELINE_PATH}")

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Re-enter through the package namespace so custom transformers are pickled
    # as ``src.data_processing.*`` (portable) rather than ``__main__.*``.
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data_processing import main as packaged_main

    packaged_main()

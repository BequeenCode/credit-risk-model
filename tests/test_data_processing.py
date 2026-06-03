"""
Unit tests for the feature-engineering pipeline (Task 3), the proxy target
(Task 4), and the training helpers (Task 5).

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    FeatureEngineer,
    WoEEncoder,
    build_pipeline,
    build_proxy_target,
    load_raw_data,
)
from src.train import evaluate_model, get_model_search_specs, split_data


@pytest.fixture
def raw_sample():
    """A small, hand-built raw frame mirroring the dataset schema."""
    return pd.DataFrame({
        "loan_id": [1, 2, 3, 4],
        "age": [30, 45, 52, 60],
        "income": [40000, 60000, 80000, 30000],
        "loan_amount": [120000, 90000, 50000, 150000],
        "loan_term": [12, 24, 36, 60],
        "interest_rate": [5.0, 6.5, 4.0, 8.0],
        "employment_years": [3.0, np.nan, 10.0, 1.0],
        "num_accounts": [5, 8, 3, 12],
        "num_delinquencies": [0.0, 2.0, np.nan, 3.0],
        "credit_score": [620, 700, 800, 540],
        "employment_type": ["Employed", "Self-Employed", "Employed", "Unemployed"],
        "home_ownership": ["Rent", "Own", "Mortgage", "Rent"],
        "loan_purpose": ["Auto", "Business", "Education", "Auto"],
    })


@pytest.fixture
def target():
    return pd.Series([1, 0, 0, 1], name="default")


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------
def test_feature_engineer_creates_expected_columns(raw_sample):
    out = FeatureEngineer().fit_transform(raw_sample)

    for col in [
        "debt_to_income",
        "monthly_installment",
        "installment_to_income",
        "accounts_per_year_employed",
        "total_interest_ratio",
        "has_delinquency",
        "credit_score_band",
    ]:
        assert col in out.columns, f"missing engineered feature: {col}"


def test_feature_engineer_drops_id(raw_sample):
    out = FeatureEngineer().fit_transform(raw_sample)
    assert "loan_id" not in out.columns


def test_debt_to_income_is_correct(raw_sample):
    out = FeatureEngineer().fit_transform(raw_sample)
    # Row 0: 120000 / 40000 == 3.0
    assert out["debt_to_income"].iloc[0] == pytest.approx(3.0, rel=1e-6)


def test_has_delinquency_flag(raw_sample):
    out = FeatureEngineer().fit_transform(raw_sample)
    # [0, 2, NaN->0, 3] -> [0, 1, 0, 1]
    assert out["has_delinquency"].tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# WoEEncoder
# ---------------------------------------------------------------------------
def test_woe_encoder_outputs_woe_columns(raw_sample, target):
    enc = WoEEncoder(columns=["credit_score", "employment_type"], n_bins=3)
    out = enc.fit_transform(raw_sample, target)
    assert list(out.columns) == ["credit_score_woe", "employment_type_woe"]
    assert out.notna().all().all()


def test_woe_iv_table_is_sorted_and_nonnegative(raw_sample, target):
    enc = WoEEncoder(columns=["credit_score", "num_delinquencies"], n_bins=3)
    enc.fit(raw_sample, target)
    iv = enc.get_iv_table()
    assert (iv["iv"] >= 0).all()
    assert iv["iv"].is_monotonic_decreasing


def test_woe_handles_unseen_categories(raw_sample, target):
    enc = WoEEncoder(columns=["loan_purpose"], n_bins=3)
    enc.fit(raw_sample, target)
    unseen = raw_sample.copy()
    unseen["loan_purpose"] = "SpaceTravel"  # never seen in fit
    out = enc.transform(unseen)
    # Unseen bin -> neutral evidence (0.0), never NaN.
    assert (out["loan_purpose_woe"] == 0.0).all()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def test_pipeline_produces_dataframe_without_nans(raw_sample, target):
    pipe = build_pipeline(woe_n_bins=3)
    out = pipe.fit_transform(raw_sample, target)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(raw_sample)
    assert not out.isnull().any().any()


def test_pipeline_transform_is_consistent(raw_sample, target):
    pipe = build_pipeline(woe_n_bins=3)
    pipe.fit(raw_sample, target)
    first = pipe.transform(raw_sample)
    second = pipe.transform(raw_sample)
    pd.testing.assert_frame_equal(first, second)


def test_pipeline_on_real_data():
    """End-to-end smoke test on the actual raw CSV, if present."""
    try:
        df = load_raw_data()
    except FileNotFoundError:
        pytest.skip("raw data file not available")
    X = df.drop(columns=["default"])
    y = df["default"]
    out = build_pipeline().fit_transform(X, y)
    assert out.shape[0] == len(df)
    assert not out.isnull().any().any()


# ---------------------------------------------------------------------------
# Task 4 - proxy target (RFM + K-Means)
# ---------------------------------------------------------------------------
@pytest.fixture
def rfm_sample():
    """Three clearly separable engagement segments for K-Means.

    Rows 0-3   : disengaged (low tenure, few accounts, low income) -> high risk
    Rows 4-7   : mid engagement
    Rows 8-11  : highly engaged (high on all axes)
    """
    return pd.DataFrame({
        "employment_years": [0.0, 1.0, 0.5, 1.0, 6.0, 7.0, 6.5, 7.0, 15.0, 16.0, 14.0, 15.0],
        "num_accounts":     [1,   2,   1,   2,   7,   8,   7,   8,   13,   14,   13,   14],
        "income":           [20000, 22000, 21000, 20000, 55000, 56000, 54000, 55000, 90000, 92000, 91000, 90000],
    })


def test_proxy_target_is_binary_and_aligned(rfm_sample):
    labels, info = build_proxy_target(rfm_sample)
    assert labels.name == "is_high_risk"
    assert set(labels.unique()) <= {0, 1}
    assert len(labels) == len(rfm_sample)
    assert labels.index.equals(rfm_sample.index)


def test_proxy_flags_least_engaged_segment(rfm_sample):
    labels, info = build_proxy_target(rfm_sample)
    # The first four rows are the disengaged segment -> should be high risk.
    assert labels.iloc[:4].sum() == 4
    # The most-engaged segment (last four) must not be high risk.
    assert labels.iloc[8:].sum() == 0


def test_proxy_target_is_reproducible(rfm_sample):
    first, _ = build_proxy_target(rfm_sample, random_state=42)
    second, _ = build_proxy_target(rfm_sample, random_state=42)
    pd.testing.assert_series_equal(first, second)


def test_proxy_target_handles_missing_values():
    df = pd.DataFrame({
        "employment_years": [0.0, np.nan, 6.0, 15.0, 16.0],
        "num_accounts":     [1,   2,      7,   13,   14],
        "income":           [20000, 22000, 55000, 90000, 92000],
    })
    labels, _ = build_proxy_target(df, n_clusters=3)
    assert not labels.isnull().any()
    assert set(labels.unique()) <= {0, 1}


def test_proxy_target_missing_column_raises():
    df = pd.DataFrame({"employment_years": [1.0], "num_accounts": [2]})
    with pytest.raises(ValueError):
        build_proxy_target(df)  # 'income' column absent


# ---------------------------------------------------------------------------
# Task 5 - training helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def processed_sample():
    """A small processed-style frame with both target columns."""
    rng = np.random.RandomState(0)
    n = 60
    return pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        "feat_c": rng.normal(size=n),
        "default": rng.randint(0, 2, size=n),
        "is_high_risk": rng.randint(0, 2, size=n),
    })


def test_split_data_drops_all_targets(processed_sample):
    X_train, X_test, y_train, y_test = split_data(processed_sample, target="default")
    # Neither target may remain in the feature matrix (no leakage).
    assert "default" not in X_train.columns
    assert "is_high_risk" not in X_train.columns
    assert y_train.name == "default"


def test_split_data_is_reproducible(processed_sample):
    first = split_data(processed_sample, target="default", random_state=1)
    second = split_data(processed_sample, target="default", random_state=1)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[2], second[2])


def test_split_data_unknown_target_raises(processed_sample):
    with pytest.raises(ValueError):
        split_data(processed_sample, target="not_a_column")


def test_get_model_search_specs_has_two_models():
    specs = get_model_search_specs()
    assert len(specs) >= 2
    assert "LogisticRegression" in specs and "RandomForest" in specs


def test_evaluate_model_returns_all_metrics(processed_sample):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = split_data(processed_sample, target="default")
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

"""
Unit tests for the feature-engineering pipeline (Task 3).

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    FeatureEngineer,
    WoEEncoder,
    build_pipeline,
    load_raw_data,
)


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

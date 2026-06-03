"""
Pydantic request/response schemas for the credit-risk API (Task 6).

The request mirrors the *raw* customer fields (the columns of the original
dataset, minus the identifier and target). The service runs these through the
fitted feature pipeline before scoring, so callers never deal with the 50
engineered/encoded columns directly.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Raw customer/loan attributes accepted by ``/predict``."""

    age: int = Field(..., ge=18, le=100, examples=[45])
    income: float = Field(..., gt=0, examples=[60000])
    loan_amount: float = Field(..., gt=0, examples=[150000])
    loan_term: int = Field(..., gt=0, examples=[36], description="Term in months")
    interest_rate: float = Field(..., ge=0, examples=[5.5])
    employment_years: float = Field(..., ge=0, examples=[8])
    num_accounts: int = Field(..., ge=0, examples=[5])
    num_delinquencies: float = Field(..., ge=0, examples=[0])
    credit_score: int = Field(..., ge=300, le=850, examples=[650])
    employment_type: Literal["Employed", "Self-Employed", "Unemployed"] = Field(
        ..., examples=["Employed"]
    )
    home_ownership: Literal["Rent", "Own", "Mortgage"] = Field(
        ..., examples=["Mortgage"]
    )
    loan_purpose: Literal[
        "Debt Consolidation", "Home Improvement", "Business", "Auto", "Education"
    ] = Field(..., examples=["Auto"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 45,
                    "income": 60000,
                    "loan_amount": 150000,
                    "loan_term": 36,
                    "interest_rate": 5.5,
                    "employment_years": 8,
                    "num_accounts": 5,
                    "num_delinquencies": 0,
                    "credit_score": 650,
                    "employment_type": "Employed",
                    "home_ownership": "Mortgage",
                    "loan_purpose": "Auto",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Risk score returned by ``/predict``."""

    risk_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of high credit risk"
    )
    risk_label: int = Field(..., description="1 = high risk, 0 = low risk")
    threshold: float = Field(..., description="Decision threshold applied")


class HealthResponse(BaseModel):
    """Service / model readiness."""

    status: str
    model_loaded: bool
    model_source: str

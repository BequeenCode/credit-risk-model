"""
Credit-Risk FastAPI service (Task 6).

Exposes a ``/predict`` endpoint that accepts raw customer data, runs it through
the fitted feature pipeline (Task 3) and the best model (Task 5), and returns a
risk probability.

Model loading order
-------------------
1. MLflow Model Registry (``credit-risk-best-model``) when ``MLFLOW_TRACKING_URI``
   / a registry is reachable.
2. Local pickle ``models/best_model.pkl`` as a fallback (CI, offline, Docker).

Run locally:
    uvicorn src.api.main:app --reload
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    CustomerData,
    HealthResponse,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_PIPELINE_PATH = os.getenv("FEATURE_PIPELINE_PATH", "models/feature_pipeline.pkl")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/best_model.pkl")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "credit-risk-best-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

app = FastAPI(
    title="Credit Risk API",
    description="Predicts the probability that a customer is high credit risk.",
    version="1.0.0",
)

# Populated at startup.
_state = {"model": None, "pipeline": None, "model_source": "none"}


# ---------------------------------------------------------------------------
# Model / pipeline loading
# ---------------------------------------------------------------------------
def _load_feature_pipeline():
    import joblib

    if os.path.exists(FEATURE_PIPELINE_PATH):
        return joblib.load(FEATURE_PIPELINE_PATH)
    return None


def _load_model():
    """Return ``(model, source)`` from the MLflow registry or local pickle.

    Prefers the ``mlflow.sklearn`` flavour so the native estimator (with
    ``predict_proba``) is returned rather than a pyfunc wrapper that only
    yields class labels.
    """
    # 1) Try the MLflow Model Registry (sklearn flavour for predict_proba).
    try:
        import mlflow.sklearn

        uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(uri)
        return model, f"mlflow:{uri}"
    except Exception:  # noqa: BLE001 - registry not reachable in many envs
        pass

    # 2) Fall back to the locally saved best model.
    try:
        import joblib

        if os.path.exists(LOCAL_MODEL_PATH):
            return joblib.load(LOCAL_MODEL_PATH), f"local:{LOCAL_MODEL_PATH}"
    except Exception:  # noqa: BLE001
        pass

    return None, "none"


@app.on_event("startup")
def _startup():
    _state["pipeline"] = _load_feature_pipeline()
    _state["model"], _state["model_source"] = _load_model()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _features_from_request(data: CustomerData) -> pd.DataFrame:
    """Turn a request into a one-row raw DataFrame, then apply the pipeline."""
    raw = pd.DataFrame([data.model_dump()])
    pipeline = _state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Feature pipeline not loaded (expected at {FEATURE_PIPELINE_PATH}).",
        )
    return pipeline.transform(raw)


def _predict_proba(model, X: pd.DataFrame) -> float:
    """Extract the positive-class probability across model flavours."""
    # scikit-learn estimator
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    # mlflow.pyfunc - predict() may yield probabilities or labels
    pred = model.predict(X)
    if hasattr(pred, "iloc"):
        pred = pred.iloc[:, -1] if getattr(pred, "ndim", 1) > 1 else pred
    value = float(pd.Series(pred).iloc[0])
    return value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {"service": "Credit Risk API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if _state["model"] is not None else "degraded",
        model_loaded=_state["model"] is not None,
        model_source=_state["model_source"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(data: CustomerData):
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it (src/train.py) or mount models/.",
        )

    X = _features_from_request(data)
    proba = _predict_proba(model, X)
    proba = min(max(proba, 0.0), 1.0)  # clamp to [0, 1]

    return PredictionResponse(
        risk_probability=round(proba, 6),
        risk_label=int(proba >= DECISION_THRESHOLD),
        threshold=DECISION_THRESHOLD,
    )

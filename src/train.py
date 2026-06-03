"""
Model Training & Experiment Tracking - Credit Risk Model (Task 5)
================================================================

Trains and compares at least two classifiers on the model-ready dataset
produced by ``data_processing.py``, with:

  * a reproducible train/test split (fixed ``random_state``);
  * hyperparameter tuning via ``GridSearchCV`` / ``RandomizedSearchCV``;
  * experiment tracking with MLflow (params, metrics, model artifacts);
  * full evaluation (accuracy, precision, recall, F1, ROC-AUC);
  * registration of the best model in the MLflow Model Registry.

Target
------
The processed dataset carries two candidate targets:

  * ``default``       - the actual historical label (used here as ground truth).
  * ``is_high_risk``  - the engineered behavioural proxy from Task 4.

We train on ``default`` by default (the genuine outcome). Pass ``--target
is_high_risk`` to train the proxy-based scorecard instead. Either way the other
target column is dropped from the feature matrix so it cannot leak.

Usage
-----
    python src/train.py
    python src/train.py --target is_high_risk
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# Optional dependency - the workflow still runs (and saves the best model to
# disk) if MLflow is not installed; tracking/registry steps are simply skipped.
try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH = "data/processed/credit_data_processed.csv"
BEST_MODEL_PATH = "models/best_model.pkl"
EXPERIMENT_NAME = "credit-risk-model"
REGISTERED_MODEL_NAME = "credit-risk-best-model"

DEFAULT_TARGET = "default"
PROXY_TARGET = "is_high_risk"
ALL_TARGETS = [DEFAULT_TARGET, PROXY_TARGET]

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_processed_data(path=PROCESSED_DATA_PATH):
    """Load the model-ready dataset written by ``data_processing.py``."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{path}'. "
            "Run `python src/data_processing.py` first."
        )
    return pd.read_csv(path)


def split_data(df, target=DEFAULT_TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split into stratified train/test sets.

    Drops *all* target columns from the feature matrix so neither the actual
    label nor the proxy can leak into training.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    drop_cols = [c for c in ALL_TARGETS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Model definitions + hyperparameter search spaces
# ---------------------------------------------------------------------------
def get_model_search_specs():
    """Return ``{name: (estimator, param_grid, search_kind)}`` to train/tune.

    Two models are tuned, satisfying the "at least two models" requirement and
    covering both an interpretable baseline (Logistic Regression) and an
    ensemble (Random Forest), each with a different search strategy.
    """
    return {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            },
            "grid",  # exhaustive GridSearch over a small space
        ),
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "random",  # RandomizedSearch over a larger space
        ),
    }


def tune_model(estimator, param_grid, search_kind, X_train, y_train):
    """Run Grid or Randomized search; return the fitted best estimator + params."""
    if search_kind == "grid":
        search = GridSearchCV(
            estimator, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=10,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Compute the five required classification metrics."""
    y_pred = model.predict(X_test)

    # ROC-AUC needs scores/probabilities; fall back to decision_function.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover
        y_score = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


# ---------------------------------------------------------------------------
# Training workflow
# ---------------------------------------------------------------------------
def train_and_track(target=DEFAULT_TARGET):
    """End-to-end: split, tune, evaluate, log to MLflow, register best model."""
    print("=" * 80)
    print(f"MODEL TRAINING & TRACKING  (target = '{target}')")
    print("=" * 80)

    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df, target=target)
    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
          f"| Features: {X_train.shape[1]}")

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow tracking -> experiment '{EXPERIMENT_NAME}'")
    else:
        print("MLflow not installed - training without experiment tracking.")

    results = []  # (name, model, params, metrics)

    for name, (estimator, grid, kind) in get_model_search_specs().items():
        print(f"\n--- {name}  ({kind} search) ---")
        best_estimator, best_params = tune_model(
            estimator, grid, kind, X_train, y_train
        )
        metrics = evaluate_model(best_estimator, X_test, y_test)

        print(f"  best params : {best_params}")
        print("  metrics     : " + ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()))

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("target", target)
                mlflow.log_param("search_kind", kind)
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_estimator, name="model")

        results.append((name, best_estimator, best_params, metrics))

    # Select the best model by ROC-AUC (primary metric for ranking risk).
    best_name, best_model, best_params, best_metrics = max(
        results, key=lambda r: r[3]["roc_auc"]
    )
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_name}  (ROC-AUC = {best_metrics['roc_auc']:.4f})")
    print("=" * 80)

    # Persist the best model to disk (always) ...
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved best model -> {BEST_MODEL_PATH}")

    # ... and register it in the MLflow Model Registry (if available).
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"{best_name}-registered"):
            mlflow.log_param("model", best_name)
            mlflow.log_param("target", target)
            mlflow.log_params(best_params)
            mlflow.log_metrics(best_metrics)
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
        print(f"Registered '{REGISTERED_MODEL_NAME}' in the MLflow Model Registry.")

    return best_name, best_model, best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train credit-risk models.")
    parser.add_argument(
        "--target",
        choices=ALL_TARGETS,
        default=DEFAULT_TARGET,
        help="Target column to train on (default: 'default').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    args = parse_args()
    train_and_track(target=args.target)

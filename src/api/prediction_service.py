"""
src/api/prediction_service.py
------------------------------
Core prediction logic — transforms request data and runs model inference.

This service layer separates business logic from the FastAPI route handlers.
The route handlers call these functions; this makes the code testable
without spinning up an HTTP server.

Also handles:
  - Feature engineering from raw request input
  - Prediction logging to PostgreSQL
"""

import uuid
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
    CreditRiskRequest,
    CreditRiskResponse,
    CreditRiskTier,
    CombinedPredictionResponse,
)
from src.api.model_loader import ModelStore
from src.utils.logger import logger

# Map integer predictions to tier names
CREDIT_TIER_MAP = {0: CreditRiskTier.GOOD, 1: CreditRiskTier.MODERATE, 2: CreditRiskTier.POOR}

# Fraud decision threshold (can be adjusted without retraining)
FRAUD_THRESHOLD = 0.5


def _build_fraud_features(req: FraudPredictionRequest) -> pd.DataFrame:
    """
    Convert a FraudPredictionRequest into a single-row DataFrame with
    all features the preprocessing pipeline expects.

    This mirrors the feature engineering done in clean_data.py.
    """
    # Derived features (computed the same way as in training)
    balance_error_orig = abs(req.oldbalance_org - req.amount - req.newbalance_orig)
    balance_error_dest = abs(req.oldbalance_dest + req.amount - req.newbalance_dest)
    orig_balance_zeroed = int(req.newbalance_orig == 0)

    return pd.DataFrame([{
        "step": req.step,
        "type": req.type.value,
        "amount": req.amount,
        "oldbalance_org": req.oldbalance_org,
        "newbalance_orig": req.newbalance_orig,
        "oldbalance_dest": req.oldbalance_dest,
        "newbalance_dest": req.newbalance_dest,
        "balance_error_orig": balance_error_orig,
        "balance_error_dest": balance_error_dest,
        "orig_balance_zeroed": orig_balance_zeroed,
    }])


def _build_credit_features(req: CreditRiskRequest) -> pd.DataFrame:
    """Convert a CreditRiskRequest into a single-row DataFrame."""
    return pd.DataFrame([{
        "age": req.age,
        "duration_months": req.duration_months,
        "credit_amount": req.credit_amount,
        "installment_rate": req.installment_rate,
        "residence_since": req.residence_since,
        "existing_credits": req.existing_credits,
        "num_dependents": req.num_dependents,
        "checking_account": req.checking_account,
        "credit_history": req.credit_history,
        "purpose": req.purpose,
        "savings": req.savings,
        "employment": req.employment,
        "housing": req.housing,
        "job": req.job,
    }])


def predict_fraud(
    request: FraudPredictionRequest,
    store: ModelStore,
    threshold: float = FRAUD_THRESHOLD,
) -> FraudPredictionResponse:
    """
    Run fraud detection inference.

    Steps:
      1. Build feature DataFrame
      2. Apply fitted preprocessor (no SMOTE at inference time)
      3. Get probability from XGBoost model
      4. Apply decision threshold

    Args:
        request: Validated FraudPredictionRequest.
        store: Loaded ModelStore.
        threshold: Decision threshold for binary fraud label.

    Returns:
        FraudPredictionResponse with probability and binary label.

    Raises:
        RuntimeError: If fraud model is not loaded.
    """
    if not store.fraud_ready:
        raise RuntimeError(
            "Fraud model is not loaded. "
            "Run 'python src/models/train.py' to train and save models."
        )

    features_df = _build_fraud_features(request)

    # Apply ONLY the preprocessor step (not SMOTE — SMOTE is training-only)
    X = store.fraud_pipeline.named_steps["preprocessor"].transform(features_df)

    fraud_prob = float(store.fraud_model.predict_proba(X)[0, 1])
    is_fraud = fraud_prob >= threshold

    logger.debug(f"Fraud prediction: prob={fraud_prob:.4f}, is_fraud={is_fraud}")

    return FraudPredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        threshold=threshold,
        model_version=store.fraud_model_version,
    )


def predict_credit_risk(
    request: CreditRiskRequest,
    store: ModelStore,
) -> CreditRiskResponse:
    """
    Run credit risk scoring inference.

    Args:
        request: Validated CreditRiskRequest.
        store: Loaded ModelStore.

    Returns:
        CreditRiskResponse with tier label and confidence scores per class.

    Raises:
        RuntimeError: If credit model is not loaded.
    """
    if not store.credit_ready:
        raise RuntimeError(
            "Credit model is not loaded. "
            "Run 'python src/models/train.py' to train and save models."
        )

    features_df = _build_credit_features(request)
    X = store.credit_pipeline.named_steps["preprocessor"].transform(features_df)

    y_pred = int(store.credit_model.predict(X)[0])
    y_prob = store.credit_model.predict_proba(X)[0]

    confidence_scores = {
        CreditRiskTier.GOOD.value: round(float(y_prob[0]), 4),
        CreditRiskTier.MODERATE.value: round(float(y_prob[1]), 4),
        CreditRiskTier.POOR.value: round(float(y_prob[2]), 4),
    }

    tier = CREDIT_TIER_MAP.get(y_pred, CreditRiskTier.POOR)

    logger.debug(f"Credit prediction: tier={tier}, scores={confidence_scores}")

    return CreditRiskResponse(
        credit_tier=tier,
        confidence_scores=confidence_scores,
        model_version=store.credit_model_version,
    )


def predict_combined(
    fraud_request: FraudPredictionRequest,
    credit_request: CreditRiskRequest,
    store: ModelStore,
) -> CombinedPredictionResponse:
    """
    Run both fraud detection and credit risk scoring in a single call.
    This is the showcase endpoint that mirrors real fintech production systems.
    """
    request_id = str(uuid.uuid4())

    fraud_result = predict_fraud(fraud_request, store)
    credit_result = predict_credit_risk(credit_request, store)

    return CombinedPredictionResponse(
        fraud=fraud_result,
        credit=credit_result,
        request_id=request_id,
    )

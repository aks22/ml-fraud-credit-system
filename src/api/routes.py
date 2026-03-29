"""
src/api/routes.py
------------------
FastAPI route handlers for all prediction endpoints.

Endpoints:
  POST /predict/fraud           — Fraud detection
  POST /predict/credit-risk     — Credit risk scoring
  POST /predict/combined        — Both models, one request (showcase endpoint)
  GET  /health                  — System health and model version
  GET  /                        — API information

All routes log predictions to PostgreSQL and expose metrics to Prometheus.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
    CreditRiskRequest,
    CreditRiskResponse,
    CombinedPredictionRequest,
    CombinedPredictionResponse,
    HealthResponse,
)
from src.api.prediction_service import predict_fraud, predict_credit_risk, predict_combined
from src.api.model_loader import model_store
from src.api.database import get_db, FraudPredictionLog, CreditPredictionLog
from src.utils.logger import logger

router = APIRouter()


# ── Helper: safe DB write ─────────────────────────────────────────────────────

def _log_fraud_prediction(
    request: FraudPredictionRequest,
    response: FraudPredictionResponse,
    request_id: str,
    db: Session,
) -> None:
    """Write fraud prediction to PostgreSQL. Silently fails — prediction still returned."""
    try:
        log = FraudPredictionLog(
            request_id=request_id,
            step=request.step,
            transaction_type=request.type.value,
            amount=request.amount,
            oldbalance_org=request.oldbalance_org,
            newbalance_orig=request.newbalance_orig,
            oldbalance_dest=request.oldbalance_dest,
            newbalance_dest=request.newbalance_dest,
            fraud_probability=response.fraud_probability,
            is_fraud=response.is_fraud,
            threshold=response.threshold,
            model_version=response.model_version,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log fraud prediction to DB: {e}")
        db.rollback()


def _log_credit_prediction(
    request: CreditRiskRequest,
    response: CreditRiskResponse,
    request_id: str,
    db: Session,
) -> None:
    """Write credit prediction to PostgreSQL. Silently fails."""
    try:
        log = CreditPredictionLog(
            request_id=request_id,
            age=request.age,
            duration_months=request.duration_months,
            credit_amount=request.credit_amount,
            employment=request.employment,
            purpose=request.purpose,
            credit_tier=response.credit_tier.value,
            confidence_good=response.confidence_scores.get("Good", 0),
            confidence_moderate=response.confidence_scores.get("Moderate", 0),
            confidence_poor=response.confidence_scores.get("Poor", 0),
            model_version=response.model_version,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log credit prediction to DB: {e}")
        db.rollback()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", tags=["Info"])
async def root():
    """API root — returns basic information about the system."""
    return {
        "name": "ML Fraud & Credit Risk System",
        "version": "1.0.0",
        "description": "Production ML API for fraud detection and credit risk scoring",
        "endpoints": {
            "fraud_detection": "POST /predict/fraud",
            "credit_risk": "POST /predict/credit-risk",
            "combined": "POST /predict/combined",
            "health": "GET /health",
            "docs": "GET /docs",
            "metrics": "GET /metrics",
        },
    }


@router.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """
    System health check.

    Returns model load status and version info.
    Use this endpoint to verify the system is ready before sending predictions.
    """
    status_val = "healthy" if model_store.fully_ready else (
        "degraded" if (model_store.fraud_ready or model_store.credit_ready) else "unhealthy"
    )

    return HealthResponse(
        status=status_val,
        fraud_model_loaded=model_store.fraud_ready,
        credit_model_loaded=model_store.credit_ready,
        fraud_model_version=model_store.fraud_model_version,
        credit_model_version=model_store.credit_model_version,
    )


@router.post(
    "/predict/fraud",
    response_model=FraudPredictionResponse,
    tags=["Predictions"],
    summary="Fraud Detection",
    description="Classify a financial transaction as fraudulent or legitimate.",
)
async def fraud_prediction(
    request: FraudPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Predict whether a transaction is fraudulent.

    Returns:
      - **fraud_probability**: Raw probability score (0–1)
      - **is_fraud**: Binary label (True = flagged as fraud)
      - **threshold**: Decision threshold used (default 0.5)
      - **model_version**: Which model version produced this result
    """
    if not model_store.fraud_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud model is not loaded. Please train models first.",
        )

    try:
        response = predict_fraud(request, model_store)
        _log_fraud_prediction(request, response, str(uuid.uuid4()), db)
        return response
    except Exception as e:
        logger.error(f"Fraud prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/credit-risk",
    response_model=CreditRiskResponse,
    tags=["Predictions"],
    summary="Credit Risk Scoring",
    description="Score a loan applicant into Good / Moderate / Poor credit risk tier.",
)
async def credit_risk_prediction(
    request: CreditRiskRequest,
    db: Session = Depends(get_db),
):
    """
    Predict credit risk tier for a loan applicant.

    Returns:
      - **credit_tier**: Good | Moderate | Poor
      - **confidence_scores**: Probability per tier
      - **model_version**: Model version used
    """
    if not model_store.credit_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credit model is not loaded. Please train models first.",
        )

    try:
        response = predict_credit_risk(request, model_store)
        _log_credit_prediction(request, response, str(uuid.uuid4()), db)
        return response
    except Exception as e:
        logger.error(f"Credit prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/combined",
    response_model=CombinedPredictionResponse,
    tags=["Predictions"],
    summary="Combined Fraud + Credit Risk (Showcase Endpoint)",
    description=(
        "Run both fraud detection and credit risk scoring in a single request. "
        "This mirrors what a bank's loan approval system would call in production: "
        "simultaneously flag transaction risk AND score applicant creditworthiness."
    ),
)
async def combined_prediction(
    request: CombinedPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    The showcase endpoint — demonstrates the full dual-model system.

    Accepts a combined payload with transaction data (for fraud) and
    applicant data (for credit risk), returns both predictions in one response.
    """
    if not model_store.fully_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="One or both models are not loaded.",
        )

    try:
        response = predict_combined(request.fraud, request.credit, model_store)

        # Log both predictions to DB
        _log_fraud_prediction(request.fraud, response.fraud, response.request_id, db)
        _log_credit_prediction(request.credit, response.credit, response.request_id, db)

        return response
    except Exception as e:
        logger.error(f"Combined prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

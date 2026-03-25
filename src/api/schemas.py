"""
src/api/schemas.py
-------------------
Pydantic models for FastAPI request and response validation.

Using Pydantic v2 syntax (which ships with FastAPI >= 0.100).
All fields include descriptions for auto-generated /docs page.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    CASH_IN = "CASH_IN"


class CreditRiskTier(str, Enum):
    GOOD = "Good"
    MODERATE = "Moderate"
    POOR = "Poor"


# ── Request Schemas ───────────────────────────────────────────────────────────

class FraudPredictionRequest(BaseModel):
    """Input features for a single fraud detection prediction."""

    step: int = Field(..., ge=1, le=743, description="Hour of the simulation (1–743)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0, description="Transaction amount in currency units")
    oldbalance_org: float = Field(..., ge=0, description="Sender balance before transaction")
    newbalance_orig: float = Field(..., ge=0, description="Sender balance after transaction")
    oldbalance_dest: float = Field(..., ge=0, description="Recipient balance before transaction")
    newbalance_dest: float = Field(..., ge=0, description="Recipient balance after transaction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalance_org": 181.0,
                "newbalance_orig": 0.0,
                "oldbalance_dest": 0.0,
                "newbalance_dest": 0.0,
            }
        }
    }


class CreditRiskRequest(BaseModel):
    """Input features for a single credit risk scoring prediction."""

    age: int = Field(..., ge=18, le=100, description="Applicant age")
    duration_months: int = Field(..., ge=1, le=84, description="Loan duration in months")
    credit_amount: float = Field(..., gt=0, description="Requested loan amount")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate as % of income")
    residence_since: int = Field(..., ge=1, le=4, description="Years at current residence")
    existing_credits: int = Field(..., ge=1, le=4, description="Number of existing credits")
    num_dependents: int = Field(..., ge=1, le=2, description="Number of dependents")
    checking_account: str = Field(..., description="Checking account status")
    credit_history: str = Field(..., description="Credit history category")
    purpose: str = Field(..., description="Purpose of the loan")
    savings: str = Field(..., description="Savings account/bonds status")
    employment: str = Field(..., description="Employment duration")
    housing: str = Field(..., description="Housing situation")
    job: str = Field(..., description="Job category")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "duration_months": 24,
                "credit_amount": 5000.0,
                "installment_rate": 2,
                "residence_since": 3,
                "existing_credits": 1,
                "num_dependents": 1,
                "checking_account": "0-200 DM",
                "credit_history": "existing_paid",
                "purpose": "car",
                "savings": "< 100 DM",
                "employment": "1-4 years",
                "housing": "own",
                "job": "skilled",
            }
        }
    }


class CombinedPredictionRequest(BaseModel):
    """Combined request for both fraud detection and credit risk scoring."""
    fraud: FraudPredictionRequest
    credit: CreditRiskRequest


# ── Response Schemas ──────────────────────────────────────────────────────────

class FraudPredictionResponse(BaseModel):
    """Fraud detection prediction result."""
    fraud_probability: float = Field(..., description="Probability that transaction is fraudulent (0–1)")
    is_fraud: bool = Field(..., description="Binary fraud flag (True if probability >= threshold)")
    threshold: float = Field(default=0.5, description="Decision threshold used")
    model_version: str = Field(..., description="Model version used for this prediction")


class CreditRiskResponse(BaseModel):
    """Credit risk scoring prediction result."""
    credit_tier: CreditRiskTier = Field(..., description="Predicted credit risk tier")
    confidence_scores: dict[str, float] = Field(..., description="Probability per risk tier")
    model_version: str = Field(..., description="Model version used for this prediction")


class CombinedPredictionResponse(BaseModel):
    """Combined fraud + credit risk prediction result — the showcase endpoint."""
    fraud: FraudPredictionResponse
    credit: CreditRiskResponse
    request_id: str = Field(..., description="Unique ID for this request (used for logging)")


class HealthResponse(BaseModel):
    """System health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    fraud_model_loaded: bool
    credit_model_loaded: bool
    fraud_model_version: str
    credit_model_version: str
    api_version: str = "1.0.0"

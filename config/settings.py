"""
config/settings.py
------------------
Central application configuration loaded from environment variables.
All modules import from here — never hardcode paths or credentials elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (if present)
load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


# ── Application ───────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
PORT: int = int(os.getenv("PORT", 8000))


# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://mluser:mlpassword@localhost:5432/fraud_credit_db"
)
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "fraud_credit_db")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "mluser")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "mlpassword")


# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-credit-system")


# ── Model Registry ────────────────────────────────────────────────────────────
FRAUD_MODEL_NAME: str = os.getenv("FRAUD_MODEL_NAME", "FraudDetector-v1")
CREDIT_MODEL_NAME: str = os.getenv("CREDIT_MODEL_NAME", "CreditRiskScorer-v1")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")


# ── Data Paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR: Path = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: Path = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODELS_DIR: Path = ROOT_DIR / os.getenv("MODELS_DIR", "outputs/models")
OUTPUTS_DIR: Path = ROOT_DIR / os.getenv("OUTPUTS_DIR", "outputs")

# Specific processed data files
CLEANED_FRAUD_PATH: Path = PROCESSED_DATA_DIR / "cleaned_fraud.csv"
CLEANED_CREDIT_PATH: Path = PROCESSED_DATA_DIR / "cleaned_credit.csv"

# Saved pipeline / model artefacts
FRAUD_PIPELINE_PATH: Path = MODELS_DIR / "fraud_pipeline.pkl"
CREDIT_PIPELINE_PATH: Path = MODELS_DIR / "credit_pipeline.pkl"
FRAUD_MODEL_PATH: Path = MODELS_DIR / "fraud_model.pkl"
CREDIT_MODEL_PATH: Path = MODELS_DIR / "credit_model.pkl"

# Training data snapshots (used by drift detection)
FRAUD_TRAIN_REFERENCE_PATH: Path = PROCESSED_DATA_DIR / "fraud_train_reference.csv"
CREDIT_TRAIN_REFERENCE_PATH: Path = PROCESSED_DATA_DIR / "credit_train_reference.csv"


# ── Monitoring ────────────────────────────────────────────────────────────────
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", 0.15))


# ── Create directories on import ──────────────────────────────────────────────
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

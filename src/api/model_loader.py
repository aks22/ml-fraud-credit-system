"""
src/api/model_loader.py
------------------------
Singleton model loader for FastAPI.

Loads both models (fraud + credit risk) and their preprocessing pipelines
on application startup. This ensures models are loaded once and reused
across all requests (not re-loaded on every request, which would be very slow).

Models are loaded from:
  1. Local .pkl files (fast, used in Docker / production)
  2. MLflow Model Registry (if local files are not found and MLflow is accessible)

Usage:
    from src.api.model_loader import ModelStore

    models = ModelStore()
    fraud_pipeline = models.fraud_pipeline
    credit_model = models.credit_model
"""

import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from config.settings import (
    FRAUD_PIPELINE_PATH,
    CREDIT_PIPELINE_PATH,
    FRAUD_MODEL_PATH,
    CREDIT_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    FRAUD_MODEL_NAME,
    CREDIT_MODEL_NAME,
    MODEL_STAGE,
)
from src.utils.logger import logger


@dataclass
class ModelStore:
    """
    Holds references to all loaded models and pipelines.
    Loaded once on app startup, shared across all API request handlers.
    """
    fraud_pipeline: Any = field(default=None, init=False)
    credit_pipeline: Any = field(default=None, init=False)
    fraud_model: Any = field(default=None, init=False)
    credit_model: Any = field(default=None, init=False)
    fraud_model_version: str = field(default="unknown", init=False)
    credit_model_version: str = field(default="unknown", init=False)

    def load_all(self) -> None:
        """Load all models. Called once at application startup.
        If models are already loaded (e.g. injected in tests), this is a no-op."""
        if self.fully_ready:
            logger.debug("Models already loaded — skipping load_all()")
            return
        self._load_fraud_assets()
        self._load_credit_assets()

    def _load_fraud_assets(self) -> None:
        """Load fraud pipeline + model from local files, or fall back to MLflow."""
        # Attempt local file load first
        if FRAUD_PIPELINE_PATH.exists() and FRAUD_MODEL_PATH.exists():
            self.fraud_pipeline = joblib.load(FRAUD_PIPELINE_PATH)
            self.fraud_model = joblib.load(FRAUD_MODEL_PATH)
            self.fraud_model_version = "local-pkl"
            logger.info(f"Fraud assets loaded from local files")
        else:
            logger.warning(
                f"Local fraud model not found. "
                "Run 'python src/models/train.py' to train models first, "
                "or ensure MLflow is accessible."
            )
            self._try_load_from_mlflow("fraud")

    def _load_credit_assets(self) -> None:
        """Load credit pipeline + model from local files, or fall back to MLflow."""
        if CREDIT_PIPELINE_PATH.exists() and CREDIT_MODEL_PATH.exists():
            self.credit_pipeline = joblib.load(CREDIT_PIPELINE_PATH)
            self.credit_model = joblib.load(CREDIT_MODEL_PATH)
            self.credit_model_version = "local-pkl"
            logger.info(f"Credit assets loaded from local files")
        else:
            logger.warning(
                f"Local credit model not found. "
                "Run 'python src/models/train.py' to train models first."
            )
            self._try_load_from_mlflow("credit")

    def _try_load_from_mlflow(self, model_type: str) -> None:
        """
        Attempt to load a model from the MLflow Model Registry.
        Silently fails if MLflow is not accessible (e.g. during testing).
        """
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_name = FRAUD_MODEL_NAME if model_type == "fraud" else CREDIT_MODEL_NAME
            model_uri = f"models:/{model_name}/{MODEL_STAGE}"

            if model_type == "fraud":
                self.fraud_model = mlflow.xgboost.load_model(model_uri)
                self.fraud_model_version = f"mlflow-{MODEL_STAGE}"
                logger.info(f"Fraud model loaded from MLflow: {model_uri}")
            else:
                self.credit_model = mlflow.sklearn.load_model(model_uri)
                self.credit_model_version = f"mlflow-{MODEL_STAGE}"
                logger.info(f"Credit model loaded from MLflow: {model_uri}")

        except Exception as e:
            logger.error(
                f"Failed to load {model_type} model from MLflow registry: {e}\n"
                "The API will start but predictions will fail until models are loaded."
            )

    @property
    def fraud_ready(self) -> bool:
        """True if fraud model and pipeline are both loaded."""
        return self.fraud_pipeline is not None and self.fraud_model is not None

    @property
    def credit_ready(self) -> bool:
        """True if credit model and pipeline are both loaded."""
        return self.credit_pipeline is not None and self.credit_model is not None

    @property
    def fully_ready(self) -> bool:
        return self.fraud_ready and self.credit_ready


# Global singleton — instantiated once when this module is imported
model_store = ModelStore()

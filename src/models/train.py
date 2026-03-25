"""
src/models/train.py
--------------------
Master training script — Phases 3 + 4.

Runs the complete training pipeline:
  1. Data cleaning (Phase 1)
  2. Feature engineering (Phase 2)
  3. Model training + evaluation (Phase 3)
  4. MLflow experiment tracking + model registry (Phase 4)

Usage:
    python src/models/train.py

    # With custom hyperparameters:
    python src/models/train.py --fraud-n-estimators 500 --credit-max-depth 15
"""

import argparse
import mlflow
from mlflow import MlflowClient

from config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    FRAUD_MODEL_NAME,
    CREDIT_MODEL_NAME,
)
from src.utils.logger import logger


def register_model(run_id: str, artifact_path: str, model_name: str, stage: str = "Staging") -> None:
    """
    Register a trained model in the MLflow Model Registry.

    Args:
        run_id: MLflow run ID that contains the model artefact.
        artifact_path: Path within the run to the model artefact.
        model_name: Registry name (e.g. 'FraudDetector-v1').
        stage: Initial stage ('Staging' or 'Production').
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info(f"Registering model '{model_name}' from run {run_id}")

    try:
        # Register the model (creates a new version)
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = model_version.version

        # Transition to the desired stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,  # Archive old versions of same stage
        )
        logger.info(f"Model '{model_name}' v{version} → stage: {stage}")
    except Exception as e:
        logger.warning(f"MLflow model registration failed (is MLflow server running?): {e}")
        logger.info("Tip: Start MLflow with: mlflow ui --port 5000")


def run_full_pipeline(
    fraud_params: dict | None = None,
    credit_params: dict | None = None,
    register_in_registry: bool = True,
) -> dict:
    """
    Execute the full training pipeline end-to-end.

    Args:
        fraud_params: Override params for fraud XGBoost model.
        credit_params: Override params for credit RF model.
        register_in_registry: Whether to register models in MLflow registry.

    Returns:
        Dict containing both models' metrics.
    """
    logger.info("=" * 60)
    logger.info("Starting full ML training pipeline")
    logger.info("=" * 60)

    # ── Phase 1: Data Ingestion & Cleaning ───────────────────────────────────
    logger.info("\n[Phase 1] Data Ingestion & Cleaning")
    from src.ingestion.clean_data import run_cleaning_pipeline
    fraud_df, credit_df = run_cleaning_pipeline()

    # ── Phase 2: Feature Engineering ─────────────────────────────────────────
    logger.info("\n[Phase 2] Feature Engineering")
    from src.features.pipeline import (
        prepare_fraud_data, prepare_credit_data,
        get_feature_names,
        FRAUD_NUMERIC_FEATURES, FRAUD_CATEGORICAL_FEATURES,
        CREDIT_NUMERIC_FEATURES, CREDIT_CATEGORICAL_FEATURES,
    )

    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, fraud_pipeline = prepare_fraud_data(fraud_df)
    X_credit_train, X_credit_test, y_credit_train, y_credit_test, credit_pipeline = prepare_credit_data(credit_df)

    credit_feature_names = get_feature_names(
        credit_pipeline, CREDIT_NUMERIC_FEATURES, CREDIT_CATEGORICAL_FEATURES
    )

    # ── Phase 3 + 4: Train, Evaluate, Track ──────────────────────────────────
    logger.info("\n[Phase 3+4] Fraud Model Training")
    from src.models.fraud.train_fraud import train_fraud_model, DEFAULT_PARAMS as FRAUD_DEFAULTS
    fraud_params_final = {**FRAUD_DEFAULTS, **(fraud_params or {})}
    fraud_model, fraud_metrics = train_fraud_model(
        X_fraud_train, y_fraud_train,
        X_fraud_test, y_fraud_test,
        params=fraud_params_final,
        run_name="fraud_xgboost_run",
    )

    logger.info("\n[Phase 3+4] Credit Risk Model Training")
    from src.models.credit_risk.train_credit import train_credit_model, DEFAULT_PARAMS as CREDIT_DEFAULTS
    credit_params_final = {**CREDIT_DEFAULTS, **(credit_params or {})}
    credit_model, credit_metrics = train_credit_model(
        X_credit_train, y_credit_train,
        X_credit_test, y_credit_test,
        feature_names=credit_feature_names,
        params=credit_params_final,
        run_name="credit_rf_run",
    )

    # ── Phase 4: Model Registry ───────────────────────────────────────────────
    if register_in_registry:
        logger.info("\n[Phase 4] Registering models in MLflow Registry")
        if "mlflow_run_id" in fraud_metrics:
            register_model(
                run_id=fraud_metrics["mlflow_run_id"],
                artifact_path="fraud_model",
                model_name=FRAUD_MODEL_NAME,
                stage="Staging",
            )
        if "mlflow_run_id" in credit_metrics:
            register_model(
                run_id=credit_metrics["mlflow_run_id"],
                artifact_path="credit_model",
                model_name=CREDIT_MODEL_NAME,
                stage="Staging",
            )

    all_metrics = {
        "fraud": {k: v for k, v in fraud_metrics.items() if k != "mlflow_run_id"},
        "credit": {k: v for k, v in credit_metrics.items() if k != "mlflow_run_id"},
    }

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Fraud metrics:  {all_metrics['fraud']}")
    logger.info(f"Credit metrics: {all_metrics['credit']}")
    logger.info("=" * 60)

    return all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection and credit risk models")
    parser.add_argument("--fraud-n-estimators", type=int, default=None)
    parser.add_argument("--fraud-max-depth", type=int, default=None)
    parser.add_argument("--fraud-learning-rate", type=float, default=None)
    parser.add_argument("--credit-n-estimators", type=int, default=None)
    parser.add_argument("--credit-max-depth", type=int, default=None)
    parser.add_argument("--no-registry", action="store_true", help="Skip MLflow model registry")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fraud_overrides = {}
    if args.fraud_n_estimators:
        fraud_overrides["n_estimators"] = args.fraud_n_estimators
    if args.fraud_max_depth:
        fraud_overrides["max_depth"] = args.fraud_max_depth
    if args.fraud_learning_rate:
        fraud_overrides["learning_rate"] = args.fraud_learning_rate

    credit_overrides = {}
    if args.credit_n_estimators:
        credit_overrides["n_estimators"] = args.credit_n_estimators
    if args.credit_max_depth:
        credit_overrides["max_depth"] = args.credit_max_depth

    run_full_pipeline(
        fraud_params=fraud_overrides or None,
        credit_params=credit_overrides or None,
        register_in_registry=not args.no_registry,
    )

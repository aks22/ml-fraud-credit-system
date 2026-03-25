"""
src/models/fraud/train_fraud.py
--------------------------------
Phase 3: Fraud detection model training and evaluation.

Algorithm: XGBoost classifier
Primary metric: Precision-Recall AUC (correct for severe class imbalance)
Secondary metrics: F1, ROC-AUC, confusion matrix

The training function is wrapped in an MLflow run context (Phase 4 integration)
so all parameters, metrics, and artefacts are automatically tracked.

Usage:
    python src/models/fraud/train_fraud.py
"""

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend — saves to file instead of displaying

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

from config.settings import (
    CLEANED_FRAUD_PATH,
    FRAUD_MODEL_PATH,
    OUTPUTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.utils.logger import logger


# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # scale_pos_weight: ratio of negative to positive class samples.
    # This is set dynamically based on the data, but we provide a fallback.
    "scale_pos_weight": 10,
    "eval_metric": "aucpr",       # Optimise for PR-AUC
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini Coefficient = 2 * ROC-AUC - 1. Industry standard for credit/fraud scoring."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def train_fraud_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
    run_name: str = "fraud_xgboost",
) -> tuple[XGBClassifier, dict]:
    """
    Train XGBoost fraud detection model with MLflow tracking.

    Args:
        X_train: SMOTE-resampled training features (already preprocessed).
        y_train: Training labels.
        X_test: Preprocessed test features (no SMOTE applied).
        y_test: Test labels.
        params: XGBoost hyperparameters. Defaults to DEFAULT_PARAMS.
        run_name: MLflow run name.

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    params = params or DEFAULT_PARAMS.copy()

    # Dynamically compute scale_pos_weight if not explicitly overridden
    if "scale_pos_weight" not in params or params["scale_pos_weight"] == 10:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        params["scale_pos_weight"] = round(neg / max(pos, 1), 2)
        logger.info(f"Computed scale_pos_weight: {params['scale_pos_weight']} ({neg:,} neg / {pos:,} pos)")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info(f"Training XGBoost fraud model with params: {params}")
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        gini = compute_gini(y_test, y_prob)

        metrics = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1_score": round(f1, 4),
            "gini_coefficient": round(gini, 4),
        }

        logger.info(f"Fraud model metrics: {metrics}")

        # ── MLflow logging ────────────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log model artefact to MLflow
        mlflow.xgboost.log_model(model, artifact_path="fraud_model")

        # ── Save evaluation report ────────────────────────────────────────────
        report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
        report_path = OUTPUTS_DIR / "fraud_evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== Fraud Detection Model Evaluation ===\n\n")
            f.write(f"Params: {params}\n\n")
            f.write(f"Metrics:\n  PR-AUC:  {pr_auc:.4f}\n  ROC-AUC: {roc_auc:.4f}\n")
            f.write(f"  F1:      {f1:.4f}\n  Gini:    {gini:.4f}\n\n")
            f.write(report)
        mlflow.log_artifact(str(report_path))
        logger.info(f"Saved evaluation report → {report_path}")

        # ── Precision-Recall curve chart ──────────────────────────────────────
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="steelblue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Fraud Detection — Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        chart_path = OUTPUTS_DIR / "fraud_pr_curve.png"
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(chart_path))
        logger.info(f"Saved PR curve → {chart_path}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legitimate", "Fraud"])
        ax.set_yticklabels(["Legitimate", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Fraud Detection — Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="black")
        fig.colorbar(im)
        cm_path = OUTPUTS_DIR / "fraud_confusion_matrix.png"
        fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # ── Save model locally ────────────────────────────────────────────────
        FRAUD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(FRAUD_MODEL_PATH).replace('.pkl', '.json'))
        joblib.dump(model, FRAUD_MODEL_PATH)
        logger.info(f"Saved fraud model → {FRAUD_MODEL_PATH}")

        # Return MLflow run ID for model registry step
        run_id = mlflow.active_run().info.run_id
        metrics["mlflow_run_id"] = run_id

    return model, metrics


if __name__ == "__main__":
    from src.ingestion.clean_data import run_cleaning_pipeline
    from src.features.pipeline import prepare_fraud_data

    logger.info("=== Phase 3: Fraud Model Training ===")
    fraud_df, _ = run_cleaning_pipeline()
    X_train, X_test, y_train, y_test, _ = prepare_fraud_data(fraud_df)
    model, metrics = train_fraud_model(X_train, y_train, X_test, y_test)
    logger.info(f"Training complete. Final metrics: {metrics}")

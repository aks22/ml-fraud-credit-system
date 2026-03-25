"""
src/models/credit_risk/train_credit.py
---------------------------------------
Phase 3: Credit risk scoring model training and evaluation.

Algorithm: scikit-learn RandomForestClassifier
Target: Multi-class — 0=Good, 1=Moderate, 2=Poor
Primary metric: Weighted F1 Score
Secondary metrics: Gini Coefficient per class, confusion matrix, feature importance

Usage:
    python src/models/credit_risk/train_credit.py
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend — saves to file instead of displaying

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config.settings import (
    CREDIT_MODEL_PATH,
    OUTPUTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.utils.logger import logger

CREDIT_RISK_LABELS = {0: "Good", 1: "Moderate", 2: "Poor"}

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced",   # Handles multi-class imbalance
    "random_state": 42,
    "n_jobs": -1,
}


def compute_gini_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute Gini Coefficient for each credit risk class using one-vs-rest ROC-AUC.
    Gini = 2 * AUC - 1.
    """
    gini_scores = {}
    n_classes = y_prob.shape[1]
    for i in range(n_classes):
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            gini_scores[CREDIT_RISK_LABELS[i]] = round(2 * auc - 1, 4)
        except Exception:
            gini_scores[CREDIT_RISK_LABELS[i]] = 0.0
    return gini_scores


def train_credit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
    params: dict | None = None,
    run_name: str = "credit_risk_random_forest",
) -> tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest credit risk model with MLflow tracking.

    Args:
        X_train: SMOTE-resampled training features.
        y_train: Training labels (0/1/2).
        X_test: Preprocessed test features.
        y_test: Test labels.
        feature_names: Feature names for importance chart.
        params: RF hyperparameters.
        run_name: MLflow run name.

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    params = params or DEFAULT_PARAMS.copy()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info(f"Training Random Forest credit model with params: {params}")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        gini_scores = compute_gini_per_class(y_test, y_prob)

        metrics = {
            "weighted_f1": round(weighted_f1, 4),
            "macro_f1": round(macro_f1, 4),
            **{f"gini_{k.lower()}": v for k, v in gini_scores.items()},
        }
        logger.info(f"Credit model metrics: {metrics}")

        # ── MLflow logging ────────────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="credit_model")

        # ── Save evaluation report ────────────────────────────────────────────
        report = classification_report(
            y_test, y_pred,
            target_names=[CREDIT_RISK_LABELS[i] for i in sorted(CREDIT_RISK_LABELS)]
        )
        report_path = OUTPUTS_DIR / "credit_evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== Credit Risk Scoring Model Evaluation ===\n\n")
            f.write(f"Params: {params}\n\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n")
            f.write(f"Gini Scores: {gini_scores}\n\n")
            f.write(report)
        mlflow.log_artifact(str(report_path))
        logger.info(f"Saved evaluation report → {report_path}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        labels = [CREDIT_RISK_LABELS[i] for i in sorted(CREDIT_RISK_LABELS)]
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Credit Risk — Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        fig.colorbar(im)
        cm_path = OUTPUTS_DIR / "credit_confusion_matrix.png"
        fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # ── Feature importance chart ──────────────────────────────────────────
        if feature_names and len(feature_names) == model.n_features_in_:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_features)), top_importances[::-1], color="steelblue")
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features[::-1])
            ax.set_xlabel("Feature Importance (Gini)")
            ax.set_title("Credit Risk — Top 20 Feature Importances")
            ax.grid(True, axis="x", alpha=0.3)
            imp_path = OUTPUTS_DIR / "credit_feature_importance.png"
            fig.savefig(imp_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(imp_path))
            logger.info(f"Saved feature importance chart → {imp_path}")

        # ── Save model locally ────────────────────────────────────────────────
        CREDIT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CREDIT_MODEL_PATH)
        logger.info(f"Saved credit model → {CREDIT_MODEL_PATH}")

        run_id = mlflow.active_run().info.run_id
        metrics["mlflow_run_id"] = run_id

    return model, metrics


if __name__ == "__main__":
    from src.ingestion.clean_data import run_cleaning_pipeline
    from src.features.pipeline import prepare_credit_data, get_feature_names

    logger.info("=== Phase 3: Credit Risk Model Training ===")
    _, credit_df = run_cleaning_pipeline()
    X_train, X_test, y_train, y_test, pipeline = prepare_credit_data(credit_df)

    from src.features.pipeline import CREDIT_NUMERIC_FEATURES, CREDIT_CATEGORICAL_FEATURES
    feature_names = get_feature_names(pipeline, CREDIT_NUMERIC_FEATURES, CREDIT_CATEGORICAL_FEATURES)

    model, metrics = train_credit_model(X_train, y_train, X_test, y_test, feature_names=feature_names)
    logger.info(f"Training complete. Final metrics: {metrics}")

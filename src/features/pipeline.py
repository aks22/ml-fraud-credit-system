"""
src/features/pipeline.py
-------------------------
Phase 2: Feature engineering pipeline.

Builds a scikit-learn / imblearn Pipeline that:
  1. Applies StandardScaler to numeric columns
  2. Applies OneHotEncoder to categorical columns
     (both via ColumnTransformer — processed simultaneously)
  3. Applies SMOTE to the training set only (handles class imbalance)

CRITICAL — DATA LEAKAGE PREVENTION:
  The pipeline is FITTED on training data ONLY.
  The fitted pipeline is then used to TRANSFORM both train and test sets.
  SMOTE is applied ONLY to the training set — never to test data.

The fitted pipelines are saved to disk with joblib so the API can
load them without re-training.

Usage:
    from src.features.pipeline import build_fraud_pipeline, build_credit_pipeline
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# imblearn Pipeline supports SMOTE inside pipeline steps
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config.settings import (
    CLEANED_FRAUD_PATH,
    CLEANED_CREDIT_PATH,
    FRAUD_PIPELINE_PATH,
    CREDIT_PIPELINE_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
from src.utils.logger import logger


# ── Feature column definitions ────────────────────────────────────────────────

FRAUD_NUMERIC_FEATURES = [
    "step", "amount", "oldbalance_org", "newbalance_orig",
    "oldbalance_dest", "newbalance_dest",
    "balance_error_orig", "balance_error_dest", "orig_balance_zeroed",
]

FRAUD_CATEGORICAL_FEATURES = ["type"]

FRAUD_TARGET = "is_fraud"


CREDIT_NUMERIC_FEATURES = [
    "age", "duration_months", "credit_amount",
    "installment_rate", "residence_since",
    "existing_credits", "num_dependents",
]

CREDIT_CATEGORICAL_FEATURES = [
    "checking_account", "credit_history", "purpose",
    "savings", "employment", "housing", "job",
]

CREDIT_TARGET = "credit_risk"


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Scales numeric features with StandardScaler (mean=0, std=1)
      - Encodes categorical features with OneHotEncoder
        (handle_unknown='ignore' avoids errors on unseen categories at inference)

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any columns not listed above
    )

    return preprocessor


def build_fraud_pipeline() -> ImbPipeline:
    """
    Build the fraud detection preprocessing pipeline.
    Includes SMOTE oversampling as the final step.

    The ImbPipeline from imbalanced-learn is used so that SMOTE
    is correctly integrated and only applied during fit(), not transform().

    Returns:
        ImbPipeline: preprocessor → SMOTE (ready to be fit on training data)
    """
    preprocessor = _build_preprocessor(FRAUD_NUMERIC_FEATURES, FRAUD_CATEGORICAL_FEATURES)

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        # SMOTE generates synthetic minority class samples.
        # random_state ensures reproducibility.
        # k_neighbors=5 is the default; reduce to 3 if minority class is very small.
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
    ])

    return pipeline


def build_credit_pipeline() -> ImbPipeline:
    """
    Build the credit risk preprocessing pipeline.
    Uses SMOTE for multi-class imbalance handling.

    Returns:
        ImbPipeline: preprocessor → SMOTE
    """
    preprocessor = _build_preprocessor(CREDIT_NUMERIC_FEATURES, CREDIT_CATEGORICAL_FEATURES)

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
    ])

    return pipeline


# ── Train/test split and fit ──────────────────────────────────────────────────

def prepare_fraud_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ImbPipeline]:
    """
    Split, preprocess, and SMOTE-oversample fraud training data.

    IMPORTANT: The pipeline is fit ONLY on X_train. X_test is transformed
    using the already-fitted pipeline. This prevents data leakage.

    Args:
        df: Cleaned fraud DataFrame (output of clean_fraud_data).
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train_resampled, X_test_transformed, y_train_resampled, y_test, fitted_pipeline
    """
    # Validate required columns exist
    missing = set(FRAUD_NUMERIC_FEATURES + FRAUD_CATEGORICAL_FEATURES + [FRAUD_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fraud DataFrame: {missing}")

    feature_cols = FRAUD_NUMERIC_FEATURES + FRAUD_CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[FRAUD_TARGET]

    # Split BEFORE fitting the pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Fraud split — Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Train fraud rate: {y_train.mean():.4%}"
    )

    # Build and fit pipeline on TRAINING DATA ONLY
    pipeline = build_fraud_pipeline()

    # fit_resample = fit preprocessor + apply SMOTE on training data
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

    logger.info(
        f"After SMOTE — Train samples: {len(X_train_res):,} | "
        f"Fraud rate: {y_train_res.mean():.4%}"
    )

    # Transform test data using the fitted preprocessor (no SMOTE on test data)
    # Access only the preprocessor step for transform
    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)

    # Save a reference copy of training data (used later by drift detection)
    ref_path = PROCESSED_DATA_DIR / "fraud_train_reference.csv"
    X_train.assign(is_fraud=y_train.values).to_csv(ref_path, index=False)
    logger.info(f"Saved fraud training reference → {ref_path}")

    # Save fitted pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, FRAUD_PIPELINE_PATH)
    logger.info(f"Saved fraud pipeline → {FRAUD_PIPELINE_PATH}")

    return X_train_res, X_test_transformed, y_train_res.to_numpy(), y_test.to_numpy(), pipeline


def prepare_credit_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ImbPipeline]:
    """
    Split, preprocess, and SMOTE-oversample credit risk training data.

    Returns:
        X_train_resampled, X_test_transformed, y_train_resampled, y_test, fitted_pipeline
    """
    missing = set(CREDIT_NUMERIC_FEATURES + CREDIT_CATEGORICAL_FEATURES + [CREDIT_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in credit DataFrame: {missing}")

    feature_cols = CREDIT_NUMERIC_FEATURES + CREDIT_CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[CREDIT_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Credit split — Train: {len(X_train):,} | Test: {len(X_test):,}")

    pipeline = build_credit_pipeline()
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

    logger.info(
        f"After SMOTE — Train samples: {len(X_train_res):,} | "
        f"Class distribution: {pd.Series(y_train_res).value_counts().to_dict()}"
    )

    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)

    ref_path = PROCESSED_DATA_DIR / "credit_train_reference.csv"
    X_train.assign(credit_risk=y_train.values).to_csv(ref_path, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, CREDIT_PIPELINE_PATH)
    logger.info(f"Saved credit pipeline → {CREDIT_PIPELINE_PATH}")

    return X_train_res, X_test_transformed, y_train_res.to_numpy(), y_test.to_numpy(), pipeline


def get_feature_names(pipeline: ImbPipeline, numeric_features: list, categorical_features: list) -> list[str]:
    """
    Extract the final feature names after ColumnTransformer encoding.
    Useful for feature importance charts.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_feature_names

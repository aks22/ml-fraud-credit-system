"""
src/ingestion/load_data.py
--------------------------
Phase 1: Data loading and schema validation.

Loads raw CSV files (or generates synthetic data if files are not present),
validates schema, and returns a clean DataFrame ready for cleaning.

Usage:
    from src.ingestion.load_data import load_fraud_data, load_credit_data
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from config.settings import RAW_DATA_DIR
from src.utils.logger import logger


# ── Expected schema definitions ───────────────────────────────────────────────

FRAUD_SCHEMA = {
    "step": "int64",
    "type": "object",
    "amount": "float64",
    "nameOrig": "object",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "object",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int64",
}

CREDIT_SCHEMA = {
    "age": "int64",
    "duration_months": "int64",
    "credit_amount": "float64",
    "installment_rate": "int64",
    "residence_since": "int64",
    "existing_credits": "int64",
    "num_dependents": "int64",
    "checking_account": "object",
    "credit_history": "object",
    "purpose": "object",
    "savings": "object",
    "employment": "object",
    "housing": "object",
    "job": "object",
    "credit_risk": "int64",
}


def _validate_schema(df: pd.DataFrame, expected_schema: dict, dataset_name: str) -> None:
    """
    Validate that the DataFrame contains the expected columns with compatible dtypes.

    Raises:
        ValueError: if required columns are missing or empty.
    """
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{dataset_name}] Missing required columns: {missing_cols}"
        )

    if df.empty:
        raise ValueError(f"[{dataset_name}] Dataset is empty after loading.")

    logger.debug(f"[{dataset_name}] Schema validation passed — {len(df):,} rows, {len(df.columns)} columns")


def _enforce_dtypes(df: pd.DataFrame, schema: dict, dataset_name: str) -> pd.DataFrame:
    """
    Coerce columns to their expected dtypes where possible.
    Logs a warning (not error) for columns that cannot be coerced.
    """
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        try:
            if dtype in ("int64", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "object":
                df[col] = df[col].astype(str)
        except Exception as e:
            logger.warning(f"[{dataset_name}] Could not coerce '{col}' to {dtype}: {e}")
    return df


def load_fraud_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw fraud transaction data from CSV.

    If the file is not found, automatically generates synthetic data
    (useful for development / CI without the full Kaggle dataset).

    Args:
        path: Override path to the CSV. Defaults to data/raw/fraud_transactions.csv.

    Returns:
        Raw (unprocessed) DataFrame.
    """
    file_path = path or (RAW_DATA_DIR / "fraud_transactions.csv")

    if not file_path.exists():
        logger.warning(
            f"Fraud data not found at {file_path}. "
            "Generating synthetic dataset. "
            "For production, download PaySim from Kaggle."
        )
        from src.utils.data_generator import generate_fraud_data
        df = generate_fraud_data()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Synthetic fraud data saved to {file_path}")
    else:
        logger.info(f"Loading fraud data from {file_path}")
        df = pd.read_csv(file_path)

    _validate_schema(df, FRAUD_SCHEMA, "FraudData")
    df = _enforce_dtypes(df, FRAUD_SCHEMA, "FraudData")

    logger.info(
        f"Fraud data loaded — {len(df):,} rows | "
        f"fraud rate: {df['isFraud'].mean():.4%}"
    )
    return df


def load_credit_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw credit risk applicant data from CSV.

    If the file is not found, automatically generates synthetic data.

    Args:
        path: Override path to the CSV. Defaults to data/raw/credit_risk.csv.

    Returns:
        Raw (unprocessed) DataFrame.
    """
    file_path = path or (RAW_DATA_DIR / "credit_risk.csv")

    if not file_path.exists():
        logger.warning(
            f"Credit data not found at {file_path}. "
            "Generating synthetic dataset. "
            "For production, download German Credit Dataset from UCI."
        )
        from src.utils.data_generator import generate_credit_data
        df = generate_credit_data()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Synthetic credit data saved to {file_path}")
    else:
        logger.info(f"Loading credit data from {file_path}")
        df = pd.read_csv(file_path)

    _validate_schema(df, CREDIT_SCHEMA, "CreditData")
    df = _enforce_dtypes(df, CREDIT_SCHEMA, "CreditData")

    distribution = df["credit_risk"].value_counts().to_dict()
    logger.info(f"Credit data loaded — {len(df):,} rows | risk distribution: {distribution}")
    return df

"""
src/ingestion/clean_data.py
----------------------------
Phase 1: Data cleaning pipeline.

Handles:
  - Null value imputation or row dropping (based on column importance)
  - Duplicate row removal
  - Column name standardisation to snake_case
  - Domain-specific feature creation (balance error flags for fraud)
  - Outlier capping for numeric features

IMPORTANT: This module never modifies files in data/raw/.
            All outputs are written to data/processed/.

Usage:
    from src.ingestion.clean_data import clean_fraud_data, clean_credit_data
"""

import re
import pandas as pd
import numpy as np
from config.settings import PROCESSED_DATA_DIR
from src.utils.logger import logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_snake_case(name: str) -> str:
    """Convert a camelCase or mixed column name to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s1.lower().replace(" ", "_").replace("-", "_")


def _standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to snake_case."""
    df.columns = [_to_snake_case(c) for c in df.columns]
    return df


def _log_null_summary(df: pd.DataFrame, dataset_name: str) -> None:
    """Log a summary of null values per column."""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if null_cols.empty:
        logger.debug(f"[{dataset_name}] No null values found.")
    else:
        logger.warning(f"[{dataset_name}] Null values found:\n{null_cols.to_string()}")


def _cap_outliers(df: pd.DataFrame, columns: list[str], lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Cap outliers at the specified percentiles (Winsorization).

    This prevents extreme values from dominating model training
    without removing entire rows.
    """
    for col in columns:
        if col not in df.columns:
            continue
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.debug(f"Capped '{col}' to [{lower:.2f}, {upper:.2f}]")
    return df


# ── Fraud Data Cleaning ───────────────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw fraud transaction data.

    Steps:
      1. Standardise column names to snake_case
      2. Drop exact duplicate rows
      3. Impute / drop nulls
      4. Engineer domain-specific features (balance error flags)
      5. Cap numeric outliers
      6. Drop irrelevant ID columns

    Args:
        df: Raw DataFrame from load_data.load_fraud_data()

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    original_len = len(df)
    logger.info(f"Starting fraud data cleaning — {original_len:,} rows")

    # 1. Standardise column names
    df = _standardise_column_names(df)

    # 2. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} duplicate rows")

    # 3. Handle nulls
    _log_null_summary(df, "FraudData")
    # Numeric columns: fill with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col in ("is_fraud",):
            # Target column: drop rows with null labels — cannot impute
            null_mask = df[col].isnull()
            if null_mask.any():
                logger.warning(f"Dropping {null_mask.sum()} rows with null target 'is_fraud'")
                df = df[~null_mask]
        else:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Imputed {null_count} nulls in '{col}' with median={median_val:.2f}")

    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug(f"Imputed {null_count} nulls in '{col}' with mode='{mode_val}'")

    # 4. Domain-specific feature engineering
    # Balance error: difference between expected and actual balance change.
    # Fraud transactions often show inconsistent balance updates.
    df["balance_error_orig"] = (
        df["oldbalance_org"] - df["amount"] - df["newbalance_orig"]
    ).abs()
    df["balance_error_dest"] = (
        df["oldbalance_dest"] + df["amount"] - df["newbalance_dest"]
    ).abs()
    # Binary flag: zero balance after transaction (account drained — fraud signal)
    df["orig_balance_zeroed"] = (df["newbalance_orig"] == 0).astype(int)

    # 5. Cap outliers for amount and balance columns
    amount_cols = [
        "amount", "oldbalance_org", "newbalance_orig",
        "oldbalance_dest", "newbalance_dest",
        "balance_error_orig", "balance_error_dest",
    ]
    df = _cap_outliers(df, amount_cols)

    # 6. Drop ID columns (not predictive, and not generalisable)
    id_cols = [c for c in ["name_orig", "name_dest"] if c in df.columns]
    df = df.drop(columns=id_cols, errors="ignore")

    logger.info(f"Fraud cleaning complete — {len(df):,} rows (removed {original_len - len(df):,})")
    return df


# ── Credit Data Cleaning ──────────────────────────────────────────────────────

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw credit risk applicant data.

    Steps:
      1. Standardise column names
      2. Drop duplicates
      3. Impute / drop nulls
      4. Validate domain constraints (age > 18, amount > 0)
      5. Cap numeric outliers

    Args:
        df: Raw DataFrame from load_data.load_credit_data()

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    original_len = len(df)
    logger.info(f"Starting credit data cleaning — {original_len:,} rows")

    # 1. Standardise column names
    df = _standardise_column_names(df)

    # 2. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} duplicate rows")

    # 3. Handle nulls
    _log_null_summary(df, "CreditData")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col == "credit_risk":
            null_mask = df[col].isnull()
            if null_mask.any():
                logger.warning(f"Dropping {null_mask.sum()} rows with null target 'credit_risk'")
                df = df[~null_mask]
        else:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    # 4. Domain constraint validation
    invalid_age = df["age"] < 18
    if invalid_age.any():
        logger.warning(f"Removing {invalid_age.sum()} rows with age < 18")
        df = df[~invalid_age]

    invalid_amount = df["credit_amount"] <= 0
    if invalid_amount.any():
        logger.warning(f"Removing {invalid_amount.sum()} rows with credit_amount <= 0")
        df = df[~invalid_amount]

    # 5. Cap outliers
    df = _cap_outliers(df, ["credit_amount", "age", "duration_months"])

    logger.info(f"Credit cleaning complete — {len(df):,} rows (removed {original_len - len(df):,})")
    return df


# ── Pipeline Entrypoint ───────────────────────────────────────────────────────

def run_cleaning_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Phase 1 pipeline: load → validate → clean → save.

    Returns:
        Tuple of (cleaned_fraud_df, cleaned_credit_df)
    """
    from src.ingestion.load_data import load_fraud_data, load_credit_data
    from config.settings import CLEANED_FRAUD_PATH, CLEANED_CREDIT_PATH

    # Load
    fraud_raw = load_fraud_data()
    credit_raw = load_credit_data()

    # Clean
    fraud_clean = clean_fraud_data(fraud_raw)
    credit_clean = clean_credit_data(credit_raw)

    # Save to processed/
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fraud_clean.to_csv(CLEANED_FRAUD_PATH, index=False)
    credit_clean.to_csv(CLEANED_CREDIT_PATH, index=False)

    logger.info(f"Saved cleaned fraud data → {CLEANED_FRAUD_PATH}")
    logger.info(f"Saved cleaned credit data → {CLEANED_CREDIT_PATH}")

    return fraud_clean, credit_clean


if __name__ == "__main__":
    run_cleaning_pipeline()

"""
tests/test_ingestion.py
------------------------
Unit tests for Phase 1: data loading and cleaning.

Run with:
    pytest tests/test_ingestion.py -v
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_fraud_df() -> pd.DataFrame:
    """Minimal valid fraud DataFrame for testing."""
    return pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [1000.0, 5000.0, 200.0],
        "nameOrig": ["C123", "C456", "C789"],
        "oldbalanceOrg": [1000.0, 5000.0, 200.0],
        "newbalanceOrig": [0.0, 0.0, 0.0],
        "nameDest": ["D123", "D456", "D789"],
        "oldbalanceDest": [0.0, 100.0, 50.0],
        "newbalanceDest": [1000.0, 5100.0, 250.0],
        "isFraud": [0, 1, 0],
    })


@pytest.fixture
def sample_credit_df() -> pd.DataFrame:
    """Minimal valid credit DataFrame for testing."""
    return pd.DataFrame({
        "age": [35, 25, 45],
        "duration_months": [24, 12, 36],
        "credit_amount": [5000.0, 2000.0, 10000.0],
        "installment_rate": [2, 1, 3],
        "residence_since": [3, 1, 4],
        "existing_credits": [1, 2, 1],
        "num_dependents": [1, 1, 2],
        "checking_account": ["0-200 DM", "no_account", ">= 200 DM"],
        "credit_history": ["existing_paid", "all_paid", "critical"],
        "purpose": ["car", "furniture", "business"],
        "savings": ["< 100 DM", "no_savings", "500-1000 DM"],
        "employment": ["1-4 years", "< 1 year", ">= 7 years"],
        "housing": ["own", "rent", "own"],
        "job": ["skilled", "unskilled", "highly_skilled"],
        "credit_risk": [0, 1, 2],
    })


# ── Tests: load_data ──────────────────────────────────────────────────────────

def test_fraud_data_generates_when_file_missing(tmp_path, monkeypatch):
    """When fraud CSV doesn't exist, synthetic data should be generated."""
    from src.ingestion import load_data
    monkeypatch.setattr(load_data, "RAW_DATA_DIR", tmp_path)

    df = load_data.load_fraud_data(path=tmp_path / "fraud_transactions.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "is_fraud" in df.columns or "isFraud" in df.columns


def test_credit_data_generates_when_file_missing(tmp_path, monkeypatch):
    """When credit CSV doesn't exist, synthetic data should be generated."""
    from src.ingestion import load_data
    monkeypatch.setattr(load_data, "RAW_DATA_DIR", tmp_path)

    df = load_data.load_credit_data(path=tmp_path / "credit_risk.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "credit_risk" in df.columns


def test_schema_validation_raises_on_missing_columns():
    """Schema validation should raise ValueError if required columns are missing."""
    from src.ingestion.load_data import _validate_schema, FRAUD_SCHEMA
    broken_df = pd.DataFrame({"wrong_column": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        _validate_schema(broken_df, FRAUD_SCHEMA, "TestData")


def test_schema_validation_raises_on_empty_dataframe():
    """Schema validation should raise ValueError on empty DataFrame."""
    from src.ingestion.load_data import _validate_schema, FRAUD_SCHEMA
    empty_df = pd.DataFrame(columns=list(FRAUD_SCHEMA.keys()))

    with pytest.raises(ValueError, match="empty"):
        _validate_schema(empty_df, FRAUD_SCHEMA, "TestData")


# ── Tests: clean_data ─────────────────────────────────────────────────────────

def test_clean_fraud_standardises_column_names(sample_fraud_df):
    """Column names should be converted to snake_case after cleaning."""
    from src.ingestion.clean_data import clean_fraud_data
    cleaned = clean_fraud_data(sample_fraud_df.copy())

    # camelCase columns should be snake_case now
    assert "old_balance_org" in cleaned.columns or "oldbalance_org" in cleaned.columns
    assert "is_fraud" in cleaned.columns


def test_clean_fraud_creates_balance_error_features(sample_fraud_df):
    """Fraud cleaning should create balance_error_orig and balance_error_dest."""
    from src.ingestion.clean_data import clean_fraud_data
    cleaned = clean_fraud_data(sample_fraud_df.copy())

    assert "balance_error_orig" in cleaned.columns
    assert "balance_error_dest" in cleaned.columns
    assert "orig_balance_zeroed" in cleaned.columns


def test_clean_fraud_removes_id_columns(sample_fraud_df):
    """ID columns (nameOrig, nameDest) should be dropped."""
    from src.ingestion.clean_data import clean_fraud_data
    cleaned = clean_fraud_data(sample_fraud_df.copy())

    assert "name_orig" not in cleaned.columns
    assert "name_dest" not in cleaned.columns


def test_clean_fraud_handles_duplicates():
    """Duplicate rows should be removed."""
    from src.ingestion.clean_data import clean_fraud_data
    df_with_dups = pd.DataFrame({
        "step": [1, 1],
        "type": ["PAYMENT", "PAYMENT"],
        "amount": [100.0, 100.0],
        "nameOrig": ["C1", "C1"],
        "oldbalanceOrg": [100.0, 100.0],
        "newbalanceOrig": [0.0, 0.0],
        "nameDest": ["D1", "D1"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [100.0, 100.0],
        "isFraud": [0, 0],
    })
    cleaned = clean_fraud_data(df_with_dups)
    assert len(cleaned) == 1


def test_clean_credit_removes_underage_applicants():
    """Applicants under 18 should be removed during cleaning."""
    from src.ingestion.clean_data import clean_credit_data
    df = pd.DataFrame({
        "age": [16, 30],
        "duration_months": [12, 24],
        "credit_amount": [1000.0, 5000.0],
        "installment_rate": [1, 2],
        "residence_since": [1, 3],
        "existing_credits": [1, 1],
        "num_dependents": [1, 1],
        "checking_account": ["0-200 DM", "no_account"],
        "credit_history": ["existing_paid", "all_paid"],
        "purpose": ["car", "furniture"],
        "savings": ["< 100 DM", "no_savings"],
        "employment": ["1-4 years", "< 1 year"],
        "housing": ["own", "rent"],
        "job": ["skilled", "unskilled"],
        "credit_risk": [0, 1],
    })
    cleaned = clean_credit_data(df)
    assert len(cleaned) == 1
    assert cleaned["age"].iloc[0] == 30


def test_clean_credit_handles_null_imputation():
    """Null values in numeric columns should be imputed with the median."""
    from src.ingestion.clean_data import clean_credit_data
    df = pd.DataFrame({
        "age": [35, None, 40],
        "duration_months": [24, 12, 36],
        "credit_amount": [5000.0, 2000.0, 10000.0],
        "installment_rate": [2, 1, 3],
        "residence_since": [3, 1, 4],
        "existing_credits": [1, 2, 1],
        "num_dependents": [1, 1, 2],
        "checking_account": ["0-200 DM", "no_account", ">= 200 DM"],
        "credit_history": ["existing_paid", "all_paid", "critical"],
        "purpose": ["car", "furniture", "business"],
        "savings": ["< 100 DM", "no_savings", "500-1000 DM"],
        "employment": ["1-4 years", "< 1 year", ">= 7 years"],
        "housing": ["own", "rent", "own"],
        "job": ["skilled", "unskilled", "highly_skilled"],
        "credit_risk": [0, 1, 2],
    })
    cleaned = clean_credit_data(df)
    assert cleaned["age"].isnull().sum() == 0


# ── Tests: full pipeline ──────────────────────────────────────────────────────

def test_processed_files_created(tmp_path, monkeypatch):
    """
    Running the cleaning pipeline should create cleaned CSV files in processed/.
    """
    from src.ingestion import clean_data
    from src.ingestion import load_data

    monkeypatch.setattr(clean_data, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(load_data, "RAW_DATA_DIR", tmp_path)

    # Patch the output paths
    monkeypatch.setattr("config.settings.CLEANED_FRAUD_PATH", tmp_path / "cleaned_fraud.csv")
    monkeypatch.setattr("config.settings.CLEANED_CREDIT_PATH", tmp_path / "cleaned_credit.csv")
    monkeypatch.setattr("config.settings.PROCESSED_DATA_DIR", tmp_path)

    fraud_df, credit_df = clean_data.run_cleaning_pipeline()

    assert isinstance(fraud_df, pd.DataFrame)
    assert isinstance(credit_df, pd.DataFrame)
    assert len(fraud_df) > 0
    assert len(credit_df) > 0
    # Target column must be present
    assert "is_fraud" in fraud_df.columns
    assert "credit_risk" in credit_df.columns

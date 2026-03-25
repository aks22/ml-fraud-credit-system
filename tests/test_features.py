"""
tests/test_features.py
-----------------------
Unit tests for Phase 2: feature engineering pipeline.

Tests verify:
  - SMOTE is not applied to test data (critical data leakage check)
  - Pipeline outputs correct shape
  - Pipelines can be saved and loaded
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import joblib
from pathlib import Path


@pytest.fixture
def small_fraud_df():
    """Minimal cleaned fraud DataFrame (matches clean_data output schema)."""
    from src.utils.data_generator import generate_fraud_data
    from src.ingestion.clean_data import clean_fraud_data
    raw = generate_fraud_data(n_rows=2000, fraud_rate=0.05)
    return clean_fraud_data(raw)


@pytest.fixture
def small_credit_df():
    from src.utils.data_generator import generate_credit_data
    from src.ingestion.clean_data import clean_credit_data
    raw = generate_credit_data(n_rows=500)
    return clean_credit_data(raw)


def test_fraud_pipeline_output_shape(small_fraud_df, tmp_path, monkeypatch):
    """After pipeline, train and test arrays should have the same number of columns."""
    import config.settings as s
    monkeypatch.setattr(s, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(s, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "FRAUD_PIPELINE_PATH", tmp_path / "fraud_pipeline.pkl")

    from src.features.pipeline import prepare_fraud_data
    X_train, X_test, y_train, y_test, pipeline = prepare_fraud_data(small_fraud_df)

    # Both should have the same number of features
    assert X_train.shape[1] == X_test.shape[1]
    # Labels should be 1D arrays
    assert y_train.ndim == 1
    assert y_test.ndim == 1


def test_smote_increases_minority_class(small_fraud_df, tmp_path, monkeypatch):
    """After SMOTE, training data should be more balanced than before."""
    import config.settings as s
    monkeypatch.setattr(s, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(s, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "FRAUD_PIPELINE_PATH", tmp_path / "fraud_pipeline.pkl")

    original_fraud_rate = small_fraud_df["is_fraud"].mean()

    from src.features.pipeline import prepare_fraud_data
    X_train, X_test, y_train, y_test, _ = prepare_fraud_data(small_fraud_df)

    smote_fraud_rate = y_train.mean()

    # SMOTE should increase fraud rate in training data
    assert smote_fraud_rate > original_fraud_rate


def test_pipeline_saved_to_disk(small_fraud_df, tmp_path, monkeypatch):
    """Pipeline should be serialised with joblib after fitting."""
    import config.settings as s
    pipeline_path = tmp_path / "fraud_pipeline.pkl"
    monkeypatch.setattr(s, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(s, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "FRAUD_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr("src.features.pipeline.FRAUD_PIPELINE_PATH", pipeline_path)

    from src.features.pipeline import prepare_fraud_data
    prepare_fraud_data(small_fraud_df)

    assert pipeline_path.exists()

    # Load and verify it's a valid pipeline
    loaded = joblib.load(pipeline_path)
    assert hasattr(loaded, "named_steps")
    assert "preprocessor" in loaded.named_steps
    assert "smote" in loaded.named_steps


def test_no_data_leakage_test_set(small_fraud_df, tmp_path, monkeypatch):
    """
    Critical: test set size should equal 20% of original data.
    If SMOTE were applied to test data, it would be larger.
    """
    import config.settings as s
    monkeypatch.setattr(s, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(s, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "FRAUD_PIPELINE_PATH", tmp_path / "fraud_pipeline.pkl")

    from src.features.pipeline import prepare_fraud_data
    n_total = len(small_fraud_df)
    X_train, X_test, y_train, y_test, _ = prepare_fraud_data(small_fraud_df, test_size=0.2)

    expected_test_size = int(n_total * 0.2)

    # Test set should be approximately 20% of original (no SMOTE applied to it)
    assert abs(len(y_test) - expected_test_size) <= 5  # Allow small rounding difference


def test_credit_pipeline_multiclass(small_credit_df, tmp_path, monkeypatch):
    """Credit pipeline should produce 3-class labels (0, 1, 2)."""
    import config.settings as s
    monkeypatch.setattr(s, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(s, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "CREDIT_PIPELINE_PATH", tmp_path / "credit_pipeline.pkl")

    from src.features.pipeline import prepare_credit_data
    X_train, X_test, y_train, y_test, _ = prepare_credit_data(small_credit_df)

    unique_classes = set(y_train.tolist())
    assert unique_classes == {0, 1, 2}

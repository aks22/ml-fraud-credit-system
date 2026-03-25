"""
src/monitoring/drift_report.py
--------------------------------
Phase 6: Data drift detection using Evidently AI.

Compares the distribution of live incoming prediction data against
the training data distribution. Generates an HTML drift report.

If drift exceeds the configured threshold, an alert is logged
(and can be extended to send email/Slack notifications).

Usage:
    python src/monitoring/drift_report.py

    # Or call from a cron job / Airflow DAG:
    from src.monitoring.drift_report import run_drift_detection
    run_drift_detection()
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from config.settings import (
    FRAUD_TRAIN_REFERENCE_PATH,
    CREDIT_TRAIN_REFERENCE_PATH,
    OUTPUTS_DIR,
    DRIFT_THRESHOLD,
    DATABASE_URL,
)
from src.utils.logger import logger


def load_live_fraud_data(n_recent: int = 1000) -> pd.DataFrame | None:
    """
    Load the most recent N fraud prediction inputs from PostgreSQL.
    These represent the live data distribution the model is seeing in production.
    """
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DATABASE_URL)
        query = f"""
            SELECT step, transaction_type as type, amount, 
                   oldbalance_org, newbalance_orig, 
                   oldbalance_dest, newbalance_dest
            FROM fraud_predictions
            ORDER BY timestamp DESC
            LIMIT {n_recent}
        """
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} recent fraud predictions from DB")
        return df
    except Exception as e:
        logger.warning(f"Could not load live fraud data from DB: {e}")
        return None


def load_live_credit_data(n_recent: int = 1000) -> pd.DataFrame | None:
    """Load the most recent N credit prediction inputs from PostgreSQL."""
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DATABASE_URL)
        query = f"""
            SELECT age, duration_months, credit_amount, employment, purpose
            FROM credit_predictions
            ORDER BY timestamp DESC
            LIMIT {n_recent}
        """
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} recent credit predictions from DB")
        return df
    except Exception as e:
        logger.warning(f"Could not load live credit data from DB: {e}")
        return None


def generate_fraud_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Generate Evidently AI drift report comparing reference vs current fraud data.

    Args:
        reference_df: Training data distribution (reference baseline).
        current_df: Live prediction data (current distribution).
        output_dir: Directory to save the HTML report.

    Returns:
        Dict with drift results summary.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(reference_data=reference_df, current_data=current_df)

        # Save HTML report
        report_path = output_dir / f"fraud_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(report_path))
        logger.info(f"Fraud drift report saved → {report_path}")

        # Extract drift summary
        report_dict = report.as_dict()
        drift_summary = _extract_drift_summary(report_dict)

        return {"report_path": str(report_path), **drift_summary}

    except ImportError:
        logger.error("Evidently AI not installed. Run: pip install evidently")
        return {"error": "evidently not installed"}
    except Exception as e:
        logger.error(f"Drift report generation failed: {e}")
        return {"error": str(e)}


def generate_credit_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Generate drift report for credit risk data."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        report_path = output_dir / f"credit_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(str(report_path))
        logger.info(f"Credit drift report saved → {report_path}")

        report_dict = report.as_dict()
        drift_summary = _extract_drift_summary(report_dict)

        return {"report_path": str(report_path), **drift_summary}
    except Exception as e:
        logger.error(f"Credit drift report failed: {e}")
        return {"error": str(e)}


def _extract_drift_summary(report_dict: dict) -> dict:
    """
    Extract key drift metrics from the Evidently report dictionary.
    Returns number of drifted features and overall drift share.
    """
    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                return {
                    "drift_detected": result.get("dataset_drift", False),
                    "drift_share": result.get("share_of_drifted_columns", 0.0),
                    "n_drifted_features": result.get("number_of_drifted_columns", 0),
                    "n_features": result.get("number_of_columns", 0),
                }
    except Exception:
        pass
    return {"drift_detected": False, "drift_share": 0.0}


def check_drift_alert(drift_results: dict, model_name: str) -> None:
    """
    Log an alert if drift exceeds the configured threshold.
    Extend this to send Slack / email notifications in production.
    """
    drift_share = drift_results.get("drift_share", 0.0)
    if drift_share > DRIFT_THRESHOLD:
        logger.warning(
            f"⚠️  DRIFT ALERT [{model_name}]: "
            f"{drift_share:.1%} of features have drifted "
            f"(threshold: {DRIFT_THRESHOLD:.1%}). "
            f"Consider retraining the model."
        )
    else:
        logger.info(
            f"✅  Drift check [{model_name}]: OK "
            f"(drift share: {drift_share:.1%} < threshold: {DRIFT_THRESHOLD:.1%})"
        )


def run_drift_detection() -> None:
    """
    Main drift detection pipeline.
    Loads reference data, loads live data, generates reports, checks thresholds.
    """
    output_dir = OUTPUTS_DIR / "drift_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fraud drift ───────────────────────────────────────────────────────────
    if FRAUD_TRAIN_REFERENCE_PATH.exists():
        reference_fraud = pd.read_csv(FRAUD_TRAIN_REFERENCE_PATH)
        live_fraud = load_live_fraud_data()

        if live_fraud is not None and len(live_fraud) >= 50:
            # Align columns
            common_cols = list(set(reference_fraud.columns) & set(live_fraud.columns))
            fraud_results = generate_fraud_drift_report(
                reference_df=reference_fraud[common_cols],
                current_df=live_fraud[common_cols],
                output_dir=output_dir,
            )
            check_drift_alert(fraud_results, "FraudDetector")
        else:
            logger.info("Not enough live fraud data for drift detection (need >= 50 samples)")
    else:
        logger.warning(
            f"Fraud training reference not found at {FRAUD_TRAIN_REFERENCE_PATH}. "
            "Train the model first."
        )

    # ── Credit drift ──────────────────────────────────────────────────────────
    if CREDIT_TRAIN_REFERENCE_PATH.exists():
        reference_credit = pd.read_csv(CREDIT_TRAIN_REFERENCE_PATH)
        live_credit = load_live_credit_data()

        if live_credit is not None and len(live_credit) >= 50:
            common_cols = list(set(reference_credit.columns) & set(live_credit.columns))
            credit_results = generate_credit_drift_report(
                reference_df=reference_credit[common_cols],
                current_df=live_credit[common_cols],
                output_dir=output_dir,
            )
            check_drift_alert(credit_results, "CreditRiskScorer")
        else:
            logger.info("Not enough live credit data for drift detection (need >= 50 samples)")
    else:
        logger.warning(
            f"Credit training reference not found at {CREDIT_TRAIN_REFERENCE_PATH}. "
            "Train the model first."
        )


if __name__ == "__main__":
    run_drift_detection()

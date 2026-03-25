"""
src/utils/data_generator.py
----------------------------
Generates synthetic fraud and credit risk datasets for development
and testing. This lets the pipeline run end-to-end without requiring
the full Kaggle datasets to be downloaded first.

The synthetic data mirrors the schema of the real datasets:
  - Fraud:  PaySim-style mobile transactions
  - Credit: German Credit Dataset-style applicant records

Run directly to generate and save data:
    python src/utils/data_generator.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config.settings import RAW_DATA_DIR
from src.utils.logger import logger


def generate_fraud_data(n_rows: int = 100_000, fraud_rate: float = 0.013, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic PaySim-style transaction data.

    Class imbalance: ~1.3% fraud (mirrors real PaySim distribution).

    Columns match PaySim schema:
        step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        nameDest, oldbalanceDest, newbalanceDest, isFraud
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_weights = [0.35, 0.08, 0.35, 0.14, 0.08]  # Approximate PaySim distribution

    # Legitimate transactions
    legit_amounts = rng.lognormal(mean=6.5, sigma=1.8, size=n_legit)
    legit_old_bal_orig = rng.lognormal(mean=9, sigma=2, size=n_legit)
    legit_new_bal_orig = np.maximum(0, legit_old_bal_orig - legit_amounts * rng.uniform(0.8, 1.0, n_legit))
    legit_old_bal_dest = rng.lognormal(mean=8, sigma=2.5, size=n_legit)
    legit_new_bal_dest = legit_old_bal_dest + legit_amounts * rng.uniform(0.7, 1.0, n_legit)

    # Fraudulent transactions — typically large TRANSFER or CASH_OUT
    fraud_amounts = rng.lognormal(mean=10, sigma=1.5, size=n_fraud)  # Larger amounts
    fraud_old_bal_orig = rng.lognormal(mean=10, sigma=1.5, size=n_fraud)
    fraud_new_bal_orig = np.zeros(n_fraud)  # Account emptied
    fraud_old_bal_dest = rng.lognormal(mean=5, sigma=2, size=n_fraud)
    fraud_new_bal_dest = fraud_old_bal_dest + fraud_amounts

    legit_df = pd.DataFrame({
        "step": rng.integers(1, 744, size=n_legit),
        "type": rng.choice(transaction_types, size=n_legit, p=type_weights),
        "amount": legit_amounts.round(2),
        "nameOrig": [f"C{rng.integers(1e9, 1e10)}" for _ in range(n_legit)],
        "oldbalanceOrg": legit_old_bal_orig.round(2),
        "newbalanceOrig": legit_new_bal_orig.round(2),
        "nameDest": [f"C{rng.integers(1e9, 1e10)}" for _ in range(n_legit)],
        "oldbalanceDest": legit_old_bal_dest.round(2),
        "newbalanceDest": legit_new_bal_dest.round(2),
        "isFraud": 0,
    })

    fraud_types = rng.choice(["TRANSFER", "CASH_OUT"], size=n_fraud)
    fraud_df = pd.DataFrame({
        "step": rng.integers(1, 744, size=n_fraud),
        "type": fraud_types,
        "amount": fraud_amounts.round(2),
        "nameOrig": [f"C{rng.integers(1e9, 1e10)}" for _ in range(n_fraud)],
        "oldbalanceOrg": fraud_old_bal_orig.round(2),
        "newbalanceOrig": fraud_new_bal_orig.round(2),
        "nameDest": [f"C{rng.integers(1e9, 1e10)}" for _ in range(n_fraud)],
        "oldbalanceDest": fraud_old_bal_dest.round(2),
        "newbalanceDest": fraud_new_bal_dest.round(2),
        "isFraud": 1,
    })

    df = pd.concat([legit_df, fraud_df], ignore_index=True).sample(frac=1, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Generated fraud dataset: {len(df):,} rows | {df['isFraud'].sum():,} fraud ({df['isFraud'].mean():.2%})")
    return df


def generate_credit_data(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic German Credit-style applicant data.

    Target variable: credit_risk — 0=Good, 1=Moderate, 2=Poor
    Approximate class distribution: 60% Good, 25% Moderate, 15% Poor
    """
    rng = np.random.default_rng(seed)

    # Simulate correlated features using a simple logistic relationship
    age = rng.integers(18, 75, size=n_rows)
    duration_months = rng.integers(6, 72, size=n_rows)
    credit_amount = rng.lognormal(mean=7.5, sigma=1.2, size=n_rows).round(2)
    installment_rate = rng.integers(1, 5, size=n_rows)
    residence_since = rng.integers(1, 5, size=n_rows)
    existing_credits = rng.integers(1, 5, size=n_rows)
    num_dependents = rng.integers(1, 3, size=n_rows)

    # Categorical features
    checking_account = rng.choice(
        ["no_account", "< 0 DM", "0-200 DM", ">= 200 DM"],
        size=n_rows,
        p=[0.39, 0.27, 0.27, 0.07],
    )
    credit_history = rng.choice(
        ["critical", "delay", "existing_paid", "all_paid", "no_credits"],
        size=n_rows,
        p=[0.29, 0.08, 0.53, 0.05, 0.05],
    )
    purpose = rng.choice(
        ["car", "furniture", "radio/TV", "domestic", "repairs", "education", "business", "vacation"],
        size=n_rows,
    )
    savings = rng.choice(
        ["no_savings", "< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM"],
        size=n_rows,
        p=[0.22, 0.60, 0.10, 0.06, 0.02],
    )
    employment = rng.choice(
        ["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"],
        size=n_rows,
        p=[0.04, 0.17, 0.34, 0.17, 0.28],
    )
    housing = rng.choice(["free", "rent", "own"], size=n_rows, p=[0.11, 0.18, 0.71])
    job = rng.choice(
        ["unemployed", "unskilled", "skilled", "highly_skilled"],
        size=n_rows,
        p=[0.02, 0.20, 0.63, 0.15],
    )

    # Risk score: higher = worse credit
    risk_score = (
        (credit_amount / 5000)
        + (duration_months / 12)
        + (installment_rate * 0.3)
        - (age * 0.01)
        + rng.normal(0, 0.5, size=n_rows)
    )

    # Assign credit risk tiers based on risk score percentiles
    low_thresh = np.percentile(risk_score, 60)
    mid_thresh = np.percentile(risk_score, 85)
    credit_risk = np.where(risk_score <= low_thresh, 0,
                  np.where(risk_score <= mid_thresh, 1, 2))

    df = pd.DataFrame({
        "age": age,
        "duration_months": duration_months,
        "credit_amount": credit_amount,
        "installment_rate": installment_rate,
        "residence_since": residence_since,
        "existing_credits": existing_credits,
        "num_dependents": num_dependents,
        "checking_account": checking_account,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings": savings,
        "employment": employment,
        "housing": housing,
        "job": job,
        "credit_risk": credit_risk,  # 0=Good, 1=Moderate, 2=Poor
    })

    logger.info(
        f"Generated credit dataset: {len(df):,} rows | "
        f"Good={int((credit_risk==0).sum()):,} "
        f"Moderate={int((credit_risk==1).sum()):,} "
        f"Poor={int((credit_risk==2).sum()):,}"
    )
    return df


if __name__ == "__main__":
    logger.info("Generating synthetic datasets...")

    fraud_df = generate_fraud_data(n_rows=100_000)
    fraud_path = RAW_DATA_DIR / "fraud_transactions.csv"
    fraud_df.to_csv(fraud_path, index=False)
    logger.info(f"Saved fraud data → {fraud_path}")

    credit_df = generate_credit_data(n_rows=10_000)
    credit_path = RAW_DATA_DIR / "credit_risk.csv"
    credit_df.to_csv(credit_path, index=False)
    logger.info(f"Saved credit data → {credit_path}")

    logger.info("Synthetic data generation complete.")

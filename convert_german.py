import pandas as pd
import numpy as np

cols = [
    "checking_account", "duration_months", "credit_history",
    "purpose", "credit_amount", "savings", "employment",
    "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker", "credit_risk"
]

df = pd.read_csv("data/raw/german.data", sep=" ", header=None, names=cols)

# Split Bad class into Moderate and Poor
# Logic: Bad credit + high loan amount = Poor (2)
#        Bad credit + low loan amount  = Moderate (1)
#        Good credit                   = Good (0)

median_amount = df[df["credit_risk"] == 2]["credit_amount"].median()

def assign_risk(row):
    if row["credit_risk"] == 1:
        return 0  # Good
    elif row["credit_risk"] == 2:
        if row["credit_amount"] >= median_amount:
            return 2  # Poor
        else:
            return 1  # Moderate
    return 0

df["credit_risk"] = df.apply(assign_risk, axis=1)

df.to_csv("data/raw/credit_risk.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/credit_risk.csv")
print(df["credit_risk"].value_counts().sort_index())
print(f"\nClass labels: 0=Good, 1=Moderate, 2=Poor")
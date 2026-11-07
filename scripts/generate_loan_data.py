"""Generate the synthetic ``data/loan_approval.csv`` sample (mixed types + missing values).

Run from the repository root:  python scripts/generate_loan_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 500

age = rng.integers(21, 66, n)
income = np.round(rng.lognormal(mean=10.6, sigma=0.45, size=n), -2)          # annual income
credit = np.clip(rng.normal(680, 70, n), 350, 850).round()
loan_amount = np.round(income * rng.uniform(0.2, 1.4, n), -2)
employment = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n, p=[.55, .22, .15, .08])
education = rng.choice(["Graduate", "Post-graduate", "High school"], n, p=[.5, .2, .3])
property_area = rng.choice(["Urban", "Semi-urban", "Rural"], n, p=[.4, .35, .25])
dependents = rng.choice(["0", "1", "2", "3+"], n, p=[.45, .25, .2, .1])

score = (
    (credit - 650) / 40
    + (income / loan_amount - 1.2) * 1.5
    + np.where(employment == "Unemployed", -2.5, 0)
    + np.where(employment == "Salaried", 0.6, 0)
    + np.where(education == "High school", -0.4, 0.2)
    + np.where(property_area == "Semi-urban", 0.4, 0)
    + rng.normal(0, 0.8, n)
)
approved = np.where(score > 0, "Approved", "Rejected")

df = pd.DataFrame({
    "applicant_id": [f"A{i:04d}" for i in range(1, n + 1)],
    "age": age,
    "annual_income": income,
    "loan_amount": loan_amount,
    "credit_score": credit,
    "employment_type": employment,
    "education": education,
    "property_area": property_area,
    "dependents": dependents,
    "loan_status": approved,
})

# inject ~5% missing values in four columns
for col in ["annual_income", "credit_score", "employment_type", "property_area"]:
    mask = rng.random(n) < 0.05
    df[col] = df[col].astype(object)
    df.loc[mask, col] = np.nan

out = Path(__file__).resolve().parents[1] / "data" / "loan_approval.csv"
df.to_csv(out, index=False)
print(f"wrote {out} {df.shape}; class balance:\n{df['loan_status'].value_counts()}")

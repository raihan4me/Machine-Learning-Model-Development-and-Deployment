"""Generate a synthetic loan-default dataset (optionally drifted)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int, seed: int, drift: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if drift:
        # New marketing campaign: younger, lower-income, thinner credit files
        age = rng.integers(19, 45, n)
        income = rng.lognormal(mean=10.8, sigma=0.4, size=n) * 0.75
        credit_score = np.clip(rng.normal(630, 60, n), 300, 850).astype(int)
        employment_years = np.clip(rng.exponential(3, n), 0, 40).round(1)
    else:
        age = rng.integers(21, 70, n)
        income = rng.lognormal(mean=10.8, sigma=0.4, size=n)
        credit_score = np.clip(rng.normal(680, 60, n), 300, 850).astype(int)
        employment_years = np.clip(rng.exponential(6, n), 0, 40).round(1)

    income = income.round(2)
    loan_amount = (rng.uniform(0.05, 0.6, n) * income).round(2)
    num_open_accounts = rng.poisson(4, n)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.4, 0.2, 0.4])

    logit = (
        -2.0
        + 3.0 * (loan_amount / income)
        - 0.012 * (credit_score - 680)
        - 0.05 * employment_years
        + 0.08 * num_open_accounts
        + 0.4 * (home_ownership == "RENT")
        + 0.8 * (age < 25)
        + rng.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "num_open_accounts": num_open_accounts,
        "home_ownership": home_ownership,
        "default": default,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--no-label", action="store_true", help="drop the target (production batch)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = generate(a.n, a.seed, a.drift)
    if a.no_label:
        df = df.drop(columns=["default"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")
    if "default" in df:
        print(f"default rate: {df['default'].mean():.3f}")

import pandas as pd
import numpy as np
from pathlib import Path


def generate_loan_data(
    n_samples: int = 5000, output_path: str = "loan_data.csv", seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income = rng.integers(20_000, 200_000, size=n_samples)
    loan_amount = rng.integers(500_000, 10_000_000, size=n_samples)
    interest_rate = rng.uniform(7, 12, size=n_samples)
    tenure = rng.integers(5, 30, size=n_samples)
    credit_score = rng.integers(300, 900, size=n_samples)
    existing_emi = rng.integers(0, 30_000, size=n_samples)
    employment_years = rng.integers(0, 35, size=n_samples)
    loan_purpose = rng.choice(
        ["home", "renovation", "plot", "construction"], size=n_samples
    )

    r = interest_rate / (12 * 100)
    n = tenure * 12
    emi = (loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1)

    emi_ratio = emi / income
    total_burden = (emi + existing_emi) / income
    ltv = loan_amount / (income * 12 * 5)

    risk_score = (
        (emi_ratio > 0.4).astype(int) * 3
        + (total_burden > 0.5).astype(int) * 2
        + (credit_score < 650).astype(int) * 2
        + (ltv > 0.8).astype(int) * 1
        + (employment_years < 2).astype(int) * 1
    )
    risk = (risk_score >= 3).astype(int)

    df = pd.DataFrame(
        {
            "income": income,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate.round(2),
            "tenure": tenure,
            "emi": emi.round(2),
            "credit_score": credit_score,
            "existing_emi": existing_emi,
            "employment_years": employment_years,
            "loan_purpose": loan_purpose,
            "emi_ratio": emi_ratio.round(4),
            "total_burden": total_burden.round(4),
            "ltv": ltv.round(4),
            "risk": risk,
        }
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} rows → {output_path}  (risk rate: {risk.mean():.1%})")
    return df


if __name__ == "__main__":
    generate_loan_data()

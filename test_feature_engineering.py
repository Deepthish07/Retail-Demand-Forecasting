import pandas as pd

from src.feature_engineering import FeatureEngineer


df = pd.read_excel(
    "data/processed/aggregated_sales.xlsx"
)

feature_engineer = FeatureEngineer(df)


sparsity_report = (
    feature_engineer
    .analyze_demand_sparsity()
)


print("\nDemand Sparsity Analysis:")

print(
    sparsity_report.to_string(
        index=False
    )
)
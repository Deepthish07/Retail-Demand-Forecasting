import pandas as pd

from src.feature_engineering import FeatureEngineer

df = pd.read_excel(
    "data/processed/modeling_sales_store_style.xlsx"
)

feature_engineer = FeatureEngineer(df)

calendar_df = (
    feature_engineer
    .create_complete_calendar()
)

feature_engineer.create_complete_calendar()
feature_df = feature_engineer.create_lag_features()

feature_engineer.validate_lag_features()

print("\nFeature Shape")
print(feature_df.shape)

print("\nColumns")
print(feature_df.columns)

print("\nFirst 20 Rows")
print(feature_df.head(20))
print("\nSample Store × Style")
print("-" * 80)

sample = feature_df[
    (feature_df["store"] == "TBBA - TBF028 - FIVESTAR APPARELS") &
    (feature_df["style"] == "1191")
]

print(
    sample[
        [
            "date",
            "qty",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28"
        ]
    ].head(40)
)

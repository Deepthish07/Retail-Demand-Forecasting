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
feature_df = feature_engineer.create_rolling_features()

feature_engineer.validate_rolling_features()
sample = feature_df[
    (feature_df["store"] == "TBBA - TBF028 - FIVESTAR APPARELS") &
    (feature_df["style"] == "1191")
]

print(
    sample[
        [
            "date",
            "qty",
            "rolling_mean_7",
            "rolling_sum_7",
            "rolling_std_7"
        ]
    ].head(40)
)
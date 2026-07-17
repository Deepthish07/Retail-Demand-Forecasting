import pandas as pd

from src.feature_engineering import FeatureEngineer


# Load aggregated dataset
df = pd.read_excel(
    "data/processed/aggregated_sales.xlsx"
)

# Initialize feature engineering
feature_engineer = FeatureEngineer(df)

# Create calendar features
feature_df = (
    feature_engineer
    .create_calendar_features()
)

print("Shape:", feature_df.shape)

print(
    feature_df[
        [
            "date",
            "year",
            "month",
            "quarter",
            "week_of_year",
            "day_of_month",
            "day_of_week",
            "is_weekend",
            "is_month_start",
            "is_month_end"
        ]
    ].head(10)
)
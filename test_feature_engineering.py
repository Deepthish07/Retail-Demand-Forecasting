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
feature_df = feature_engineer.create_date_features()
feature_df = feature_engineer.create_business_features()

feature_engineer.validate_business_features()

print(feature_df.shape)

print(feature_df.columns)
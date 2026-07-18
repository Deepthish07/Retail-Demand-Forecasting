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
feature_engineer.validate_calendar()
feature_engineer.add_calendar_features(calendar_df)
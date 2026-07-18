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

print("\nOriginal Shape:")
print(df.shape)

print("\nCalendar Shape:")
print(calendar_df.shape)

print("\nFirst 20 Rows")
print(calendar_df.head(20))
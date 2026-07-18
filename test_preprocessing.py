from src.preprocessing import RetailPreprocessor
from pathlib import Path
preprocessor = RetailPreprocessor()
preprocessor.load_data(Path("data/raw/one year ebo sale data.xlsx"))
preprocessor.detect_columns()
preprocessor.rename_columns()
df=preprocessor.remove_empty_columns()
df=preprocessor.standardize_text_columns()
df=preprocessor.convert_data_types()
validation_report = preprocessor.business_validation()
aggregated_df = preprocessor.aggregate_daily_sales()

output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

aggregated_df.to_excel(
    output_path / "aggregated_sales.xlsx",
    index=False
)

print("Aggregated dataset saved successfully.")
modeling_df = preprocessor.prepare_modeling_data()

print(
    "Detailed Aggregated Shape:",
    preprocessor.aggregated_df.shape
)

print(
    "Modeling Dataset Shape:",
    modeling_df.shape
)

state_check = (
    preprocessor.aggregated_df
    .groupby("store")["state"]
    .nunique()
    .sort_values(ascending=False)
)
store_check = (
    preprocessor.aggregated_df[
        preprocessor.aggregated_df["store"] == "TBBA - TBF028 - FIVESTAR APPARELS"
    ][["store", "state"]]
    .drop_duplicates()
)

print(store_check)
print(preprocessor.aggregated_df.columns.tolist())
from pathlib import Path

from src.preprocessing import RetailPreprocessor

preprocessor = RetailPreprocessor()

# ----------------------------
# Load Data
# ----------------------------
preprocessor.load_data(
    Path("data/raw/one year ebo sale data.xlsx")
)

# ----------------------------
# Preprocessing
# ----------------------------
preprocessor.detect_columns()
preprocessor.rename_columns()
preprocessor.remove_empty_columns()
preprocessor.standardize_text_columns()
preprocessor.convert_data_types()
preprocessor.business_validation()

# ----------------------------
# Aggregate Sales
# ----------------------------
aggregated_df = preprocessor.aggregate_daily_sales()

output_path = Path("data/processed")
output_path.mkdir(
    parents=True,
    exist_ok=True
)

aggregated_df.to_excel(
    output_path / "aggregated_sales.xlsx",
    index=False
)

# ----------------------------
# Prepare Modeling Dataset
# ----------------------------
modeling_df = preprocessor.prepare_modeling_data()

preprocessor.save_dataset(
    modeling_df,
    "modeling_sales_store_style.xlsx"
)
print("=" * 60)
print("CHECKING DUPLICATES")
print("=" * 60)

duplicate_count = modeling_df.duplicated(
    subset=["date", "store", "style"]
).sum()

print(f"Duplicate rows : {duplicate_count}")

# ----------------------------
# Product Master
# ----------------------------
product_master = preprocessor.create_product_master()

print("\nProduct Master Shape:")
print(product_master.shape)

print("\nFirst 10 Products")
print(product_master.head(10))
print("\nDuplicate Styles:")
print(product_master["style"].duplicated().sum())
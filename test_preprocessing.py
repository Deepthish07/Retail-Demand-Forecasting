from src.preprocessing import RetailPreprocessor
preprocessor = RetailPreprocessor()
preprocessor.load_data("data/raw/one year ebo sale data.xlsx")
preprocessor.detect_columns()
preprocessor.rename_columns()
df=preprocessor.remove_empty_columns()
df=preprocessor.standardize_text_columns()
df=preprocessor.convert_data_types()
validation_report = preprocessor.business_validation()

print("Validation Report")

for column, result in validation_report["columns"].items():

    print(
        f"{column}: {result['count']} violations"
    )

print(
    f"\nTotal Violations: "
    f"{validation_report['total_violations']}"
)

print(
    f"Status: {validation_report['status']}"
)
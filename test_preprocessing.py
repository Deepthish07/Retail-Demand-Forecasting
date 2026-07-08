from src.preprocessing import RetailPreprocessor
preprocessor = RetailPreprocessor()
preprocessor.load_data("data/raw/one year ebo sale data.xlsx")
preprocessor.detect_columns()
df=preprocessor.rename_columns()
print(df.columns.tolist())
print(df.head())

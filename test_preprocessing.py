from src.preprocessing import RetailPreprocessor
preprocessor = RetailPreprocessor()
preprocessor.load_data("data/raw/one year ebo sale data.xlsx")
preprocessor.detect_columns()
preprocessor.rename_columns()
df=preprocessor.remove_empty_columns()
print(df.columns.tolist())




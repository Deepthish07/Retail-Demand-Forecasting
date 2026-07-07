from src.preprocessing import RetailPreprocessor
preprocessor = RetailPreprocessor()
sale_data=preprocessor.load_data("data/raw/one year ebo sale data.xlsx")
mapping = preprocessor.detect_columns()
print(mapping)

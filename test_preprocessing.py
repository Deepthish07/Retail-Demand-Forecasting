from src.preprocessing import RetailPreprocessor
preprocessor = RetailPreprocessor()
sale_data=preprocessor.load_data("data/raw/one year ebo sale data.xlsx")
print(sale_data.shape)
print(sale_data.head())


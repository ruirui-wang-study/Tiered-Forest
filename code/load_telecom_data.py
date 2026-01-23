import pandas as pd

# Load the telecom terminology dataset
df = pd.read_excel('C:/good/11/data/1.xlsx')

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

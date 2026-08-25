import pandas as pd

df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nClass distribution:")
print(df["Class"].value_counts())
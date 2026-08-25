import pandas as pd
from predict import predict_transaction

# Load dataset
df = pd.read_csv("creditcard.csv")

# Test one normal transaction
normal_transaction = df[df["Class"] == 0].iloc[0]
normal_data = normal_transaction.drop("Class").to_dict()

print("NORMAL TRANSACTION TEST")
print(predict_transaction(normal_data))


# Test one fraud transaction
fraud_transaction = df[df["Class"] == 1].iloc[0]
fraud_data = fraud_transaction.drop("Class").to_dict()

print("\nFRAUD TRANSACTION TEST")
print(predict_transaction(fraud_data))
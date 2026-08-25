import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

THRESHOLD = 0.4


def predict_transaction(transaction_data):
    """
    Predict whether a transaction is fraudulent.
    """

    # Convert input into DataFrame
    df = pd.DataFrame([transaction_data])

    # Scale Time and Amount
    df[["Time", "Amount"]] = scaler.transform(
        df[["Time", "Amount"]]
    )

    # Get fraud probability
    fraud_probability = model.predict_proba(df)[0][1]

    # Make prediction using custom threshold
    prediction = 1 if fraud_probability >= THRESHOLD else 0

    # Risk level
    if fraud_probability >= 0.8:
        risk = "HIGH"
    elif fraud_probability >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "prediction": "FRAUD" if prediction == 1 else "NORMAL",
        "risk_level": risk
    }
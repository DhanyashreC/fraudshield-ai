from fastapi import FastAPI, UploadFile, File, HTTPException
from src.predict import predict_transaction
import pandas as pd
import io


app = FastAPI(
    title="FraudShield AI",
    description="Credit Card Fraud Detection API",
    version="1.0"
)


# Features expected by the trained model
REQUIRED_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to FraudShield AI",
        "status": "API is running"
    }


# CSV prediction endpoint
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

    # Check whether the uploaded file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    # Read the uploaded file
    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV file."
        )

    # Check whether all required columns are present
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing_columns}"
        )

    # Store prediction results
    results = []

    # Predict each transaction
    for index, row in df.iterrows():

        # Select only the required features
        transaction_data = row[REQUIRED_COLUMNS].to_dict()

        # Get prediction from our trained model
        prediction = predict_transaction(transaction_data)

        # Store result
        results.append({
            "transaction_id": int(index),
            "amount": float(row["Amount"]),
            **prediction
        })

    # Count fraud transactions
    fraud_alerts = sum(
        1
        for result in results
        if result["prediction"] == "FRAUD"
    )

    # Return final response
    return {
        "total_transactions": len(df),
        "fraud_alerts": fraud_alerts,
        "normal_transactions": len(df) - fraud_alerts,
        "results": results
    }
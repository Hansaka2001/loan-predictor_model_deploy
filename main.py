from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize App
app = FastAPI(title="Loan Approval AI")

# 2. load the model
model = joblib.load("loan_model.pkl")

# 3. Define the Input Data Structure

class LoanApplication(BaseModel):
    income: int
    credit_score: int

# 4. Define the Prediction Endpoint

@app.post("/predict")
def predict_loan_approval(application: LoanApplication):
    try:
        data = pd.DataFrame([{
            'income': application.income,
            'credit_score': application.credit_score
        }])

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0].tolist()

        return {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "confidence_score": max(probability),
            "details": {
                "income_input": application.income,
                "credit_score_input": application.credit_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Loan Approval API is running! Go to /docs to test."}

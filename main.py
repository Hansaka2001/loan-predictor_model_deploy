from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize App
app = FastAPI(title="Loan Approval AI")

# 2. Load the Random Forest Model
# We load it here so it stays in memory (fast!)
model = joblib.load("loan_model.pkl")

# 3. Define the Input Data Structure
# This matches the inputs we used in Colab: Income & Credit Score
class LoanApplication(BaseModel):
    income: int       # e.g., 60
    credit_score: int # e.g., 700

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_loan_approval(application: LoanApplication):
    try:
        # Prepare data exactly how the model expects it (DataFrame)
        # Note: We must use the same column names as we did in training: 'income', 'credit_score'
        data = pd.DataFrame([{
            'income': application.income, 
            'credit_score': application.credit_score
        }])
        
        # Make Prediction
        # result will be [1] (Approved) or [0] (Rejected)
        prediction = model.predict(data)[0]
        
        # Get Probability (Confidence score)
        # probe_result will look like [0.1, 0.9] (10% reject chance, 90% approve chance)
        probability = model.predict_proba(data)[0].tolist() 
        
        return {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "confidence_score": max(probability), # Returns the higher probability
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
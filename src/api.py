from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import sys
sys.path.append("./src")
from preprocess import preprocess_data

# Load model
model = joblib.load("models/model_pipeline.joblib")

app = FastAPI()

class InputData(BaseModel):
    # Accepts dict for single sample
    data: dict

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Preprocess input
        processed = preprocess_data(input_data.data, is_single_sample=True)
        prediction = model.predict(processed)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

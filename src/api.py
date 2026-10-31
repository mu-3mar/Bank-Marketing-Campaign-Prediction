from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import joblib
from pathlib import Path

from src.preprocess import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
model = joblib.load(PROJECT_ROOT / "models" / "model_pipeline.joblib")

app = FastAPI(title="Bank Marketing Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://0.0.0.0:5500",
        "null",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    # Accepts dict for single sample
    data: dict

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Preprocess input
        processed = preprocess_data(input_data.data, is_single_sample=True)
        prediction = model.predict(processed)
        probability = float(model.predict_proba(processed)[0, 1])
        return {
            "prediction": int(prediction[0]),
            "probability": round(probability, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_ROOT / "index.html")

app.mount("/", StaticFiles(directory=FRONTEND_ROOT), name="frontend")

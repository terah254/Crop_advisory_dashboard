from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
import traceback
from dotenv import load_dotenv
from train_model import train_and_save
from weather import fetch_forecast, fetch_current
from trees import analyze_farm_image
from model import engineer_features, generate_advisory

# Auto-train model if not already trained
MODEL_PATH = "/models/risk_model.pkl"
if not os.path.exists(MODEL_PATH):
    train_and_save()

risk_model = joblib.load(MODEL_PATH)
LABELS     = ["Low Risk", "Moderate Risk", "High Risk"]
FEATURE_COLS = ["temp_max", "temp_min", "precipitation_sum", "precipitation_probability", "wind_max"]


# Load environment variables
load_dotenv()  # Railway injects env vars directly, no .env file needed, but this allows local dev with a .env file containing WEATHER_AI_KEY=your_key_here

app = FastAPI(
    title="Crop Advisory API",
    description="ML-powered crop risk scoring using WeatherAI geo-intelligence APIs",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained model once at startup
MODEL_PATH = "../models/risk_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the notebook first.")

risk_model = joblib.load(MODEL_PATH)
LABELS     = ["Low Risk", "Moderate Risk", "High Risk"]
FEATURE_COLS = ["temp_max", "temp_min", "precipitation_sum", "precipitation_probability", "wind_max"]


#  Health check 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": risk_model is not None}


# Main analysis endpoint
import traceback

@app.post("/analyze")
async def analyze(
    lat:         float      = Form(...),
    lon:         float      = Form(...),
    county:      str        = Form(""),
    land_acres:  float      = Form(None),
    image:       UploadFile = File(None)
):
    try:
      
        forecast_raw = await fetch_forecast(lat, lon, days=7)
        current_raw  = await fetch_current(lat, lon)

        daily_list = (
            forecast_raw.get("daily") or
            forecast_raw.get("forecast") or
            []
        )

        if not daily_list:
            raise HTTPException(status_code=500, detail="No forecast data returned from API.")

        df = engineer_features(daily_list)


        X            = df[FEATURE_COLS].values
        predictions  = risk_model.predict(X)
        risk_labels  = [LABELS[p] for p in predictions]
        risk_probs   = risk_model.predict_proba(X).tolist()

        df["risk_label"]      = risk_labels
        df["risk_confidence"] = [round(max(p), 2) for p in risk_probs]

        tree_analysis = None
        if image and image.filename:
            img_bytes     = await image.read()
            tree_analysis = await analyze_farm_image(
                img_bytes, image.filename, county, land_acres
            )

        high_risk_days = sum(1 for l in risk_labels if l == "High Risk")
        advisory       = generate_advisory(df, tree_analysis, high_risk_days)

        return {
            "location":     {"lat": lat, "lon": lon, "county": county},
            "current":      current_raw,
            "forecast":     df.to_dict(orient="records"),
            "risk_summary": {
                "high_risk_days": high_risk_days,
                "risk_labels":    risk_labels,
                "risk_probs":     risk_probs
            },
            "tree_analysis": tree_analysis,
            "advisory":      advisory
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
   
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# Forecast only endpoint
@app.get("/forecast")
async def forecast(lat: float, lon: float, days: int = 7):
    try:
        data = await fetch_forecast(lat, lon, days)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
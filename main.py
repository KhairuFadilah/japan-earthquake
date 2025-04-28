from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os
from fastapi.responses import JSONResponse

app = FastAPI(title="Earthquake Magnitude Prediction API")

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load scaler dan model saat startup
@app.on_event("startup")
async def load_model():
    global scaler, model
    if not os.path.exists("scaler.pkl") or not os.path.exists("model.pkl"):
        raise FileNotFoundError("Scaler or Model file not found.")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

# Schema input
class InputData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    nst: float
    gap: float
    rms: float
    depthError: float
    magNst: float
    distance_from_tokyo_km: float
    magType_m: bool
    magType_mb: bool
    magType_ms: bool
    magType_mwb: bool
    magType_mwc: bool
    magType_mwr: bool
    magType_mww: bool

# Endpoint home
@app.get("/")
async def home():
    return {"message": "Welcome to Earthquake Magnitude Prediction API!"}

# Endpoint prediksi
@app.post("/predict")
async def predict(input_data: InputData):
    # Convert input ke DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Rename kolom supaya sesuai dengan model
    input_df.rename(columns={
        'distance_from_tokyo_km': 'distance_from_tokyo (km)'
    }, inplace=True)

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)

    return {"predicted_magnitude": float(prediction[0])}

# Tambahkan di atas /predict atau setelah home()
@app.get("/earthquake-data")
async def get_earthquake_data():
    df = pd.read_json("earthquake_data.json")  # Pastikan file JSON ada
    data = df.to_dict(orient="records")
    return JSONResponse(content=data)

# Endpoint sederhana untuk cek status
@app.get("/status")
async def status():
    return {"status": "API is running", "model_loaded": model is not None}

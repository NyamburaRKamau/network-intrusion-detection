from prometheus_fastapi_instrumentator import Instrumentator
from src.preprocess import load_artifacts, preprocess_input
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Network Intrusion Detection API",
    description="Detects malicious network traffic using ML",
    version="1.0.0"
)

# Load model and scaler on startup
model = None
scaler = None


@app.on_event("startup")
def load_model():
    global model, scaler
    model, scaler = load_artifacts()


# Prometheus monitoring
Instrumentator().instrument(app).expose(app)


class NetworkConnection(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: float
    wrong_fragment: float
    urgent: float
    hot: float
    num_failed_logins: float
    logged_in: float
    num_compromised: float
    root_shell: float
    su_attempted: float
    num_root: float
    num_file_creations: float
    num_shells: float
    num_access_files: float
    num_outbound_cmds: float
    is_host_login: float
    is_guest_login: float
    count: float
    srv_count: float
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float


@app.get("/")
def root():
    return {"message": "Network Intrusion Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(connection: NetworkConnection):
    try:
        data = connection.dict()
        processed = preprocess_input(data)
        scaled = scaler.transform(processed)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0]

        return {
            "prediction": int(prediction),
            "label": "ATTACK" if prediction == 1 else "NORMAL",
            "confidence": round(float(max(probability)), 4),
            "attack_probability": round(float(probability[1]), 4),
            "normal_probability": round(float(probability[0]), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import joblib
import numpy as np
import pandas as pd
import os

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "intrusion_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])

    # Label encode categorical columns
    # These mappings match what we used during training
    protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 0}
    flag_map = {
        'REJ': 0, 'SF': 1, 'RSTO': 2, 'S0': 3, 'RSTR': 4,
        'SH': 5, 'S3': 6, 'S2': 7, 'S1': 8, 'RSTOS0': 9, 'OTH': 10
    }

    df['protocol_type'] = df['protocol_type'].map(protocol_map)
    df['flag'] = df['flag'].map(flag_map)
    df['service'] = df['service'].astype('category').cat.codes

    return df.values

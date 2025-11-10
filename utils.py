
import os, io, tempfile
import numpy as np
import joblib
from feature_extractor import extract_feature

MODEL_PATH = os.path.join("model", "classifier.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
LABELS_PATH = os.path.join("model", "labels.npy")

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    return model, scaler, labels

def predict_from_file(file_path):
    model, scaler, labels = load_model()
    feat = extract_feature(file_path)
    X = scaler.transform([feat])
    proba = None
    try:
        proba = model.predict_proba(X)[0]
    except Exception:
        proba = None
    pred = model.predict(X)[0]
    return pred, proba, labels

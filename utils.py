import os
import numpy as np
import joblib
import librosa
from feature_extractor import extract_feature_raw

# ============================
# パス解決（このファイル基準）
# ============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

# ============================
# キャッシュ
# ============================
_model = None
_scaler = None
_labels = None

def _ensure_files_exist():
    missing = []
    for p in (MODEL_PATH, SCALER_PATH, LABELS_PATH):
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

def load_model():
    """
    モデル・スケーラ・ラベルを一度だけロード（キャッシュ）
    """
    global _model, _scaler, _labels
    if _model is None or _scaler is None or _labels is None:
        _ensure_files_exist()
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    return _model, _scaler, _labels

def predict_from_file(file_path):
    """
    音声ファイル → 特徴量 → スケーリング → 感情予測
    戻り値:
      pred_label : str
      proba      : np.ndarray | None
      labels     : list[str]
    """
    model, scaler, labels = load_model()

    # 音声ロード
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 特徴量抽出（学習と同一）
    feat = extract_feature_raw(y, sr)
    X = scaler.transform([feat])

    # 予測
    raw_pred = model.predict(X)[0]

    try:
        proba = model.predict_proba(X)[0]
    except Exception:
        proba = None

    # ラベル整形（SVC は文字列を直接返す想定）
    pred_label = str(raw_pred)

    return pred_label, proba, labels

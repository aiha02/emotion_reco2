import os
import numpy as np
import joblib
from feature_extractor import extract_feature

# model ディレクトリをこのファイル基準で解決（カレントディレクトリ依存を避ける）
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

# キャッシュ用（プロセス単位）
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
    モデル・スケーラ・ラベルをキャッシュして返す。
    予測ごとにファイルを再ロードしないようにする。
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
    file_path の音声ファイルから特徴量抽出→スケーリング→予測を行う。
    戻り値:
      pred_label: 予測されたラベル文字列（labels に基づく）
      proba: 各ラベルの確率配列（predict_proba が使えない場合は None）
      labels: ラベル一覧（predict 出力の順序）
    """
    model, scaler, labels = load_model()
    feat = extract_feature(file_path)
    X = scaler.transform([feat])

    proba = None
    try:
        proba = model.predict_proba(X)[0]
    except Exception:
        proba = None

    raw_pred = model.predict(X)[0]

    # raw_pred が index なのかラベル文字列なのか両方に対応してラベル文字列へ変換
    pred_label = None
    try:
        # numpy integer の可能性もあるので int 変換を試みる
        if isinstance(raw_pred, (int, np.integer)):
            pred_label = labels[int(raw_pred)]
        else:
            # raw_pred がラベル文字列ならそのまま。数値文字列の場合に備えて int 変換も試す。
            if raw_pred in labels:
                pred_label = raw_pred
            else:
                try:
                    idx = int(raw_pred)
                    pred_label = labels[idx]
                except Exception:
                    pred_label = str(raw_pred)
    except Exception:
        pred_label = str(raw_pred)

    return pred_label, proba, labels

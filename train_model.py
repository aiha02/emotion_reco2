import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from feature_extractor import extract_feature_raw
import librosa

# =========================
# クラス統合ルール
# =========================
EMOTION_MAP = {
    "JOY": "POS",
    "ACC": "POS",

    "ANG": "NEG_H",
    "DIS": "NEG_H",

    "SAD": "NEG_L",
    "FEA": "NEG_L",

    "NEU": "NEU",

    "ANT": "OTH",
    "SUR": "OTH",
    "OTH": "OTH",
}

def map_emotion(label):
    return EMOTION_MAP.get(label, None)

# =========================
# パス設定
# =========================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# データ読み込み
# =========================
df = pd.read_csv(CATEGORY_FILE)
df = df.dropna(subset=["fid", "ans1"])
df["emotion"] = df["ans1"].apply(map_emotion)
df = df.dropna(subset=["emotion"])

X, y = [], []

for _, row in df.iterrows():
    wav_path = os.path.join(WAV_DIR, f"{row['fid']}.wav")
    if not os.path.exists(wav_path):
        continue

    try:
        y_audio, sr = librosa.load(wav_path, sr=None, mono=True)
        feat = extract_feature_raw(y_audio, sr)
        X.append(feat)
        y.append(row["emotion"])
    except Exception as e:
        print(f"Error: {row['fid']} {e}")

X = np.array(X)
y = np.array(y)

print("サンプル数:", len(y))
print("クラス分布:", Counter(y))

# =========================
# スケーリング + SVM
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

# =========================
# 交差検証
# =========================
min_count = min(Counter(y).values())
n_splits = min(5, min_count)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv)

print("CV accuracy:", scores)
print("CV mean:", scores.mean())

# =========================
# 全データで学習
# =========================
clf.fit(X_scaled, y)

# =========================
# 保存
# =========================
joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
np.save(os.path.join(MODEL_DIR, "labels.npy"), np.unique(y))

print("学習完了")

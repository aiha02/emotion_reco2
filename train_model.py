import argparse
import pandas as pd
import numpy as np
import librosa
import os
import sys
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

# =========================================================
# パス
# =========================================================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
TRANS_DIR = os.path.join(DATASET_DIR, "trans")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")

CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")
INTENSITY_FILE = os.path.join(EVAL_DIR, "intensity.txt")

MODEL_DIR = "model"
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "classifier.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.npy")

# =========================================================
# クラス統合ルール
# =========================================================
EMOTION_MAP = {
    "JOY": "POS", "ACC": "POS",
    "ANG": "NEG_H", "DIS": "NEG_H",
    "SAD": "NEG_L", "FEA": "NEG_L",
    "NEU": "NEU",
    "ANT": "OTH", "SUR": "OTH", "OTH": "OTH",
}

def map_emotion(label):
    return EMOTION_MAP.get(label, None)

# =========================================================
# Δ / ΔΔ 特徴量抽出（260次元）
# =========================================================
def extract_feature_raw(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_m = mfcc.mean(axis=1)
    delta_m = delta.mean(axis=1)
    delta2_m = delta2.mean(axis=1)

    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).mean(axis=1)

    return np.hstack([mfcc_m, delta_m, delta2_m, chroma, mel])

# =========================================================
# 既存モデルチェック
# =========================================================
if not args.force and os.path.exists(CLASSIFIER_FILE):
    print("既存モデルあり。--force で再学習")
    sys.exit(0)

# =========================================================
# category.txt（多数決）
# =========================================================
cat = pd.read_csv(CATEGORY_FILE)
cat = cat.dropna(subset=["fid", "uid", "ans1", "ans2", "ans3"])

def majority(row):
    return Counter([row["ans1"], row["ans2"], row["ans3"]]).most_common(1)[0][0]

cat["raw_label"] = cat.apply(majority, axis=1)
cat["label"] = cat["raw_label"].apply(map_emotion)
cat = cat.dropna(subset=["label"])

category_map = {
    (row.fid, str(row.uid).zfill(3)): row.label
    for _, row in cat.iterrows()
}

# =========================================================
# intensity.txt（平均）
# =========================================================
inten = pd.read_csv(INTENSITY_FILE)
cols = [c for c in inten.columns if c.startswith("E")]
inten["intensity"] = inten[cols].mean(axis=1)

intensity_map = {
    (row.fid, str(row.uid).zfill(3)): row.intensity
    for _, row in inten.iterrows()
}

# =========================================================
# 発話単位特徴抽出
# =========================================================
X, y, inten_list = [], [], []
short_skip = 0
nolabel_skip = 0

for trans_file in os.listdir(TRANS_DIR):
    if not trans_file.endswith(".txt"):
        continue

    fid = trans_file.replace(".txt", "")
    wav_path = os.path.join(WAV_DIR, f"{fid}.wav")
    trans_path = os.path.join(TRANS_DIR, trans_file)

    if not os.path.exists(wav_path):
        continue

    audio, sr = librosa.load(wav_path, sr=None)

    with open(trans_path, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 3:
                continue

            uid, start, end = cols[0], float(cols[1]), float(cols[2])

            if end - start < 0.2:
                short_skip += 1
                continue

            key = (fid, uid)
            if key not in category_map:
                nolabel_skip += 1
                continue

            seg = audio[int(start*sr):int(end*sr)]
            feat = extract_feature_raw(seg, sr)

            X.append(feat)
            y.append(category_map[key])
            inten_list.append(intensity_map.get(key, 3.0))

print(f"抽出発話数: {len(X)}")
print(f"短発話除外: {short_skip}")
print(f"ラベルなし除外: {nolabel_skip}")

X = np.array(X)
y = np.array(y)

# =========================================================
# クラス数チェック
# =========================================================
print("クラス分布:", Counter(y))
valid = {k for k, v in Counter(y).items() if v >= 2}

mask = [lab in valid for lab in y]
X = X[mask]
y = y[mask]

# =========================================================
# 交差検証
# =========================================================
min_cnt = min(Counter(y).values())
n_splits = min(5, min_cnt)
n_splits = max(n_splits, 2)

pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# =========================================================
# 最終学習
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
clf.fit(X_scaled, y)

print("Training accuracy:", clf.score(X_scaled, y))

# =========================================================
# 保存
# =========================================================
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, CLASSIFIER_FILE)
joblib.dump(scaler, SCALER_FILE)
np.save(LABELS_FILE, np.unique(y))

print("🎉 クラス統合 + Δ/ΔΔ 学習完了")

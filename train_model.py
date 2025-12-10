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
# CLI: --force があると再学習
# =========================================================
parser = argparse.ArgumentParser(description="Train emotion recognition model using utterance-level data.")
parser.add_argument("--force", action="store_true", help="Force retraining.")
args = parser.parse_args()

# =========================================================
# パス設定
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
# extract_feature_raw (numpy array 対応)
# =========================================================
def extract_feature_raw(y, sr):
    """Return 1D feature vector: MFCC(40) + Chroma(12) + Mel(128) = 180 dims"""
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# =========================================================
# 既に学習済みならスキップ
# =========================================================
if not args.force and os.path.exists(CLASSIFIER_FILE):
    print("学習済みモデルが存在します。--force で再学習できます。")
    sys.exit(0)

# =========================================================
# category.txt 読み込み（多数決）
# =========================================================
cat = pd.read_csv(CATEGORY_FILE)
cat = cat.dropna(subset=["fid", "uid", "ans1", "ans2", "ans3"])

def majority_label(row):
    labels = [row["ans1"], row["ans2"], row["ans3"]]
    cnt = Counter(labels)
    return cnt.most_common(1)[0][0]  # 最頻値

cat["label"] = cat.apply(majority_label, axis=1)

category_map = {(row.fid, str(row.uid).zfill(3)): row.label for _, row in cat.iterrows()}

# =========================================================
# intensity.txt 読み込み（平均を取る）
# =========================================================
inten = pd.read_csv(INTENSITY_FILE)
inten_cols = [c for c in inten.columns if c.startswith("E")]
inten["intensity"] = inten[inten_cols].mean(axis=1)

intensity_map = {(row.fid, str(row.uid).zfill(3)): row.intensity for _, row in inten.iterrows()}

# =========================================================
# 発話単位の特徴量抽出
# =========================================================
X, y, inten_list = [], [], []
short_skipped = 0
no_label_skipped = 0

for trans_file in os.listdir(TRANS_DIR):
    if not trans_file.endswith(".txt"):
        continue

    fid = trans_file.replace(".txt", "")
    wav_path = os.path.join(WAV_DIR, f"{fid}.wav")
    trans_path = os.path.join(TRANS_DIR, trans_file)

    if not os.path.exists(wav_path):
        print(f"⚠️ Missing wav: {wav_path}")
        continue

    # 音声ロード
    audio, sr = librosa.load(wav_path, sr=None)

    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 5:
                continue

            uid, start, end = cols[0], float(cols[1]), float(cols[2])

            # 時間が短すぎる発話は除外
            if end - start < 0.2:
                short_skipped += 1
                continue

            seg = audio[int(start * sr): int(end * sr)]

            key = (fid, uid)
            if key not in category_map:
                no_label_skipped += 1
                continue

            label = category_map[key]
            intensity = intensity_map.get(key, 3.0)  # なければ中間値3で補完

            feat = extract_feature_raw(seg, sr)

            X.append(feat)
            y.append(label)
            inten_list.append(intensity)

print(f"抽出完了: {len(X)} samples")
print(f"短すぎて除外: {short_skipped}")
print(f"ラベルなし除外: {no_label_skipped}")

X = np.array(X)
y = np.array(y)
inten_arr = np.array(inten_list)

# =========================================================
# 各クラスのサンプル数チェック
# =========================================================
print("Label count:", Counter(y))

valid_classes = {lab for lab, cnt in Counter(y).items() if cnt >= 2}
X = np.array([x for x, lab in zip(X, y) if lab in valid_classes])
inten_arr = np.array([i for i, lab in zip(inten_arr, y) if lab in valid_classes])
y = np.array([lab for lab in y if lab in valid_classes])

print("After filtering:", Counter(y))

# =========================================================
# 交差検証
# =========================================================
min_cnt = min(Counter(y).values())
n_splits = min(5, min_cnt)
if n_splits < 2:
    n_splits = 2

print(f"Using n_splits={n_splits}")

pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# =========================================================
# 全データで最終学習
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
clf.fit(X_scaled, y)

acc = clf.score(X_scaled, y)
print(f"Training accuracy: {acc:.4f}")

# =========================================================
# 保存
# =========================================================
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(clf, CLASSIFIER_FILE)
joblib.dump(scaler, SCALER_FILE)
np.save(LABELS_FILE, np.unique(y))

print("Model saved")

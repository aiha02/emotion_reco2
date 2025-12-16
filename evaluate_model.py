import os
import joblib
import numpy as np
import pandas as pd
import librosa
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================
# 1. パス
# ============================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
TRANS_DIR = os.path.join(DATASET_DIR, "trans")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")

CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")
INTENSITY_FILE = os.path.join(EVAL_DIR, "intensity.txt")

MODEL_DIR = "model"

# ============================
# 2. モデル読み込み
# ============================
clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
labels = np.load(os.path.join(MODEL_DIR, "labels.npy"), allow_pickle=True)

# ============================
# 3. クラス統合ルール（学習と同一）
# ============================
EMOTION_MAP = {
    "JOY": "POS", "ACC": "POS",
    "ANG": "NEG_H", "DIS": "NEG_H",
    "SAD": "NEG_L", "FEA": "NEG_L",
    "NEU": "NEU",
    "ANT": "OTH", "SUR": "OTH", "OTH": "OTH",
}

def map_emotion(label):
    return EMOTION_MAP.get(label, None)

# ============================
# 4. 特徴量抽出（Δ / ΔΔ 含む：260次元）
# ============================
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

# ============================
# 5. category（多数決 → クラス統合）
# ============================
cat = pd.read_csv(CATEGORY_FILE)

def majority(row):
    return Counter([row.ans1, row.ans2, row.ans3]).most_common(1)[0][0]

cat["raw_label"] = cat.apply(majority, axis=1)
cat["label"] = cat["raw_label"].apply(map_emotion)
cat = cat.dropna(subset=["label"])

label_map = {
    (r.fid, str(r.uid).zfill(3)): r.label
    for _, r in cat.iterrows()
}

# ============================
# 6. intensity（平均）
# ============================
inten = pd.read_csv(INTENSITY_FILE)
inten_cols = [c for c in inten.columns if c.startswith("E")]
inten["mean_intensity"] = inten[inten_cols].mean(axis=1)

inten_map = {
    (r.fid, str(r.uid).zfill(3)): r.mean_intensity
    for _, r in inten.iterrows()
}

# ============================
# 7. 発話単位で評価データ作成
# ============================
X, y_true, intensity_true = [], [], []

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
                continue

            key = (fid, uid)
            if key not in label_map:
                continue

            seg = audio[int(start * sr): int(end * sr)]
            feat = extract_feature_raw(seg, sr)

            X.append(feat)
            y_true.append(label_map[key])
            intensity_true.append(inten_map.get(key, 3.0))

X = np.array(X)
y_true = np.array(y_true)
intensity_true = np.array(intensity_true)

print("評価サンプル数:", len(y_true))
print("クラス分布:", Counter(y_true))

# ============================
# 8. 予測
# ============================
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)

# ============================
# 9. 分類評価
# ============================
acc = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true, y_pred, labels=labels, digits=4, zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nAccuracy:", acc)
print("\nClassification Report:\n", report)

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

pd.DataFrame(cm, index=labels, columns=labels).to_csv(
    os.path.join(MODEL_DIR, "confusion_matrix.csv")
)

# ============================
# 10. 感情強度 MAE
# ============================
proba = clf.predict_proba(X_scaled)
pred_intensity = proba.max(axis=1) * 5  # 疑似強度（1〜5）

mae = mean_absolute_error(intensity_true, pred_intensity)
print("Intensity MAE:", mae)

# ============================
# 11. t-SNE（論文用図）
# ============================
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_embedded = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for lab in labels:
    idx = np.where(y_true == lab)
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=lab, s=10)

plt.legend()
plt.title("t-SNE of Utterance-level Emotion Features (Merged Classes)")
plt.savefig(os.path.join(MODEL_DIR, "tsne_plot.png"))
plt.close()

print("評価結果保存完了")

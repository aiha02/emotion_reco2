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
labels = np.load(os.path.join(MODEL_DIR, "labels.npy"))

# ============================
# 3. 特徴量抽出（学習と同一）
# ============================
def extract_feature_raw(y, sr):
    stft = np.abs(librosa.stft(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# ============================
# 4. category（多数決）
# ============================
cat = pd.read_csv(CATEGORY_FILE)

def majority(row):
    return Counter([row.ans1, row.ans2, row.ans3]).most_common(1)[0][0]

cat["label"] = cat.apply(majority, axis=1)
label_map = {(r.fid, str(r.uid).zfill(3)): r.label for _, r in cat.iterrows()}

# ============================
# 5. intensity（平均）
# ============================
inten = pd.read_csv(INTENSITY_FILE)
inten_cols = [c for c in inten.columns if c.startswith("E")]
inten["mean_intensity"] = inten[inten_cols].mean(axis=1)
inten_map = {(r.fid, str(r.uid).zfill(3)): r.mean_intensity for _, r in inten.iterrows()}

# ============================
# 6. 発話単位で評価データ作成
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
            uid, start, end, *_ = line.strip().split(",")
            start, end = float(start), float(end)

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
# 7. 予測
# ============================
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)

# ============================
# 8. 分類評価（卒論必須）
# ============================
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4, zero_division=0)
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
# 9. 強度誤差（卒論で差がつく）
# ============================
proba = clf.predict_proba(X_scaled)
pred_intensity = proba.max(axis=1) * 5  # 疑似強度

mae = mean_absolute_error(intensity_true, pred_intensity)
print("Intensity MAE:", mae)

# ============================
# 10. t-SNE（論文用図）
# ============================
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_embedded = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for lab in labels:
    idx = np.where(y_true == lab)
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=lab, s=10)

plt.legend()
plt.title("t-SNE of Utterance-level Emotion Features")
plt.savefig(os.path.join(MODEL_DIR, "tsne_plot.png"))
plt.close()

print("評価結果保存完了")

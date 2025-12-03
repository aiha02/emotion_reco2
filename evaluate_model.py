import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from feature_extractor import extract_feature
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================
# 1) モデル読み込み
# ============================
clf = joblib.load("model/classifier.pkl")
scaler = joblib.load("model/scaler.pkl")
labels = np.load("model/labels.npy")

# ============================
# 2) データセット読み込み
# ============================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")

df = pd.read_csv(CATEGORY_FILE)
df = df.dropna(subset=['fid', 'ans1'])

fid_to_label = dict(zip(df["fid"], df["ans1"]))

X, y = [], []

for fid, label in fid_to_label.items():
    wav_path = os.path.join(WAV_DIR, f"{fid}.wav")
    if os.path.exists(wav_path):
        try:
            feat = extract_feature(wav_path)
            X.append(feat)
            y.append(label)
        except:
            print(f"❌ Feature error: {fid}")

X = np.array(X)
y = np.array(y)

print("Loaded samples:", len(y))

# ============================
# 3) スケーリング
# ============================
X_scaled = scaler.transform(X)

# ============================
# 4) 予測
# ============================
y_pred = clf.predict(X_scaled)

# ============================
# 5) Accuracy
# ============================
acc = accuracy_score(y, y_pred)
print(f"\n=== Accuracy ===\n{acc:.4f}")

# ============================
# 6) Classification Report
# ============================
print("\n=== Classification Report ===")
report = classification_report(y, y_pred, digits=4)
print(report)

# 保存
with open("model/classification_report.txt", "w") as f:
    f.write(report)

# ============================
# 7) Confusion Matrix
# ============================
cm = confusion_matrix(y, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\n=== Confusion Matrix ===")
print(cm_df)

cm_df.to_csv("model/confusion_matrix.csv")

# ============================
# 8) Cross Validation (5-fold)
# ============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(clf, X_scaled, y, cv=skf)
print("\n=== 5-Fold CV Scores ===")
print(cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# ============================
# 9) t-SNE 可視化
# ============================
print("\nRunning t-SNE (this may take 20–40 seconds)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_embedded = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for lab in labels:
    idx = np.where(y == lab)
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=lab, s=40)

plt.legend()
plt.title("t-SNE visualization of emotion features")
plt.savefig("model/tsne_plot.png")
plt.close()

print("t-SNE saved to model/tsne_plot.png")

# ============================
# 終了
# ============================
print("\nAll evaluation results saved in /model/")

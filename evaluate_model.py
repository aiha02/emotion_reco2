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
from collections import Counter

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
        except Exception as e:
            print(f"❌ Feature error: {fid} ({e})")
    else:
        print(f"⚠️ Missing file: {wav_path}")

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
# zero_division=0 で undefined metric の表示を制御
report = classification_report(y, y_pred, digits=4, zero_division=0)
print(report)

# 保存（model ディレクトリが無ければ作る）
os.makedirs("model", exist_ok=True)
with open("model/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# ============================
# 7) Confusion Matrix
# ============================
cm = confusion_matrix(y, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\n=== Confusion Matrix ===")
print(cm_df)

cm_df.to_csv("model/confusion_matrix.csv", encoding="utf-8")

# ============================
# 8) Cross Validation (adjust n_splits or skip if too few samples)
# ============================
counts = Counter(y)
print("\nClass distribution:", counts)
min_count = min(counts.values()) if counts else 0

if min_count < 2:
    print("\n=== Skipping cross-validation ===")
    print("At least one class has fewer than 2 samples; StratifiedKFold requires at least 2 samples per class.")
else:
    # n_splits は 2 以上、かつ各クラスの最小サンプル数以下にする
    n_splits = min(5, min_count)
    if n_splits < 2:
        print("\n=== Skipping cross-validation ===")
        print("After adjustment, n_splits < 2; cannot run StratifiedKFold.")
    else:
        print(f"\n=== Running StratifiedKFold with n_splits = {n_splits} ===")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(clf, X_scaled, y, cv=skf)
            print("\n=== Cross-validation scores ===")
            print(cv_scores)
            print("Mean CV Accuracy:", cv_scores.mean())
        except Exception as e:
            print("Error during cross-validation:", e)

# ============================
# 9) t-SNE 可視化
# ============================
print("\nRunning t-SNE (this may take 20–40 seconds)...")
try:
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
except Exception as e:
    print("t-SNE failed:", e)

# ============================
# 終了
# ============================
print("\nAll evaluation results saved in /model/")

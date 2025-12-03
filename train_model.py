import pandas as pd
from feature_extractor import extract_feature
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
import joblib
import numpy as np
import os
from collections import Counter
from sklearn.pipeline import make_pipeline

# ==========================
# 1. パス設定
# ==========================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")

# ==========================
# 2. category.txt の読み込み
# ==========================
df = pd.read_csv(CATEGORY_FILE)
df = df.dropna(subset=['fid', 'ans1'])  # 欠損除去

# fid ごとに代表ラベルを使用
fid_to_label = dict(zip(df['fid'], df['ans1']))

# ==========================
# 3. 特徴量抽出
# ==========================
X, y = [], []

for fid, label in fid_to_label.items():
    wav_path = os.path.join(WAV_DIR, f"{fid}.wav")

    if os.path.exists(wav_path):
        try:
            feat = extract_feature(wav_path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"❌ Error with {fid}: {e}")
    else:
        print(f"⚠️ Missing file: {wav_path}")

# numpy変換
X = np.array(X)
y = np.array(y)

# ==========================
# 4. クラス数が1のクラスを除外（重要）
#    交差検証のために各クラスに最低2サンプル必要
# ==========================
print("Label count BEFORE:", Counter(y))

# 最低サンプル数を 2 に設定（1 サンプルのクラスは削除）
min_required_per_class = 2
valid_classes = {lab for lab, cnt in Counter(y).items() if cnt >= min_required_per_class}

X = np.array([x for x, lab in zip(X, y) if lab in valid_classes])
y = np.array([lab for lab in y if lab in valid_classes])

print("Label count AFTER:", Counter(y))

# 基本チェック：クラス数が2未満の場合は学習不可
unique_labels = np.unique(y)
n_classes = len(unique_labels)
n_samples = len(y)

if n_classes < 2:
    raise ValueError(f"訓練できるクラスが不足しています。現在のクラス数={n_classes}、サンプル数={n_samples}")

# ==========================
# 5. 層化交差検証（評価）
#    データが少ないので StratifiedKFold を使って安定評価する
# ==========================
# 各クラスの最小サンプル数を求め、それに合わせて n_splits を決定
class_counts = Counter(y)
min_class_count = min(class_counts.values())

# n_splits は 2..5 の間で、min_class_count を超えない値にする
max_splits = 5
n_splits = min(max_splits, min_class_count)
if n_splits < 2:
    n_splits = 2  # 安全措置（ただし min_class_count が 1 の場合はここに来るべきでない）

print(f"n_samples={n_samples}, n_classes={n_classes}, min_class_count={min_class_count}, using n_splits={n_splits} for StratifiedKFold")

# パイプライン：スケーラー + SVM
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"✅ Cross-validation accuracy scores: {scores}")
print(f"✅ CV mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# ==========================
# 6. 最終モデルを全データで学習して保存
#    実行時には全データで学習してデプロイ用モデルを作るのが一般的
# ==========================
# scaler を個別に保存したかったので、pipeline ではなく個別に fit して保存する
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
clf.fit(X_scaled, y)

acc = clf.score(X_scaled, y)
print(f"✅ Training accuracy on full dataset: {acc:.4f}")

# ==========================
# 7. モデル保存
# ==========================
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/classifier.pkl")
joblib.dump(scaler, "model/scaler.pkl")
np.save("model/labels.npy", np.unique(y))

print("🎉 Training complete! Model and scaler saved in ./model/")

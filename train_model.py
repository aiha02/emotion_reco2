import pandas as pd
from feature_extractor import extract_feature
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import numpy as np
import os

# パス設定
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")
CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")

# category.txt の読み込み
df = pd.read_csv(CATEGORY_FILE)
df = df.dropna(subset=['fid', 'ans1'])  # 欠損除去

# fid ごとに代表ラベルを決定（ans1 を使用）
fid_to_label = dict(zip(df['fid'], df['ans1']))

# 特徴量とラベルのリスト
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

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# モデル学習（SVM）
clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
clf.fit(X_train, y_train)

# 精度確認
acc = clf.score(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.3f}")

# モデル保存
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/classifier.pkl")
joblib.dump(scaler, "model/scaler.pkl")
np.save("model/labels.npy", np.unique(y))

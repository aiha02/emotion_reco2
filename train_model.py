import os
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_extractor import extract_feature

# ==========================
# 1️⃣ データセットのパス設定
# ==========================
DATASET_DIR = "dataset"
WAV_DIR = os.path.join(DATASET_DIR, "wav")
TRANS_DIR = os.path.join(DATASET_DIR, "trans")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")

CATEGORY_FILE = os.path.join(EVAL_DIR, "category.txt")

# ==========================
# 2️⃣ データを読み込む関数
# ==========================
def load_dataset():
    """
    dataset/wav/*.wav と dataset/eval/category.txt を対応づけて
    特徴量と感情ラベルを抽出
    """
    df = pd.read_csv(
        CATEGORY_FILE,
        header=None,
        names=["file_id", "utt_id", "emotion1", "emotion2", "emotion3"],
    )

    X, y = [], []

    for _, row in df.iterrows():
        wav_name = f"{row['file_id']}.wav"
        wav_path = os.path.join(WAV_DIR, wav_name)

        if not os.path.exists(wav_path):
            print(f"⚠️ {wav_path} が見つかりません。スキップします。")
            continue

        try:
            features = extract_feature(wav_path)
            X.append(features)
            y.append(row["emotion1"])  # メインの感情ラベル
        except Exception as e:
            print(f"❌ {wav_name} の特徴抽出に失敗: {e}")

    return np.array(X), np.array(y)

# ==========================
# 3️⃣ モデルの学習
# ==========================
def train_model():
    X, y = load_dataset()

    if len(X) == 0:
        raise ValueError("❌ データが読み込めませんでした。dataset/ のパスを確認してください。")

    print(f"✅ 読み込んだサンプル数: {len(X)}")

    # ラベルエンコード
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 特徴量スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # モデル構築
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # 精度評価
    acc = model.score(X_test, y_test)
    print(f"🎯 テスト精度: {acc:.3f}")

    # ==========================
    # 4️⃣ モデル保存（utils.pyと連携）
    # ==========================
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/classifier.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    np.save("model/labels.npy", label_encoder.classes_)

    print("✅ モデルを保存しました：model/classifier.pkl")
    print("✅ スケーラーを保存しました：model/scaler.pkl")
    print("✅ ラベルを保存しました：model/labels.npy")

if __name__ == "__main__":
    train_model()

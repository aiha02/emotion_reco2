import os
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ====== 設定 ======
DATASET_CSV = "dataset/transcripts.csv"   # CSV: filename, text, emotion
AUDIO_DIR = "dataset/audio"               # WAVファイルのディレクトリ
MODEL_DIR = "model"                       # 学習済みモデル保存先
SAMPLE_RATE = 16000                       # JSUT/一般的音声のサンプリングレート

# ====== 特徴量抽出関数 ======
def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        # MFCC特徴量を抽出
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # 各次元の平均を取る（固定長ベクトル化）
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ {file_path} の処理中にエラー: {e}")
        return None

# ====== データセット読み込み ======
def load_dataset(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    features, labels = [], []

    print(f"🎧 データセット読み込み中 ({len(df)} 件)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = str(row["filename"]).strip()
        emotion = str(row["emotion"]).strip()

        audio_path = os.path.join(audio_dir, filename)
        if not os.path.exists(audio_path):
            print(f"⚠️ ファイルが見つかりません: {audio_path}")
            continue

        feat = extract_features(audio_path)
        if feat is not None:
            features.append(feat)
            labels.append(emotion)

    print(f"✅ 有効サンプル数: {len(features)}")
    return np.array(features), np.array(labels)

# ====== 学習処理 ======
def train_and_save_model(X, y):
    print("⚙️ データの標準化中...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🧠 モデル学習中 (RandomForest)...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"✅ テスト精度: {acc:.3f}")

    # 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    np.save(os.path.join(MODEL_DIR, "labels.npy"), np.unique(y))
    print(f"💾 モデル保存完了 → {MODEL_DIR}/")

# ====== メイン ======
if __name__ == "__main__":
    X, y = load_dataset(DATASET_CSV, AUDIO_DIR)
    if len(X) == 0:
        print("❌ 有効なデータが見つかりませんでした。CSVやパスを確認してください。")
    else:
        train_and_save_model(X, y)

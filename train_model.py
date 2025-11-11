import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===============================
# 1. パス設定
# ===============================
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WAV_DIR = os.path.join(DATASET_DIR, "wav")
TRANS_DIR = os.path.join(DATASET_DIR, "trans")
EVAL_DIR = os.path.join(DATASET_DIR, "eval")

CATEGORY_PATH = os.path.join(EVAL_DIR, "category.txt")
INTENSITY_PATH = os.path.join(EVAL_DIR, "intensity.txt")


# ===============================
# 2. データ読み込み
# ===============================
def load_metadata():
    """category.txt と trans/*.txt を統合して DataFrame を作成"""
    cat_df = pd.read_csv(CATEGORY_PATH, header=None, names=["fid", "uid", "emo1", "emo2", "emo3"])
    cat_df["uid"] = cat_df["uid"].astype(str).str.zfill(3)

    records = []
    for trans_file in os.listdir(TRANS_DIR):
        if not trans_file.endswith(".txt"):
            continue
        fid = trans_file.replace(".txt", "")
        trans_path = os.path.join(TRANS_DIR, trans_file)
        trans_df = pd.read_csv(trans_path, header=None, names=["uid", "start", "end", "speaker", "text"], encoding="utf-8")

        for _, row in trans_df.iterrows():
            match = cat_df[(cat_df["fid"] == fid) & (cat_df["uid"] == str(row["uid"]).zfill(3))]
            if len(match) > 0:
                emotion = match.iloc[0]["emo1"]  # 主ラベルとして emo1 を採用
            else:
                emotion = "OTH"
            records.append({
                "fid": fid,
                "uid": row["uid"],
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "emotion": emotion
            })

    df = pd.DataFrame(records)
    return df


# ===============================
# 3. 特徴抽出関数
# ===============================
def extract_features(wav_path, start, end, sr=16000):
    """指定区間の音声からMFCC特徴を抽出"""
    y, sr = librosa.load(wav_path, sr=sr)
    start_sample = int(float(start) * sr)
    end_sample = int(float(end) * sr)
    y = y[start_sample:end_sample]

    if len(y) == 0:
        return np.zeros((20, 50))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = librosa.util.fix_length(mfcc, size=50, axis=1)
    return mfcc


# ===============================
# 4. Datasetクラス
# ===============================
class EmotionDataset(Dataset):
    def __init__(self, df, wav_dir):
        self.df = df
        self.wav_dir = wav_dir
        self.label_encoder = LabelEncoder()
        self.df["label"] = self.label_encoder.fit_transform(df["emotion"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, f"{row['fid']}.wav")
        features = extract_features(wav_path, row["start"], row["end"])
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return X, y


# ===============================
# 5. モデル定義
# ===============================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.fc1 = nn.Linear(32 * 5 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ===============================
# 6. 学習ループ
# ===============================
def train_model(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])
    train_dataset = EmotionDataset(train_df, WAV_DIR)
    test_dataset = EmotionDataset(test_df, WAV_DIR)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    num_classes = len(train_dataset.label_encoder.classes_)
    model = EmotionCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # 評価
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())

    print(classification_report(trues, preds, target_names=train_dataset.label_encoder.classes_))

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/emotion_cnn.pth")
    print("✅ モデルを保存しました！ model/emotion_cnn.pth")


# ===============================
# 7. 実行
# ===============================
if __name__ == "__main__":
    df = load_metadata()
    print("Loaded samples:", len(df))
    train_model(df)

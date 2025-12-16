import numpy as np
import librosa

def extract_feature_raw(y, sr):
    """MFCC(40) + Chroma(12) + Mel(128) = 180"""
    stft = np.abs(librosa.stft(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([mfcc, chroma, mel])


# 👇 Streamlit 互換用ラッパー（重要）
def extract_feature(file_path):
    """wav/mp3 ファイルパスから特徴量抽出（UI用）"""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return extract_feature_raw(y, sr)

import numpy as np
import librosa

def extract_feature_raw(y, sr):
    """
    MFCC(40) + ΔMFCC(40) + ΔΔMFCC(40)
    + Chroma(12) + Mel(128)
    = 260次元
    """

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Δ / ΔΔ
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 時間平均
    mfcc_mean = np.mean(mfcc.T, axis=0)
    delta_mean = np.mean(delta.T, axis=0)
    delta2_mean = np.mean(delta2.T, axis=0)

    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sr).T,
        axis=0
    )

    # Mel
    mel = np.mean(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T,
        axis=0
    )

    return np.hstack([
        mfcc_mean,
        delta_mean,
        delta2_mean,
        chroma,
        mel
    ])


def extract_feature(file_path):
    """
    wav / mp3 ファイル用（Streamlit・utils 互換）
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return extract_feature_raw(y, sr)

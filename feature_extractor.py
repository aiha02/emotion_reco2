
import numpy as np
import librosa

def extract_feature(file_path):
    """Return 1D feature vector: MFCC(40) + Chroma(12) + Mel(128) => total 180 features"""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    feat = np.hstack([mfccs, chroma, mel])
    return feat

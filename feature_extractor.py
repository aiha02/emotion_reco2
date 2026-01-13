import numpy as np
import librosa


def extract_feature_raw(file_path: str, duration: float = 15.0) -> np.ndarray:
    """
    学習時と互換の260次元特徴量
    """

    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=True,
        duration=duration
    )

  #  y, _ = librosa.effects.trim(y, top_db=30)

    return extract_features(y, sr)


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    features = []

    # =========================
    # MFCC (40)
    # =========================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for feat in (mfcc, delta, delta2):
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))

    # MFCC系: 40 × 3 × 2 = 240

    # =========================
    # RMS (energy)
    # =========================
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # =========================
    # Zero Crossing Rate
    # =========================
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # =========================
    # Spectral Centroid
    # =========================
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # =========================
    # Spectral Bandwidth
    # =========================
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # 追加特徴: 4 × 2 = 8

    # 合計: 240 + 8 = 248
    # ↓ 安定化のため Chroma を少量追加

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))  # 12

    # 合計: 248 + 12 = 260

    return np.array(features, dtype=np.float32)

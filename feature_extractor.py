import numpy as np
import librosa


def extract_feature_raw(
    file_path: str,
    duration: float = 15.0
) -> np.ndarray:
    """
    utils.py から呼ばれる互換用関数
    内部で改善済み特徴量抽出を行う
    """

    # =========================
    # Load audio
    # =========================
    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=True,
        duration=duration
    )

    # =========================
    # Remove silence (IMPORTANT)
    # =========================
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    return extract_features(y, sr)


def extract_features(
    y: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    感情推定用の改善済み特徴量
    - mean + std
    - 複数音響特徴
    """

    features = []

    # =========================
    # MFCC
    # =========================
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # =========================
    # Chroma
    # =========================
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # =========================
    # Spectral centroid
    # =========================
    spec_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))

    # =========================
    # Spectral bandwidth
    # =========================
    spec_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )
    features.append(np.mean(spec_bandwidth))
    features.append(np.std(spec_bandwidth))

    # =========================
    # Energy (RMS)
    # =========================
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)

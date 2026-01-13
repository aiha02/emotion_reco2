import numpy as np
import librosa


def extract_features(
    y: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    音声波形から感情推定用の特徴量を抽出する
    改善点：
    - 平均＋標準偏差を使用
    - 無音区間除去後の特徴量
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
    # Spectral features
    # =========================
    spec_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))

    spec_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )
    features.append(np.mean(spec_bandwidth))
    features.append(np.std(spec_bandwidth))

    # =========================
    # Energy
    # =========================
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


def extract_features_from_file(
    file_path: str,
    duration: float = 15.0
) -> np.ndarray:
    """
    音声ファイルから特徴量を抽出する
    - 無音区間除去
    - 最大15秒（感情差を出しやすい）
    """

    # 音声読み込み
    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=True,
        duration=duration
    )

    # 無音除去（超重要）
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # デバッグ用（必要ならコメントアウト）
    # print("Audio std:", np.std(y))

    return extract_features(y, sr)

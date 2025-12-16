# emotion_state.py
# ============================================
# 感情確率 + 感情強度 → Spotify Audio Features
# ============================================

from typing import Dict

# 使用する感情ラベル（学習後のクラス統合に合わせる）
EMOTIONS = ["POS", "NEU", "NEG_L", "NEG_H", "OTH"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """値を [lo, hi] に収める"""
    return max(lo, min(hi, x))


def normalize_intensity(intensity: float) -> float:
    """
    感情強度（1〜5）→ 0〜1
    """
    return clamp((intensity - 1.0) / 4.0)


def emotion_state_to_audio_features(
    emotion_probs: Dict[str, float],
    intensity: float,
) -> Dict[str, float]:
    """
    感情確率 + 強度から Spotify Recommendation API 用の
    Audio Feature ベクトルを生成する
    """

    # ============================
    # 1. 感情確率の安全化
    # ============================
    probs = {e: emotion_probs.get(e, 0.0) for e in EMOTIONS}
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s for k, v in probs.items()}
    else:
        probs["NEU"] = 1.0

    # ============================
    # 2. 強度正規化
    # ============================
    inten = normalize_intensity(intensity)

    # ============================
    # 3. 各 Audio Feature を計算
    # ============================

    # --- valence（明るさ） ---
    valence = (
        probs["POS"] * 0.85 +
        probs["NEU"] * 0.55 +
        probs["NEG_L"] * 0.25 +
        probs["NEG_H"] * 0.15 +
        probs["OTH"] * 0.50
    )

    # --- energy（激しさ） ---
    energy = (
        probs["POS"] * (0.50 + 0.30 * inten) +
        probs["NEU"] * 0.35 +
        probs["NEG_L"] * (0.25 + 0.20 * inten) +
        probs["NEG_H"] * (0.60 + 0.40 * inten) +
        probs["OTH"] * 0.30
    )

    # --- danceability ---
    danceability = (
        probs["POS"] * 0.65 +
        probs["NEU"] * 0.45 +
        probs["NEG_L"] * 0.35 +
        probs["NEG_H"] * 0.40 +
        probs["OTH"] * 0.45
    )

    # --- tempo（BPMスケール） ---
    # 60〜180 BPM にマッピング
    tempo_norm = (
        probs["POS"] * (0.55 + 0.25 * inten) +
        probs["NEU"] * 0.45 +
        probs["NEG_L"] * 0.35 +
        probs["NEG_H"] * (0.60 + 0.30 * inten) +
        probs["OTH"] * 0.45
    )
    tempo = 60.0 + 120.0 * clamp(tempo_norm)

    # --- acousticness（生音感） ---
    acousticness = (
        probs["NEG_L"] * 0.65 +
        probs["NEU"] * 0.55 +
        probs["POS"] * 0.40 +
        probs["NEG_H"] * 0.20 +
        probs["OTH"] * 0.50
    )

    # --- instrumentalness（歌詞少なめ） ---
    instrumentalness = (
        probs["NEU"] * 0.55 +
        probs["OTH"] * 0.50 +
        probs["NEG_L"] * 0.40 +
        probs["POS"] * 0.30 +
        probs["NEG_H"] * 0.20
    )

    # ============================
    # 4. クリップして返す
    # ============================
    return {
        "target_valence": clamp(valence),
        "target_energy": clamp(energy),
        "target_danceability": clamp(danceability),
        "target_tempo": round(tempo, 1),
        "target_acousticness": clamp(acousticness),
        "target_instrumentalness": clamp(instrumentalness),
    }

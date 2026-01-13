# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import os

from utils import predict_from_file
from emotion_state import emotion_state_to_audio_features
from spotify_recommender import SpotifyRecommender


# =====================================
# ページ設定
# =====================================
st.set_page_config(
    page_title="音声感情 × Spotify 楽曲推薦",
    page_icon="🎧",
)

st.title("🎙 音声感情認識 × 楽曲推薦デモ")
st.caption("音声から感情を推定し、Spotifyで楽曲を推薦します")

recommender = SpotifyRecommender(market="JP")


# =====================================
# 音声アップロード
# =====================================
st.header("① 音声ファイルをアップロード")

uploaded_file = st.file_uploader(
    "WAV / MP3 ファイルを選択してください",
    type=["wav", "mp3", "m4a", "ogg"],
)

if uploaded_file is None:
    st.stop()


# =====================================
# 一時保存
# =====================================
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded_file.read())
    audio_path = tmp.name


# =====================================
# 感情推定
# =====================================
st.header("② 感情推定結果")

try:
    # 🔴 重要：返り値は3つ
    pred_label, proba, labels = predict_from_file(audio_path)

    # 確率配列 → dict
    emotion_probs = {
        labels[i]: float(proba[i])
        for i in range(len(labels))
    }

    emotion_map = {
        "POS": "😊 ポジティブ",
        "NEU": "😌 ニュートラル",
        "NEG": "😢 ネガティブ",
    }

    st.success(
        f"推定感情：**{emotion_map.get(pred_label, pred_label)}**"
    )

    st.caption("感情確率")
    st.json(emotion_probs)

except Exception as e:
    st.error(f"感情推定エラー: {e}")
    st.stop()


# =====================================
# 感情 → 音楽特徴量
# =====================================
st.header("③ 感情から生成された音楽特徴量")

# 🔴 emotion_state.py 側の定義に合わせて補正
emotion_probs_fixed = {
    "POS": emotion_probs.get("POS", 0.0),
    "NEU": emotion_probs.get("NEU", 0.0),
    "NEG_L": emotion_probs.get("NEG", 0.0),
    "NEG_H": 0.0,
    "OTH": 0.0,
}

# 強度（とりあえず固定：後で主観入力にできる）
intensity = 3

audio_features = emotion_state_to_audio_features(
    emotion_probs=emotion_probs_fixed,
    intensity=intensity,
)

st.json(audio_features)


# =====================================
# 楽曲推薦
# =====================================
st.header("④ おすすめ楽曲")

emotion_text = {
    "POS": "😊 明るく前向きな気分に合う曲",
    "NEU": "😌 落ち着いて聴ける曲",
    "NEG": "😢 気持ちに寄り添う曲",
}

st.info(
    emotion_text.get(pred_label, "あなたの気分に合う曲")
)

try:
    recommendations = recommender.recommend_tracks(
        audio_features=audio_features,
        emotion_label=pred_label,
        limit=8,
    )
except Exception as e:
    st.error(f"Spotify 推薦エラー: {e}")
    st.stop()

if not recommendations:
    st.warning("おすすめ楽曲が取得できませんでした")
    st.stop()


# =====================================
# UI表示（カード形式）
# =====================================
for track in recommendations:
    col1, col2 = st.columns([1, 3])

    with col1:
        if track.get("image_url"):
            st.image(
                track["image_url"],
                use_container_width=True,
            )

    with col2:
        st.markdown(f"### 🎵 {track['track_name']}")
        st.markdown(f"**👤 アーティスト**：{track['artist']}")

        if track.get("album"):
            st.markdown(f"**💿 アルバム**：{track['album']}")

        if track.get("preview_url"):
            st.audio(track["preview_url"])
        else:
            st.caption("※ プレビュー音源なし")

        st.link_button(
            "🔗 Spotifyで開く",
            track["external_url"],
        )

        st.caption(
            f"推薦条件："
            f"valence={audio_features['target_valence']:.2f}, "
            f"energy={audio_features['target_energy']:.2f}, "
            f"danceability={audio_features['target_danceability']:.2f}"
        )

    st.divider()


# =====================================
# 後処理
# =====================================
os.remove(audio_path)

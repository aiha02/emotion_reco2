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
# 初期設定
# =====================================
st.set_page_config(
    page_title="音声感情 × Spotify 楽曲推薦",
    page_icon="🎧",
    layout="centered",
)

st.title("🎙 音声感情認識 × 楽曲推薦デモ")
st.caption("あなたの声から気分を推定し、Spotifyで楽曲を推薦します")

recommender = SpotifyRecommender(market="JP")


# =====================================
# 音声アップロード
# =====================================
st.header("① 音声ファイルをアップロード")

uploaded_file = st.file_uploader(
    "WAV / MP3 ファイルを選択してください",
    type=["wav", "mp3"],
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
    emotion_label, emotion_prob = predict_from_file(audio_path)

    emotion_map = {
        "POS": "😊 ポジティブ",
        "NEU": "😌 ニュートラル",
        "NEG": "😢 ネガティブ",
    }

    st.success(
        f"推定感情：**{emotion_map.get(emotion_label, emotion_label)}**"
    )

    st.caption(
        "（モデル出力確率）"
    )
    st.json(emotion_prob)

except Exception as e:
    st.error(f"感情推定エラー: {e}")
    st.stop()


# =====================================
# 感情 → 音楽特徴量
# =====================================
audio_features = emotion_state_to_audio_features(emotion_label)

st.header("③ 感情に基づく音楽特徴量")
st.caption("Spotify Recommendation API に渡す目標値")

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

st.info(emotion_text.get(emotion_label, "あなたの気分に合う曲"))

try:
    recommendations = recommender.recommend_tracks(
        audio_features=audio_features,
        emotion_label=emotion_label,
        limit=8,
    )

except Exception as e:
    st.error(f"Spotify 推薦でエラーが発生しました: {e}")
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
            f"推薦理由：valence={audio_features['target_valence']:.2f}, "
            f"energy={audio_features['target_energy']:.2f}, "
            f"danceability={audio_features['target_danceability']:.2f}"
        )

    st.divider()


# =====================================
# 後処理
# =====================================
os.remove(audio_path)

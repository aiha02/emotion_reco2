# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from utils import predict_from_file
from emotion_state import emotion_state_to_audio_features
from spotify_recommender import SpotifyRecommender

# =====================================
# ページ設定
# =====================================
st.set_page_config(
    page_title="音声感情 × Spotify 楽曲推薦",
    layout="centered",
)

st.title("🎙️ 音声感情認識 × 🎵 Spotify 楽曲推薦")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（wav / mp3）",
    type=["wav", "mp3", "m4a", "ogg"],
)

audio_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(uploaded.read())
    tmp.close()
    audio_path = tmp.name

if audio_path:
    st.audio(audio_path)

    pred_label, proba, labels = predict_from_file(audio_path)
    prob_dict = dict(zip(labels, proba))

    intensity = np.max(proba) * 5.0

    audio_features = emotion_state_to_audio_features(
        emotion_probs=prob_dict,
        intensity=intensity,
    )

    st.subheader("🎵 おすすめ楽曲")

    try:
        recommender = SpotifyRecommender(market="JP")
        tracks = recommender.recommend_tracks(
            audio_features,
            limit=8,
            candidate_size=80,
        )

        for t in tracks:
            st.markdown(
                f"🎶 **{t['track_name']}**  \n"
                f"👤 {t['artist']}  \n"
                f"[🔗 Spotifyで開く]({t['external_url']})"
            )
            if t["preview_url"]:
                st.audio(t["preview_url"])
            st.markdown("---")

    except Exception as e:
        st.error(f"Spotify 推薦でエラーが発生しました: {e}")

# app.py 
from dotenv import load_dotenv 
load_dotenv()

import streamlit as st 
import tempfile 
import os 
import numpy as np 
import matplotlib.pyplot as plt

from utils import predict_from_file 
from emotion_state import emotion_state_to_audio_features 
from spotify_recommender import SpotifyRecommender
recommender = SpotifyRecommender(market="JP")


st.set_page_config( page_title="音声感情 × Spotify 楽曲推薦", layout="centered", )

st.title("🎙️ 音声感情認識 × 🎵 Spotify 楽曲推薦") 
st.markdown( """ 音声から **感情・強度** を推定し、 その感情状態に合わせた **Spotify 楽曲** を推薦します。 """ )

uploaded = st.file_uploader
( "音声ファイルをアップロード（wav / mp3）", type=["wav", "mp3", "m4a", "ogg"], ) 
audio_path = None 
if uploaded: 
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav") 
    tmp.write(uploaded.read()) 
    tmp.close() 
    audio_path = tmp.name

if audio_path: 
    st.audio(audio_path) 
with st.spinner("🎧 感情を解析しています..."): 
    pred_label, proba, labels = predict_from_file(audio_path)

st.subheader("📊 感情推定結果") 
prob_dict = dict(zip(labels, proba)) 
st.write("**予測感情:**", pred_label) 
fig, ax = plt.subplots(figsize=(6, 3)) 
ax.bar(prob_dict.keys(), prob_dict.values()) 
ax.set_ylim(0, 1) ax.set_ylabel("Probability") 
ax.set_title("Emotion Probability") 
st.pyplot(fig)

intensity = np.max(proba) * 5.0 
st.subheader("🔥 感情強度") 
st.progress(intensity / 5.0) 
st.write(f"推定強度: **{intensity:.2f} / 5**")

audio_features = emotion_state_to_audio_features( 
    emotion_probs=prob_dict, 
    intensity=intensity, ) 
st.subheader("🎚️ 推薦用オーディオ特徴量") 
st.json(audio_features)

st.subheader("🎵 おすすめ楽曲") 
try: 
    recommender = SpotifyRecommender() 
    tracks = recommender.recommend_tracks( 
        audio_features, limit=8, 
    ) 
for t in tracks: 
    st.markdown( 
        f"🎶 **{t['track_name']}** \n" 
        f"👤 {t['artist']} \n" 
        f"[🔗 Spotifyで開く]({t['external_url']})" )
    if t["preview_url"]: 
        st.audio(t["preview_url"]) 
        st.markdown("---")

except Exception as e: 
st.error(f"Spotify 推薦でエラーが発生しました: {e}") 
else: st.info("音声ファイルをアップロードしてください。") 
st.caption("© Graduation Research Demo | Emotion-based Music Recommendation")

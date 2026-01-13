# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import os
import csv
from datetime import datetime

from utils import predict_from_file
from emotion_state import emotion_state_to_audio_features
from spotify_recommender import SpotifyRecommender


# =====================================
# ページ設定
# =====================================
st.set_page_config(
    page_title="感情 × 楽曲推薦 主観評価実験",
    page_icon="🎧",
)

st.title("🎙 感情音声 × Spotify 楽曲推薦")
st.caption("音声から感情を推定し、楽曲推薦に対する主観評価を収集します")

recommender = SpotifyRecommender(market="JP")


# =====================================
# 被験者情報
# =====================================
st.header("① 被験者情報")

participant_id = st.text_input("被験者ID（例：P01）")

if not participant_id:
    st.stop()


# =====================================
# 音声アップロード
# =====================================
st.header("② 音声ファイルをアップロード")

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
st.header("③ 感情推定")

try:
    result = predict_from_file(audio_path)
    emotion_label = result[0]
    emotion_prob = result[1]

    emotion_map = {
        "POS": "😊 ポジティブ",
        "NEU": "😌 ニュートラル",
        "NEG": "😢 ネガティブ",
    }

    st.success(f"推定感情：{emotion_map.get(emotion_label, emotion_label)}")

except Exception as e:
    st.error(f"感情推定エラー: {e}")
    st.stop()


# =====================================
# 感情 → 音楽特徴量
# =====================================
audio_features = emotion_state_to_audio_features(emotion_label)


# =====================================
# 楽曲推薦
# =====================================
st.header("④ おすすめ楽曲")

tracks = recommender.recommend_tracks(
    audio_features=audio_features,
    emotion_label=emotion_label,
    limit=1,  # 実験なので1曲に絞る
)

if not tracks:
    st.error("楽曲が取得できませんでした")
    st.stop()

track = tracks[0]


# =====================================
# 楽曲表示
# =====================================
st.subheader("🎵 推薦楽曲")

st.markdown(f"**曲名**：{track['track_name']}")
st.markdown(f"**アーティスト**：{track['artist']}")
st.markdown(f"**アルバム**：{track.get('album', '-')}")
st.link_button("Spotifyで開く", track["external_url"])

if track.get("preview_url"):
    st.audio(track["preview_url"])


# =====================================
# 主観評価フォーム
# =====================================
st.header("⑤ 楽曲に対する評価")

with st.form("evaluation_form"):
    q1 = st.slider(
        "Q1. この曲は今の気分に合っていましたか？",
        1, 5, 3
    )
    q2 = st.slider(
        "Q2. この曲に満足していますか？",
        1, 5, 3
    )
    q3 = st.slider(
        "Q3. 今後もこの曲を聴きたいと思いますか？",
        1, 5, 3
    )
    comment = st.text_area(
        "自由記述（任意）"
    )

    submitted = st.form_submit_button("評価を送信")


# =====================================
# 結果保存
# =====================================
if submitted:
    save_path = "evaluation_results.csv"
    file_exists = os.path.isfile(save_path)

    with open(save_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "participant_id",
                "emotion_label",
                "track_name",
                "artist",
                "spotify_url",
                "q1_mood_match",
                "q2_satisfaction",
                "q3_listen_again",
                "comment",
            ])

        writer.writerow([
            datetime.now().isoformat(),
            participant_id,
            emotion_label,
            track["track_name"],
            track["artist"],
            track["external_url"],
            q1,
            q2,
            q3,
            comment,
        ])

    st.success("評価を保存しました。ご協力ありがとうございました！")


# =====================================
# 後処理
# =====================================
os.remove(audio_path)

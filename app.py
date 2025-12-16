import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import predict_from_file
from feature_extractor import extract_feature
import soundfile as sf

# ===============================
# ページ設定
# ===============================
st.set_page_config(
    page_title="音声感情認識（Web版）",
    layout="centered"
)

st.title("🎙️ 音声感情認識 — Webデモ")
st.markdown(
    """
WAV / MP3 / M4A / OGG ファイルをアップロード、  
またはマイクから録音して感情を推定します。
""",
    unsafe_allow_html=True
)

# ===============================
# 音声ファイルアップロード
# ===============================
uploaded = st.file_uploader(
    "音声ファイルをアップロード",
    type=["wav", "mp3", "m4a", "ogg"]
)

# ===============================
# マイク録音（streamlit-audiorec）
# ===============================
uploaded_file_path = None
try:
    import streamlit_audiorec as sar
    st.info("🎤 マイク録音が利用可能です")
    rec = sar.st_audiorec()
    if rec is not None and len(rec) > 0:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(rec)
        tmp.close()
        uploaded_file_path = tmp.name
except Exception:
    st.info("⚠️ マイク録音は利用できません")

# ===============================
# ファイルアップロード処理
# ===============================
if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.read())
    tfile.close()
    uploaded_file_path = tfile.name

# ===============================
# 推論処理
# ===============================
if uploaded_file_path is not None:
    st.audio(open(uploaded_file_path, "rb").read())
    st.write("🎧 感情を推定しています...")

    try:
        pred, proba, labels = predict_from_file(uploaded_file_path)

        st.header(f"🎯 予測された感情: {pred}")

        if proba is not None:
            st.subheader("感情ごとの確率")
            for lab, p in zip(labels, proba):
                st.write(f"- {lab}: {p:.3f}")

        # ===============================
        # 特徴量可視化（MFCC）
        # ===============================
        feat = extract_feature(uploaded_file_path)
        mfcc = feat[:40]

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(mfcc)
        ax.set_title("MFCC (mean)")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Value")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ 予測中にエラーが発生しました: {e}")

else:
    st.info("⏳ 音声をアップロードまたは録音してください")

st.markdown("---")
st.caption("© bp22008 卒業研究デモ")

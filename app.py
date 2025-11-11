import streamlit as st
import tempfile, os, base64, io
from utils import predict_from_file
from feature_extractor import extract_feature
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# ===============================
# ページ設定
# ===============================
st.set_page_config(page_title="日本語音声感情認識（Web版）", layout="centered")

st.title("🎙️ 日本語音声感情認識 — Webデモ")
st.markdown("WAV / MP3 ファイルをアップロード、またはマイクから録音して感情を推定します。<br>予測される感情: 😊 喜び、😠 怒り、😢 悲しみ、😐 中立", unsafe_allow_html=True)

# ===============================
# 音声ファイルアップロード
# ===============================
uploaded = st.file_uploader('音声ファイルをアップロード（wav, mp3, m4a, ogg）', type=['wav', 'mp3', 'm4a', 'ogg'])

# ===============================
# マイク録音（streamlit-audiorec対応）
# ===============================
use_recorder = False
try:
    import streamlit_audiorec as sar
    use_recorder = True
except Exception:
    use_recorder = False

if use_recorder:
    st.info("🎤 マイク録音機能が有効です（streamlit-audiorecによる提供）")
    rec = sar.st_audiorec()
    if rec is not None and len(rec) > 0:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(rec)
        tmp.flush()
        tmp.close()
        uploaded_file_path = tmp.name
    else:
        uploaded_file_path = None
else:
    st.info("⚠️ マイク録音機能が利用できません。音声ファイルをアップロードしてください。")

# ===============================
# アップロード処理
# ===============================
if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read())
    tfile.flush()
    tfile.close()
    uploaded_file_path = tfile.name

# ===============================
# 推論処理
# ===============================
if 'uploaded_file_path' in locals() and uploaded_file_path is not None:
    st.audio(open(uploaded_file_path, 'rb').read())
    st.write('🎧 特徴量を抽出して感情を推定しています...')
    try:
        pred, proba, labels = predict_from_file(uploaded_file_path)
        label_names = labels
        st.header(f"🎯 予測された感情: {pred}")
        if proba is not None:
            st.subheader("感情ごとの確率")
            for name, p in zip(label_names, proba):
                st.write(f"- {name}: {p:.2f}")

        # ===============================
        # 特徴量可視化
        # ===============================
        feat = extract_feature(uploaded_file_path)
        mfcc = feat[:40]
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(mfcc)
        ax.set_title('MFCC（平均値）', fontproperties='Meiryo')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ 予測中にエラーが発生しました: {e}")
else:
    st.info('⏳ 音声をアップロードまたは録音してください。')

st.markdown("---")
st.caption("© 2025 日本語音声感情認識プロジェクト")

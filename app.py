
import streamlit as st
import tempfile, os, base64, io
from utils import predict_from_file
from feature_extractor import extract_feature
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

st.set_page_config(page_title="Speech Emotion Recognition (Web)", layout="centered")

st.title("🎙️ Speech Emotion Recognition — Web Demo")
st.markdown("Upload a WAV/MP3 file or record via microphone (if supported). Predicts one of: 😊 happy, 😠 angry, 😢 sad, 😐 neutral.")

uploaded = st.file_uploader('Upload audio file (wav or mp3)', type=['wav','mp3','m4a','ogg'])
use_recorder = False
try:
    import streamlit_audiorec as sar
    use_recorder = True
except Exception:
    use_recorder = False

if use_recorder:
    st.info("Microphone recorder enabled (provided by streamlit-audiorec).")
    rec = sar.st_audiorec()
    if rec is not None and len(rec) > 0:
        b = rec
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(b)
        tmp.flush()
        tmp.close()
        uploaded_file_path = tmp.name
    else:
        uploaded_file_path = None
else:
    st.info("Microphone recorder not available. You can still upload audio files.")

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read())
    tfile.flush()
    tfile.close()
    uploaded_file_path = tfile.name

if 'uploaded_file_path' in locals() and locals().get('uploaded_file_path', None):
    st.audio(open(uploaded_file_path,'rb').read())
    st.write('Extracting features and predicting...')
    try:
        pred, proba, labels = predict_from_file(uploaded_file_path)
        label_names = labels
        st.header(f"Prediction: {pred}")
        if proba is not None:
            for name, p in zip(label_names, proba):
                st.write(f"- {name}: {p:.2f}")
        feat = extract_feature(uploaded_file_path)
        mfcc = feat[:40]
        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(mfcc)
        ax.set_title('MFCC (mean values)')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write('Waiting for audio...')

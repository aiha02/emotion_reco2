
# 🎵 Emotion Recognition Streamlit Web App

日本語の音声認識と感情分類を行うWebアプリケーションです。  
Streamlitベースで動作し、音声ファイルを入力すると感情（喜び・怒り・悲しみなど）を推定します。

---

## 🚀 機能概要
- 🎙️ **音声アップロード**：WAV形式の音声をアップロードして解析  
- 🧠 **感情認識モデル**：MFCC特徴量＋SVMモデルを使用  
- 📊 **結果の可視化**：感情スコアをグラフで表示  
- 🧩 **日本語音声データ対応**：独自データセットを使用可能  

---

## 📦 環境構築

```bash
git clone https://github.com/USERNAME/emotion-recognition-streamlit-web.git
cd emotion-recognition-streamlit-web
pip install -r requirements.txt

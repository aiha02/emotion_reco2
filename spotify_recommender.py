# spotify_recommender.py
# ============================================
# Emotion-based Spotify Recommendation Module
# ============================================

import os
from typing import Dict, List
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyRecommender:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        """
        Spotify Recommendation API 用クラス
        環境変数 or 引数から認証情報を取得
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify client_id / client_secret が設定されていません")

        auth = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        self.sp = spotipy.Spotify(auth_manager=auth)

    # ----------------------------------------
    # 感情ベース推薦
    # ----------------------------------------
    def recommend_tracks(
        self,
        audio_features: Dict[str, float],
        limit: int = 10,
        seed_genres: List[str] | None = None,
        market: str = "JP",
    ) -> List[Dict]:
        """
        audio_features:
          emotion_state_to_audio_features() の出力

        return:
          track 情報のリスト
        """

        if seed_genres is None:
            # 日本で無難に使えるジャンル
            seed_genres = ["pop", "rock", "indie"]

        # Spotify Recommendation API 用パラメータ
        params = {
            "limit": limit,
            "market": market,
            "seed_genres": seed_genres[:5],  # 最大5
        }

        # audio feature を target_* として追加
        for k, v in audio_features.items():
            params[k] = v

        results = self.sp.recommendations(**params)

        tracks = []
        for t in results["tracks"]:
            tracks.append({
                "track_name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "album": t["album"]["name"],
                "preview_url": t["preview_url"],
                "external_url": t["external_urls"]["spotify"],
                "popularity": t["popularity"],
            })

        return tracks

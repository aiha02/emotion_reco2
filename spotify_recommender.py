# spotify_recommender.py
import os
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyRecommender:
    """
    Spotify Recommendation API を用いた
    感情適合型楽曲推薦クラス（最終・安定版）

    - Audio Features API は使用しない
    - Recommendation API のみ使用（403回避）
    - 感情 → seed_genres + target_* で推薦
    """

    def __init__(self, market: str = "JP"):
        if not os.getenv("SPOTIPY_CLIENT_ID"):
            raise RuntimeError("SPOTIPY_CLIENT_ID が設定されていません")
        if not os.getenv("SPOTIPY_CLIENT_SECRET"):
            raise RuntimeError("SPOTIPY_CLIENT_SECRET が設定されていません")

        self.market = market
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials()
        )

        # 感情 → ジャンル対応
        self.emotion_to_genres = {
            "POS": ["pop", "dance"],
            "NEU": ["chill", "ambient"],
            "NEG": ["acoustic", "sad"],
        }

    # ======================================================
    # public
    # ======================================================
    def recommend_tracks(
        self,
        audio_features: Dict[str, float],
        emotion_label: str = None,
        limit: int = 8,
    ) -> List[Dict]:
        """
        感情推定結果から Spotify 推薦を取得
        """

        # 感情ラベルが未指定なら valence から推定
        if emotion_label is None:
            v = audio_features.get("target_valence", 0.5)
            if v >= 0.6:
                emotion_label = "POS"
            elif v <= 0.4:
                emotion_label = "NEG"
            else:
                emotion_label = "NEU"

        genres = self.emotion_to_genres.get(
            emotion_label, ["pop"]
        )

        res = self.sp.recommendations(
            seed_genres=genres,
            target_valence=audio_features.get("target_valence", 0.5),
            target_energy=audio_features.get("target_energy", 0.5),
            target_danceability=audio_features.get(
                "target_danceability", 0.5
            ),
            limit=limit,
            market=self.market,
        )

        tracks = []
        for t in res["tracks"]:
            tracks.append({
                "track_name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "external_url": t["external_urls"]["spotify"],
                "preview_url": t["preview_url"],
            })

        return tracks

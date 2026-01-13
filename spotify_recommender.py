# spotify_recommender.py
import os
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


class SpotifyRecommender:
    """
    Spotify Recommendation API 最終安定版

    ✔ seed_tracks 使用
    ✔ market を recommendations では指定しない（404回避）
    ✔ 失敗時は search ベースに自動フォールバック
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

    # ======================================================
    # public
    # ======================================================
    def recommend_tracks(
        self,
        audio_features: Dict[str, float],
        emotion_label: str,
        limit: int = 8,
    ) -> List[Dict]:

        seed_tracks = self._get_seed_tracks(emotion_label)

        # --------------------------------------------------
        # 1. Recommendation API（market 指定しない）
        # --------------------------------------------------
        try:
            res = self.sp.recommendations(
                seed_tracks=seed_tracks[:2],
                target_valence=audio_features.get("target_valence", 0.5),
                target_energy=audio_features.get("target_energy", 0.5),
                target_danceability=audio_features.get(
                    "target_danceability", 0.5
                ),
                limit=limit,
            )
            tracks = res["tracks"]

        except SpotifyException:
            # --------------------------------------------------
            # 2. フォールバック：Search API
            # --------------------------------------------------
            tracks = self._fallback_search(emotion_label, limit)

        return [
            {
                "track_name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "external_url": t["external_urls"]["spotify"],
                "preview_url": t["preview_url"],
            }
            for t in tracks
        ]

    # ======================================================
    # private
    # ======================================================
    def _get_seed_tracks(self, emotion_label: str) -> List[str]:
        query_map = {
            "POS": "happy pop",
            "NEU": "chill",
            "NEG": "sad acoustic",
        }
        q = query_map.get(emotion_label, "pop")

        res = self.sp.search(
            q=q,
            type="track",
            limit=5,
            market=self.market,
        )
        return [t["id"] for t in res["tracks"]["items"]]

    def _fallback_search(
        self,
        emotion_label: str,
        limit: int,
    ) -> List[Dict]:

        query_map = {
            "POS": "happy pop",
            "NEU": "chill",
            "NEG": "sad acoustic",
        }
        q = query_map.get(emotion_label, "pop")

        res = self.sp.search(
            q=q,
            type="track",
            limit=limit,
            market=self.market,
        )

        return res["tracks"]["items"]

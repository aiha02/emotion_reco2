# spotify_recommender.py
import os
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


class SpotifyRecommender:
    """
    Spotify Recommendation 安定動作版（genre seed 不使用）

    ✔ seed_tracks のみ使用（404回避）
    ✔ audio features による制御は維持
    ✔ Client Credentials Flow 完全対応
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

        # -----------------------------------------
        # 1. seed_tracks を取得（必ず成功する）
        # -----------------------------------------
        seed_tracks = self._get_seed_tracks(emotion_label)

        # -----------------------------------------
        # 2. Recommendation API
        # -----------------------------------------
        try:
            res = self.sp.recommendations(
                seed_tracks=seed_tracks[:2],  # 1〜2が最安定
                target_valence=audio_features.get("target_valence", 0.5),
                target_energy=audio_features.get("target_energy", 0.5),
                target_danceability=audio_features.get(
                    "target_danceability", 0.5
                ),
                limit=limit,
                market=self.market,
            )
        except SpotifyException as e:
            raise RuntimeError(f"Spotify 推薦失敗: {e}")

        # -----------------------------------------
        # 3. 整形
        # -----------------------------------------
        results = []
        for t in res["tracks"]:
            results.append({
                "track_name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "external_url": t["external_urls"]["spotify"],
                "preview_url": t["preview_url"],
            })

        return results

    # ======================================================
    # private
    # ======================================================
    def _get_seed_tracks(self, emotion_label: str) -> List[str]:
        """
        感情ごとに安定した検索クエリを使う
        """
        query_map = {
            "POS": "happy pop",
            "NEU": "chill",
            "NEG": "sad acoustic",
        }

        q = query_map.get(emotion_label, "pop")

        tracks = self.sp.search(
            q=q,
            type="track",
            limit=5,
            market=self.market,
        )["tracks"]["items"]

        return [t["id"] for t in tracks]

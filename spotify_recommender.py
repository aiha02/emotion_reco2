# spotify_recommender.py
import os
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


class SpotifyRecommender:
    """
    Spotify Recommendation API 安定動作版（404完全回避）

    - 利用可能な genre seed を事前取得
    - 無効な genre は自動除外
    - genre が使えない場合は seed_tracks にフォールバック
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

        # Spotify公式が許可している genre seed
        self.available_genres = set(
            self.sp.recommendation_genre_seeds()["genres"]
        )

        # 感情 → genre（候補）
        self.emotion_to_genres = {
            "POS": ["pop", "dance", "happy"],
            "NEU": ["chill", "ambient"],
            "NEG": ["acoustic", "sad"],
        }

    # ======================================================
    # public
    # ======================================================
    def recommend_tracks(
        self,
        audio_features: Dict[str, float],
        emotion_label: str,
        limit: int = 8,
    ) -> List[Dict]:

        # --------------------------
        # 1. 有効な genre のみ使用
        # --------------------------
        candidate_genres = self.emotion_to_genres.get(
            emotion_label, ["pop"]
        )

        seed_genres = [
            g for g in candidate_genres
            if g in self.available_genres
        ]

        # --------------------------
        # 2. Recommendation API
        # --------------------------
        try:
            if seed_genres:
                res = self.sp.recommendations(
                    seed_genres=seed_genres[:2],  # 多すぎると不安定
                    target_valence=audio_features.get("target_valence", 0.5),
                    target_energy=audio_features.get("target_energy", 0.5),
                    target_danceability=audio_features.get(
                        "target_danceability", 0.5
                    ),
                    limit=limit,
                    market=self.market,
                )
            else:
                raise SpotifyException(404, -1, "No valid genre seed")

        except SpotifyException:
            # --------------------------
            # 3. フォールバック：seed_tracks
            # --------------------------
            tracks = self.sp.search(
                q="top hits",
                type="track",
                limit=5,
                market=self.market,
            )["tracks"]["items"]

            seed_tracks = [t["id"] for t in tracks]

            res = self.sp.recommendations(
                seed_tracks=seed_tracks[:2],
                target_valence=audio_features.get("target_valence", 0.5),
                target_energy=audio_features.get("target_energy", 0.5),
                limit=limit,
                market=self.market,
            )

        # --------------------------
        # 4. 整形
        # --------------------------
        results = []
        for t in res["tracks"]:
            results.append({
                "track_name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "external_url": t["external_urls"]["spotify"],
                "preview_url": t["preview_url"],
            })

        return results

# spotify_recommender.py
import numpy as np
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity


class SpotifyRecommender:
    """
    Spotify Search API + Audio Features を用いた
    感情適合型楽曲推薦クラス（安定版）
    """

    FEATURE_KEYS = [
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "tempo",
    ]

    def __init__(self, market: str = "JP"):
        self.market = market
        auth = SpotifyClientCredentials()
        self.sp = spotipy.Spotify(auth_manager=auth)

        self.tempo_min = 60.0
        self.tempo_max = 180.0

        # 感情系検索クエリ（安全）
        self.search_queries = [
            "mood",
            "emotion",
            "chill",
            "happy",
            "sad",
            "relax",
        ]

    # ======================================================
    def recommend_tracks(
        self,
        target_audio_features: Dict[str, float],
        limit: int = 8,
    ) -> List[Dict]:

        target_vec = self._build_target_vector(target_audio_features)

        candidates = self._collect_candidate_tracks()

        if not candidates:
            return []

        track_ids = [t["id"] for t in candidates]
        features = self.sp.audio_features(track_ids)

        scored = []
        for track, feat in zip(candidates, features):
            if feat is None:
                continue

            vec = self._feature_dict_to_vector(feat)
            sim = cosine_similarity(
                target_vec.reshape(1, -1),
                vec.reshape(1, -1)
            )[0][0]

            scored.append((sim, track))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._format_track(t) for _, t in scored[:limit]]

    # ======================================================
    def _collect_candidate_tracks(self, max_tracks: int = 100):
        tracks = {}

        for q in self.search_queries:
            results = self.sp.search(
                q=q,
                type="track",
                limit=20,
                market=self.market,
            )

            for t in results["tracks"]["items"]:
                if t["id"] not in tracks:
                    tracks[t["id"]] = {
                        "id": t["id"],
                        "name": t["name"],
                        "artist": ", ".join(a["name"] for a in t["artists"]),
                        "external_url": t["external_urls"]["spotify"],
                        "preview_url": t["preview_url"],
                    }

                if len(tracks) >= max_tracks:
                    break

        return list(tracks.values())

    def _build_target_vector(self, af: Dict[str, float]) -> np.ndarray:
        vec = []
        for k in self.FEATURE_KEYS:
            if k == "tempo":
                t = af.get("target_tempo", 120.0)
                t = (t - self.tempo_min) / (self.tempo_max - self.tempo_min)
                vec.append(np.clip(t, 0.0, 1.0))
            else:
                vec.append(af.get(f"target_{k}", 0.5))
        return np.array(vec, dtype=np.float32)

    def _feature_dict_to_vector(self, feat: Dict) -> np.ndarray:
        vec = []
        for k in self.FEATURE_KEYS:
            if k == "tempo":
                t = feat["tempo"]
                t = (t - self.tempo_min) / (self.tempo_max - self.tempo_min)
                vec.append(np.clip(t, 0.0, 1.0))
            else:
                vec.append(feat[k])
        return np.array(vec, dtype=np.float32)

    def _format_track(self, t: Dict) -> Dict:
        return {
            "track_name": t["name"],
            "artist": t["artist"],
            "external_url": t["external_url"],
            "preview_url": t["preview_url"],
        }

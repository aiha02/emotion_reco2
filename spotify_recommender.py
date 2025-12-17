# spotify_recommender.py
import numpy as np
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity


class SpotifyRecommender:
    """
    Spotify Audio Features を用いた
    自前・感情適合楽曲推薦クラス

    - Client Credentials Flow を使用
    - market を明示指定（地域制限による 404 対策）
    """

    # 使用する audio feature 順序（重要）
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

        # tempo を他と同スケールにするための正規化用
        self.tempo_min = 60.0
        self.tempo_max = 180.0

        # 候補曲を集めるプレイリスト（公式・比較的安定）
        self.seed_playlists = [
            "37i9dQZF1DX4WYpdgoIcn6",  # Mood Booster
            "37i9dQZF1DX3rxVfibe1L0",  # Mood Ring
            "37i9dQZF1DX889U0CL85jj",  # Life Sucks
        ]

    # ======================================================
    # public API
    # ======================================================
    def recommend_tracks(
        self,
        target_audio_features: Dict[str, float],
        limit: int = 8,
    ) -> List[Dict]:

        # 1. 目標ベクトル
        target_vec = self._build_target_vector(target_audio_features)

        # 2. 候補曲収集
        tracks = self._collect_candidate_tracks()

        if len(tracks) == 0:
            return []

        # 3. Audio Features 取得
        track_ids = [t["id"] for t in tracks]
        features = self.sp.audio_features(track_ids)

        # 4. 類似度計算
        scored = []
        for track, feat in zip(tracks, features):
            if feat is None:
                continue

            vec = self._feature_dict_to_vector(feat)
            sim = cosine_similarity(
                target_vec.reshape(1, -1),
                vec.reshape(1, -1)
            )[0][0]

            scored.append((sim, track))

        # 5. 類似度順に並べて返す
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        return [self._format_track(t) for _, t in top]

    # ======================================================
    # internal
    # ======================================================
    def _collect_candidate_tracks(self, max_tracks: int = 200):
        """
        seed_playlists から楽曲候補を収集
        market を明示指定して 404 を回避
        """
        tracks = {}

        for pid in self.seed_playlists:
            try:
                results = self.sp.playlist_items(
                    pid,
                    additional_types=["track"],
                    limit=100,
                    market=self.market,
                )
            except Exception as e:
                print(f"プレイリスト取得失敗 ({pid}):", e)
                continue

            for item in results.get("items", []):
                track = item.get("track")
                if not track or track.get("id") is None:
                    continue

                tracks[track["id"]] = {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join(a["name"] for a in track["artists"]),
                    "external_url": track["external_urls"]["spotify"],
                    "preview_url": track["preview_url"],
                }

                if len(tracks) >= max_tracks:
                    break

        return list(tracks.values())

    def _build_target_vector(self, audio_features: Dict[str, float]) -> np.ndarray:
        """
        emotion_state_to_audio_features の出力を
        推薦用ベクトルに変換
        """
        vec = []

        for k in self.FEATURE_KEYS:
            if k == "tempo":
                t = audio_features.get("target_tempo", 120.0)
                t = (t - self.tempo_min) / (self.tempo_max - self.tempo_min)
                vec.append(np.clip(t, 0.0, 1.0))
            else:
                vec.append(audio_features.get(f"target_{k}", 0.5))

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

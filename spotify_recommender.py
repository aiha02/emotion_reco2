# spotify_recommender.py
import os
import numpy as np
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from sklearn.metrics.pairwise import cosine_similarity


class SpotifyRecommender:
    """
    Spotify Search API + Audio Features を用いた
    感情適合型楽曲推薦クラス（403対策・安定動作版）

    - Recommendation API は使用しない
    - Search API で候補曲を収集
    - Audio Features を安全にバッチ取得
    - 感情特徴ベクトルとのコサイン類似度で再ランキング
    """

    # 使用する Audio Features（順序重要）
    FEATURE_KEYS = [
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "tempo",
    ]

    def __init__(self, market: str = "JP"):
        # ===============================
        # 環境変数チェック（重要）
        # ===============================
        if not os.getenv("SPOTIPY_CLIENT_ID"):
            raise RuntimeError("SPOTIPY_CLIENT_ID が設定されていません")
        if not os.getenv("SPOTIPY_CLIENT_SECRET"):
            raise RuntimeError("SPOTIPY_CLIENT_SECRET が設定されていません")

        self.market = market

        auth = SpotifyClientCredentials()
        self.sp = spotipy.Spotify(auth_manager=auth)

        # tempo 正規化用
        self.tempo_min = 60.0
        self.tempo_max = 180.0

        # Search API 用の安定クエリ
        self.search_queries = [
            "mood",
            "emotion",
            "chill",
            "happy",
            "sad",
            "relax",
        ]

    # ======================================================
    # public API
    # ======================================================
    def recommend_tracks(
        self,
        target_audio_features: Dict[str, float],
        limit: int = 8,
        candidate_size: int = 120,
    ) -> List[Dict]:
        """
        感情推定結果から楽曲推薦を行う
        """

        # 1. 感情 → 目標ベクトル
        target_vec = self._build_target_vector(target_audio_features)

        # 2. 候補曲収集
        candidates = self._collect_candidate_tracks(
            max_tracks=candidate_size
        )

        if not candidates:
            return []

        # 3. Audio Features を安全にバッチ取得
        track_ids = [t["id"] for t in candidates]
        features = self._get_audio_features_batched(track_ids)

        # 4. コサイン類似度による再ランキング
        scored = []
        for track, feat in zip(candidates, features):
            if feat is None:
                continue

            vec = self._feature_dict_to_vector(feat)
            sim = cosine_similarity(
                target_vec.reshape(1, -1),
                vec.reshape(1, -1),
            )[0][0]

            scored.append((sim, track))

        # 5. 類似度順に上位を返す
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._format_track(t) for _, t in scored[:limit]]

    # ======================================================
    # internal
    # ======================================================
    def _collect_candidate_tracks(self, max_tracks: int = 120) -> List[Dict]:
        """
        Search API により候補楽曲を収集
        - is_local=True の曲は除外（403対策）
        """
        tracks = {}

        for q in self.search_queries:
            results = self.sp.search(
                q=q,
                type="track",
                limit=20,
                market=None,  # market制限を外す（重要）
            )

            for t in results["tracks"]["items"]:
                if t["is_local"]:
                    continue

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

            if len(tracks) >= max_tracks:
                break

        return list(tracks.values())

    def _get_audio_features_batched(
        self,
        track_ids: List[str],
        batch_size: int = 50,  # 安全のため50
    ) -> List[Dict]:
        """
        Audio Features API 制約対応（403耐性あり）
        """
        all_features = []

        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]
            try:
                feats = self.sp.audio_features(batch)
                for f in feats:
                    all_features.append(f)
            except SpotifyException as e:
                print("Audio features error, skip batch:", e)
                all_features.extend([None] * len(batch))

        return all_features

    def _build_target_vector(self, af: Dict[str, float]) -> np.ndarray:
        """
        感情推定結果 → 推薦用特徴ベクトル
        """
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
        """
        Spotify Audio Features → 推薦用特徴ベクトル
        """
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
        """
        出力形式を統一
        """
        return {
            "track_name": t["name"],
            "artist": t["artist"],
            "external_url": t["external_url"],
            "preview_url": t["preview_url"],
        }

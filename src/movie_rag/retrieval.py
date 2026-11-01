"""基于真实 API 的电影检索与可追溯训练评分证据。"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class Retriever:
    def __init__(self, store, client, cache_dir, top_k=30, coarse_k=50):
        if not 10 <= top_k <= coarse_k:
            raise ValueError("筛选数量须满足 10 <= top_k <= coarse_k")
        if not getattr(client, "embedding_model", None):
            raise ValueError("client.embedding_model 必须指定，防止混用缓存")
        self.store = store
        self.client = client
        self.top_k = top_k
        self.coarse_k = coarse_k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 缓存身份同时绑定模型、维度及服务地址，避免跨地域或配置复用。
        self.embedding_identity = {
            "model": client.embedding_model,
            "dimensions": getattr(client, "dimensions", None),
            "base_url": getattr(client, "base_url", None),
        }
        model_key = hashlib.sha256(json.dumps(self.embedding_identity, sort_keys=True).encode()).hexdigest()[:16]
        self.cache_path = self.cache_dir / f"embeddings_{model_key}.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        self.movie_ids = sorted(store.movies)
        self.movie_columns = {mid: index for index, mid in enumerate(self.movie_ids)}
        self.user_ids = sorted(store.user_ratings)
        self.matrix = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        for i, uid in enumerate(self.user_ids):
            for mid, rating in store.user_ratings[uid].items():
                self.matrix[i, self.movie_columns[mid]] = (rating - 3) / 2
        self.norms = np.linalg.norm(self.matrix, axis=1)

    def _vectors(self, texts, text_type):
        keys = [hashlib.sha256(json.dumps([self.embedding_identity, text_type, text],
                                        ensure_ascii=False).encode()).hexdigest() for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        text_by_key = dict(zip(keys, texts))
        for start in range(0, len(missing), 10):
            batch = missing[start:start + 10]
            vectors = self.client.embed([text_by_key[key] for key in batch], text_type=text_type)
            array = np.asarray(vectors, dtype=float)
            if array.ndim != 2 or len(array) != len(batch) or not np.isfinite(array).all():
                raise ValueError("Embedding API 返回无效向量")
            if np.any(np.linalg.norm(array, axis=1) == 0):
                raise ValueError("Embedding API 返回零向量")
            self.cache.update({key: vector for key, vector in zip(batch, array.tolist())})
            # 每批完成即保存，API 中断后可以从已完成批次继续。
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.cache_path)
        array = np.asarray([self.cache[key] for key in keys], dtype=float)
        return array / np.linalg.norm(array, axis=1, keepdims=True)

    def _profile(self, case):
        genre_ratings = defaultdict(list)
        for row in case["history"]:
            for genre in row["genres"]:
                genre_ratings[genre].append(row["rating"])
        ordered = sorted(genre_ratings, key=lambda genre: (-np.mean(genre_ratings[genre]), genre))
        genre_text = "; ".join(f"{genre}: mean={np.mean(genre_ratings[genre]):.2f}/5, n={len(genre_ratings[genre])}"
                               for genre in ordered)
        high = sorted((row for row in case["history"] if row["rating"] >= 4),
                      key=lambda row: (-row["rating"], -row.get("timestamp", 0), row["movie_id"]))[:8]
        low = sorted((row for row in case["history"] if row["rating"] <= 2),
                     key=lambda row: (row["rating"], -row.get("timestamp", 0), row["movie_id"]))[:5]
        describe = lambda rows: "; ".join(f"{row['title']} ({', '.join(row['genres'])}), {row['rating']}/5" for row in rows)
        query = ("Recommend unseen candidate movies for this user. Genre preferences from training history: "
                 + genre_text + ". Liked movies: " + describe(high) + ". Disliked movies: " + describe(low)
                 + ". Prefer genres and films matching the positive preferences; avoid negative preferences.")
        return query

    def _document(self, mid):
        movie = self.store.movies[mid]
        ratings = self.store.movie_ratings[mid]
        mean = float(np.mean(ratings)) if ratings else None
        text = (f"Movie ID: {mid}. Title: {movie['title']}. Release date: {movie['release_date']}. "
                f"Genres: {', '.join(movie['genres'])}. Training ratings: {len(ratings)}; "
                f"mean rating: {mean:.2f}/5; positive ratings (>=4): {self.store.positive_counts[mid]}."
                if ratings else
                f"Movie ID: {mid}. Title: {movie['title']}. Release date: {movie['release_date']}. "
                f"Genres: {', '.join(movie['genres'])}. No training ratings.")
        return text

    def _neighbors(self, case, candidate_ids):
        profile = np.zeros(len(self.movie_ids), dtype=np.float32)
        for row in case["history"]:
            profile[self.movie_columns[int(row["movie_id"])]] = (row["rating"] - 3) / 2
        denominator = self.norms * np.linalg.norm(profile)
        similarities = np.divide(self.matrix @ profile, denominator,
                                 out=np.zeros(len(self.user_ids), dtype=float), where=denominator > 0)
        for i, uid in enumerate(self.user_ids):
            if uid == case["user_id"]:
                similarities[i] = 0
        indices = sorted((i for i, score in enumerate(similarities) if score > 0),
                         key=lambda i: (-similarities[i], self.user_ids[i]))[:40]
        scores, evidence = {}, {}
        for mid in candidate_ids:
            support = [(self.user_ids[i], float(similarities[i]), self.store.user_ratings[self.user_ids[i]][mid])
                       for i in indices if mid in self.store.user_ratings[self.user_ids[i]]]
            weight = sum(item[1] for item in support)
            # 两个中性伪观测减小单个近邻评分带来的极端结果。
            scores[mid] = sum(sim * (rating - 3) / 2 for _, sim, rating in support) / (weight + 2)
            evidence[mid] = [{"user_id": uid, "similarity": round(sim, 6), "rating": rating}
                             for uid, sim, rating in support[:5]]
        return scores, evidence

    def retrieve(self, case):
        seen = {int(row["movie_id"]) for row in case["history"]}
        original_ids = sorted({int(movie["movie_id"]) for movie in case["candidates"]} - seen)
        if len(original_ids) < self.coarse_k:
            raise ValueError(f"不足 {self.coarse_k} 部未看候选电影")
        query = self._profile(case)
        documents = [self._document(mid) for mid in original_ids]
        vectors = self._vectors(documents, "document")
        query_vector = self._vectors([query], "query")[0]
        dense = vectors @ query_vector
        collaborative, neighbors = self._neighbors(case, original_ids)
        coarse_scores = {mid: 0.65 * (float(dense[i]) + 1) / 2
                         + 0.35 * (collaborative[mid] + 1) / 2
                         for i, mid in enumerate(original_ids)}
        # 粗筛只使用向量与近邻支持，后续仅由重排模型决定精筛顺序。
        selected = sorted(range(len(original_ids)),
                          key=lambda i: (-coarse_scores[original_ids[i]], original_ids[i]))[:self.coarse_k]
        candidate_ids = [original_ids[i] for i in selected]
        selected_documents = [documents[i] for i in selected]
        response = self.client.rerank(query, selected_documents, top_n=len(selected_documents))
        rerank_scores = {}
        for result in response:
            index = int(result["index"])
            score = float(result["relevance_score"])
            if index < 0 or index >= len(selected) or index in rerank_scores or not np.isfinite(score):
                raise ValueError("Rerank API 返回无效索引或分数")
            rerank_scores[index] = score
        if len(rerank_scores) != len(selected):
            raise ValueError("Rerank API 未返回请求的全部候选结果")
        evidence = []
        for i, mid in enumerate(candidate_ids):
            movie = self.store.movies[mid]
            evidence.append({"evidence_id": f"movie_{mid}", "movie_id": mid,
                             "title": movie["title"], "genres": movie["genres"],
                             "content": selected_documents[i], "score": rerank_scores[i],
                             "dense_score": float(dense[selected[i]]),
                             "collaborative_score": collaborative[mid],
                             "coarse_score": coarse_scores[mid], "rerank_score": rerank_scores[i],
                             "neighbor_ratings": neighbors[mid],
                             "source": ["movies.csv", "train_ratings.csv"],
                             "positive_rating_count": self.store.positive_counts[mid]})
        evidence.sort(key=lambda row: (-row["score"], row["movie_id"]))
        evidence = evidence[:self.top_k]
        return {"query": query, "evidence": evidence,
                "meta": {"pipeline": "cascade", "embedding_model": self.client.embedding_model,
                         "top_k": self.top_k, "coarse_k": self.coarse_k,
                         "candidate_count": len(original_ids), "rerank_candidate_count": len(selected),
                         "neighbor_limit": 40,
                         "stage_candidate_ids": {"coarse": candidate_ids,
                                                 "rerank": [row["movie_id"] for row in evidence]},
                         "coarse_score_formula": "0.65*(cosine+1)/2+0.35*(cf+1)/2",
                         "score_formula": "rerank",
                         "profile_rule": "全部历史计算类型均分；最多8部高分和5部低分电影，评分优先、时间降序"}}

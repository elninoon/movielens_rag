"""使用合成数据检查检索契约，不访问真实 API 或验证标签。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from movie_rag.data import DataStore, load_cases, popular_recommendations
from movie_rag.retrieval import Retriever


class FakeClient:
    embedding_model = "fixture-embedding"
    dimensions = 3
    base_url = "https://fixture.invalid/v1"

    def __init__(self):
        self.embedding_calls = []
        self.rerank_calls = 0

    def embed(self, texts, text_type="document"):
        self.embedding_calls.append((list(texts), text_type))
        return [[1.0, (len(text) % 11 + 1) / 11, 0.25] for text in texts]

    def rerank(self, query, documents, top_n):
        self.rerank_calls += 1
        return [{"index": i, "relevance_score": 1 / (i + 1)} for i in range(top_n)]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with (self.root / "movies.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["movie_id", "title", "release_date", "genres"])
            writer.writeheader()
            for mid in range(1, 15):
                writer.writerow({"movie_id": mid, "title": f"Movie {mid}",
                                 "release_date": "1995", "genres": "Comedy|Drama"})
        with (self.root / "train_ratings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["user_id", "movie_id", "rating", "timestamp"])
            writer.writeheader()
            for uid, mid, rating in [(1, 1, 5), (2, 1, 5), (3, 1, 4),
                                     (2, 2, 4), (3, 2, 5), (2, 3, 5), (3, 3, 5),
                                     (4, 4, 3), (5, 4, 1)]:
                writer.writerow({"user_id": uid, "movie_id": mid, "rating": rating, "timestamp": 1})
        self.store = DataStore(self.root)
        self.case = {
            "user_id": 1,
            "history": [{"movie_id": 1, "title": "Movie 1", "genres": ["Comedy", "Drama"],
                         "rating": 5, "timestamp": 1}],
            # 故意包括已看电影并排除库内电影14，检查边界。
            "candidates": [self.store.movies[mid] for mid in range(1, 14)],
        }
        self.client = FakeClient()

    def tearDown(self):
        self.temporary.cleanup()

    def test_popularity_ties_history_and_low_ratings(self):
        result = popular_recommendations(self.store, self.case)
        ids = [row["movie_id"] for row in result["recommendations"]]
        self.assertEqual(ids, list(range(2, 12)))
        self.assertEqual([row["rank"] for row in result["recommendations"]], list(range(1, 11)))
        self.assertEqual(self.store.positive_counts[4], 0)
        self.assertEqual(result["method"], "popular")

    def test_candidate_boundary_and_top_k(self):
        retriever = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        result = retriever.retrieve(self.case)
        self.assertEqual(len(result["evidence"]), 10)
        ids = [row["movie_id"] for row in result["evidence"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= set(range(2, 14)))
        self.assertEqual(result["meta"]["candidate_count"], 12)
        self.assertTrue(all(row["source"] == ["movies.csv", "train_ratings.csv"] for row in result["evidence"]))
        self.assertTrue(all(len(texts) <= 10 for texts, _ in self.client.embedding_calls))
        self.assertEqual(len(self.client.embedding_calls), 3)

    def test_persistent_cache_avoids_repeated_embedding(self):
        retriever = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        first = retriever.retrieve(self.case)
        calls = len(self.client.embedding_calls)
        second = retriever.retrieve(self.case)
        self.assertEqual(len(self.client.embedding_calls), calls)
        self.assertEqual(first, second)
        restarted = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        self.assertEqual(restarted.retrieve(self.case), first)
        self.assertEqual(len(self.client.embedding_calls), calls)

    def test_cache_identity_tracks_dimensions_and_service(self):
        original = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        self.client.dimensions = 1024
        dimensional = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        self.client.base_url = "https://other-region.invalid/v1"
        regional = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        self.assertEqual(len({original.cache_path, dimensional.cache_path, regional.cache_path}), 3)

    def test_no_validation_labels_required_or_read(self):
        # 不存在标签时可运行，存在无法解析的标签时结果也不变。
        retriever = Retriever(self.store, self.client, self.root / "cache", top_k=10)
        first = retriever.retrieve(self.case)
        (self.root / "validation_labels.jsonl").write_text("THIS IS NOT JSON", encoding="utf-8")
        cases_path = self.root / "validation_cases.jsonl"
        cases_path.write_text(json.dumps(self.case) + "\n", encoding="utf-8")
        reloaded = DataStore(self.root)
        cases = load_cases(cases_path)
        result = Retriever(reloaded, self.client, self.root / "cache", top_k=10).retrieve(cases[0])
        self.assertEqual(result, first)

    def test_no_rerank_and_minimum_top_k(self):
        result = Retriever(self.store, self.client, self.root / "cache", top_k=30,
                           use_rerank=False).retrieve(self.case)
        self.assertEqual(len(result["evidence"]), 12)
        self.assertEqual(self.client.rerank_calls, 0)
        with self.assertRaises(ValueError):
            Retriever(self.store, self.client, self.root / "cache", top_k=9)


if __name__ == "__main__":
    unittest.main()

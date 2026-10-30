"""使用合成数据验证排序指标、固定分母及输入输出隔离约束。"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movie_rag.evaluation import compare_methods, evaluate_results, select_failure_cases


def case(uid):
    return {"user_id": uid,
            "history": [{"movie_id": 999, "rating": 5, "genres": ["Drama"]}],
            "candidates": [{"movie_id": mid, "title": str(mid), "genres": ["Drama"]}
                           for mid in range(1, 101)]}


def result(uid, ids=range(1, 11)):
    return {"user_id": uid, "status": "ok", "method": "synthetic",
            "recommendations": [{"movie_id": mid, "rank": rank,
                                 "reason": "依据类型", "evidence": ["movie metadata"]}
                                for rank, mid in enumerate(ids, 1)]}


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_rank_and_binary_relevance(self):
        rec = result(1)
        rec["recommendations"].reverse()
        report = evaluate_results([case(1)], [{"user_id": 1, "relevant_movie_ids": [1, 3, 99]}], [rec])
        summary = report["summary"]
        self.assertAlmostEqual(summary["recall_at_10"], 2 / 3)
        self.assertAlmostEqual(summary["precision_at_10"], .2)
        self.assertEqual(summary["hit_rate_at_10"], 1)
        self.assertEqual(summary["mrr_at_10"], 1)
        ideal = sum(1 / math.log2(k + 2) for k in range(3))
        self.assertAlmostEqual(summary["ndcg_at_10"], (1 + 1 / math.log2(4)) / ideal)

    def test_missing_failed_users_remain_in_denominator(self):
        report = evaluate_results([case(1), case(2), case(3)],
                                  [{"user_id": uid, "relevant_movie_ids": [1]} for uid in (1, 2, 3)],
                                  [result(1), {"user_id": 2, "status": "error", "recommendations": []}])
        self.assertEqual(report["summary"]["metric_user_count"], 3)
        self.assertAlmostEqual(report["summary"]["recall_at_10"], 1 / 3)
        self.assertEqual(report["summary"]["failed_or_invalid_user_count"], 2)

    def test_invalid_id_history_duplicate_and_rank(self):
        rec = result(1)
        rec["recommendations"][0]["movie_id"] = 999
        rec["recommendations"][1]["movie_id"] = 999
        rec["recommendations"][1]["rank"] = 1
        report = evaluate_results([case(1)], [{"user_id": 1, "relevant_movie_ids": [3]}], [rec])
        codes = {issue["code"] for issue in report["issues"]}
        self.assertTrue({"movie_in_history", "movie_outside_candidates", "duplicate_movie", "invalid_ranks"} <= codes)
        self.assertEqual(report["summary"]["recall_at_10"], 0)

    def test_empty_relevance_is_explicitly_excluded(self):
        report = evaluate_results([case(1), case(2)],
                                  [{"user_id": 1, "relevant_movie_ids": []},
                                   {"user_id": 2, "relevant_movie_ids": [1]}], [result(1), result(2)])
        self.assertEqual(report["summary"]["excluded_label_user_count"], 1)
        self.assertIsNone(report["per_user"][0]["recall_at_10"])
        self.assertEqual(report["summary"]["recall_at_10"], 1)

    def test_label_in_history_or_outside_candidates_is_rejected(self):
        report = evaluate_results([case(1)], [{"user_id": 1, "relevant_movie_ids": [999]}], [result(1)])
        self.assertEqual(report["summary"]["metric_user_count"], 0)
        self.assertIsNone(report["summary"]["ndcg_at_10"])

    def test_methods_share_population(self):
        report = compare_methods([case(1), case(2)],
                                 [{"user_id": uid, "relevant_movie_ids": [1]} for uid in (1, 2)],
                                 {"full": [result(1), result(2)], "partial": [result(1)]})
        self.assertEqual(report["summary"]["full"]["recall_at_10"], 1)
        self.assertEqual(report["summary"]["partial"]["recall_at_10"], .5)

    def test_duplicate_results_are_invalid(self):
        report = evaluate_results([case(1)], [{"user_id": 1, "relevant_movie_ids": [1]}], [result(1), result(1)])
        self.assertEqual(report["summary"]["valid_rate"], 0)
        self.assertEqual(report["summary"]["recall_at_10"], 0)

    def test_failure_analysis_distinguishes_retrieval_and_ranking(self):
        rec = result(1)
        rec["retrieval"] = {"evidence": [{"movie_id": 50}]}
        failures = select_failure_cases([case(1)], [{"user_id": 1, "relevant_movie_ids": [50, 60]}], [rec], [result(1)])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["retrieval_analysis"]["missed_after_retrieval"], [50])
        self.assertEqual(failures[0]["retrieval_analysis"]["missed_not_retrieved"], [60])
        self.assertEqual(failures[0]["history_profile"]["liked_genre_counts"], {"Drama": 1})

    def test_api_error_is_not_selected_as_recommendation_failure(self):
        # 接口失败的差值可能比真正漏推荐更差，但不应占用3个失败案例名额。
        cases = [case(uid) for uid in (1, 2, 3, 4)]
        labels = [{"user_id": uid, "relevant_movie_ids": [99]} for uid in (1, 2, 3, 4)]
        results = [{"user_id": 1, "status": "error", "error": "HTTP403", "recommendations": []}]
        results += [result(uid) for uid in (2, 3, 4)]
        baselines = [result(uid, [99] + list(range(1, 10))) for uid in (1, 2, 3, 4)]
        failures = select_failure_cases(cases, labels, results, baselines, n=3)
        self.assertEqual({row["user_id"] for row in failures}, {2, 3, 4})
        report = evaluate_results(cases, labels, results)
        self.assertEqual(report["summary"]["metric_user_count"], 4)
        self.assertEqual(report["summary"]["failed_or_invalid_user_count"], 1)


if __name__ == "__main__":
    unittest.main()

"""验证生成约束与消融输入，使用合成数据且不调用模型。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movie_rag.generation import build_messages, validate_response


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "user_id": 7,
            "history": [{"movie_id": mid, "title": f"历史电影{mid}",
                         "genres": ["Drama"], "rating": mid % 5 + 1,
                         "timestamp": mid * 100}
                        for mid in range(1001, 1201)],
            "candidates": [{"movie_id": mid, "title": f"候选电影{mid}",
                            "genres": ["Drama"], "release_date": "01-Jan-1995"}
                           for mid in range(1, 101)],
        }
        self.retrieval = {"query": "偏好Drama", "top_k": 30,
                          "evidence": [{"evidence_id": "movie_1", "movie_id": 1,
                                        "content": "候选1类型为Drama", "score": .9}]}
        _, self.sources = build_messages(self.case, self.retrieval)
        self.payload = {"recommendations": [
            {"movie_id": mid, "rank": mid, "reason": "候选属于Drama类型，可结合训练历史判断。",
             "evidence_ids": [f"candidate_{mid}", "history_1001"]}
            for mid in range(1, 11)]}

    def test_correct_sources_are_resolved_and_duplicates_removed(self):
        self.payload["recommendations"][0]["evidence_ids"] += ["movie_1", "movie_1"]
        output = validate_response(self.payload, self.case, self.sources)
        self.assertEqual(len(output), 10)
        self.assertEqual([r["rank"] for r in output], list(range(1, 11)))
        first = output[0]
        self.assertEqual([ev["evidence_id"] for ev in first["evidence"]],
                         ["candidate_1", "history_1001", "movie_1"])
        self.assertEqual(first["evidence"][2]["content"], "候选1类型为Drama")

    def test_movie_outside_candidates_rejected(self):
        self.payload["recommendations"][0]["movie_id"] = 2000
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, self.sources)

    def test_history_movie_rejected_even_if_in_candidates(self):
        self.case["candidates"][0] = deepcopy(self.case["history"][0])
        self.payload["recommendations"][0]["movie_id"] = 1001
        self.payload["recommendations"][0]["evidence_ids"] = ["candidate_1001"]
        _, sources = build_messages(self.case, self.retrieval)
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, sources)

    def test_duplicate_movie_rejected(self):
        self.payload["recommendations"][1]["movie_id"] = 1
        self.payload["recommendations"][1]["evidence_ids"] = ["candidate_1"]
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, self.sources)

    def test_wrong_or_reordered_rank_rejected(self):
        for invalid in (0, 2, "1", 1.5, None, True, 1.0):
            with self.subTest(rank=invalid):
                payload = deepcopy(self.payload)
                payload["recommendations"][0]["rank"] = invalid
                with self.assertRaises(ValueError):
                    validate_response(payload, self.case, self.sources)
        self.payload["recommendations"].reverse()
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, self.sources)

    def test_non_object_recommendation_is_rejected(self):
        self.payload['recommendations'][0] = '非法字符串'
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, self.sources)

    def test_forged_history_or_retrieval_reference_rejected(self):
        for invalid in ("history_5000", "movie_5000", "candidate_2000", 1, None):
            with self.subTest(reference=invalid):
                payload = deepcopy(self.payload)
                payload["recommendations"][0]["evidence_ids"].append(invalid)
                with self.assertRaises(ValueError):
                    validate_response(payload, self.case, self.sources)

    def test_each_recommendation_must_reference_itself(self):
        self.payload["recommendations"][0]["evidence_ids"] = ["candidate_2", "history_1001"]
        with self.assertRaises(ValueError):
            validate_response(self.payload, self.case, self.sources)

    def test_exactly_ten_recommendations_required(self):
        for count in (0, 9, 11):
            with self.subTest(count=count):
                payload = {"recommendations": (self.payload["recommendations"] +
                                                [deepcopy(self.payload["recommendations"][-1])])[:count]}
                with self.assertRaises(ValueError):
                    validate_response(payload, self.case, self.sources)

    def test_empty_reason_rejected(self):
        for invalid in ("", "  ", None):
            with self.subTest(reason=invalid):
                payload = deepcopy(self.payload)
                payload["recommendations"][0]["reason"] = invalid
                with self.assertRaises(ValueError):
                    validate_response(payload, self.case, self.sources)

    def test_full_history_preserved_without_compression(self):
        original = deepcopy(self.case)
        messages, sources = build_messages(self.case, self.retrieval)
        sent = json.loads(messages[1]["content"])
        self.assertEqual(len(sent["history"]), 200)
        self.assertEqual(len(sent["candidates"]), 100)
        for original_row, sent_row in zip(original["history"], sent["history"]):
            self.assertEqual(sent_row,
                             {"evidence_id": f"history_{original_row['movie_id']}", **original_row})
            self.assertEqual(sources[sent_row["evidence_id"]], sent_row)
        self.assertEqual(self.case, original)

    def test_cascade_prompt_and_validation_only_allow_retained_candidates(self):
        case = {**self.case, 'candidates': self.case['candidates'][:30]}
        messages, sources = build_messages(case, self.retrieval)
        self.assertIn('给定30部候选', messages[0]['content'])
        self.assertEqual(len(json.loads(messages[1]['content'])['history']), 200)
        self.assertNotIn('candidate_31', sources)
        payload = deepcopy(self.payload)
        payload['recommendations'][0]['movie_id'] = 31
        payload['recommendations'][0]['evidence_ids'] = ['candidate_31']
        with self.assertRaises(ValueError):
            validate_response(payload, case, sources)

    def test_ablation_only_changes_retrieved_evidence_in_messages(self):
        ablation = {**self.retrieval, "evidence": []}
        full_messages, full_sources = build_messages(self.case, self.retrieval)
        ablated_messages, ablated_sources = build_messages(self.case, ablation)
        self.assertEqual(full_messages[0], ablated_messages[0])
        full = json.loads(full_messages[1]["content"])
        removed = json.loads(ablated_messages[1]["content"])
        self.assertEqual(full.pop("retrieved_evidence"), self.retrieval["evidence"])
        self.assertEqual(removed.pop("retrieved_evidence"), [])
        self.assertEqual(full, removed)
        self.assertEqual(set(full_sources) - set(ablated_sources), {"movie_1"})
        self.assertTrue(all(full_sources[key] == value for key, value in ablated_sources.items()))


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from movie_rag.grounding import ground_recommendations


class GroundingTests(unittest.TestCase):
    def test_low_rating_is_not_called_preference(self):
        case = {'candidates': [{'movie_id': 1, 'title': '候选', 'genres': ['Drama']}],
                'history': [{'movie_id': 2, 'title': '低分片', 'genres': ['Drama'], 'rating': 3},
                            {'movie_id': 3, 'title': '高分片', 'genres': ['Drama'], 'rating': 4}]}
        row = {'movie_id': 1, 'rank': 1, 'reason': '用户喜欢3分片',
               'evidence': [{'evidence_id': 'history_2', **case['history'][0]}]}
        result = ground_recommendations(case, [row])[0]
        self.assertIn('高分片', result['reason'])
        self.assertIn('4分', result['reason'])
        self.assertNotIn('低分片', result['reason'])
        self.assertEqual(result['model_reason'], row['reason'])
        self.assertEqual(result['movie_id'], 1)
        self.assertEqual(result['rank'], 1)

    def test_missing_support_is_explicit(self):
        case = {'candidates': [{'movie_id': 1, 'title': '候选', 'genres': ['Comedy']}],
                'history': [{'movie_id': 2, 'title': '历史', 'genres': ['Drama'], 'rating': 5}]}
        result = ground_recommendations(case, [{'movie_id': 1, 'rank': 1, 'reason': '模型文本',
            'evidence': [{'evidence_id': 'movie_1', 'positive_rating_count': 7}]}])[0]
        self.assertIn('探索', result['reason'])
        self.assertIn('7次', result['reason'])
        self.assertEqual([e['evidence_id'] for e in result['evidence']], ['candidate_1', 'movie_1'])

    def test_rendering_is_idempotent(self):
        case = {'candidates': [{'movie_id': 1, 'title': '候选', 'genres': ['Drama']}], 'history': []}
        rows = [{'movie_id': 1, 'rank': 1, 'reason': '原始理由', 'evidence': []}]
        once = ground_recommendations(case, rows)
        twice = ground_recommendations(case, once)
        self.assertEqual(once, twice)


if __name__ == '__main__':
    unittest.main()

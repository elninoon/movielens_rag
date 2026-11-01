"""用完整100候选与假API验证CLI级联、消融、缓存及服务续跑边界。"""
import argparse
import csv
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from movie_rag import cli


class FakeClient:
    embedding_model = 'fixture-embedding'
    dimensions = 3
    base_url = 'https://fixture.invalid/v1'
    requests = []
    rerank_sizes = []

    def __init__(self, root):
        self.calls = []

    def describe(self):
        return {'generation_model': 'fixture-generation'}

    def embed(self, texts, text_type='document'):
        return [[1.0, (len(text) % 11 + 1) / 11, .25] for text in texts]

    def rerank(self, query, documents, top_n):
        self.rerank_sizes.append(len(documents))
        return [{'index': i, 'relevance_score': 1 / (i + 1)} for i in range(top_n)]

    def chat(self, messages):
        payload = json.loads(messages[1]['content'])
        self.requests.append((messages[0]['content'], payload))
        rows = [{'movie_id': movie['movie_id'], 'rank': rank,
                 'reason': '依据真实候选类型进行推荐。',
                 'evidence_ids': [movie['evidence_id']]}
                for rank, movie in enumerate(payload['candidates'][:10], 1)]
        return {'choices': [{'message': {'content': json.dumps({'recommendations': rows})}}]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data' / 'movielens_dev'
        self.data.mkdir(parents=True)
        movies = [{'movie_id': mid, 'title': f'Movie {mid}', 'genres': 'Drama',
                   'release_date': '1995'} for mid in range(1, 103)]
        ratings = [{'user_id': uid, 'movie_id': mid, 'rating': rating, 'timestamp': 1}
                   for uid, mid, rating in [(1, 101, 5), (2, 102, 4), (3, 101, 5), (3, 1, 4)]]
        for name, rows in [('movies', movies), ('train_ratings', ratings)]:
            with (self.data / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        candidates = [{**movie, 'genres': ['Drama']} for movie in movies[:100]]
        self.cases = [{'user_id': uid,
                       'history': [{'movie_id': mid, 'title': f'Movie {mid}', 'genres': ['Drama'],
                                    'rating': 5, 'timestamp': 1}],
                       'candidates': candidates} for uid, mid in [(1, 101), (2, 102)]]
        for name, rows in [('validation_cases', self.cases), ('validation_labels', [
                {'user_id': uid, 'relevant_movie_ids': [1, 2]} for uid in (1, 2)])]:
            (self.data / f'{name}.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.args = argparse.Namespace(data=str(self.data), name='cascade', limit=None,
                                       retry_errors=False, workers=1)
        FakeClient.requests = []
        FakeClient.rerank_sizes = []

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self):
        with patch.object(cli, 'ROOT', self.root), patch.object(cli, 'BailianClient', FakeClient), contextlib.redirect_stdout(io.StringIO()):
            cli.run_experiment(self.args)

    def test_default_pipeline_and_ablation_share_the_same_thirty_candidates(self):
        self.run_pipeline()
        self.assertEqual(FakeClient.rerank_sizes, [50, 50])
        self.assertEqual(len(FakeClient.requests), 4)
        for case in self.cases:
            sent = [(system, row) for system, row in FakeClient.requests if row['user_id'] == case['user_id']]
            self.assertEqual(len(sent), 2)
            self.assertEqual(sent[0][0], sent[1][0])
            self.assertIn('给定30部候选', sent[0][0])
            self.assertEqual(sent[0][1]['candidates'], sent[1][1]['candidates'])
            self.assertEqual(len(sent[0][1]['candidates']), 30)
            for _, row in sent:
                self.assertEqual(row['history'][0]['movie_id'], case['history'][0]['movie_id'])
            self.assertEqual(sorted(len(row['retrieved_evidence']) for _, row in sent), [0, 30])
        evaluation = json.loads((self.root / 'outputs/cascade/evaluation.json').read_text(encoding='utf-8'))
        self.assertTrue(all(row['user_count'] == 2 for row in evaluation['summary'].values()))
        self.run_pipeline()
        self.assertEqual(len(FakeClient.requests), 4)

    def test_retry_does_not_resample_model_validation_failure(self):
        self.run_pipeline()
        result_path = self.root / 'outputs/cascade/rag/1.json'
        failed = {'user_id': 1, 'status': 'error', 'error': '必须恰好返回10条推荐',
                  'recommendations': [], 'calls': [{'kind': 'generation', 'ok': False}, {'kind': 'generation', 'ok': True}]}
        result_path.write_text(json.dumps(failed), encoding='utf-8')
        self.args.retry_errors = True
        self.run_pipeline()
        self.assertEqual(len(FakeClient.requests), 4)
        self.assertEqual(json.loads(result_path.read_text(encoding='utf-8')), failed)

    def test_service_retry_reuses_retrieval_and_only_regenerates_failed_user(self):
        self.run_pipeline()
        result_path = self.root / 'outputs/cascade/rag/1.json'
        failed = {'user_id': 1, 'status': 'error', 'error': 'HTTP 403',
                  'recommendations': [], 'calls': [{'kind': 'generation', 'ok': False}]}
        result_path.write_text(json.dumps(failed), encoding='utf-8')
        self.args.retry_errors = True
        self.run_pipeline()
        self.assertEqual(len(FakeClient.requests), 5)
        self.assertEqual(FakeClient.rerank_sizes, [50, 50])
        self.assertEqual(json.loads(result_path.read_text(encoding='utf-8'))['status'], 'ok')
        self.assertEqual(len(list((self.root / 'outputs/cascade/errors/rag').glob('1_*.json'))), 1)

    def test_verifier_uses_configured_users_for_limited_experiment(self):
        self.args.limit = 1
        self.run_pipeline()
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_outputs.py'
        spec = importlib.util.spec_from_file_location('verify_outputs_fixture', script)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with patch.object(verifier, 'ROOT', self.root), patch('sys.argv', ['verify_outputs.py']), contextlib.redirect_stdout(io.StringIO()):
            verifier.main()
        result = json.loads((self.root / 'outputs/cascade/verification.json').read_text(encoding='utf-8'))
        self.assertEqual(result['users_per_method'], 1)
        self.assertEqual(result['grounded_recommendation_count'], 20)


if __name__ == '__main__':
    unittest.main()

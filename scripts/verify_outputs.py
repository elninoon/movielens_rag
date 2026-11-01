"""核查实际交付结果：完整用户集合、原始模型排序和事实依据。"""
import argparse
import json
import sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from movie_rag.data import DataStore, load_cases, popular_recommendations
from movie_rag.generation import build_messages, validate_response
from movie_rag.grounding import ground_recommendations


def main():
    data = ROOT / 'data' / 'movielens_dev'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cascade')
    args = parser.parse_args()
    experiment = ROOT / 'outputs' / args.name
    config = json.loads((experiment / 'config.json').read_text(encoding='utf-8'))
    store = DataStore(data)
    all_cases = load_cases(data / 'validation_cases.jsonl')
    selected = set(config['user_ids'])
    cases = [case for case in all_cases if case['user_id'] in selected]
    assert len(selected) == len(config['user_ids']) == len(cases), '实验用户集合缺失或重复'
    count = 0
    failures = []
    for method in config['methods']:
        paths = list((experiment / method).glob('*.json'))
        assert {int(path.stem) for path in paths} == {case['user_id'] for case in cases}
        for case in cases:
            result = json.loads((experiment / method / f"{case['user_id']}.json").read_text(encoding='utf-8'))
            if result.get('status', 'ok') != 'ok':
                # 消融中的模型失败保留原始证据并计零，不伪装成合法推荐。
                assert result['recommendations'] == [] and result.get('error')
                assert result.get('attempts') or result.get('calls')
                failures.append({'method': method, 'user_id': case['user_id'], 'error': result['error']})
                continue
            if method == 'popular':
                assert result == popular_recommendations(store, case)
                continue
            # 当前级联及历史对照均以实际送入模型的候选为准，原始100候选仍用于核查来源。
            allowed = set(result.get('allowed_candidate_ids', [row['movie_id'] for row in case['candidates']]))
            generation_case = {**case, 'candidates': [row for row in case['candidates'] if row['movie_id'] in allowed]}
            assert len(generation_case['candidates']) == len(allowed)
            if config.get('pipeline') == '100-50-30-10' or method == 'cascade':
                assert len(allowed) == 30
                stages = result['retrieval']['meta']['stage_candidate_ids']
                assert allowed == set(stages['rerank'])
                assert allowed <= set(stages['coarse']) and len(set(stages['coarse'])) == 50
            messages, sources = build_messages(generation_case, result['retrieval'])
            actual = result['attempts'][0]['messages']
            assert json.loads(actual[1]['content']) == json.loads(messages[1]['content'])
            raw = result['attempts'][-1]['response']['choices'][0]['message']['content']
            original = validate_response(json.loads(raw), generation_case, sources)
            assert [(r['movie_id'], r['rank']) for r in original] == [
                (r['movie_id'], r['rank']) for r in result['recommendations']]
            assert result['recommendations'] == ground_recommendations(generation_case, result['recommendations'])
            for item in result['retrieval']['evidence']:
                mid = item['movie_id']
                assert item['positive_rating_count'] == store.positive_counts[mid]
                assert item['genres'] == store.movies[mid]['genres']
                for neighbor in item['neighbor_ratings']:
                    assert neighbor['user_id'] != case['user_id']
                    assert store.user_ratings[neighbor['user_id']][mid] == neighbor['rating']
            if method == 'no_retrieval':
                assert not result['retrieval']['evidence']
            count += len(result['recommendations'])
    archive_path = ROOT / 'deliverables' / 'MovieLens_RAG_源码.zip'
    if archive_path.exists():
        # 白名单之外再检查实际归档，不显示或输出凭证。
        secret = None
        env = ROOT / '.env'
        if env.exists():
            secret = next((line.split('=', 1)[1].strip().encode() for line in env.read_text(encoding='utf-8').splitlines()
                           if line.startswith('DASHSCOPE_API_KEY=')), None)
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                assert not {'data', 'cache', 'outputs', '.env'} & set(Path(name).parts)
                assert not secret or secret not in archive.read(name)
    result = {'status': 'passed', 'users_per_method': len(cases), 'methods': config['methods'],
        'grounded_recommendation_count': count, 'failed_generation_cases': failures,
        'original_model_rankings_preserved': True,
        'candidate_and_history_sources_verified': True, 'training_neighbor_ratings_verified': True,
        'source_package_excludes_credentials_and_data': archive_path.exists()}
    (experiment / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

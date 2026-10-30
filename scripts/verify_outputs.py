"""核查实际交付结果：完整用户集合、原始模型排序和事实依据。"""
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
    experiment = ROOT / 'outputs' / 'formal'
    store = DataStore(data)
    cases = load_cases(data / 'validation_cases.jsonl')
    count = 0
    failures = []
    for method in ['popular', 'rag', 'no_retrieval']:
        paths = list((experiment / method).glob('*.json'))
        assert {int(path.stem) for path in paths} == {case['user_id'] for case in cases}
        for case in cases:
            result = json.loads((experiment / method / f"{case['user_id']}.json").read_text(encoding='utf-8'))
            if result.get('status', 'ok') != 'ok':
                # 消融中的模型失败保留原始证据并计零，不伪装成合法推荐。
                assert method == 'no_retrieval'
                assert result['recommendations'] == [] and result.get('error')
                assert result.get('attempts') or result.get('calls')
                failures.append({'method': method, 'user_id': case['user_id'], 'error': result['error']})
                continue
            if method == 'popular':
                assert result == popular_recommendations(store, case)
                continue
            messages, sources = build_messages(case, result['retrieval'])
            raw = result['attempts'][-1]['response']['choices'][0]['message']['content']
            original = validate_response(json.loads(raw), case, sources)
            assert [(r['movie_id'], r['rank']) for r in original] == [
                (r['movie_id'], r['rank']) for r in result['recommendations']]
            assert result['recommendations'] == ground_recommendations(case, result['recommendations'])
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
    result = {'status': 'passed', 'users_per_method': len(cases), 'methods': 3,
        'grounded_recommendation_count': count, 'failed_ablation_cases': failures,
        'original_model_rankings_preserved': True,
        'candidate_and_history_sources_verified': True, 'training_neighbor_ratings_verified': True,
        'source_package_excludes_credentials_and_data': archive_path.exists()}
    (ROOT / 'outputs' / 'delivery_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

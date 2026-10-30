"""对已完成实验统一应用事实约束，保持推荐编号和顺序不变。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from movie_rag.grounding import ground_recommendations, POLICY_VERSION
from movie_rag.client import BailianClient
from movie_rag.cli import fingerprint
from movie_rag.generation import validate_response, build_messages


def main():
    experiment = ROOT / 'outputs' / 'formal'
    cases = {row['user_id']: row for row in map(json.loads,
        (ROOT / 'data/movielens_dev/validation_cases.jsonl').read_text(encoding='utf-8').splitlines())}
    count = 0
    for method in ['rag', 'no_retrieval']:
        for path in (experiment / method).glob('*.json'):
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('status') != 'ok':
                continue
            # 新的严格格式校验也必须接受原始成功响应，才能保留其模型排序。
            case = cases[result['user_id']]
            _, sources = build_messages(case, result['retrieval'])
            raw = result['attempts'][-1]['response']['choices'][0]['message']['content']
            validate_response(json.loads(raw), case, sources)
            before = [(r['movie_id'], r['rank']) for r in result['recommendations']]
            result['recommendations'] = ground_recommendations(cases[result['user_id']], result['recommendations'])
            assert before == [(r['movie_id'], r['rank']) for r in result['recommendations']]
            result['reason_policy'] = POLICY_VERSION
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            count += 1
    config_path = experiment / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    client = BailianClient(ROOT)
    config['postprocessing'] = {'policy': POLICY_VERSION, 'ranking_unchanged': True,
        'original_fingerprint': config.get('postprocessing', {}).get('original_fingerprint', config['fingerprint']) if config.get('postprocessing') else config['fingerprint'],
        'description': '保留原始模型响应；严格校验原始排序后，统一从实际历史评分与电影类型渲染理由。'}
    config['reason_policy'] = POLICY_VERSION
    config['fingerprint'] = fingerprint(client, ROOT / 'data/movielens_dev', config['top_k'], config['use_rerank'])
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已核验{count}份结果的推荐理由，推荐编号及顺序保持不变。')


if __name__ == '__main__':
    main()

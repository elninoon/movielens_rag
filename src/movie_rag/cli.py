"""数据生成、真实实验、评测与演示的统一入口。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

from .client import BailianClient
from .data import DataStore, load_cases, popular_recommendations
from .evaluation import compare_methods, select_failure_cases
from .generation import PROMPT_VERSION, recommend, select_generation_case
from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'movielens_dev'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def fingerprint(client, data_dir):
    # 版本与训练/输入绑定，标签不参与推荐配置或缓存身份。
    digest = hashlib.sha256(json.dumps({'models': client.describe(), 'pipeline': '100-50-30-10',
        'prompt': PROMPT_VERSION}, sort_keys=True).encode())
    for name in ['movies.csv', 'train_ratings.csv', 'validation_cases.jsonl']:
        digest.update((data_dir / name).read_bytes())
    for name in ['retrieval.py', 'generation.py', 'grounding.py', 'cli.py']:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def run_experiment(args):
    data_dir = Path(args.data)
    cases = load_cases(data_dir / 'validation_cases.jsonl')
    if args.limit:
        cases = cases[:args.limit]
    if any(len(case['candidates']) != 100 for case in cases):
        raise ValueError('实验输入必须每位用户恰好100部候选')
    store = DataStore(data_dir)
    client = BailianClient(ROOT)
    identity = fingerprint(client, data_dir)
    output = ROOT / 'outputs' / args.name
    config_path = output / 'config.json'
    old_config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    if old_config and (old_config['fingerprint'] != identity or
                       old_config['user_ids'] != [case['user_id'] for case in cases]):
        raise ValueError('同名实验配置或用户集合已改变，请使用新的 --name，避免混用结果')
    if args.retry_errors:
        for method in ['rag', 'no_retrieval']:
            for path in (output / method).glob('*.json'):
                previous = json.loads(path.read_text(encoding='utf-8'))
                calls = previous.get('calls', [])
                if previous.get('status') == 'error' and calls and not calls[-1].get('ok', False):
                    archive = output / 'errors' / method / f'{path.stem}_{time.time_ns()}.json'
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(archive)
    write_json(config_path, {'fingerprint': identity, 'models': client.describe(),
        'pipeline': '100-50-30-10', 'coarse_k': 50, 'top_k': 30,
        'data_dir': str(data_dir.resolve()),
        'user_ids': [case['user_id'] for case in cases], 'prompt_version': PROMPT_VERSION,
        'methods': ['popular', 'rag', 'no_retrieval'], 'generation_workers': args.workers,
        'reason_policy': 'exact-rating-and-shared-genre-v1',
        'ablation_policy': '相同精筛30候选，只移除增强证据'})
    retriever = Retriever(store, client, ROOT / 'cache')
    tasks = []
    for index, case in enumerate(cases, 1):
        uid = case['user_id']
        write_json(output / 'popular' / f'{uid}.json', popular_recommendations(store, case))
        rag_path = output / 'rag' / f'{uid}.json'
        retrieval_path = output / 'retrieval' / f'{uid}.json'
        ablation_path = output / 'no_retrieval' / f'{uid}.json'
        if not rag_path.exists() or not ablation_path.exists():
            if retrieval_path.exists():
                retrieved = json.loads(retrieval_path.read_text(encoding='utf-8'))
            else:
                before = len(client.calls)
                start = time.perf_counter()
                retrieved = retriever.retrieve(case)
                retrieved['calls'] = client.calls[before:]
                retrieved['seconds'] = time.perf_counter() - start
                write_json(retrieval_path, retrieved)
            generation_case = select_generation_case(case, retrieved)
            if not rag_path.exists():
                tasks.append((generation_case, 'rag', retrieved, rag_path))
            if not ablation_path.exists():
                # 消融保留相同精筛候选、完整历史和生成配置，仅删除增强证据。
                ablated = {'evidence': [], 'meta': {'ablation': 'remove_retrieval',
                           'stage_candidate_ids': retrieved['meta']['stage_candidate_ids']}}
                tasks.append((generation_case, 'no_retrieval', ablated, ablation_path))
        print(f'检索准备 {index}/{len(cases)} user={uid}', flush=True)

    def generate(task):
        case, method, retrieved, path = task
        worker = BailianClient(ROOT)
        start = time.perf_counter()
        try:
            result = recommend(worker, case, retrieved, method)
        except Exception as error:
            result = {'user_id': case['user_id'], 'method': method, 'status': 'error',
                      'recommendations': [], 'error': str(error), 'retrieval': retrieved,
                      'allowed_candidate_ids': [row['movie_id'] for row in case['candidates']]}
        result['seconds'] = time.perf_counter() - start
        result['calls'] = worker.calls
        write_json(path, result)
        return method, case['user_id'], result['status']

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            method, uid, status = future.result()
            print(f'生成完成 {done}/{len(tasks)} {method} user={uid} status={status}', flush=True)
    evaluate_output(data_dir, output)


def evaluate_output(data_dir: Path, output: Path):
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    selected = set(config['user_ids'])
    cases = [case for case in load_cases(data_dir / 'validation_cases.jsonl') if case['user_id'] in selected]
    # 标签只在推荐文件全部生成后进入独立评测模块。
    labels = read_jsonl(data_dir / 'validation_labels.jsonl')
    by_method = {}
    for method in config['methods']:
        directory = output / method
        by_method[method] = [json.loads(path.read_text(encoding='utf-8')) for path in directory.glob('*.json')]
    evaluation = compare_methods(cases, labels, by_method)
    write_json(output / 'evaluation.json', evaluation)
    if 'rag' in by_method and 'popular' in by_method:
        failures = select_failure_cases(cases, labels, by_method['rag'], by_method['popular'], n=3)
        write_json(output / 'failure_cases.json', failures)
    print(json.dumps(evaluation['summary'], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='MovieLens 100K / Qwen RAG 课程作业')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='原样执行教师数据准备脚本')
    prepare.add_argument('--zip')
    check = commands.add_parser('check-api', help='小请求验证三个模型')
    baseline = commands.add_parser('baseline', help='不调用API的热门基线')
    baseline.add_argument('--data', default=str(DATA))
    experiment = commands.add_parser('experiment', help='运行热门、RAG和移除检索的消融，自动断点续跑')
    experiment.add_argument('--data', default=str(DATA))
    experiment.add_argument('--name', default='cascade')
    experiment.add_argument('--limit', type=int)
    experiment.add_argument('--workers', type=int, default=3)
    experiment.add_argument('--retry-errors', action='store_true', help='保留失败日志并仅续跑服务失败，不重新抽样模型约束失败')
    evaluate = commands.add_parser('evaluate', help='重新评测已有结果，不调用API')
    evaluate.add_argument('--name', default='cascade')
    evaluate.add_argument('--data', default=str(DATA))
    demo = commands.add_parser('demo', help='展示真实实验中某位用户的推荐与依据，不重复收费')
    demo.add_argument('--name', default='cascade')
    demo.add_argument('--user', type=int, required=True)
    demo.add_argument('--method', default='rag', choices=['rag', 'no_retrieval', 'popular', 'cascade', 'all100'])
    predict = commands.add_parser('recommend', help='对新输入case.json运行推荐，不读取任何标签')
    predict.add_argument('--case', required=True)
    predict.add_argument('--data', default=str(DATA))
    predict.add_argument('--output', default=str(ROOT / 'outputs' / 'prediction.json'))
    args = parser.parse_args()
    if args.command == 'prepare':
        sys.argv = ['prepare_movielens.py', '--mode', 'student', '--output', str(DATA), '--seed', '20261018']
        if args.zip:
            sys.argv += ['--zip', args.zip]
        runpy.run_path(str(ROOT / 'scripts' / 'prepare_movielens.py'), run_name='__main__')
    elif args.command == 'check-api':
        client = BailianClient(ROOT)
        vectors = client.embed(['Movie genre: Drama. Rating: 5.'], 'document')
        ranking = client.rerank('Recommend drama films', ['Drama movie', 'Horror movie'], 2)
        response = client.chat([{'role': 'user', 'content': '仅返回JSON对象：{"ok":true}'}])
        result = {'models': client.describe(), 'embedding_dimension': len(vectors[0]),
                  'rerank_count': len(ranking), 'chat': response['choices'][0]['message']['content'],
                  'calls': client.calls}
        write_json(ROOT / 'outputs' / 'api_check.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'baseline':
        data_dir = Path(args.data)
        cases = load_cases(data_dir / 'validation_cases.jsonl')
        store = DataStore(data_dir)
        from .evaluation import evaluate_results
        results = [popular_recommendations(store, case) for case in cases]
        value = evaluate_results(cases, read_jsonl(data_dir / 'validation_labels.jsonl'), results)
        write_json(ROOT / 'outputs' / 'baseline_evaluation.json', value)
        print(json.dumps(value['summary'], ensure_ascii=False, indent=2))
    elif args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'evaluate':
        evaluate_output(Path(args.data), ROOT / 'outputs' / args.name)
    elif args.command == 'demo':
        path = ROOT / 'outputs' / args.name / args.method / f'{args.user}.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        print(json.dumps({'user_id': args.user, 'status': result.get('status', 'ok'),
                          'recommendations': result['recommendations']}, ensure_ascii=False, indent=2))
    elif args.command == 'recommend':
        case = json.loads(Path(args.case).read_text(encoding='utf-8-sig'))
        if len(case['candidates']) != 100:
            raise ValueError('老师规定每次输入100部候选电影')
        client = BailianClient(ROOT)
        store = DataStore(args.data)
        retrieval = Retriever(store, client, ROOT / 'cache').retrieve(case)
        result = recommend(client, select_generation_case(case, retrieval), retrieval, 'rag')
        result['calls'] = client.calls
        write_json(args.output, result)
        print(json.dumps({'user_id': case['user_id'], 'status': result.get('status', 'ok'),
                          'recommendations': result['recommendations']}, ensure_ascii=False, indent=2))

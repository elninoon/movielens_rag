"""离线汇总级联三方法实验、同候选消融核查与调用记录，不改原始结果。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from movie_rag.data import load_cases
from movie_rag.evaluation import METRICS, compare_methods, select_failure_cases
from analyze_cascade import call_summary, format_delta, read_json, stage_coverage

METHODS = ('popular', 'rag', 'no_retrieval')


def read_rows(directory, recursive=False):
    paths = directory.rglob('*.json') if recursive else directory.glob('*.json')
    return [read_json(path) for path in sorted(paths)]


def execution_status(users, records, evaluated):
    by_uid = {row['user_id']: row for row in records}
    valid = {row['user_id']: row['valid'] for row in evaluated}
    details = []
    for uid in users:
        row = by_uid.get(uid)
        if row is None:
            category = 'missing'
        elif valid[uid]:
            category = 'success'
        elif row.get('calls') and not row['calls'][-1].get('ok', False):
            category = 'service_failure'
        else:
            category = 'content_failure'
        details.append({'user_id': uid, 'category': category, 'error': row.get('error') if row else None})
    counts = Counter(row['category'] for row in details)
    return {**{key: counts[key] for key in ('success', 'service_failure', 'content_failure', 'missing')},
        'per_user': details}


def paired_metrics(evaluation, users):
    by_method = {method: {row['user_id']: row for row in evaluation['methods'][method]['per_user']}
                 for method in ('rag', 'no_retrieval')}
    per_user = [{'user_id': uid, 'delta': {metric: by_method['rag'][uid][metric]
        - by_method['no_retrieval'][uid][metric] for metric in METRICS}} for uid in users]
    summary = {}
    for metric in METRICS:
        values = [row['delta'][metric] for row in per_user]
        summary[metric] = {'mean_delta': sum(values) / len(users),
            'wins': sum(x > 1e-12 for x in values), 'ties': sum(abs(x) <= 1e-12 for x in values),
            'losses': sum(x < -1e-12 for x in values)}
    samples = np.random.default_rng(20261018).integers(0, len(users), (10000, len(users)))
    intervals = {}
    for metric in ('recall_at_10', 'ndcg_at_10'):
        means = np.array([row['delta'][metric] for row in per_user])[samples].mean(axis=1)
        low, high = np.quantile(means, [.025, .975])
        intervals[metric] = {'lower': float(low), 'upper': float(high)}
    return {'direction': 'rag - no_retrieval', 'summary': summary, 'per_user': per_user,
        'bootstrap': {'seed': 20261018, 'iterations': 10000,
            'method': '用户级配对percentile bootstrap 95%', 'intervals': intervals}}


def check_ablation(cases, results):
    indexed = {method: {row['user_id']: row for row in results[method]}
               for method in ('rag', 'no_retrieval')}
    details = []
    for case in cases:
        uid = case['user_id']
        rows = [indexed[method].get(uid, {}) for method in ('rag', 'no_retrieval')]
        if any(not row.get('attempts') for row in rows):
            details.append({'user_id': uid, 'verified': False, 'issue': '缺少至少一组实际提示词'})
            continue
        messages = [row['attempts'][0]['messages'] for row in rows]
        payloads = [json.loads(value[1]['content']) for value in messages]
        ids = [[movie['movie_id'] for movie in payload['candidates']] for payload in payloads]
        expected_history = [{'evidence_id': f"history_{row['movie_id']}", **row} for row in case['history']]
        checks = {'same_system_prompt': messages[0][0] == messages[1][0],
            'same_payload_except_retrieved_evidence':
                {k: v for k, v in payloads[0].items() if k != 'retrieved_evidence'}
                == {k: v for k, v in payloads[1].items() if k != 'retrieved_evidence'},
            'same_30_candidates': ids[0] == ids[1] and len(ids[0]) == len(set(ids[0])) == 30,
            'same_allowed_candidate_ids': all(row.get('allowed_candidate_ids') == ids[0] for row in rows),
            'complete_original_history': all(payload['history'] == expected_history for payload in payloads),
            'original_candidate_subset': set(ids[0]) <= {row['movie_id'] for row in case['candidates']},
            'only_rag_has_evidence': bool(payloads[0].get('retrieved_evidence'))
                and payloads[1].get('retrieved_evidence') == [],
            'outputs_within_30': all(all(rec['movie_id'] in ids[0] for rec in row.get('recommendations', [])) for row in rows)}
        details.append({'user_id': uid, 'verified': all(checks.values()), 'checks': checks})
    return {'verified_count': sum(row['verified'] for row in details), 'user_count': len(details),
        'all_verified': all(row['verified'] for row in details), 'per_user': details}


def failure_material(cases, labels, results, evaluation, retrieved, saved_cases=None):
    # 沿用CLI的真实漏推荐样例选择，不推断用户主观原因。
    failures = saved_cases if saved_cases is not None else select_failure_cases(
        cases, labels, results['rag'], results['popular'], n=3)
    indexed = {method: {row['user_id']: row for row in evaluation['methods'][method]['per_user']}
               for method in METHODS}
    materials = []
    for failure in failures[:3]:
        uid = failure['user_id']
        rag = indexed['rag'][uid]
        missed = set(rag['missed_movie_ids'])
        stages = retrieved.get(uid, {}).get('meta', {}).get('stage_candidate_ids', {})
        coarse, rerank = set(stages.get('coarse', [])), set(stages.get('rerank', []))
        present = 'coarse' in stages and 'rerank' in stages
        materials.append({'user_id': uid, 'relevant_count': rag['relevant_count'],
            'methods': {method: {'valid': indexed[method][uid]['valid'],
                'hit_count': indexed[method][uid]['hit_count'],
                'recall_at_10': indexed[method][uid]['recall_at_10'],
                'ndcg_at_10': indexed[method][uid]['ndcg_at_10']} for method in METHODS},
            'stage_record_present': present,
            'missed_notcoarse': len(missed - coarse) if present else None,
            'missed_aftercoarse_before30': len((missed & coarse) - rerank) if present else None,
            'missed_after30': len(missed & rerank) if present else None})
    return {'selection': '沿用CLI按RAG相对热门基线NDCG差值升序选择的最多3位合法且漏推荐用户。',
        'interpretation': '漏失阶段是相关电影未被保留或最终未被选中的统计位置，不表示用户主观原因；消融非法输出命中按主评测计零。',
        'cases': materials}


def summarize(output):
    config = read_json(output / 'config.json')
    users = config['user_ids']
    if not users or len(users) != len(set(users)):
        raise ValueError('实验用户集合必须非空且不重复')
    data = Path(config.get('data_dir', ROOT / 'data' / 'movielens_dev'))
    all_cases = {row['user_id']: row for row in load_cases(data / 'validation_cases.jsonl')}
    cases = [all_cases[uid] for uid in users]
    if any(len(row['candidates']) != 100 for row in cases):
        raise ValueError('评测必须使用原始100候选')
    labels = [json.loads(line) for line in (data / 'validation_labels.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    results = {method: read_rows(output / method) for method in METHODS}
    evaluation = compare_methods(cases, labels, results)
    if any(row['metric_user_count'] != len(users) for row in evaluation['summary'].values()):
        raise ValueError('标签不完整或非法，不能改变配置用户分母')
    status = {method: execution_status(users, results[method], evaluation['methods'][method]['per_user']) for method in METHODS}
    indexed = {method: {row['user_id']: row for row in evaluation['methods'][method]['per_user']}
               for method in ('rag', 'no_retrieval')}
    common = [uid for uid in users if all(indexed[method][uid]['valid'] for method in indexed)]
    exploratory = {'user_count': len(common), 'user_ids': common,
        'policy': '仅双方均成功子集，受结果合法性筛选影响；用来诊断排序与合法性，不代表固定配置用户总体结果，不替代主指标。',
        'metrics': {method: {metric: sum(indexed[method][uid][metric] for uid in common) / len(common)
            if common else None for metric in METRICS} for method in indexed},
        'paired': paired_metrics(evaluation, common) if common else None}
    complete = all(not (row['service_failure'] or row['missing']) for row in status.values())
    retrieved = {int(path.stem): read_json(path) for path in (output / 'retrieval').glob('*.json')}
    label_index = {row['user_id']: row for row in labels}
    stages = {name: stage_coverage(cases, label_index, retrieved, stage)
              for name, stage in (('coarse50', 'coarse'), ('rerank30', 'rerank'))}
    total_relevant = sum(len(label_index[uid]['relevant_movie_ids']) for uid in users)
    for name, value in stages.items():
        covered = sum(row['covered_relevant_count'] for row in value['per_user'])
        value['totals'] = {'original_relevant_count': total_relevant, 'covered_count': covered,
            'lost_count': total_relevant - covered}
    stages['rerank30']['totals']['lost_since_coarse'] = (stages['coarse50']['totals']['covered_count']
        - stages['rerank30']['totals']['covered_count'])
    provenance = read_json(output / 'provenance.json') if (output / 'provenance.json').exists() else None
    return {'experiment_name': output.name, 'user_count': len(users), 'config': config,
        'provenance': provenance, 'experiment_complete': complete, 'execution_status': status,
        'metrics': evaluation['summary'], 'per_user_metrics': {method: value['per_user'] for method, value in evaluation['methods'].items()},
        'evaluation_matches_saved': evaluation == read_json(output / 'evaluation.json') if (output / 'evaluation.json').exists() else None,
        'issues': {method: value['issues'] for method, value in evaluation['methods'].items()},
        'paired': paired_metrics(evaluation, users), 'exploratory_common_success': exploratory,
        'failure_material': failure_material(cases, labels, results, evaluation, retrieved,
            read_json(output / 'failure_cases.json') if (output / 'failure_cases.json').exists() else None),
        'ablation_input_audit': check_ablation(cases, results),
        'stage_coverage': stages,
        'cost': {'generation': {method: call_summary(results[method]) for method in ('rag', 'no_retrieval')},
            'retrieval': call_summary(list(retrieved.values())),
            'historical_errors': {method: call_summary(read_rows(output / 'errors' / method, True)) for method in METHODS}},
        'notes': [
            '所有方法以配置用户和原始100候选评测，失败及非法结果计零；不覆盖CLI生成的evaluation.json或failure_cases.json。',
            '配对差值为rag减no_retrieval；只删除同30候选输入的增强证据，不能解释为整个检索流程的独立收益。',
            '系统提示词、首次用户输入除增强证据外一致，完整历史与实际allowed30另行核查。',
            '当前结果顶层calls汇总生成成本，不重复计算内嵌retrieval.calls；检索与历史归档分别列出。',
            '复用来源以provenance为准；汇总已保存调用不等于本次新增调用，缓存命中不等于冷启动零成本。',
            'usage字段按原名分列，input/output/total及cache明细可能有包含关系，不相加；缺少usage不能推定免费。',
            '耗时按用户结果记录统计，包含该记录内重试；请求秒数之和不等于并发墙钟时间，不推算人民币费用。',
            'bootstrap仅供当前用户样本参考；服务中断未完成时不可据此判断方法优劣，区间跨0不证明等效。']}


def render(summary):
    lines = ['# 级联三方法实验汇总', '', f"实验：{summary['experiment_name']}；配置用户数：{summary['user_count']}。", '']
    if not summary['experiment_complete']:
        lines += ['> **实验有服务失败或缺失，固定分母指标仅记录当前状态，不宜据此比较算法优劣。**', '']
    lines += ['| 方法 | 成功 | 服务失败 | 内容失败 | 缺失 | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | MRR@10 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for method in METHODS:
        status = summary['execution_status'][method]
        lines.append(f'| {method} | ' + ' | '.join(str(status[key]) for key in ('success', 'service_failure', 'content_failure', 'missing'))
            + ' | ' + ' | '.join(f"{summary['metrics'][method][metric]:.6f}" for metric in METRICS) + ' |')
    audit = summary['ablation_input_audit']
    lines += ['', f"同候选消融输入核查：{audit['verified_count']}/{audit['user_count']}；保存的CLI评测一致性：{summary['evaluation_matches_saved']}。", '',
        '## 同候选消融配对差异', '', '方向为rag减no_retrieval；bootstrap种子20261018、10000次、用户级配对percentile 95%区间。', '',
        '| 指标 | 平均差值 | 胜 | 平 | 负 | 95%区间 |', '| --- | --- | --- | --- | --- | --- |']
    for metric, row in summary['paired']['summary'].items():
        ci = summary['paired']['bootstrap']['intervals'].get(metric)
        bounds = f"[{format_delta(ci['lower'])}, {format_delta(ci['upper'])}]" if ci else '—'
        lines.append(f"| {metric} | {format_delta(row['mean_delta'])} | {row['wins']} | {row['ties']} | {row['losses']} | {bounds} |")
    exploratory = summary['exploratory_common_success']
    if 0 < exploratory['user_count'] < summary['user_count']:
        lines += ['', '## 双方均成功子集（仅诊断）', '', exploratory['policy'], '',
            f"用户数：{exploratory['user_count']}；编号：{exploratory['user_ids']}。", '',
            '| 方法 | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | MRR@10 |', '| --- | --- | --- | --- | --- | --- |']
        for method, values in exploratory['metrics'].items():
            lines.append(f'| {method} | ' + ' | '.join(f'{values[metric]:.6f}' for metric in METRICS) + ' |')
    materials = summary['failure_material']
    lines += ['', '## 真实漏推荐样例统计', '', materials['selection'], '', materials['interpretation'], '',
        '| 用户 | 相关数 | 热门命中 | RAG命中 | 消融命中 | 未进50 | 进50未进30 | 进30未选10 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in materials['cases']:
        values = [row['user_id'], row['relevant_count'],
            *[row['methods'][method]['hit_count'] for method in METHODS],
            row['missed_notcoarse'], row['missed_aftercoarse_before30'], row['missed_after30']]
        lines.append('| ' + ' | '.join(str(value) if value is not None else '缺检索记录' for value in values) + ' |')
    lines += ['', '| 用户 | 方法 | Recall@10 | NDCG@10 | 合法 |', '| --- | --- | --- | --- | --- |']
    for row in materials['cases']:
        for method, values in row['methods'].items():
            lines.append(f"| {row['user_id']} | {method} | {values['recall_at_10']:.6f} | {values['ndcg_at_10']:.6f} | {values['valid']} |")
    lines += ['', '## 筛选覆盖', '', '| 阶段 | 合法用户数 | 宏平均相关覆盖 | 保留相关数 | 累计丢失相关数 | Recall上限 | NDCG上限 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for stage, value in summary['stage_coverage'].items():
        row, total = value['summary'], value['totals']
        lines.append(f"| {stage} | {row['valid_stage_user_count']} | {row['relevant_coverage']:.6f} | {total['covered_count']} | {total['lost_count']} | {row['recall_at_10_ceiling']:.6f} | {row['ndcg_at_10_ceiling']:.6f} |")
    lines += ['', '## 调用用量与耗时', '', '| 分类 | 记录数 | 记录耗时合计秒 | 平均秒 | 中位秒 |', '| --- | --- | --- | --- | --- |']
    costs = [(f'生成/{method}', value) for method, value in summary['cost']['generation'].items()]
    costs += [('检索（可能复用）', summary['cost']['retrieval'])]
    costs += [(f'历史归档/{method}', value) for method, value in summary['cost']['historical_errors'].items()]
    usage_lines = []
    for label, value in costs:
        mean = f"{value['record_seconds_mean']:.3f}" if value['record_seconds_mean'] is not None else '—'
        mid = f"{value['record_seconds_median']:.3f}" if value['record_seconds_median'] is not None else '—'
        lines.append(f"| {label} | {value['record_count']} | {value['record_seconds_sum']:.3f} | {mean} | {mid} |")
        for group in value['by_kind_and_model']:
            usage_lines += [f"### {label}：{group['kind']} / {group['model']}", '',
                f"调用{group['call_count']}次，失败{group['failed_call_count']}次，含usage {group['calls_with_usage']}次；请求累计{group['request_seconds_sum']:.3f}秒。", '',
                '| usage原始字段 | 累计值 |', '| --- | --- |',
                *[f'| {field} | {count} |' for field, count in group['usage_fields'].items()], '']
    lines += ['', *usage_lines, '## 复用来源', '',
        '无provenance文件，不能从用量汇总推断哪些调用在本轮新增。' if summary['provenance'] is None
        else '下列为本机记录的复用来源，不等于所有已保存调用都在本轮发生：\n\n```json\n' + json.dumps(summary['provenance'], ensure_ascii=False, indent=2) + '\n```',
        '', '## 统计口径', '', *[f'- {note}' for note in summary['notes']], '',
        '详细逐用户指标、配对差值及输入审查见同目录summary.json。本文件不包含逐条历史或原始提示词。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cascade')
    args = parser.parse_args()
    output = ROOT / 'outputs' / args.name
    summary = summarize(output)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'summary.md').write_text(render(summary), encoding='utf-8')
    print(json.dumps({'experiment_complete': summary['experiment_complete'], 'metrics': summary['metrics'],
        'ablation_input_verified': summary['ablation_input_audit']['all_verified'],
        'summary': str(output / 'summary.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

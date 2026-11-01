"""离线比较全候选与级联推荐，统一使用原始100候选和固定30位用户评测。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from movie_rag.data import load_cases
from movie_rag.evaluation import METRICS, compare_methods


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_results(directory):
    return [read_json(path) for path in sorted(directory.glob('*.json'))]


def usage_leaves(value, prefix=''):
    # 明细字段单独保留，不能与父级总量相加，否则会重复计算token。
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            yield from usage_leaves(item, name)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            yield name, item


def call_summary(rows):
    groups = {}
    for row in rows:
        for call in row.get('calls', []):
            key = (call.get('kind', 'unknown'), call.get('model', 'unknown'))
            group = groups.setdefault(key, {'kind': key[0], 'model': key[1],
                'call_count': 0, 'failed_call_count': 0, 'calls_with_usage': 0,
                'request_seconds_sum': 0.0, 'usage_fields': Counter()})
            group['call_count'] += 1
            group['failed_call_count'] += int(not call.get('ok', False))
            group['request_seconds_sum'] += call.get('seconds', 0)
            if call.get('usage'):
                group['calls_with_usage'] += 1
                group['usage_fields'].update(dict(usage_leaves(call['usage'])))
    return {'record_count': len(rows),
        'record_seconds_sum': sum(row.get('seconds', 0) for row in rows),
        'record_seconds_mean': sum(row.get('seconds', 0) for row in rows) / len(rows) if rows else None,
        'record_seconds_median': median(row.get('seconds', 0) for row in rows) if rows else None,
        'by_kind_and_model': list(groups.values())}


def stage_coverage(cases, labels, retrieval, stage):
    rows = []
    expected_count = 50 if stage == 'coarse' else 30
    for case in cases:
        uid = case['user_id']
        record = retrieval.get(uid)
        ids = record.get('meta', {}).get('stage_candidate_ids', {}).get(stage, []) if record else []
        selected = set(ids)
        relevant = set(labels[uid]['relevant_movie_ids'])
        hit_count = len(selected & relevant)
        best_hits = min(10, hit_count)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
        best_dcg = sum(1 / math.log2(i + 2) for i in range(best_hits))
        candidates = {movie['movie_id'] for movie in case['candidates']}
        rows.append({'user_id': uid, 'record_present': record is not None,
            'candidate_count': len(ids), 'distinct_candidate_count': len(selected),
            'stage_valid': len(ids) == len(selected) == expected_count and selected <= candidates,
            'relevant_count': len(relevant), 'covered_relevant_count': hit_count,
            'relevant_coverage': hit_count / len(relevant),
            'recall_at_10_ceiling': best_hits / len(relevant),
            'ndcg_at_10_ceiling': best_dcg / ideal})
    fields = ('covered_relevant_count', 'relevant_coverage', 'recall_at_10_ceiling', 'ndcg_at_10_ceiling')
    return {'summary': {'user_count': len(rows),
        'missing_record_count': sum(not row['record_present'] for row in rows),
        'valid_stage_user_count': sum(row['stage_valid'] for row in rows),
        **{field: sum(row[field] for row in rows) / len(rows) for field in fields}},
        'per_user': rows}


def analyze(output):
    config = read_json(output / 'config.json')
    data = ROOT / 'data' / 'movielens_dev'
    cases = load_cases(data / 'validation_cases.jsonl')
    users = config['user_ids']
    if len(users) != 30 or len(set(users)) != 30 or set(users) != {case['user_id'] for case in cases}:
        raise ValueError('本比较要求配置与原始数据使用同一组完整30位验证用户')
    labels = [json.loads(line) for line in (data / 'validation_labels.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    label_index = {row['user_id']: row for row in labels}
    if any(not label_index.get(uid, {}).get('relevant_movie_ids') for uid in users):
        raise ValueError('每位用户必须有非空相关标签，保证固定30人指标分母')
    results = {method: read_results(output / method) for method in ('all100', 'cascade')}
    retrieval_rows = {method: read_results(output / f'retrieval_{method}') for method in results}
    # 检索文件没有顶层user_id，文件名即用户编号；不从生成结果嵌套检索重复统计成本。
    retrieval = {method: {int(path.stem): read_json(path)
        for path in (output / f'retrieval_{method}').glob('*.json')} for method in results}
    evaluation = compare_methods(cases, labels, results)
    if any(summary['metric_user_count'] != 30 for summary in evaluation['summary'].values()):
        raise ValueError('标签非法导致评测分母变化，请检查标签')
    indexed = {method: {row['user_id']: row for row in evaluation['methods'][method]['per_user']} for method in results}
    execution = {}
    for method in results:
        by_uid = {row['user_id']: row for row in results[method]}
        status = {'success_count': 0, 'http_service_failure_count': 0,
            'other_service_failure_count': 0, 'model_validation_failure_count': 0,
            'other_failure_count': 0, 'missing_count': 0, 'per_user': []}
        for uid in users:
            result = by_uid.get(uid)
            if result is None:
                category = 'missing'
            elif indexed[method][uid]['valid']:
                category = 'success'
            elif any(not call.get('ok', False) and call.get('http_status', 0) >= 400
                     for call in result.get('calls', [])) or 'HTTP ' in result.get('error', ''):
                category = 'http_service_failure'
            elif any(not call.get('ok', False) for call in result.get('calls', [])):
                category = 'other_service_failure'
            elif any('validation_error' in attempt for attempt in result.get('attempts', [])):
                category = 'model_validation_failure'
            else:
                category = 'other_failure'
            status[f'{category}_count'] += 1
            status['per_user'].append({'user_id': uid, 'category': category,
                'error': result.get('error') if result else None})
        status['retrieval_record_count'] = len(set(retrieval[method]) & set(users))
        execution[method] = status
    experiment_complete = all(not any(row[key] for key in
        ('http_service_failure_count', 'other_service_failure_count', 'other_failure_count', 'missing_count'))
        and row['retrieval_record_count'] == 30 for row in execution.values())
    common_users = [uid for uid in users if all(indexed[method][uid]['valid'] for method in results)]
    exploratory = {'user_count': len(common_users), 'user_ids': common_users,
        'policy': '仅双方均成功用户的探索性配对；受服务中断及成功样本选择影响，不代表完整30用户结果，不能替代主指标。',
        'methods': {method: {metric: sum(indexed[method][uid][metric] for uid in common_users) / len(common_users)
            if common_users else None for metric in METRICS} for method in results},
        'paired': {metric: {'mean_delta': sum(indexed['cascade'][uid][metric] - indexed['all100'][uid][metric]
            for uid in common_users) / len(common_users) if common_users else None,
            'wins': sum(indexed['cascade'][uid][metric] - indexed['all100'][uid][metric] > 1e-12 for uid in common_users),
            'ties': sum(abs(indexed['cascade'][uid][metric] - indexed['all100'][uid][metric]) <= 1e-12 for uid in common_users),
            'losses': sum(indexed['cascade'][uid][metric] - indexed['all100'][uid][metric] < -1e-12 for uid in common_users)} for metric in METRICS}}
    paired = []
    for uid in users:
        paired.append({'user_id': uid,
            'all100_valid': indexed['all100'][uid]['valid'],
            'cascade_valid': indexed['cascade'][uid]['valid'],
            'delta': {metric: indexed['cascade'][uid][metric] - indexed['all100'][uid][metric] for metric in METRICS}})
    paired_summary = {}
    for metric in METRICS:
        values = [row['delta'][metric] for row in paired]
        paired_summary[metric] = {'mean_delta': sum(values) / 30,
            'wins': sum(value > 1e-12 for value in values),
            'ties': sum(abs(value) <= 1e-12 for value in values),
            'losses': sum(value < -1e-12 for value in values)}
    coarse = stage_coverage(cases, label_index, retrieval['cascade'], 'coarse')
    rerank = stage_coverage(cases, label_index, retrieval['cascade'], 'rerank')
    cascade_results = {row['user_id']: row for row in results['cascade']}
    strict = []
    for uid in users:
        record = retrieval['cascade'].get(uid, {})
        stages = record.get('meta', {}).get('stage_candidate_ids', {})
        allowed = set(stages.get('rerank', []))
        result = cascade_results.get(uid, {})
        rec_ids = [row['movie_id'] for row in result.get('recommendations', [])]
        outside = sorted(set(rec_ids) - allowed)
        evidence_ids = {row['movie_id'] for row in record.get('evidence', [])}
        strict.append({'user_id': uid, 'outside_rerank_movie_ids': outside,
            'rerank_subset_of_coarse': allowed <= set(stages.get('coarse', [])),
            'evidence_matches_rerank': evidence_ids == allowed and len(allowed) == 30,
            'strict_valid': indexed['cascade'][uid]['valid']
                and len(stages.get('rerank', [])) == len(allowed) == 30 and not outside})
    comparison = {'user_count': 30, 'methods': evaluation['summary'],
        'experiment_complete': experiment_complete, 'execution_status': execution,
        'exploratory_common_success': exploratory,
        'delta_direction': 'cascade - all100',
        'paired': {'summary': paired_summary, 'per_user': paired},
        'cascade_stage_coverage': {'coarse50': coarse, 'retrieval30': rerank},
        'cascade_candidate_check': {'strict_valid_user_count': sum(row['strict_valid'] for row in strict),
            'all_strict_valid': all(row['strict_valid'] for row in strict), 'per_user': strict},
        'cost': {method: {'generation': call_summary(results[method]),
            'retrieval': call_summary(retrieval_rows[method])} for method in results},
        'shared_embedding_cost': call_summary(read_results(output / 'shared_embedding_batches')),
        'service_check_cost': call_summary(read_results(output / 'service_checks')),
        'service_error_archive_cost': {method: call_summary(read_results(output / 'service_errors' / method))
            for method in results},
        'notes': [
            '统一以原始100候选、完整30位用户评测；缺失、失败或非法结果计零。',
            '阶段相关覆盖为各用户保留下来的相关电影比例的宏平均；缺失检索按空集合计，单列缺失数。',
            '理论上限假定在该阶段候选中理想选择并排序前10；用原始相关集作分母。',
            'all100的30条证据不限制最终候选，故不能把其证据覆盖当作all100推荐指标上限。',
            '仅汇总各文件顶层calls，避免重复计算生成结果内嵌的检索记录。',
            'token字段分别累计，total_tokens与明细不能相加；缓存token通常是输入token的子集。',
            'retrieval可能复用向量缓存或历史检索文件；记录只反映已保存调用，不代表公平冷启动成本。',
            '共享向量预计算的全部批次调用单列shared_embedding_cost，不分摊给任一方法；两组检索耗时不含共享向量构建。',
            '共享预计算也可能命中此前缓存；零Embedding调用不表示从零构建向量不需要成本。',
            'service_errors归档的历史服务失败调用另列成本，当前执行状态仅依据当前方法结果文件；归档不存在时记录数为0。',
            'service_checks中的服务恢复预检调用单列service_check_cost，不计入任一方法的推荐调用或成功数；目录不存在时为0。',
            '本实验同时改变粗筛、精筛打分和生成候选限制：all100使用融合分选证据，cascade粗筛后仅按rerank分选30候选；差异不能归因于单个因素。',
            '请求耗时之和含失败调用，不等于并发实验墙钟耗时；未返回usage的调用不能视为零计费。',
            '没有价格和账单依据，不估算人民币费用。']}
    # 用户级配对重采样保留同一用户两种方法的对应关系。
    rng = np.random.default_rng(20261018)
    samples = rng.integers(0, 30, size=(10000, 30))
    intervals = {}
    for metric in ('recall_at_10', 'ndcg_at_10'):
        delta = np.array([row['delta'][metric] for row in paired])
        bounds = np.quantile(delta[samples].mean(axis=1), [.025, .975])
        intervals[metric] = {'mean_delta': float(delta.mean()), 'lower': float(bounds[0]), 'upper': float(bounds[1])}
    comparison['paired']['bootstrap'] = {'seed': 20261018, 'iterations': 10000,
        'method': '用户级配对percentile bootstrap 95%区间；服务失败计零的未完整实验不作方法优劣推断。',
        'metrics': intervals}
    efficiency = {}
    for method in results:
        records = results[method]
        totals = Counter({key: 0 for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        for row in records:
            for call in row.get('calls', []):
                totals.update({key: call.get('usage', {}).get(key, 0)
                               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        timing = comparison['cost'][method]['generation']
        efficiency[method] = {'record_count': len(records),
            'seconds_mean': timing['record_seconds_mean'], 'seconds_median': timing['record_seconds_median'],
            **{f'{key}_mean': value / len(records) if records else None for key, value in totals.items()}}
    efficiency['cascade_reduction'] = {field: 1 - efficiency['cascade'][field] / efficiency['all100'][field]
        if efficiency['all100'][field] else None for field in ('prompt_tokens_mean', 'total_tokens_mean', 'seconds_mean')}
    efficiency['policy'] = '按当前用户结果逐人均值，包含同用户内容校验重试；历史服务失败和恢复预检另列。仅完整实验用于效率比较，调用时段及并发状态未严格控制。'
    comparison['generation_efficiency'] = efficiency
    return evaluation, comparison


def format_delta(value):
    return f'{0.0 if abs(value) < 0.0000005 else value:+.6f}'


def render_markdown(comparison):
    lines = ['# 全候选与级联推荐离线比较', '']
    if not comparison['experiment_complete']:
        lines += ['> **实验尚未完成：生成服务中断或结果缺失。目前固定30用户总指标包含服务失败计零，不宜据此判断两种方法优劣。阶段覆盖可独立核查；双方均成功子集仅供探索。**', '']
    else:
        lines += ['实验已完成；模型内容或格式约束失败保留计零。', '']
    lines += ['| 方法 | 成功 | HTTP服务失败 | 其他服务失败 | 模型校验失败 | 其他失败 | 缺失 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for method, row in comparison['execution_status'].items():
        lines.append(f"| {method} | {row['success_count']} | {row['http_service_failure_count']} | {row['other_service_failure_count']} | {row['model_validation_failure_count']} | {row['other_failure_count']} | {row['missing_count']} |")
    lines += ['', '同一组30位用户，使用原始100候选与同一标签；差值方向为 cascade − all100。', '',
        '| 方法 | 合法用户 | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | MRR@10 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for method, row in comparison['methods'].items():
        lines.append(f"| {method} | {row['valid_user_count']}/30 | " + ' | '.join(f'{row[field]:.6f}' for field in METRICS) + ' |')
    lines += ['', '## 配对差异', '', '| 指标 | 宏平均差值 | 胜 | 平 | 负 |', '| --- | --- | --- | --- | --- |']
    for metric, row in comparison['paired']['summary'].items():
        lines.append(f"| {metric} | {format_delta(row['mean_delta'])} | {row['wins']} | {row['ties']} | {row['losses']} |")
    bootstrap = comparison['paired']['bootstrap']
    lines += ['', '用户级配对 bootstrap（固定种子20261018，10000次）95%区间：', '',
        '| 指标 | 差值 | 下界 | 上界 |', '| --- | --- | --- | --- |']
    for metric, row in bootstrap['metrics'].items():
        lines.append(f"| {metric} | {format_delta(row['mean_delta'])} | {format_delta(row['lower'])} | {format_delta(row['upper'])} |")
    lines += ['', '区间跨0时，当前样本不足以支持两种方法存在明确差异；这也不能证明两种方法等效。服务中断时区间仅反映失败计零结果。']
    exploratory = comparison['exploratory_common_success']
    if 0 < exploratory['user_count'] < 30:
        lines += ['', '## 双方均成功子集（仅探索）', '', exploratory['policy'], '',
        f"用户数：{exploratory['user_count']}；用户编号：{exploratory['user_ids']}。", '',
        '| 方法 | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | MRR@10 |', '| --- | --- | --- | --- | --- | --- |']
        for method, row in exploratory['methods'].items():
            lines.append(f'| {method} | ' + ' | '.join(f'{row[field]:.6f}' if row[field] is not None else '无样本' for field in METRICS) + ' |')
        lines += ['', '| 指标 | 探索性配对差值 | 胜 | 平 | 负 |', '| --- | --- | --- | --- | --- |']
        for metric, row in exploratory['paired'].items():
            value = format_delta(row['mean_delta']) if row['mean_delta'] is not None else '无样本'
            lines.append(f"| {metric} | {value} | {row['wins']} | {row['ties']} | {row['losses']} |")
    lines += ['', '## 阶段覆盖与理论上限', '', '| 阶段 | 完整合法阶段用户 | 缺失记录 | 宏平均相关覆盖 | Recall@10上限 | NDCG@10上限 |', '| --- | --- | --- | --- | --- | --- |']
    for stage, value in comparison['cascade_stage_coverage'].items():
        row = value['summary']
        lines.append(f"| {stage} | {row['valid_stage_user_count']}/30 | {row['missing_record_count']} | {row['relevant_coverage']:.6f} | {row['recall_at_10_ceiling']:.6f} | {row['ndcg_at_10_ceiling']:.6f} |")
    check = comparison['cascade_candidate_check']
    lines += ['', f"级联最终输出合法且全部来自30候选的用户：**{check['strict_valid_user_count']}/30**。", '', '## 生成效率', '',
        '| 方法 | 每用户输入token均值 | 每用户总token均值 | 每用户生成秒数均值 | 秒数中位数 |', '| --- | --- | --- | --- | --- |']
    efficiency = comparison['generation_efficiency']
    for method in ('all100', 'cascade'):
        row = efficiency[method]
        lines.append(f"| {method} | {row['prompt_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} | {row['seconds_mean']:.3f} | {row['seconds_median']:.3f} |")
    reduction = efficiency['cascade_reduction']
    reductions = '；'.join(f"{label}：{reduction[field]:.2%}" if reduction[field] is not None else f'{label}：无数据'
        for label, field in [('输入token降低', 'prompt_tokens_mean'), ('总token降低', 'total_tokens_mean'), ('平均生成秒数降低', 'seconds_mean')])
    lines += ['', reductions + '。', '', efficiency['policy'], '', '## 调用记录与成本依据', '',
        '| 方法 | 阶段 | 类型 / 模型 | 调用数 | 失败数 | 含usage调用数 | 请求秒数之和 |', '| --- | --- | --- | --- | --- | --- | --- |']
    usage = []
    summaries = [(method, stage, value) for method, stages in comparison['cost'].items()
                 for stage, value in stages.items()]
    summaries.append(('共享', 'embedding预计算', comparison['shared_embedding_cost']))
    summaries.append(('共享', '服务恢复预检', comparison['service_check_cost']))
    summaries += [(method, '历史服务失败归档', value)
                  for method, value in comparison['service_error_archive_cost'].items()]
    for method, stage, value in summaries:
        for group in value['by_kind_and_model']:
            label = f"{group['kind']} / {group['model']}"
            lines.append(f"| {method} | {stage} | {label} | {group['call_count']} | {group['failed_call_count']} | {group['calls_with_usage']} | {group['request_seconds_sum']:.3f} |")
            for field, count in group['usage_fields'].items():
                usage.append(f'| {method} | {stage} | {label} | {field} | {count} |')
    shared = comparison['shared_embedding_cost']
    lines += ['', f"共享向量预计算记录批次：{shared['record_count']}；批次耗时之和：{shared['record_seconds_sum']:.3f}秒。该部分单列，不计入all100或cascade。"]
    lines += ['', f"服务恢复预检记录数：{comparison['service_check_cost']['record_count']}；预检调用另列，不参与两组推荐指标。"]
    lines += ['', '各usage字段按原名分别统计，字段之间可能包含关系：', '', '| 方法 | 阶段 | 类型 / 模型 | usage字段 | 累计值 |', '| --- | --- | --- | --- | --- |', *usage,
        '', '## 统计口径', '', *[f'- {note}' for note in comparison['notes']], '',
        '逐用户配对差异、阶段覆盖和候选检查见同目录 `comparison.json`，完整评测见 `evaluation.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cascade_comparison')
    args = parser.parse_args()
    output = ROOT / 'outputs' / args.name
    evaluation, comparison = analyze(output)
    for name, value in [('evaluation.json', evaluation), ('comparison.json', comparison)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'comparison.md').write_text(render_markdown(comparison), encoding='utf-8')
    print(json.dumps({'methods': comparison['methods'], 'paired': comparison['paired']['summary'],
        'strict_valid_user_count': comparison['cascade_candidate_check']['strict_valid_user_count'],
        'report': str(output / 'comparison.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

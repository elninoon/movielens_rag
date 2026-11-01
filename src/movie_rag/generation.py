"""从同一候选集生成推荐，保留提示词与可核查的依据引用。"""
from __future__ import annotations

import json
from .grounding import ground_recommendations

PROMPT_VERSION = 'movielens-cascade-1'
SYSTEM_PROMPT = '''你是电影推荐系统。仅依据输入数据推断偏好，评分>=4表示喜欢，<=2表示不喜欢。
必须从给定{candidate_count}部候选中推荐10部不同电影，按适合用户的程度排序，不能推荐历史已看电影。
电影类型、上映信息、用户训练历史及检索证据是唯一事实来源。禁止使用记忆中的剧情、演员、影评或编造事实。
优先考虑个性化匹配，同时参考证据中的训练评分统计及近邻支持，但不要把预测分数描述成真实评分。
仅返回JSON对象 {"recommendations":[{"movie_id":整数,"rank":1,"reason":"简短中文理由","evidence_ids":["candidate_编号","history_编号","movie_编号"]}]}。
每条理由必须引用该候选的candidate_编号。引用历史时必须使用存在的history_编号。
存在对应检索证据且理由使用其信息时引用movie_编号。引用ID必须实际出现在输入中。
每条理由最多80个汉字，准确描述支持推荐的事实，不把单次评分写成多次，不把类型偏好写成必然喜欢。'''


def build_messages(case: dict, retrieval: dict) -> tuple[list[dict], dict]:
    # 全量训练历史都提供给模型，不按验证标签筛选或压缩。
    history = [{'evidence_id': f"history_{h['movie_id']}", **h} for h in case['history']]
    candidates = [{'evidence_id': f"candidate_{c['movie_id']}", **c} for c in case['candidates']]
    evidence = retrieval.get('evidence', [])
    payload = {'user_id': case['user_id'], 'history': history, 'candidates': candidates,
               'retrieved_evidence': evidence}
    sources = {row['evidence_id']: row for row in history + candidates + evidence}
    # 候选数与实际输入一致，在线推荐入口只接受精筛后的30部。
    system_prompt = SYSTEM_PROMPT.replace('{candidate_count}', str(len(candidates)))
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    return messages, sources


def validate_response(payload: dict, case: dict, sources: dict) -> list[dict]:
    rows = payload.get('recommendations')
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError('必须恰好返回10条推荐')
    candidate_ids = {row['movie_id'] for row in case['candidates']}
    history_ids = {row['movie_id'] for row in case['history']}
    seen = set()
    output = []
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError('每条推荐必须是JSON对象')
        mid = row.get('movie_id')
        if type(mid) is not int or mid not in candidate_ids or mid in history_ids or mid in seen:
            raise ValueError(f'第{rank}条电影编号非法或重复')
        if type(row.get('rank')) is not int or row['rank'] != rank:
            raise ValueError('排名须为连续1到10')
        reason = row.get('reason')
        refs = row.get('evidence_ids')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('推荐理由为空')
        if not isinstance(refs, list) or f'candidate_{mid}' not in refs:
            raise ValueError(f'电影{mid}缺少自身候选依据')
        if any(not isinstance(ref, str) or ref not in sources for ref in refs):
            raise ValueError('引用了不存在的证据编号')
        seen.add(mid)
        output.append({'movie_id': mid, 'rank': rank, 'reason': reason.strip(),
                       'evidence': [sources[ref] for ref in dict.fromkeys(refs)]})
    return output


def select_generation_case(case: dict, retrieval: dict) -> dict:
    """按精筛证据限制生成候选，完整保留历史与原始候选顺序。"""
    ids = [row['movie_id'] for row in retrieval['evidence']]
    allowed = set(ids)
    candidates = [row for row in case['candidates'] if row['movie_id'] in allowed]
    seen = {row['movie_id'] for row in case['history']}
    if len(ids) != 30 or len(allowed) != 30 or len(candidates) != 30 or allowed & seen:
        raise ValueError('精筛必须返回30部不同且未看的原始候选电影')
    return {**case, 'candidates': candidates}


def recommend(client, case: dict, retrieval: dict, method: str) -> dict:
    if len(case["candidates"]) != 30:
        raise ValueError("级联生成阶段必须恰好提供30部候选电影")
    messages, sources = build_messages(case, retrieval)
    attempts = []
    for attempt in range(2):
        response = client.chat(messages)
        content = response['choices'][0]['message']['content']
        attempts.append({'response': response, 'messages': list(messages)})
        try:
            rows = validate_response(json.loads(content), case, sources)
            return {'user_id': case['user_id'], 'method': method, 'status': 'ok',
                    'recommendations': ground_recommendations(case, rows),
                    'reason_policy': 'exact-rating-and-shared-genre-v1',
                    'allowed_candidate_ids': [row['movie_id'] for row in case['candidates']],
                    'retrieval': retrieval, 'attempts': attempts}
        except (ValueError, TypeError, KeyError) as error:
            attempts[-1]['validation_error'] = str(error)
            if attempt == 1:
                return {'user_id': case['user_id'], 'method': method, 'status': 'error',
                        'recommendations': [], 'error': str(error), 'retrieval': retrieval,
                        'allowed_candidate_ids': [row['movie_id'] for row in case['candidates']],
                        'attempts': attempts}
            messages = messages + [{'role': 'assistant', 'content': content},
                {'role': 'user', 'content': f'输出校验失败：{error}。请依据原始输入重新输出完整JSON。'}]
    raise RuntimeError('生成未结束')

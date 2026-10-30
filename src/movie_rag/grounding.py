"""将模型排序转成基于真实字段的推荐理由，原始理由留作审计。"""
from __future__ import annotations

POLICY_VERSION = 'exact-rating-and-shared-genre-v1'


def ground_recommendations(case: dict, rows: list[dict]) -> list[dict]:
    candidates = {movie['movie_id']: movie for movie in case['candidates']}
    output = []
    for row in rows:
        mid = row['movie_id']
        movie = candidates[mid]
        genres = set(movie['genres'])
        cited = {item.get('evidence_id') for item in row['evidence']}
        # 优先使用模型选择的有效高分依据；没有时从输入历史中寻找可核查的类型匹配。
        supporting = [h for h in case['history'] if h['rating'] >= 4 and genres & set(h['genres'])]
        supporting.sort(key=lambda h: (f"history_{h['movie_id']}" not in cited,
            -len(genres & set(h['genres'])), -h['rating'], -h.get('timestamp', 0), h['movie_id']))
        evidence = [{'evidence_id': f'candidate_{mid}', **movie}]
        if supporting:
            history = supporting[0]
            common = '、'.join(sorted(genres & set(history['genres'])))
            reason = f"你给《{history['title']}》评为{history['rating']}分；本片同属{common}，据此作为类型匹配推荐。"
            evidence.append({'evidence_id': f"history_{history['movie_id']}", **history})
        else:
            reason = f"本片类型为{'、'.join(movie['genres']) or 'unknown'}；历史中缺少同类型高分依据，作为探索项推荐。"
        retrieved = next((item for item in row['evidence'] if item.get('evidence_id') == f'movie_{mid}'), None)
        if retrieved is not None:
            count = retrieved['positive_rating_count']
            reason += f'训练集中有{count}次不低于4分的评分。'
            evidence.append(retrieved)
        output.append({**row, 'model_reason': row.get('model_reason', row['reason']),
                       'reason': reason, 'evidence': evidence, 'reason_policy': POLICY_VERSION})
    return output

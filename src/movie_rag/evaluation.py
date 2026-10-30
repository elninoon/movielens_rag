"""独立离线评测；只有本模块读取调用方提供的验证标签。"""

from __future__ import annotations

from collections import Counter
import math


METRICS = ("recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10", "mrr_at_10")


def _index(rows, name):
    index, issues = {}, []
    for row in rows:
        uid = row["user_id"]
        if uid in index:
            issues.append({"user_id": uid, "code": f"duplicate_{name}"})
        index[uid] = row
    return index, issues


def _validate(case, result):
    """非法或失败结果整条计零，避免部分合法结果取得不公平优势。"""
    problems = []
    if result is None:
        return ["missing_result"]
    if result.get("status", "ok") != "ok":
        problems.append("run_error")
    recs = result.get("recommendations")
    if not isinstance(recs, list):
        return problems + ["recommendations_not_list"]
    if len(recs) != 10:
        problems.append("recommendation_count_not_10")
    candidates = {movie["movie_id"] for movie in case["candidates"]}
    history = {movie["movie_id"] for movie in case["history"]}
    ids, ranks = [], []
    for rec in recs:
        if not isinstance(rec, dict):
            problems.append("recommendation_not_object")
            continue
        mid, rank = rec.get("movie_id"), rec.get("rank")
        if type(mid) is not int:
            problems.append("movie_id_not_integer")
        else:
            ids.append(mid)
            if mid not in candidates:
                problems.append("movie_outside_candidates")
            if mid in history:
                problems.append("movie_in_history")
        if type(rank) is not int:
            problems.append("rank_not_integer")
        else:
            ranks.append(rank)
        if not isinstance(rec.get("reason"), str) or not rec["reason"].strip():
            problems.append("missing_reason")
        if not isinstance(rec.get("evidence"), list) or not rec["evidence"]:
            problems.append("missing_evidence")
    if len(ids) != len(set(ids)):
        problems.append("duplicate_movie")
    if sorted(ranks) != list(range(1, 11)):
        problems.append("invalid_ranks")
    return sorted(set(problems))


def evaluate_results(cases, labels, results):
    """返回 summary、per_user 和 issues；指标以固定输入用户集合为分母。"""
    cases_by_uid, issues = _index(cases, "case")
    labels_by_uid, more = _index(labels, "label")
    issues.extend(more)
    results_by_uid, more = _index(results, "result")
    issues.extend(more)
    duplicate_uids = {i["user_id"] for i in issues if i["code"] == "duplicate_result"}
    for uid in results_by_uid.keys() - cases_by_uid.keys():
        issues.append({"user_id": uid, "code": "unexpected_result_user"})
    per_user = []
    for uid, case in cases_by_uid.items():
        result = results_by_uid.get(uid)
        problems = _validate(case, result)
        if uid in duplicate_uids:
            problems.append("duplicate_result")
        label = labels_by_uid.get(uid)
        relevant = set(label.get("relevant_movie_ids", [])) if label else set()
        label_error = None
        if label is None:
            label_error = "missing_label"
        elif not relevant:
            label_error = "empty_relevant_set"
        elif not relevant.issubset({m["movie_id"] for m in case["candidates"]}):
            label_error = "relevant_movie_outside_candidates"
        elif relevant & {m["movie_id"] for m in case["history"]}:
            label_error = "relevant_movie_in_history"
        if label_error:
            issues.append({"user_id": uid, "code": label_error})
        issues.extend({"user_id": uid, "code": p} for p in problems)
        valid = not problems
        ranked = sorted(result["recommendations"], key=lambda r: r["rank"]) if valid else []
        selected = [rec["movie_id"] for rec in ranked]
        hits = [int(mid in relevant) for mid in selected]
        row = {
            "user_id": uid, "valid": valid, "issues": problems,
            "metric_eligible": label_error is None, "label_issue": label_error,
            "relevant_count": len(relevant), "hit_count": sum(hits),
            "selected_movie_ids": selected,
            "relevant_movie_ids": sorted(relevant),
            "missed_movie_ids": sorted(relevant - set(selected)),
        }
        if label_error:
            row.update(dict.fromkeys(METRICS, None))
        else:
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
            row.update({
                "recall_at_10": sum(hits) / len(relevant),
                "ndcg_at_10": dcg / ideal,
                "hit_rate_at_10": float(any(hits)),
                "precision_at_10": sum(hits) / 10,
                "mrr_at_10": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
            })
        per_user.append(row)
    eligible = [row for row in per_user if row["metric_eligible"]]
    summary = {
        "user_count": len(per_user), "metric_user_count": len(eligible),
        "excluded_label_user_count": len(per_user) - len(eligible),
        "valid_user_count": sum(row["valid"] for row in per_user),
        "valid_rate": sum(row["valid"] for row in per_user) / len(per_user) if per_user else None,
        "failed_or_invalid_user_count": sum(not row["valid"] for row in per_user),
        "policy": "宏平均；缺失、失败或非法结果计零；空相关集及非法标签排除并单独报告。",
    }
    summary.update({metric: sum(row[metric] for row in eligible) / len(eligible) if eligible else None
                    for metric in METRICS})
    return {"summary": summary, "per_user": per_user, "issues": issues}


def compare_methods(cases, labels, by_method):
    """所有方法共享 cases 与 labels，禁止按成功结果取交集。"""
    evaluations = {method: evaluate_results(cases, labels, results)
                   for method, results in by_method.items()}
    return {"methods": evaluations,
            "summary": {method: result["summary"] for method, result in evaluations.items()},
            "user_ids": [case["user_id"] for case in cases]}


def _genre_profile(history):
    positive, negative = Counter(), Counter()
    for item in history:
        rating = item["rating"]
        genres = item.get("genres", [])
        if isinstance(genres, str):
            genres = genres.split("|")
        if rating >= 4:
            positive.update(genres)
        elif rating <= 2:
            negative.update(genres)
    return {"liked_genre_counts": dict(positive.most_common()),
            "disliked_genre_counts": dict(negative.most_common()),
            "interpretation": "统计为高低评分电影的类型出现次数，多类型电影会重复计入不同类型。"}


def select_failure_cases(cases, labels, results, baseline_results, n=3):
    """仅分析有效前10的漏推荐；服务/格式错误单独报告，不代替推荐失败。"""
    evaluated = evaluate_results(cases, labels, results)
    baseline = {row["user_id"]: row for row in evaluate_results(cases, labels, baseline_results)["per_user"]}
    cases_index = {case["user_id"]: case for case in cases}
    results_index = {row["user_id"]: row for row in results}
    failures = [row for row in evaluated["per_user"]
                if row["valid"] and row["metric_eligible"] and row["missed_movie_ids"]]
    failures.sort(key=lambda row: (
        row["ndcg_at_10"] - (baseline[row["user_id"]]["ndcg_at_10"] or 0),
        row["ndcg_at_10"], row["user_id"]))
    selected_failures = []
    for row in failures[:n]:
        uid = row["user_id"]
        case, result = cases_index[uid], results_index.get(uid, {})
        retrieval = result.get("retrieval", {})
        evidence = retrieval.get("evidence", []) if isinstance(retrieval, dict) else []
        retrieved_ids = {ev["movie_id"] for ev in evidence if isinstance(ev, dict) and "movie_id" in ev}
        relevant = set(row["relevant_movie_ids"])
        missed = set(row["missed_movie_ids"])
        candidates = {m["movie_id"]: m for m in case["candidates"]}
        stage = {
            "retrieved_movie_ids": sorted(retrieved_ids),
            "retrieved_relevant_movie_ids": sorted(retrieved_ids & relevant),
            "missed_not_retrieved": sorted(missed - retrieved_ids),
            "missed_after_retrieval": sorted(missed & retrieved_ids),
            "diagnosis": (
                "结果失败或不满足格式约束，应先检查调用错误及输出校验。" if not row["valid"] else
                "已进入增强证据但未入选的相关电影存在生成选择损失；其余漏推荐电影未被增强证据覆盖。全部100候选仍提供给模型，不能据此断定是唯一失败原因。"
                if evidence else "没有可核查的检索记录，不能将漏推荐直接归因为检索失败。"
            ),
        }
        audit = []
        for rec in result.get("recommendations", []):
            if isinstance(rec, dict):
                audit.append({"movie_id": rec.get("movie_id"), "reason": rec.get("reason"),
                              "evidence": rec.get("evidence", []),
                              "review_question": "逐项核对引用是否存在、历史评分及类型是否吻合；剧情或看过影片的断言必须有实际数据依据。"})
        selected_failures.append({
            "user_id": uid, "metrics": {metric: row[metric] for metric in METRICS},
            "baseline_metrics": {metric: baseline[uid][metric] for metric in METRICS},
            "valid": row["valid"], "issues": row["issues"],
            "selected_movie_ids": row["selected_movie_ids"],
            "missed_movies": [candidates[mid] for mid in sorted(missed) if mid in candidates],
            "history_profile": _genre_profile(case["history"]),
            "history": case["history"], "retrieval_analysis": stage, "reason_audit": audit,
            "preference_note": "类型统计仅能支持偏好倾向，不能据此确定漏推荐的主观原因。",
        })
    return selected_failures

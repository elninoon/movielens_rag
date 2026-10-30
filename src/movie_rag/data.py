"""加载官方脚本生成的数据；推荐侧只读取电影信息和训练评分。"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


class DataStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        with (self.data_dir / "movies.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.movies = {}
            for row in csv.DictReader(stream):
                row["movie_id"] = int(row["movie_id"])
                row["genres"] = row["genres"].split("|") if row["genres"] else []
                self.movies[row["movie_id"]] = row
        with (self.data_dir / "train_ratings.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.ratings = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        self.positive_counts = Counter()
        self.movie_ratings = defaultdict(list)
        self.user_ratings = defaultdict(dict)
        for row in self.ratings:
            mid, rating = row["movie_id"], row["rating"]
            if mid not in self.movies or not 1 <= rating <= 5:
                raise ValueError("训练评分包含无效电影编号或评分")
            self.positive_counts[mid] += int(rating >= 4)
            self.movie_ratings[mid].append(rating)
            self.user_ratings[row["user_id"]][mid] = rating


def load_cases(path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    for case in cases:
        if not {"user_id", "history", "candidates"} <= case.keys():
            raise ValueError("case 缺少 user_id、history 或 candidates")
        ids = [int(movie["movie_id"]) for movie in case["candidates"]]
        if len(ids) != len(set(ids)):
            raise ValueError("候选电影编号重复")
    return cases


def popular_recommendations(store: DataStore, case: dict) -> dict:
    seen = {int(row["movie_id"]) for row in case["history"]}
    candidates = {int(movie["movie_id"]) for movie in case["candidates"]} - seen
    if len(candidates) < 10:
        raise ValueError("不足 10 部未在历史中出现的候选电影")
    ranked = sorted(candidates, key=lambda mid: (-store.positive_counts[mid], mid))[:10]
    return {
        "user_id": case["user_id"],
        "method": "popular",
        "recommendations": [
            {"movie_id": mid, "rank": rank,
             "reason": f"训练集中获得至少4分的评分共{store.positive_counts[mid]}次，按该次数排序推荐。",
             "evidence": [{"source": "train_ratings.csv", "movie_id": mid,
                           "positive_rating_count": store.positive_counts[mid]}]}
            for rank, mid in enumerate(ranked, 1)
        ],
    }

#!/usr/bin/env python3
"""下载 MovieLens 100K，并在本机生成课程开发包或教师私有测试包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5"
KNOWN_MD5 = "0e33842e24a9c977be4e0107933c0723"
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def ssl_context() -> ssl.SSLContext:
    cafile = Path("/etc/ssl/cert.pem")
    return ssl.create_default_context(cafile=str(cafile)) if cafile.exists() else ssl.create_default_context()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"下载 {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "course-data-preparer/1.0"})
    with urllib.request.urlopen(request, timeout=90, context=ssl_context()) as response, partial.open("wb") as fout:
        shutil.copyfileobj(response, fout)
    partial.replace(target)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_md5(zip_path: Path, md5_path: Path) -> None:
    matches = re.findall(r"\b[0-9a-fA-F]{32}\b", md5_path.read_text(encoding="utf-8"))
    if not matches:
        raise RuntimeError("官方 MD5 文件中没有找到 32 位哈希")
    published = matches[-1].lower()
    if published != KNOWN_MD5:
        raise RuntimeError(f"官方 MD5 与脚本固定值不一致：{published}")
    actual = md5sum(zip_path)
    if actual != KNOWN_MD5:
        raise RuntimeError(f"压缩包校验失败：期望 {KNOWN_MD5}，实际 {actual}")
    print(f"MD5 校验通过：{actual}")


def read_ratings(path: Path) -> list[dict]:
    ratings = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("\t")
            ratings.append({
                "user_id": int(user_id), "movie_id": int(movie_id),
                "rating": int(rating), "timestamp": int(timestamp),
            })
    return ratings


def read_movies(path: Path) -> dict[int, dict]:
    movies = {}
    with path.open(encoding="latin-1") as fin:
        for line in fin:
            fields = line.rstrip("\n").split("|")
            flags = [int(x) for x in fields[5:24]]
            movie_id = int(fields[0])
            movies[movie_id] = {
                "movie_id": movie_id,
                "title": fields[1],
                "release_date": fields[2],
                "genres": [name for name, flag in zip(GENRES, flags) if flag],
            }
    return movies


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_case(
    user_id: int, history: list[dict], heldout: list[dict], movies: dict[int, dict],
    all_rated: set[int], rng: random.Random,
) -> tuple[dict, dict]:
    pool = sorted(set(movies) - all_rated)
    negatives = rng.sample(pool, 95)
    candidate_ids = [r["movie_id"] for r in heldout] + negatives
    rng.shuffle(candidate_ids)
    case = {
        "user_id": user_id,
        "history": [
            {
                "movie_id": r["movie_id"], "title": movies[r["movie_id"]]["title"],
                "genres": movies[r["movie_id"]]["genres"], "rating": r["rating"],
                "timestamp": r["timestamp"],
            }
            for r in history
        ],
        "candidates": [movies[mid] for mid in candidate_ids],
    }
    label = {
        "user_id": user_id,
        "heldout": heldout,
        "relevant_movie_ids": sorted(r["movie_id"] for r in heldout if r["rating"] >= 4),
    }
    return case, label


def check_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"输出目录非空：{path}。请换一个新目录，避免覆盖已有文件。")


def excluded_users(manifest_path: str | None) -> set[int]:
    if not manifest_path:
        return set()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {int(x) for x in manifest["selected_user_ids"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True, help="teacher 模式必须使用未公开的私有整数")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--zip", dest="zip_path", help="可选：已下载的 ml-100k.zip")
    parser.add_argument("--exclude-manifest", help="teacher 模式可排除 student 模式已选用户")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    check_output(output)
    excluded = excluded_users(args.exclude_manifest)

    with tempfile.TemporaryDirectory(prefix="movielens_course_") as tmp:
        temp = Path(tmp)
        zip_path = Path(args.zip_path).resolve() if args.zip_path else temp / "ml-100k.zip"
        md5_path = temp / "ml-100k.zip.md5"
        if not zip_path.exists():
            download(DATA_URL, zip_path)
        download(MD5_URL, md5_path)
        verify_md5(zip_path, md5_path)

        raw_dir = temp / "raw"
        wanted = {"ml-100k/u.data", "ml-100k/u.item"}
        with zipfile.ZipFile(zip_path) as archive:
            missing = wanted - set(archive.namelist())
            if missing:
                raise RuntimeError(f"压缩包缺少文件：{sorted(missing)}")
            archive.extractall(raw_dir, members=sorted(wanted))
        ratings = read_ratings(raw_dir / "ml-100k/u.data")
        movies = read_movies(raw_dir / "ml-100k/u.item")

    by_user: dict[int, list[dict]] = defaultdict(list)
    for row in ratings:
        by_user[row["user_id"]].append(row)
    for rows in by_user.values():
        rows.sort(key=lambda r: (r["timestamp"], r["movie_id"]))

    eligible = [
        uid for uid, rows in by_user.items()
        if uid not in excluded and len(rows) >= 20 and any(r["rating"] >= 4 for r in rows[-5:])
    ]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = set(eligible[: args.users])
    if len(selected) < args.users:
        raise RuntimeError(f"符合条件且未被排除的用户只有 {len(selected)} 位")

    train_rows, cases, labels = [], [], []
    for uid, rows in sorted(by_user.items()):
        heldout = rows[-5:] if uid in selected else []
        history = rows[:-5] if heldout else rows
        train_rows.extend(history)
        if heldout:
            case, label = build_case(
                uid, history, heldout, movies, {r["movie_id"] for r in rows},
                random.Random(f"{args.seed}:{uid}"),
            )
            cases.append(case)
            labels.append(label)

    output.mkdir(parents=True, exist_ok=True)
    movie_rows = [
        {"movie_id": m["movie_id"], "title": m["title"], "release_date": m["release_date"],
         "genres": "|".join(m["genres"])}
        for m in movies.values()
    ]
    write_csv(output / "movies.csv", movie_rows, ["movie_id", "title", "release_date", "genres"])
    write_csv(output / "train_ratings.csv", train_rows, ["user_id", "movie_id", "rating", "timestamp"])
    prefix = "validation" if args.mode == "student" else "test"
    write_jsonl(output / f"{prefix}_cases.jsonl", sorted(cases, key=lambda x: x["user_id"]))
    write_jsonl(output / f"{prefix}_labels.jsonl", sorted(labels, key=lambda x: x["user_id"]))
    (output / "split_manifest.json").write_text(json.dumps({
        "mode": args.mode,
        "seed": args.seed,
        "selected_user_ids": sorted(selected),
        "excluded_user_ids": sorted(excluded),
        "data_url": DATA_URL,
        "zip_md5": KNOWN_MD5,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{output}")
    if args.mode == "student":
        print("该目录供本组本地开发使用；请勿上传或再次分发 MovieLens 派生数据。")
    else:
        print("教师测试目录和 seed 必须保密，不得发给学生。")


if __name__ == "__main__":
    main()

"""核查教师脚本生成的数据，标签仅用于事后审计，不进入推荐模块。"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'movielens_dev'


def main():
    manifest = json.loads((DATA / 'split_manifest.json').read_text(encoding='utf-8'))
    assert manifest['mode'] == 'student' and manifest['seed'] == 20261018
    assert manifest['zip_md5'] == '0e33842e24a9c977be4e0107933c0723'
    with (DATA / 'train_ratings.csv').open(encoding='utf-8-sig') as stream:
        train = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    with (DATA / 'movies.csv').open(encoding='utf-8-sig') as stream:
        movies = list(csv.DictReader(stream))
    cases = [json.loads(line) for line in (DATA / 'validation_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    labels = {row['user_id']: row for row in map(json.loads, (DATA / 'validation_labels.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(cases) == len(labels) == len(manifest['selected_user_ids']) == 30
    assert len(movies) == 1682 and len(train) == 99850
    by_user = defaultdict(list)
    for row in train:
        by_user[row['user_id']].append(row)
    assert len(by_user) == 943
    pairs = {(r['user_id'], r['movie_id']) for r in train}
    assert len(pairs) == len(train)
    for case in cases:
        uid = case['user_id']
        history = case['history']
        heldout = labels[uid]['heldout']
        assert len(heldout) == 5
        assert {(r['movie_id'], r['rating'], r['timestamp']) for r in history} == {
            (r['movie_id'], r['rating'], r['timestamp']) for r in by_user[uid]}
        assert all((uid, r['movie_id']) not in pairs for r in heldout)
        assert max((r['timestamp'], r['movie_id']) for r in history) <= min((r['timestamp'], r['movie_id']) for r in heldout)
        candidate_ids = {r['movie_id'] for r in case['candidates']}
        history_ids = {r['movie_id'] for r in history}
        heldout_ids = {r['movie_id'] for r in heldout}
        assert len(candidate_ids) == len(case['candidates']) == 100
        assert not candidate_ids & history_ids
        assert heldout_ids <= candidate_ids
        assert len(candidate_ids - heldout_ids) == 95
        assert set(labels[uid]['relevant_movie_ids']) == {r['movie_id'] for r in heldout if r['rating'] >= 4}
        assert labels[uid]['relevant_movie_ids']
    result = {'status': 'passed', 'mode': 'student', 'seed': manifest['seed'],
        'movie_count': len(movies), 'user_count': len(by_user), 'train_rating_count': len(train),
        'validation_user_count': len(cases), 'heldout_rating_count': 150,
        'candidates_per_user': 100, 'unrated_candidates_per_user': 95,
        'checks': ['官方MD5', '历史与训练一致', '留出评分不进入训练', '每用户最后5条留出',
                   '100候选含5留出和95未评分', '历史与候选无交集', '相关标签为留出评分>=4']}
    out = ROOT / 'outputs' / 'data_audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

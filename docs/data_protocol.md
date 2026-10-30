# 数据生成与隔离协议

本项目沿用教师附件 `02_学生附件/prepare_movielens.py`，不自行重做划分、不改用随机评分切分、不使用教师私有测试模式。

## 生成命令

在材料根目录执行：

```powershell
python .\02_学生附件\prepare_movielens.py --mode student --output .\movielens_rag\data\movielens_dev --seed 20261018
```

输出目录必须为空。附件默认 `--users 30`，保持默认值。若已有官方原始压缩包，可增加 `--zip <ml-100k.zip 的本地路径>`；脚本仍会下载官方 MD5 文件并验证，不能据此跳过校验。

官方地址：

- 数据主页：[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- 压缩包：[ml-100k.zip](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
- 校验文件：[ml-100k.zip.md5](https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5)

教师脚本固定 MD5 为 `0e33842e24a9c977be4e0107933c0723`，同时核对官方发布值和下载文件实际值。脚本仅解析 `u.data` 与 `u.item`，不需要外部剧情资料。

## 真实划分规则

1. 每位用户的评分按 `(timestamp, movie_id)` 升序排列。
2. 候选验证用户须至少有 20 条评分，且最后 5 条评分中至少一条评分 ≥4。固定种子打乱符合条件的用户，取前 30 位。
3. 对这 30 位用户，各自最后 5 条评分留出，其余评分进入训练集；其他用户的全部评分进入训练集。因此这不是整个数据集统一时间点切分，也没有为所有用户都留出 5 条。
4. 对每位验证用户，把 5 部留出电影与 95 部从该用户从未评分电影中无放回抽取的电影合并，形成 100 部候选。95 部的抽样池排除了该用户全部已评分电影，包含已留出的部分。
5. 每位用户的候选抽样与打乱使用独立的 `Random(f"{seed}:{user_id}")`。不得重新随机生成候选以获得更好结果。
6. 标签中的 `relevant_movie_ids` 仅包含 5 条留出记录里评分 ≥4 的电影。5 部留出电影不一定全部相关；95 部未评分电影也不能解释为真实负反馈。

## 输出文件及使用权限

| 文件 | 内容 | 允许读取的模块 |
| --- | --- | --- |
| `movies.csv` | 编号、片名、上映日期、类型 | 数据、基线、检索、推荐、评测 |
| `train_ratings.csv` | 训练用户编号、电影编号、评分、时间戳 | 数据、热门统计、训练偏好构建、检索、评测 |
| `validation_cases.jsonl` | 每用户编号、训练历史及 100 部候选基本信息 | 基线、检索、推荐、评测 |
| `validation_labels.jsonl` | 留出评分及相关电影编号 | 仅离线评测与事后失败分析 |
| `split_manifest.json` | 模式、种子、选中用户、数据地址及 MD5 | 复现记录与数据检查 |

禁止通过标签构建向量、查询、用户偏好、热门次数、推荐提示词或重排特征。电影候选元数据可进入模型，但留出评分不可进入模型。正式比较使用同一组 30 位用户、相同历史、相同 100 部候选和同一标签定义。

## 结果检查与保存

逐用户检查：候选数为 100，历史与候选不相交；推荐恰为 10 部、不重复、均在候选内且不在历史中。评测不应静默删除调用失败或输出非法的用户，应记录失败并使用一致分母。

保存生成命令、manifest、实验配置、实际推荐与检索证据、调用记录和评测汇总，以便从结果追溯到输入。所有含 MovieLens 原始或派生数据的文件保留在本机，公共代码包只保留生成方式，不打包数据、标签、向量缓存或包含数据内容的运行输出。

报告引用：F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)。

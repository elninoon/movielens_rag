// 历史材料构建器：只重建outputs/formal旧流程PPT，不代表当前100→50→30→10方法。
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const workspaceDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dependencies = 'C:/Users/Zhehao Yang/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skillDir = 'C:/Users/Zhehao Yang/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
process.env.RUNTIME_NODE_MODULES = `${dependencies}/node/node_modules`;
const { Presentation, PresentationFile } = await import(pathToFileURL(`${process.env.RUNTIME_NODE_MODULES}/@oai/artifact-tool/dist/artifact_tool.mjs`).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(`${skillDir}/container_tools/artifact_tool_utils.mjs`).href);

const buildDir = path.join(workspaceDir, 'build', 'presentation');
const outputDir = path.join(workspaceDir, 'deliverables');
const formalDir = path.join(workspaceDir, 'outputs', 'formal');
const FONT = 'Microsoft YaHei';
const INK = '#182638', MUTED = '#556476', BLUE = '#1C5EA8', TEAL = '#228B80', ORANGE = '#C27638';
const methodNames = { popular: '热门基线', rag: 'RAG', no_retrieval: '移除检索' };
const methodKeys = ['popular', 'rag', 'no_retrieval'];
const readJson = async file => JSON.parse(await fs.readFile(file, 'utf8'));
// 汇总文件缺失即停止，不生成填入估算数字的演示稿。
const evaluation = await readJson(path.join(formalDir, 'evaluation.json'));
const failures = await readJson(path.join(formalDir, 'failure_cases.json'));
const config = await readJson(path.join(formalDir, 'config.json'));
if (failures.length < 3) throw new Error('需要至少三个真实失败用户');
for (const method of methodKeys) {
  if (!evaluation.summary[method] || evaluation.summary[method].user_count !== config.user_ids.length) {
    throw new Error(`正式结果缺失或用户数不一致: ${method}`);
  }
}
const summaries = evaluation.summary;
// 权限中断与模型输出校验失败分别报告，避免把真实模型失败当作待续跑。
let interrupted = false;
const pct = n => `${(n * 100).toFixed(1)}%`;
const fixed = n => Number(n).toFixed(3);
const pp = n => `${n >= 0 ? '+' : ''}${(n * 100).toFixed(1)} 个百分点`;
const delta = summaries.rag.ndcg_at_10 - summaries.popular.ndcg_at_10;
const ablationDelta = summaries.rag.ndcg_at_10 - summaries.no_retrieval.ndcg_at_10;
const citation = '来源：outputs/formal/evaluation.json，failure_cases.json，config.json；教师作业说明及 prepare_movielens.py。';
const storeData = path.join(workspaceDir, 'data', 'movielens_dev');
const moviesCount = (await fs.readFile(path.join(storeData, 'movies.csv'), 'utf8')).trim().split(/\r?\n/).length - 1;
const ratingsCount = (await fs.readFile(path.join(storeData, 'train_ratings.csv'), 'utf8')).trim().split(/\r?\n/).length - 1;

const cost = {};
for (const method of ['rag', 'no_retrieval']) {
  const rows = await Promise.all(config.user_ids.map(uid => readJson(path.join(formalDir, method, `${uid}.json`))));
  interrupted ||= rows.some(row => /HTTP\s*403|AccessDenied/.test(row.error ?? ''));
  const calls = rows.flatMap(row => row.calls ?? []);
  const tokens = calls.reduce((sum, call) => sum + (call.usage?.total_tokens ??
    ((call.usage?.prompt_tokens ?? 0) + (call.usage?.completion_tokens ?? 0))), 0);
  cost[method] = { calls: calls.length, tokens, seconds: rows.reduce((sum, row) => sum + row.seconds, 0),
    usageComplete: calls.every(call => call.usage && Object.keys(call.usage).length > 0) };
}
const retrievalRows = await Promise.all(config.user_ids.map(uid => readJson(path.join(formalDir, 'retrieval', `${uid}.json`))));
const retrievalCalls = retrievalRows.flatMap(row => row.calls ?? []);
const retrievalSeconds = retrievalRows.reduce((sum, row) => sum + row.seconds, 0);
let archived403Calls = 0;
for (const method of ['rag', 'no_retrieval']) {
  const folder = path.join(formalDir, 'errors', method);
  const files = await fs.readdir(folder).catch(error => {
    if (error.code === 'ENOENT') return [];
    throw error;
  });
  for (const file of files.filter(name => name.endsWith('.json'))) {
    const result = await readJson(path.join(folder, file));
    archived403Calls += (result.calls ?? []).filter(call => call.http_status === 403).length;
  }
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const notes = [];
function text(slide, content, left, top, width, height, size = 28, bold = false, color = INK) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left, top, width, height },
    fill: 'none', line: { style: 'solid', fill: 'none', width: 0 } });
  shape.text = String(content);
  shape.text.style = { typeface: FONT, fontSize: size, bold, color };
  return shape;
}
function slide(title) {
  const page = deck.slides.add();
  page.background.fill = '#FFFFFF';
  text(page, title, 76, 48, 1118, 75, 43, true);
  text(page, `${deck.slides.items.length} / 10`, 1135, 664, 92, 30, 18, false, MUTED);
  return page;
}
function note(page, content) {
  page.speakerNotes.text = `${content}\n\n${citation}`;
  notes.push(content);
}
function table(page, values, widths, top = 180, height = 350, size = 25) {
  const grid = page.tables.add({ rows: values.length, columns: values[0].length,
    left: 78, top, width: 1120, height, columnWidths: widths, values });
  grid.borders.assign({ style: 'solid', fill: '#DCE3EA', width: 1 });
  for (let row = 0; row < values.length; row++) {
    for (let column = 0; column < values[0].length; column++) {
      const cell = grid.getCell(row, column);
      cell.fill = row === 0 ? '#E7EEF5' : row % 2 === 0 ? '#F7F9FB' : '#FFFFFF';
      cell.text.style = { typeface: FONT, fontSize: size, bold: row === 0, color: INK };
    }
  }
  return grid;
}

{
  const page = deck.slides.add(); page.background.fill = '#FFFFFF';
  text(page, 'MovieLens 电影推荐', 82, 187, 1110, 90, 62, true);
  text(page, '基于 Qwen 的检索增强推荐与评测', 86, 296, 1100, 62, 34, false, BLUE);
  text(page, `${config.user_ids.length} 位验证用户   每位 100 部候选   推荐 Top-10`, 87, 410, 1090, 50, 28, false, MUTED);
  if (interrupted) text(page, 'API 权限中断，实验待续跑', 87, 493, 1090, 48, 30, true, ORANGE);
  text(page, 'LLM 课程大作业   八人小组', 87, 591, 1090, 40, 25, false, MUTED);
  note(page, `约30秒。介绍课题：利用用户历史电影评分推断偏好，在固定的一百部候选中推荐十部电影，给出能够核查的依据。选择题目一的RAG方案，使用真实百炼接口。${interrupted ? '当前API返回HTTP403 AccessDenied.Unpurchased，实验中断。本稿展示阶段记录，不将失败计零后的指标解释为算法优劣，待授权恢复后续跑。' : '正式结果已按固定用户集合完成。'}介绍成员时按小组实际名单填写，后面的分工页只列建议职责。`);
}
{
  const page = slide('任务与验证数据');
  text(page, '输入与输出', 80, 165, 470, 50, 31, true, BLUE);
  text(page, '用户训练历史及评分\n100 部候选电影的基本信息\n10 部未看电影、排序及依据', 80, 233, 550, 210, 29);
  text(page, 'MovieLens 100K', 710, 165, 485, 50, 31, true, BLUE);
  text(page, `${moviesCount.toLocaleString()} 部电影\n${ratingsCount.toLocaleString()} 条训练评分\n${config.user_ids.length} 位验证用户，各留出最近 5 条评分`, 710, 233, 500, 210, 28);
  text(page, '相关电影：留出评分 ≥4。验证标签只在生成完成后进入评测。', 80, 515, 1115, 70, 27);
  text(page, '电影信息包含名称、上映日期、类型，不包含剧情或影评。', 80, 600, 1115, 44, 24, false, MUTED);
  note(page, `约55秒。数据来自官方MovieLens 100K，由教师脚本固定种子20261018生成。实际训练集为${ratingsCount}条评分。每位验证用户最后五条评分作为留出记录，其中评分至少四分的电影构成相关集，其余候选由脚本采样。检索和生成不读取标签。数据仅用于本组本地课程开发，不上传公共仓库。引用：Harper and Konstan, 2015, The MovieLens Datasets: History and Context, DOI 10.1145/2827872。`);
}
{
  const page = slide('RAG 推荐流程');
  const rows = [
    ['01', '训练历史构造查询', '类型均分、高分电影和低分电影'],
    ['02', '候选检索与重排序', '电影元数据、训练统计、相似用户评分'],
    ['03', 'Qwen 生成推荐', '完整历史、全部 100 候选及 30 条检索证据'],
    ['04', '结果校验与独立评测', '候选范围、去重、排序、引用及 Top-10 指标'],
  ];
  rows.forEach((row, i) => {
    const y = 168 + i * 111;
    text(page, row[0], 82, y, 85, 58, 34, true, BLUE);
    text(page, row[1], 195, y, 1000, 47, 30, true);
    text(page, row[2], 195, y + 49, 1000, 43, 25, false, MUTED);
  });
  note(page, '约1分钟。先用训练历史描述用户偏好，再把候选电影的真实元数据和训练统计组织为可检索文本。真实向量接口和重排序接口产生候选证据，协同近邻信号补充相似用户对候选的训练评分。生成模型同时看到完整历史和一百部候选，因此检索证据负责补充信息，并不把生成选择硬限制在三十部电影。最后检查输出格式与引用，再由独立评测模块读取标签。');
}
{
  const page = slide('检索对象与排序依据');
  table(page, [
    ['阶段', '实现与依据'],
    ['用户查询', '全部历史的类型均分，最多 8 部高分与 5 部低分电影'],
    ['检索文本', '电影名称、上映日期、类型及训练评分统计'],
    ['向量与重排序', 'text-embedding-v4，1024 维；qwen3-rerank'],
    ['协同支持', '最多 40 位相似训练用户，排除目标用户自身'],
    ['证据输出', '候选集合内 Top-30，保留来源、分数及近邻评分'],
  ], [245, 875], 165, 385, 25);
  text(page, '融合分数 = 0.45 × 重排序 + 0.35 × 协同支持 + 0.20 × 向量相似度', 80, 577, 1120, 65, 24, false, BLUE);
  note(page, '约1分10秒。说明检索对象是电影信息及训练评分统计，不是模型编写的剧情。类型均分使用全部训练历史，高低分电影只用于压缩查询文本。向量余弦和协同评分映射到零到一后参与融合，相似用户以训练评分减三再除二的向量计算余弦，最多取四十个正相似邻居。协同评分含两个中性伪观测，降低少量支持造成的极端值。融合系数是预设配置，并非从验证标签调出的最优值。证据包括来源、电影编号、训练统计及实际近邻评分。');
}
{
  const page = slide('生成约束与消融设计');
  text(page, 'qwen3.8-27b', 80, 165, 1035, 58, 36, true, BLUE);
  text(page, '完整训练历史和全部候选均提供给模型\nQwen 负责排序，最终理由从实际历史、类型和训练统计构建\n保留原始模型理由及调用记录，核对依据中的事实', 80, 244, 1115, 169, 29);
  text(page, '消融只清空 retrieved_evidence', 80, 453, 1115, 55, 32, true);
  text(page, '历史、100 部候选、系统提示词、模型及温度均保持一致。\n热门基线按训练集中 ≥4 分次数降序，同次数按电影编号升序。', 80, 528, 1115, 112, 26, false, MUTED);
  note(page, '约1分钟。模型仅能使用输入中的历史评分、类型、上映信息和检索内容，禁止调用记忆中的剧情。程序验证恰好十部、候选范围、未看、无重复、排名连续及引用存在，失败可纠正一次。实际抽查发现引用存在仍可能误述评分，因此保留model_reason和原始调用，最终理由由确定性程序依据实际历史评分、类型重合及训练统计构建，Qwen仍负责排序。RAG和消融均执行相同理由处理，不改变排序。消融严格只移除检索证据，其余输入和参数一致。热门基线按训练集中至少四分的评分次数排序。');
}
{
  const page = slide(interrupted ? '阶段结果：失败按零计入' : '三种方法的推荐效果');
  if (interrupted) text(page, 'API 权限中断，实验待续跑', 80, 128, 1110, 37, 24, true, ORANGE);
  const chart = page.charts.add('bar', {
    position: { left: 80, top: 168, width: 1120, height: 380 },
    categories: ['Recall@10', 'NDCG@10', 'HitRate@10'],
    series: methodKeys.map((method, i) => ({ name: methodNames[method],
      values: ['recall_at_10', 'ndcg_at_10', 'hit_rate_at_10'].map(metric => +summaries[method][metric].toFixed(6)),
      fill: [BLUE, TEAL, ORANGE][i] })),
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 90 },
    hasLegend: true, legend: { position: 'bottom', overlay: false, textStyle: { fontSize: 22, fill: INK } },
    xAxis: { textStyle: { fontSize: 23, fill: INK } },
    yAxis: { minimumScale: 0, maximumScale: 1, numberFormatCode: '0%',
      textStyle: { fontSize: 20, fill: MUTED }, majorGridlines: { style: 'solid', fill: '#E3E8EE', width: 1 } },
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0.0%', textStyle: { fontSize: 19, fill: INK } },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  text(page, interrupted ? '成功用户集合不一致，当前指标不能用于比较模型优劣。'
    : `RAG 相对热门基线的 NDCG@10：${pp(delta)}`, 82, 579, 1115, 55, 28, true, interrupted ? ORANGE : BLUE);
  if (!interrupted && summaries.no_retrieval.failed_or_invalid_user_count) {
    text(page, `消融 ${summaries.no_retrieval.failed_or_invalid_user_count} 次输出约束失败，按固定 30 位用户分母计零。`, 82, 634, 1050, 35, 23, false, MUTED);
  }
  note(page, `约1分10秒。图表是实际${config.user_ids.length}位用户宏平均，不是挑选成功用户后的平均。热门基线Recall为${pct(summaries.popular.recall_at_10)}，RAG为${pct(summaries.rag.recall_at_10)}，消融为${pct(summaries.no_retrieval.recall_at_10)}。NDCG同时考虑是否命中和命中位置。${interrupted ? 'RAG和消融均包含接口调用失败，成功用户集合也不完全一致。图表是固定分母的运行记录，不能据此判断算法不如热门或推断检索是否有效。应在授权恢复后完成同一用户集合再比较。' : `RAG相对热门基线为${pp(delta)}，相对消融为${pp(ablationDelta)}。消融用户921在两轮输出中违反候选或历史约束，被校验拒绝，保留为一次失败并计零，不通过反复重试挑选成功答案。这些指标同时反映推荐质量与输出可靠性，不是仅成功请求的模型能力比较。同一验证集上的一次差异不代表统计显著。`}缺失、失败及非法推荐计零，标签无效用户另行报告。`);
}
{
  const page = slide('输出有效性与调用开销');
  table(page, [
    ['方法', '有效用户', '生成调用', '已记录 Token', '均用户生成耗时'],
    ['热门基线', `${summaries.popular.valid_user_count}/${config.user_ids.length}`, '0', '0', '不调用模型'],
    ['RAG', `${summaries.rag.valid_user_count}/${config.user_ids.length}`, String(cost.rag.calls),
      cost.rag.tokens.toLocaleString(), `${(cost.rag.seconds / config.user_ids.length).toFixed(1)} 秒`],
    ['移除检索', `${summaries.no_retrieval.valid_user_count}/${config.user_ids.length}`, String(cost.no_retrieval.calls),
      cost.no_retrieval.tokens.toLocaleString(), `${(cost.no_retrieval.seconds / config.user_ids.length).toFixed(1)} 秒`],
  ], [195, 185, 200, 245, 295], 175, 280, 25);
  text(page, `检索阶段另有 ${retrievalCalls.length} 次 API 调用，累计 ${retrievalSeconds.toFixed(1)} 秒。`, 80, 500, 1115, 66, 28, true, BLUE);
  const pending = ['rag', 'no_retrieval'].reduce((sum, method) => sum + summaries[method].failed_or_invalid_user_count, 0);
  text(page, interrupted ? `两方案各 ${summaries.rag.failed_or_invalid_user_count} 个失败，共 ${pending} 个请求待续跑。Token 仅统计已返回 usage。`
    : archived403Calls ? `另有 ${archived403Calls} 次历史 HTTP403 已归档，不含在表中调用数，最终结果均已续跑。`
      : '生成耗时为各用户请求耗时的均值，检索累计耗时受已有向量缓存影响。', 80, 585, 1115, 64, 24, false, MUTED);
  note(page, `约55秒。表中的Token仅累计当前结果文件中生成API实际返回的usage，包含该结果中的模型纠正或重试，不包含检索Token，也不估算失败请求未返回的usage。${archived403Calls ? `另有${archived403Calls}次历史HTTP403请求归档于outputs/formal/errors，未计入表内调用数，历史失败不作为最终方法失败。` : ''}检索调用数单独列出，其耗时包含向量查询和重排序等步骤，向量缓存已经命中的文本不会重复请求。生成任务可并发，累计请求耗时不能当成整个实验墙钟时间。有效性指标衡量结构和引用编号，最终理由另有基于真实数据的确定性构建，原始模型理由仍然保留用于审查。`);
}
{
  const page = slide('三个失败用户的证据定位');
  table(page, [
    ['用户', 'RAG / 热门 NDCG', '漏选相关电影', '未进检索证据', '已有证据仍漏选'],
    ...failures.slice(0, 3).map(row => [String(row.user_id),
      `${fixed(row.metrics.ndcg_at_10)} / ${fixed(row.baseline_metrics.ndcg_at_10)}`,
      String(row.missed_movies.length), String(row.retrieval_analysis.missed_not_retrieved.length),
      String(row.retrieval_analysis.missed_after_retrieval.length)]),
  ], [130, 330, 220, 220, 220], 171, 280, 25);
  const allMissedRetrieved = failures.slice(0, 3).every(row => row.retrieval_analysis.missed_not_retrieved.length === 0);
  text(page, allMissedRetrieved ? '三个用户的漏选电影均已进入检索证据，问题集中在生成排序。'
    : '未进入证据的漏选反映检索覆盖问题，已有证据仍漏选反映排序问题。', 80, 496, 1120, 78, 27);
  text(page, '模型仍能看到全部候选。原始模型理由另行保留，最终理由依据实际数据构建。', 80, 589, 1120, 58, 24, false, MUTED);
  const details = failures.slice(0, 3).map(row => `用户${row.user_id}漏选${row.missed_movies.map(movie => `${movie.title}（${movie.movie_id}）`).join('、')}，其中${row.retrieval_analysis.missed_after_retrieval.length}部已有检索证据。`).join('');
  note(page, `约1分10秒。失败案例按RAG相对热门基线的NDCG差值从低到高选择。${details}这些用户的结果都合法，表中衡量真实推荐损失，不混入API失败。完整训练历史和推荐理由位于failure_cases.json。由于生成器保留全部候选，未检索到并不构成不能推荐的硬限制。历史类型偏好只能说明倾向，不能确定用户主观原因。答辩时可以打开demo展示一个理由及其真实历史或候选类型来源。`);
}
{
  const page = slide('八人分工建议');
  table(page, [
    ['小组', '成员一职责', '成员二职责'],
    ['A 数据与基线', '成员1（姓名待补）\n数据准备与隔离', '成员2（姓名待补）\n热门基线与数据检查'],
    ['B 检索', '成员3（姓名待补）\n索引、向量及缓存', '成员4（姓名待补）\n重排序及检索分析'],
    ['C 生成与集成', '成员5（姓名待补）\n提示词和依据输出', '成员6（姓名待补）\n接口集成与演示入口'],
    ['D 实验与评测', '成员7（姓名待补）\n评测与消融调度', '成员8（姓名待补）\n失败分析与汇报统稿'],
  ], [240, 440, 440], 162, 395, 24);
  text(page, '先约定 case、retrieval、result 接口，各组用模拟样例独立开发。', 80, 590, 1115, 57, 25, false, MUTED);
  note(page, '约45秒。本页是拟定工作分配，姓名和实际完成情况应由组员据实补齐，不将建议写成已完成贡献。四组分别承担数据和基线、检索、生成集成、实验评测，每组各负责自己的报告章节和汇报页面。首日固定三个接口并用模拟样例开发，评测组先使用基线或模拟结果开发指标，减少上下游等待。');
}
{
  const page = slide('结果与反思');
  text(page, interrupted ? '接口权限恢复后，完成固定用户集合的实验' : `RAG 的 NDCG@10 为 ${fixed(summaries.rag.ndcg_at_10)}`, 80, 171, 1120, 61, 35, true, BLUE);
  text(page, interrupted ? '当前结果保留失败记录，不能据此判断推荐方法的优劣。'
    : `相对热门基线 ${pp(delta)}，相对移除检索 ${pp(ablationDelta)}。`, 80, 247, 1120, 75, 28);
  text(page, '原始理由出现评分误述，引用核查需要覆盖事实', 80, 369, 1120, 55, 31, true);
  text(page, '用户 20：模型把历史电影 121 的 3 分描述为高分。\n修复：保留原文，依据实际数据构建最终理由，保持推荐排序。', 80, 445, 1120, 119, 28);
  text(page, '一次验证实验提供当前配置的证据，后续改进需重新验证。', 80, 603, 1120, 42, 24, false, MUTED);
  note(page, `约1分钟。${interrupted ? '当前真实实验因接口权限中断，下一步先恢复授权并续跑失败用户，再在同一固定集合比较基线、RAG和消融。失败必须保留，不能用成功子集替代完整结果。' : `当前RAG相对热门NDCG差值为${pp(delta)}，相对消融为${pp(ablationDelta)}。`}实际抽查用户20发现，模型把历史电影121的三分误述为高分，虽然引用编号存在，文字事实仍然错误。我们保留model_reason和原始调用，由确定性代码根据实际历史、电影类型和训练统计构建最终理由，两个模型方案采用相同规则，推荐顺序不变。后续需要用独立数据比较检索与排序配置，类型和有限评分也难以完整表达个人偏好。`);
}

await fs.writeFile(path.join(workspaceDir, 'docs', 'presentation_notes.md'),
  '# 汇报讲稿\n\n约十分钟，逐页讲稿同时写入 PPTX 备注。成员姓名与实际贡献需据实补充。\n\n'
  + notes.map((content, index) => `## 第 ${index + 1} 页\n\n${content}\n`).join('\n'), 'utf8');

for (const [index, page] of deck.slides.items.entries()) {
  const blob = await deck.export({ slide: page, format: 'png', scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${String(index + 1).padStart(2, '0')}.png`), new Uint8Array(await blob.arrayBuffer()));
}
const candidatePath = path.join(buildDir, `candidate-${Date.now()}.pptx`);
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const defaultFinal = path.join(outputDir, 'MovieLens_RAG_汇报.pptx');
const finalPath = process.argv[2] ? path.resolve(process.argv[2]) : defaultFinal;
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const result = await finalizePresentation({ workspaceDir, candidatePath, finalPath,
  pythonExecutable: `${dependencies}/python/python.exe`,
  integrityValidatorPath: `${skillDir}/container_tools/inspect_presentation_package_integrity.py`,
  layoutValidatorPath: `${skillDir}/container_tools/inspect_presentation_layout_geometry.py`,
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit',
    ...[4, 7, 8, 9].flatMap(number => ['--require-native-table-slide', String(number)])],
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [4, 7, 8, 9], requiredNativeChartOwnerSlides: [6],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [FONT] }, verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, `validation-${Date.now()}.json`),
});
console.log(JSON.stringify({ finalPath, result }, null, 2));

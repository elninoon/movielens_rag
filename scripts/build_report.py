"""重建旧版 outputs/formal 实验报告；不代表当前级联流程，当前对照用 analyze_cascade.py。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
FONT_NAME = 'ReportCN' if Path('C:/Windows/Fonts/simsun.ttc').exists() else 'STSong-Light'
sys.path.insert(0, str(ROOT / 'src'))
from movie_rag.generation import SYSTEM_PROMPT, build_messages


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def number(value, digits=4):
    return f'{value:.{digits}f}' if value is not None else '未定义'


def movies_text(ids, movies):
    return '；'.join(f"{mid} {movies[mid]['title']}" for mid in ids) if ids else '无'


def usage(calls):
    output = {'calls': len(calls), 'failed_calls': sum(not c.get('ok', False) for c in calls),
              'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
              'seconds': sum(c.get('seconds', 0) for c in calls),
              'input_reported': False, 'output_reported': False}
    for call in calls:
        used = call.get('usage', {})
        output['input_reported'] |= 'prompt_tokens' in used or 'input_tokens' in used
        output['output_reported'] |= 'completion_tokens' in used or 'output_tokens' in used
        inputs = used.get('prompt_tokens', used.get('input_tokens', 0))
        outputs = used.get('completion_tokens', used.get('output_tokens', 0))
        output['input_tokens'] += inputs
        output['output_tokens'] += outputs
        output['total_tokens'] += used.get('total_tokens', inputs + outputs)
    return output


class Report:
    def __init__(self):
        if FONT_NAME == 'ReportCN':
            pdfmetrics.registerFont(TTFont(FONT_NAME, 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
        else:
            pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
        self.styles = getSampleStyleSheet()
        for name in ('Normal', 'BodyText', 'Heading1', 'Heading2', 'Title'):
            self.styles[name].fontName = FONT_NAME
            self.styles[name].textColor = colors.black
            self.styles[name].wordWrap = 'CJK'
        self.styles['BodyText'].fontSize = 10.2
        self.styles['BodyText'].leading = 16
        self.styles['BodyText'].spaceAfter = 7
        self.styles['Title'].fontSize = 22
        self.styles['Title'].leading = 30
        self.styles['Title'].spaceAfter = 13
        self.styles['Heading1'].fontSize = 14
        self.styles['Heading1'].leading = 21
        self.styles['Heading1'].spaceBefore = 13
        self.styles['Heading1'].spaceAfter = 7
        self.styles['Heading2'].fontSize = 11.8
        self.styles['Heading2'].leading = 18
        self.styles['Heading2'].spaceBefore = 10
        self.styles['Heading2'].spaceAfter = 5
        self.styles.add(ParagraphStyle('SmallCN', fontName=FONT_NAME, fontSize=8.5,
                                      leading=12, wordWrap='CJK', spaceAfter=3))
        self.styles.add(ParagraphStyle('CellCN', fontName=FONT_NAME, fontSize=8.8,
                                      leading=12.5, wordWrap='CJK'))
        self.styles.add(ParagraphStyle('FormulaCN', fontName=FONT_NAME, fontSize=9.2,
                                      leading=14, wordWrap='CJK', leftIndent=10, spaceAfter=8))
        self.story, self.markdown, self.pending_heading = [], [], []

    def heading(self, text, level=1):
        style = 'Title' if level == 0 else 'Heading1' if level == 1 else 'Heading2'
        self.pending_heading.append(Paragraph(escape(text), self.styles[style]))
        self.markdown.append('#' * (level + 1 if level else 1) + ' ' + text + '\n')

    def para(self, text, small=False):
        paragraph = Paragraph(escape(text).replace('\n', '<br/>'), self.styles['SmallCN' if small else 'BodyText'])
        if self.pending_heading:
            self.story.append(KeepTogether(self.pending_heading + [paragraph]))
            self.pending_heading = []
        else:
            self.story.append(paragraph)
        self.markdown.append(text + '\n')

    def formula(self, text):
        self.story.append(Paragraph(escape(text), self.styles['FormulaCN']))
        self.markdown.append('`' + text + '`\n')

    def table(self, headers, rows, widths):
        cells = [[Paragraph(escape(str(cell)), self.styles['CellCN']) for cell in row]
                 for row in [headers] + rows]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8edf3')),
            ('GRID', (0, 0), (-1, -1), .45, colors.HexColor('#d9d9d9')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        # 本报告均为短表，整体保留，避免只剩一行移到下一页。
        self.story.append(KeepTogether(self.pending_heading + [table, Spacer(1, 9)]))
        self.pending_heading = []
        self.markdown.append('| ' + ' | '.join(headers) + ' |')
        self.markdown.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        self.markdown.extend('| ' + ' | '.join(str(c).replace('|', '/') for c in row) + ' |' for row in rows)
        self.markdown.append('')

    def save(self, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / 'MovieLens_RAG_报告.md').write_text('\n'.join(self.markdown), encoding='utf-8')
        document = SimpleDocTemplate(str(output / 'MovieLens_RAG_报告.pdf'), pagesize=A4,
                                     leftMargin=42, rightMargin=42, topMargin=40, bottomMargin=42,
                                     title='MovieLens电影推荐RAG课程报告', author='课程小组')
        def footer(canvas, doc):
            canvas.setFont(FONT_NAME, 8)
            canvas.setFillColor(colors.HexColor('#657080'))
            canvas.drawString(42, 24, 'MovieLens 100K / Qwen RAG / 验证集实验')
            canvas.drawRightString(A4[0] - 42, 24, str(doc.page))
        document.build(self.story, onFirstPage=footer, onLaterPages=footer)


def build(name='formal'):
    experiment = ROOT / 'outputs' / name
    config = read_json(experiment / 'config.json')
    evaluation = read_json(experiment / 'evaluation.json')
    failures = read_json(experiment / 'failure_cases.json')
    data = Path(config['data_dir'])
    manifest = read_json(data / 'split_manifest.json')
    cases = read_jsonl(data / 'validation_cases.jsonl')
    cases = [case for case in cases if case['user_id'] in config['user_ids']]
    labels = {row['user_id']: row for row in read_jsonl(data / 'validation_labels.jsonl')}
    with (data / 'movies.csv').open(encoding='utf-8-sig', newline='') as stream:
        movies = {int(row['movie_id']): row for row in csv.DictReader(stream)}
    with (data / 'train_ratings.csv').open(encoding='utf-8-sig', newline='') as stream:
        training = list(csv.DictReader(stream))
    results = {}
    for method in config['methods']:
        results[method] = []
        for uid in config['user_ids']:
            path = experiment / method / f'{uid}.json'
            if not path.exists():
                raise RuntimeError(f'正式结果未完成，禁止生成最终报告：{method}/{uid}.json')
            results[method].append(read_json(path))
    retrieval = [read_json(experiment / 'retrieval' / f'{uid}.json') for uid in config['user_ids']]
    summary = evaluation['summary']
    archived_errors = [read_json(path) for path in sorted((experiment / 'errors').glob('*/*.json'))]
    for method, rows in results.items():
        if sum(row.get('status', 'ok') == 'ok' for row in rows) != summary[method]['valid_user_count']:
            raise RuntimeError('推荐结果与evaluation快照不一致，请先运行evaluate再重建报告')
    count = len(cases)
    if count != 30 or len(config['user_ids']) != 30:
        raise RuntimeError('最终报告要求完整30用户正式实验')
    if len(failures) < 3:
        raise RuntimeError('少于3个真实失败用户，需要检查失败案例输出')
    metrics = ['recall_at_10', 'ndcg_at_10', 'hit_rate_at_10', 'precision_at_10', 'mrr_at_10']
    names = {'popular': '热门基线', 'rag': '完整RAG', 'no_retrieval': '移除检索'}
    report = Report()
    report.heading('MovieLens电影推荐RAG课程报告', 0)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    report.para(f'题目一 · 30位验证用户 · 生成日期 {stamp} · 实验标识 {config["fingerprint"]}', small=True)
    failed_generations = [(method, row['user_id'], row.get('error', '结构校验失败'))
                          for method in ('rag', 'no_retrieval') for row in results[method]
                          if row.get('status') != 'ok']
    pending_api = [(method, uid, error) for method, uid, error in failed_generations
                   if 'HTTP 403' in error or 'AccessDenied' in error]
    if pending_api:
        report.para(f'实验状态：API权限中断，待续跑。60个生成任务中已有{60-len(failed_generations)}个成功，'
                    f'{len(failed_generations)}个失败。以下指标按固定30用户分母、失败计零如实计算，属于当前运行快照；'
                    '恢复模型权限并仅重试错误任务后应重新生成报告，不能将此版作为完整成功实验结论。')
    else:
        report.para(f'实验状态：正式运行已完成。完整RAG生成{summary["rag"]["valid_user_count"]}/30份合法结果，'
                    f'移除检索生成{summary["no_retrieval"]["valid_user_count"]}/30份合法结果。'
                    '协议内重试后仍违反约束的输出保留为失败并计零，不反复采样直到成功。历史API权限中断已恢复，错误请求单列归档。')
    rag_score, popular_score = summary['rag']['ndcg_at_10'], summary['popular']['ndcg_at_10']
    compare = '高于' if rag_score > popular_score else '低于' if rag_score < popular_score else '等于'
    report.para(f'本项目使用教师提供的脚本生成MovieLens 100K学生开发集，搭建“训练历史偏好构建、电影证据检索、Qwen结构化推荐”的完整流程。'
                f'在相同30位用户及各自100部候选上，完整RAG的NDCG@10为{rag_score:.4f}，{compare}热门基线的{popular_score:.4f}。'
                '以下结果均来自保存的真实API响应与离线评测，不以模拟结果代替实验。')

    report.heading('1 任务与验收目标')
    report.para('给定用户训练历史与100部候选电影，输出10部历史未出现、互不重复的候选电影，按推荐程度排序。每条结果包含电影编号、排名、简短中文理由及理由依据。'
                '使用Qwen API，不训练或微调模型，不要求Agent。系统采用命令行入口，演示可直接读取真实结果，避免答辩时重复调用。')
    report.table(['教师要求', '本项目落实'], [
        ['热门基线', '训练集中评分>=4的次数降序，次数相同按电影ID升序'],
        ['RAG', '电影元数据与训练评分统计构成文档，语义检索、重排及训练近邻证据融合'],
        ['消融实验', '仅删除检索证据；保留相同训练历史、100候选、系统提示词与生成参数'],
        ['失败分析', '选取3个验证用户，检查漏推荐、偏好表达、检索覆盖、最终排序及理由依据'],
        ['共同提交', '可运行代码、中文报告、约10分钟presentation及运行说明'],
    ], [92, 419])

    report.heading('2 严格沿用教师数据生成方法')
    report.para('数据来源为GroupLens官方MovieLens 100K。scripts/prepare_movielens.py为教师附件的副本，未重写划分规则。'
                '执行student模式、固定seed=20261018，默认选择30位用户；官方压缩包通过附件要求的MD5校验。')
    report.formula('python run.py prepare')
    report.para('上述入口等价于执行附件脚本：python prepare_movielens.py --mode student --output data/movielens_dev --seed 20261018。'
                '若已有官方压缩包，可使用--zip指定，本机仍按脚本获取并核对官方MD5。')
    report.para('脚本先按(timestamp, movie_id)排列每位用户评分。仅选择至少20条记录、且最后5条中存在评分>=4电影的用户，按固定随机种子打乱并选30人。'
                '被选用户的最后5条作为留出，其余记录进入训练集；未被选用户的全部评分进入训练集。每位验证用户的100候选由其5部留出电影与95部从未评分电影组成，按用户独立随机种子打乱。'
                '相关电影仅指5部留出中评分>=4的电影，并非所有留出电影。')
    related = [len(labels[case['user_id']]['relevant_movie_ids']) for case in cases]
    histories = [len(case['history']) for case in cases]
    report.table(['核对项', '本机生成结果'], [
        ['电影数 / 训练用户数', f"{len(movies)} / {len({row['user_id'] for row in training})}"],
        ['训练评分数 / 留出评分数', f'{len(training):,} / {count * 5}'],
        ['验证用户 / 候选数 / 推荐数', f'{count} / 每用户100 / 每用户10'],
        ['训练历史长度', f'最少{min(histories)}，最多{max(histories)}，均值{sum(histories)/count:.1f}'],
        ['相关电影数', f'每用户{min(related)}至{max(related)}部，合计{sum(related)}部'],
        ['划分模式 / seed', f"{manifest['mode']} / {manifest['seed']}"],
        ['官方ZIP的MD5', manifest['zip_md5']],
    ], [156, 355])
    report.para('movies.csv和train_ratings.csv供检索模块使用；validation_cases.jsonl仅包含允许输入的信息。'
                'validation_labels.jsonl只在所有推荐结果生成后交给独立评测模块。索引、偏好、热门统计、近邻以及模型输入均不读取标签。'
                '未生成教师私有测试包。MovieLens缺少完整剧情和影评，本项目不补写剧情或引入外部电影资料。生成数据、包含电影数据的缓存和原始模型输入输出仅供本组本机开发，不上传公共仓库或再次分发。')

    report.heading('3 检索设计与证据来源')
    report.para('检索对象为当前100部候选中的电影文档。每篇文档含电影ID、名称、上映日期、类型，以及仅由训练评分计算的评分人数、均分和>=4分次数。'
                '程序按需向量化各用户候选，重复电影通过缓存复用，不先引入候选集外的电影。证据同时注明movies.csv与train_ratings.csv来源。')
    report.para('查询使用全部历史计算各类型的评分均值与样本数；额外选最多8部高分电影和5部低分电影作为正负例。高分按评分降序、时间降序及电影ID排序，低分按评分升序、时间降序及电影ID排序。'
                '这种压缩仅用于检索查询；Qwen生成阶段仍收到完整训练历史。')
    report.para('语义部分使用text-embedding-v4，1024维，按每批不超过10条请求。向量单位化后计算余弦相似度。缓存身份绑定模型、维度、服务地址、文本角色与文档文本；逐批保存，可中断续跑。'
                '重排部分使用qwen3-rerank，对全部100部未看候选文档评分，再参与融合，不能将其描述成仅对语义Top-30重排。')
    report.para('协同证据来自训练用户的评分向量：已评分电影取(rating-3)/2，缺失位置为0；计算目标历史向量与各训练用户向量的余弦相似度，排除目标用户，取相似度大于0的前40位。'
                '每部候选的协同分为近邻中心化评分的相似度加权和除以“相似度之和+2”，分母中的2用于收缩稀疏支持。每条证据展示最多5个真实近邻评分样例，但计算可使用40位近邻中对该片有评分者。')
    report.formula('cf(i) = sum(sim(u,v) * (rating(v,i)-3)/2) / (sum(sim(u,v)) + 2)')
    report.formula('score(i) = 0.45 * rerank(i) + 0.35 * (cf(i)+1)/2 + 0.20 * (cosine(i)+1)/2')
    report.para(f'按融合分降序、电影ID升序，返回前{config["top_k"]}条证据给Qwen。该权重是预先设定的工程配置，本次未通过验证标签反复搜索最优权重。'
                '协同分是预测信号，训练均分与近邻打分是实际统计，两者不能在推荐理由中混称。')

    report.heading('4 Qwen输入与结构化约束')
    model = config['models']
    report.table(['配置', '实际取值'], [
        ['生成模型', model['generation_model']], ['Embedding / Rerank', f"{model['embedding_model']} / {model['rerank_model']}"],
        ['temperature / seed', f"{model['temperature']} / {model['seed']}"],
        ['max_tokens / thinking', f"{model['max_tokens']} / {model['enable_thinking']}"],
        ['证据条数 / 生成并发', f"{config['top_k']} / {config.get('generation_workers', 1)}"],
        ['输出模式', 'JSON对象；每条电影ID、rank、reason、evidence_ids'],
    ], [156, 355])
    report.para('用户消息包含user_id、完整history、原顺序100部candidates，以及retrieved_evidence。候选、历史、检索证据分别以candidate_电影ID、history_电影ID和movie_电影ID引用。'
                '以下为保存到正式调用中的系统提示词，未省略约束：')
    report.para(SYSTEM_PROMPT, small=True)
    report.para('机器校验要求：恰好10条、电影ID为整数、候选内、历史外、不重复、rank为连续1至10、理由非空、每条必须引用自身候选，且每个引用ID确实存在。'
                '通过校验后将证据ID展开为原始证据对象，便于复查。格式校验失败时仅反馈具体错误并最多重试一次，不悄悄用热门基线替代失败。网络失败会保留错误状态，评测中计零。')
    report.para('机器引用合法并不保证自然语言理由正确：例如引用了一个真实history编号，却把其低分说成高分，仍可能通过结构校验。'
                '剧情、类型归纳、“多次喜欢”等表述需要将理由逐项对照实际评分、类型和证据内容。本报告的失败案例保留原话及事实摘录，不能把100%结构合法率写成100%事实正确率。')
    rendered_count = sum('model_reason' in rec for method in ('rag', 'no_retrieval')
                         for result in results[method] for rec in result.get('recommendations', []))
    if rendered_count:
        report.para('联调中发现了真实的语义错误：用户20的原始首条理由将Twister与ID4统称为高分电影，但history_118实际为4分、history_121实际为3分。'
                    '最终交付因此增加确定性理由渲染：Qwen仍决定推荐编号、顺序及引用选择，交付reason只使用真实候选类型、引用历史的精确评分/共同类型与可核实训练统计构造；'
                    '优先选择模型已引用的同类型高分历史，若没有则从完整训练历史选择一条真实支持；若没有同类型高分依据，则明确作为探索项。'
                    '原始自然语言保留为model_reason，原始响应也保留在attempts中。完整RAG和移除检索应用同一后处理，不改推荐电影或排序，因此不改变排序指标。'
                    '这项输出事实约束提高已展示事实的可核查性，不等价于证明偏好推断正确。')

    report.heading('5 实验协议与评价指标')
    report.para('三种方法使用同一30位用户、同一训练划分、相同的100候选及同一相关性标签。热门基线只用训练统计；完整RAG包含历史、候选、检索证据；消融仅移除retrieved_evidence。'
                '完整RAG与消融的系统提示词、完整历史、候选顺序、模型、temperature、seed和max_tokens均相同。该消融测量整包检索证据的效果，不能单独证明Embedding、Rerank或协同分各自的贡献。')
    report.para('指标对用户宏平均。Recall@10为命中相关电影数除以该用户相关电影数；Precision@10为命中数除以10；HitRate@10为是否至少命中1部；MRR@10为首部相关电影排名的倒数。'
                'NDCG@10使用二元相关性，按1/log2(rank+1)折扣，并以同一用户理想排序归一化。缺失、失败或格式非法结果整条计零，不只评测成功用户；空相关集排除并单独报告。')
    report.formula('NDCG@10 = [sum(rel(rank)/log2(rank+1), rank=1..10)] / IDCG@10')
    report.table(['方法', 'Recall@10', 'NDCG@10', 'HitRate@10', 'Precision@10', 'MRR@10'],
                 [[names[method]] + [number(summary[method][metric]) for metric in metrics]
                  for method in config['methods']], [79, 85, 85, 85, 91, 86])
    report.table(['方法', '评测用户', '结构合法', '失败或非法', '空集/非法标签排除'],
                 [[names[method], summary[method]['metric_user_count'],
                   f"{summary[method]['valid_user_count']}/{summary[method]['user_count']}",
                   summary[method]['failed_or_invalid_user_count'], summary[method]['excluded_label_user_count']]
                  for method in config['methods']], [95, 85, 100, 100, 131])
    delta = summary['rag']['ndcg_at_10'] - summary['no_retrieval']['ndcg_at_10']
    direction = '增加' if delta >= 0 else '减少'
    report.para(f'移除检索时NDCG@10为{summary["no_retrieval"]["ndcg_at_10"]:.4f}；加入检索后{direction}{abs(delta):.4f}。'
                f'完整RAG相对热门基线的Recall@10差值为{summary["rag"]["recall_at_10"]-summary["popular"]["recall_at_10"]:+.4f}，'
                f'NDCG@10差值为{rag_score-popular_score:+.4f}。这些是本次30用户验证集上的观察，不报告未做的显著性检验或教师私有测试表现。')
    if pending_api:
        report.para('当前RAG及消融各有接口失败用户，热门基线没有API可用性限制。全用户指标同时反映模型推荐效果与接口失败，'
                    '因此本版不能仅凭RAG低于热门就归因于检索或偏好建模；完整实验结论待权限恢复后续跑确认。')
    elif failed_generations:
        report.para('消融保留1个输出约束失败用户并计零，完整RAG无失败。该差异同时包含候选排序与指令遵循的影响；'
                    '本报告不通过删除失败用户或反复抽样来提高消融成功率，也不把所有指标差距仅归因于检索排序质量。')

    report.heading('6 三个失败用户的证据分析')
    report.para('推荐失败定义为输出合法10条推荐但至少漏掉1部相关电影。接口或格式错误单独归入服务可用性，不占用这3个案例名额。'
                '案例优先选择RAG相对热门基线NDCG下降较大的用户，再按RAG NDCG和用户ID排序。'
                '检索未覆盖意味着相关电影不在交给模型的30条增强证据中，仍可能出现在100候选元数据里；因此“未检索”不等于模型没有任何机会推荐。')
    result_by_uid = {row['user_id']: row for row in results['rag']}
    case_by_uid = {case['user_id']: case for case in cases}
    for index, failure in enumerate(failures[:3], 1):
        uid = failure['user_id']
        row = result_by_uid[uid]
        case = case_by_uid[uid]
        fmetrics, bmetrics = failure['metrics'], failure['baseline_metrics']
        report.heading(f'6.{index} 用户{uid}', 2)
        report.para(f'历史{len(case["history"])}条，相关电影{len(labels[uid]["relevant_movie_ids"])}部。'
                    f'RAG Recall@10={fmetrics["recall_at_10"]:.4f}、NDCG@10={fmetrics["ndcg_at_10"]:.4f}；'
                    f'热门基线对应值为{bmetrics["recall_at_10"]:.4f}、{bmetrics["ndcg_at_10"]:.4f}。')
        report.para('实际推荐顺序：' + ', '.join(map(str, failure['selected_movie_ids'])) + '。')
        missed_rows = [[movie['movie_id'], movie['title'], ', '.join(movie.get('genres', []))]
                       for movie in failure['missed_movies']]
        report.table(['漏掉的相关ID', '电影名称', '类型'], missed_rows, [93, 263, 155])
        profile = failure['history_profile']
        liked = list(profile['liked_genre_counts'].items())[:5]
        disliked = list(profile['disliked_genre_counts'].items())[:4]
        report.para('偏好证据：历史>=4分电影中的主要类型计数为' + '、'.join(f'{g}:{n}' for g, n in liked)
                    + '；<=2分主要类型计数为' + ('、'.join(f'{g}:{n}' for g, n in disliked) or '无')
                    + '。多类型电影分别计入各类型；高低分可以出现相同类型，因此类型频率本身不足以证明偏好判断正确。')
        stage = failure['retrieval_analysis']
        retrieval_ranks = {ev['movie_id']: rank for rank, ev in enumerate(row.get('retrieval', {}).get('evidence', []), 1)}
        selected_genres = Counter(genre for rec in row.get('recommendations', [])
                                  for genre in next(movie['genres'] for movie in case['candidates'] if movie['movie_id'] == rec['movie_id']))
        report.para('最终10部推荐的主要类型计数：' + '、'.join(f'{genre}:{n}' for genre, n in selected_genres.most_common(4))
                    + '。漏推荐电影在增强证据中的名次：'
                    + '、'.join(f"{movie['movie_id']}第{retrieval_ranks[movie['movie_id']]}位" if movie['movie_id'] in retrieval_ranks else f"{movie['movie_id']}未进入前30"
                               for movie in failure['missed_movies']) + '。')
        unsupported = [movie for movie in failure['missed_movies']
                       if not any(profile['liked_genre_counts'].get(genre, 0) for genre in movie['genres'])]
        if unsupported:
            report.para('偏好表示的具体局限：' + '；'.join(f"{movie['movie_id']}的类型{','.join(movie['genres'])}在高分训练历史中均未出现" for movie in unsupported)
                        + '。这使基于历史类型的正向解释缺少支持，说明现有偏好表示难以覆盖这些实际留出兴趣；不能由此断定用户真实不喜欢这些类型。')
        if not failure['valid']:
            report.para('该用户结果未通过调用或结构校验，指标按协议计零。实际错误：'
                        + str(row.get('error', failure['issues']))
                        + '。已有检索证据中的相关电影为' + movies_text(stage['retrieved_relevant_movie_ids'], movies)
                        + '。接口/格式失败没有可用的最终前10排序，不能把这些电影全部归因为偏好判断或生成排序错误。')
        else:
            report.para('检索覆盖损失：' + movies_text(stage['missed_not_retrieved'], movies)
                        + '。生成选择损失（已有增强证据但未进前10）：' + movies_text(stage['missed_after_retrieval'], movies) + '。'
                        '这一分解可定位发生损失的环节；缺少反事实实验时，不能断言单一类型偏好或模型推理错误就是主因。')
        retrieval_rows = {ev['movie_id']: ev for ev in row.get('retrieval', {}).get('evidence', [])}
        missed_score_rows = []
        for mid in stage['missed_after_retrieval']:
            ev = retrieval_rows[mid]
            missed_score_rows.append([mid, number(ev['score']), number(ev['dense_score']),
                                      number(ev['collaborative_score']), number(ev['rerank_score'])])
        if missed_score_rows:
            report.table(['已有证据ID', '融合分', '余弦', '协同分', '重排分'], missed_score_rows, [100, 103, 103, 103, 102])
        for rec in row.get('recommendations', [])[:2]:
            report.para(f'理由核查样例，rank={rec["rank"]}、电影{rec["movie_id"]}：{rec["reason"]}')
            if 'model_reason' in rec:
                report.para('Qwen原始理由（保留用于分析，不作为已核验事实）：' + rec['model_reason'], small=True)
                raw_payload = json.loads(row['attempts'][-1]['response']['choices'][0]['message']['content'])
                raw_rec = next(item for item in raw_payload['recommendations'] if item['movie_id'] == rec['movie_id'])
                _, raw_sources = build_messages(case, row['retrieval'])
                raw_histories = [raw_sources[ref] for ref in raw_rec['evidence_ids'] if ref.startswith('history_')]
                if raw_histories:
                    report.para('原始理由实际引用的历史：' + '；'.join(f"{item['evidence_id']}《{item['title']}》={item['rating']}分" for item in raw_histories)
                                + '。这组原始事实用于核查模型原话；以下交付证据可能因确定性渲染重新选择，二者应分别核对。', small=True)
            cited = []
            for ev in rec['evidence']:
                eid = ev['evidence_id']
                if eid.startswith('history_'):
                    cited.append(f"{eid}实际评分{ev['rating']}，类型{','.join(ev['genres'])}")
                elif eid.startswith('candidate_'):
                    cited.append(f"{eid}类型{','.join(ev['genres'])}")
                else:
                    cited.append(f"{eid}训练高分次数{ev.get('positive_rating_count','未提供')}，近邻样例{len(ev.get('neighbor_ratings', []))}条")
            report.para('可直接核对的事实：' + '；'.join(cited) + '。引用存在性已由程序检查；需按这组事实逐句核对原理由是否把低分说成喜欢、是否扩大类型结论。', small=True)

    report.heading('7 调用量 Token与耗时')
    if failed_generations:
        report.para('最终输出可用性：当前失败任务如下。若输出在协议内两次生成后仍违反候选或历史约束，'
                    '程序拒绝交付并按零分纳入固定用户分母，保留两次原始响应。它与历史403服务权限错误分别记录。')
        report.table(['方法', '失败用户', '实际错误类型'],
                     [[names[method], ', '.join(str(uid) for m, uid, _ in failed_generations if m == method),
                       'HTTP403 AccessDenied.Unpurchased' if all('AccessDenied.Unpurchased' in err for m, _, err in failed_generations if m == method)
                       else '详见逐用户错误日志']
                      for method in ('rag', 'no_retrieval') if any(m == method for m, _, _ in failed_generations)],
                     [90, 261, 160])
        for method, uid, error in failed_generations:
            result = next(row for row in results[method] if row['user_id'] == uid)
            case = case_by_uid[uid]
            candidate_ids = {movie['movie_id'] for movie in case['candidates']}
            history_ids = {movie['movie_id'] for movie in case['history']}
            rejected = []
            for attempt_no, attempt in enumerate(result.get('attempts', []), 1):
                payload = json.loads(attempt['response']['choices'][0]['message']['content'])
                invalid = [f"{rec.get('movie_id')}（" + '/'.join(
                    label for condition, label in [(rec.get('movie_id') not in candidate_ids, '候选外'),
                                                     (rec.get('movie_id') in history_ids, '历史内')] if condition) + '）'
                           for rec in payload.get('recommendations', [])
                           if rec.get('movie_id') not in candidate_ids or rec.get('movie_id') in history_ids]
                rejected.append(f"第{attempt_no}轮：" + ('、'.join(invalid) or attempt.get('validation_error', '见原始记录')))
            report.para(f'{names[method]}用户{uid}，实际错误为“{error}”。' + '；'.join(rejected)
                        + '。这是约束遵循失败，不能当作一个已输出合法前10的推荐失败案例。')
    if archived_errors:
        archived_calls = [call for result in archived_errors for call in result.get('calls', [])]
        historical_errors = usage(archived_calls)
        statuses = Counter(str(call.get('http_status', call.get('error', '未知'))) for call in archived_calls)
        report.para(f'服务恢复记录：此前{len(archived_errors)}个生成任务因模型访问权限中断，失败日志已保留在errors归档。'
                    '权限恢复后仅重试失败任务，已成功的推荐编号与排序继续复用；归档错误不占用最终推荐失败案例名额。'
                    f'归档共有{historical_errors["calls"]}次请求，HTTP状态分布{dict(statuses)}，累计请求耗时{historical_errors["seconds"]:.1f}秒。'
                    '这些错误响应未提供usage，不能据此编造Token或费用；以下主表只统计最终逐用户结果和检索日志，归档失败开销单列于此。')
    generation_usage = {}
    for method in ('rag', 'no_retrieval'):
        generation_usage[method] = usage([call for result in results[method] for call in result.get('calls', [])])
    retrieval_calls = [call for record in retrieval for call in record.get('calls', [])]
    embedding_usage = usage([call for call in retrieval_calls if call['kind'] == 'embedding'])
    rerank_usage = usage([call for call in retrieval_calls if call['kind'] == 'rerank'])
    cost_rows = []
    for label, used in [('RAG生成', generation_usage['rag']), ('消融生成', generation_usage['no_retrieval']),
                        ('本次检索Embedding', embedding_usage), ('本次检索Rerank', rerank_usage)]:
        cost_rows.append([label, used['calls'], used['input_tokens'] if used['input_reported'] else '未提供',
                          used['output_tokens'] if used['output_reported'] else '不适用/未提供',
                          used['total_tokens'], f"{used['seconds']:.1f}"])
    report.table(['阶段', '请求数', '输入Token', '输出Token', '总Token', '累计请求秒'], cost_rows, [114, 57, 88, 80, 88, 84])
    report.para('Token来自API返回的usage，输入输出字段按接口命名归一；仅计本次正式结果中保存的调用。API只提供total_tokens而未提供输入/输出拆分时明确标为未提供，不以0代替。'
                '表中秒数是各次HTTP请求耗时之和，并非并行实验墙钟时间；不据此套用未核实单价估算费用。热门基线不调用模型。')
    rag_seconds = sum(row.get('seconds', 0) for row in results['rag'])
    ablation_seconds = sum(row.get('seconds', 0) for row in results['no_retrieval'])
    retrieval_seconds = sum(row.get('seconds', 0) for row in retrieval)
    report.para(f'每用户处理耗时累计：RAG生成{rag_seconds:.1f}秒（均值{rag_seconds/count:.1f}秒），'
                f'消融生成{ablation_seconds:.1f}秒（均值{ablation_seconds/count:.1f}秒），检索{retrieval_seconds:.1f}秒。'
                '这些值包括相应阶段的本地开销及可能的重试等待；各用户生成按3个worker并行，不能将累计值解释成实际等待时长。')
    runtime_file = experiment / 'runtime.json'
    if runtime_file.exists():
        runtime = read_json(runtime_file)
        report.para('另行保存的运行计时：' + json.dumps(runtime, ensure_ascii=False) + '。以记录的字段及起止范围解释墙钟，断点续跑不得当作全新冷启动。')
    else:
        report.para('本次正式实验没有单独保存完整进程起止墙钟，因此不报告推算的并行总时长。复现实验如需该值，应在启动和结束处另行计时。')
    all_embedding_calls = []
    seen_requests = set()
    for path in sorted((ROOT / 'outputs').glob('*/retrieval/*.json')):
        for call in read_json(path).get('calls', []):
            request_id = call.get('request_id')
            if call.get('kind') == 'embedding' and request_id and request_id not in seen_requests:
                seen_requests.add(request_id)
                all_embedding_calls.append(call)
    historical = usage(all_embedding_calls)
    cache_entries = sum(len(read_json(path)) for path in (ROOT / 'cache').glob('embeddings_*.json'))
    report.para(f'向量缓存目前保存{cache_entries}个文本角色条目。正式实验前已进行小规模联调，正式调用可复用暖缓存；'
                f'已保存各实验检索日志中去重后共有{historical["calls"]}次Embedding请求、{historical["total_tokens"]:,}个Token。'
                '该数是可追溯的缓存建设记录，不包含未落盘的API检查或网络中断记录，不能宣称为完整冷启动账单。RAG与消融对比不重复计向量缓存的一次建设成本。')

    report.heading('8 结果讨论与改进')
    report.para('热门推荐在这一候选协议下是有竞争力的基线：训练评分多的电影更可能被留出用户评分，单纯按类型匹配可能高估冷门候选。'
                'RAG增加了训练统计和近邻依据，但电影元数据只有类型、名称及上映信息，难以表达节奏、叙事风格等细粒度偏好；语义相关也不直接等于用户喜欢。')
    report.para('本次消融只检验整体检索内容，未分离重排、协同信号、证据条数、权重及提示词各自贡献。30用户验证集较小，且只使用一次固定划分；生成温度设为0和固定seed也不能保证服务端完全逐位复现。'
                '95部未评分候选按课堂协议作为非相关项，这不等于用户真实不喜欢它们。当前结果不能直接外推到全库推荐或线上点击收益。')
    report.para('优先改进方向是增加独立开发划分后再选择融合权重，分别做移除近邻和移除重排实验，统计检索Recall@30，并对推荐理由逐条标注“事实正确、引用充分、偏好推断合理”。'
                '这些是后续建议，本报告不将未完成实验描述为已实现提升。')

    report.heading('9 复现路径与八人协作建议')
    report.para('在项目目录安装requirements.txt，按.env.example填写自己的百炼Key及地域地址；密钥不写入源代码、报告或提交包。'
                '依次运行python run.py prepare、python run.py check-api、python run.py experiment --name formal --workers 3，'
                '即可完成数据、接口验证及正式实验。python run.py evaluate --name formal只重算指标；python run.py demo --user 用户编号读取已保存真实结果。'
                '报告使用支持reportlab的Python执行python scripts/build_report.py --name formal重建。')
    report.para('以下为后续协作及答辩分配建议。成员姓名待小组填写，不代表对真实个人贡献的认证；提交前应按实际参与情况修改。')
    report.table(['成员', '建议承担', '报告与汇报内容'], [
        ['成员1、2', '数据准备、数据审计及热门基线', '解释最后5条留出、95负候选、相关性阈值与热门规则'],
        ['成员3、4', '向量检索、协同证据及重排', '说明文档、查询、融合公式与来源'],
        ['成员5、6', 'Qwen生成、提示词、约束与集成', '演示结构化结果及引用，说明机器校验局限'],
        ['成员7、8', '评测、消融、失败分析与材料整理', '展示真实指标、失败用户与反思'],
    ], [83, 184, 244])
    report.para('正式结果路径：outputs/formal/config.json、evaluation.json、failure_cases.json；每用户的原始响应、提示词和调用usage位于rag/与no_retrieval/，检索日志位于retrieval/。'
                '项目提交包应排除.env、原始与派生MovieLens数据、向量缓存以及含电影数据的原始模型日志；由接收者从官方地址在本机重建。')

    report.heading('10 参考资料与致谢')
    report.para('感谢GroupLens提供MovieLens研究数据。F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. '
                'ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. DOI: 10.1145/2827872。')
    report.para('MovieLens 100K官方主页：https://grouplens.org/datasets/movielens/100k/\n'
                '官方数据说明：https://files.grouplens.org/datasets/movielens/ml-100k-README.txt\n'
                '百炼Embedding与Rerank说明：https://help.aliyun.com/en/model-studio/embedding-rerank-model/\n'
                '生成模型说明：https://help.aliyun.com/en/model-studio/qwen3-8-27b\n'
                '课程依据：教师提供的《26年秋课程大作业：电影推荐与数据分析》及prepare_movielens.py。', small=True)
    report.save(ROOT / 'deliverables')
    print(json.dumps({'report': str(ROOT / 'deliverables' / 'MovieLens_RAG_报告.pdf'),
                      'users': count, 'methods': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='formal')
    build(parser.parse_args().name)

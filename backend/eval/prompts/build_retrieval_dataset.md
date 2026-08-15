# 构建 Retrieval 数据集提示词

请继续构建本项目的 RAG retrieval 评测数据集。请直接检查代码、Markdown、现有数据集和 Milvus 后执行，不要只给方案或问题列表。

项目目录：`E:\pythonprojects\fund_analyser`

## 目标与边界

`retrieval.jsonl` 只评测两级 RAG 的第二级：在已经确定正确 `filter_fund_code` 的前提下，是否从**该单只基金的一份报告**中召回并排序正确 chunk。

- 输出：`backend/eval/datasets/retrieval.jsonl`
- 在现有 `backend/eval/runners/build_retrieval_dataset.py` 上增量修改，不要创建功能重复的脚本。
- 每条样本只能对应一只基金和一份 `_analyzed.md` 报告。
- 每条必须有 `filter_fund_code`；它与 `expected_fund_codes` 必须都是同一个基金代码。
- `relevant_chunk_ids` 只能来自该基金的 Milvus chunk。
- 跨基金比较、主题/板块范围筛选、"有哪些基金持有某股票"等多报告问题不属于本数据集，应构建到 `answer_cross_fund.jsonl`。

## 先检查上下文

1. 检查 git/worktree 状态，保留已有用户修改，不得 reset、checkout、覆盖或回退无关改动。
2. 阅读：
   - `backend/eval/schemas.py`
   - `backend/eval/datasets/retrieval.jsonl`
   - `backend/eval/runners/build_retrieval_dataset.py`
   - `backend/eval/runners/query_fund_report.py`
   - `vectorize/vectorize_to_milvus.py`
3. 读取 `annual_reports_2025_funds/_pdf_review.json`。不得使用 `code_mismatches` 或 `extraction_issues` 中的报告；SHA256 重复本身不是排除条件。
4. 核对 Markdown 文件名与正文、Milvus `fund_code`/`file_path`、PDF 校验结果。发现冲突时先定位数据问题。

## 报告与问题构思

1. 每批处理后续 10 份尚未构建的正常报告；每份 10 条，共 100 条。沿用现有数据集的排序、ID 和去重逻辑。
2. 从 `markdown_mineru/*/*_analyzed.md` 的实际正文构思问题，再逐条到 Milvus 查询相应 chunk；不能只根据目录名、基金名称或常识编题。
3. 同一报告的 10 个问题覆盖不同章节，优先涵盖：
   - 基金基本信息、成立/上市、基金代码
   - 指数、复制方法、跟踪误差
   - 基金经理或人员变更
   - 净值、业绩与基准
   - 市场回顾、投资策略和运作
   - 资产/行业配置、股票或债券持仓
   - 持有人结构、风险、费用、关联交易或重大事项
4. 问题必须自然、具体、可由标注 chunk 明确回答；数值必须核对报告期、单位和上下文。避免仅替换基金名或数字形成模板化重复。

## Milvus Ground Truth

Milvus：`http://localhost:19595`；collection：`fund_reports_mineru`。

1. 必须查询在线 Milvus，使用真实 `id`、`fund_code`、`file_path`、`chunk_index` 和 `content`。
2. 不得根据 Markdown 推测或编造 chunk ID；`chunk_index` 仅用于定位和 `note`，不能代替 `id`。
3. 只标注足以直接支持问题答案的最少 chunk；多章节事实才使用多个正例。
4. 不得因关键词相同就标为正例，不得混用其他基金 chunk。
5. `relevant_keywords` 必须是正例 `content` 内的原始子串，并足以支持相关性判断。
6. `note` 记录基金代码、主题、实际 `chunk_index`、章节和 Milvus ID。
7. 缺关键词、chunk 不存在或 metadata 不一致时，使用 `query_fund_report.py` 查询实际 content 后修正样本，不要绕过校验。

每行保持以下字段风格：

```json
{
  "id": "retrieval-001",
  "query": "...",
  "filter_fund_code": "159103",
  "top_k": 10,
  "expected_fund_codes": ["159103"],
  "relevant_chunk_ids": ["真实 Milvus id"],
  "relevant_keywords": ["正例原文短语"],
  "category": "single_fund",
  "note": "主题；chunk=... 章节"
}
```

## 生成与验证

运行构建脚本后独立验证：

1. JSONL 每行合法，ID 连续，历史顺序未破坏，query 无重复。
2. 本批恰好新增 100 条，覆盖 10 份正常报告，每份恰好 10 条。
3. 不包含 PDF 校验异常基金。
4. 每条 `filter_fund_code == expected_fund_codes[0]`，且只有一个期望基金代码。
5. 所有正例 ID 可在 Milvus 查询到，`fund_code` 与过滤代码一致，`file_path` 指向该基金的 `_analyzed.md`。
6. 每个关键词在对应正例正文中存在，问题与正例语义匹配，且没有跨基金或不相关 chunk。
7. 构建脚本通过语法检查并可重复运行。

完成后报告：本批报告、基金代码和名称；历史/新增/累计条数；实际验证结果；处理过的一致性问题；修改文件列表。

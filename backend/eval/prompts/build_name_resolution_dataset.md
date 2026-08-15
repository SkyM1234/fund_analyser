# 构建基金名称识别数据集提示词

请继续构建本项目的基金名称识别评测数据集。请直接检查代码、Markdown 和现有数据后执行，不要只给方案或问题列表。

项目目录：`E:\pythonprojects\fund_analyser`

## 目标与边界

`fund_name_resolution.jsonl` 评测两级 RAG 的第一级 `rag_identify_funds`：用户给出基金名称、别名、报告标题或描述时，系统能否识别正确基金代码。该数据集不评测 Milvus 报告 chunk 检索，因此不包含 `relevant_chunk_ids` 或 `filter_fund_code`。

- 输出：`backend/eval/datasets/fund_name_resolution.jsonl`
- 在 `backend/eval/runners/build_name_resolution_dataset.py` 上增量修改，不要创建功能重复的脚本。
- 先阅读 `backend/eval/schemas.py`、现有 JSONL、构建脚本，以及 `rag_identify_funds` 的实现和评测器。

## 数据来源与样本设计

1. 从 `markdown_mineru` 的报告目录、`_analyzed.md` 一级标题和报告正文提取真实的基金名称变体。
2. 保留并覆盖已有脚本的标题类样本：
   - `short_name`：报告目录中的基金短名
   - `full_name`：完整基金名称
   - `short_report_title`：短名加年度报告后缀
   - `markdown_title`：`_analyzed.md` 的一级标题
   - `source_title`：报告目录标题
3. 在不与其他基金混淆的前提下，可增补真实、可验证的别名和简称，类别使用 `alias` 并在 `note` 说明来源。
4. 另行补充有代表性的：
   - `vague_description`：仅在当前系统确有可验证、唯一答案时使用
   - `no_match`：无对应基金、无意义名称或明确不应命中的输入，`expected_fund_code` 为 `null`
5. 不得凭常识虚构简称、基金关系、主题描述或不存在的基金；名称变体必须能回溯到报告目录、Markdown 正文、业务配置或已验证的名称映射来源。
6. 对同名、近似名或无法唯一确定的输入，不要标为某只基金的正确命中；改为跳过、设计成 no-match，或在评测逻辑支持时标记为歧义。

每行符合 `NameResolutionExample`：

```json
{
  "id": "name-res-001",
  "query": "港股通科技ETF华夏",
  "expected_fund_code": "159101",
  "category": "short_name",
  "note": "目录中的基金短名"
}
```

## 校验

1. JSONL 每行合法；ID 连续且无重复；query 全局无重复。
2. 每个非空 `expected_fund_code` 都可由源报告目录、Markdown 标题或已验证映射追溯。
3. 标题类样本每份正常报告覆盖完整；新增 alias、vague_description、no_match 的类别和数量清晰可审阅。
4. 不使用 `annual_reports_2025_funds/_pdf_review.json` 中 `code_mismatches` 或 `extraction_issues` 的异常报告。
5. 运行构建脚本并通过 Python 语法检查；重复运行不产生不稳定排序、重复 query 或 ID 变化。

完成后报告：覆盖报告数、各 category 的样本数、增补别名/no-match 的来源、验证结果及修改文件列表。

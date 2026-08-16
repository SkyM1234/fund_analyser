# 构建单基金 Answer 数据集提示词

请继续构建本项目的 answer 评测数据集。请直接检查代码、修改必要文件、生成数据并完成验证，不要只给方案或问题列表。

项目目录：`E:\pythonprojects\fund_analyser`

## 目标与边界

`answer_single_fund.jsonl` 用于单基金 Agent 回答评测。该数据集只包含能够由一只基金的一份或多份报告回答的问题，

当前主线为：

```text
intent = "single_fund_query"
category = "single_fund_strategy"
```

跨基金、板块和多基金比较问题使用`answer_cross_fund.jsonl`，构建规则见`build_cross_fund_answer_dataset.md`。不要把跨基金问题放入本数据集，也不要把这类问题放入默认 `retrieval.jsonl`。

## 先检查上下文

1. 检查 git/worktree 状态，保留已有用户修改，不得 reset、checkout 或覆盖无关改动。
2. 阅读：
   - `backend/eval/README.md`
   - `backend/eval/schemas.py`
   - `backend/eval/datasets/answer_single_fund.jsonl`
   - `backend/eval/datasets/retrieval.jsonl`
   - `backend/eval/runners/build_single_fund_answer_dataset.py`
   - `backend/eval/runners/run_answer_eval.py`
   - `backend/eval/targets/service_target.py`
   - `backend/eval/evaluators/answer_metrics.py`
   - `backend/app/agent/rag_agent.py`
   - `backend/app/services/rag_result_parser.py`
   - `backend/mcp/rag-mcp/src/server.py`
   - `vectorize/vectorize_to_milvus.py`
3. 优先在既有构建脚本上增量修改；已实现的功能先验证，避免创建重复脚本。
4. 读取 `annual_reports_2025_funds/_pdf_review.json`，排除 `code_mismatches` 和 `extraction_issues` 中的报告。

## 单基金样本

### 分类规则

- 所有样本的 `intent` 必须是 `single_fund_query`。
- 所有样本的 `category` 必须是 `single_fund_strategy`。
- query 必须明确指向一只基金。
- query 不能同时指向多只基金、泛泛询问基金类别，或要求扫描全部候选基金。
- query 必须能够由该基金的一份或多份报告回答；不能只依赖常识、基金名称或相邻数据推断答案。

### 候选与 query 构建

1. 优先使用 `retrieval.jsonl` 中已经验证的单基金问题。
2. 优先选择 query 中直接出现六位基金代码、完整基金名称，或经过核验且能唯一映射到基金的简称/别名的样本。
3. 如果需要从 `markdown_mineru` 报告构思新 query，必须先阅读报告正文中的真实字段，再到 Milvus 查询对应 chunk；不能只根据基金名称、主题标签或常识出题。
4. query 无法唯一确定基金时跳过，并记录跳过原因。
5. 发现候选样本质量不合格时，换用下一个候选。
6. 已有 answer 样本按既有 ID、顺序和去重逻辑增量维护，不得删除其他有效样本。

### 字段与事实标注

每行使用 `AnswerExample` 字段，单基金样本至少包含：

```json
{
  "id": "answer-001",
  "query": "...",
  "reference_answer": "...",
  "expected_fund_codes": ["159103"],
  "key_facts": ["..."],
  "should_refuse": false,
  "intent": "single_fund_query",
  "category": "single_fund_strategy",
  "relevant_keywords": ["..."],
  "relevant_chunk_ids": ["真实 Milvus id"],
  "expected_tool_calls": [],
  "note": "来源 retrieval-...；基金；chunk=... id=..."
}
```

1. `reference_answer` 必须严格依据 `relevant_chunk_ids` 的正文撰写，不得用常识补充报告中没有的事实。
2. `expected_fund_codes`、query 和相关 chunk 必须指向同一只基金。
3. `key_facts` 应是答案必须出现的事实、数字、指数、策略或风险，不能是无判别力的泛化词。
4. `relevant_keywords` 必须是相关 chunk 原文中存在的短语，且能够证明答案中的关键结论。
5. `note` 应包含 retrieval 来源、基金、问题主题、`chunk_index` 和真实 Milvus ID。

## Milvus Ground Truth

Milvus：`http://localhost:19595`；collection：`fund_reports_mineru`。

1. 对每个候选问题直接查询在线 Milvus；`relevant_chunk_ids` 必须是存在的主键 `id`，不得根据 Markdown 推测。
2. 核对每个相关 chunk 的 `fund_code`、`file_path`、`chunk_index` 和 `content`。
3. `file_path` 必须对应正确的 `markdown_mineru` 下的 `_analyzed.md` 文件。
4. 仅选择直接支持 `reference_answer` 和 `key_facts` 的最少 chunk；不要使用只共享关键词但不能证明结论的 chunk。
5. `relevant_keywords` 必须来自对应 chunk 正文的原始短语。
6. `note` 可以使用 `chunk_index` 和章节信息辅助审阅，但不能用它们代替真实 Milvus ID。

## 工具调用和服务评测

1. `expected_tool_calls` 必须以当前 Agent、MCP 和评测器的实际比较语义为准；先检查实现，再写入能够稳定验证的关键调用。
2. query 未直接包含六位基金代码时，必须按顺序标注：

```json
[
  {"name": "rag_identify_funds"},
  {"name": "rag_search", "args": {"filter_fund_code": "最终确定的基金代码"}}
]
```

3. query 直接包含六位基金代码时，只标注：

```json
[
  {"name": "rag_search", "args": {"filter_fund_code": "基金代码"}}
]
```

4. 六位代码检测不能使用仅适用于英文单词边界的实现，必须识别如 `159128报告期内` 这类与中文直接相连的写法。`rag_identify_funds` 仅要求工具名，不能比较其 args，因为 Agent 可能改写识别 query。
5. answer 评测必须通过已启动 Docker 服务的完整链路运行。`service_target` 应从 `retrieval_context` SSE 事件收集实际 chunk ID，不得从最终回答反推或伪造。
6. 不得新增或保留 `--target local-agent` 作为 answer 评测目标；不要为评测重置 MCP 统计、禁用限流或改用 local-agent，除非用户明确要求。
7. `--concurrency` 是同时执行的评测样本数，服务端应提供相应 worker/并发槽位；它不是单条样本内的工具调用数。

## 生成与验证

运行现有构建脚本，并验证：

1. JSONL 合法，ID 连续且不重复，query 无重复。
2. 所有 `relevant_chunk_ids` 都能在 Milvus 查询到，且属于 `expected_fund_codes` 指定的基金。
3. 所有 `relevant_keywords` 都存在于对应 chunk 正文中。
4. `reference_answer` 和 `key_facts` 都能由已标注 chunk 支持，不得引入报告之外的事实。
5. 每个样本只指向一只基金；未含六位代码的样本使用 `rag_identify_funds -> rag_search`，含代码的样本只使用 `rag_search`。
6. 构建脚本通过语法检查且可重复运行，不会重复追加或破坏既有数据。
7. 不得出现 local-agent target；服务评测应能使用 SSE 收集的检索上下文计算 Agentic RAG 的 hit rate、MRR 和 NDCG。
8. 如果 Milvus、Docker、依赖或权限阻塞验证，记录实际失败步骤和原因，同时完成可执行的本地检查。

完成后报告：修改文件；新增或更新的样本数和类别；使用的基金、chunk IDs 和构建来源；工具调用验证；Milvus、JSONL、去重、语法和幂等性结果；仍受外部依赖阻塞的项目。

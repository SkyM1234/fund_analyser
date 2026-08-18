# 构建跨基金 Answer 数据集提示词

请继续构建本项目的 answer 评测数据集。请直接检查代码、修改必要文件、生成数据并完成验证，不要只给方案或问题列表。

项目目录：`E:\pythonprojects\fund_analyser`

## 目标与边界

`answer_cross_fund.jsonl` 用于已明确基金或板块范围的跨基金 Agent 回答评测。该数据集用于评测同一问题下多个指定基金或同一板块内多个基金的跨报告检索、比较和汇总能力。

当前主线为：

```text
intent = "cross_fund_query"
category = "cross_fund_strategy"
```

## 先检查上下文

1. 检查 git/worktree 状态，保留已有用户修改，不得 reset、checkout 或覆盖无关改动。
2. 阅读：
   - `backend/eval/README.md`
   - `backend/eval/schemas.py`
   - `backend/eval/datasets/answer_cross_fund.jsonl`
   - `backend/eval/runners/build_cross_fund_answer_dataset.py`
   - `backend/eval/runners/run_answer_eval.py`
   - `backend/eval/targets/service_target.py`
   - `backend/eval/evaluators/answer_metrics.py`
   - `backend/app/agent/rag_agent.py`
   - `backend/app/services/rag_result_parser.py`
   - `backend/mcp/rag-mcp/src/server.py`
   - `vectorize/vectorize_to_milvus.py`
3. 优先在既有构建脚本上增量修改；已实现的功能先验证，避免创建重复脚本。
4. 读取 `annual_reports_2025_funds/_pdf_review.json`，排除 `code_mismatches` 和 `extraction_issues` 中的报告。

## 跨基金样本

### 分类规则

1. 所有样本的 `intent` 必须是 `cross_fund_query`。
2. 所有样本的 `category` 必须是 `cross_fund_strategy`。

### 候选与 query 构建

1. 阅读多个 `markdown_mineru/*/*_analyzed.md`，从报告正文中的真实字段构思 query；不能只根据基金名称、主题标签或常识出题。
2. query 必须包含明确的比较维度、筛选条件或报告范围，并要求逐基金给出证据。
3. 不能用两个基金的简单并列冒充跨基金能力；问题应体现真实的跨报告比较、筛选或汇总需求。
4. query 中应明确列出基金、基金代码、基金名称或板块范围。query 中少出现基金代码，多用基金名称代替。
5. query 范围内的每只基金都必须有可核验的报告依据；无法找到直接依据时，应放弃该候选问题或缩小问题范围。
6. 已有样本按既有 ID、顺序和去重逻辑增量维护，不得删除其他有效样本。

### 字段与事实标注

每行使用 `AnswerExample` 字段，跨基金样本至少包含：

```json
{
  "id": "answer-cross-fund-001",
  "query": "...",
  "reference_answer": "...",
  "expected_fund_codes": ["159101", "159125"],
  "key_facts": ["..."],
  "should_refuse": false,
  "intent": "cross_fund_query",
  "category": "cross_fund_strategy",
  "relevant_chunk_ids": ["真实 Milvus id"],
  "expected_tool_calls": [],
  "note": "159101: chunk=... id=...；159125: chunk=... id=..."
}
```

1. `reference_answer` 和 `key_facts` 只能由已标注 chunk 支持，不得夸大“全部”“仅有”或“唯一”等结论。
2. `expected_fund_codes` 必须列出 query 范围内、且由答案逐一说明的基金。
3. 每个 `expected_fund_codes` 都必须至少对应一个直接支持答案的 chunk。
4. `key_facts` 应覆盖各基金之间实际需要比较的事实、数字、指数、策略或风险。
5. `note` 必须按基金代码记录依据，例如：`159213: chunk=... id=...；159526: chunk=... id=...`。

## Milvus Ground Truth

Milvus：`http://localhost:19595`；collection：`fund_reports_mineru`。

1. 阅读 Markdown 只是为了构思 query 和定位报告内容；`relevant_chunk_ids` 必须来自在线 Milvus 的真实主键 `id`。
2. 对 query 范围内的每一只基金逐个查询 Milvus，并核对每个相关 chunk 的 `fund_code`、`file_path`、`chunk_index` 和 `content`。
3. `file_path` 必须对应正确的 `markdown_mineru` 下的 `_analyzed.md` 文件。
4. 仅选择能够直接支持答案的最少 chunk；不要使用只共享关键词但不能证明结论的 chunk。
5. `note` 可以使用 `chunk_index` 和章节信息辅助审阅，但不能用它们代替真实 Milvus ID。

## 工具调用和服务评测

1. `expected_tool_calls` 必须以当前 Agent、MCP 和评测器的实际比较语义为准；先检查实现，再写入能够稳定验证的关键调用。
2. query 未直接包含六位基金代码时，必须按顺序标注：

```json
[
  {"name": "rag_identify_funds"},
  {"name": "rag_search", "args": {"filter_fund_code": "最终确定的基金代码"}},
  {"name": "rag_identify_funds"},
  {"name": "rag_search", "args": {"filter_fund_code": "最终确定的基金代码"}}
]
```

4. query 直接包含六位基金代码时，只标注：

```json
[
  {"name": "rag_search", "args": {"filter_fund_code": "基金代码"}},
  {"name": "rag_search", "args": {"filter_fund_code": "基金代码"}}
]
```
5. 六位代码检测不能使用仅适用于英文单词边界的实现，必须识别如 `159128报告期内` 这类与中文直接相连的写法。`rag_identify_funds` 仅要求工具名，不能比较其 args，因为 Agent 可能改写识别 query。
6. answer 评测必须通过已启动 Docker 服务的完整链路运行。`service_target` 应从 `retrieval_context` SSE 事件收集实际 chunk ID，不得从最终回答反推或伪造。
7. 不得新增或保留 `--target local-agent` 作为 answer 评测目标；不要为评测重置 MCP 统计、禁用限流或改用 local-agent，除非用户明确要求。
8. `--concurrency` 是同时执行的评测样本数，服务端应提供相应 worker/并发槽位；它不是单条样本内的工具调用数。

## 生成与验证

运行现有构建脚本，并验证：

1. JSONL 合法，ID 不重复，query 不重复。
2. 所有 `relevant_chunk_ids` 都能在 Milvus 查询到，且基金归属正确。
3. 每个 `expected_fund_codes` 都有直接支持答案的 chunk。
4. `reference_answer` 和 `key_facts` 都能由已标注 chunk 支持，不得引入报告之外的事实。
5. 构建脚本通过语法检查且可重复运行，不会重复追加或改变样本数量。
7. 不得出现 local-agent target；服务评测应能使用 SSE 收集的检索上下文计算 Agentic RAG 的 hit rate、MRR 和 NDCG。
8. 如果 Milvus、Docker、依赖或权限阻塞验证，记录实际失败步骤和原因，同时完成可执行的本地检查。

完成后报告：修改文件；新增或更新的样本数和类别；使用的基金、chunk IDs 和构建来源；工具调用验证；Milvus、JSONL、去重、语法和幂等性结果；仍受外部依赖阻塞的项目。

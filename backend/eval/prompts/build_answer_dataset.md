# 构建 Answer 数据集提示词

请继续构建本项目的 answer 评测数据集。请直接检查代码、修改必要文件、生成数据并完成验证，不要只给方案或问题列表。

项目目录：`E:\pythonprojects\fund_analyser`

## 目标与边界

`answer.jsonl` 评测真实服务中的 Agent 路由、工具调用、跨报告检索和最终回答。它承接两类问题：

1. **单基金回答**：`intent = "fund_query"`。优先复用已验证的 `retrieval.jsonl` 单基金样本。
2. **跨基金/主题/板块回答**：`intent = "cross_fund_query"` 或 `"fund_screening"`。例如比较同类基金的跟踪误差、筛选某主题内持有某公司或满足某条件的基金。这类问题可以、也应覆盖多份报告。

不要把跨基金问题放入默认 `retrieval.jsonl`：该数据集仅测已知基金后的单报告 chunk 检索。

## 先检查上下文

1. 检查 git/worktree 状态，保留已有用户修改，不得 reset、checkout 或覆盖无关改动。
2. 阅读：
   - `backend/eval/README.md`
   - `backend/eval/schemas.py`
   - `backend/eval/datasets/answer.jsonl`
   - `backend/eval/datasets/retrieval.jsonl`
   - `backend/eval/runners/build_answer_dataset.py`
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

数据来源优先使用 `retrieval.jsonl` 中已经验证的单基金问题。默认从尚未用于
answer 的合适样本继续处理；若已有 answer 样本，按已有 ID、顺序和去重逻辑增量维护，不得
删除其他 category 的既有样本。

### 候选筛选

1. `single_fund_strategy` 的 query 必须明确指向一只基金。
2. 优先选择 query 中直接出现六位基金代码、完整基金名称，或经过核验且能唯一映射到基金的简称/别名的样本。
3. 不得仅依赖相邻数据、基金代码顺序或常识推断 query 对应基金。
4. query 无法唯一确定基金时跳过；不要选择同时指向多只基金、泛泛询问基金类别，或不能由单份报告回答的问题。
5. 发现候选样本质量不合格时，记录跳过原因并换用下一个候选。

### 字段与事实标注

单基金样本至少保持以下字段风格：

```json
{
  "id": "answer-001",
  "query": "...",
  "reference_answer": "...",
  "expected_fund_codes": ["159103"],
  "key_facts": ["..."],
  "should_refuse": false,
  "intent": "fund_query",
  "category": "single_fund_strategy",
  "relevant_keywords": ["..."],
  "relevant_chunk_ids": ["真实 Milvus id"],
  "expected_tool_calls": [],
  "note": "来源 retrieval-...；主题；chunk=... id=..."
}
```

1. `reference_answer` 必须严格依据 `relevant_chunk_ids` 的正文撰写，不得用常识补充报告中没有的事实。
2. `expected_fund_codes`、相关 chunk 和 query 指向必须一致。
3. `key_facts` 应是答案必须出现的事实、数字、指数、策略或风险，不能是无判别力的泛化词。
4. `relevant_keywords` 必须是相关 chunk 原文中存在的短语，且能证明语义相关性。
5. `note` 应包含 retrieval 来源、基金、问题主题、`chunk_index` 和真实 chunk ID。

## 跨基金与板块样本

1. 先阅读多个 `markdown_mineru/*/*_analyzed.md`，从真实报告内容构思问题；再逐个到 Milvus 解析对应支持 chunk。不能只从基金名称、主题标签或常识生成。
2. query 必须有明确比较或筛选条件、可验证的报告范围和可回答的结果。例如：
   - 明确列出的同类基金中，哪些期末持有某股票？
   - 指定主题基金中，哪些报告披露了某项跟踪误差目标？
   - 多只指定基金的期末某项配置或业绩分别如何？
3. 用 `cross_fund_query` 表示比较/汇总，用 `fund_screening` 表示按条件筛选。`category` 进一步记录主题和任务，如 `robotics_holding_screening`。
4. `expected_fund_codes` 必须列出答案中应出现和引用的所有基金代码；`relevant_chunk_ids` 包含每只基金的真实支持 chunk。
5. `note` 必须按基金代码映射支持依据，例如 `159213: chunk=... id=...；159526: chunk=... id=...`。
6. 参考答案逐一列出符合条件的基金及其必要证据。除非候选范围被 query 明确列出且已完整核验，否则不得声称“全部”“仅有”或“唯一”。
7. 不要用两个基金的简单并列冒充多基金能力；应至少反映真实的跨报告比较、筛选或聚合需求。

## Milvus Ground Truth

Milvus：`http://localhost:19595`；collection：`fund_reports_mineru`。

1. 直接查询在线 Milvus；`relevant_chunk_ids` 必须是存在的主键 `id`，不得根据 Markdown 推测。
2. 对每个正例核对 `fund_code`、`file_path`、`chunk_index` 和 `content`。`file_path` 必须对应正确的 `_analyzed.md`。
3. 仅选择直接支持 reference answer 的最少 chunk；不要用只共享关键词、但不能证明结论的 chunk。
4. `relevant_keywords` 必须是正例正文中的原始短语；跨基金样本应覆盖每个必要基金的支持内容。
5. `note` 使用 `chunk_index` 和章节辅助审阅，但不得以其代替真实 Milvus ID。

每行使用 `AnswerExample` 字段；该 schema **没有** `filter_fund_code`：

```json
{
  "id": "answer-001",
  "query": "...",
  "reference_answer": "...",
  "expected_fund_codes": ["159213", "159526"],
  "key_facts": ["..."],
  "should_refuse": false,
  "intent": "fund_screening",
  "category": "robotics_holding_screening",
  "relevant_keywords": ["..."],
  "relevant_chunk_ids": ["真实 Milvus id"],
  "expected_tool_calls": [],
  "note": "159213: chunk=... id=...；159526: chunk=... id=..."
}
```

## 工具调用与服务评测

1. `expected_tool_calls` 必须以当前 Agent/MCP 实现和评测器实际比较语义为准；先检查实现，再为样本写入可稳定验证的关键调用。
2. 单基金 query 未直接包含六位代码时，必须按顺序标注：

```json
[
  {"name": "rag_identify_funds"},
  {"name": "rag_search", "args": {"filter_fund_code": "最终确定的基金代码"}}
]
```

3. 单基金 query 直接包含六位代码时，只标注：

```json
[
  {"name": "rag_search", "args": {"filter_fund_code": "基金代码"}}
]
```

4. 六位代码检测不能使用仅适用于英文单词边界的实现，必须识别如 `159128报告期内` 这类与中文直接相连的写法。`rag_identify_funds` 仅要求工具名，不能比较其 args，因为 Agent 可能改写识别 query。
5. 跨基金样本不应强行套用单基金 `filter_fund_code` 断言。根据当前实现标注可验证的识别/检索调用；如现有评测器无法稳定表达多基金调用链，保留空列表并在 `note` 说明，先不要伪造精确 args。
6. answer 评测必须通过已启动 Docker 服务的完整链路运行。`service_target` 应从 `retrieval_context` SSE 事件收集实际 chunk ID，不得从最终回答反推或伪造。
7. 不得新增或保留 `--target local-agent` 作为 answer 评测目标；不要为评测重置 MCP 统计、禁用限流或改用 local-agent，除非用户明确要求。
8. `--concurrency` 是同时执行的评测样本数，服务端应提供相应 worker/并发槽位；它不是单条样本内的工具调用数。

## 生成与验证

运行现有构建脚本，并验证：

1. JSONL 合法，ID 连续且不重复，query 无重复；保留其他类别样本。
2. 所有 `relevant_chunk_ids` 可在 Milvus 查询到，且各自属于 `expected_fund_codes` 中对应基金。
3. 所有关键词在相应正例正文中存在，reference answer 和 key facts 都能由已标注 chunk 支持。
4. 单基金样本的基金指向唯一；未含六位代码的样本均为 `rag_identify_funds -> rag_search`，含代码的样本仅有 `rag_search`。
5. 跨基金样本的每个期望基金都有明确支持 chunk，参考答案不夸大筛选范围或完整性。
6. 构建脚本通过语法检查且可重复运行，不会重复追加或破坏既有数据。
7. 不得出现 local-agent target；服务评测应能使用 SSE 收集的检索上下文计算 Agentic RAG 的 hit rate、MRR 和 NDCG。
8. 若 Milvus、Docker、依赖或权限阻塞验证，记录实际失败步骤和原因，同时完成可执行的本地检查。

完成后报告：修改文件；新增或更新的样本数和类别；使用的基金、chunk IDs 和构建来源；工具调用验证；Milvus、JSONL、去重、语法和幂等性结果；仍受外部依赖阻塞的项目。

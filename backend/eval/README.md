# Fund Analyser 评测框架

基于 **LangSmith** 的评测框架，分别评测 **RAG 检索** 与 **Agent 端到端回答** 两个层面，定位是为后续多 Agent 升级、RAG 调优提供量化基线。

## 评测对象

| 层面 | Target | 关注点 |
|---|---|---|
| RAG 检索 | `targets/rag_target.py`（直连 GPU `/fund_reports/search`） | 检索器本身的召回 & 排序质量 |
| Agent 端到端 | `targets/service_target.py`（调用 Docker 后端） | 路由→工具调用→回答 全链路 |

## 指标

### RAG 检索（5 个）

| Key | 类型 | 含义 |
|---|---|---|
| `hit_rate` | 规则 | top-K 是否包含至少 1 个相关 chunk |
| `mrr` | 规则 | 首个相关结果排名的倒数（1/rank） |
| `ndcg` | 规则 | 折损累计增益，0/1 相关性权重 |
| `fund_code_recall` | 规则 | 期望基金代码被召回的比例 |
| `context_relevance` | LLM-judge | 整体片段对问题的语义相关性 |

相关性判定优先级：`relevant_chunk_ids` → `expected_fund_codes` (+ keywords 加强) → `relevant_keywords`。

### Agent 回答（7 个 + 5 个检索指标）

**回答质量指标**：

| Key | 类型 | 含义 |
|---|---|---|\n| `citation_accuracy` | 规则 | 引用基金代码 F1 |
| `refusal_correctness` | 规则 | 敏感问题拒绝准确率 |
| `key_fact_coverage` | 规则 | 关键事实子串命中率 |
| `intent_accuracy` | 规则 | 路由意图准确率 |
| `correctness` | LLM-judge | 与参考答案的语义一致性 |
| `faithfulness` | LLM-judge | 是否仅基于检索上下文（抓幻觉） |
| `answer_relevance` | LLM-judge | 切题程度 |

**检索质量指标（Agent 驱动的 RAG）**：

| Key | 类型 | 含义 |
|---|---|---|
| `hit_rate` | 规则 | top-K 是否包含至少 1 个相关 chunk |
| `session_mrr` | 规则 | 会话内合并检索结果中首个相关 chunk 排名的倒数 |
| `session_ndcg` | 规则 | 会话内合并检索结果的折损累计增益 |
| `fund_code_recall` | 规则 | 期望基金代码被召回的比例 |
| `context_relevance` | LLM-judge | 整体片段对问题的语义相关性 |

**对比价值**：
- 检索评测（`run_retrieval_eval`）测的是"直接 RAG"（GPU `/fund_reports/search`）
- 回答评测（`run_answer_eval`）测的是"Agent 驱动的 RAG"（可能会改写 query、多轮检索）
- 对比两者的 `context_relevance` / `hit_rate` 可以量化 Agent 层的增益

## 数据集格式

### `datasets/retrieval.jsonl`（每行一个 `RetrievalExample`）

```json
{
  "id": "retrieval-001",
  "query": "159103 的投资策略是什么",
  "filter_fund_code": ["159103"],
  "top_k": 5,
  "expected_fund_codes": ["159103"],
  "relevant_chunk_ids": ["778279457f99a750795ed1dec5d4c072"],
  "relevant_keywords": ["金融科技", "指数"],
  "category": "single_fund"
}
```

### `datasets/answer.jsonl`（每行一个 `AnswerExample`）

```json
{
  "id": "answer-001",
  "query": "159103 的投资策略是什么",
  "reference_answer": "159103 是汇添富中证金融科技主题 ETF，...",
  "expected_fund_codes": ["159103"],
  "key_facts": ["金融科技", "被动", "指数"],
  "relevant_chunk_ids": ["778279457f99a750795ed1dec5d4c072"],
  "should_refuse": false,
  "intent": "fund_query"
}
```

完整回答评测会从 `retrieval_result` SSE 事件收集 RAG 子 Agent 实际命中的
Milvus chunk，并计算 `hit_rate`、`session_mrr`、`session_ndcg`。多次
`rag_search` 的结果按调用顺序合并，相同 chunk 只保留首次出现的位置。
服务 target 会按 `--concurrency` 创建同等数量的评测用户，每个并发槽位独占
一个 MCP 限流桶。评测使用服务的真实按用户限流，不会重置 MCP 调用计数；
若样本触发限流，将按实际服务错误记录。使用 `--no-service-auto-register`
时需预先创建这些带 `_1`、`_2` 后缀的评测用户。

从数据库导出后，按上述结构写入 jsonl 即可。

## 快速开始

```bash
cd backend
pip install -r eval/requirements.txt
cp eval/.env.example eval/.env
# 编辑 eval/.env，填入 LANGSMITH_API_KEY 和 JUDGE_LLM_API_KEY

# 1) 上传数据集到 LangSmith
python -m eval.runners.upload_dataset --kind all --mode append

# 2) 跑 RAG 检索评测
python -m eval.runners.run_retrieval_eval --experiment-prefix v1-baseline

# 3) 跑 Agent 端到端评测（默认调用 http://127.0.0.1:8800）
python -m eval.runners.run_answer_eval --experiment-prefix v1-baseline --concurrency 2

# 跳过 LLM-judge 仅跑规则指标（更快、零成本）
python -m eval.runners.run_retrieval_eval --no-judge
python -m eval.runners.run_answer_eval --no-judge
```

`--no-judge` 只跳过 Judge LLM 调用。数据集读取、实验记录和结果上报仍由
LangSmith 完成，因此始终需要有效的 `LANGSMITH_API_KEY`。

每次运行后：
- 结果同步到 LangSmith 项目 `fund-analyser-eval`
- 本地落盘 JSON 到 `eval/reports/<评测类型>/`；每次运行同时生成明细报告和 `-summary.json` 聚合得分报告
- 控制台打印每个指标的均值

## 设计取舍

1. **Judge LLM 与业务 LLM 解耦**：用单独的 `JUDGE_LLM_*` 配置，避免同源偏差。建议 judge 用更强模型。
2. **服务 target 使用独立 UUID 会话**：每条样本通过完整 FastAPI + Celery + MCP 链路执行，互不共享上下文。
3. **相关性判定多级兜底**：精确 chunk_id → 基金代码+关键词 → 仅关键词，对应标注成本由高到低。
4. **`fund_code_recall` 单独成项**：专门给 `filter_fund_code` 改造做回归——若该值下降说明硬注入失效。
5. **规则指标 + LLM-judge 双轨**：规则指标确定性强可用作 CI gate；LLM-judge 给细粒度信号。
6. **Answer 评测只调用服务**：MCP 开关与 worker 数量由 Docker 部署配置控制，评测进程不再维护独立 Agent 运行时。

## 后续可加

- `tool_call_accuracy`：与期望的工具调用序列对比（多 Agent 升级后必备）
- `latency_p50 / p95`：从 LangSmith trace 直接拉
- `cost_per_query`：基于 token 计数
- 数据集生成脚手架：从问答日志半自动生成 ground truth

## 构建 retrieval 数据集的提示词

  请继续构建本项目的 RAG retrieval（检索）评测数据集。请直接检查代码、Markdown、现有数据集和 Milvus 后执行，不要只给方案
  或问题列表。

  项目目录：
  E:\pythonprojects\fund_analyser

  一、执行要求

  1. 先检查工作区状态、现有代码和数据，不要覆盖或回退已有改动。
  2. 相关代码和数据位于：
     - backend/eval
     - markdown_mineru
     - annual_reports_2025_funds/_pdf_review.json
  3. 输出文件参考并更新：
     - backend/eval/datasets/retrieval.jsonl
  4. 必须在现有构建脚本基础上修改：
     - backend/eval/runners/build_retrieval_dataset.py
     不要重复创建功能相同的脚本。
  5. 用于查看 Milvus 原始 chunk 的辅助脚本是：
     - backend/eval/runners/query_fund_report.py
     如果该脚本因 Python 版本、连接参数或输出编码问题不能运行，可以做最小必要优化。

  二、报告选择

  1. 每次处理接下来的 10 份尚未构建 retrieval 数据的正常报告。
  2. 每份报告生成 10 条数据，本批共 100 条。
  3. 根据现有 retrieval.jsonl 和构建脚本判断哪些报告已经处理，继续选择按既有排序规则排列的后续 10 份，不要重复。
  4. Markdown 位于 markdown_mineru，每份报告一个子目录。
  5. 主要使用 `_analyzed.md` 结尾的文件，这是实际向量化的原始内容。
  6. 必须阅读报告实际正文，不能只根据文件名、基金名称或常识生成问题。

  三、异常报告过滤

  1. 读取 annual_reports_2025_funds/_pdf_review.json。
  2. 不得使用 `code_mismatches` 或 `extraction_issues` 中的异常报告。
  3. SHA256 重复不能单独判定异常；如果基金代码与 PDF 正文代码一致，可以保留。
  4. 核对 Markdown 文件名和正文、Milvus fund_code/file_path、PDF 校验结果。
  5. 如果三者不一致，先定位并处理数据一致性问题，不要带着冲突继续生成。

  四、Milvus Ground Truth

  Milvus 地址：
  http://localhost:19595

  主要 collection：
  fund_reports_mineru

  参考以下代码理解切分、metadata 和入库方式：
  vectorize/vectorize_to_milvus.py

  要求：

  1. 必须直接查询当前 Milvus，使用真实存在的 chunk、主键和 metadata。
  2. 不允许推测、编造或根据 Markdown 自行生成 Milvus ID。
  3. 优先选择能够完整支持问题答案的 chunk。
  4. 只有问题确实需要多个章节时才记录多个正例。
  5. 不得把只包含相同关键词但不能回答问题的 chunk 标为正例。
  6. 不得混用其他基金的 chunk。
  7. 保持现有数据集字段格式，包括：
     - id
     - query
     - filter_fund_code
     - top_k
     - expected_fund_codes
     - relevant_chunk_ids
     - relevant_keywords
     - category
     - note
  8. relevant_chunk_ids 必须使用 Milvus 中真实的 `id`，不能用 chunk_index 代替。
  9. note 应记录主题和实际 chunk_index/章节。

  五、构建报错处理

  如果 build_retrieval_dataset.py 报缺少关键词、chunk 不存在或 metadata 不一致：

  1. 使用 query_fund_report.py 查询对应基金和 chunk，例如：
     python backend/eval/runners/query_fund_report.py 159269 --chunks 100,101
  2. 以 Milvus 实际 content 为准检查：
     - 日期格式，如 `2025-06-26` 与 `2025年6月26日`
     - 数字和单位之间的空格
     - 行业名称的实际写法
     - 监管机构的完整名称
     - Markdown 表格解析产生的空格
  3. 修正问题规范或关键词后继续运行，不能因首次报错停止。
  4. 关键词必须是实际正例正文中的原始子串，同时应足以证明语义相关性。

  六、问题质量

  每份报告的 10 个问题应覆盖不同章节，尽量包括：

  - 基金基本信息、成立日、上市日和基金代码
  - 跟踪指数、复制策略及跟踪误差目标
  - 基金经理、经理助理或人员变更
  - 报告期业绩、净值增长率及基准比较
  - 投资策略、市场回顾和运作分析
  - 资产配置
  - 行业配置
  - 股票或债券持仓
  - 前十大或主要持仓
  - 持有人结构、风险、费用、关联交易或重大事项

  问题必须：

  1. 能由指定正例 chunk 中的明确内容回答。
  2. 表述自然，接近真实用户检索问题。
  3. 不把答案直接写入问题。
  4. 不依赖跨报告推断。
  5. 尽量命中不同章节和 chunk。
  6. 数值类问题核对报告期、单位和上下文。
  7. 基金名称、代码、人物和日期必须来自真实报告。
  8. 避免仅替换数字或基金名称形成模板化重复问题。

  七、生成与验证

  运行构建脚本，更新累计 retrieval.jsonl，并独立验证：

  1. 每行均为合法 JSON。
  2. 本批新增恰好 100 条。
  3. 本批覆盖恰好 10 份不同的正常报告。
  4. 每份报告恰好 10 条。
  5. ID 连续，历史数据顺序和 ID 不被破坏。
  6. 不存在重复 query。
  7. 不包含 PDF 校验异常基金。
  8. 所有 relevant_chunk_ids 均可在 Milvus 查询到。
  9. 每个正例的 fund_code 与 filter_fund_code 一致。
  10. file_path 指向对应基金的 `_analyzed.md`。
  11. relevant_keywords 均存在于对应正例正文。
  12. 问题与正例内容语义匹配，正例足以回答问题。
  13. 没有使用同基金的不相关 chunk 或其他基金的 chunk。
  14. 构建脚本通过语法检查并可重复运行。

  八、最终输出

  完成代码修改、数据集生成和验证后，给出：

  1. 本批使用的 10 份报告、基金代码和基金名称。
  2. 历史累计条数、本批新增条数和累计基金数。
  3. 实际执行的验证结果。
  4. 发现并处理的数据格式或一致性问题。
  5. 修改过的文件列表。

  请持续执行到数据集成功生成并验证完成，不要停留在方案、问题草稿或未解决的构建报错。

## 构建 answer 数据集的提示词

请继续构建本项目的 answer 评测数据集。不要只给方案，请直接检查代码、修改必要文件、生成数据并完成验证。

项目目录：
E:\pythonprojects\fund_analyser

一、先检查上下文

1. 检查当前 git/worktree 状态，保留已有用户修改，不要 reset、checkout 或覆盖无关改动。
2. 阅读以下文件，理解现有字段、构建逻辑、评测逻辑和服务调用链：
   - backend/eval/README.md
   - backend/eval/datasets/answer.jsonl
   - backend/eval/datasets/retrieval.jsonl
   - backend/eval/runners/build_answer_dataset.py
   - backend/eval/runners/run_answer_eval.py
   - backend/eval/targets/service_target.py
   - backend/eval/evaluators/answer_metrics.py
   - backend/app/agent/rag_agent.py
   - backend/app/services/rag_result_parser.py
   - backend/mcp/rag-mcp/src/server.py
   - vectorize/vectorize_to_milvus.py
3. 如果已有构建脚本或数据，不要另起一个重复脚本；优先在现有脚本上增量修改。
4. 如果发现当前代码已经实现了某项要求，先验证其行为，再避免重复修改。

二、当前 answer 数据集的目标

本次优先构建：

- intent = "fund_query"
- category = "single_fund_strategy"
- 数据来源优先使用 retrieval.jsonl 中已经存在且经过验证的问题。

默认先处理 retrieval.jsonl 中尚未用于 answer 的合适样本；如果已有 answer 样本，应按现有 ID、顺序和去重逻辑继续，不要重复添加相同 query。

三、问题筛选规则

1. single_fund_strategy 的 query 必须明确指向一个基金。
2. 优先选择：
   - query 中直接出现六位基金代码；或
   - query 中出现完整基金名称；或
   - query 中出现经过核验、能够唯一映射到该基金的简称/别名。
3. 不要仅凭相邻数据、基金代码顺序或常识推断 query 指向的基金。
4. 如果 query 不能唯一确定某个基金，跳过该样本，换用下一个合适样本。
5. 不要选择同时指向多个基金、泛泛询问基金类别、或无法由单一基金报告内容回答的问题。
6. 如果某个候选样本质量不合适，应明确跳过并记录原因。例如，single_fund_strategy 不应使用没有明确基金指向的 query。

四、Milvus Ground Truth

Milvus 地址：
http://localhost:19595

主要 collection：
fund_reports_mineru

必须参考：
vectorize/vectorize_to_milvus.py

要求：

1. 直接查询当前 Milvus，使用真实存在的 chunk、主键 id 和 metadata。
2. 不要猜测、编造或根据 Markdown 内容自行生成 chunk ID。
3. relevant_chunk_ids 必须填写 Milvus 中真实的 id，不能用 chunk_index 替代。
4. 每个 relevant_chunk_id 必须：
   - 在 Milvus 中存在；
   - 属于 expected_fund_codes 对应的基金；
   - 内容能够直接支持 query 的答案。
5. 优先选择能够完整回答问题的最少 chunk。
6. 只有问题确实需要多个章节或多个事实时，才添加多个 relevant_chunk_ids。
7. 不要把只有相同关键词、但不能支持答案的 chunk 标为 Ground Truth。
8. 不要混入其他基金的 chunk。
9. 核对 metadata 中的 fund_code、file_path、chunk_index 等字段，发现不一致时先定位数据问题，不要带着冲突继续生成。
10. 可以使用现有辅助脚本查询原始 chunk；如果辅助脚本不可用，做最小必要修复，不要另建重复工具。

五、answer.jsonl 字段要求

每行必须是一个合法 JSON 对象，至少包含：

{
  "id": "answer-001",
  "query": "...",
  "reference_answer": "...",
  "expected_fund_codes": ["159xxx"],
  "key_facts": ["..."],
  "should_refuse": false,
  "intent": "fund_query",
  "category": "single_fund_strategy",
  "relevant_keywords": ["..."],
  "relevant_chunk_ids": ["真实 Milvus chunk id"],
  "expected_tool_calls": [],
  "note": "来源、选择理由、chunk_index 和 chunk id"
}

字段规则：

1. reference_answer 必须严格依据 relevant_chunk_ids 中的内容撰写，不要凭常识补充报告中没有的事实。
2. expected_fund_codes 必须与 query 和相关 chunk 的基金一致。
3. key_facts 应是答案中应明确出现的关键事实、数字、指数名称、策略或风险；不要填泛化词。
4. relevant_keywords 必须是相关 chunk 原文中真实存在的短语，并且能够证明语义相关性。
5. note 应说明 retrieval 来源、基金、问题主题、chunk_index 和真实 chunk id。
6. 保持现有 answer.jsonl 的字段风格、编码和 JSONL 格式。
7. 保留其他 category 的既有样本，不要因为重建 strategy 样本而删除它们。

六、expected_tool_calls 规则

必须根据 query 是否直接包含六位基金代码生成：

1. query 不直接包含六位基金代码时，expected_tool_calls 必须按以下顺序包含：

[
  {
    "name": "rag_identify_funds"
  },
  {
    "name": "rag_search",
    "args": {
      "filter_fund_code": "最终确定的基金代码"
    }
  }
]

2. query 直接包含六位基金代码时，只需要：

[
  {
    "name": "rag_search",
    "args": {
      "filter_fund_code": "基金代码"
    }
  }
]

3. 六位代码检测不能依赖只适用于英文的单词边界；必须正确识别类似“159128报告期内……”这种数字与中文直接相连的情况。
4. rag_identify_funds 只要求工具名被调用，不要标注或比较 args；Agent 可能改写用于识别的 query。
5. expected_tool_calls 表示期望的工具调用链，不要因为普通 RAG 和 Agentic RAG 的结果不同而删除工具调用标注。

七、Agentic RAG Ground Truth

1. answer 评测通过 Docker 中已经启动的完整后端服务执行，不要把 answer 评测改回本机 local-agent。
2. 不要新增或保留 --target local-agent 作为 answer 评测目标。
3. service_target 应通过服务的完整对话流程收集实际 retrieval_result SSE 事件中的 chunk IDs。
4. 如果当前流程无法收集子 RAG Agent 实际命中的 chunk IDs，应优先修复 SSE/结果解析/target 链路，使 relevant_chunk_ids 能用于 Agentic RAG 的 hit_rate、MRR、NDCG 比较。
5. 不要伪造 Agentic RAG 的 chunk IDs，也不要只从最终答案中反推 chunk IDs。
6. 不要在评测前调用 MCP reset-stats；保留服务真实的按用户限流行为。
7. 不要为了评测创建独立 Docker 环境或禁用限流，除非用户明确要求。
8. --concurrency 表示同时运行的评测样本数；服务端应有相匹配数量的可用 worker/并发槽位。不要把它误解成每个样本内部的工具调用数。

八、构建与验证

1. 运行现有 build_answer_dataset.py 生成或更新 answer.jsonl。
2. 如果脚本需要访问 Milvus，使用：

python backend/eval/runners/build_answer_dataset.py --milvus-uri http://localhost:19595 --collection fund_reports_mineru

3. 运行以下检查：
   - 每行都是合法 JSON；
   - id 连续且没有重复；
   - query 没有重复；
   - expected_fund_codes、基金名称和 relevant_chunk_ids 一致；
   - 所有 relevant_chunk_ids 都能在 Milvus 查询到；
   - 所有 Ground Truth chunk 都属于目标基金；
   - relevant_keywords 出现在对应 chunk 内容中；
   - reference_answer 能由 Ground Truth chunk 支持；
   - 未写出不应存在的 local-agent target；
   - query 不含六位代码的样本均包含 rag_identify_funds -> rag_search；
   - query 含六位代码的样本仅包含 rag_search；
   - 构建脚本可以重复运行，不会重复追加或破坏既有数据；
   - Python 语法检查通过。
4. 如果构建过程出现 chunk 不存在、metadata 不一致、关键词缺失或 query 不明确，不要绕过校验；查询 Milvus 后修正样本或更换候选样本。
5. 如果 Docker、Milvus、Python 依赖或权限导致某项验证无法执行，明确记录实际失败步骤和原因，同时完成不依赖该外部服务的其他校验。

九、完成时报告

完成后简要报告：

1. 修改了哪些文件；
2. 新增或更新了多少条 answer 样本；
3. 使用了哪些 retrieval 样本、基金代码和 chunk IDs；
4. expected_tool_calls 规则验证结果；
5. Milvus、JSONL、重复数据、语法和构建幂等性验证结果；
6. 仍存在的外部依赖或未完成验证。

请持续执行到数据集生成和验证完成，不要停留在方案、问题列表或未解决的构建报错上。

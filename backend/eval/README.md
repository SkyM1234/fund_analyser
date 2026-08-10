# Fund Analyser 评测框架

基于 **LangSmith** 的评测框架，分别评测 **RAG 检索** 与 **Agent 端到端回答** 两个层面，定位是为后续多 Agent 升级、RAG 调优提供量化基线。

## 评测对象

| 层面 | Target | 关注点 |
|---|---|---|
| RAG 检索 | `targets/rag_target.py`（直连 GPU `/fund_reports/search`） | 检索器本身的召回 & 排序质量 |
| Agent 端到端 | `targets/agent_target.py`（驱动 LangGraph） | 路由→工具调用→回答 全链路 |

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
| `mrr` | 规则 | 首个相关结果排名的倒数 |
| `ndcg` | 规则 | 折损累计增益 |
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
  "relevant_chunk_ids": [],
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
  "should_refuse": false,
  "intent": "fund_query"
}
```

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

# 3) 跑 Agent 端到端评测（启动 MCP，并发建议 1~2）
python -m eval.runners.run_answer_eval --experiment-prefix v1-baseline --concurrency 2

# 单独测试 RAG 部分（禁用 cn-funds-mcp，Agent 只能用 rag_search）
python -m eval.runners.run_answer_eval --experiment-prefix v1-rag-only --concurrency 2 --no-cn-funds-mcp

# 跳过 LLM-judge 仅跑规则指标（更快、零成本）
python -m eval.runners.run_retrieval_eval --no-judge
python -m eval.runners.run_answer_eval --no-judge
```

`--no-judge` 只跳过 Judge LLM 调用。数据集读取、实验记录和结果上报仍由
LangSmith 完成，因此始终需要有效的 `LANGSMITH_API_KEY`。

每次运行后：
- 结果同步到 LangSmith 项目 `fund-analyser-eval`
- 本地落盘 JSON 到 `eval/reports/`
- 控制台打印每个指标的均值

## 设计取舍

1. **Judge LLM 与业务 LLM 解耦**：用单独的 `JUDGE_LLM_*` 配置，避免同源偏差。建议 judge 用更强模型。
2. **Agent target 用 MemorySaver**：评测期会话不污染生产 PG checkpoint。
3. **相关性判定多级兜底**：精确 chunk_id → 基金代码+关键词 → 仅关键词，对应标注成本由高到低。
4. **`fund_code_recall` 单独成项**：专门给 `filter_fund_code` 改造做回归——若该值下降说明硬注入失效。
5. **规则指标 + LLM-judge 双轨**：规则指标确定性强可用作 CI gate；LLM-judge 给细粒度信号。
6. **`--no-cn-funds-mcp` 选项**：用于单独测试 RAG 部分，避免 Agent 通过 `get_fund_list` 等工具获取额外信息，导致 `faithfulness` / `context_relevance` 等指标失真。

## 后续可加

- `tool_call_accuracy`：与期望的工具调用序列对比（多 Agent 升级后必备）
- `latency_p50 / p95`：从 LangSmith trace 直接拉
- `cost_per_query`：基于 token 计数
- 数据集生成脚手架：从问答日志半自动生成 ground truth

## 构建数据集的提示词

  我需要继续构建本项目的 RAG 评测数据集，请直接检查代码、数据和 Milvus 后执行，不要只给方案。

  项目目录：
  E:\pythonprojects\fund_analyser

  当前先构建 retrieval（检索）评测数据集，要求如下：

  1. 数据集位置与格式
  - 相关代码和数据位于 backend/eval。
  - 输出文件参考现有的 backend/eval/datasets/retrieval.jsonl。
  - 如果已有构建脚本 backend/eval/runners/build_retrieval_dataset.py，先检查并在其基础上修改，不要重复创建功能相同的脚
  本。

  1. 原始 Markdown
  - Markdown 位于 markdown_mineru。
  - 每份报告一个子目录。
  - 主要使用文件名以 `_analyzed.md` 结尾的文件，这是最终用于向量化的原始内容。
  - 不要只根据文件名、基金名称或常识生成问题，必须阅读报告实际内容。

  1. Milvus 数据
  - 参考 vectorize/vectorize_to_milvus.py，理解 Markdown 的切分、metadata 和入库方式。
  - 必须直接访问 Milvus 查询实际入库数据，用真实 chunk、主键和 metadata 构建 ground truth，不能自行猜测。
  - Milvus 地址通常是：
    http://localhost:19595
  - 主要 collection 是：
    fund_reports_mineru
  - 如有需要，同时检查 fund_index 等相关 collection。
  - 构建后要验证每条正例在 Milvus 中确实存在，并且与问题对应。

  1. 异常报告过滤
  - PDF 校验结果位于：
    annual_reports_2025_funds/_pdf_review.json
  - 不要使用 `code_mismatches` 或 `extraction_issues` 中的异常报告。
  - SHA256 重复本身不能直接判定异常：如果基金代码和 PDF 正文代码一致，可以保留。
  - 如果 Markdown、Milvus 和 PDF 校验结果不一致，先报告并处理数据一致性问题。

  1. 数据规模
  - 每次处理 10 份正常报告。
  - 每份报告构建 10 条 retrieval 评测数据，共 100 条。
  - 如果前面的报告已经处理过，则继续选择接下来的 10 份未处理报告。
  - 输出时说明本次使用了哪些报告和基金代码。

  1. 问题质量
  每份报告的 10 个问题应尽量覆盖不同方面，避免只是替换数字或改写句式。可覆盖：
  - 基金基本信息和基金代码
  - 基金经理及变更
  - 报告期内业绩表现
  - 净值增长率及基准比较
  - 资产配置
  - 股票或债券持仓
  - 前十大持仓
  - 行业配置
  - 投资策略和运作分析
  - 风险、费用、关联交易或重大事项

  问题必须满足：
  - 能由报告中的明确内容回答。
  - 表述自然，接近真实用户检索问题。
  - 不把答案直接写进问题。
  - 不生成含糊、无法定位或需要跨报告推断的问题。
  - 尽量让不同问题命中不同章节和 chunk。
  - 数值类问题必须核对单位、报告期和上下文。
  - 问题中的基金名称、代码、人物和日期必须来自真实内容。

  7. Ground truth 要求
  - 每条数据的正例必须来自 Milvus 中真实存在的相关 chunk。
  - 优先选择能够完整支持答案的 chunk，不要只选择出现关键词但无法回答问题的片段。
  - 如一个问题确实需要多个 chunk，可以记录多个正例，但不要无意义增加正例。
  - 检查 fund_code、source、section、chunk_id 或主键等字段是否与现有数据集格式一致。
  - 不要把同基金但不相关的 chunk 标成正例。
  - 避免不同基金之间的数据串用。

  8. 验证
  完成后至少检查：
  - JSONL 每行都是合法 JSON。
  - 总数是否为 100 条。
  - 是否每份报告正好 10 条。
  - 是否覆盖 10 份不同的正常报告。
  - 是否存在重复问题。
  - 所有正例 ID 是否能在 Milvus 中查到。
  - 问题与正例内容是否语义匹配。
  - 是否误用了异常基金或其他基金的 chunk。
  - 给出构建结果摘要和发现的数据问题。

  请直接完成数据选择、Milvus 查询、数据集生成和验证。不要只生成问题列表，也不要使用虚构的 Milvus 数据。

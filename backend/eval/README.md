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

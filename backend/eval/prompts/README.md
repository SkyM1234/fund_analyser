# 评测数据集构建提示词

本目录按评测数据集类型维护构建提示词。构建前先阅读对应 schema、现有数据集和构建脚本；不得覆盖或回退工作区中的无关改动。

| 数据集 | 提示词 | 目标 |
|---|---|---|
| `retrieval.jsonl` | [build_retrieval_dataset.md](build_retrieval_dataset.md) | 已知基金代码后的单报告 chunk 检索 |
| `answer_single_fund.jsonl` | [build_single_fund_answer_dataset.md](build_single_fund_answer_dataset.md) | Agent 端到端单基金回答 |
| `answer_cross_fund.jsonl` | [build_cross_fund_answer_dataset.md](build_cross_fund_answer_dataset.md) | 已明确基金或板块范围的跨基金比较与策略回答 |
| `fund_name_resolution.jsonl` | [build_name_resolution_dataset.md](build_name_resolution_dataset.md) | 名称、别名和报告标题的基金代码识别 |

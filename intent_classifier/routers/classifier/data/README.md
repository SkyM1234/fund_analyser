# 意图分类起步数据：200 条

`demo.jsonl` 保留原文件名以兼容训练命令，目前总计 200 条，并非额外新增 200 条。
原有 36 条扩充至 200 条，原有 4 条与路由回归集完全相同的输入已调整问法。
标签由助手按项目 `backend/app/services/router.py` 的规则编写，尚未经过独立人工审核。

| 类别 | 数量 |
| --- | ---: |
| chitchat | 25 |
| out_of_scope | 25 |
| sensitive | 35 |
| fund_query | 50 |
| fund_screening | 35 |
| general_finance | 30 |
| 合计 | 200 |

32 条包含多轮上下文，涉及单个对象、多个对象、前者/后者、指代歧义、补充基金代码、
继承板块条件、继续无范围筛选、切换主题、事实查询转为投资建议等情形。

## 年报来源

138 条参考 `markdown_mineru` 中 17 份 2025 年基金年报的名称、披露字段及概念，涉及代码：

`159101`、`159102`、`159103`、`159110`、`159115`、`159201`、`159202`、`159213`、
`159227`、`159230`、`159256`、`159278`、`159299`、`159326`、`159366`、`159369`、`159731`。

每条的 `source_refs` 对应 `sources.json` 的 `sources` 中的代码键，后者包含本地 MD 路径、
参考章节的行号和摘录。62 条 `origin=synthetic` 的样本不声称来自年报。
`origin=report_inspired` 代表内容启发来源，不能据此认定问题、标签或筛选结果为年报原文。
数据只标注意图，不提供基金事实答案。条件筛选可能没有匹配结果，这不影响意图类别。
训练输入只读取 query/history，来源、分组、备注等字段不会作为模型特征。

## 标注边界

- 明确基金代码、基金名称或板块/主题的事实查询，属于 `fund_query`，即使附带筛选条件。
- 没有上述范围，只按持仓公司、规模、费率、持有人比例等事实条件反查，属于 `fund_screening`。
- 持仓公司的股票代码不能直接当作基金代码；不根据公司名称猜测基金板块。
- 请求具体推荐、买卖时点、仓位配置、未来收益预测，属于 `sensitive`。
- 客观历史排名、筛选和比较不是推荐；反过来，“不要推荐”也不能消除实际买卖请求。
- 金融概念和一般原理属于 `general_finance`；提到主题名称不等于正在查询该主题下的基金。
- “它的托管费率是多少”在无历史时仍是 `fund_query`，缺少对象或歧义应由后续澄清处理。
- 新话题不继承历史基金；纯问候加实质问题时按实质问题标注。

## 数据检查与划分

同一语义模板或对照样本使用相同 group_id。默认 seed=42、validation-ratio=0.2 时，
现有脚本划分为 159 条训练、41 条验证，两边均覆盖六类，group_id 无交集。
200 条输入无完全重复，与 `backend/eval/datasets/routing.jsonl` 无完全相同的 query/history。
分组和完全去重不能保证语义上的完全独立，验证分数仍需用真实测试集确认。
统计和分布记录在 `sources.json` 的 `validation` 中。

在项目根目录运行：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --train-file intent_classifier/routers/classifier/data/demo.jsonl --output-dir models/intent-macbert-200-v1
```

训练输出在 models 目录，不会修改这 200 条源数据。若输出目录已存在内容，请换用新目录。

## 额外的 20 条测试数据

`test.jsonl` 独立于上述 200 条，含 10 条单轮、10 条多轮样本。
标签分布：chitchat 3、out_of_scope 3、sensitive 4、fund_query 4、fund_screening 3、general_finance 3。
`tags` 描述测试场景，`note` 说明易混淆的标注边界；两者都不会输入模型。
`intent-test-013` 和 `intent-test-020` 的当前问题相同，但历史分别指向具体基金风险和通用风险概念，
因此预期标签不同；两条共用一个测试分组。
新样本没有与 demo.jsonl 完全相同的 query/history，也没有共享 group_id。
不要把 test.jsonl 加入训练或验证划分。用 evaluate.py 独立评测，完整命令见上一级 README。

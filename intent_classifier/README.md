# 离线意图路由

统一入口为 `cascade.py`。通过 `--routes` 选择启用的路由，按
**规则 → 向量 → 分类模型 → 澄清** 的顺序执行，首个接受结果立即返回。
全程离线，只考虑当前 `query`，不调用 LLM 或在线 API。

## 目录和职责

```text
intent_classifier/
├── cascade.py                  # 唯一运行入口：选择路由、初始化、执行验证
├── pipeline.py                 # 级联顺序、提前返回、最终澄清
├── common.py                   # 标签、query 数据校验、分数接受条件
├── routers/
│   ├── rules/
│   │   ├── router.py           # 整句规则匹配
│   │   └── data/rules.json     # 规则 ID、标签、正则表达式
│   ├── vector/
│   │   ├── router.py           # BGE 编码、相似度聚合、阈值判断
│   │   └── data/examples.jsonl # 134 条向量例句
│   └── classifier/
│       ├── router.py           # 本地六分类推理、阈值判断
│       ├── train.py            # 分类模型训练
│       ├── predict.py          # 分类模型原始预测
│       ├── evaluate.py         # 分类模型独立评测
│       ├── common.py           # 训练数据划分与原有编码格式
│       ├── data/               # 分类模型训练/测试数据与来源
│       └── README.md           # 分类模型训练和评测说明
├── validation/
│   ├── queries.jsonl           # 所有路由组合共用的 22 条验证问题
│   └── reporting.py            # 汇总指标
├── reports/                    # 验证结果：cascade/ 与 classifier/
└── tests/
```

每个路由实现 `route(query)`，返回 `accepted` 和候选意图等判定信息。
`pipeline.py` 只组合这些判定，`cascade.py` 只创建已启用的路由；禁用的路由不加载模型或数据。
各路由的代码和专属数据分别维护，所有组合共享同一份验证问题。

## 运行

推荐 Python 3.11；本机环境为 `E:\python_envs\llm\python.exe`。
依赖安装：`python -m pip install -r intent_classifier/requirements.txt`。
在项目根目录运行：

```powershell
# 全部启用
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes rules vector classifier
# 单独验证
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes rules
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes vector
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes classifier
# 任意组合，执行顺序固定
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes rules classifier
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes vector classifier
# 全部关闭，直接澄清
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes none
```

支持模块方式：`python -m intent_classifier.cascade --routes vector`。
验证单个问题或自己的数据：

```powershell
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --query "不要推荐，只比较159103和159104的费率"
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --input-file path/to/queries.jsonl --routes vector --vector-threshold 0.85 --vector-margin 0.06
```

输入与向量例句为 UTF-8 JSONL，不接受 `history` 字段或相同 query 的冲突标签：

```json
{"id":"example-001","query":"哪些基金持有宁德时代？","intent":"fund_screening"}
```

六类标签为 `chitchat`、`out_of_scope`、`sensitive`、`fund_query`、`fund_screening`、`general_finance`。
规则可通过 `--rules-file` 指定，向量例句可通过 `--routes-file` 指定。

## 模型和接受条件

默认向量模型为 `embedding_model/bge-base-zh-v1.5`，分类模型为
`models/intent-macbert-200-v1/best`；分别通过 `--embedding-model`、`--classifier-model` 指定。
模型初始化一次并常驻内存，向量例句在初始化时编码一次。
默认自动选择 GPU，`--cpu --cpu-threads 2` 可切换 CPU。

向量路由对每类取最大余弦相似度，比较最高分和第二类别的分差；分类模型检查
softmax 最高分及类别分差。两项都达标才接受，否则交给下一层。

| 路由 | 最高分阈值 | 类别分差 | 参数 |
|---|---:|---:|---|
| 向量 | 0.85 | 0.06 | `--vector-threshold`、`--vector-margin` |
| 分类模型 | 0.75 | 0.15 | `--classifier-threshold`、`--classifier-margin` |

初始阈值尚未校准，分数不代表正确概率。可用
`--classifier-threshold 0 --classifier-margin 0` 查看分类器不拒识的基础效果。
各层都未接受时返回 `intent=null`、`needs_clarification=true`。
模型加载失败及输入超长会报错。此线路只验证意图，不补全指代或检查基金查询参数是否完整。

## 验证结果

级联报告写入 `intent_classifier/reports/cascade/<时间戳>/`，分类模型独立评测报告写入
`intent_classifier/reports/classifier/<时间戳>/`。通过 `--output-dir` 可以指定其他目录。

- `metrics.json`：启用路由、阈值、覆盖率、已接受样本准确率、各层尝试与命中数、候选准确率、延迟。
- `predictions.jsonl`：每条问题的意图或澄清、逐层分数/分差/阈值及耗时。
- `report.md`：逐条问题的可读报告，展示每层六类分数及是否接受。

每层判定带有 `scores` 和 `score_type`：规则的 `rule_match` 为命中标记（0/1），
向量的 `cosine_similarity` 为余弦相似度，分类模型的 `softmax_probability` 为未校准概率。
Markdown 报告将分类模型概率显示为百分比，向量相似度保留原始数值。
未启用或因前序命中而未执行的路由标记为 —，报告生成不会额外运行路由。

延迟按 batch=1 测量，排除初始化与预热；初始化时间单列。`--warmup-steps` 默认 3。
比较效果时同时查看覆盖率和已接受样本准确率。
演示问题与训练/例句可能重叠，报告记录与向量例句相同的 query 数量。
正式评估需使用独立数据集；当前分类权重曾使用部分带历史样本训练，本线路始终以空历史编码。

分类模型的训练数据、来源、训练和评测命令见 [routers/classifier/README.md](routers/classifier/README.md)。
运行测试：`python -m unittest discover -s intent_classifier/tests -v`。

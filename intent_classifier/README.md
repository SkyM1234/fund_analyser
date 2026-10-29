# 本地意图路由

本模块负责 LLM 前置分流，按 **规则 → 向量 → 分类模型 → LLM 回退** 的顺序执行，
首个接受结果立即返回。未接受的问题由上层调用 LLM 继续识别意图。
本模块只处理当前 `query`，模型和例句全部本地加载；LLM 调用由上层系统负责。

当前基线使用 `models/intent-macbert-200-v2-epoch6/best`，分类接受阈值为 **0.85**。
训练、数据来源和回归集说明分别见：

- [分类模型：训练、预测与独立评测](routers/classifier/README.md)
- [分类模型数据：标签边界与来源](routers/classifier/data/README.md)
- [级联回归集：100 条问题与标注边界](validation/README.md)

## 目录和职责

```text
intent_classifier/
├── cascade.py                  # 级联 CLI：选择路由、初始化、生成报告
├── pipeline.py                 # 级联顺序、提前返回、标记 LLM 回退
├── common.py                   # 标签、默认分类模型、query 校验、接受条件
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
│       ├── common.py           # 训练数据校验、分组划分、输入编码
│       ├── data/               # 分类模型训练/测试数据与来源
│       └── README.md           # 分类模型训练和评测说明
├── validation/
│   ├── queries.jsonl           # 所有路由组合共用的 100 条回归问题
│   └── reporting.py            # 本地路由指标与 Markdown 报告
├── reports/                    # 本地生成的报告和诊断记录，不纳入版本控制
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
# 使用当前基线，默认启用全部本地路由
& E:\python_envs\llm\python.exe -m intent_classifier.cascade
# 单独验证
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes rules
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes vector
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes classifier
# 组合验证，执行顺序固定
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes rules classifier
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes vector classifier
# 全部关闭，只输出 LLM 回退标记
& E:\python_envs\llm\python.exe intent_classifier/cascade.py --routes none
```

同时支持直接执行 `python intent_classifier/cascade.py`。
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
`models/intent-macbert-200-v2-epoch6/best`；分别通过 `--embedding-model`、`--classifier-model` 指定。
分类模型默认路径统一定义在 `common.py`，原始预测脚本也使用该路径。
权重目录不纳入版本控制；换机器时需要准备模型或显式指定路径。
模型初始化一次并常驻内存，向量例句在初始化时编码一次。
默认自动选择 GPU，`--cpu --cpu-threads 2` 可切换 CPU。

向量路由对每类取最大余弦相似度，比较最高分和第二类别的分差；分类模型检查
softmax 最高分及类别分差。两项都达标才接受，否则交给下一层。

| 路由 | 最高分阈值 | 类别分差 | 参数 |
|---|---:|---:|---|
| 向量 | 0.85 | 0.06 | `--vector-threshold`、`--vector-margin` |
| 分类模型 | 0.85 | 0.15 | `--classifier-threshold`、`--classifier-margin` |

当前阈值是回归集上的保守运行配置，未经独立概率校准，分数不等于实际正确率。可用
`--classifier-threshold 0 --classifier-margin 0` 查看分类器不拒识的基础效果。
模型加载失败及输入超长会报错。此线路只验证意图，不补全指代或检查基金查询参数是否完整。

## 结果与 LLM 交接

`CascadeRouter.route(query)` 返回本地结果和逐层 `trace`。以下字段沿用已有格式：

| 字段 | 本地接受 | 本地全部未接受 |
| --- | --- | --- |
| `intent` | 六类意图之一 | `null` |
| `stage` | `rules` / `vector` / `classifier` | `clarification` |
| `needs_clarification` | `false` | `true`，表示交给 LLM 识别意图 |
| `clarification_question` | `null` | `null` |

`clarification` 是历史字段名称。在本模块中它表示 LLM 回退，上层应先调用 LLM，
不能据此直接要求用户补充问题。是否需要用户澄清，由后续业务流程决定。
本模块不调用 LLM，也不生成用户澄清问题。

## 验证结果

级联报告写入 `intent_classifier/reports/cascade/<时间戳>/`，分类模型独立评测报告写入
`intent_classifier/reports/classifier/<时间戳>/`。通过 `--output-dir` 可以指定其他目录。

- `metrics.json`：启用路由、阈值、覆盖率、已接受样本准确率、各层尝试与命中数、候选准确率、延迟。
- `predictions.jsonl`：每条问题的本地意图或 LLM 回退标记、逐层分数/分差/阈值及耗时。
- `report.md`：逐条问题的可读报告，展示每层六类分数及是否接受。

每层判定带有 `scores` 和 `score_type`：规则的 `rule_match` 为命中标记（0/1），
向量的 `cosine_similarity` 为余弦相似度，分类模型的 `softmax_probability` 为未校准概率。
Markdown 报告将分类模型概率显示为百分比，向量相似度保留原始数值。
未启用或因前序命中而未执行的路由标记为 —，报告生成不会额外运行路由。

延迟按 batch=1 测量，排除初始化与预热；初始化时间单列。`--warmup-steps` 默认 3。
核心指标口径：

| 指标 | 含义 |
| --- | --- |
| `coverage` | 本地接受数 ÷ 总样本数 |
| `accepted_accuracy` | 本地接受且有标注的样本中，分类正确的比例 |
| `clarification_count` | 需要交给 LLM 的样本数，沿用历史字段名称 |
| `by_stage.*.candidate_accuracy` | 实际执行到该层、有候选且有标注的样本中，候选分类正确的比例；包含未达接受阈值的候选 |

因前序接受而未执行的层不计入候选准确率；规则未命中时没有候选，也不参与候选准确率。
这些指标仅统计本地路由，完整系统的最终准确率、调用成本和延迟还需要纳入 LLM 阶段。

当前基线报告为 `reports/cascade/20261029-124150-158795/`：100 条中本地接受 44 条且全部正确，
其中规则 3 条、向量 6 条、分类模型 35 条，其余 56 条交给 LLM。报告未执行 LLM，
不能据此推算完整系统准确率，也不能把本批零误接受视为线上保证。

演示问题与训练/例句可能重叠，报告记录与向量例句相同的 query 数量。
验证问题的类别分布、年报参考和标注边界见 [validation/README.md](validation/README.md)。
正式评估需使用独立数据集；当前分类权重曾使用部分带历史样本训练，本线路始终以空历史编码。

运行测试：`python -m unittest discover -s intent_classifier/tests -v`。

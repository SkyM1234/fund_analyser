# 本地六分类模型

使用本地 MacBERT 微调项目的六种意图，也可通过 `--model-path` 换成本地 RBT3。
`router.py` 由级联入口调用，`predict.py` 查看原始分类分数，`train.py` 和 `evaluate.py`
分别负责训练和独立评测。这里不补全基金范围、指代或业务参数。
级联接受条件与 LLM 回退约定见 [主说明](../../README.md)。

## 环境

推荐 Python 3.11，本机解释器为 `E:\python_envs\llm\python.exe`。
依赖安装统一见 [主说明](../../README.md#运行)。以下命令均在项目根目录执行。

## 当前基线

级联和原始预测默认使用 `models/intent-macbert-200-v2-epoch6/best`。
该模型从基础 MacBERT 训练 6 轮，使用 v2 保存的 159 条训练、41 条验证划分，
`warmup-ratio=0.25`，最佳检查点为 `checkpoint-60`。
级联最高概率阈值为 0.85，类别分差为 0.15；原始预测脚本不执行接受或回退判断。

当前基线的复现命令：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train `
  --model-path models/chinese-macbert-base `
  --train-file models/intent-macbert-200-v2/train_split.jsonl `
  --validation-file models/intent-macbert-200-v2/validation_split.jsonl `
  --output-dir models/intent-macbert-epoch6-reproduced `
  --epochs 6 --warmup-ratio 0.25
```

当前数据每轮约 10 次参数更新，6 轮共约 60 步；预热比例 0.25 对应 15 步，
与原 v2 的 15 轮、0.1 预热比例保持一致。复现依赖原 v2 的数据划分文件。

## 数据格式

UTF-8 JSONL，每行一个样本，必须有 `query`、`intent`，可以有 `history`：

```json
{"id":"sample-001","history":[["human","159103的持仓"],["ai","159103的持仓如下。"]],"query":"它的费率呢","intent":"fund_query","group_id":"conversation-001"}
```

标签为 `chitchat`、`out_of_scope`、`sensitive`、`fund_query`、`fund_screening`、`general_finance`。
历史也支持 `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`。
其他字段会保留在输出的数据划分中，但不作为模型输入。

同一会话或同一模板生成的相似样本请使用相同 `group_id`，脚本会按组隔离训练与验证。
没有 `group_id` 时按类别分层随机划分，无法自动识别近似改写；应自行提供分组或独立验证集。
完全相同的 query/history 会被拒绝，独立训练与验证文件之间也会检查重复和分组泄漏。
训练、验证均须覆盖六类。自动划分要求每类至少两个样本，且验证集样本数足够覆盖六类。

`data/demo.jsonl` 为 200 条起步数据，`data/test.jsonl` 为额外的 20 条独立测试数据。
类别数量、标注边界、分组和来源说明统一见 [数据说明](data/README.md)。
级联回归集 `intent_classifier/validation/queries.jsonl` 保留用于回归，不加入训练集。

## 新训练与继续微调

在项目根目录运行，使用这 200 条数据训练（自动按 group_id 隔离训练集和验证集）：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --train-file intent_classifier/routers/classifier/data/demo.jsonl --output-dir models/intent-macbert-next
```

正式训练可显式提供训练集和验证集（将路径换成实际文件）：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --train-file path/to/train.jsonl --validation-file path/to/validation.jsonl --output-dir models/intent-macbert-next
```

默认从 `models/chinese-macbert-base` 离线加载，输入上限 256 token，单卡 batch=4，梯度累积 4 步，
有效 batch=16，学习率 2e-5，最多 15 个 epoch。默认 `constant_with_warmup` 调度：前 10% 的优化步数
用于预热，此后保持学习率，给小数据集更充分的训练机会。CUDA 下自动使用 FP16 和梯度检查点。
默认按验证交叉熵损失选最佳模型，连续 4 轮不下降则提前停止，并在训练结束后导出最佳模型。
即使分类准确率或 Macro-F1 暂时不变，只要验证损失仍在下降，训练就可以继续。
这是一组适合尝试的 6GB 显存配置，实际占用取决于其他 GPU 进程。
这些是通用训练默认值；复现当前第 6 轮基线时使用上节的显式参数。

- 训练预算：`--epochs 15 --patience 4`；这是起始配置，实际收敛情况以验证曲线为准。
- 学习率调度：`--lr-scheduler-type constant_with_warmup --warmup-ratio 0.1`；也支持 `linear` 和 `cosine`。
- 模型选优：默认 `--selection-metric loss`，越低越好；可用 `--selection-metric macro_f1` 按分类分数选优。
- 显存不足：加 `--batch-size 2 --gradient-accumulation-steps 8`，必要时将 `--max-length` 降为 128。
- GPU 不支持 FP16 或出现数值问题：加 `--no-fp16`；CPU 调试用 `--cpu`。
- 每次训练使用新的 `--output-dir`；脚本拒绝覆盖非空输出目录。

每轮同时记录验证损失、准确率、Macro-F1、`mean_confidence` 和 `brier_score`。
`mean_confidence` 是平均最高 Softmax 概率，越高不一定越好；`brier_score` 是六类概率与真实
one-hot 标签之间的平方误差之和再取样本平均，范围为 0 到 2，越低越好，错误的高置信预测会受到更大惩罚。
选择模型时结合分类效果和概率误差判断，训练本身不会保证概率已经校准。
最终模型确定后，使用独立校准数据选择路由接受阈值和分差。

如果要从当前基线继续微调，同时保持原来的训练/验证划分，可运行：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --model-path models/intent-macbert-200-v2-epoch6/best --train-file models/intent-macbert-200-v2-epoch6/train_split.jsonl --validation-file models/intent-macbert-200-v2-epoch6/validation_split.jsonl --output-dir models/intent-macbert-next-continued
```

这会从旧模型权重开始，以新的优化器和学习率调度执行额外微调；`--epochs` 表示额外轮数，
不恢复旧训练的优化器、调度器或步数。继续微调时，`--max-length`、`--history-rounds` 应与旧模型
保存的输入配置一致；当前基线对应默认的 256 和 3。

当前问题独立放在第一个文本段，保留完整 token；剩余预算保留最近历史的尾部。
默认最多保留最近 6 条历史消息（3 轮）。当前问题本身超预算会报错，而不会静默丢弃条件。
推理脚本读取保存的输入配置，使用完全相同的编码方式。

输出包括：

- `<output-dir>/best/`：最佳模型、分词器、标签映射、输入配置。
- `validation_metrics.json`：最佳模型的验证损失、准确率、Macro-F1、概率指标、各类别精确率/召回率/F1、混淆矩阵，以及模型选优依据。
- `training_history.json`：完整训练日志，包含每轮验证指标和训练过程的损失、学习率。
- `train_split.jsonl`、`validation_split.jsonl`：本次实际数据划分。
- Trainer 检查点：包含优化器等训练状态；`save_total_limit=1`，训练完成后只保留最佳检查点，不能依赖它回看任意历史轮次。

`best/` 是前述选优指标最佳的模型，不一定来自最后一轮。查看 `validation_metrics.json`
中的 `best_checkpoint` 确认实际步数。推理和级联应使用带 `intent_classifier_config.json` 的
`best/` 目录，原始 Trainer 检查点不包含这个自定义输入配置。

## 分类预测

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.predict --query "哪些基金持有宁德时代"
```

指定其他训练输出时：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.predict --model-path models/intent-macbert-next/best --query "不要推荐，只比较159103和159104的费率"
```

有历史时，把消息列表保存为 JSON 文件，并通过 `--history-file path/to/history.json` 传入。
输出包含 `intent`、`confidence` 和六类概率。softmax 分数未经校准，不能直接当作可靠的正确率；
回退阈值需要在独立验证数据上选择。

## 独立测试集评测

`data/test.jsonl` 是额外的 20 条测试数据，不包含在训练用的 200 条中。
类别分布、历史对照和来源见 [数据说明](data/README.md#额外的-20-条测试数据)。
不要加入训练集，也不要使用其结果选择阈值或反复调参；阈值选择使用验证集。
20 条数据中错一条会使准确率变化 5 个百分点，不足以代表真实线上分布。

只检查数据及与训练数据的重叠：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.evaluate --check-data-only
```

评测已训练的模型，路径应指向带有 intent_classifier_config.json 的 best 目录：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.evaluate --model-path models/intent-macbert-200-v2-epoch6/best
```

默认自动使用 GPU；CPU 评测加 `--cpu`。如训练输出目录不同，请修改 `--model-path`。
默认读取 `data/test.jsonl`，可用 `--test-file` 指定其他六类测试集。
默认与 `data/demo.jsonl`、模型旁保存的实际 train_split.jsonl / validation_split.jsonl 检查
完全相同的 query/history 和共享 group_id；也可重复使用 `--reference-file` 增加其他数据文件。
这些检查不能判断所有近义改写是否泄漏，仍需人工维护训练、验证、测试的边界。

报告默认保存在 `intent_classifier/reports/classifier/<时间戳>/`，也可以用 `--output-dir` 指定新的空目录：

- `metrics.json`：准确率、Macro-F1、Weighted-F1、各类别指标、混淆矩阵、
  有/无历史分组准确率、各测试标签准确率，以及设备和输入配置。
- `predictions.jsonl`：逐条真实标签、预测标签、六类概率、历史、token 数和延迟。
- `errors.jsonl`：只保留分类错误的样本，方便分析误判。

推理保持与 predict.py 相同的默认模型精度，单条请求（batch=1）。
延迟先预热 3 次，排除加载和预热时间，包含分词、张量构造与设备传输、前向计算、
softmax 及概率转回 CPU；GPU 计时会同步。报告给出均值、P50 和 P95，20 条的尾延迟仅供参考。
`--warmup-steps` 可修改预热次数。模型加载时间单独记录。

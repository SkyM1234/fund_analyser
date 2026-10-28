# 本地六分类模型

使用本地 MacBERT 微调项目的六种意图，也可通过 `--model-path` 换成本地 RBT3。
这里只预测 `intent`，不生成 `scope_basis`、`resolved_query` 或 `needs_clarification`。
脚本独立于后端，暂未接入线上路由。

## 环境

推荐 Python 3.11。当前 `E:\python_envs\llm` 已包含所需依赖。
在其他环境安装时，先按显卡和驱动选择 PyTorch 的 CUDA 版本，再安装独立依赖：

```powershell
python -m pip install -r intent_classifier/requirements.txt
```

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

`data/demo.jsonl` 已扩充为 200 条起步数据，其中 138 条参考 `markdown_mineru` 中的 17 份基金年报，
另有 62 条用于问候、越界、推荐请求和分类边界。32 条包含历史对话。
样本均包含 `id`、`group_id`、`source_refs` 和 `origin`；来源路径、行号和摘录见 `data/sources.json`。
参考年报是为了生成真实的基金名称、字段和概念问法，意图标签按项目规则编写；
问句不是年报原文，也不表示年报中一定有满足筛选条件的基金。
这批合成数据可用于第一轮训练，不能作为上线效果依据。正式评估应补充审核过的真实数据和独立测试集。
详细类别分布和标注边界见 `data/README.md`。
已有 `backend/eval/datasets/routing.jsonl` 请保留做回归测试，避免加入训练集。

## 开始训练

在项目根目录运行，使用这 200 条数据训练（自动按 group_id 隔离训练集和验证集）：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --train-file intent_classifier/routers/classifier/data/demo.jsonl
```

正式训练可显式提供训练集和验证集（将路径换成实际文件）：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.train --train-file intent_classifier/routers/classifier/data/train.jsonl --validation-file intent_classifier/routers/classifier/data/validation.jsonl --output-dir models/intent-macbert-v1
```

默认从 `models/chinese-macbert-base` 离线加载，输入上限 256 token，单卡 batch=4，梯度累积 4 步，
有效 batch=16，学习率 2e-5，最多 5 个 epoch。CUDA 下自动使用 FP16 和梯度检查点，
验证集 Macro-F1 连续两轮不改善则提前停止，最后保存验证分数最好的模型。
这是一组适合尝试的 6GB 显存配置，实际占用取决于其他 GPU 进程。

- 显存不足：加 `--batch-size 2 --gradient-accumulation-steps 8`，必要时将 `--max-length` 降为 128。
- GPU 不支持 FP16 或出现数值问题：加 `--no-fp16`；CPU 调试用 `--cpu`。
- 每次训练使用新的 `--output-dir`；脚本拒绝覆盖非空输出目录。

当前问题独立放在第一个文本段，保留完整 token；剩余预算保留最近历史的尾部。
默认最多保留最近 6 条历史消息（3 轮）。当前问题本身超预算会报错，而不会静默丢弃条件。
推理脚本读取保存的输入配置，使用完全相同的编码方式。

输出包括：

- `models/intent-macbert/best/`：训练后模型、分词器、标签映射、输入配置。
- `validation_metrics.json`：准确率、Macro-F1、各类别精确率/召回率/F1、混淆矩阵。
- `train_split.jsonl`、`validation_split.jsonl`：本次实际数据划分。
- Trainer 自动保存的检查点：包含优化器等训练状态。

## 分类预测

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.predict --query "哪些基金持有宁德时代"
```

指定其他训练输出时：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.predict --model-path models/intent-macbert-v1/best --query "不要推荐，只比较159103和159104的费率"
```

有历史时，把消息列表保存为 JSON 文件，并通过 `--history-file path/to/history.json` 传入。
输出包含 `intent`、`confidence` 和六类概率。softmax 分数未经校准，不能直接当作可靠的正确率；
回退阈值需要在独立验证数据上选择。

## 独立测试集评测

`data/test.jsonl` 是额外的 20 条测试数据，不包含在训练用的 200 条中。
六类数量依次为：chitchat 3、out_of_scope 3、sensitive 4、fund_query 4、
fund_screening 3、general_finance 3；其中 10 条带历史，10 条不带历史。
包含相同当前问题“还有哪些风险？”在不同历史下分别标为 fund_query 和 general_finance 的对照。
测试数据由助手按项目规则编写，适合初步评测，尚未经过独立人工审核。
不要加入训练集，也不要使用其结果选择阈值或反复调参；阈值选择使用验证集。
20 条数据中错一条会使准确率变化 5 个百分点，不足以代表真实线上分布。

只检查数据及与训练数据的重叠：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.evaluate --check-data-only
```

评测已训练的模型，路径应指向带有 intent_classifier_config.json 的 best 目录：

```powershell
& E:\python_envs\llm\python.exe -m intent_classifier.routers.classifier.evaluate --model-path models/intent-macbert-200-v1/best
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

# embedding-service Docker 化踩坑总结

从裸机 `python embedding_service.py` 到 `docker compose up -d` 一键启动，整个过程踩过的坑。


## 设计要点（面试可讲的点）


### 1. Milvus `load_collection` 是异步的——"连上了"≠"能查了"

这是本次改动中最值得讲的根因分析：

- **排查路径**：同一查询重启后偶发失效 → 排除模型和数据问题（不变） → 锁定启动阶段的竞态窗口 → 代码里搜不到任何 `load_collection` 调用 → 怀疑是 Milvus collection 未完全 load 时 `search()` 静默返回不完整结果。
- **验证方法**：读 pymilvus 3.0.0 的 `MilvusClient.get_load_state` 源码，确认它返回 `{"state": LoadState}` 枚举（`NotExist`/`NotLoad`/`Loading`/`Loaded`），且 `load_collection()` 只是**发起**加载请求，不阻塞等待完成。`IVF_FLAT` 索引的加载是异步的——需要构建倒排文件表结构、将数据段从磁盘/对象存储载入内存——在此期间 `search()` 不会报错或阻塞，而是基于已载入的部分分片返回结果。
- **修复**：在 FastAPI `lifespan` 的 startup 阶段，对 `fund_reports_mineru` 和 `fund_index` 两个 collection 显式调用 `load_collection()` 后轮询 `get_load_state()`，直到状态变为 `LoadState.Loaded` 才 `yield` 放行服务。`time.sleep` 阻塞轮询在 startup 阶段是合理的——服务在完全 ready 之前不应该开始接受请求。

面试可讲：**排查"中间件客户端库的静默失败"类 bug 时，不能只看自己的代码有没有显式错误处理——要确认客户端的每个"看起来是同步"的 API 调用，在服务端是否真的是同步完成的。** `MilvusClient()` 构造函数返回了、`has_collection()` 返回 True，不等于 collection 已经可以正常服务查询——这和数据库连接池的"连接建立了但后台健康检查还没跑"是同一类问题。

另一点：`get_load_state` 返回的是 `LoadState` 枚举实例（`int` 子类），不是字符串 `"Loaded"`。一度写成 `state.get("state") == "Loaded"` 会导致条件永远不满足、60 秒后超时报错——这是"不读库源码、凭字段名猜测类型"的典型坑。修复方式不是给字符串比较加引号包裹，而是从 `pymilvus.client.types` 导入 `LoadState` 枚举做 identity 比较。


### 2. Docker 内网 vs 宿主机端口——为什么 localhost:19595 在容器里连不上 Milvus

- **现象**：embedding-service 容器启动后连接 Milvus 报错，不断重启。宿主机 `curl localhost:19595` 正常。
- **根因**：`docker-compose.yml` 里的端口映射 `"19595:19530"` 含义是 **宿主机 19595 → 容器内 19530**。Milvus standalone 进程在容器内监听的是 gRPC 端口 19530，不是 19595。同一个 Docker 网络内的其他容器（如 embedding-service）直接访问 `milvus-standalone:19530`，走的是 Docker 内部 DNS + 容器真实端口。`19595` 是宿主机侧的映射端口，只在宿主机上有效。
- **修复**：`MILVUS_URL` 环境变量在 compose 里设为 `http://milvus-standalone:19530`（容器内部端口），裸机运行时仍用 `http://localhost:19595`（宿主机映射端口）。

面试可讲：**Docker 网络有三种视角：宿主机视角（端口映射）、容器间视角（内部端口 + service name DNS）、容器自身视角（localhost:真实端口）。** 很多"容器 A 连不上容器 B"的问题都是把宿主机映射端口当成了容器间通信端口。


### 3. `pip install FlagEmbedding` 自动拉 transformers 5.x ——语义版本破坏

- **现象**：Docker 日志 `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model`，但相同版本组合在 Windows 裸机上正常。
- **排查**：`docker exec embedding-service pip show transformers` 发现装了 **5.14.1**，而 Windows 环境是 **4.57.6**。`FlagEmbedding==1.4.0` 的 `setup.py` 没有约束 transformers 上限（`transformers>=4.36`），pip 解析到最新的 5.x 大版本。transformers 5.x 删除了 `PreTrainedTokenizer.prepare_for_model` 方法（该方法在 4.x 中是 deprecated 但保留的），导致 `FlagReranker.compute_score` 内部调用链断裂。
- **修复**：Dockerfile 中显式加 `"transformers>=4.36,<5.0"` 约束。

面试可讲：**容器化的可重复构建（reproducible build）需要用显式版本约束对抗 pip 的依赖解析。** 裸机上"刚好能用"不代表 Docker 里也能用——裸机环境日积月累的依赖已经被锁定在某个 `pip freeze` 快照里，而 Docker 每次 `docker build` 都是全新的依赖解析。解决方式不是"把裸机的 `pip freeze` 全量搬进 Dockerfile"，而是搞清楚**每个包的有效版本范围**，用 `>=lower,<upper` 约束上下界。


### 4. PyTorch CPU vs CUDA ——pip 在 Docker 构建期"看不到" GPU

- **现象**：容器启动成功，API 正常返回，但 `nvidia-smi` 在宿主机上看不到显存占用，所有推理落在 CPU 上。
- **根因**：`docker build` 阶段没有 GPU 设备。pip 在安装 `FlagEmbedding` 时检测到系统无 CUDA，自动拉取 PyTorch CPU 版本。即使 `docker-compose.yml` 里配了 `deploy.resources.reservations.devices`（运行时 GPU 透传），构建阶段装的是 CPU 版 PyTorch，运行时也变不出 CUDA kernel。
- **修复**：Dockerfile 中分两步——**先**从 PyTorch 官方 CUDA 索引显式安装 CUDA 版 PyTorch（`--index-url https://download.pytorch.org/whl/cu124`），**再**装 FlagEmbedding。第二步检测到已有 CUDA 版 PyTorch，不会覆盖安装 CPU 版。

面试可讲：**Docker 构建时环境和运行时环境是不同的——`docker build` 没有 GPU、没有网络特权、没有 volume mount。** 任何依赖"运行时硬件特性"的 pip 包（PyTorch、TensorFlow、JAX）都需要在 Dockerfile 里显式指定 CUDA 版本，不能依赖 pip 自动检测——pip 检测的是构建环境，不是运行环境。


### 5. WSL2 GPU-PV 虚拟化——容器内 `nvidia-smi` 失败 ≠ GPU 不可用

- **现象链条**：
  1. WSL2 内 `nvidia-smi` 正常
  2. PowerShell 运行 `nbody` CUDA 示例容器 正常
  3. `docker info | grep Runtimes` 看不到 nvidia runtime
- **根因**：Windows Docker Desktop 使用 **WSL2 GPU-PV（GPU Paravirtualization）虚拟化方案**，而非传统的 NVIDIA Container Toolkit。GPU-PV 只转发 **CUDA 计算 API**（模型推理、矩阵运算等所有深度学习框架依赖的核心 API），但**不转发 NVML 显卡管理接口**。`nvidia-smi` 依赖 NVML 库读取硬件寄存器来获取温度、功耗、显存占用等信息，在 GPU-PV 模式下这些寄存器对容器不可见，所以 `nvidia-smi` 在容器内无法执行。但 PyTorch/TensorRT/vLLM 等框架的 CUDA kernel 调用完全不受影响——它们走的是另一条 API 路径。
- **如何验证 GPU 真的在用**：
  ```python
  import torch
  print(torch.cuda.is_available())        # True
  print(torch.cuda.get_device_name(0))    # "NVIDIA GeForce RTX 3060 Laptop GPU"
  print(torch.cuda.memory_allocated())    # 实际已分配显存
  ```
  或者在模型推理前后对比宿主机 `nvidia-smi` 的显存占用变化——容器内看不到，但宿主机看得到。

面试可讲：**GPU 虚拟化方案不是只有一种——"能跑 CUDA 程序"和"能调 nvidia-smi"是两个正交的能力。** 传统方案（NVIDIA Container Toolkit + `--gpus all`）通过透传整个 GPU 设备节点实现，nvidia-smi 在容器内可用；WSL2 GPU-PV 方案通过 API 转发实现，更轻量但损失了管理接口。排查"GPU 到底有没有生效"时的正确验证链是：① `torch.cuda.is_available()` → ② 跑一次前向推理看速度/显存 → ③ 宿主机 `nvidia-smi` 看显存变化。**不要**把"容器里 nvidia-smi 没输出"当成"GPU 不可用"——那只是在当前虚拟化方案下 NVML 不可访问而已。


### 6. PyTorch 版本和 CUDA 索引的对应关系——2.6 之后 cu121 不再更新

- **现象**：`ERROR: Could not find a version that satisfies the requirement torch>=2.6.0`，但 PyTorch 明明已经发到 2.7+。
- **根因**：`https://download.pytorch.org/whl/cu121` 索引只更新到 PyTorch 2.5.1。PyTorch 2.6 起官方最低支持 CUDA 12.4，cu121 索引停止更新。而 `transformers<5.0` 解析到的最新版（4.57.x）出于安全原因（CVE-2025-32434）要求 `torch>=2.6`——于是形成了一个交叉依赖死锁：cu121 没 torch 2.6，pip 默认源也没 CUDA torch。
- **修复**：将 PyTorch 安装索引从 `cu121` 切换到 `cu124`。

面试可讲：**GPU 相关的基础设施栈有三层独立的版本号——CUDA 驱动版本、PyTorch CUDA 编译版本、transformers 框架版本——它们之间的兼容矩阵不是线性的。** 排查"Docker 里能跑但裸机不能"或反过来时，第一个检查项应该是 `torch.__version__` 和 `torch.version.cuda`，而不是 transformers 的版本号。


### 7. 代码卷挂载——告别"每次改一行就 rebuild"

- **痛点**：调试期间改一次 `embedding_service.py` 就要 `docker compose up -d --build`——重新拉基础镜像、重新 pip install、重新 COPY。每次改动等待 2-5 分钟。
- **方案**：在 `docker-compose.yml` 里把 `embedding_service.py` 和 `hybrid_search.py` 以 `:ro` 卷挂载进容器。Dockerfile 里的 `COPY` 保留作为生产部署的 fallback（没有卷时使用镜像内置的代码）。开发时改代码 → `docker compose restart`（3 秒），生产部署时去掉卷挂载 → 镜像自包含。
- **权衡**：卷挂载意味着容器依赖宿主机文件系统，破坏了"镜像即交付物"的纯净化。但开发/调试阶段的时间收益远大于纯净化收益。这和 Kubernetes 里用 ConfigMap 挂载配置文件的权衡是同构的。

面试可讲：**Docker 开发效率的核心杠杆是"哪些文件需要 rebuild，哪些只需要 restart"。** 分层策略：依赖（Dockerfile RUN）→ rebuild；代码（volume mount）→ restart；模型（volume mount）→ 无需重启。把变化频率不同的文件分到不同的"变更层"，最小化每次改动的时间成本。


### 8. 压测必须分层——GPU 快不等于完整聊天链路快

最初只通过 `/api/chat/stream` 做端到端 SSE 压测，结果只能看到首 Token 延迟和总耗时，无法判断耗时来自 GPU、MCP、LLM 还是 Celery 排队。因此增加了两套职责不同的压测脚本：

- `backend/scripts/gpu_load_test.py`：直接请求 embedding service，绕过 FastAPI 聊天接口、Celery、LangGraph、MCP 和外部 LLM，单独测量 embedding、Milvus 检索和 reranker。
- `backend/scripts/chat_load_test.py`：请求完整 `/api/chat/stream` 链路，测量真实用户视角下的首 Token 延迟、总耗时、成功率和吞吐量。

`chat_load_test.py` 同时替代了原来语义不明确的 `load_test.py`，并做了以下标准化：

- 注册和登录属于准备阶段，不计入正式请求延迟。
- `--requests` 表示总请求数，`--concurrency` 表示客户端最大并发数，避免把总量和并发度混为一谈。
- 支持 `--warmup`，减少首次连接、模型预热和缓存冷启动对结果的干扰。
- 按 SSE 规范解析 `event:` 和多行 `data:`，以首个非空 `token` 事件计算 TTFT。
- 输出成功率、请求吞吐量、成功吞吐量以及 avg/p50/p90/p95/max/min。
- 支持 `--same-session`，用于验证服务端会话锁是否正确拒绝同一会话的并发请求。

面试可讲：**性能分析的第一步不是调参数，而是建立可隔离的测量边界。** 端到端测试回答"用户实际等多久"，组件测试回答"时间花在哪一层"；只有两者结合，才能避免把上游排队误判成 GPU 性能问题。


### 9. Reranker 的 `max_batch` 必须是硬上限——触发阈值不等于执行上限

- **原实现的问题**：`BatchReranker` 在累计 pair 数达到 `max_batch` 后只负责提前触发 flush，但 `_flush()` 会一次取走当时队列中的全部请求，再把所有 pairs 合并后一次性传给 `compute_score()`。高并发下，事件循环在 flush 真正执行前可能继续接收请求，因此实际输入规模可能远大于配置的 `max_batch`。
- **风险**：超大 batch 会造成显存峰值、单次推理长尾甚至 OOM；此时把 `BATCH_RERANK_MAX=128` 理解为"单次最多 128 条"是错误的，它原来只是一个触发阈值。
- **修复**：合并请求后按 `self.max_batch` 对 `all_pairs` 严格切片，每个 chunk 单独调用 `compute_score()`，最后校验 score 数量并按原请求边界拆分结果。
- **异常处理**：任意 chunk 失败时，将同一个 flush 批次中尚未完成的 Future 全部设置异常，避免请求永久等待。
- **参数校验**：`max_batch <= 0` 在初始化阶段直接抛出 `ValueError`，避免运行时出现无效步长或静默错误。

面试可讲：**批处理系统中要区分"何时触发 flush"和"单次执行多少数据"。** 前者控制等待时间和聚合效率，后者控制资源上限；只实现触发阈值而没有执行分片，并不能形成真正的背压保护。


### 10. Fast tokenizer 不是线程安全的——并发信号量不能代替模型实例锁

- **现象**：并发调用 `asyncio.to_thread(model.encode/compute_score)` 时，fast tokenizer 偶发报 `Already borrowed`。
- **根因**：`GPU_SEMAPHORE` 只限制全局允许多少个 GPU 推理进入临界区。当其值大于 1 时，同一个 encoder 或 reranker 实例仍可能被多个线程同时调用；Hugging Face fast tokenizer 和模型实例不保证这种并发方式是线程安全的。
- **修复**：`BatchEncoder` 和 `BatchReranker` 分别增加独立的 `_inference_lock`。同一个模型实例一次只执行一个推理批次，但请求仍可在等待窗口内合并成动态 batch。
- **结果**：并发请求不再直接竞争同一个 tokenizer；吞吐优化由"同实例并行调用"改为"短时间聚合后批量推理"，避免用线程并发换取不稳定的表面吞吐。

`GPU_CONCURRENCY` 的准确语义是：**embedding service 进程内，允许同时进入 GPU 推理区的批次数量上限**。它不是 HTTP 并发数，也不是 Celery worker 数：

- 设置为 `1`：encoder 与 reranker 的 GPU 推理全局串行，显存峰值最低，适合 6GB 显存。
- 设置大于 `1`：可能允许 encoder 与 reranker 两个不同模型实例重叠执行，但同一个模型仍受各自 `_inference_lock` 保护。
- 从 `1` 提高到 `8` 不保证提高吞吐量。单卡已经被一个 batch 跑满、reranker 占主导或动态批处理已充分利用 GPU 时，增加并发只会增加调度竞争和显存风险。

本机测试中，提高 `GPU_CONCURRENCY` 没有带来明显吞吐增益，因此最终默认值使用 `1`，优先保证稳定性。修改 `.env` 后需要重新创建 embedding-service 容器，已经启动的容器不会自动读取宿主机上更新后的环境变量。

面试可讲：**并发上限、批处理大小和线程安全是三个不同维度。** 信号量解决资源竞争，动态 batch 解决 GPU 利用率，模型实例锁解决第三方推理库的线程安全；不能用一个 `concurrency` 参数同时替代三者。


### 11. 推理后显存不下降——通常是 CUDA 缓存，不是内存泄漏

- **现象**：embedding service 刚启动时显存占用较低，第一次请求后显存明显上升；请求结束后显存没有回到启动值。
- **原因**：模型可能在首次推理时完成 CUDA 上下文、kernel workspace 和中间 tensor 的懒初始化；PyTorch CUDA caching allocator 会保留已经申请的显存块，供后续请求复用，避免每次推理重复执行昂贵的 `cudaMalloc/cudaFree`。
- **判断方法**：如果显存经过预热后稳定在一个平台值，连续压测不再持续增长，通常属于正常缓存。只有显存随每轮请求持续单调增长、最终 OOM，才应继续排查对象引用或 tensor 生命周期泄漏。
- **监控口径**：`nvidia-smi dmon` 中 `fb` 才是显存占用 MB；`mem` 是显存控制器利用率，不是"显存已占百分比"。Windows/WSL2 下该利用率还可能存在长期显示 100% 的监控偏差。
- **不建议**：不要在每个请求后调用 `torch.cuda.empty_cache()`。它不能释放模型权重，只会降低缓存复用效率并增加延迟；只有明确需要把空闲缓存让给同卡其他进程时才考虑主动清理。

面试可讲：**GPU 内存观察要区分 allocated、reserved 和设备层看到的占用。** 深度学习框架保留显存是性能策略，不应仅凭"请求结束后 nvidia-smi 数字没下降"判定泄漏。


### 12. MCP 不会把所有 worker 串行化——真正的并发边界在 Celery worker

完整聊天链路为：

```text
FastAPI /api/chat/stream
  → Celery agent_queue
  → LangGraph 多 Agent 工作流
  → 每个 worker 自己的 MCP stdio 子进程
  → rag-mcp HTTP 请求
  → 单个 embedding-service
```

backend worker 使用 `--pool=solo`。每个 worker 容器是独立 OS 进程，并在启动时创建自己的 asyncio loop、MCP client、rag-mcp 子进程和 stdio 管道。因此使用 `docker compose --scale worker=N` 时，得到的是 N 条独立 MCP 调用路径，而不是 N 个 worker 共用一个 MCP 串行连接。

MCP 会增加 JSON-RPC 编解码、stdio 进程通信和一次 HTTP 转发，但直接 GPU 压测与完整 SSE 压测的数量级差异表明，它不是几十秒 TTFT 的主要来源。完整聊天的首 Token 只从 `synthesizer` 节点开始转发，因此 TTFT 还包含：

```text
Celery 排队
→ LLM 意图分类
→ Supervisor 规划
→ Agent 多轮 LLM/tool 调用
→ MCP/RAG
→ 自检与反思
→ Synthesizer 首次输出
```

当请求数大于 worker 数时，多出的请求会在 `agent_queue` 等待，并随着 worker 释放而分批进入完整工作流；这部分排队时间也会被客户端计入首 Token 延迟。

worker 可以通过 Docker Compose 的 service scale 机制横向扩容。扩容后，每个新增 worker 都会获得独立的 MCP 子进程和资源连接，但仍然共享外部 LLM、Redis、数据库和 embedding-service。

面试可讲：**排查异步链路时必须画出每一层的并发边界。** HTTP 接受的连接数不等于正在执行的 Agent 数，MCP 独立管道数也不等于 GPU 可同时执行的 reranker batch 数。队列、进程、协议连接和 GPU 临界区分别有自己的容量上限。


### 13. 压测结论——先消除 worker 排队，再优化首 Token 前的串行工作流

关键结论：

1. 增加 worker 后，完整 SSE 的吞吐量明显提高、首 Token 延迟明显下降，说明 Celery 排队是端到端链路中的主要瓶颈之一。
2. 请求数超过 worker 数时会形成滚动的多批执行。吞吐量可能继续提高，但排队会增加 TTFT 和尾延迟。
3. 直接 GPU RAG 明显快于完整 SSE，且 GPU `SM` 呈现短时满载、长时间空闲的脉冲形态，说明完整链路的大部分时间在等待 LLM、Agent 编排或队列，而不是持续等待 GPU。
4. 完整 SSE 中 TTFT 与总耗时较为接近，说明主要延迟发生在答案开始输出之前。继续优化时，应优先减少简单事实查询中的串行 LLM 调用，例如合并"基金识别 + RAG 搜索"工具、为明确查询增加快速路径、减少不必要的规划和反思步骤。
5. worker 不是越多越好。扩容会增加 MCP 子进程、数据库连接、Redis 连接和外部 LLM 并发，应通过固定负载下的对照测试寻找吞吐量、尾延迟和资源消耗之间的平衡点。

下一步最有价值的观测项是把 TTFT 拆成阶段耗时：

```text
请求提交 → worker 开始：Celery queue delay
worker 开始 → plan_created：路由与规划
tool_call → tool_result：MCP/RAG
最后一次 tool_result → 首 Token：反思与 Synthesizer
首 Token → done：答案输出
```

面试可讲：**吞吐量增加和单请求延迟下降不是同一个目标。** 增加 worker 可以提高系统吞吐并减少排队，但无法消除单个请求内部的串行 LLM 链路；容量规划需要同时观察 throughput、TTFT p95、错误率和资源饱和点。


## 最终 Dockerfile 关键决策一览

| 决策 | 选择 | 原因 |
|------|------|------|
| 基础镜像 | `python:3.11-slim-bookworm` | 足够小，PyTorch 自带 CUDA runtime 无需系统级 CUDA |
| PyTorch 来源 | PyTorch 官方 cu124 索引 | pip 默认源不给 CUDA 版本 |
| transformers 约束 | `>=4.36,<5.0` | FlagEmbedding 1.4.0 不兼容 5.x |
| 模型 | volume 挂载 `:ro` | 4GB 太大，换模型不用 rebuild |
| 代码 | volume 挂载 `:ro`（开发）/ COPY（生产） | 开发效率 vs 交付纯度 |
| GPU | `deploy.resources.reservations.devices` | Docker Desktop + WSL2 原生支持，无需额外安装 toolkit |
| ipc | `host` | PyTorch CUDA 共享内存通信需要 |
| memlock | `-1` | CUDA 需要锁定物理内存页 |
| pip 源 | 清华镜像（除 torch 外） | torch CUDA 包清华不一定有，走官方 |
| GPU 推理并发 | `GPU_CONCURRENCY=1` | 单卡 6GB 下优先稳定；同模型实例由独立 inference lock 串行保护 |
| Reranker 动态批处理 | `wait=50ms, max=128` | 短窗口聚合请求，单次 `compute_score` 严格限制为 128 pairs |

## 最终 docker-compose.yml 服务总览

| 服务 | 镜像 | 端口 | 作用 |
|------|------|------|------|
| etcd | `quay.io/coreos/etcd:v3.5.5` | — | Milvus 元数据 |
| minio | `minio/minio:RELEASE.2023-03-20…` | 9000/9001 | Milvus 对象存储 |
| milvus-standalone | `milvusdb/milvus:v2.5.9` | 19595:19530 | 向量数据库 |
| attu | `zilliz/attu:latest` | 8000:3000 | Milvus 管理 UI |
| **embedding-service** | **自建** | **8001** | **BGE-M3 + Reranker 查询** |

## 快速启动

```bash
cd milvus-docker
cp .env.example .env       # 首次，编辑配置
docker compose up -d       # 首次加 --build
curl http://localhost:8001/health
```

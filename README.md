# 基金分析 Agent

面向中国基金公开披露数据的问答与分析系统。系统检索已入库的基金年报，结合实时基金数据生成可追溯的回答；前端实时展示任务规划、Agent 执行和工具调用过程。

系统用于信息检索、数据整理和分析辅助，不提供基金推荐、买卖建议或收益预测。

## 功能

- 基金年报检索：基于 BGE-M3、Milvus 和重排序的混合检索。
- 基金范围识别：支持基金代码、名称、别名和跨基金问题。
- 多 Agent 编排：路由、范围确认、规划、RAG 检索、实时数据、结果分析、冲突反思和合规检查。
- 流式问答：FastAPI 通过 SSE 实时返回回答、计划、Agent 状态和工具调用。
- 用户与会话：注册登录、刷新令牌、持久化会话和会话回溯。
- 管理能力：用户状态、密码重置、令牌撤销和全局会话管理。
- 数据管道：基金年报下载、MinerU PDF 解析、Markdown 清洗和视觉补充、向量入库。
- 评测框架：使用 LangSmith 评估检索质量和端到端回答质量。

## 架构

```text
Vue 3 前端 (5173)
        |
        v
FastAPI API (8800) <----- Redis Pub/Sub -----> Celery Worker
        |                                           |
        |                                           v
        |                                   LangGraph 多 Agent
        |                                    |               |
        v                                    v               v
MySQL / PostgreSQL / Redis                rag-mcp      cn-funds-mcp
                                               |
                                               v
                                    Embedding Service (8001)
                                               |
                                               v
                              Milvus + BGE-M3 / BGE-Reranker-v2-m3

离线数据流：
基金年报 PDF -> MinerU -> Markdown 清洗/图表补充 -> 向量化 -> Milvus
```

## 仓库结构

```text
fund_analyser/
├── frontend/                 # Vue 3 前端
├── frontend-docker/          # 前端 Docker Compose
├── backend/                  # FastAPI、Celery、LangGraph、MCP 与测试
├── backend-docker/           # 后端 API 和 Worker Docker Compose
├── sql-docker/               # PostgreSQL、MySQL、Redis
├── milvus-docker/            # Milvus、Attu
├── embedding-service-docker/ # embedding/RAG 服务
├── vectorize/                # PDF 到 Milvus 的离线入库脚本
├── mineru-docker/            # MinerU PDF 解析服务
├── annual_reports_2025_funds/# 原始基金年报（本地数据，已忽略）
├── markdown_mineru/          # 解析后的 Markdown（本地数据，已忽略）
├── embedding_model/          # BGE 模型文件（本地数据，已忽略）
└── download_fund_reports.py  # 基金年报下载与校验脚本
```

## 快速启动

以下步骤启动已具备数据的完整问答系统。首次导入或更新年报数据，见后文的“数据入库”。

### 前置条件

- Docker Desktop 与 Docker Compose。
- NVIDIA GPU、NVIDIA 驱动和 NVIDIA Container Toolkit，用于 embedding/RAG 服务。
- 本地模型目录 `embedding_model/`，其中包含 `bge-m3` 和 `bge-reranker-v2-m3`。
- DeepSeek 或其他 OpenAI 兼容 LLM 的 API 密钥。

### 1. 配置环境变量

为各 Compose 项目准备环境文件。后端和 embedding 服务中的 MySQL 密码必须一致。

```powershell
# 在仓库根目录执行
Copy-Item milvus-docker\.env.example milvus-docker\.env
Copy-Item embedding-service-docker\.env.example embedding-service-docker\.env
Copy-Item backend-docker\.env.example backend-docker\.env
New-Item -ItemType File -Path sql-docker\.env
```

在 `sql-docker/.env` 中设置：

```dotenv
MYSQL_ROOT_PASSWORD=replace-with-a-strong-password
```

在 `embedding-service-docker/.env` 中设置相同的 `MYSQL_PASSWORD`，并确认模型目录：

```dotenv
MYSQL_PASSWORD=replace-with-a-strong-password
EMBEDDING_MODEL_DIR=../embedding_model
```

在 `backend-docker/.env` 中至少设置：

```dotenv
DEEPSEEK_API_KEY=sk-your-key
MYSQL_ROOT_PASSWORD=replace-with-a-strong-password
JWT_SECRET_KEY=replace-with-a-random-production-secret
CORS_ORIGINS=http://localhost:5173
```

### 2. 启动基础服务

```powershell
cd sql-docker
docker compose up -d

cd ..\milvus-docker
docker compose up -d

cd ..\embedding-service-docker
docker compose up -d

```

等待 embedding 服务就绪：

```powershell
Invoke-WebRequest http://localhost:8001/health
```

### 3. 启动后端

```powershell
cd ..\backend-docker
docker compose up -d --build

# 首次启动或迁移更新后执行
docker compose exec backend alembic -c /app/alembic.ini upgrade head
docker compose exec backend python -m scripts.create_admin --username admin --password change-me
```

默认启动一个 Celery Worker。要增加并发处理能力，扩展独立 Worker 实例：

```powershell
docker compose up -d --scale worker=4
```

### 4. 启动前端

```powershell
cd ..\frontend-docker
docker compose up -d --build
```

打开 `http://localhost:5173`，使用注册账号登录。后端 OpenAPI 文档位于 `http://localhost:8800/docs`，健康检查位于 `http://localhost:8800/api/health`。

`backend-docker` 依赖 `sql-docker_default` 与 `milvus` 两个 Docker 网络，因此必须在启动后端前完成前两步。

## 数据入库

数据入库仅在首次建设索引或需要更新基金年报时执行。所有脚本均从仓库根目录运行。

### 1. 下载基金年报

```powershell
pip install pymupdf requests
python download_fund_reports.py
```

默认下载或断点续传至 `annual_reports_2025_funds/`，并生成 PDF 校验记录。使用 `--help` 查看数量、输出目录和清理选项。

### 2. 启动 MinerU 并解析 PDF

MinerU 的部署文件位于 `mineru-docker/`。根据本机显存选择 `compose.yaml` 或 `compose-6g.yaml`，先参考 [MINERU_DEPLOY_SUMMARY.md](mineru-docker/MINERU_DEPLOY_SUMMARY.md) 配置模型和服务。

```powershell
cd mineru-docker
# 使用 compose.yaml
docker compose --profile api up -d

# 若选择 compose-6g.yaml，改用：
# docker compose -f compose-6g.yaml --profile api up -d

cd ..
pip install -r vectorize\requirements.txt
python vectorize\batch_parse_pdfs.py
```

解析结果写入 `markdown_mineru/`。批处理脚本默认连接 `http://localhost:8000/file_parse`，并会跳过 PDF 校验失败的文件。

### 3. 清洗并补充 Markdown

```powershell
python vectorize\md_preprocessor.py
python vectorize\md_image_table_analyze.py
```

第二个脚本使用 Qwen VL 分析图片和表格，运行前设置 `QWEN_API_KEY`。

### 4. 写入 Milvus

确认 `milvus-docker` 已启动，且 `embedding_model/bge-m3` 可访问：

```powershell
python vectorize\vectorize_to_milvus.py
python vectorize\query_fund_report.py --list
```

向量化过程会将年报分块写入 `fund_reports_mineru`，并构建用于基金名称识别的 `fund_index`。

详细流程及 CPU 环境注意事项见 [vectorize/README.md](vectorize/README.md)。

## 本地开发

### 前端

```powershell
cd frontend
npm install
$env:VITE_API_TARGET = "http://localhost:8800"
npm run dev
```

开发服务器运行在 `http://localhost:5173`，`/api` 请求代理到 `VITE_API_TARGET`。

### 后端

先启动 `sql-docker` 和 `milvus-docker`，再安装后端与 MCP 依赖：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r mcp\rag-mcp\requirements.txt
npm install --prefix mcp\cn-funds-mcp-master
```

设置开发环境变量并启动 API、Worker：

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key"
$env:GPU_HOST = "localhost"
$env:POSTGRES_HOST = "localhost"
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PASSWORD = "your-mysql-password"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:CELERY_BROKER_URL = "redis://localhost:6379/1"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/2"
$env:JWT_SECRET_KEY = "local-development-secret"

alembic upgrade head
python -m app.main
```

在另一个终端启动 Worker。Windows 下必须使用 `solo` pool：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
celery -A app.core.celery_app worker --pool=solo -Q agent_queue --loglevel=info
```

更多后端配置、API、SSE 事件和运维说明见 [backend/README.md](backend/README.md)。

## 常用地址与端口

| 服务 | 地址/端口 | 用途 |
| --- | --- | --- |
| 前端 | `http://localhost:5173` | 用户界面。 |
| 后端 API | `http://localhost:8800` | FastAPI 服务。 |
| API 文档 | `http://localhost:8800/docs` | OpenAPI / Swagger UI。 |
| Embedding/RAG | `http://localhost:8001` | 检索、重排序和健康检查。 |
| Attu | `http://localhost:8000` | Milvus 管理界面。 |
| Milvus | `localhost:19595` | 向量数据库宿主机端口。 |
| PostgreSQL | `localhost:5432` | LangGraph checkpoint。 |
| MySQL | `localhost:3306` | 业务数据。 |
| Redis | `localhost:6379` | 缓存、消息和 Celery。 |

## 测试与评测

后端单元测试：

```powershell
cd backend
pytest -q
```

评测框架位于 `backend/eval/`，提供 RAG 检索与 Agent 端到端回答的 LangSmith 评测：

```powershell
cd backend
pip install -r eval\requirements.txt
Copy-Item eval\.env.example eval\.env
python -m eval.runners.upload_dataset --kind all --mode append
python -m eval.runners.run_retrieval_eval --experiment-prefix baseline
python -m eval.runners.run_answer_eval --experiment-prefix baseline --concurrency 2
```

评测配置、指标和数据集格式见 [backend/eval/README.md](backend/eval/README.md)。

## 运维与排查

```powershell
# 查看各组件日志
cd sql-docker
docker compose logs -f

cd ..\milvus-docker
docker compose logs -f embedding-service

cd ..\backend-docker
docker compose logs -f backend
docker compose logs -f worker
```

- SSE 无响应：确认 Worker 正在消费 `agent_queue`，并检查 Redis 的 broker/result 配置。
- MCP 初始化失败：确认容器中可执行 Node.js 和 Python，检查 embedding 服务的 `/health`。
- 登录或迁移失败：确认 MySQL、PostgreSQL、Redis 已启动，且三个 Compose 项目的 MySQL 密码一致。
- GPU 服务无法启动：检查 NVIDIA Container Toolkit、GPU 可见性和 `EMBEDDING_MODEL_DIR` 挂载路径。

## 关联文档

- [前端说明](frontend/README.md)
- [后端说明](backend/README.md)
- [后端架构](backend/docs/ARCHITECTURE.md)
- [向量化流程](vectorize/README.md)
- [评测框架](backend/eval/README.md)
- [MinerU 部署摘要](mineru-docker/MINERU_DEPLOY_SUMMARY.md)

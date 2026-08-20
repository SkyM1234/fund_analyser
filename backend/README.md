# 基金分析 Agent 后端

基金分析系统的后端服务，基于 FastAPI、Celery 和 LangGraph。服务提供认证、会话和流式问答 API；问答任务由 Celery Worker 执行，通过 Redis Pub/Sub 转发为 SSE 事件。

默认接入两个 MCP 服务：

- `rag-mcp`：调用 embedding/RAG 服务，检索基金年报与披露数据。
- `cn-funds-mcp`：查询基金实时估值、持仓和基础信息等市场数据。

系统不提供基金推荐、买卖建议或收益预测。

## 架构概览

```text
Frontend
   |
   v
FastAPI API (8800) ---- Redis Pub/Sub ---- Celery Worker (agent_queue)
   |                                            |
   |                                            v
   |                                     LangGraph 多 Agent
   |                                      |             |
   v                                      v             v
MySQL / PostgreSQL / Redis             rag-mcp     cn-funds-mcp
                                            |
                                            v
                                  Embedding / RAG 服务 (8001)
```

- MySQL：用户、刷新令牌、会话归属、合规审计等业务数据。
- PostgreSQL：LangGraph checkpoint 和会话消息状态。
- Redis：缓存、限流、令牌状态、Celery broker/result backend 和 SSE 事件转发。
- 多 Agent：路由、基金范围识别、任务规划、RAG/市场数据获取、依赖结果分析、结果冲突反思、合规检查和最终回答生成。

详细的图编排说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 前置条件

- Python 3.11+。
- Node.js 18+，用于启动 `cn-funds-mcp`。
- 可访问的 DeepSeek 兼容 LLM API。
- PostgreSQL、MySQL、Redis。
- 可访问的 embedding/RAG 服务；默认地址为 `http://<GPU_HOST>:8001`。

Docker 部署还需要 Docker Compose。embedding 服务使用 GPU 时，需要 NVIDIA Container Toolkit 和本地 embedding 模型目录。

## Docker 部署

仓库将基础设施、向量检索服务和后端拆分为三个 Compose 项目。请按下列顺序启动。

```powershell
# 在仓库根目录执行
cd sql-docker
docker compose up -d

cd ..\milvus-docker
docker compose up -d

cd ..\backend-docker
Copy-Item .env.example .env
```

编辑 `backend-docker/.env`，至少设置以下变量：

```dotenv
DEEPSEEK_API_KEY=sk-your-key
MYSQL_ROOT_PASSWORD=replace-with-the-password-used-by-sql-docker
JWT_SECRET_KEY=replace-with-a-random-production-secret
CORS_ORIGINS=http://localhost:5173
```

启动 API 和一个 Worker：

```powershell
docker compose up -d --build
```

生产环境建议通过增加独立 Worker 实例提升并发，而非增加单个 Worker 的并发度：

```powershell
docker compose up -d --scale worker=4
```

首次部署或迁移更新后执行：

```powershell
docker compose exec backend alembic -c /app/alembic.ini upgrade head
docker compose exec backend python -m scripts.create_admin --username admin --password change-me
```

服务启动后可访问：

- API：`http://localhost:8800`
- OpenAPI 文档：`http://localhost:8800/docs`
- 健康检查：`http://localhost:8800/api/health`

`backend-docker` 使用 `sql-docker_default` 和 `milvus` 两个外部网络；若前两个 Compose 项目未启动，后端容器无法启动或无法连接依赖服务。

## 本地开发

在 `backend/` 目录中创建虚拟环境并安装依赖：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r mcp\rag-mcp\requirements.txt
npm install --prefix mcp\cn-funds-mcp-master
```

设置本地环境变量。以下示例假设数据库、Redis 和 embedding 服务均通过本机端口访问：

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key"
$env:GPU_HOST = "localhost"
$env:GPU_PORT = "8001"
$env:POSTGRES_HOST = "localhost"
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PASSWORD = "your-mysql-password"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:CELERY_BROKER_URL = "redis://localhost:6379/1"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/2"
$env:JWT_SECRET_KEY = "local-development-secret"
$env:CORS_ORIGINS = "http://localhost:5173"
```

先执行数据库迁移：

```powershell
alembic upgrade head
```

分别启动 API 和 Worker：

```powershell
# 终端 1：Windows 下建议使用模块启动方式
python -m app.main

# 终端 2：Windows 必须使用 solo pool
celery -A app.core.celery_app worker --pool=solo -Q agent_queue --loglevel=info
```

Worker 是独立进程，会自行初始化 Redis、数据库连接池和 MCP 客户端。Windows 下不要省略 `--pool=solo`；默认的 prefork 不受支持。每个 Worker 只消费一个 agent 任务，扩展吞吐量时启动多个 Worker 进程。

## 配置

所有配置由环境变量读取，定义见 `app/core/config.py`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | LLM API 密钥。代码同时兼容 `LLM_API_KEY`。 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 LLM 基础地址。 |
| `LLM_MODEL` | `deepseek-v4-flash` | LLM 模型名。 |
| `LLM_MAX_CONCURRENCY` | `10` | 单进程内 LLM 调用并发上限。 |
| `GPU_HOST` / `GPU_PORT` | `localhost` / `8001` | embedding/RAG 服务地址。 |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `GPU_HOST` / `5432` | LangGraph checkpoint 数据库地址。 |
| `MYSQL_HOST` / `MYSQL_PORT` | `GPU_HOST` / `3306` | MySQL 地址。 |
| `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | `root` / 空 / `fund_analyser` | MySQL 连接信息。 |
| `REDIS_URL` | `redis://<GPU_HOST>:6379/0` | 业务 Redis 库。 |
| `CELERY_BROKER_URL` | Redis DB 1 | Celery broker。 |
| `CELERY_RESULT_BACKEND` | Redis DB 2 | Celery result backend。 |
| `JWT_SECRET_KEY` | 开发默认值 | JWT 签名密钥；生产环境必须替换。 |
| `CORS_ORIGINS` | `http://localhost:5173` | 逗号分隔的允许来源。 |
| `MCP_ENABLED` | `true` | 是否初始化 MCP 服务。 |
| `MCP_MAX_TOTAL_CALLS` | `40` | 单用户限流窗口内的 MCP 总调用上限。 |
| `MCP_MAX_CALLS_PER_TOOL` | `20` | 单用户、单工具的限流窗口内调用上限。 |

默认 MCP 配置会从 `backend/mcp/` 中启动 `cn-funds-mcp-master` 和 `rag-mcp`。因此本地运行时，`node` 和当前 Python 环境必须位于 `PATH` 中。

## API

除注册、登录、刷新令牌和健康检查外，接口均需携带：

```http
Authorization: Bearer <access_token>
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 服务名称和版本。 |
| `GET` | `/api/health` | API、LLM 配置和 embedding/RAG 服务状态。 |
| `POST` | `/api/auth/register` | 注册用户。 |
| `POST` | `/api/auth/login` | 登录并获取 access/refresh token。 |
| `POST` | `/api/auth/refresh` | 使用 refresh token 换取新 token 对。 |
| `POST` | `/api/auth/logout` | 撤销 refresh token。 |
| `GET` | `/api/auth/me` | 当前用户信息。 |
| `POST` | `/api/chat/stream` | 提交问答任务并接收 SSE 流。 |
| `GET` | `/api/sessions` | 当前用户的会话列表。 |
| `GET` | `/api/sessions/{thread_id}` | 会话详情和工具调用记录。 |
| `DELETE` | `/api/sessions/{thread_id}` | 删除会话及其 checkpoint。 |
| `POST` | `/api/sessions/{thread_id}/rewind` | 清除会话 checkpoint，供客户端回溯后重新对话。 |
| `GET` | `/api/mcp/stats` | 当前用户的 MCP 调用统计。 |
| `GET` | `/api/mcp/tools` | 可用 MCP 工具列表。 |

`/api/admin/*` 接口仅管理员可访问，包含用户启停/删除、密码重置、令牌撤销和全局会话管理；完整契约以 `/docs` 为准。

### 流式问答

`POST /api/chat/stream` 的请求体：

```json
{
  "message": "比较两只基金近一期披露的重仓行业",
  "session_id": "a-client-generated-uuid",
  "history": []
}
```

响应为 `text/event-stream`。前端应至少处理以下事件：

| SSE 事件 | 说明 |
| --- | --- |
| `message_start` | 新的助手消息开始；合规重试后也会重新发送，客户端应重置当前回答内容。 |
| `token` | 增量文本，数据格式为 `{"delta": "..."}`。 |
| `route_result` | 路由识别结果。 |
| `plan_created` | Supervisor 生成的任务计划。 |
| `agent_start` / `agent_end` | 子 Agent 的执行状态。 |
| `tool_call` / `tool_result` | MCP 工具调用及其结果。 |
| `retrieval_context` | 实际用于生成回答的检索上下文元数据。 |
| `retry_notice` / `tool_retry` | 合规或工具重试信息。 |
| `done` | 请求完成。 |
| `error` | 任务或事件转发错误。 |

## 测试与运维

```powershell
# 在 backend/ 下运行单元测试
pytest -q

# 创建或提升管理员
python -m scripts.create_admin --username admin --password change-me --email admin@example.com

# Docker 日志
cd ..\backend-docker
docker compose logs -f backend
docker compose logs -f worker
```

应用日志写入 `logs/fund_api.log`；Docker 部署下该目录使用名为 `backend_logs` 的卷。

## 常见问题

### SSE 一直等待，没有事件

确认 Worker 已运行，且它消费 `agent_queue`：

```powershell
celery -A app.core.celery_app worker --pool=solo -Q agent_queue --loglevel=info
```

同时检查 `CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND` 和 `REDIS_URL` 是否指向同一个 Redis 实例的正确 DB。

### MCP 初始化失败

检查 `node`、`python` 是否可执行，以及两个 MCP 目录的依赖是否安装。还应检查 embedding 服务：

```powershell
Invoke-WebRequest http://localhost:8001/health
```

可通过 `/api/health` 查看后端感知到的 RAG 服务状态，通过 `/api/mcp/tools` 确认可用工具。

### 数据库迁移或登录失败

确认 MySQL、PostgreSQL 和 Redis 已启动，环境变量与部署地址一致，然后重新执行：

```powershell
alembic upgrade head
```

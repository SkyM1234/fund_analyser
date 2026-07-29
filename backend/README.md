# 基金分析 Agent - 后端

## 快速开始

```bash
# Docker 部署（推荐）
cd backend-docker
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
docker compose up -d --build

# 启动多个 worker（多用户并发）
docker compose up -d --scale worker=4

# 数据库迁移
docker exec fund-backend alembic -c /app/alembic.ini upgrade head

# 创建管理员
docker exec fund-backend python -m scripts.create_admin --username alice --password secret123
```

裸机开发：

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-xxx
uvicorn app.main:app --host 0.0.0.0 --port 8800

# Celery worker（另开终端）
celery -A app.core.celery_app worker --pool=solo --loglevel=info -Q default,agent_queue
```

## 项目结构

```
backend/
├── app/
│   ├── agent/          # LangGraph Agent
│   ├── api/            # FastAPI 路由
│   ├── services/       # 核心服务（路由、MCP）
│   └── core/           # 配置
├── mcp/
│   ├── rag-mcp/        # RAG MCP 服务（Python）
│   └── cn-funds-mcp-master/  # 基金查询 MCP（Node.js）
├── scripts/            # 管理脚本
├── alembic/            # 数据库迁移
└── docs/
    └── ARCHITECTURE.md  # 详细架构文档
```

## 核心功能

- **多Agent架构** - Supervisor + 专家Agent，任务拆解与专业化执行
- **智能路由** - 三层路由（规则/实体/LLM）自动识别用户意图
- **敏感过滤** - 自动拒绝投资建议、基金推荐等敏感问题
- **RAG 检索** - 基于向量数据库的基金报告检索
- **MCP 工具** - 统一的工具调用接口（25+ 工具）
- **会话管理** - PostgreSQL 持久化对话历史

## Agent 架构

多Agent架构（Supervisor + 专家Agent）：

```
        ┌─────────────────────────┐
        │   Supervisor (Planner)  │
        └───┬───┬───┬───┬─────────┘
            │   │   │   │
   ┌────────┘   │   │   └────────────┐
   ▼            ▼   ▼                ▼
┌──────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│ RAG  │  │ 实时行情  │  │ 对比分析 │  │ 合规检查 │
│Agent │  │  Agent   │  │  Agent   │  │  Agent   │
└──────┘  └──────────┘  └─────────┘  └──────────┘
```

**优势**：
- 单Agent上下文体积下降 60%+
- 复杂任务成功率提升（显式任务拆解）
- 合规审计点独立，更可控

## MCP 工具

### RAG 工具（rag-mcp）
- `rag_search` - 检索基金报告
- `rag_identify_funds` - 从查询中语义识别基金代码
- `rag_health` - 健康检查
- `rag_list_funds` - 基金清单

### 基金工具（cn-funds-mcp）
- `search_fund` - 搜索基金
- `get_fund_info` - 基金详情
- `get_fund_estimate` - 实时估值
- `get_fund_position` - 持仓信息
- ... 共 20+ 工具

## 环境变量

```bash
# LLM（必填）
DEEPSEEK_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash

# GPU RAG
GPU_HOST=localhost
GPU_PORT=8001

# PostgreSQL（Docker 部署时自动指向 postgres 服务）
POSTGRES_HOST=postgres

# MySQL（Docker 部署时自动指向 mysql 服务）
MYSQL_HOST=mysql
MYSQL_PASSWORD=skc393720

# Redis
REDIS_URL=redis://redis:6379/0

# JWT
JWT_SECRET_KEY=dev-secret-change-me

# CORS（允许的前端地址，逗号分隔）
CORS_ORIGINS=http://localhost:5173,http://SKC:5173
```

## Celery 任务队列

聊天请求（`/api/chat/stream`）不在 FastAPI 进程内直接跑 agent 图，而是投递给
Celery worker 执行；FastAPI 通过 Redis Pub/Sub 订阅 worker 发出的事件并转发给
浏览器的 SSE 连接。

```bash
# Docker 部署
docker compose up -d --scale worker=4     # 4 个并发 worker

# 裸机开发
celery -A app.core.celery_app worker --pool=solo --loglevel=info -Q default,agent_queue

# 监控面板（可选）
celery -A app.core.celery_app flower --port=5555
```

**注意**：
- Docker 下用 `--scale worker=N` 扩展，每个 worker 是独立的 solo 进程，
  拥有自己的 MCP client 和 stdio 管道，真正并发无竞态。
- 每个 worker 进程会自行初始化 MCP client、checkpoint 连接池、MySQL engine、
  Redis client（与 FastAPI 的 lifespan 逻辑完全独立）。
- broker/result backend 使用 Redis 的 db 1/2，与业务缓存/锁使用的 db 0 隔离
  （见 `CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND`），避免 `FLUSHDB` 误操作
  互相影响。
- Windows 上必须用 `--pool=solo`；Linux/Docker 下也推荐 solo + 多实例，
  因为 MCP stdio 连接不能跨 fork 共享。

## API 端点

- `POST /api/chat/stream` - 流式对话（SSE）
- `GET /api/health` - 健康检查
- `GET /api/sessions` - 会话列表
- `POST /api/auth/login` - 登录
- `POST /api/auth/register` - 注册

## 故障排查

### MCP 连接失败
```bash
# 检查 embedding 服务
curl http://localhost:8001/health

# 查看日志
docker logs fund-backend
```

### Agent 无响应
检查 MCP 工具是否加载成功，日志中应该有：
```
✓ [worker] MCP client initialized with 2 servers
✓ [worker] Loaded and cached N MCP tools
```

---

详细架构说明：[ARCHITECTURE.md](docs/ARCHITECTURE.md)

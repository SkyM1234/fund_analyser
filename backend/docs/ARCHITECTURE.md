# 项目架构文档

## 概述

基金分析 Agent 系统，采用多Agent架构，通过 MCP 协议统一访问外部服务。

## 架构图

```
用户 → React 前端 → FastAPI 后端
                      ├─ 多Agent系统 (LangGraph)
                      │   ├─ Supervisor (任务规划)
                      │   ├─ RAG Agent (年报检索)
                      │   ├─ Market Agent (实时数据)
                      │   ├─ Synthesizer (结果汇总)
                      │   └─ Compliance Agent (合规检查)
                      │   └─ MCP Client → MCP 工具
                      │       ├─ rag-mcp (GPU RAG 服务)
                      │       └─ cn-funds-mcp (天天基金 API)
                      └─ 查询路由器 (意图识别)
```

## 核心模块

### 1. MCP 服务层 (`backend/mcp/`)

所有外部服务访问的统一入口：

**rag-mcp** (Python) - GPU RAG 服务封装
- `src/server.py` - MCP 服务器
- `src/rag_client.py` - HTTP 客户端（内部）
- `src/fund_code_matcher.py` - 基金代码字符串匹配兜底（内部）

提供工具：
- `rag_search` - 检索基金报告（混合检索+重排序）
- `rag_identify_funds` - 语义识别基金代码（两级RAG第一级）
- `rag_health` - 健康检查
- `rag_stats` - 统计信息
- `rag_list_funds` - 基金清单
- `rag_match_fund_codes` - 基金代码字符串匹配（语义识别的兜底）

**cn-funds-mcp** (Node.js) - 天天基金 API 封装
- 20+ 基金查询工具（实时净值、持仓、经理等）

### 2. Agent 系统 (`backend/app/agent/`)

**多Agent架构** - 基于 LangGraph 的多Agent编排

核心组件：
- `multi_agent_state.py` - 状态定义
- `supervisor.py` - 任务规划与调度
- `rag_agent.py` - 年报检索专家
- `market_agent.py` - 实时数据专家
- `compliance_agent.py` - 合规检查
- `synthesizer.py` - 结果汇总
- `multi_agent_controller.py` - LangGraph编排控制器

详见 [MULTI_AGENT.md](MULTI_AGENT.md)

### 3. 查询路由器 (`backend/app/services/router.py`)

两层路由策略：
1. **规则过滤** - 闲聊/越界/敏感问题（快速拒绝）
2. **LLM 分类** - 结合近几轮对话历史判断意图（闲聊/越界/敏感/基金查询/基金筛选/通用知识）

路由结果：
- `intent` - 意图类型

基金代码识别不再由路由器承担，交由后续 Agent（如 rag_agent）在需要时通过语义识别（`rag_identify_funds`）或字符串匹配兜底（`rag_match_fund_codes`）动态获取，避免路由阶段凭 LLM 知识猜测代码导致的误判。

### 4. MCP 适配器 (`backend/app/services/`)

轻量层，仅用于协议转换：
- `rag_client_mcp.py` - RAG 调用适配
- `mcp_client.py` - MCP 客户端管理器

## 配置

### MCP 服务配置 (`backend/app/core/config.py`)

```python
MCP_SERVERS = [
    {
        "name": "cn-funds-mcp",
        "command": "node",
        "args": ["src/index.js"],
        "cwd": "backend/mcp/cn-funds-mcp-master"
    },
    {
        "name": "rag-mcp",
        "command": "python",
        "args": ["src/server.py"],
        "cwd": "backend/mcp/rag-mcp",
        "env": {
            "GPU_HOST": "localhost",
            "GPU_PORT": "8001"
        }
    }
]
```

### 环境变量

```bash
# LLM
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-v4-flash

# GPU RAG
GPU_HOST=localhost
GPU_PORT=8001

# PostgreSQL (Checkpoint)
POSTGRES_URI=postgresql://fund_user:fund_pass@localhost:5432/fund_chat
```

## 启动

```bash
cd backend

# 安装依赖
pip install -r requirements.txt
cd mcp/rag-mcp && pip install -r requirements.txt && cd ../..

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8800
```

## 测试

```bash
# 测试 RAG MCP 服务
python test_new_rag_mcp.py

# 验证迁移完成
python verify_migration.py

# 检查 MCP 状态
python check_mcp_status.py
```

## 关键特性

### 1. 敏感问题过滤
自动拒绝投资建议、基金推荐、收益预测等敏感问题。

### 2. 智能路由
- 规则匹配快速拦截
- 基金代码/名称自动识别
- LLM 意图分类兜底

### 3. 统一工具接口
Agent 通过 MCP 调用所有工具，无需区分来源。

### 4. 会话持久化
使用 PostgreSQL 存储对话历史，支持断点续聊。

## 重要约束

⚠️ **不提供投资建议**
- 不推荐具体基金
- 不预测收益
- 不给出买卖建议

## 故障排查

### MCP 连接失败
```bash
# 检查 GPU 服务
curl http://localhost:8001/health

# 查看日志
cat backend/logs/fund_api.log | grep "rag-mcp"
```

### Agent 无响应
检查 MCP 工具是否加载：
```python
from app.services.mcp_client import get_mcp_client
mcp_client = await get_mcp_client()
tools = await mcp_client.list_all_tools()
print(f"Loaded {len(tools)} tools")
```

---

**架构版本**: v2.0 (Full MCP)  
**更新日期**: 2026-06-26

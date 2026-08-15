# RAG MCP Server

GPU RAG 服务的 MCP 封装，提供基金报告检索功能。

## 提供的工具

1. **rag_search** - 检索基金报告（混合检索+重排序）
2. **rag_identify_funds** - 从查询中语义识别基金代码
3. **rag_health** - 健康检查
4. **rag_list_funds** - 基金清单

## 目录结构

```
rag-mcp/
├── src/
│   ├── server.py             # MCP 服务器
│   ├── rag_client.py         # HTTP 客户端
│   └── fund_code_matcher.py  # 基金代码字符串匹配兜底
└── requirements.txt
```

## 使用

由 FastAPI 后端自动启动，配置见 `backend/app/core/config.py`。

## 独立测试

```bash
# 直接调 embedding service
curl -X POST http://localhost:8001/fund_reports/search \
  -H "Content-Type: application/json" \
  -d '{"query":"投资策略","top_k":10}'
```

## 环境变量

- `GPU_HOST` - GPU 服务器地址（默认 localhost）
- `GPU_PORT` - GPU 服务器端口（默认 8001）

## 工具示例

### rag_search
```json
{
  "query": "投资策略",
  "filter_fund_code": "161725",
  "top_k": 5
}
```

---

详细架构见 [backend/docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)

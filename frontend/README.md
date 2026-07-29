# 前端：基金问答助手

Vue 3 + Vite + TypeScript + Pinia + Element Plus + Markdown，通过 SSE 实时渲染助手输出与工具调用过程。

## 启动

### Docker 部署（推荐）

```bash
cd frontend-docker
docker compose up -d --build
```

访问 `http://localhost:5173`

### 裸机开发

```bash
cd frontend
npm install
npm run dev
```

默认端口 `5173`，已配置 `/api` 反代到后端 `http://backend:8800`（Docker 内）
或 `http://localhost:8800`（裸机开发）。

## 目录

```
frontend/src/
├── api/chat.ts              # 手动 SSE 解析（fetch + ReadableStream）
├── stores/chat.ts           # Pinia: 消息列表、工具步骤、流式状态
├── components/
│   ├── ChatWindow.vue       # 聊天主面板
│   ├── MessageBubble.vue    # 单条消息（Markdown 渲染）
│   ├── ToolCallCard.vue     # 工具调用折叠卡
│   └── SessionSidebar.vue   # 会话历史侧栏（Drawer）
├── App.vue
└── main.ts
```

## 流式事件 → UI 映射

| 后端事件 | 前端动作 |
|---|---|
| `token` | 追加到当前助手气泡正文 |
| `tool_call` | 新增一个 ToolCallCard（可折叠） |
| `tool_result` | 把结果填入对应 ToolCallCard |
| `done` | 关闭 pending |
| `error` | 显示错误条 |

## 局域网访问

同一局域网下，另一台电脑通过主机名 `SKC` 访问：

1. `frontend/vite.config.ts` 已配置 `allowedHosts: ['skc', '.local']`
2. `backend-docker/.env` 中 `CORS_ORIGINS` 需包含 `http://SKC:5173`
3. 另一台电脑浏览器访问 `http://SKC:5173`

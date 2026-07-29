# ============================================================================
# Backend — FastAPI + MCP（Python + Node.js）
# ============================================================================
# 构建:  docker build -t fund-backend:latest .
# 启动:  docker compose up -d
# ============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# 安装 Node.js 22.x（cn-funds-mcp MCP server 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 cn-funds-mcp 的 Node 依赖
COPY mcp/cn-funds-mcp-master/package*.json /app/mcp/cn-funds-mcp-master/
RUN cd /app/mcp/cn-funds-mcp-master && \
    npm config set registry https://registry.npmmirror.com && \
    npm install

# 安装 Python 依赖
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r /tmp/requirements.txt

EXPOSE 8800

CMD ["python", "-m", "app.main"]

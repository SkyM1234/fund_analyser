# ============================================================================
# Frontend — Vite Dev Server（Node.js 镜像）
# ============================================================================
# 构建:  docker build -t fund-frontend:latest .
# 启动:  docker compose up -d
# ============================================================================
FROM node:22-alpine

WORKDIR /app

# 复制 package 文件安装依赖
COPY package*.json ./
RUN npm config set registry https://registry.npmmirror.com && \
    npm install --legacy-peer-deps

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ============================================================================
# Embedding Service — Docker Image（GPU 机器上运行）
# ============================================================================
# 构建:  docker build -t embedding-service:latest .
# 启动:  在 docker-compose.yml 同目录下 docker compose up -d
# ============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# 先装 CUDA 版 PyTorch（官方源，防止 FlagEmbedding 自动装 CPU 版）
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch>=2.6.0"

# 安装 embedding_service 其余依赖（清华源）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    FlagEmbedding==1.4.0 \
    "transformers>=4.36,<5.0" \
    pymilvus==3.0.0 \
    fastapi==0.136.1 \
    uvicorn==0.40.0 \
    pydantic==2.10.6 \
    numpy==2.4.4

# 复制应用代码
COPY embedding_service.py /app/
COPY hybrid_search.py /app/

EXPOSE 8001

CMD ["python", "-u", "embedding_service.py"]

# MinerU 2.5 Docker 部署总结

## 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3060 Laptop (6GB) |
| CPU | AMD Ryzen 7 5800H (8C16T) |
| 内存 | 16GB |
| 系统 | Windows 11 25H2 |
| CUDA | 13.2 (Driver 596.49) |

## 部署步骤

### 1. 安装 WSL2 + Docker Desktop

```powershell
wsl --update
```
安装 Docker Desktop，启用 WSL2 集成。

### 2. 调整 WSL2 资源

创建 `%USERPROFILE%\.wslconfig`：
```ini
[wsl2]
memory=12GB
swap=8GB
```

重启 WSL：`wsl --shutdown`

### 3. 更新 NVIDIA 驱动

CUDA ≥12.8 才能用 vLLM。RTX 3060 建议 Game Ready ≥570.65，实测 596.49 稳定。

### 4. 构建 Docker 镜像

```powershell
git clone https://github.com/opendatalab/MinerU.git
cd MinerU\docker\china
docker build -t mineru:latest -f Dockerfile .
```

国内用户用 `china/Dockerfile`（DaoCloud 加速基础镜像 + 阿里云 pip 源 + ModelScope 模型下载）。构建约 40 分钟，最终镜像 ~43GB。

### 5. 编写 Compose 配置

创建 `compose-6g.yaml`：
```yaml
services:
  mineru-gradio:
    image: mineru:latest
    container_name: mineru-gradio
    restart: unless-stopped
    profiles: ["gradio"]
    ports:
      - "7860:7860"
    environment:
      MINERU_MODEL_SOURCE: local
    entrypoint: mineru-gradio
    command: >
      --server-name 0.0.0.0
      --server-port 7860
      --gpu-memory-utilization 0.55
      --max-num-seqs 1
      --max-model-len 3072
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    shm_size: "8gb"
    volumes:
      - ./data:/data
      - ./mineru.json:/root/mineru.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

### 6. 配置 LLM 标题分级

创建 `mineru.json`：
```json
{
    "llm-aided-config": {
        "title_aided": {
            "api_key": "你的API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "enable_thinking": false,
            "enable": true
        }
    }
}
```

### 7. 启动

```powershell
docker compose -f compose-6g.yaml --profile gradio up -d
```

访问 `http://localhost:7860`。

## 6GB 显存调参

| 参数 | 值 | 说明 |
|------|------|------|
| `gpu-memory-utilization` | 0.55 | 给 vLLM 分配 3.3GB（模型 2.16GB + 缓存 0.74GB） |
| `max-num-seqs` | 1 | 串行处理，减少显存占用 |
| `max-model-len` | 3072 | 每页最多 3072 tokens（太小会截断，太大显存不够） |

**注意**：`gpu-memory-utilization` 不是"越小越省"，而是控制 vLLM 可用显存比例。过低会导致模型装不下。6GB 显存建议从 0.5 起调。

## 常用命令

```powershell
# 启动
docker compose -f compose-6g.yaml --profile gradio up -d

# 停止/重启
docker compose -f compose-6g.yaml --profile gradio down
docker compose -f compose-6g.yaml --profile gradio restart

# 查看日志
docker logs mineru-gradio
```

## 修改容器内文件（挂载方式）

如果需要对 MinerU 做自定义修改，把文件从容器拷出来，改完再挂载回去：

```powershell
# 1. 复制文件到本地
docker cp mineru-gradio:/usr/local/lib/python3.12/dist-packages/mineru/目标路径/文件.py ./文件.py

# 2. 编辑本地文件

# 3. 在 compose-6g.yaml 添加挂载
#   volumes:
#     - ./文件.py:/usr/local/lib/python3.12/dist-packages/mineru/目标路径/文件.py

# 4. 重建容器
docker compose -f compose-6g.yaml --profile gradio down
docker compose -f compose-6g.yaml --profile gradio up -d
```

**注意**：添加/删除挂载必须 `down && up`，`restart` 不更新挂载配置。修改配置类文件（如 `mineru.json`）只需 `restart`。

## 可选参数

--enforce-eager
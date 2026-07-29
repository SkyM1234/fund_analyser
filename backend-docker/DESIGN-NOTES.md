# Backend Docker 化踩坑总结

从裸机 `python -m app.main` 到 `docker compose up -d --scale worker=4`，整个过程踩过的坑。


## 设计要点（面试可讲的点）


### 1. `requirements.txt` 精确钉死——Docker 构建时的依赖地狱

这是在 Docker 化过程中出现次数最多的报错类型：`ResolutionImpossible`。

- **现象链条**：
  1. `pydantic==2.10.6` + `mcp==1.26.0` → 冲突，mcp 要求 `>=2.11.0`
  2. `pydantic_core==2.27.2` → 阻止 pydantic 升级（每个 pydantic 版本绑定一个精确的 pydantic-core 版本）
  3. `psycopg==3.3.4` → Linux slim 镜像没有 libpq，import 报错 `no pq wrapper available`
- **根因**：`pip freeze` 导出的是**当前环境的快照**，包含大量传递依赖（pydantic-core 是 pydantic 的依赖，不是项目的直接依赖）。Docker 构建时是全新的依赖解析环境，pip 会严格检查所有版本约束——快照中某个传递依赖的精确版本可能与另一个包的更新版本冲突。
- **修复**：
  - 将传递依赖（`pydantic_core`）的版本钉死**删除**，让 pip 自动解析
  - 直接依赖用范围约束而非精确版本：`pydantic>=2.11.0,<3.0.0`
  - `psycopg` 改用 `psycopg[binary]`——自带 C 库静态链接，不依赖系统 libpq

面试可讲：**`pip freeze > requirements.txt` 不是"可重复构建"方案。** Docker 重新 build 时，pip 是从空白环境开始解析的——这和增量安装了一年的 conda 环境完全不同。正确的做法是：只钉**直接依赖**的版本范围，传递依赖交给 pip 解析。这和 Rust 的 `Cargo.lock`（锁依赖）vs `Cargo.toml`（声明依赖）的区别是同一类问题。


### 2. MCP stdio + Celery prefork = 死锁

这是整个架构中最关键的并发设计决策。

- **现象**：`--pool=prefork --concurrency=4` 会导致子进程共享父进程的 MCP stdio pipe，连接损坏。
- **根因**：MCP server 通过 **stdio 子进程** 通信（Python 的 `mcp` 包 + `StdioServerParameters` + `stdio_client`）。每个 MCP server 是一个独立的 OS 子进程，通过一对 stdin/stdout pipe 与主进程通信。当 Celery 使用 prefork pool 时：
  - `worker_init` 在主进程初始化 MCP（创建子进程 + pipe）
  - fork 后子进程继承 pipe fd，但**后台事件循环线程死亡**（Linux fork 只保留调用线程）
  - 结果：子进程中的 MCP 连接不可用
- **修复思路**（按优劣排序）：
  1. **多 solo worker**（当前方案）：`--scale worker=N`，每个 worker 独立进程、独立 MCP、独立 pipe。每个 worker 真正独立，扩展只需要 `docker compose up -d --scale worker=4`
  2. prefork + `worker_process_init`：每个 fork 子进程内重新初始化 MCP。但需要改信号绑定，且子进程异常退出时 MCP 子进程可能泄漏
  3. HTTP transport 替代 stdio：MCP server 改用 SSE/HTTP transport，天然支持多路复用

面试可讲：**进程池（prefork）和长连接（stdio pipe）是冲突的。** fork 能复制内存，但复制不了 I/O 连接和线程。MCP 引入了"每进程一个事件循环 + 每进程一组 stdio 子进程"的约束，这让 prefork 从"最佳实践"变成了"不可用"。架构决策的连锁反应：加一个 MCP → 废掉 prefork → 引入多实例部署模式 → 最终形成 `--scale worker=N` 的弹性伸缩方案。


### 3. Docker Compose 跨项目网络——`external: true` 的陷阱

- **现象**：`fund-backend` 容器启动后连不上 PostgreSQL，报 `could not translate host name "postgres"`。
- **根因**：项目和基础设施分在三个 compose 文件（`sql-docker/`、`milvus-docker/`、`backend-docker/`），各自有独立的默认网络。`backend-docker` 的容器默认只能看到同文件内的服务。
- **修复**：backend 显式加入两个外部网络：

  ```yaml
  networks:
    default:
      external: true
      name: sql-docker_default    # 访问 postgres / mysql / redis
    milvus:
      external: true
      name: milvus                # 访问 embedding-service / milvus
  ```

- **副作用**：跨项目的 `depends_on` + `condition: service_healthy` 不可用（Docker Compose 只能检查同文件内的服务健康状态）。解决方案：启动顺序由运维脚本/文档约定保证，不依赖 compose 的 `depends_on`。
- **副作用 2**：`docker compose down -v` 清理的是项目自身的卷，**不会清理外部网络的卷**——三个 compose 文件的卷需要各自管理。

面试可讲：**Docker Compose 的"项目"概念是按 compose 文件隔离的，不是按目录隔离的。** 微服务拆到多个 compose 文件后，服务发现、健康检查、卷管理都从"内置功能"变成了"需要手动设计"。这和 K8s 里跨 namespace 访问 Service 的问题是同构的——只不过 compose 没有 Ingress/Service Mesh 来兜底。


### 4. Volume 挂载 + 构建产物 = 静默覆盖

- **现象**：构建时 `npm install` 装了 cn-funds-mcp 的 Node 依赖，运行时 import 报 `Module not found`。
- **根因**：`docker-compose.yml` 里挂载 `../backend:/app:ro` 把整个宿主目录覆盖到容器 `/app`。构建时在镜像里装的 `/app/mcp/cn-funds-mcp-master/node_modules` 被宿主机目录（没有 node_modules）**完全覆盖**。
- **修复**：用**匿名卷**保护构建产物：

  ```yaml
  volumes:
    - ../backend:/app:ro
    - /app/mcp/cn-funds-mcp-master/node_modules   # 匿名卷，优先于宿主目录
  ```

  匿名卷在 Docker 的挂载优先级中高于 bind mount——`/app` 来自宿主，但 `/app/mcp/.../node_modules` 保留镜像内容。首次运行时 Docker 会将镜像中的内容复制到匿名卷；后续 rebuild 时，匿名卷**不会**自动更新——需要 `docker compose down -v` 清理。

面试可讲：**Docker 的挂载优先级：匿名卷 > bind mount > 镜像层。** 这个机制可以用"白名单"方式保护构建产物——大部分代码从宿主挂载（开发效率），少数构建产物保留在容器内（依赖隔离）。


### 5. `aiomysql` → `asyncmy` ——MySQL 异步驱动的坑

- **现象**：登出/会话列表偶发 `TypeError: AsyncAdapt_aiomysql_connection.ping() missing 1 required positional argument: 'reconnect'`。
- **根因**：`aiomysql 0.3.x` 修改了 `connection.ping()` 签名（增加了必填参数 `reconnect`），但 SQLAlchemy 的 MySQL 异步 dialect 调用 `ping()` 时不传这个参数。版本不兼容。
- **修复**：切换到 `asyncmy`——SQLAlchemy 官方推荐的 MySQL 异步驱动。API 设计与 `pymysql` 对齐，无此兼容性问题。改动涉及：
  1. `requirements.txt`：`aiomysql` → `asyncmy>=0.2.9`
  2. `config.py`：`mysql+aiomysql://` → `mysql+asyncmy://`
  3. `alembic/env.py`：同步驱动替换逻辑同步更新（`asyncmy` → `pymysql`）
  4. `requirements.txt` 额外加 `pymysql`（Alembic 迁移用同步驱动）

面试可讲：**选异步驱动时，优先看 SQLAlchemy 官方的 dialect 支持列表，而不是 PyPI 下载量。** `aiomysql` 下载量高但不代表兼容性好——SQLAlchemy 对它的测试覆盖不如 `asyncmy`。大多数"异步驱动连不上数据库"的问题，不是驱动有 bug，而是驱动和 ORM 之间的适配层（dialect）没跟上驱动自己的 breaking change。


### 6. 国内网络——Docker Hub + pip + deb 三连断

- **Docker Hub**：`docker.io` 被墙，Docker Desktop Settings → Docker Engine → `registry-mirrors` 配镜像加速器。
- **pip**：PyTorch CUDA 包清华源不一定有，Dockerfile 里 torch 走官方 `--index-url https://download.pytorch.org/whl/cu124`，其余走清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- **deb.nodesource.com**：安装 Node.js 时可能超时，备选方案是 `apt-get install nodejs npm`（版本旧但 cn-funds-mcp 够用）。


## 启动

```bash
cd backend-docker
cp .env.example .env   # 编辑填入 DEEPSEEK_API_KEY
docker compose up -d --build

# 数据库初始化（首次）
docker exec fund-backend alembic -c /app/alembic.ini upgrade head
docker exec fund-backend python -m scripts.create_admin --username alice --password secret123

# 多 worker 扩展
docker compose up -d --scale worker=4
```

# Frontend Docker 化踩坑总结

从裸机 `npm run dev` 到 `docker compose up -d` 一键启动，整个过程踩过的坑。


## 设计要点（面试可讲的点）


### 1. `crypto.randomUUID()` 只在安全上下文可用

- **现象**：本机 `localhost:5173` 正常，局域网另一台电脑访问 `http://SKC:5173` 白屏。控制台报 `crypto.randomUUID is not a function`。
- **根因**：Web Crypto API 的 `randomUUID()` 只在 **localhost** 或 **HTTPS** 下可用。`http://SKC:5173` 既不是 localhost 也不是 HTTPS，浏览器禁用了该 API。
- **修复**：增加 fallback——手动生成 UUID v4：

  ```typescript
  const uid = () => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID()
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16)
    })
  }
  ```

面试可讲：**Web API 的安全性限制是按源（origin）判断的，不是按 IP。** `http://192.168.0.119:5173` 和 `http://skc:5173` 虽然是"内网地址"，但对浏览器来说和公网 HTTP 一样不可信。这就是为什么很多内部工具用自签名证书也要上 HTTPS——不仅是防中间人，还是为了让浏览器开放完整的 Web API。


### 2. Vite `allowedHosts` ——开发服务器的主机名白名单

- **现象**：局域网电脑访问 `http://SKC:5173` 被 Vite 拦截，显示 `Blocked request. This host ("skc") is not allowed.`。
- **根因**：Vite 3+ 默认只允许 `localhost` 和 `127.0.0.1` 访问 dev server，防止 DNS 重绑定攻击（恶意网站通过 DNS 将域名指向 `127.0.0.1` 来攻击本地 dev server）。
- **修复**：

  ```typescript
  // vite.config.ts
  server: {
    host: '0.0.0.0',
    allowedHosts: ['skc', '.local'],
  }
  ```

  同时后端需要允许对应的 CORS origin：

  ```bash
  # backend-docker/.env
  CORS_ORIGINS=http://localhost:5173,http://SKC:5173
  ```

面试可讲：**Dev server 的安全策略是分层的——`host: '0.0.0.0'` 控制监听地址（网络层），`allowedHosts` 控制 HTTP Host 头校验（应用层），CORS 控制跨域请求（浏览器层）。** 三层都放开才能从局域网访问。


### 3. npm peer dependency 版本冲突——`--legacy-peer-deps` 的正确理解

- **现象**：前端 build 时报 `ERESOLVE: @vitejs/plugin-vue@5.2.4` 要求 `vite@^5.0.0 || ^6.0.0`，但项目用的是 `vite@8.0.16`。
- **根因**：`package.json` 的 `^5.1.4` 语义版本允许安装 5.x 的最新版（5.2.4），但该版本不支持 vite 8。vite 8 是较新的大版本，插件生态还没跟上。
- **修复**：`npm install --legacy-peer-deps`——跳过 peer dependency 的严格检查。这是开发阶段的实用妥协；生产构建时应该升级 `@vitejs/plugin-vue` 到支持 vite 8 的版本。

面试可讲：**peer dependency 是库作者声明"我和什么版本兼容"，npm 的职责是检查这个声明是否被满足。** 新版 npm（7+）默认严格检查 peer dependency 并拒绝安装——这是正确的行为，防止运行时出现隐蔽的 API 不兼容。`--legacy-peer-deps` 和 `--force` 的区别：前者是"忽略 peer 冲突但保留其他约束"，后者是"完全跳过依赖解析，直接用我给的版本"。


### 4. Volume 挂载 + node_modules = 静默覆盖

- **现象**：构建时 `npm install` 装了依赖，运行时 import 报 `Module not found`。
- **根因**：`docker-compose.yml` 里挂载 `../frontend:/app` 把整个宿主目录覆盖到容器 `/app`。构建时在镜像里装的 `/app/node_modules` 被宿主机目录（Windows 上的 node_modules）**完全覆盖**。且 Windows 的 `node_modules` 可能包含平台相关的原生模块，在 Linux 容器里无法运行。
- **修复**：用**匿名卷**保护构建产物：

  ```yaml
  volumes:
    - ../frontend:/app
    - /app/node_modules   # 匿名卷，优先于宿主目录
  ```

  匿名卷在 Docker 的挂载优先级中高于 bind mount。首次运行时 Docker 会将镜像中的 node_modules 复制到匿名卷；改了 `package.json` 需要 `docker compose down -v && docker compose up -d --build`。

面试可讲：**Docker 的挂载优先级：匿名卷 > bind mount > 镜像层。** 这个机制可以用"白名单"方式保护构建产物——大部分代码从宿主挂载（开发效率），少数构建产物保留在容器内（依赖隔离）。


### 5. 国内网络——Docker Hub + npm 双连断

- **Docker Hub**：`docker.io` 被墙，Docker Desktop Settings → Docker Engine → `registry-mirrors` 配镜像加速器。
- **npm**：`registry.npmjs.org` 慢/断，Dockerfile 里 `npm config set registry https://registry.npmmirror.com`。


## 启动

```bash
cd frontend-docker
docker compose up -d --build
```

访问 `http://localhost:5173`。

## 局域网访问

1. `vite.config.ts` 已配置 `allowedHosts: ['skc', '.local']`
2. `backend-docker/.env` 中 `CORS_ORIGINS` 需包含 `http://SKC:5173`
3. 另一台电脑浏览器访问 `http://SKC:5173`

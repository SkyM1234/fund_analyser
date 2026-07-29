# Alembic 数据库迁移

## 快速开始

### Docker 部署（推荐）

```bash
# 迁移到最新版本
docker exec fund-backend alembic -c /app/alembic.ini upgrade head

# 查看当前版本
docker exec fund-backend alembic -c /app/alembic.ini current

# 回滚一个版本
docker exec fund-backend alembic -c /app/alembic.ini downgrade -1
```

### 裸机运行

```bash
cd backend
export MYSQL_PASSWORD=your_password
alembic upgrade head
```

## 已有迁移版本

| 版本 | 说明 |
|------|------|
| `88116c7fba19` | 创建 users、refresh_tokens、sessions 表 |
| `a1b2c3d4e5f6` | 添加 user role 字段 |

## 注意事项

- 数据库连接从 `app.core.config.get_settings()` 读取，通过环境变量配置，不依赖 `alembic.ini` 中的占位 URL
- 迁移用同步驱动 `pymysql`，运行期用异步驱动 `asyncmy`，`env.py` 会自动转换
- SQLite / MySQL 不支持事务性 DDL，`downgrade` 失败时可能需要手动修复

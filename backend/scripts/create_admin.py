"""管理员账号初始化脚本。

用法：
    # Docker 内运行
    docker exec fund-backend python -m scripts.create_admin --username alice --password secret123
    # 裸机运行（在 backend/ 目录下）
    python -m scripts.create_admin --username alice --password secret123 --email a@x.com

若用户名已存在，则将该用户提升为 admin（不改密码）；否则创建新用户并设为 admin。
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.security import hash_password
from app.db.models import User
from app.db.mysql import close_engine, get_session_factory


async def create_or_promote_admin(username: str, password: str | None, email: str | None) -> None:
    factory = get_session_factory()
    async with factory() as db:
        user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()

        if user is not None:
            user.role = "admin"
            await db.commit()
            print(f"已将现有用户 '{username}' (id={user.id}) 提升为管理员。")
            return

        if not password:
            print(f"用户 '{username}' 不存在，创建新管理员账号需要 --password。", file=sys.stderr)
            sys.exit(1)

        user = User(
            username=username,
            email=email,
            password_hash=await hash_password(password),
            role="admin",
        )
        db.add(user)
        await db.commit()
        print(f"已创建新管理员账号 '{username}' (id={user.id})。")


def main() -> None:
    parser = argparse.ArgumentParser(description="创建或提升管理员账号")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", help="仅在用户不存在、需要新建账号时必填")
    parser.add_argument("--email", default=None)
    args = parser.parse_args()

    try:
        asyncio.run(create_or_promote_admin(args.username, args.password, args.email))
    finally:
        asyncio.run(close_engine())


if __name__ == "__main__":
    main()

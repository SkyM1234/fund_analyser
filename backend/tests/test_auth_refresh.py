import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from app.api.auth import refresh
from app.models.auth import RefreshRequest


class RefreshRotationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_refresh_consumes_token_only_once(self):
        redis = SimpleNamespace(getdel=AsyncMock(side_effect=["valid", None]))
        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock())
        db.execute.return_value.scalar_one_or_none.return_value = SimpleNamespace(is_active=True)
        with patch("app.api.auth.decode_token", return_value=1), patch(
            "app.api.auth.get_redis_client", return_value=redis,
        ), patch("app.api.auth._issue_tokens", new=AsyncMock(return_value="new-tokens")) as issue:
            results = await asyncio.gather(
                refresh(RefreshRequest(refresh_token="old-token"), db),
                refresh(RefreshRequest(refresh_token="old-token"), db),
                return_exceptions=True,
            )
        self.assertEqual(results[0], "new-tokens")
        self.assertIsInstance(results[1], HTTPException)
        self.assertEqual(results[1].status_code, 401)
        issue.assert_awaited_once()

    async def test_disabled_account_cannot_issue_new_tokens(self):
        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock())
        db.execute.return_value.scalar_one_or_none.return_value = SimpleNamespace(is_active=False)
        with patch("app.api.auth.decode_token", return_value=1), patch(
            "app.api.auth.get_redis_client", return_value=SimpleNamespace(getdel=AsyncMock(return_value="valid")),
        ), patch("app.api.auth._issue_tokens", new_callable=AsyncMock) as issue:
            with self.assertRaises(HTTPException):
                await refresh(RefreshRequest(refresh_token="token"), db)
            issue.assert_not_awaited()

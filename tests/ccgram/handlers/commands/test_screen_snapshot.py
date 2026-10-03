"""Tests for screen_snapshot — terminal image reply for TUI-only commands."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.commands.screen_snapshot import _maybe_send_screen_snapshot

_SN = "ccgram.handlers.commands.screen_snapshot"


def _message() -> MagicMock:
    message = MagicMock()
    message.chat.id = -100999
    message.message_thread_id = 42
    message.message_id = 7
    return message


@pytest.fixture
def snapshot_env() -> Iterator[SimpleNamespace]:
    with (
        patch(f"{_SN}.tmux_manager") as tmux_manager,
        patch(f"{_SN}.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch(f"{_SN}.safe_reply", new_callable=AsyncMock) as reply,
        patch(
            "ccgram.screenshot.text_to_image",
            new_callable=AsyncMock,
            return_value=b"PNGDATA",
        ) as render,
    ):
        client = MagicMock()
        client.send_document = AsyncMock()
        yield SimpleNamespace(
            tmux_manager=tmux_manager,
            sleep=sleep,
            client=client,
            reply=reply,
            render=render,
        )


class TestMaybeSendScreenSnapshot:
    async def test_sends_image_of_settled_screen(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        """The reply shows the pane once it stops changing, not the first draft."""
        snapshot_env.tmux_manager.capture_pane = AsyncMock(
            side_effect=["drawing...", "Context Usage panel", "Context Usage panel"]
        )

        await _maybe_send_screen_snapshot(
            snapshot_env.client, _message(), "@1", "stock-chooser", "/context"
        )

        assert snapshot_env.render.await_count == 1
        assert snapshot_env.render.await_args[0][0] == "Context Usage panel"
        sent = snapshot_env.client.send_document.await_args
        assert sent.kwargs["chat_id"] == -100999
        assert sent.kwargs["message_thread_id"] == 42
        assert "Context Usage" not in sent.kwargs["caption"]
        assert "/context" in sent.kwargs["caption"]
        assert sent.kwargs["document"].getvalue() == b"PNGDATA"
        snapshot_env.reply.assert_not_called()

    async def test_reports_failure_when_capture_is_empty(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        snapshot_env.tmux_manager.capture_pane = AsyncMock(return_value="")

        await _maybe_send_screen_snapshot(
            snapshot_env.client, _message(), "@1", "display", "/context"
        )

        snapshot_env.client.send_document.assert_not_called()
        assert "could not capture" in snapshot_env.reply.await_args[0][1]

    async def test_reports_failure_when_send_fails(
        self, snapshot_env: SimpleNamespace
    ) -> None:
        snapshot_env.tmux_manager.capture_pane = AsyncMock(return_value="panel")
        snapshot_env.client.send_document = AsyncMock(side_effect=RuntimeError("boom"))

        await _maybe_send_screen_snapshot(
            snapshot_env.client, _message(), "@1", "display", "/usage"
        )

        assert "could not capture" in snapshot_env.reply.await_args[0][1]

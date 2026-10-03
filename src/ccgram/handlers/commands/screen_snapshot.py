"""Terminal-screen fallback for commands the provider only draws in the TUI.

Some provider commands — Oh My Pi's ``/context``, ``/usage``, ``/tools`` — paint
a full-screen view and write nothing to the transcript. The reply pipeline has
no output to relay, so the topic stayed silent after the send acknowledgement.
Providers declare those commands in ``ProviderCapabilities.tui_screen_commands``;
for a forwarded one, ccgram waits for the screen to settle and answers with a
captured terminal image.

Public surface is the one helper used by ``forward.py``:
  - _spawn_screen_snapshot(): post-send pane capture + image reply
"""

from __future__ import annotations

import asyncio
import io

import structlog
from telegram import Message

from ...multiplexer import multiplexer as tmux_manager
from ...telegram_client import TelegramClient
from ...utils import task_done_callback
from ..messaging_pipeline.message_sender import safe_reply

logger = structlog.get_logger()

# Poll until two pane captures match, so a screen that is still drawing (for
# example /usage, which fetches provider limits) is captured settled.
_SCREEN_SETTLE_POLL_SECONDS = 0.5
_SCREEN_SETTLE_MAX_POLLS = 8


async def _capture_settled_screen(window_id: str) -> str | None:
    """Capture the pane once its text stops changing, within the poll budget."""
    previous = await tmux_manager.capture_pane(window_id, with_ansi=True)
    for _ in range(_SCREEN_SETTLE_MAX_POLLS):
        await asyncio.sleep(_SCREEN_SETTLE_POLL_SECONDS)
        current = await tmux_manager.capture_pane(window_id, with_ansi=True)
        if current and current == previous:
            return current
        previous = current
    return previous


async def _maybe_send_screen_snapshot(
    client: TelegramClient,
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
) -> None:
    """Send the settled terminal screen as an image for a TUI-only command."""
    pane_text = await _capture_settled_screen(window_id)
    if not pane_text:
        logger.warning("Screen snapshot capture returned nothing", window_id=window_id)
        await safe_reply(
            message, f"❌ [{display}] `{cc_slash}` — could not capture the screen."
        )
        return

    # Lazy: screenshot pulls in the image stack (PIL); only needed here.
    from ...screenshot import text_to_image

    png_bytes = await text_to_image(pane_text, with_ansi=True)
    try:
        await client.send_document(
            chat_id=message.chat.id,
            document=io.BytesIO(png_bytes),
            filename="screen.png",
            caption=f"🖥️ [{display}] {cc_slash} — terminal screen",
            message_thread_id=message.message_thread_id,
            reply_to_message_id=message.message_id,
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure must never escape
        logger.error("Failed to send screen snapshot: %s", exc)
        await safe_reply(
            message, f"❌ [{display}] `{cc_slash}` — could not capture the screen."
        )


def _spawn_screen_snapshot(
    client: TelegramClient,
    message: Message,
    window_id: str,
    display: str,
    cc_slash: str,
) -> None:
    """Run the screen-snapshot reply in the background."""

    async def _run() -> None:
        await _maybe_send_screen_snapshot(client, message, window_id, display, cc_slash)

    task = asyncio.create_task(_run())
    task.add_done_callback(task_done_callback)

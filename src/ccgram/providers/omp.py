"""Oh My Pi (``omp``) provider — https://github.com/can1357/oh-my-pi.

Oh My Pi is a fork of pi: the transcript format is the same pi-family JSONL v3
envelope, parsed by ``pi_format`` (roles, ``toolCall``/``toolResult`` blocks,
``stopReason`` errors). What differs is the storage layout and the CLI surface.

Sessions live at ``~/.omp/agent/sessions/<encoded-cwd>/<timestamp>_<uuid>.jsonl``
— the same one-JSONL-per-session shape as pi, but the cwd bucket encoding is
omp's own (home- and tmp-relative prefixes, see ``encode_cwd_dirname``). Every
transcript starts with a ``type: title`` entry *before* the ``type: session``
header, so the header is not line 1 (``pi_format.read_session_header`` scans
past it).

ccgram treats omp as hookless: there is no omp equivalent of the pi
hook-runner extension and ccgram installs no provider hooks, so session
tracking is transcript discovery (``discover_transcript``) and the transcript
file is the message source of truth. ``make_launch_args`` is inherited from
``PiProvider``: resume always uses ``--session <path>`` because ``--resume``
with no value opens an interactive picker ccgram cannot drive over
``send_keys``.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import replace
from pathlib import Path

from ccgram.providers.base import (
    DiscoveredCommand,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.providers.omp_discovery import (
    _OMP_TELEGRAM_BUILTINS,
    discover_omp_commands,
)
from ccgram.providers.pi import PiProvider
from ccgram.providers.session_scan import newest_matching_transcript


def _omp_sessions_dir() -> Path:
    return Path.home() / ".omp" / "agent" / "sessions"


# Cap transcript age when the pane is dead — guards against picking up an
# unrelated historical transcript for the same cwd.
_OMP_STALE_TRANSCRIPT_MAX_AGE_SECS = 120.0

# How many recent session files to inspect when searching for a cwd match.
_OMP_DISCOVERY_SCAN_LIMIT = 20


def _resolve(path: str | Path) -> Path:
    """Canonicalize *path*, falling back to the literal path on OSError."""
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _relative_bucket(prefix: str, target: Path, root: Path) -> str | None:
    """Render ``prefix + <target relative to root>``, or None if not under root."""
    try:
        relative = target.relative_to(_resolve(root))
    except ValueError:
        return None
    parts = [part for part in relative.parts if part not in ("", ".")]
    return prefix + "-".join(parts)


def encode_cwd_dirname(cwd: str) -> str:
    """Encode a working directory into omp's session subdirectory name.

    omp canonicalizes the cwd first (so symlink aliases share a bucket), then:

    - under the home directory → ``-`` + home-relative path, separators ``-``
    - under the temp root → ``-tmp-`` + temp-relative path, separators ``-``
    - otherwise → ``--`` + absolute path minus the leading ``/`` + ``--``

    Edges: the home directory itself renders as ``-`` and the temp root as
    ``-tmp-`` (empty relative path), and the filesystem root renders as
    ``----`` (the absolute branch with nothing left to encode). None of them
    raise.
    """
    resolved = _resolve(cwd)

    for prefix, root in (("-", Path.home()), ("-tmp-", tempfile.gettempdir())):
        bucket = _relative_bucket(prefix, resolved, Path(root))
        if bucket is not None:
            return bucket

    return "--" + str(resolved).strip("/").replace("/", "-") + "--"


# ── Interactive ``ask`` prompt ───────────────────────────────────────────
#
# The ``ask`` tool draws its question as a box titled "Ask" and writes nothing
# to the transcript while it waits, so that box is the only place the question
# and its options appear.  Verified against a live omp 18.2.1 TUI:
#
#   ╭─ Ask ───────────────────────────────────────╮
#   │ Which color do you prefer?                   │
#   ├──────────────────────────────────────────────┤
#   │   󱊔 󰄌 Red                                     │
#   │       Warm and bold.                         │
#   │     󰄌 Green                                   │
#   ├──────────────────────────────────────────────┤
#   │ Enter select · n note · ↑/↓ move · Esc cancel │
#   ╰──────────────────────────────────────────────╯
#
# The footer differs per flavour — "Enter select" for single and tabbed asks,
# "Space toggle · Enter next" for multi-select, with "Tab/←/→" added once the
# ask has tabs — but the arrow/Esc hint is on all of them.  An answered ask
# collapses to a box without a footer, so the hint is what proves the prompt
# is still waiting.
_ASK_BOX_TITLE_RE = re.compile(r"^\s*╭─+\s*Ask\b")
_ASK_BOX_CLOSE_RE = re.compile(r"^\s*╰")
_ASK_HINT_FRAGMENTS = ("↑/↓ move", "Esc cancel")

# Nerd Font icons omp draws at the start of an option row, transliterated to
# standard Unicode so Telegram renders the rows instead of private-use boxes.
# Each replacement is one column wide, so the captured alignment survives.
_ASK_ROW_ICONS = str.maketrans(
    {
        "\uf054": "❯",  # cursor on the highlighted row
        "\uf10c": "○",  # radio button, not chosen
        "\uf096": "☐",  # checkbox, not chosen
        "\uf14a": "☑",  # checkbox, chosen
    }
)

# A wide pane pads every box line to its full width. Dropping the trailing
# right border and collapsing the long rules keeps the panel readable in a
# Telegram message without losing a row of content.
_ASK_TRAILING_BORDER_RE = re.compile(r"\s*│\s*$")
_ASK_RULE_RE = re.compile(r"─{5,}")


def _fit_ask_panel(box: str) -> str:
    """Trim a captured box to its content width for display."""
    lines = [
        _ASK_RULE_RE.sub("─────", _ASK_TRAILING_BORDER_RE.sub("", line)).rstrip()
        for line in box.split("\n")
    ]
    return "\n".join(lines)


def extract_ask_panel(pane_text: str) -> str | None:
    """Return the ``ask`` prompt box a pane is waiting on, else None.

    Reads the last complete box of the capture. The box counts only when its
    title is "Ask" and its footer carries the arrow/Esc hint, so a half-drawn
    frame or a collapsed, already answered box never becomes a live prompt.
    """
    lines = pane_text.split("\n")
    close_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if _ASK_BOX_CLOSE_RE.match(lines[i])),
        None,
    )
    if close_idx is None:
        return None
    open_idx = next(
        (i for i in range(close_idx - 1, -1, -1) if _ASK_BOX_TITLE_RE.match(lines[i])),
        None,
    )
    if open_idx is None:
        return None

    box = "\n".join(lines[open_idx : close_idx + 1])
    if not all(fragment in box for fragment in _ASK_HINT_FRAGMENTS):
        return None
    return _fit_ask_panel(box.translate(_ASK_ROW_ICONS))


class OmpProvider(PiProvider):
    """AgentProvider implementation for the Oh My Pi CLI."""

    _CAPS = replace(
        PiProvider._CAPS,
        name="omp",
        launch_command="omp",
        # pi's hook support comes from the third-party hook-runner extension;
        # omp has no such contract and ccgram installs no provider hooks, so
        # every hook-driven path must stay off for it.
        supports_hook=False,
        builtin_commands=tuple(_OMP_TELEGRAM_BUILTINS.keys()),
        # omp's app.message.followUp default is ctrl+q (also ctrl+enter); pi's
        # follow-up key is Alt+Enter, so the key differs between the two.
        followup_key="C-q",
        # Unverified against a live omp TUI (pi's set was verified by driving
        # the real CLI): kept to commands whose registry entry clearly opens an
        # in-TUI menu the user drives with arrows/Enter/Esc.
        tui_picker_commands=frozenset(
            {
                "agents",
                "copy",
                "extensions",
                "login",
                "logout",
                "mcp",
                "model",
                "session",
                "settings",
                "share",
                "switch",
                "todo",
            }
        ),
        # Commands omp draws as a full-screen view and never writes to the
        # transcript, verified by driving a live omp 18.2.1 TUI. Forwarding one
        # otherwise produced no reply at all, so ccgram answers with a captured
        # terminal image. Text dumps (/changelog), one-line toasts (/dirs,
        # /dump, /export) and the modal pickers above keep their existing path.
        tui_screen_commands=frozenset(
            {"context", "hotkeys", "jobs", "stats", "tools", "usage"}
        ),
    )

    _BUILTINS = _OMP_TELEGRAM_BUILTINS

    # ── Discovery ────────────────────────────────────────────────────────

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Return the newest omp transcript whose header cwd matches."""
        age_limit = (
            _OMP_STALE_TRANSCRIPT_MAX_AGE_SECS if max_age is None else float(max_age)
        )
        return newest_matching_transcript(
            _omp_sessions_dir() / encode_cwd_dirname(cwd),
            cwd,
            window_key=window_key,
            max_age=age_limit,
            scan_limit=_OMP_DISCOVERY_SCAN_LIMIT,
        )

    # ── Commands ────────────────────────────────────────────────────────

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        return discover_omp_commands(base_dir)

    # ── Terminal status ─────────────────────────────────────────────────

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002 — omp sets no title-derived status
    ) -> StatusUpdate | None:
        """Report a waiting ``ask`` prompt as an interactive UI.

        omp draws no spinner or status line ccgram can read, so the only pane
        state worth surfacing is the question box the user has to answer.
        """
        panel = extract_ask_panel(pane_text)
        if panel is None:
            return None
        return StatusUpdate(
            raw_text=panel,
            display_label="Ask",
            is_interactive=True,
            ui_type="Ask",
        )

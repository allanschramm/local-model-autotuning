"""Trial server log helpers for dashboard Idle / Em execução (#26)."""

from __future__ import annotations

import time
from pathlib import Path

# Default monitored tuning log directory.
_LOG_REL = Path("tuning-artifacts")
_GROWTH_WINDOW_SEC = 10.0
_TAIL_LINES = 80


def server_log_path() -> Path:
    """Newest bounded tuning server log, relative to the repository."""
    directory = Path(__file__).resolve().parents[1] / _LOG_REL
    candidates = list(directory.glob("*-server.log"))
    return (
        max(candidates, key=lambda p: p.stat().st_mtime) if candidates else directory / "server.log"
    )


def run_state_and_tail() -> tuple[str, str | None]:
    """Return (Idle|Em execução, tail text or None if log missing).

    Em execução when the log's mtime is within the last ~10s (recent growth).
    Missing file → Idle + None (UI shows pt-BR empty, no crash).
    """
    try:
        path = server_log_path()
        if not path.is_file():
            return "Idle", None
        st = path.stat()
        age = time.time() - st.st_mtime
        state = "Em execução" if age <= _GROWTH_WINDOW_SEC else "Idle"
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        tail = "\n".join(lines[-_TAIL_LINES:])
        return state, tail
    except OSError:
        return "Idle", None

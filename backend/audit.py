"""
Audit Module for Orion Multi-Agent RAG System
Persistent, append-only structured logging in JSON Lines format.
Thread-safe via a module-level lock.
"""

import json
import os
import random
import string
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────

AUDIT_LOG_PATH = Path(__file__).parent.parent / "outputs" / "audit_log.jsonl"
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_write_lock = threading.Lock()

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_request_id() -> str:
    """Generate a unique request ID e.g. ORION-20260324-025229-A3F7"""
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"ORION-{now}-{rand}"


def log_event(
    event_type: str,
    user: str,
    request_id: str,
    payload: Dict[str, Any],
    role: Optional[str] = None,
) -> None:
    """
    Append a structured audit event to the JSONL log.

    Args:
        event_type:  One of: preprocess_block, preprocess_pass, tasks_identified,
                     task_start, tool_call, task_complete, postprocess_redaction,
                     final, error
        user:        Username or "system"
        request_id:  Unique request ID (from generate_request_id())
        payload:     Arbitrary dict of event-specific data
        role:        User role if known ("admin" / "limited" / None)
    """
    event: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "user": user or "anonymous",
        "role": role or "unknown",
        "event_type": event_type,
        "payload": payload,
    }

    line = json.dumps(event, ensure_ascii=False, default=str)

    try:
        with _write_lock:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        # Never let audit failures crash the application
        print(f"[AUDIT] ERROR writing log: {exc}")


def get_recent_events(n: int = 50) -> list:
    """Read the last N events from the audit log (for UI/debug use)."""
    events = []
    try:
        if not AUDIT_LOG_PATH.exists():
            return events
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        print(f"[AUDIT] ERROR reading log: {exc}")
    return events


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rid = generate_request_id()
    print(f"Generated request_id: {rid}")

    log_event("test_event", "test_user", rid, {"message": "audit module self-test"}, role="admin")

    events = get_recent_events(5)
    assert any(e["event_type"] == "test_event" for e in events), "Test event not found!"
    print(f"audit_log.jsonl exists at: {AUDIT_LOG_PATH}")
    print("audit.py: ALL CHECKS PASSED ✅")

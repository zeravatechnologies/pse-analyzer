"""The Guardian shield in the status bar -- [SPEC] 10.4, AZ-12.

"Deliberately near-invisible: a shield in the status bar." This module ONLY
READS the Guardian audit log (the hash-chained jsonl from
SystemForEric/guardian); it does not import its code and it writes nothing. The
Guardian protects the machine; it never touches the analytical artefacts -- and
the analyzer never touches it.

States:
  gray   -- the Guardian is not running on this machine (there is no audit log)
  green  -- running, nothing was escalated in the last 24 h
  yellow -- there are detected deviations
  red    -- an escalation, or a failed owner authentication
Green is never faked when there is no data.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

RED = {"escalation", "owner_auth_failed", "capability_revoked"}
YELLOW = {"pathology_detected"}


def shield_state(window_s: float = 86400) -> dict:
    home = Path(os.environ.get("GUARDIAN_HOME", str(Path.home() / ".guardian")))
    logs = sorted((home / "audit").glob("*.audit.jsonl")) if (home / "audit").is_dir() else []
    if not logs:
        return {"level": "gray", "text": "The Guardian is not running on this machine"}
    now = time.time()
    level, n = "green", 0
    for lp in logs:
        try:
            with open(lp, "rb") as f:
                start = max(0, os.path.getsize(lp) - 256_000)
                f.seek(start)
                tail = f.read().decode("utf-8", "ignore").splitlines()
                if start > 0:
                    tail = tail[1:]            # the first line is probably cut in half
        except OSError:
            continue
        for line in tail:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if now - float(rec.get("ts", 0)) > window_s:
                continue
            et = str(rec.get("event_type", ""))
            if any(et.startswith(r) for r in RED):
                level, n = "red", n + 1
            elif et in YELLOW and level != "red":
                level, n = "yellow", n + 1
    text = {"green": "The Guardian is running, with no events in the last 24 h",
            "yellow": f"The Guardian noticed deviations in the last 24 h: {n}",
            "red": f"The Guardian needs attention — events in the last 24 h: {n}"}[level]
    return {"level": level, "text": text}

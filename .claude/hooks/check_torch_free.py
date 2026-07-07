#!/usr/bin/env python3
"""PostToolUse guard: keep the audio/IO layer torch-free.

hear_embed/audio.py, pipeline.py, and writers.py must stay importable without
torch so ``uv sync --no-group model`` and CI's torch-free tests keep working.
Heavy deps (torch, transformers) belong in *lazy* imports inside functions,
never at module top level. This hook re-reads an edited file from that layer and
warns — feeding the message back to Claude and to the user — if a top-level
``import torch`` / ``from transformers import ...`` slipped in.

Wired up in .claude/settings.json on the ``Edit|Write`` matcher. Stdlib-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The torch-free layer. embedder.py is intentionally excluded — it *is* the model
# wrapper and imports torch lazily inside its methods.
GUARDED = {"audio.py", "pipeline.py", "writers.py"}

# A *top-level* import (column 0 -> not indented inside a function) of a heavy dep.
# Lazy imports inside functions are indented and correctly do NOT match.
PATTERN = re.compile(r"^(?:import|from)\s+(torch|transformers)\b", re.MULTILINE)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_input = data.get("tool_input") or {}
    fp = tool_input.get("file_path")
    if not fp:
        sys.exit(0)

    target = Path(fp)
    # Only guard the three torch-free modules directly under hear_embed/.
    if target.name not in GUARDED or target.parent.name != "hear_embed":
        sys.exit(0)

    try:
        source = target.read_text(encoding="utf-8")
    except OSError:
        sys.exit(0)

    hits = sorted({m.group(1) for m in PATTERN.finditer(source)})
    if not hits:
        sys.exit(0)

    names = " / ".join(hits)
    warning = (
        f"Torch-free boundary violated: {target.name} now has a top-level import of "
        f"{names}. The audio/IO layer must stay importable without torch so "
        "`uv sync --no-group model` and CI's torch-free tests keep working. Move the "
        f"{names} import inside the function that needs it (a lazy import), matching "
        "the pattern in audio.py/embedder.py. See CLAUDE.md 'the torch-free boundary'."
    )
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": warning,
                "systemMessage": (
                    f"⚠ hear-embed: top-level {names} import added to "
                    f"{target.name} (torch-free boundary)."
                ),
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

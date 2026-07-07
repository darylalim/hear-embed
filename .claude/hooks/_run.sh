#!/usr/bin/env bash
# Resolve a Python interpreter for hear-embed's hooks and exec the given script,
# passing stdin through untouched. Interpreter preference:
#   1. the project venv  (.venv/bin/python) — fastest, always the pinned 3.11
#   2. a system python3/python — works on a fresh clone before `uv sync`
#   3. `uv run --no-sync python` — last resort; --no-sync avoids reinstalling deps
# If none exists we exit 0 (allow the tool call) rather than bricking the session.
set -u
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
script="$1"

if [ -x "$root/.venv/bin/python" ]; then
  exec "$root/.venv/bin/python" "$script"
fi
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    exec "$py" "$script"
  fi
done
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-sync --project "$root" python "$script"
fi
exit 0

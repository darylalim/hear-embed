"""Tests for the Claude Code invariant-enforcement hooks in ``.claude/hooks/``.

These hooks are project tooling rather than part of the ``hear_embed`` package,
but they mechanically guard the repo's hard invariants — vendored code is
immutable, ``.env`` secrets and ``uv.lock`` are off-limits, and the audio/IO
layer stays torch-free — so their behavior is locked in here. The scripts are
stdlib-only and torch-free, so these tests run in CI's ``-m "not model"`` job.

Each hook is exercised exactly as Claude Code runs it: a JSON payload on stdin,
with the decision read back from stdout (empty stdout == allow). We shell out
via ``subprocess`` rather than importing, because the scripts read stdin and
call ``sys.exit``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
GUARD = HOOKS_DIR / "guard_paths.py"
CHECK = HOOKS_DIR / "check_torch_free.py"
WRAPPER = HOOKS_DIR / "_run.sh"


def _run_hook(script: Path, tool_name: str, tool_input: dict) -> dict | None:
    """Run a hook with a synthesized payload; return its JSON decision or None.

    ``None`` means the hook allowed the call (exited 0 with empty stdout).
    """
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    result = subprocess.run(
        [sys.executable, str(script)],
        input=payload,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    return json.loads(out) if out else None


def _denied(decision: dict | None) -> bool:
    """True if a PreToolUse decision denied the call."""
    return (
        decision is not None
        and decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    )


# --- guard_paths.py (PreToolUse) ---------------------------------------------

VENDORED = str(REPO_ROOT / "hear_embed" / "_vendor" / "audio_utils.py")
VENDORED_NB = str(REPO_ROOT / "hear_embed" / "_vendor" / "notebook.ipynb")
ENV = str(REPO_ROOT / ".env")
ENV_LOCAL = str(REPO_ROOT / ".env.local")
ENV_PROD = str(REPO_ROOT / ".env.production")
ENV_NESTED = str(REPO_ROOT / "config" / ".env")
ENV_EXAMPLE = str(REPO_ROOT / ".env.example")
LOCK = str(REPO_ROOT / "uv.lock")
NORMAL = str(REPO_ROOT / "hear_embed" / "audio.py")


@pytest.mark.parametrize(
    ("tool", "tool_input", "denied"),
    [
        # Vendored Google code: writes blocked, reads fine.
        ("Edit", {"file_path": VENDORED}, True),
        ("Write", {"file_path": VENDORED}, True),
        ("NotebookEdit", {"notebook_path": VENDORED_NB}, True),
        ("Read", {"file_path": VENDORED}, False),
        # MultiEdit is a write tool too (some Claude Code builds expose it) and
        # must be guarded on every deny rule, not silently allowed.
        ("MultiEdit", {"file_path": VENDORED}, True),
        ("MultiEdit", {"file_path": ENV}, True),
        ("MultiEdit", {"file_path": LOCK}, True),
        # .env holds secrets: read + write blocked; .env.example is fine.
        ("Read", {"file_path": ENV}, True),
        ("Edit", {"file_path": ENV}, True),
        ("Write", {"file_path": ENV}, True),
        ("Read", {"file_path": ENV_EXAMPLE}, False),
        ("Edit", {"file_path": ENV_EXAMPLE}, False),
        # .env.* variants (matched by basename) and a nested .env are secrets too.
        ("Read", {"file_path": ENV_LOCAL}, True),
        ("Edit", {"file_path": ENV_LOCAL}, True),
        ("Write", {"file_path": ENV_PROD}, True),
        ("Read", {"file_path": ENV_NESTED}, True),
        # uv.lock is machine-generated: writes blocked, reads fine.
        ("Edit", {"file_path": LOCK}, True),
        ("Write", {"file_path": LOCK}, True),
        ("Read", {"file_path": LOCK}, False),
        # Ordinary source is untouched.
        ("Edit", {"file_path": NORMAL}, False),
        ("Write", {"file_path": NORMAL}, False),
    ],
)
def test_guard_paths(tool, tool_input, denied):
    decision = _run_hook(GUARD, tool, tool_input)
    assert _denied(decision) is denied
    if denied:
        # A deny always carries a human-readable reason.
        assert decision is not None
        assert decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_guard_paths_allows_when_no_path():
    # A payload with no file_path/notebook_path is allowed, not an error.
    assert _run_hook(GUARD, "Edit", {}) is None


def test_env_example_documents_the_token_workflow():
    # The guard blocks reading .env (test above), so CLAUDE.md points users to
    # .env.example for the gated-model token setup. Keep that template honest: it
    # must define HF_TOKEN and show how to load it into a uv command, or the
    # CLAUDE.md pointer rots silently.
    text = Path(ENV_EXAMPLE).read_text()
    assert "HF_TOKEN" in text, ".env.example no longer defines HF_TOKEN"
    assert "--env-file" in text or "UV_ENV_FILE" in text, (
        ".env.example no longer shows a uv token-loading workflow (`--env-file` or "
        "`UV_ENV_FILE`) that CLAUDE.md references for the gated model"
    )


# --- check_torch_free.py (PostToolUse) ---------------------------------------


def _write_module(tmp_path: Path, name: str, source: str) -> str:
    """Write ``source`` to ``tmp_path/hear_embed/<name>`` and return its path."""
    pkg = tmp_path / "hear_embed"
    pkg.mkdir(exist_ok=True)
    module = pkg / name
    module.write_text(source)
    return str(module)


def test_torch_free_top_level_torch_warns(tmp_path):
    path = _write_module(tmp_path, "audio.py", "import torch\nimport numpy as np\n")
    decision = _run_hook(CHECK, "Edit", {"file_path": path})
    assert decision is not None
    assert decision["decision"] == "block"
    assert "torch" in decision["reason"]


def test_torch_free_top_level_transformers_warns(tmp_path):
    src = "from transformers import AutoModel\n"
    path = _write_module(tmp_path, "writers.py", src)
    decision = _run_hook(CHECK, "Edit", {"file_path": path})
    assert decision is not None
    assert "transformers" in decision["reason"]


def test_torch_free_lazy_import_allowed(tmp_path):
    # An indented (in-function) import is the sanctioned lazy pattern.
    src = "def load():\n    import torch\n    return torch\n"
    path = _write_module(tmp_path, "pipeline.py", src)
    assert _run_hook(CHECK, "Edit", {"file_path": path}) is None


def test_torch_free_comment_mention_allowed(tmp_path):
    # "import torch" inside a comment must not trip the column-0 regex.
    src = "# a comment: import torch\nimport numpy\n"
    path = _write_module(tmp_path, "audio.py", src)
    assert _run_hook(CHECK, "Edit", {"file_path": path}) is None


def test_torch_free_unguarded_module_ignored(tmp_path):
    # embedder.py is the model wrapper — intentionally allowed to import torch.
    path = _write_module(tmp_path, "embedder.py", "import torch\n")
    assert _run_hook(CHECK, "Edit", {"file_path": path}) is None


def test_torch_free_outside_package_ignored(tmp_path):
    # A guarded *filename* outside a hear_embed/ dir is not the real module.
    stray = tmp_path / "audio.py"
    stray.write_text("import torch\n")
    assert _run_hook(CHECK, "Edit", {"file_path": str(stray)}) is None


def test_torch_free_multiedit_top_level_torch_warns(tmp_path):
    # MultiEdit reaches the guard via the PostToolUse matcher just like Edit/Write.
    path = _write_module(tmp_path, "audio.py", "import torch\n")
    decision = _run_hook(CHECK, "MultiEdit", {"file_path": path})
    assert decision is not None
    assert decision["decision"] == "block"


@pytest.mark.parametrize("name", ["audio.py", "pipeline.py", "writers.py"])
def test_torch_free_real_modules_pass(name):
    # The actual shipped torch-free modules must not trip the guard.
    path = str(REPO_ROOT / "hear_embed" / name)
    assert _run_hook(CHECK, "Edit", {"file_path": path}) is None


# --- _run.sh interpreter wrapper (the shipped launch path) -------------------


def _run_via_wrapper(script: Path, tool_name: str, tool_input: dict) -> dict | None:
    """Drive a hook through _run.sh exactly as settings.json does (bash -> python)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    result = subprocess.run(
        [bash, str(WRAPPER), str(script)],
        input=payload,
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(REPO_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    return json.loads(out) if out else None


def test_run_sh_wrapper_denies_and_allows():
    # The production path is settings.json -> bash _run.sh -> python hook. Exercise
    # it end to end so a wrapper regression (interpreter resolution, quoting,
    # stdin passthrough) can't ship green while every direct-invocation test passes.
    assert _denied(_run_via_wrapper(GUARD, "Edit", {"file_path": VENDORED}))
    assert _run_via_wrapper(GUARD, "Read", {"file_path": NORMAL}) is None

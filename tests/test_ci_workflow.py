"""Invariants for the GitHub Actions CI workflow (``.github/workflows/ci.yml``).

The workflow is project config, not shipped code, but it encodes invariants that
stay *green even when broken* — the class of regression a passing build hides:

- two jobs sharing a uv cache key silently re-download torch every run, so each
  cache-enabled ``setup-uv`` step needs its own ``cache-suffix``;
- a bare ``uv run`` re-syncs the default groups, quietly reinstalling a
  dependency group a job just chose to skip, so every ``uv run`` needs
  ``--no-sync``;
- a job without ``timeout-minutes`` inherits GitHub's 360-minute default, so a
  hung download can burn hours unnoticed.

Parsing needs PyYAML, a dev-group dependency; ``importorskip`` keeps the module
ty-clean (``yaml`` typed ``Any``) and skips gracefully if it is somehow absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml: Any = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_workflow = yaml.safe_load(CI.read_text())
_jobs = _workflow["jobs"]


def _steps(job: Any) -> list[Any]:
    return job.get("steps", [])


def _run_lines():
    """Yield ``(job_name, stripped_line)`` for every line of every ``run:`` step."""
    for name, job in _jobs.items():
        for step in _steps(job):
            run = step.get("run")
            if run:
                for line in run.splitlines():
                    yield name, line.strip()


def test_concurrency_cancels_superseded_runs() -> None:
    assert _workflow["concurrency"]["cancel-in-progress"] is True


def test_every_job_has_a_timeout() -> None:
    for name, job in _jobs.items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and timeout > 0, (
            f"job {name!r} has no timeout-minutes; it would inherit the 360-min default"
        )


def test_uv_cache_slots_are_distinct_per_job() -> None:
    suffixes = []
    for job in _jobs.values():
        for step in _steps(job):
            uses = step.get("uses", "")
            with_block = step.get("with") or {}
            if uses.startswith("astral-sh/setup-uv") and with_block.get("enable-cache"):
                suffixes.append(with_block.get("cache-suffix"))
    assert suffixes, "no cache-enabled setup-uv steps found"
    assert all(suffixes), "a cache-enabled setup-uv step is missing a cache-suffix"
    assert len(suffixes) == len(set(suffixes)), f"cache-suffixes collide: {suffixes}"


def test_uv_run_steps_use_no_sync() -> None:
    for name, line in _run_lines():
        if "uv run" in line:
            assert "--no-sync" in line, f"{name}: `uv run` without --no-sync: {line!r}"


def test_build_job_builds_and_smoke_tests_the_wheel() -> None:
    assert "build" in _jobs, "no build job"
    runs = [line for name, line in _run_lines() if name == "build"]
    assert any("uv build" in line for line in runs), "build job never runs `uv build`"
    assert any("hear-embed --help" in line for line in runs), (
        "build job never smoke-tests the console script"
    )

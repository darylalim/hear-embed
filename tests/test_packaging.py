"""Packaging & pyproject source-tree invariants: license shipping + the torch pin.

``pyproject.toml`` declares ``license = "Apache-2.0"`` and ``license-files``
(PEP 639), and a root ``LICENSE`` carries the full Apache-2.0 text. These are
cheap source-tree checks (torch-free, run in CI's ``-m "not model"`` job). The
distribution-level guarantee — that the *built wheel's* METADATA actually
carries ``License-File`` entries for both texts — is exercised separately by
CI's ``build`` job, which unzips the wheel and asserts them (see
``.github/workflows/ci.yml``).

The last check guards a different pyproject invariant: that Linux ``torch`` stays
pinned to the CPU index (a stays-green-when-broken CI download regression). It
lives here, not in ``test_ci_workflow.py``, so a missing PyYAML can't silently
skip it behind that module's ``importorskip("yaml")``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSE = REPO_ROOT / "LICENSE"
VENDORED_LICENSE = REPO_ROOT / "hear_embed" / "_vendor" / "LICENSE.apache-2.0"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Landmarks only a complete Apache-2.0 text contains: the header plus a clause
# deep in the body, so a truncated or placeholder LICENSE is caught too.
_APACHE_MARKERS = (
    "Apache License",
    "Version 2.0, January 2004",
    "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
    "Limitation of Liability",
)


def test_root_license_is_complete_apache_2() -> None:
    assert LICENSE.is_file(), "root LICENSE is missing"
    text = LICENSE.read_text()
    for marker in _APACHE_MARKERS:
        assert marker in text, f"LICENSE missing Apache-2.0 marker: {marker!r}"


def test_vendored_license_still_present() -> None:
    # The distribution ships two license texts: this project's LICENSE and
    # Google's, for the unmodified vendored audio_utils.py. Guard both survive.
    assert VENDORED_LICENSE.is_file()


def test_pyproject_declares_license_and_ships_both_texts() -> None:
    text = PYPROJECT.read_text()
    assert 'license = "Apache-2.0"' in text, "pyproject.toml missing SPDX license"
    files_line = next(
        (ln for ln in text.splitlines() if ln.lstrip().startswith("license-files")),
        None,
    )
    assert files_line is not None, "pyproject.toml does not declare license-files"
    # Both texts must be listed so the built distribution ships them. Match each
    # as a *quoted* list element: a bare `"LICENSE" in files_line` is also
    # satisfied by the vendored path (…/LICENSE.apache-2.0), so it would not catch
    # the root LICENSE entry being dropped.
    assert '"LICENSE"' in files_line, "root LICENSE not listed in license-files"
    assert '"hear_embed/_vendor/LICENSE.apache-2.0"' in files_line


def test_pyproject_pins_linux_torch_to_cpu_index() -> None:
    # On Linux, PyPI's default ``torch`` is the CUDA build plus several GB of
    # ``nvidia-*`` wheels; CI's typecheck/build jobs only need an importable CPU
    # torch. pyproject resolves Linux torch from the pytorch-cpu index instead —
    # dropping this pin is a stays-green-when-broken regression (CI keeps passing
    # while re-downloading gigabytes every run). Parse the TOML structure, not the
    # raw text, so a whitespace/quote reflow of the inline table can't fake a
    # match and an inverted ``!= 'linux'`` marker can't slip past a substring test.
    tomllib = pytest.importorskip("tomllib")  # stdlib on 3.11+, the pinned version
    uv = tomllib.loads(PYPROJECT.read_text())["tool"]["uv"]

    indexes = {idx["name"]: idx["url"] for idx in uv.get("index", [])}
    assert indexes.get("pytorch-cpu") == "https://download.pytorch.org/whl/cpu", (
        "the pytorch-cpu index is gone from [[tool.uv.index]]"
    )

    sources = uv.get("sources", {}).get("torch", [])
    if isinstance(sources, dict):  # uv accepts a single mapping or a list of them
        sources = [sources]
    assert any(
        s.get("index") == "pytorch-cpu"
        and "== 'linux'" in s.get("marker", "").replace('"', "'")
        for s in sources
    ), "torch is not pinned to the pytorch-cpu index under a `== 'linux'` marker"

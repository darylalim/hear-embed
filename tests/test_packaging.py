"""Packaging invariants: the project ships its license.

``pyproject.toml`` declares ``license = "Apache-2.0"`` and ``license-files``
(PEP 639), and a root ``LICENSE`` carries the full Apache-2.0 text. These are
cheap source-tree checks (torch-free, run in CI's ``-m "not model"`` job). The
distribution-level guarantee — that the *built wheel's* METADATA actually
carries ``License-File`` entries — is exercised separately by CI's ``build``
job, which builds the wheel, installs it torch-free, and smoke-tests it (see
``.github/workflows/ci.yml``).
"""

from __future__ import annotations

from pathlib import Path

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
    # Both texts must be listed so the built distribution ships them.
    assert "LICENSE" in files_line
    assert "hear_embed/_vendor/LICENSE.apache-2.0" in files_line

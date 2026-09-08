"""
The package must not lie about its own version.

On a site with no network and no support, ``invisensing.__version__`` is
the only thing an operator can ask to find out which SDK is installed. It
was wrong: ``__init__.py`` said 1.2.0 while ``pyproject.toml`` and
``Cargo.toml`` said 1.2.1 — a bump that touched one file out of three.
A reported version that is not the installed version is worse than no
version at all, because it is believed.

This test is the mechanical link that stops the three drifting apart
again: bumping a release now means bumping all three or going red here.
No dependency, no network — the two manifests are parsed with a
three-line regex rather than a TOML library, because the SDK must stay
installable with nothing but numpy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import invisensing


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _first_version_in(path: Path) -> str:
    """First top-level ``version = "X.Y.Z"`` of a manifest.

    Both manifests carry the package version in their first table
    (``[project]`` / ``[package]``), before any dependency table, so the
    first match is the package's own version.
    """
    for line in path.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
        if m:
            return m.group(1)
    pytest.skip(f"no version line in {path}")


@pytest.mark.parametrize("manifest", ["pyproject.toml", "Cargo.toml"])
def test_dunder_version_matches_the_manifests(manifest):
    path = _REPO_ROOT / manifest
    if not path.exists():
        pytest.skip(f"{manifest} not shipped with the installed package")
    declared = _first_version_in(path)
    assert invisensing.__version__ == declared, (
        f"invisensing.__version__ is {invisensing.__version__!r} but "
        f"{manifest} says {declared!r}: a release bumped one and not the "
        "other, and the installed package now misreports itself."
    )


def test_version_is_a_three_part_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", invisensing.__version__), (
        f"unexpected version shape {invisensing.__version__!r}"
    )

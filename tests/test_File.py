"""
Tests for the legacy ``invisensing.File`` shim.

These mirror the original public API (``iFile.File('…').get_*``) so the
new Rust-backed implementation stays a drop-in replacement.
"""

import os
import filecmp

import numpy as np
import pytest

import invisensing.File as iFile


ASSET = "tests/assets/demodulated.dat"

# The fixture is shipped in the git repo but excluded from the PyPI
# sdist (it's 18 MB of synthetic data, only useful for our own
# round-trip tests). Skip the legacy suite cleanly when it's missing
# instead of crashing on a file-not-found.
pytestmark = pytest.mark.skipif(
    not os.path.exists(ASSET),
    reason="tests/assets/demodulated.dat not present (likely installing from sdist)",
)


def test_open_file():
    file = iFile.File(ASSET)
    assert file.get_data_type() == np.float32
    assert file.get_line_size() == 487
    assert file.get_trigger_frequency() == 1000
    assert file.get_sample_rate() == 100000000
    assert file.is_acquisition_ac()
    assert file.is_demodulated()
    assert not file.is_acquisition_hiz()
    assert file.get_duration() == 10
    assert 490 < file.get_distance() < 510
    assert file.get_num_lines() == 10000
    assert file.get_range() == 2


def test_read_file():
    file = iFile.File(ASSET)
    data = file.get_lines()
    assert file.get_lines_left() == 9999
    assert data.shape == (1, 487)
    assert data.any()
    data = file.get_lines(50)
    assert file.get_lines_left() == 9949
    assert data.shape == (50, 487)
    data = file.get_lines(9940)
    assert file.get_lines_left() == 9
    assert data.shape == (9940, 487)
    data = file.get_lines(50)
    # short-read returns what's left rather than raising
    assert file.get_lines_left() == 0
    assert data.shape == (9, 487)
    with pytest.raises(OSError):
        file.get_lines()
    with pytest.raises(ValueError):
        file.get_lines(-1)


def test_export_file():
    file = iFile.File(ASSET)
    data = file.get_lines(10000)
    os.makedirs("tests/output", exist_ok=True)
    iFile.export(
        "tests/output/output.dat",
        data,
        file.get_timestamp(),
        file.get_trigger_frequency(),
        file.get_sample_rate(),
        file.get_range(),
        file.is_demodulated(),
        file.is_acquisition_ac(),
        file.is_acquisition_hiz(),
    )
    assert filecmp.cmp(ASSET, "tests/output/output.dat", shallow=False)

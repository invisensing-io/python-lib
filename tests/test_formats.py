"""
End-to-end tests for the non-DAT format backends.

Each test synthesises a file in the relevant format (using the same
third-party library the production wrapper relies on), then opens it
through the public ``File`` facade and verifies the metadata + channel
extraction round-trip.

Skipped automatically when the optional backend isn't installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Mode,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    FLAG_PHASE,
)


# ── HDF5 ──────────────────────────────────────────────────────────────────


@pytest.fixture
def h5py_or_skip():
    h5py = pytest.importorskip("h5py")
    return h5py


def test_hdf5_iq_round_trip(h5py_or_skip):
    h5py = h5py_or_skip
    rows, positions = 4, 6
    i = np.arange(rows * positions, dtype=np.int16).reshape(rows, positions) * 5
    q = i + 1
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.h5"
        flags = FLAG_DEMODULATED | FLAG_INTERLEAVED
        with h5py.File(str(path), "w") as fp:
            fp.create_dataset("acoustic_data", data=interleaved)
            fp.attrs["line_size"] = positions * 2
            fp.attrs["trig_frequency"] = 2_000
            fp.attrs["sample_size"] = 2
            fp.attrs["sample_rate"] = 250_000_000
            fp.attrs["flags"] = flags
            fp.attrs["range"] = 1_000
            fp.attrs["pulse_width"] = 4
            fp.attrs["num_channels"] = 1
            fp.attrs["timestamp"] = "2026-01-01_hdf5"

        with File(path) as f:
            assert f.mode is Mode.IQ
            assert f.positions_per_line == positions
            assert f.num_lines == rows
            buf = f.read_all()
            np.testing.assert_array_equal(f.get_i(buf), i)
            np.testing.assert_array_equal(f.get_q(buf), q)


def test_hdf5_phase_returns_float32(h5py_or_skip):
    h5py = h5py_or_skip
    rows, positions = 3, 5
    payload = np.linspace(-np.pi, np.pi, rows * positions, dtype=np.float32).reshape(
        rows, positions
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "phase.hdf5"
        flags = FLAG_DEMODULATED | FLAG_FLOAT | FLAG_PHASE
        with h5py.File(str(path), "w") as fp:
            fp.create_dataset("acoustic_data", data=payload)
            fp.attrs["line_size"] = positions
            fp.attrs["trig_frequency"] = 1_000
            fp.attrs["sample_size"] = 4
            fp.attrs["sample_rate"] = 250_000_000
            fp.attrs["flags"] = flags
            fp.attrs["range"] = 0
            fp.attrs["pulse_width"] = 0
            fp.attrs["num_channels"] = 1
            fp.attrs["timestamp"] = "2026-01-01_hdf5_phase"
        with File(path) as f:
            assert f.mode is Mode.PHASE
            assert f.is_float
            got = f.get_phase(n=rows)
            np.testing.assert_allclose(got, payload, rtol=1e-6)


# ── TDMS ──────────────────────────────────────────────────────────────────


@pytest.fixture
def nptdms_or_skip():
    return pytest.importorskip("nptdms")


def test_tdms_iq_round_trip(nptdms_or_skip):
    nptdms = nptdms_or_skip
    from nptdms import TdmsWriter, ChannelObject, RootObject

    rows, positions = 4, 4
    i = np.arange(rows * positions, dtype=np.int16).reshape(rows, positions) * 3
    q = i + 2
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q
    flat = interleaved.flatten()

    flags = FLAG_DEMODULATED | FLAG_INTERLEAVED
    root_props = {
        "line_size": positions * 2,
        "trig_frequency": 2_000,
        "sample_size": 2,
        "sample_rate": 250_000_000,
        "flags": flags,
        "range": 1_000,
        "pulse_width": 4,
        "num_channels": 1,
        "timestamp": "2026-01-01_tdms",
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.tdms"
        with TdmsWriter(str(path)) as w:
            w.write_segment(
                [
                    RootObject(properties=root_props),
                    ChannelObject("AudaceGroup", "acoustic_data", flat),
                ]
            )
        with File(path) as f:
            assert f.mode is Mode.IQ
            buf = f.read_all()
            np.testing.assert_array_equal(f.get_i(buf), i)
            np.testing.assert_array_equal(f.get_q(buf), q)

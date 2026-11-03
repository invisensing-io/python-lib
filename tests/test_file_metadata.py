"""
What a file says about itself beyond the samples: a DAT cut short in the
middle of a line, the compression an HDF5 was written with (and the bound of
``ZSTD_LOSSY_1E4``), and how far :meth:`File.times_ns` may go.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Header,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_PHASE,
)
from invisensing._core import build_header_bytes

ROWS, COLS = 10, 4


def _dat(path: Path, cut: int = 0) -> Path:
    h = Header(line_size=COLS, trig_frequency=1_000, sample_size=2,
               sample_rate=125_000_000, flags=0, content_type=1)
    payload = np.arange(ROWS * COLS, dtype=np.int16).reshape(ROWS, COLS).tobytes()
    path.write_bytes(build_header_bytes(h) + payload[: len(payload) - cut])
    return path


# ── A DAT cut short mid-line ───────────────────────────────────────────────


def test_a_partial_last_line_is_reported_and_warned(tmp_path):
    path = _dat(tmp_path / "cut.dat", cut=3)  # 9 lines + 5 bytes of the 10th
    with pytest.warns(UserWarning, match=r"5 byte\(s\) after the last whole line"):
        f = File(path)
    with f:
        assert f.num_lines == 9
        assert f.trailing_bytes == 5
        data = f.read_all()
        assert data.shape == (9, COLS)
        np.testing.assert_array_equal(data.ravel(), np.arange(9 * COLS))


def test_a_complete_file_has_no_trailing_bytes_and_no_warning(tmp_path):
    path = _dat(tmp_path / "whole.dat")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with File(path) as f:
            assert f.trailing_bytes == 0
            assert f.num_lines == ROWS


def test_the_native_reader_counts_the_trailing_bytes(tmp_path):
    from invisensing import DatReader

    r = DatReader(str(_dat(tmp_path / "cut.dat", cut=7)))
    assert (r.num_lines, r.trailing_bytes) == (9, 1)
    r.close()


# ── Compression and the lossy bound (HDF5) ─────────────────────────────────


def _phase_h5(path: Path, *, compression: str) -> Path:
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")
    payload = np.linspace(-3, 3, 8 * 16, dtype=np.float32).reshape(8, 16)
    kw = {}
    if compression != "NONE":
        kw = dict(chunks=(4, 16), shuffle=True, **hdf5plugin.Zstd(clevel=1))
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=payload, **kw)
        for k, v in {"line_size": 16, "trig_frequency": 1_000, "sample_size": 4,
                     "sample_rate": 125_000_000,
                     "flags": FLAG_DEMODULATED | FLAG_FLOAT | FLAG_PHASE,
                     "content_type": 4}.items():
            fp.attrs[k] = np.uint32(v)
        if compression == "ZSTD_LOSSY_1E4":
            # What Audace-Lib's `write_hdf5_attrs` adds when it rounds phase.
            fp.attrs["quantization_step_rad"] = np.float64(2.0**-13)
            fp.attrs["quantization_max_error_rad"] = np.float64(2.0**-14)
    return path


@pytest.mark.parametrize("compression", ["NONE", "ZSTD", "ZSTD_LOSSY_1E4"])
def test_hdf5_says_how_it_was_compressed(tmp_path, compression):
    with File(_phase_h5(tmp_path / f"{compression}.h5", compression=compression)) as f:
        assert f.compression == compression
        if compression == "ZSTD_LOSSY_1E4":
            assert f.quantization_step_rad == 2.0**-13
            assert f.quantization_max_error_rad == 2.0**-14
        else:
            assert f.quantization_step_rad is None
            assert f.quantization_max_error_rad is None


def test_a_dat_has_no_compression(tmp_path):
    with File(_dat(tmp_path / "x.dat")) as f:
        assert f.compression is None
        assert f.quantization_step_rad is None and f.quantization_max_error_rad is None


def test_a_third_party_filter_is_named(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "gzip.h5"
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=np.zeros((4, 8), np.int16),
                          chunks=(2, 8), compression="gzip")
        for k, v in {"line_size": 8, "trig_frequency": 1_000, "sample_size": 2,
                     "sample_rate": 125_000_000, "flags": 0}.items():
            fp.attrs[k] = np.uint32(v)
    with File(path) as f:
        assert f.compression == "gzip"


# ── times_ns: never past the file, MemoryError instead of an abort ────────


def test_times_ns_is_clamped_to_the_lines_of_the_file(tmp_path):
    with File(_dat(tmp_path / "x.dat")) as f:
        assert len(f.times_ns()) == ROWS
        assert len(f.times_ns(0, 10**9)) == ROWS
        assert len(f.times_ns(7, 100)) == 3
        assert f.times_ns(7, 100).tolist() == [f.line_time_ns(i) for i in (7, 8, 9)]
        assert len(f.times_ns(ROWS + 5, 3)) == 0
        # line_time_ns still extrapolates past the last line.
        assert f.line_time_ns(ROWS + 5) == f.line_time_ns(0) + (ROWS + 5) * 1_000_000


def test_a_file_declaring_more_lines_than_memory_raises_memory_error(tmp_path):
    """An HDF5 can declare 10¹⁷ lines without storing any (a chunked dataset
    resized past its data). Its time vector (8·10¹⁷ bytes) cannot be
    allocated anywhere: the native core must raise, not abort Python."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "forged.hdf5"
    with h5py.File(str(path), "w") as fp:
        ds = fp.create_dataset("acoustic_data", data=np.zeros((3, 4), np.int16),
                               chunks=(1, 4), maxshape=(None, 4))
        ds.resize((10**17, 4))
        for k, v in {"line_size": 4, "trig_frequency": 1_000, "sample_size": 2,
                     "sample_rate": 125_000_000, "flags": 0}.items():
            fp.attrs[k] = np.uint32(v)
        fp.attrs["timestamp_ns"] = np.int64(1_790_000_000_000_000_000)
    with File(path) as f:
        assert f.num_lines == 10**17
        with pytest.raises(MemoryError):
            f.times_ns()
        # A bounded request on the same file is fine.
        assert len(f.times_ns(0, 3)) == 3


def test_the_native_vector_raises_memory_error_on_an_impossible_count():
    from invisensing import _core

    with pytest.raises(MemoryError):
        _core.time_axis_ns([0], [0], [0], (1_000, 1), 0, 2**62)

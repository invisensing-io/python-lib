"""
``File(path, phase_unit="nanostrain")``: a phase read as strain, in nm/m.

``strain = phase_rad / gauge_length_m * NANOSTRAIN_PER_RAD_METRE``, applied
while reading as one float32 multiply per phase sample by
``float32(NANOSTRAIN_PER_RAD_METRE / gauge_length_m)`` — so the expected
values here are exactly ``phase * np.float32(K / g)``, compared bit for bit.
Only a phase with a recorded gauge length is converted; everything else is
refused at open, with a message that says why.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pytest

from gauge_files import GAUGE, WRITERS, phase_samples, write_dat, write_h5
from invisensing import (
    FLAG_DEMODULATED,
    FLAG_INTERLEAVED,
    FLAG_PHASE,
    NANOSTRAIN_PER_RAD_METRE,
    File,
    Mode,
)

PHASE = 19  # dui_coherent_phase
#: The factor of float32 samples: K / g in float64, rounded once.
FACTOR = np.float32(NANOSTRAIN_PER_RAD_METRE / GAUGE)
ROWS, POSITIONS = 7, 6
_LIBS = {".h5": "h5py", ".tdms": "nptdms", ".sgy": "segyio"}


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """The reference multiplies NaNs and infinities on purpose."""
    with np.errstate(all="ignore"):
        yield


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _same(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


@pytest.fixture(params=sorted(WRITERS))
def phase_file(request, tmp_path):
    suffix = request.param
    if suffix in _LIBS:
        pytest.importorskip(_LIBS[suffix])
    data = phase_samples(ROWS, POSITIONS, seed=3)
    return WRITERS[suffix](tmp_path / f"phase{suffix}", data, PHASE, GAUGE), data


def test_every_read_path_gives_the_strain(phase_file):
    path, rad = phase_file
    want = rad * FACTOR
    with File(path, phase_unit="nanostrain") as f:
        assert f.phase_unit == "nanostrain"
        assert f.mode is Mode.PHASE and f.dtype == np.float32
        assert (f.quantity, f.unit) == ("strain", "nm/m")
        assert f.gauge_length_m == 9.6
        assert f.nanostrain_per_rad == float(FACTOR)
        assert "phase_unit='nanostrain'" in repr(f)
        _same(f.read_all(), want)
        f.rewind()
        _same(f.read_lines(3), want[:3])
        _same(f.read_lines(100), want[3:])
        f.rewind()
        _same(np.stack(list(f)), want)
        f.seek_lines(2)
        _same(f.get_phase(n=2), want[2:4])
        _same(f.channels(n=1)["phase"], want[4:5])
        f.rewind()
        _same(f.get_lines(2), want[:2])  # the legacy name
        # A buffer already read is taken as read: never converted twice.
        f.rewind()
        block = f.read_lines(ROWS)
        _same(f.get_phase(block), want)


def test_radians_stay_the_default(phase_file):
    path, rad = phase_file
    with File(path) as f:
        assert f.phase_unit == "rad"
        assert (f.quantity, f.unit) == ("phase", "rad")
        assert f.nanostrain_per_rad == float(FACTOR)  # known, not applied
        assert "nanostrain" not in repr(f)
        _same(f.read_all(), rad)
    with File(path, phase_unit="rad") as f:
        _same(f.read_all(), rad)


def test_an_export_writes_the_radians_as_stored(tmp_path):
    rad = phase_samples(ROWS, POSITIONS)
    src = write_dat(tmp_path / "src.dat", rad, content_type=PHASE, gauge=GAUGE)
    with File(src, phase_unit="nanostrain") as f:
        f.read_lines(2)
        f.export_dat(tmp_path / "copy.dat")
        _same(f.read_lines(1), rad[2:3] * FACTOR)  # the cursor is put back
    with File(tmp_path / "copy.dat") as g:
        _same(g.read_all(), rad)
        assert g.gauge_length_m == 9.6


def test_pairs_convert_the_phase_lanes_only(tmp_path):
    rad = phase_samples(ROWS, POSITIONS, seed=5)
    amp = np.abs(phase_samples(ROWS, POSITIONS, seed=6)) + 1.0
    amp[0, 1] = np.uint32(0x7F800001).view(np.float32)  # a signalling NaN stays one
    pairs = np.empty((ROWS, 2 * POSITIONS), dtype=np.float32)
    pairs[:, 0::2], pairs[:, 1::2] = rad, amp
    path = write_dat(
        tmp_path / "pairs.dat", pairs, content_type=22,
        flags=FLAG_DEMODULATED | FLAG_PHASE | FLAG_INTERLEAVED, gauge=GAUGE, format_version=0,
    )
    with File(path, phase_unit="nanostrain") as f:
        assert f.mode is Mode.PHASE_AMPLITUDE
        assert (f.quantity, f.unit) == ("strain+amplitude", "nm/m+count")
        block = f.read_all()
        _same(block[:, 0::2], rad * FACTOR)
        _same(block[:, 1::2], amp)
        _same(f.get_phase(block), rad * FACTOR)
        _same(f.get_amplitude(block), amp)
        f.rewind()
        ch = f.channels(n=ROWS)
        _same(ch["phase"], rad * FACTOR)
        _same(ch["amplitude"], amp)
        f.rewind()
        _same(next(iter(f)), block[0])


def test_quantization_stays_in_radians(tmp_path):
    pytest.importorskip("h5py")
    step = 2.0**-13
    rad = (np.round(phase_samples(ROWS, POSITIONS, seed=8) / step) * step).astype(np.float32)
    path = write_h5(
        tmp_path / "lossy.h5", rad, content_type=PHASE, gauge=GAUGE,
        extra={"quantization_step_rad": step, "quantization_max_error_rad": step / 2},
    )
    with File(path, phase_unit="nanostrain") as f:
        assert f.quantization_step_rad == step
        assert f.quantization_max_error_rad == step / 2
        _same(f.read_all(), rad * FACTOR)


def test_a_float64_or_big_endian_dataset_keeps_its_precision(tmp_path):
    """Not what a writer of the acquisition stores, but what a third-party
    HDF5 may: a float64 phase is converted in float64 (the f64 factor), a
    big-endian one is read native."""
    pytest.importorskip("h5py")
    rad = phase_samples(ROWS, POSITIONS, seed=9)
    p64 = write_h5(tmp_path / "f64.h5", rad.astype(np.float64), content_type=PHASE, gauge=GAUGE,
                   sample_size=4)
    with File(p64, phase_unit="nanostrain") as f:
        got = f.read_all()
        _same(got, rad.astype(np.float64) * (NANOSTRAIN_PER_RAD_METRE / GAUGE))
    pbe = write_h5(tmp_path / "be.h5", rad.astype(">f4"), content_type=PHASE, gauge=GAUGE)
    with File(pbe, phase_unit="nanostrain") as f:
        got = f.read_all()
        assert got.dtype == np.float32 and got.dtype.isnative
        _same(got, rad * FACTOR)


@pytest.mark.parametrize(
    "content_type, dtype, name",
    [(1, np.int16, "raw"), (3, np.int16, "hw_arctan"), (5, np.uint16, "hw_amplitude"),
     (13, np.int16, "hw_i"), (16, np.float32, "hw_amplitude_f32"), (26, np.float32, "sw_custom_f32"),
     (27, np.float32, "viz_variance")],
)
def test_a_file_that_holds_no_phase_is_refused(tmp_path, content_type, dtype, name):
    path = write_dat(tmp_path / "x.dat", np.zeros((2, 6), dtype=dtype),
                     content_type=content_type, gauge=GAUGE)
    with pytest.raises(ValueError, match=rf"not a phase") as e:
        File(path, phase_unit="nanostrain")
    assert name in str(e.value) and f"content type {content_type}" in str(e.value)


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_a_phase_without_its_gauge_is_refused(tmp_path, suffix):
    if suffix in _LIBS:
        pytest.importorskip(_LIBS[suffix])
    path = WRITERS[suffix](tmp_path / f"p{suffix}", phase_samples(2, 6), PHASE, None)
    with pytest.raises(ValueError, match="does not record its gauge length") as e:
        File(path, phase_unit="nanostrain")
    # What is true of every file without it, with no date: the producers
    # write it only from the Audace release that adds it.
    assert "only from the release that adds it" in str(e.value)
    assert "2026" not in str(e.value).replace(str(path), "")


def test_a_mozart_or_a_legacy_phase_is_refused(tmp_path):
    mozart = write_dat(tmp_path / "m.dat", phase_samples(2, 6), content_type=9)
    with pytest.raises(ValueError, match="does not record its gauge length.*MOZART"):
        File(mozart, phase_unit="nanostrain")
    legacy = write_dat(
        tmp_path / "legacy.dat", phase_samples(2, 6), content_type=0,
        flags=FLAG_DEMODULATED | FLAG_PHASE, gauge=GAUGE, format_version=0,
    )
    with pytest.raises(ValueError, match="does not record its gauge length"):
        File(legacy, phase_unit="nanostrain")


def test_a_refusal_closes_the_file(tmp_path):
    path = write_dat(tmp_path / "raw.dat", np.zeros((2, 6), np.int16), content_type=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        with pytest.raises(ValueError):
            File(path, phase_unit="nanostrain")
        gc.collect()


@pytest.mark.parametrize("bad", ["strain", "nm/m", "RAD", None, 1])
def test_an_unknown_phase_unit_is_refused(tmp_path, bad):
    path = write_dat(tmp_path / "p.dat", phase_samples(2, 6), content_type=PHASE, gauge=GAUGE)
    with pytest.raises(ValueError, match="phase_unit must be"):
        File(path, phase_unit=bad)

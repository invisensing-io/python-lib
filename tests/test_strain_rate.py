"""
``File(path, phase_unit="strain_rate")``: a phase read as strain rate, in nm/m/s.

A phase file holds the accumulated phase of each line, so the strain rate of
line ``n`` is the strain increment from line ``n − 1`` times the line rate:

    strain_rate[n] = (phase[n] − phase[n−1]) × float32(K / g × f_trig)

one float32 subtraction then one float32 multiply per phase sample, the
factor computed once in float64 and rounded once (``K`` =
``NANOSTRAIN_PER_RAD_METRE``, ``g`` = the gauge length, ``f_trig`` = the real
trigger frequency). The expected values here are that numpy expression,
compared bit for bit (NaN payloads included). Line 0 has no predecessor and
the lines after a gap of the time axis span more than one period: both read
NaN in their phase lanes.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pytest

from gauge_files import ANCHOR_NS, GAUGE, TRIG, WRITERS, phase_samples, write_dat, write_h5
from invisensing import (
    FLAG_DEMODULATED,
    FLAG_INTERLEAVED,
    FLAG_PHASE,
    NANOSTRAIN_PER_RAD_METRE,
    File,
    Mode,
)
from invisensing._core import strain_rate_factor
from invisensing._reader import _axis_time_ns

PHASE = 19  # dui_coherent_phase
#: The factor of float32 samples: K / g × f_trig in float64, rounded once.
FACTOR = np.float32(NANOSTRAIN_PER_RAD_METRE / GAUGE * TRIG)
ROWS, POSITIONS = 23, 6
_LIBS = {".h5": "h5py", ".tdms": "nptdms", ".sgy": "segyio"}


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """The reference subtracts NaNs and infinities on purpose."""
    with np.errstate(all="ignore"):
        yield


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _same(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


def _rate(rad: np.ndarray, factor=FACTOR, nan_rows=()) -> np.ndarray:
    """The reference: row 0 NaN, then `(x[n] − x[n−1]) × factor`; the rows of
    `nan_rows` NaN too."""
    want = np.empty_like(rad)
    want[0] = np.nan
    want[1:] = (rad[1:] - rad[:-1]) * factor
    for r in nan_rows:
        want[r] = np.nan
    return want


@pytest.fixture(params=sorted(WRITERS))
def phase_file(request, tmp_path):
    suffix = request.param
    if suffix in _LIBS:
        pytest.importorskip(_LIBS[suffix])
    data = phase_samples(ROWS, POSITIONS, seed=3)
    return WRITERS[suffix](tmp_path / f"phase{suffix}", data, PHASE, GAUGE), data


def test_the_definition_on_a_ramp_and_a_sine(tmp_path):
    """The values a user can check by hand: a phase ramp of `a` rad per line
    reads `a × K / g × f` nm/m/s on every line but the first; a sine reads
    the difference of the sine, i.e. a cosine to first order."""
    a = np.float32(0.25)
    ramp = np.repeat((np.arange(ROWS, dtype=np.float32) * a)[:, None], POSITIONS, axis=1)
    with File(write_dat(tmp_path / "ramp.dat", ramp, content_type=PHASE, gauge=GAUGE),
              phase_unit="strain_rate") as f:
        got = f.read_all()
    assert np.isnan(got[0]).all()
    expected = a * NANOSTRAIN_PER_RAD_METRE / GAUGE * TRIG  # 143.71 nm/m/s
    np.testing.assert_allclose(got[1:], expected, rtol=1e-6)
    w = 2 * np.pi * 0.5  # 0.5 Hz, 1 rad
    t = np.arange(400) / TRIG
    sine = np.repeat(np.sin(w * t).astype(np.float32)[:, None], 2, axis=1)
    with File(write_dat(tmp_path / "sine.dat", sine, content_type=PHASE, gauge=GAUGE),
              phase_unit="strain_rate") as f:
        got = f.read_all()
    assert np.isnan(got[0]).all()
    # sin(ωt_n) − sin(ωt_n−1) = 2 sin(ω/2f) cos(ω(t_n − 1/2f)): a cosine,
    # ω cos(ω(t_n − 1/2f)) × K / g to first order in ω/f.
    exact = 2 * TRIG * np.sin(w / (2 * TRIG)) * np.cos(w * (t[1:] - 0.5 / TRIG))
    exact *= NANOSTRAIN_PER_RAD_METRE / GAUGE
    np.testing.assert_allclose(got[1:, 0], exact, atol=1e-4 * np.abs(exact).max())
    first_order = w * np.cos(w * (t[1:] - 0.5 / TRIG)) * NANOSTRAIN_PER_RAD_METRE / GAUGE
    np.testing.assert_allclose(got[1:, 1], first_order, atol=1e-3 * np.abs(first_order).max())


def test_every_read_path_gives_the_strain_rate(phase_file):
    path, rad = phase_file
    want = _rate(rad)
    with File(path, phase_unit="strain_rate") as f:
        assert f.phase_unit == "strain_rate"
        assert f.mode is Mode.PHASE and f.dtype == np.float32
        assert (f.quantity, f.unit) == ("strain rate", "nm/m/s")
        assert f.strain_rate_per_rad == float(FACTOR)
        assert f.nanostrain_per_rad == float(np.float32(NANOSTRAIN_PER_RAD_METRE / GAUGE))
        assert f.time_gaps == ()
        assert "phase_unit='strain_rate'" in repr(f)
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


def test_blocks_of_any_size_give_the_bits_of_one_read(phase_file):
    path, rad = phase_file
    with File(path, phase_unit="strain_rate") as f:
        whole = f.read_all()
        _same(whole, _rate(rad))
        for size in range(1, ROWS + 2):
            f.rewind()
            parts = []
            while f.lines_left:
                parts.append(f.read_lines(size))
            _same(np.concatenate(parts), whole)


def test_a_seek_reads_the_line_before(phase_file):
    path, rad = phase_file
    want = _rate(rad)
    with File(path, phase_unit="strain_rate") as f:
        for k in [ROWS - 1, 0, 7, 1, 7, 8, 22, 3]:  # back, forth, the same line twice
            f.seek_lines(k)
            _same(f.read_lines(1), want[k:k + 1])
        # Iteration from a seek.
        f.seek_lines(5)
        _same(np.stack(list(f)), want[5:])
        # An oscilloscope: one line per frame, anywhere.
        rng = np.random.default_rng(0)
        for k in rng.integers(0, ROWS, 40):
            f.seek_lines(int(k))
            _same(f.get_phase(n=1), want[k:k + 1])


def test_an_export_writes_the_radians_and_keeps_the_carried_line(tmp_path):
    rad = phase_samples(ROWS, POSITIONS)
    src = write_dat(tmp_path / "src.dat", rad, content_type=PHASE, gauge=GAUGE)
    with File(src, phase_unit="strain_rate") as f:
        f.read_lines(4)
        f.export_dat(tmp_path / "copy.dat")
        _same(f.read_lines(2), _rate(rad)[4:6])  # the cursor is put back
    with File(tmp_path / "copy.dat") as g:
        _same(g.read_all(), rad)


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
    want = _rate(rad)
    with File(path, phase_unit="strain_rate") as f:
        assert f.mode is Mode.PHASE_AMPLITUDE
        assert (f.quantity, f.unit) == ("strain rate+amplitude", "nm/m/s+count")
        block = f.read_all()
        _same(block[:, 0::2], want)
        _same(block[:, 1::2], amp)
        _same(f.get_phase(block), want)
        _same(f.get_amplitude(block), amp)
        f.rewind()
        ch = f.channels(n=ROWS)
        _same(ch["phase"], want)
        _same(ch["amplitude"], amp)
        f.seek_lines(9)
        _same(next(iter(f)), block[9])


def test_the_real_trigger_frequency_is_the_line_rate(tmp_path):
    """An inexact rate (the card fires at 250 MHz / 4 999 999, not 50 Hz)
    is the one the increments are multiplied by."""
    pytest.importorskip("h5py")
    rad = phase_samples(ROWS, POSITIONS, seed=7)
    clock, ticks = 250_000_000, 4_999_999
    path = write_h5(
        tmp_path / "inexact.h5", rad, content_type=PHASE, gauge=GAUGE,
        extra={"trig_clock_hz": np.uint32(clock), "trig_period_ticks": np.uint32(ticks)},
    )
    factor = np.float32(NANOSTRAIN_PER_RAD_METRE / GAUGE * (clock / ticks))
    assert factor != FACTOR
    with File(path, phase_unit="strain_rate") as f:
        assert f.trig_frequency == TRIG and f.trig_frequency_real_hz == clock / ticks
        assert f.strain_rate_per_rad == float(factor)
        _same(f.read_all(), _rate(rad, factor))


def test_a_float64_or_big_endian_dataset_keeps_its_precision(tmp_path):
    pytest.importorskip("h5py")
    rad = phase_samples(ROWS, POSITIONS, seed=9)
    p64 = write_h5(tmp_path / "f64.h5", rad.astype(np.float64), content_type=PHASE, gauge=GAUGE,
                   sample_size=4)
    with File(p64, phase_unit="strain_rate") as f:
        r64 = rad.astype(np.float64)
        _same(f.read_all(), _rate(r64, NANOSTRAIN_PER_RAD_METRE / GAUGE * TRIG))
        f.seek_lines(10)
        _same(f.read_lines(3), _rate(r64, NANOSTRAIN_PER_RAD_METRE / GAUGE * TRIG)[10:13])
    pbe = write_h5(tmp_path / "be.h5", rad.astype(">f4"), content_type=PHASE, gauge=GAUGE)
    with File(pbe, phase_unit="strain_rate") as f:
        got = f.read_all()
        assert got.dtype == np.float32 and got.dtype.isnative
        _same(got, _rate(rad))
        f.seek_lines(5)
        _same(f.read_lines(2), _rate(rad)[5:7])


# ── Gaps of the time axis ──────────────────────────────────────────────────

PERIOD_NS = 1_000_000_000 // TRIG  # 20 ms
PERIOD_FS = PERIOD_NS * 1_000_000


def _axis_file(tmp_path, rad, axis, correction="cardclock-v1"):
    """An HDF5 stamped from a time map whose nodes are `axis` (line, ns,
    period_fs)."""
    lines, ns, fs = (list(c) for c in zip(*axis))
    return write_h5(
        tmp_path / "axis.h5", rad, content_type=PHASE, gauge=GAUGE,
        extra={
            "timestamp_ns": np.int64(_axis_time_ns(lines, ns, fs, (0, 0), 0)),
            "time_correction": correction,
            "time_map_line": np.asarray(lines, dtype=np.int64),
            "time_map_ns": np.asarray(ns, dtype=np.int64),
            "time_map_period_fs": np.asarray(fs, dtype=np.uint64),
            "time_correction_state": "tracking",
        },
    )


def _gappy_axis():
    """Two shots lost before line 10; a drift from line 20 (no gap); a stretch
    of three periods per line from line 25 (lines 26.. follow gaps)."""
    drift = PERIOD_FS + 3 * PERIOD_FS // 1_000_000  # +3 ppm
    t10 = ANCHOR_NS + 10 * PERIOD_NS + 2 * PERIOD_NS  # the step of 2 lost shots
    t20 = t10 + 10 * PERIOD_NS
    t25 = t20 + (5 * drift) // 1_000_000
    return [
        (0, ANCHOR_NS, PERIOD_FS),
        (10, t10, PERIOD_FS),
        (20, t20, drift),
        (25, t25, 3 * PERIOD_FS),
    ]


def test_a_gap_of_the_time_axis_reads_nan(tmp_path):
    pytest.importorskip("h5py")
    rows = 30
    rad = phase_samples(rows, POSITIONS, seed=11)
    path = _axis_file(tmp_path, rad, _gappy_axis())
    nan_rows = [10, 26, 27, 28, 29]
    want = _rate(rad, nan_rows=nan_rows)
    with File(path, phase_unit="strain_rate") as f:
        assert f.time_correction == "cardclock-v1"
        assert f.time_gaps == ((10, 11), (26, 30))
        t = f.times_ns()
        gap = np.flatnonzero(np.diff(t) > 1.5 * PERIOD_NS) + 1
        assert list(gap) == nan_rows  # the same lines as the times say
        _same(f.read_all(), want)
        for size in (1, 2, 3, 7, 10, 11, 29):
            f.rewind()
            parts = []
            while f.lines_left:
                parts.append(f.read_lines(size))
            _same(np.concatenate(parts), want)
        for k in (11, 10, 26, 9, 25):
            f.seek_lines(k)
            _same(f.read_lines(1), want[k:k + 1])
        f.rewind()
        _same(np.stack(list(f)), want)
    # The same file in radians or as strain: nothing is NaN.
    with File(path) as f:
        assert f.time_gaps == ((10, 11), (26, 30))  # metadata, whatever the unit
        _same(f.read_all(), rad)


def test_a_file_timed_by_its_count_has_no_gap(tmp_path):
    pytest.importorskip("h5py")
    rad = phase_samples(ROWS, POSITIONS, seed=12)
    path = _axis_file(tmp_path, rad, _gappy_axis(), correction="none")
    with File(path, phase_unit="strain_rate") as f:
        assert f.time_correction == "none" and f.time_gaps == ()
        _same(f.read_all(), _rate(rad))


def test_a_steady_axis_has_no_gap(tmp_path):
    """A drift of the card's oscillator (thousands of ppm without its
    reference) is no gap: the line rate stays the factor's."""
    pytest.importorskip("h5py")
    rad = phase_samples(ROWS, POSITIONS, seed=13)
    slow = PERIOD_FS + 7_500 * PERIOD_FS // 1_000_000
    t0 = ANCHOR_NS - 3 * PERIOD_NS
    path = _axis_file(tmp_path, rad, [(-3, t0, slow), (12, t0 + 15 * slow // 1_000_000, PERIOD_FS)])
    with File(path, phase_unit="strain_rate") as f:
        assert f.time_gaps == ()
        _same(f.read_all(), _rate(rad))


# ── Refusals ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content_type, dtype, name",
    [(1, np.int16, "raw"), (3, np.int16, "hw_arctan"), (5, np.uint16, "hw_amplitude"),
     (13, np.int16, "hw_i"), (16, np.float32, "hw_amplitude_f32"), (26, np.float32, "sw_custom_f32"),
     (27, np.float32, "viz_variance")],
)
def test_a_file_that_holds_no_phase_is_refused(tmp_path, content_type, dtype, name):
    path = write_dat(tmp_path / "x.dat", np.zeros((2, 6), dtype=dtype),
                     content_type=content_type, gauge=GAUGE)
    with pytest.raises(ValueError, match=r"strain rate \(nm/m/s\).*not a phase") as e:
        File(path, phase_unit="strain_rate")
    assert name in str(e.value) and f"content type {content_type}" in str(e.value)


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_a_phase_without_its_gauge_is_refused(tmp_path, suffix):
    if suffix in _LIBS:
        pytest.importorskip(_LIBS[suffix])
    path = WRITERS[suffix](tmp_path / f"p{suffix}", phase_samples(2, 6), PHASE, None)
    with pytest.raises(ValueError, match="'strain_rate' needs the gauge length.*does not record"):
        File(path, phase_unit="strain_rate")


def test_a_phase_without_a_trigger_frequency_is_refused(tmp_path):
    path = write_dat(tmp_path / "p.dat", phase_samples(3, 6), content_type=PHASE, gauge=GAUGE)
    raw = bytearray(path.read_bytes())
    raw[36:40] = (0).to_bytes(4, "little")  # trig_frequency = 0
    path.write_bytes(bytes(raw))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        with pytest.raises(ValueError, match="trigger frequency.*unknown.*trig_frequency = 0 Hz"):
            File(path, phase_unit="strain_rate")
        gc.collect()  # the refusal closed the file
    with File(path) as f:  # what is unknown is the rate, not the file
        assert f.trig_frequency == 0 and f.strain_rate_per_rad is None
        assert f.nanostrain_per_rad is not None
    with File(path, phase_unit="nanostrain") as f:
        assert f.read_all().shape == (3, 6)


def test_the_factor_refuses_what_is_no_rate():
    assert strain_rate_factor(GAUGE, TRIG) == float(FACTOR)
    for bad in (0.0, -50.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not a line rate"):
            strain_rate_factor(GAUGE, bad)
    with pytest.raises(ValueError, match="not a gauge length"):
        strain_rate_factor(0.0, TRIG)


@pytest.mark.parametrize("bad", ["strain-rate", "nm/m/s", "STRAIN_RATE", "rate"])
def test_an_unknown_phase_unit_is_refused(tmp_path, bad):
    path = write_dat(tmp_path / "p.dat", phase_samples(2, 6), content_type=PHASE, gauge=GAUGE)
    with pytest.raises(ValueError, match="phase_unit must be .*'strain_rate'"):
        File(path, phase_unit=bad)

"""
Corrected line times: the card's time map, as a recording carries it.

The acquisition (CardControl) measures how its card's oscillator drifts from
the host clock, and how many shots it lost, and publishes a *time map* — a
piecewise-affine function "line → corrected time" — that every module of the
chain evaluates with one integer formula. The recorder writes the part of it
that covers each file into the HDF5 attributes:

* ``timestamp_ns`` — the CORRECTED time of the first line (decision R7-A),
  ``timestamp_nominal_ns`` — the count it replaces, ``timestamp_last_ns``;
* ``time_map_line`` / ``time_map_ns`` / ``time_map_period_fs`` — the axis:
  from node ``k`` on, ``ns[k] + ⌊(i − line[k]) · period_fs[k] / 10⁶⌋``, a
  period 0 counting on the card's rational;
* ``time_correction`` (``"cardclock-v1"`` / ``"none"``), the measured cadence
  and the state of the correction.

DAT, SEG-Y and TDMS keep their format: the start time in their header is the
corrected one, and DataHeader bit ``TIME_CORRECTED`` (bytes 100..104) says so.

What these tests pin down:

* the same integers as the acquisition's ``audace::time_map::axis_time_of``,
  on the fixtures both code bases check (``tests/assets/time_axis_v1.json``);
* every file written before the map is timed exactly as 1.3.0 timed it;
* ``line_time_ns(i, nominal=True)`` keeps the count.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import invisensing
from invisensing import (
    File,
    Header,
    TIME_FLAG_CORRECTED,
    TIME_FLAG_MAP_V1,
    parse_header,
)
from invisensing._core import build_header_bytes, time_axis_ns
from invisensing._reader import _axis_time_ns

h5py = pytest.importorskip("h5py")

FIXTURES = Path(__file__).parent / "assets" / "time_axis_v1.json"
CLOCK = 250_000_000
ANCHOR = 1_760_000_000_000_000_000
ROWS = 50
LINE_SIZE = 8


def _cases():
    return json.loads(FIXTURES.read_text())["cases"]


# ── 1. The shared fixtures: the acquisition's integers ─────────────────────


def test_the_shared_fixtures_give_the_acquisition_s_integers():
    checked = 0
    for case in _cases():
        timing = (case["clock_hz"], case["period_ticks"])
        lines, ns, fs = (list(c) for c in zip(*case["axis"]))
        for line, want in case["queries"]:
            assert _axis_time_ns(lines, ns, fs, timing, line) == want, (case["name"], line)
            # The native vector: the same integer, one at a time.
            got = time_axis_ns(lines, ns, fs, timing, line, 1)
            assert got.dtype == np.int64
            assert int(got[0]) == want, (case["name"], line)
            checked += 1
    assert checked > 100


def test_the_native_vector_is_the_scalar_formula_over_a_range():
    case = _cases()[2]  # 75 kHz: nominal node, tracking, a loss step
    timing = (case["clock_hz"], case["period_ticks"])
    lines, ns, fs = (list(c) for c in zip(*case["axis"]))
    first = 29_990
    vec = time_axis_ns(lines, ns, fs, timing, first, 30_000)
    want = [_axis_time_ns(lines, ns, fs, timing, first + j) for j in range(30_000)]
    assert vec.tolist() == want
    # Strictly increasing (a forward step at the loss, never backwards).
    assert np.all(np.diff(vec) > 0)


def test_the_native_vector_refuses_a_broken_axis():
    with pytest.raises(ValueError):
        time_axis_ns([], [], [], None, 0, 3)
    with pytest.raises(ValueError):
        time_axis_ns([0, 0], [1, 2], [1, 1], None, 0, 3)
    with pytest.raises(ValueError):
        time_axis_ns([0], [1, 2], [1], None, 0, 3)


# ── 2. HDF5 written from the time map ──────────────────────────────────────


def _write_hdf5(path: Path, *, f: int = 2_000, ticks: int = 125_000, rows: int = ROWS,
                time_attrs=None, timestamp_ns: int = ANCHOR) -> Path:
    payload = np.arange(rows * LINE_SIZE, dtype=np.int16).reshape(rows, LINE_SIZE)
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=payload)
        for k, v in {
            "line_size": LINE_SIZE, "trig_frequency": f, "sample_size": 2,
            "sample_rate": 100_000_000, "flags": 0, "range": 1_000,
            "pulse_width": 40, "num_channels": 1,
        }.items():
            fp.attrs[k] = np.uint32(v)
        fp.attrs["timestamp"] = "2026-10-21 08:00:00.000+02:00"
        fp.attrs["timestamp_ns"] = np.int64(timestamp_ns)
        fp.attrs["trig_clock_hz"] = np.uint32(CLOCK)
        fp.attrs["trig_period_ticks"] = np.uint32(ticks)
        fp.attrs["trig_frequency_real_hz"] = np.float64(CLOCK / ticks)
        for k, v in (time_attrs or {}).items():
            fp.attrs[k] = v
    return path


def _corrected_attrs(axis, *, nominal_start, last=None, state="tracking"):
    lines, ns, fs = (list(c) for c in zip(*axis))
    attrs = {
        "time_correction": "cardclock-v1",
        "timestamp_nominal_ns": np.int64(nominal_start),
        "time_map_line": np.asarray(lines, dtype=np.int64),
        "time_map_ns": np.asarray(ns, dtype=np.int64),
        "time_map_period_fs": np.asarray(fs, dtype=np.uint64),
        "trig_frequency_measured_hz": np.float64(1_999.990_4),
        "trig_frequency_error_ppm": np.float64(-4.8),
        "time_uncertainty_ns": np.int64(61_000),
        "time_prior_ppm": np.float64(-4.9),
        "time_correction_state": state,
        "time_correction_ns": np.int64(0),
        "time_reference_synchronized": np.uint8(1),
    }
    if last is not None:
        attrs["timestamp_last_ns"] = np.int64(last)
    return attrs


def test_a_corrected_hdf5_is_timed_on_its_axis(tmp_path):
    case = _cases()[0]  # 2 kHz
    axis = case["axis"]
    lines, ns, fs = (list(c) for c in zip(*axis))
    timing = (CLOCK, case["period_ticks"])
    start = _axis_time_ns(lines, ns, fs, timing, 0)
    last = _axis_time_ns(lines, ns, fs, timing, ROWS - 1)
    nominal_start = ANCHOR + 123_457 * 500_000  # the count of the same line
    path = _write_hdf5(
        tmp_path / "corrected.hdf5",
        timestamp_ns=start,
        time_attrs=_corrected_attrs(axis, nominal_start=nominal_start, last=last),
    )
    with File(path) as f:
        assert f.time_correction == "cardclock-v1"
        assert f.header.time_corrected
        assert f.header.time_flags == TIME_FLAG_CORRECTED
        assert f.time_correction_state == "tracking"
        assert f.time_uncertainty_ns == 61_000
        assert f.trig_frequency_measured_hz == pytest.approx(1_999.990_4)
        # The spectrum's rate stays the programmed cadence.
        assert f.trig_frequency_real_hz == 2_000.0
        for i in [0, 1, 7, ROWS - 1, 10**6]:
            assert f.line_time_ns(i) == _axis_time_ns(lines, ns, fs, timing, i)
        assert f.line_time_ns(0) == start == f.header.acquisition_ns
        assert f.times_ns().tolist() == [f.line_time_ns(i) for i in range(ROWS)]
        assert f.times_ns(5, 3).tolist() == [f.line_time_ns(i) for i in (5, 6, 7)]
        # The count, as 1.3.0 computed it, from the nominal start.
        for i in [0, 3, ROWS - 1]:
            assert f.line_time_ns(i, nominal=True) == nominal_start + i * 500_000
        assert f.times_ns(0, 3, nominal=True).tolist() == [nominal_start + i * 500_000 for i in range(3)]
        # The duration on the corrected times: first to last over N − 1 periods.
        assert f.duration == pytest.approx((last - start) / 1e9 * ROWS / (ROWS - 1), rel=1e-12)
    # Closed: still answers (metadata).
    assert f.line_time_ns(3) == _axis_time_ns(lines, ns, fs, timing, 3)
    assert f.times_ns(0, 2).tolist() == [f.line_time_ns(0), f.line_time_ns(1)]


def test_a_loss_step_inside_a_file_is_a_real_gap(tmp_path):
    case = _cases()[0]
    lines, ns, fs = (list(c) for c in zip(*case["axis"]))
    # The same axis seen from a file that starts 10 lines before the loss
    # knot (this file's line 0 = the fixture file's line 49 990).
    shift = 49_990
    axis = [(l - shift, t, p) for l, t, p in zip(lines, ns, fs)]
    axis = [axis[1], axis[2]]  # the knot before the file, the loss knot in it
    a_lines, a_ns, a_fs = (list(c) for c in zip(*axis))
    timing = (CLOCK, 125_000)
    path = _write_hdf5(
        tmp_path / "loss.hdf5",
        timestamp_ns=_axis_time_ns(a_lines, a_ns, a_fs, timing, 0),
        time_attrs=_corrected_attrs(axis, nominal_start=0),
    )
    with File(path) as f:
        t = f.times_ns()
        step = np.diff(t)
        # 3.25 ms of lost shots between line 9 and line 10, one period elsewhere.
        assert step[9] - step[8] == pytest.approx(3_250_000, abs=2)
        assert np.all(step > 0)


def test_a_file_timed_by_the_count_says_none_and_keeps_1_3_0_s_times(tmp_path):
    path = _write_hdf5(
        tmp_path / "none.hdf5",
        time_attrs={
            "time_correction": "none",
            "timestamp_nominal_ns": np.int64(ANCHOR),
            "timestamp_last_ns": np.int64(ANCHOR + (ROWS - 1) * 500_000),
            "time_correction_ns": np.int64(0),
        },
    )
    old = _write_hdf5(tmp_path / "old.hdf5")
    with File(path) as f, File(old) as g:
        assert f.time_correction == "none"
        assert g.time_correction is None
        assert not f.header.time_corrected and not g.header.time_corrected
        for i in [0, 1, ROWS - 1, 10**9]:
            want = ANCHOR + i * 500_000
            assert f.line_time_ns(i) == want == g.line_time_ns(i)
            assert f.line_time_ns(i, nominal=True) == want == g.line_time_ns(i, nominal=True)
        assert f.times_ns().tolist() == g.times_ns().tolist() == [ANCHOR + i * 500_000 for i in range(ROWS)]
        # Duration: the very float of 1.3.0.
        assert f.duration == g.duration == ROWS / 2_000.0
        assert f.trig_frequency_measured_hz is None


def test_an_inexact_cadence_without_a_map_times_bit_identically_to_1_3_0(tmp_path):
    # 75 kHz → 3 333 ticks, the rational's own count.
    path = _write_hdf5(tmp_path / "75k.hdf5", f=75_000, ticks=3_333)
    with File(path) as f:
        for i in [0, 1, 74_999, 75_000, 270_000_000]:
            assert f.line_time_ns(i) == ANCHOR + i * 3_333 * 10**9 // CLOCK
        assert f.times_ns(74_998, 3).tolist() == [ANCHOR + i * 3_333 * 10**9 // CLOCK for i in (74_998, 74_999, 75_000)]


def test_a_broken_axis_is_ignored(tmp_path):
    attrs = _corrected_attrs([(0, ANCHOR, 500_000_000_000)], nominal_start=ANCHOR)
    attrs["time_map_ns"] = np.asarray([ANCHOR, ANCHOR], dtype=np.int64)  # lengths differ
    path = _write_hdf5(tmp_path / "broken.hdf5", time_attrs=attrs)
    with File(path) as f:
        assert f.time_correction == "cardclock-v1"
        # No usable axis: the count from the (corrected) start, as for a DAT.
        assert f.line_time_ns(10) == ANCHOR + 10 * 500_000


def test_a_forged_axis_is_refused_and_never_overflows(tmp_path):
    # A node at the most negative line with the largest period: the product
    # would overflow an i128. The file's axis is not trusted (the count from
    # its start), and the native vector refuses it — no panic either way.
    i64_min = -(2**63)
    attrs = _corrected_attrs([(i64_min, 0, 2**64 - 1)], nominal_start=ANCHOR)
    path = _write_hdf5(tmp_path / "forged.hdf5", time_attrs=attrs)
    with File(path) as f:
        assert f.line_time_ns(10) == ANCHOR + 10 * 500_000
        assert f.times_ns(0, 3).tolist() == [ANCHOR, ANCHOR + 500_000, ANCHOR + 1_000_000]
    with pytest.raises(ValueError):
        time_axis_ns([i64_min], [0], [2**64 - 1], None, 0, 3)
    # Within the bound, the farthest line still saturates instead of wrapping.
    far = time_axis_ns([i64_min], [0], [10**18], None, 2**63 - 3, 2)
    assert far.tolist() == [2**63 - 1, 2**63 - 1]


# ── 3. DAT: time_flags and line_offset in the header ───────────────────────


def _dat_header_bytes(*, line_offset: int = 0, time_flags: int = 0) -> bytes:
    buf = bytearray(128)
    struct.pack_into("<q", buf, 0, ANCHOR)
    struct.pack_into("<8i", buf, 32, LINE_SIZE, 2_000, 2, 100_000_000, 0, 1_000, 40, 1)
    buf[64] = 1
    struct.pack_into("<II", buf, 88, CLOCK, 125_000)
    struct.pack_into("<iI", buf, 96, line_offset, time_flags)
    return bytes(buf)


def test_the_dat_header_carries_time_flags_and_line_offset(tmp_path):
    h = parse_header(_dat_header_bytes(line_offset=-37, time_flags=TIME_FLAG_MAP_V1))
    assert (h.line_offset, h.time_flags) == (-37, TIME_FLAG_MAP_V1)
    assert not h.time_corrected
    assert "line_offset=-37" in repr(h)
    # The writer puts them back where the reader found them.
    assert bytes(build_header_bytes(h)) == _dat_header_bytes(line_offset=-37, time_flags=TIME_FLAG_MAP_V1)

    path = tmp_path / "corrected.dat"
    payload = np.arange(ROWS * LINE_SIZE, dtype=np.int16).reshape(ROWS, LINE_SIZE)
    path.write_bytes(_dat_header_bytes(time_flags=TIME_FLAG_CORRECTED) + payload.tobytes())
    with File(path) as f:
        assert f.header.time_corrected
        assert f.time_correction == "cardclock-v1"
        # Its start is the corrected time of its first line; it counts from there.
        assert f.line_time_ns(4) == ANCHOR + 4 * 500_000
        assert f.times_ns().tolist() == [ANCHOR + i * 500_000 for i in range(ROWS)]
        np.testing.assert_array_equal(f.read_all(), payload)


def test_an_older_dat_header_has_no_time_map():
    old = bytearray(_dat_header_bytes())
    old[96:104] = bytes(8)
    h = parse_header(bytes(old))
    assert (h.line_offset, h.time_flags, h.time_corrected) == (0, 0, False)
    assert "line_offset" not in repr(h)
    assert Header(line_size=4, trig_frequency=1, sample_size=2, sample_rate=1, flags=0).time_flags == 0


def test_the_constants_are_the_wire_bits():
    assert TIME_FLAG_MAP_V1 == 1
    assert TIME_FLAG_CORRECTED == 2
    assert "TIME_FLAG_CORRECTED" in invisensing.__all__

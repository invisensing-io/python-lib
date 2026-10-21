"""
Real trigger frequency: the card's rational, and the times counted on it.

A card fires on a divided clock: a 250 MHz PCIe7821/PCIe7818 programs
``floor(250e6 / f)`` ticks, so it fires at exactly ``f`` only when ``f``
divides the clock (27 values between 1 kHz and 200 kHz). Asked for 75 kHz,
it fires at ``250e6 / 3333 = 75007.5008`` Hz: timing lines on the requested
integer drifts by 0.36 s an hour. Recordings now carry the rational —
DataHeader bytes 88..92 / 92..96 (``trig_clock_hz`` / ``trig_period_ticks``,
u32 LE), HDF5 attributes of the same names plus ``trig_frequency_real_hz``.

What these tests pin down:

* the integer ``trig_frequency`` stays the REQUESTED rate;
* at an exact frequency, and on every older file, every time and duration
  is the very number 1.2.1 computed (bit-identical, not approximately);
* at an inexact one, times follow the card: ``anchor + ⌊i · ticks · 10⁹ /
  clock⌋``, the integer the acquisition (Audace-Lib) stamps;
* a rational that does not describe the file's ``trig_frequency`` (more
  than 1 % and more than 1 Hz away) is ignored, as the producer does.

Synthetic files only: DAT bytes are packed here with ``struct`` (not with
the SDK's own writer, which is tested separately), HDF5 files with h5py.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from invisensing import File, Header, export_dat, parse_header
from invisensing._core import build_header_bytes


CLOCK = 250_000_000
I64_MAX = 2**63 - 1
#: 2025-10-09T08:53:20Z — a realistic i64-ns acquisition anchor.
ANCHOR = 1_760_000_000_000_000_000

#: The 27 frequencies of the console's range (1 kHz–200 kHz) a 250 MHz
#: card fires exactly at: the divisors of 250 000 000 = 2⁶·5⁹.
EXACT = [
    1_000, 1_250, 1_600, 2_000, 2_500, 3_125, 3_200, 4_000, 5_000, 6_250,
    8_000, 10_000, 12_500, 15_625, 16_000, 20_000, 25_000, 31_250, 40_000,
    50_000, 62_500, 78_125, 80_000, 100_000, 125_000, 156_250, 200_000,
]

#: Named rows of the system's conformance table (Audace-Lib
#: `trig_timing.rs`, CardControl, the console's prediction): requested Hz
#: → period ticks on the 250 MHz clock.
NAMED = [
    (2_000, 125_000),
    (2_450, 102_040),
    (75_000, 3_333),
    (150_000, 1_666),
    (199_900, 1_250),
    (200_000, 1_250),
]


def _line_counts():
    """Small counts, powers of ten ± 1 up to 10¹³, and a pseudo-random
    spread — the same shape as the producer's own table."""
    v = list(range(2_000))
    p = 1
    while p <= 10**13:
        v += [p - 1, p, p + 1]
        p *= 10
    x = 0x9E3779B97F4A7C15
    for _ in range(2_000):
        x = (x * 6364136223846793005 + 1442695040888963407) % 2**64
        v.append(x % (10**13 + 1))
    return v


LINE_COUNTS = _line_counts()


def _ref_time(anchor: int, i: int, f: int, timing=None) -> int:
    """Reference line time in unbounded Python integers, saturated like the
    producer's i64: ``anchor + ⌊i·ticks·10⁹/clock⌋`` or ``⌊i·10⁹/f⌋``."""
    if timing is not None:
        clock, ticks = timing
        offset = i * ticks * 10**9 // clock
    else:
        offset = i * 10**9 // f
    return min(anchor + min(offset, I64_MAX), I64_MAX)


def _header(f: int, timing=None, anchor: int = ANCHOR) -> Header:
    clock, ticks = timing if timing is not None else (0, 0)
    return Header(
        line_size=4,
        trig_frequency=f,
        sample_size=2,
        sample_rate=100_000_000,
        flags=0,
        acquisition_ns=anchor,
        trig_clock_hz=clock,
        trig_period_ticks=ticks,
    )


# ── DAT synthesis (independent of the SDK's writer) ────────────────────────

ROWS = 64
LINE_SIZE = 8


def _dat_header_bytes(
    trig_frequency: int,
    clock: int = 0,
    ticks: int = 0,
    anchor: int = ANCHOR,
    line_size: int = LINE_SIZE,
) -> bytes:
    """A 128-byte DataHeader as the acquisition writes it: i64 ns anchor in
    0..8, i32 fields from 32, u32 rational at 88 and 92."""
    buf = bytearray(128)
    struct.pack_into("<q", buf, 0, anchor)
    struct.pack_into(
        "<8i", buf, 32,
        line_size,        # line_size
        trig_frequency,   # trig_frequency
        2,                # sample_size (i16)
        100_000_000,      # sample_rate
        0,                # flags
        1_000,            # range (mV)
        40,               # pulse_width
        1,                # num_channels
    )
    buf[64] = 1  # content_type = raw
    struct.pack_into("<I", buf, 84, 0)  # trigger_delay
    struct.pack_into("<II", buf, 88, clock, ticks)
    return bytes(buf)


def _write_dat(path: Path, trig_frequency: int, clock: int = 0, ticks: int = 0,
               anchor: int = ANCHOR, rows: int = ROWS) -> Path:
    payload = np.arange(rows * LINE_SIZE, dtype=np.int16).reshape(rows, LINE_SIZE)
    with open(path, "wb") as fp:
        fp.write(_dat_header_bytes(trig_frequency, clock, ticks, anchor))
        fp.write(payload.tobytes())
    return path


# ── HDF5 synthesis (h5py, attributes typed as the Audace writers type them) ─


def _write_hdf5(path: Path, trig_frequency: int, *, rows: int = ROWS,
                clock=None, ticks=None, real_hz=None, timestamp_ns=None,
                extra_attrs=None) -> Path:
    h5py = pytest.importorskip("h5py")
    payload = np.arange(rows * LINE_SIZE, dtype=np.int16).reshape(rows, LINE_SIZE)
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=payload)
        fp.attrs["line_size"] = np.uint32(LINE_SIZE)
        fp.attrs["trig_frequency"] = np.uint32(trig_frequency)
        fp.attrs["sample_size"] = np.uint32(2)
        fp.attrs["sample_rate"] = np.uint32(100_000_000)
        fp.attrs["flags"] = np.uint32(0)
        fp.attrs["range"] = np.uint32(1_000)
        fp.attrs["pulse_width"] = np.uint32(40)
        fp.attrs["num_channels"] = np.uint32(1)
        fp.attrs["timestamp"] = "2026-10-21 08:00:00.000"
        if timestamp_ns is not None:
            fp.attrs["timestamp_ns"] = np.int64(timestamp_ns)
        if clock is not None:
            fp.attrs["trig_clock_hz"] = np.uint32(clock)
        if ticks is not None:
            fp.attrs["trig_period_ticks"] = np.uint32(ticks)
        if real_hz is not None:
            fp.attrs["trig_frequency_real_hz"] = np.float64(real_hz)
        for k, v in (extra_attrs or {}).items():
            fp.attrs[k] = v
    return path


# ── The guard and the rational, on the native Header ───────────────────────


def test_named_rows_carry_the_card_rational():
    for f, ticks in NAMED:
        assert CLOCK // f == ticks, f
        h = _header(f, (CLOCK, ticks))
        assert h.trig_timing == (CLOCK, ticks)
        assert h.trig_frequency == f, "the integer stays the requested rate"
        assert h.trig_frequency_real_hz == CLOCK / ticks
        assert h.trig_frequency_is_exact == (CLOCK == f * ticks)


def test_the_guard_keeps_a_card_rational_and_drops_another_stream_s():
    r75 = (CLOCK, 3_333)
    assert _header(75_000, r75).trig_timing == r75
    # The PCIe7818 used to write floor(real) instead of the request.
    assert _header(75_007, r75).trig_timing == r75
    # Left behind by a header whose rate was divided or multiplied.
    for f in (750, 37_500, 150_000):
        assert _header(f, r75).trig_timing is None, f
    # The 1 % edge: 101 vs 100 is out, 100.9 is in.
    assert _header(100, (101, 1)).trig_timing is None
    assert _header(100, (1_009, 10)).trig_timing == (1_009, 10)
    # Within 1 Hz is kept even past 1 %: a variance-row rate of 19.5 Hz
    # (1 950 Hz over 100-line rows) whose integer was rounded to 20.
    row = (CLOCK, (CLOCK // 1_950) * 100)
    assert _header(20, row).trig_timing == row
    assert _header(19, row).trig_timing == row
    assert _header(21, row).trig_timing is None
    # Half a rational, a zero term, a non-positive request: unknown.
    assert _header(75_000, (CLOCK, 0)).trig_timing is None
    assert _header(75_000, (0, 3_333)).trig_timing is None
    assert _header(0, r75).trig_timing is None
    assert _header(-75_000, r75).trig_timing is None


def test_a_dropped_rational_falls_back_on_the_requested_integer():
    h = _header(750, (CLOCK, 3_333))
    assert h.trig_clock_hz == CLOCK, "the raw stored value stays readable"
    assert h.trig_frequency_real_hz == 750.0
    assert h.trig_frequency_is_exact is True
    assert h.line_time_ns(750) == ANCHOR + 1_000_000_000


def test_exact_frequencies_are_bit_identical_to_the_nominal_computation():
    """THE PROPERTY THE DESIGN RESTS ON: at an exact frequency the card's
    rational gives the very integer the nominal formula gives — no time of
    an exact recording moves by a nanosecond."""
    for f in EXACT:
        with_rational = _header(f, (CLOCK, CLOCK // f))
        nominal = _header(f)
        assert with_rational.trig_frequency_is_exact
        assert with_rational.trig_frequency_real_hz == float(f)
        for i in LINE_COUNTS:
            want = _ref_time(ANCHOR, i, f)
            assert with_rational.line_time_ns(i) == want, (f, i)
            assert nominal.line_time_ns(i) == want, (f, i)
            assert with_rational.lines_to_ns(i) == nominal.lines_to_ns(i), (f, i)


def test_inexact_frequencies_follow_the_card_rational():
    for f, ticks in NAMED:
        h = _header(f, (CLOCK, ticks))
        for i in LINE_COUNTS:
            assert h.line_time_ns(i) == _ref_time(ANCHOR, i, f, (CLOCK, ticks)), (f, i)
    # Values from the producer's own test table (Audace-Lib trig_timing.rs).
    h75 = _header(75_000, (CLOCK, 3_333), anchor=0)
    assert h75.lines_to_ns(75_000) == 999_900_000
    hour = 75_000 * 3_600
    assert h75.lines_to_ns(hour) == 3_599_640_000_000
    assert _header(75_000, anchor=0).lines_to_ns(hour) == 3_600_000_000_000
    month = 200_000 * 86_400 * 30
    assert _header(199_900, (CLOCK, 1_250), anchor=0).lines_to_ns(month) == (
        30 * 86_400 * 1_000_000_000
    )


def test_line_time_saturates_like_the_producer():
    # The slowest rational the fields can hold, within 1 Hz of 1 Hz.
    slow = _header(1, (1, 2**32 - 1), anchor=0)
    assert slow.trig_timing == (1, 2**32 - 1)
    assert slow.lines_to_ns(2**64 - 1) == I64_MAX
    assert _header(1_000).line_time_ns(2**64 - 1) == I64_MAX
    with pytest.raises(OverflowError):
        _header(1_000).line_time_ns(-1)  # the native method takes a u64


def test_no_cadence_at_all_keeps_the_anchor():
    h = _header(0)
    assert h.trig_frequency_real_hz == 0.0
    assert h.line_time_ns(12_345) == ANCHOR
    assert h.lines_to_ns(12_345) == 0


def test_header_bytes_88_to_96_round_trip():
    h = _header(75_000, (CLOCK, 3_333))
    raw = build_header_bytes(h)
    assert raw[88:96] == struct.pack("<II", CLOCK, 3_333)
    back = parse_header(raw)
    assert (back.trig_clock_hz, back.trig_period_ticks) == (CLOCK, 3_333)
    assert back.trig_timing == (CLOCK, 3_333)
    # And our parser reads what the acquisition packs.
    p = parse_header(_dat_header_bytes(2_450, CLOCK, 102_040))
    assert (p.trig_clock_hz, p.trig_period_ticks) == (CLOCK, 102_040)
    assert p.acquisition_ns == ANCHOR
    # A header without the rational still writes zeros there.
    assert build_header_bytes(_header(2_000))[88:96] == bytes(8)


def test_header_repr_is_unchanged_without_a_rational():
    assert "trig_clock_hz" not in repr(_header(2_000))
    r = repr(_header(75_000, (CLOCK, 3_333)))
    assert "trig_clock_hz=250000000, trig_period_ticks=3333" in r


# ── DAT files through the File facade ──────────────────────────────────────


def test_dat_75_khz_real_timeline(tmp_path):
    path = _write_dat(tmp_path / "a.dat", 75_000, CLOCK, 3_333)
    with File(path) as f:
        assert f.trig_frequency == 75_000 and isinstance(f.trig_frequency, int)
        assert f.trig_clock_hz == CLOCK
        assert f.trig_period_ticks == 3_333
        real = f.trig_frequency_real_hz
        assert isinstance(real, float)
        assert real == CLOCK / 3_333
        assert abs(real - 75_007.500_750_075) < 1e-6
        assert f.trig_frequency_is_exact is False
        assert f.duration == ROWS / real
        assert f.line_time_ns(0) == ANCHOR
        assert f.line_time_ns(75_000) == ANCHOR + 999_900_000
        # One hour of shots: 0.36 s shorter than the nominal count says.
        assert f.line_time_ns(75_000 * 3_600) == ANCHOR + 3_599_640_000_000
        prev = -1
        for i in range(0, 2_000_000, 997):
            t = f.line_time_ns(i)
            assert t == ANCHOR + i * 3_333 * 10**9 // CLOCK
            # Floored, within one nanosecond of i / f_real.
            exact = i * 3_333 * 10**9 / CLOCK
            assert 0 <= exact - (t - ANCHOR) < 1.0 + 1e-6
            assert t > prev
            prev = t
        assert "trig_frequency=75000 Hz (real 75007.5008 Hz)" in repr(f)


def test_dat_exact_frequency_matches_1_2_1(tmp_path):
    rows = 10_000
    path = _write_dat(tmp_path / "x.dat", 2_000, CLOCK, 125_000, rows=rows)
    with File(path) as f:
        assert f.trig_clock_hz == CLOCK and f.trig_period_ticks == 125_000
        assert f.trig_frequency_is_exact is True
        assert f.trig_frequency_real_hz == 2_000.0
        assert f.duration == rows / max(2_000, 1)  # 1.2.1's formula, to the bit
        for i in LINE_COUNTS[::50]:
            assert f.line_time_ns(i) == _ref_time(ANCHOR, i, 2_000)
        assert "(real" not in repr(f)


def test_dat_stale_rational_is_ignored(tmp_path):
    path = _write_dat(tmp_path / "s.dat", 750, CLOCK, 3_333)
    with File(path) as f:
        assert f.header.trig_clock_hz == CLOCK  # stored as is…
        assert f.trig_clock_hz is None          # …but not trusted
        assert f.trig_period_ticks is None
        assert f.trig_frequency_real_hz == 750.0
        assert f.trig_frequency_is_exact is True
        assert f.duration == ROWS / 750
        assert f.line_time_ns(750) == ANCHOR + 1_000_000_000


@pytest.mark.parametrize("trig", [1_000, 2_450, 75_000])
def test_dat_without_rational_behaves_as_1_2_1(tmp_path, trig):
    path = _write_dat(tmp_path / "old.dat", trig)
    with File(path) as f:
        assert f.trig_clock_hz is None and f.trig_period_ticks is None
        assert f.trig_frequency_real_hz == float(trig)
        assert f.trig_frequency_is_exact is True
        assert f.duration == ROWS / max(trig, 1)
        for i in LINE_COUNTS[::50]:
            assert f.line_time_ns(i) == _ref_time(ANCHOR, i, trig)
        assert "(real" not in repr(f)


def test_export_dat_leaves_the_rational_unknown(tmp_path):
    path = tmp_path / "e.dat"
    export_dat(path, np.zeros((5, 4), np.int16), sample_rate=1_000, trig_frequency=1_000,
               timestamp="2026-10-21_legacy")
    assert path.read_bytes()[88:96] == bytes(8)
    with File(path) as f:
        # Legacy ASCII timestamp: no anchor, times are offsets from line 0.
        assert f.header.acquisition_ns == 0
        assert f.line_time_ns(1_500) == 1_500_000_000
        assert f.trig_clock_hz is None


def test_line_time_ns_argument_checks(tmp_path):
    path = _write_dat(tmp_path / "a.dat", 75_000, CLOCK, 3_333)
    with File(path) as f:
        with pytest.raises(ValueError):
            f.line_time_ns(-1)
        with pytest.raises(TypeError):
            f.line_time_ns(1.5)
        assert f.line_time_ns(np.int64(75_000)) == ANCHOR + 999_900_000
        assert f.line_time_ns(np.uint32(75_000)) == ANCHOR + 999_900_000
        assert f.line_time_ns(True) == f.line_time_ns(1)
        # Beyond the recording: the cadence extrapolates.
        assert f.line_time_ns(10 * ROWS) > f.line_time_ns(ROWS)
    # Metadata, not I/O: a closed file still answers.
    assert f.closed
    assert f.line_time_ns(75_000) == ANCHOR + 999_900_000
    assert f.trig_frequency_real_hz == CLOCK / 3_333


# ── HDF5 ───────────────────────────────────────────────────────────────────


def test_hdf5_with_the_real_frequency_attributes(tmp_path):
    path = _write_hdf5(tmp_path / "r.h5", 75_000, clock=CLOCK, ticks=3_333,
                       real_hz=CLOCK / 3_333, timestamp_ns=ANCHOR)
    with File(path) as f:
        assert f.trig_frequency == 75_000
        assert f.trig_clock_hz == CLOCK and f.trig_period_ticks == 3_333
        assert f.trig_frequency_real_hz == CLOCK / 3_333
        assert f.trig_frequency_is_exact is False
        assert f.header.acquisition_ns == ANCHOR
        assert f.line_time_ns(75_000) == ANCHOR + 999_900_000
        assert f.line_time_ns(ROWS - 1) == ANCHOR + (ROWS - 1) * 3_333 * 10**9 // CLOCK
        assert f.duration == ROWS / (CLOCK / 3_333)
        assert "(real 75007.5008 Hz)" in repr(f)


def test_hdf5_exact_frequency_with_attributes_matches_1_2_1(tmp_path):
    path = _write_hdf5(tmp_path / "x.h5", 2_000, clock=CLOCK, ticks=125_000,
                       real_hz=2_000.0, timestamp_ns=ANCHOR)
    with File(path) as f:
        assert f.trig_frequency_is_exact is True
        assert f.trig_frequency_real_hz == 2_000.0
        assert f.duration == ROWS / max(2_000, 1)
        for i in LINE_COUNTS[::50]:
            assert f.line_time_ns(i) == _ref_time(ANCHOR, i, 2_000)


def test_hdf5_without_the_attributes_behaves_as_1_2_1(tmp_path):
    path = _write_hdf5(tmp_path / "old.h5", 2_450)
    with File(path) as f:
        assert f.trig_clock_hz is None and f.trig_period_ticks is None
        assert f.trig_frequency_real_hz == 2_450.0
        assert f.trig_frequency_is_exact is True
        assert f.duration == ROWS / max(2_450, 1)
        assert f.header.acquisition_ns == 0
        assert f.line_time_ns(2_450) == 1_000_000_000
        assert "(real" not in repr(f)


def test_hdf5_anchor_comes_from_timestamp_ns(tmp_path):
    """A 2026 recording made before the real-frequency attributes: the
    anchor attribute alone, nominal cadence."""
    path = _write_hdf5(tmp_path / "a.h5", 2_000, timestamp_ns=ANCHOR)
    with File(path) as f:
        assert f.header.acquisition_ns == ANCHOR
        assert f.timestamp == "2026-10-21 08:00:00.000", "display string unchanged"
        assert f.line_time_ns(2_000) == ANCHOR + 1_000_000_000


def test_hdf5_real_frequency_attribute_alone(tmp_path):
    path = _write_hdf5(tmp_path / "f.h5", 75_000, real_hz=75_007.5)
    with File(path) as f:
        assert f.trig_clock_hz is None
        assert f.trig_frequency_real_hz == 75_007.5
        assert f.trig_frequency_is_exact is False
        assert f.duration == ROWS / 75_007.5
        # No rational, no exact integer timeline: the nominal one.
        assert f.line_time_ns(75_000) == 1_000_000_000


def test_hdf5_stale_attributes_are_ignored(tmp_path):
    path = _write_hdf5(tmp_path / "s.h5", 750, clock=CLOCK, ticks=3_333,
                       real_hz=CLOCK / 3_333)
    with File(path) as f:
        assert f.trig_clock_hz is None
        assert f.trig_frequency_real_hz == 750.0
        assert f.trig_frequency_is_exact is True
        assert f.duration == ROWS / 750
        assert f.line_time_ns(750) == 1_000_000_000


@pytest.mark.parametrize(
    "attrs",
    [
        {"trig_clock_hz": np.int64(-1), "trig_period_ticks": np.int64(3_333)},
        {"trig_clock_hz": np.uint64(2**40), "trig_period_ticks": np.uint32(3_333)},
        {"trig_clock_hz": np.float64(2.5e8), "trig_period_ticks": np.float64(3_333)},
        {"trig_clock_hz": "250000000", "trig_period_ticks": "3333"},
        {"trig_frequency_real_hz": np.float64(np.nan)},
        {"trig_frequency_real_hz": np.float64(-75_007.5)},
        {"trig_frequency_real_hz": "75007.5"},
        {"timestamp_ns": np.int64(-5)},
        {"timestamp_ns": "1760000000000000000"},
    ],
)
def test_hdf5_malformed_attributes_read_as_unknown(tmp_path, attrs):
    """A value the header cannot hold is "unknown", never a refusal to open
    a recording whose samples are fine."""
    path = _write_hdf5(tmp_path / "m.h5", 75_000, extra_attrs=attrs)
    with File(path) as f:
        assert f.trig_clock_hz is None
        assert f.trig_frequency_real_hz == 75_000.0
        assert f.trig_frequency_is_exact is True
        assert f.header.acquisition_ns == 0
        assert f.read_all().shape == (ROWS, LINE_SIZE)


# ── TDMS and SEG-Y: no rational on the wire, nominal everywhere ────────────


def _write_tdms(path: Path, trig: int) -> Path:
    pytest.importorskip("nptdms")
    from nptdms import TdmsWriter, ChannelObject, RootObject

    props = {
        "line_size": LINE_SIZE, "trig_frequency": trig, "sample_size": 2,
        "sample_rate": 100_000_000, "flags": 0, "num_channels": 1,
        "timestamp": "2026-10-21_tdms",
    }
    data = np.arange(ROWS * LINE_SIZE, dtype=np.int16)
    with TdmsWriter(str(path)) as w:
        w.write_segment([RootObject(properties=props),
                         ChannelObject("AudaceGroup", "acoustic_data", data)])
    return path


def _write_segy(path: Path, trig: int) -> Path:
    segyio = pytest.importorskip("segyio")
    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE)
    spec.samples = list(range(LINE_SIZE))
    spec.tracecount = ROWS
    with segyio.create(str(path), spec) as s:
        for i in range(ROWS):
            s.trace[i] = np.full(LINE_SIZE, i, dtype=np.float32)
        s.text[0] = segyio.tools.create_text_header({
            1: "AUDACE DAS DATA",
            4: "SAMPLE RATE: 100000000 Hz",
            5: f"TRIGGER FREQUENCY: {trig} Hz",
            8: f"SAMPLES PER TRACE: {LINE_SIZE}",
            21: "HEADER FLAGS (raw bits): 0x00000002",
        })
    return path


@pytest.mark.parametrize("writer,name", [(_write_tdms, "t.tdms"), (_write_segy, "s.sgy")])
def test_tdms_and_segy_expose_the_same_api_nominally(tmp_path, writer, name):
    path = writer(tmp_path / name, 2_000)
    with File(path) as f:
        assert f.trig_frequency == 2_000
        assert f.trig_clock_hz is None and f.trig_period_ticks is None
        assert f.trig_frequency_real_hz == 2_000.0
        assert f.trig_frequency_is_exact is True
        assert f.duration == ROWS / 2_000
        assert f.line_time_ns(2_000) == 1_000_000_000

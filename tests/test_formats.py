"""
End-to-end tests for the non-DAT format backends.

Each test synthesises a file in the relevant format (using the same
third-party library the production wrapper relies on), then opens it
through the public ``File`` facade and verifies the metadata + channel
extraction round-trip.

Skipped automatically when the optional backend isn't installed.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Mode,
    export_dat,
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


def _zstd_phase_file(h5py, hdf5plugin, path: Path, payload: np.ndarray) -> None:
    """A phase recording stored like the FileWriter's `compression = ZSTD`:
    chunks of whole traces, byte shuffle, then the zstd filter (32015)."""
    rows, positions = payload.shape
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset(
            "acoustic_data",
            data=payload,
            chunks=(min(rows, 4), positions),
            shuffle=True,
            **hdf5plugin.Zstd(clevel=1),
        )
        fp.attrs["line_size"] = positions
        fp.attrs["trig_frequency"] = 2_000
        fp.attrs["sample_size"] = 4
        fp.attrs["sample_rate"] = 125_000_000
        fp.attrs["flags"] = FLAG_DEMODULATED | FLAG_FLOAT | FLAG_PHASE
        fp.attrs["range"] = 0
        fp.attrs["pulse_width"] = 0
        fp.attrs["num_channels"] = 1
        fp.attrs["timestamp"] = "2026-10-12_hdf5_zstd"


def test_hdf5_zstd_compressed_phase_reads_back_bit_for_bit(h5py_or_skip):
    h5py = h5py_or_skip
    hdf5plugin = pytest.importorskip("hdf5plugin")
    rng = np.random.default_rng(7)
    payload = rng.normal(0.0, 1.5, size=(11, 97)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "zstd.hdf5"
        _zstd_phase_file(h5py, hdf5plugin, path, payload)
        with File(path) as f:
            assert f.mode is Mode.PHASE
            assert f.num_lines == 11
            got = f.get_phase(n=11)
            np.testing.assert_array_equal(got.view(np.int32), payload.view(np.int32))


def test_a_zstd_file_without_hdf5plugin_says_what_to_install(h5py_or_skip, monkeypatch):
    h5py = h5py_or_skip
    hdf5plugin = pytest.importorskip("hdf5plugin")
    payload = np.zeros((3, 8), dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "zstd.hdf5"
        _zstd_phase_file(h5py, hdf5plugin, path, payload)
        # `import hdf5plugin` now fails, as on an install without it.
        monkeypatch.setitem(__import__("sys").modules, "hdf5plugin", None)
        with pytest.raises(ImportError, match="pip install hdf5plugin"):
            File(path)


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


# ── seek_lines (additive in 1.1.0) ─────────────────────────────────────────


def test_seek_lines_dat():
    """`seek_lines` on a .dat positions the cursor so the next read matches
    a full read sliced from the same offset, and clamps past EOF."""
    rows, line_size = 300, 16
    data = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "seek.dat"
        export_dat(path, data, sample_rate=1_000_000, trig_frequency=1_000, range_v=1.0)
        with File(path) as f:
            full = f.read_all()
        with File(path) as f:
            f.seek_lines(123)
            assert f.lines_left == rows - 123
            np.testing.assert_array_equal(f.read_all(), full[123:])
        with File(path) as f:
            f.seek_lines(10 * rows)  # clamp
            assert f.lines_left == 0


def test_seek_lines_hdf5(h5py_or_skip):
    h5py = h5py_or_skip
    rows, positions = 50, 8
    arr = np.arange(rows * positions, dtype=np.int16).reshape(rows, positions)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "seek.h5"
        with h5py.File(str(path), "w") as fp:
            fp.create_dataset("acoustic_data", data=arr)
            fp.attrs["line_size"] = positions
            fp.attrs["trig_frequency"] = 1_000
            fp.attrs["sample_size"] = 2
            fp.attrs["sample_rate"] = 1_000_000
            fp.attrs["flags"] = 0
        with File(path) as f:
            f.seek_lines(20)
            assert f.lines_left == rows - 20
            np.testing.assert_array_equal(f.read_all(), arr[20:])


# ── SEG-Y textual header: contract with the producer ──────────────────────
#
# The SEG-Y metadata the SDK needs does not fit the rev1 binary header, so
# the writer spells it out as `KEY: value` lines of the 3200-byte textual
# header and the SDK parses them back. That makes the wording of those lines
# a wire contract between two repositories.
#
# Since data format v1 there is one SEG-Y writer, Audace-Lib's
# (`src/formats/segy.rs`, `textual_header_lines`), and it writes the textual
# header in ASCII; older files came from the Filewriter's own writer (and
# may be EBCDIC, as segyio writes it). So:
#
#   * `test_the_v1_textual_header_is_parsed` reads every field of the v1
#     line list, in ASCII and in EBCDIC;
#   * `test_the_v1_fixture_matches_the_audace_lib_source` re-derives the
#     writer's own format strings and matches every fixture line against
#     them, so renaming a label over there turns this file red here (it
#     skips when Audace-Lib is not checked out beside python-lib);
#   * `test_the_filewriter_dialect_still_reads` keeps the lines of files
#     written before v1 readable, including the C24/C25 notes that contain
#     the words "SAMPLE RATE" and "TRIGGER FREQUENCY" without being fields.


V1_TEXTUAL_HEADER_LINES = [
    "C 1 AUDACE SEG-Y FILE - DISTRIBUTED ACOUSTIC SENSING DATA",
    "C 2 GENERATED BY INVISENSING AUDACE SYSTEM",
    "C 3 TIMESTAMP: 2026-11-03 10:00:00.123+01:00",
    "C 4 SAMPLE RATE: 62500000 Hz",
    "C 5 TRIGGER FREQUENCY: 50 Hz",
    "C 6 PULSE WIDTH: 100",
    "C 7 VOLTAGE RANGE: 1000 mV",
    "C 8 SAMPLES PER TRACE: 6",
    "C 9 NUMBER OF TRACES: 3",
    "C10 BYTES PER SAMPLE: 4",
    "C11 FIBER POSITIONS PER TRACE: 6",
    "C12 FIBER OPTIC DAS DATA FORMAT:",
    "C13 - EACH TRACE IS ONE TRIGGER LINE, ITS SAMPLES RUN ALONG THE FIBER",
    "C14 - SAMPLE SPACING ALONG THE FIBER: 1.6 METERS",
    "C15 - TRACE SPACING IN TIME: 0.02 S",
    "C16 - CONTENT-TYPE: dui_coherent_phase (19)",
    "C17 - QUANTITY: phase UNIT: rad",
    "C18 - FORMAT VERSION: 1",
    "C19 - SENSING RANGE: 8 METERS",
    "C20 - DURATION: 2 S",
    "C21 HEADER FLAGS (raw bits): 0x00000004",
    "C22 TRIGGER DELAY: 120 METRES",
    "C23 TIMESTAMP NS: 1790000000123456789",
    "C24 CARD: p21",
    "C25 NOTE - SAMPLE INTERVAL (DT) IS 0 HERE. THIS IS NOT A CORRUPT FILE.",
    "C26 - NO WHOLE-MICROSECOND PERIOD EXISTS AT THIS SAMPLE RATE (REV1 U16).",
    "C27 - TIME AXIS: ONE TRACE PER TRIGGER LINE, SEE C15 ABOVE.",
]

#: The textual header of a file written by the Filewriter before data format
#: v1 (the values a real acquisition produced).
FILEWRITER_TEXTUAL_HEADER_LINES = [
    "C 1 AUDACE SEG-Y FILE - DISTRIBUTED ACOUSTIC SENSING DATA",
    "C 2 GENERATED BY INVISENSING AUDACE SYSTEM",
    "C 3 TIMESTAMP: 2026-09-08 11:22:33.456",
    "C 4 SAMPLE RATE: 100000000 Hz",
    "C 5 TRIGGER FREQUENCY: 1000 Hz",
    "C 6 PULSE WIDTH: 40",
    "C 7 VOLTAGE RANGE: 1000 mV",
    "C 8 SAMPLES PER TRACE: 65535",
    "C 9 NUMBER OF TRACES: 1000",
    "C10 BYTES PER SAMPLE: 2",
    "C11 FIBER POSITIONS PER TRACE: 32767",
    "C12 FIBER OPTIC DAS DATA FORMAT:",
    "C13 - EACH TRACE REPRESENTS ONE FIBER CHANNEL",
    "C14 - TRACE SPACING: 1.03 METERS",
    "C15 - COORDINATE SYSTEM: ALONG-FIBER DISTANCE",
    "C16 - INTERLEAVED PAIRS PER POSITION: I + Q",
    "C17 DATA ACQUISITION PARAMETERS:",
    "C18 - OPTICAL INTERROGATOR: AUDACE SYSTEM",
    "C19 - SENSING RANGE: 33790 METERS",
    "C20 - SPATIAL RESOLUTION: 1.03 METERS",
    "C21 HEADER FLAGS (raw bits): 0x00000021",
    "C22 TRIGGER DELAY: 250 METRES",
    "C23 NOTE - SAMPLE INTERVAL (DT) IS 0 HERE. THIS IS NOT A CORRUPT FILE.",
    "C24 - NO WHOLE-MICROSECOND PERIOD EXISTS AT THIS SAMPLE RATE (REV1 U16).",
    "C25 - TIME AXIS: ONE TRACE PER 1/TRIGGER FREQUENCY, SEE C5 ABOVE.",
]


def _packed(lines) -> str:
    """3200-char, 40 × 80 textual header, no newlines."""
    return "".join(line.ljust(80)[:80] for line in lines).ljust(3200)[:3200]


def _segy_with_text(path: Path, text: str, encoding: str, data: np.ndarray, fmt_name: str) -> Path:
    """A SEG-Y of `data` (one trace per row) whose textual header is `text`,
    encoded as given (`"ascii"`, or `"cp037"` for EBCDIC)."""
    segyio = pytest.importorskip("segyio")
    spec = segyio.spec()
    spec.format = int(getattr(segyio.SegySampleFormat, fmt_name))
    spec.samples = list(range(data.shape[1]))
    spec.tracecount = data.shape[0]
    with segyio.create(str(path), spec) as f:
        for i in range(data.shape[0]):
            f.trace[i] = data[i]
    raw = bytearray(path.read_bytes())
    raw[:3200] = text.encode(encoding)
    path.write_bytes(bytes(raw))
    return path


@pytest.mark.parametrize("encoding", ["ascii", "cp037"])
def test_the_v1_textual_header_is_parsed(tmp_path, encoding):
    data = np.linspace(-np.pi, np.pi, 18, dtype=np.float32).reshape(3, 6)
    path = _segy_with_text(tmp_path / "phase.sgy", _packed(V1_TEXTUAL_HEADER_LINES),
                           encoding, data, "IEEE_FLOAT_4_BYTE")
    with File(path) as f:
        h = f.header
        assert (h.sample_rate, h.trig_frequency, h.pulse_width, h.range) == (62_500_000, 50, 100, 1000)
        assert f.content_type == 19 and f.content_type_name == "dui_coherent_phase"
        assert f.format_version == 1 and f.duration_s == 2
        assert f.spatial_step_m == 1.6 and f.temporal_step_s == 0.02
        assert f.flags == 0x4 and f.is_ac
        assert f.trigger_delay == 120
        assert h.acquisition_ns == 1_790_000_000_123_456_789
        assert f.timestamp == "2026-11-03 10:00:00.123+01:00"
        assert f.card == "p21"
        # The content type types the samples: f32 phase with no FLOAT flag.
        assert f.mode is Mode.PHASE and f.dtype == np.float32
        np.testing.assert_array_equal(f.get_phase(n=3), data)


def test_the_filewriter_dialect_still_reads():
    from invisensing._formats import _leading_int, _segy_content_type, _segy_text_fields

    fields = _segy_text_fields(_packed(FILEWRITER_TEXTUAL_HEADER_LINES))
    assert _leading_int(fields["SAMPLE RATE"]) == 100_000_000
    assert _leading_int(fields["TRIGGER FREQUENCY"]) == 1_000
    assert _leading_int(fields["PULSE WIDTH"]) == 40
    assert _leading_int(fields["VOLTAGE RANGE"]) == 1_000
    # The rev1 ceiling: read back exactly, not truncated.
    assert _leading_int(fields["SAMPLES PER TRACE"]) == 65_535
    assert _leading_int(fields["TRIGGER DELAY"]) == 250
    assert _leading_int(fields["HEADER FLAGS (RAW BITS)"], base=16) == 0x21
    assert fields["TIMESTAMP"] == "2026-09-08 11:22:33.456"
    # No content type before 2026-06; the older Audace-Lib dialect named it
    # by its tag alone.
    assert "CONTENT-TYPE" not in fields
    assert _segy_content_type("hw_amplitude") == 5
    assert _segy_content_type("dui_phase (17)") == 17
    assert _segy_content_type("something_else") == 0


def test_an_unrecognised_text_encoding_is_read_as_ascii_with_a_warning():
    from invisensing._formats import _decode_segy_text

    with pytest.warns(UserWarning, match="neither ASCII nor EBCDIC"):
        text = _decode_segy_text(b" 4 SAMPLE RATE: 5 Hz".ljust(3200))
    assert "SAMPLE RATE: 5 Hz" in text


# ── The v1 fixture above, checked against the writer that produces it ─────

_AUDACE_LIB_SEGY_SOURCE = (
    Path(__file__).resolve().parent.parent.parent
    / "Audace-Lib"
    / "src"
    / "formats"
    / "segy.rs"
)


def _writer_line_patterns() -> dict:
    """Every ``"C.. …"`` literal of ``textual_header_lines``, as a regex,
    plus its ``note("…")`` lines.

    Rust writes these lines either as plain literals or as ``format!``
    templates; both are string literals in the source. A ``{placeholder}``
    becomes ``.*`` — the values are what the acquisition puts there, the
    *wording around them* is the contract.

    Returns ``{"C 4": [(literal, compiled regex), …]}`` and the notes.
    """
    src = _AUDACE_LIB_SEGY_SOURCE.read_text()
    start = src.index("fn textual_header_lines")
    end = src.index("\nfn ", start + 1)
    body = src[start:end]

    def regex(literal: str):
        marked = re.sub(r"\{[^{}]*\}", "\x00", literal.rstrip())
        return re.compile(re.escape(marked).replace("\x00", ".*") + r"\s*$")

    patterns: dict = {}
    for literal in re.findall(r'"(C[ 0-9][0-9] [^"]*)"', body):
        patterns.setdefault(literal[:3], []).append((literal, regex(literal)))
    notes = [regex(n) for n in re.findall(r'note\("([^"]*)"\)', body)]
    return patterns, notes


@pytest.mark.skipif(
    not _AUDACE_LIB_SEGY_SOURCE.exists(),
    reason="Audace-Lib is not checked out next to python-lib",
)
def test_the_v1_fixture_matches_the_audace_lib_source():
    """Renaming a textual-header label in Audace-Lib must fail here.

    Without this, `V1_TEXTUAL_HEADER_LINES` is a hand-typed string that only
    proves the parser agrees with itself. Rename C4 to "ADC SAMPLE RATE"
    over there and every SEG-Y would read back with sample_rate = 0 —
    File.distance a constant, the whole spatial axis silently wrong — with
    the suite still green.
    """
    patterns, notes = _writer_line_patterns()
    assert patterns, "no C.. literals found — did textual_header_lines move?"

    for line in V1_TEXTUAL_HEADER_LINES:
        prefix = line[:3]
        if prefix in patterns:
            assert any(r.match(line) for _, r in patterns[prefix]), (
                f"fixture line {line!r} no longer matches what Audace-Lib emits "
                f"for {prefix!r}: {[lit for lit, _ in patterns[prefix]]}"
            )
        else:
            # A note line, numbered after the fields.
            assert any(r.match(line[4:]) for r in notes), f"unknown line {line!r}"

    fixture_prefixes = {line[:3] for line in V1_TEXTUAL_HEADER_LINES}
    missing = set(patterns) - fixture_prefixes
    assert not missing, (
        f"Audace-Lib writes textual-header lines the fixture does not cover: "
        f"{sorted(missing)} — add them, they may carry metadata the SDK "
        "should be parsing."
    )


# ── SEG-Y at the rev1 ceiling the writer now refuses to cross ─────────────


def test_segy_reads_the_65535_sample_boundary(tmp_path):
    """Filewriter refuses to write more than 65535 samples per trace
    (`Filewriter/src/formats/segy.rs`: the count lives in u16 fields of the
    rev1 binary and trace headers, so a larger file would declare 65535 and
    write more, desynchronising any reader at the second trace). 65535
    itself is legal and is the value a maximum-line_size acquisition
    produces, so the SDK has to read it back exactly — a truncation to 0 or
    a wrap to 65535 - 65536 here would corrupt `line_size`, and with it
    `positions_per_line`, `shape`, `distance` and every extractor.
    """
    segyio = pytest.importorskip("segyio")

    samples = 65_535
    traces = 3
    path = tmp_path / "ceiling.sgy"
    data = np.arange(traces * samples, dtype=np.float32).reshape(traces, samples) % 1000

    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE)
    spec.samples = list(range(samples))
    spec.tracecount = traces
    with segyio.create(str(path), spec) as f:
        for i in range(traces):
            f.trace[i] = data[i]
        f.text[0] = segyio.tools.create_text_header(
            {
                1: "AUDACE SEG-Y FILE - DISTRIBUTED ACOUSTIC SENSING DATA",
                4: "SAMPLE RATE: 100000000 Hz",
                5: "TRIGGER FREQUENCY: 1000 Hz",
                8: f"SAMPLES PER TRACE: {samples}",
                21: f"HEADER FLAGS (raw bits): 0x{FLAG_FLOAT:08X}",
            }
        )

    with File(path) as f:
        assert f.line_size == samples
        assert f.num_lines == traces
        assert f.shape == (traces, samples)
        assert f.dtype == np.float32
        np.testing.assert_array_equal(f.read_all(), data)
    # And the description survives the close, at the boundary too.
    assert f.line_size == samples


# ── Metadata parity between the backends ──────────────────────────────────
#
# The same capture, saved as DAT and as HDF5, must describe itself the same
# way. It did not: the Rust HDF5 writer never emitted `trigger_delay` or
# `content_type`, and this backend never looked for the second one — so a
# converted file reported no trigger pre-roll and an `undefined` content type,
# with nothing on either side to say the values had been lost rather than
# measured. `trigger_delay` in particular places every event in the file.


def test_hdf5_reports_the_trigger_preroll_and_the_content_tag(h5py_or_skip):
    h5py = h5py_or_skip
    rows, positions = 3, 8
    payload = np.arange(rows * positions, dtype=np.int16).reshape(rows, positions)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tagged.h5"
        with h5py.File(str(path), "w") as fp:
            fp.create_dataset("acoustic_data", data=payload)
            fp.attrs["line_size"] = positions
            fp.attrs["trig_frequency"] = 2_000
            fp.attrs["sample_size"] = 2
            fp.attrs["sample_rate"] = 250_000_000
            fp.attrs["flags"] = 0
            fp.attrs["range"] = 1_000
            fp.attrs["pulse_width"] = 4
            fp.attrs["num_channels"] = 1
            fp.attrs["timestamp"] = "2026-01-01_hdf5"
            fp.attrs["trigger_delay"] = 400
            fp.attrs["content_type"] = 7  # interleaved_hw_arctan_amplitude
            fp.attrs["content_type_name"] = "interleaved_hw_arctan_amplitude"

        with File(path) as f:
            assert f.header.trigger_delay == 400
            assert f.header.content_type == 7
            assert f.header.content_type_name == "interleaved_hw_arctan_amplitude"


def test_a_file_written_before_the_attributes_existed_still_opens(h5py_or_skip):
    """No attribute is not a failure — it is an older file, and it says so."""
    h5py = h5py_or_skip
    payload = np.zeros((2, 4), dtype=np.int16)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.h5"
        with h5py.File(str(path), "w") as fp:
            fp.create_dataset("acoustic_data", data=payload)
            fp.attrs["line_size"] = 4
            fp.attrs["trig_frequency"] = 1_000
            fp.attrs["sample_size"] = 2
            fp.attrs["sample_rate"] = 125_000_000
            fp.attrs["flags"] = 0

        with File(path) as f:
            assert f.header.trigger_delay == 0
            assert f.header.content_type == 0
            assert f.header.content_type_name == "undefined"

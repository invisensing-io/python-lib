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
# Filewriter spells it out in the EBCDIC textual header and the SDK scans
# it back. That makes the exact wording of those lines a wire contract
# between two repositories.
#
# The fixture below is one rendering of that header, with the values a real
# acquisition produces. On its own it would prove nothing about the writer:
# a hand-typed copy stays green while the writer is renamed underneath it,
# which is exactly how a fixture that *claims* to be verbatim becomes a
# fixture that is not. So there are two tests, and they do different jobs:
#
#   * `test_segy_textual_header_scanners_match_the_filewriter_wording`
#     checks the SDK reads the right value out of each line, including the
#     C24/C25 ambiguity the scanners must not fall into;
#   * `test_fixture_matches_the_filewriter_source` re-derives the writer's
#     own format strings from `Filewriter/src/formats/segy.rs` and matches
#     every fixture line against them, so renaming a label over there turns
#     this file red here. It skips when Filewriter is not checked out
#     beside python-lib (an SDK-only install), which is the one case where
#     the cross-check cannot be made.


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
    # Emitted only when no whole-microsecond sample interval exists. C24
    # also contains the words "SAMPLE RATE" and C25 "TRIGGER FREQUENCY",
    # which is exactly the ambiguity the scanners must not fall into.
    "C23 NOTE - SAMPLE INTERVAL (DT) IS 0 HERE. THIS IS NOT A CORRUPT FILE.",
    "C24 - NO WHOLE-MICROSECOND PERIOD EXISTS AT THIS SAMPLE RATE (REV1 U16).",
    "C25 - TIME AXIS: ONE TRACE PER 1/TRIGGER FREQUENCY, SEE C5 ABOVE.",
]


def _filewriter_textual_header() -> str:
    """3200-char, 40 × 80 textual header, no newlines — the shape segyio
    hands back from ``f.text[0]``."""
    packed = "".join(line.ljust(80)[:80] for line in FILEWRITER_TEXTUAL_HEADER_LINES)
    return packed.ljust(3200)[:3200]


def test_segy_textual_header_scanners_match_the_filewriter_wording():
    from invisensing._formats import _scan_segy_text_int, _scan_segy_text_str

    text = _filewriter_textual_header()
    assert _scan_segy_text_int(text, "SAMPLE RATE") == 100_000_000
    assert _scan_segy_text_int(text, "TRIGGER FREQUENCY") == 1_000
    assert _scan_segy_text_int(text, "PULSE WIDTH") == 40
    assert _scan_segy_text_int(text, "VOLTAGE RANGE") == 1_000
    # The rev1 ceiling Filewriter now refuses to exceed: the SDK must read
    # the boundary value back exactly, not truncate it.
    assert _scan_segy_text_int(text, "SAMPLES PER TRACE") == 65_535
    assert _scan_segy_text_int(text, "TRIGGER DELAY") == 250
    assert _scan_segy_text_int(text, "HEADER FLAGS", base=16) == 0x21
    assert _scan_segy_text_str(text, "TIMESTAMP") == "2026-09-08 11:22:33.456"


# ── The fixture above, checked against the writer that produces it ─────────

_FILEWRITER_SEGY_SOURCE = (
    Path(__file__).resolve().parent.parent.parent
    / "Filewriter"
    / "src"
    / "formats"
    / "segy.rs"
)


def _writer_line_patterns() -> dict:
    """Every ``"C.. …"`` literal of ``build_textual_header``, as a regex.

    Rust writes these lines either as plain literals or as ``format!``
    templates; both are string literals in the source. A ``{placeholder}``
    becomes ``.+`` — the values are what the acquisition puts there, the
    *wording around them* is the contract. Only the body of
    ``build_textual_header`` is scanned, so the crate's own unit tests
    below it cannot vouch for themselves.

    Returns ``{"C 4": [compiled regex, …]}`` — a prefix can have several
    variants (C14 and C20 switch precision, C16 is empty when the file is
    not interleaved).
    """
    src = _FILEWRITER_SEGY_SOURCE.read_text()
    start = src.index("fn build_textual_header")
    end = src.index("#[cfg(test)]", start)
    body = src[start:end]

    patterns: dict = {}
    for literal in re.findall(r'"(C[ 0-9][0-9] [^"]*)"', body):
        # Substitute placeholders before escaping so the braces don't get
        # escaped into the pattern.
        marked = re.sub(r"\{[^{}]*\}", "\x00", literal.rstrip())
        regex = re.compile(re.escape(marked).replace("\x00", ".+") + r"\s*$")
        patterns.setdefault(literal[:3], []).append((literal, regex))
    return patterns


@pytest.mark.skipif(
    not _FILEWRITER_SEGY_SOURCE.exists(),
    reason="Filewriter is not checked out next to python-lib",
)
def test_fixture_matches_the_filewriter_source():
    """Renaming a textual-header label in Filewriter must fail here.

    Without this, `FILEWRITER_TEXTUAL_HEADER_LINES` is a hand-typed string
    that only proves the scanners agree with themselves. Rename C4 to
    "ADC SAMPLE RATE" over there and every SEG-Y the site writes would
    read back with sample_rate = 0 — File.distance a constant, the whole
    spatial axis silently wrong — with the suite still green.
    """
    patterns = _writer_line_patterns()
    assert patterns, "no C.. literals found — did build_textual_header move?"

    for line in FILEWRITER_TEXTUAL_HEADER_LINES:
        prefix = line[:3]
        assert prefix in patterns, (
            f"the fixture has a {prefix!r} line that Filewriter no longer "
            f"writes: {line!r}"
        )
        assert any(regex.match(line) for _, regex in patterns[prefix]), (
            f"fixture line {line!r} no longer matches what Filewriter emits "
            f"for {prefix!r}: {[lit for lit, _ in patterns[prefix]]}"
        )

    fixture_prefixes = {line[:3] for line in FILEWRITER_TEXTUAL_HEADER_LINES}
    missing = set(patterns) - fixture_prefixes
    assert not missing, (
        f"Filewriter writes textual-header lines the fixture does not cover: "
        f"{sorted(missing)} — add them, they may carry metadata the SDK "
        "should be scanning."
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

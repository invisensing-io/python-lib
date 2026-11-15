"""
The gauge length a producer applied to a phase: DAT header bytes
120..124, HDF5 attribute and TDMS property ``gauge_length_m``, SEG-Y line
``GAUGE LENGTH`` (Audace-Lib DATA_FORMAT.md §3.4).

``Header.gauge_length_m`` is the value as stored — a DAT's f32 read back as
the f64 the other formats store (``k · 1e8 / sample_rate`` for a whole
``k``) — and ``File.gauge_length_m`` answers it for a phase only.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from gauge_files import GAUGE, SAMPLE_RATE, WRITERS, dat_header, phase_samples, write_dat, write_h5
from invisensing import FLAG_DEMODULATED, FLAG_INTERLEAVED, FLAG_PHASE, File, Header, Mode, export_dat, parse_header
from invisensing._core import HEADER_LAYOUT, build_header_bytes
from invisensing._formats import replace_header

PHASE = 19  # dui_coherent_phase
RAW = 1


def _bytes_120(path) -> bytes:
    return path.read_bytes()[120:124]


def test_the_field_is_bytes_120_to_124_as_f32():
    assert ("gauge_length_m", 120, 4, "f32") in HEADER_LAYOUT


def test_a_dat_gauge_reads_back_as_the_f64_the_other_formats_store():
    h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, gauge=GAUGE))
    # float32(9.6) = 9.6000003814697265625 on disk, 6 positions of 1.6 m.
    assert h.gauge_length_m == 9.6
    assert np.float32(h.gauge_length_m) == np.float32(GAUGE)
    # Not a whole number of positions: the f32 as it is.
    h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, gauge=10.0))
    assert h.gauge_length_m == 10.0
    h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, gauge=2.5))
    assert h.gauge_length_m == float(np.float32(2.5))
    # No sample rate: the f32 widened.
    h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, gauge=GAUGE, sample_rate=0))
    assert h.gauge_length_m == float(np.float32(GAUGE))


@pytest.mark.parametrize("stored", [0.0, -9.6, float("nan"), float("inf"), float("-inf")])
def test_a_dat_without_a_length_there_has_no_gauge(tmp_path, stored):
    path = write_dat(tmp_path / "p.dat", phase_samples(2, 6), content_type=PHASE, gauge=stored)
    with File(path) as f:
        assert f.header.gauge_length_m == 0.0
        assert f.gauge_length_m is None


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_every_format_gives_the_gauge_of_a_phase(tmp_path, suffix):
    pytest.importorskip({".h5": "h5py", ".tdms": "nptdms", ".sgy": "segyio"}.get(suffix, "numpy"))
    data = phase_samples(3, 6)
    path = WRITERS[suffix](tmp_path / f"phase{suffix}", data, PHASE, GAUGE)
    with File(path) as f:
        assert f.mode is Mode.PHASE
        assert f.gauge_length_m == f.header.gauge_length_m == 9.6
        # Metadata: still answered once closed.
    assert f.gauge_length_m == 9.6
    absent = WRITERS[suffix](tmp_path / f"none{suffix}", data, PHASE, None)
    with File(absent) as f:
        assert f.header.gauge_length_m == 0.0
        assert f.gauge_length_m is None


@pytest.mark.parametrize("suffix", [".dat", ".h5"])
def test_a_stray_value_on_a_non_phase_file_is_no_gauge(tmp_path, suffix):
    pytest.importorskip("h5py" if suffix == ".h5" else "numpy")
    raw = np.arange(12, dtype=np.int16).reshape(2, 6)
    path = WRITERS[suffix](tmp_path / f"raw{suffix}", raw, RAW, GAUGE)
    with File(path) as f:
        assert f.mode is Mode.RAW
        assert f.header.gauge_length_m == 9.6  # as stored
        assert f.gauge_length_m is None


def test_a_file_read_from_its_flags_has_no_gauge(tmp_path):
    """Content type 0 (a capture written before the content type existed):
    its phase is read from the flags, and no such capture had a gauge."""
    path = write_dat(
        tmp_path / "legacy.dat", phase_samples(2, 6),
        content_type=0, flags=FLAG_DEMODULATED | FLAG_PHASE, gauge=GAUGE, format_version=0,
    )
    with File(path) as f:
        assert f.mode is Mode.PHASE
        assert f.gauge_length_m is None


def test_pairs_whose_first_member_is_a_phase_carry_the_stream_s_gauge(tmp_path):
    """A producer stamps the gauge on a phase stream and on the pairs that
    carry one (15, 20–22, 31, 32): a DAT of such a stream keeps it."""
    pairs = np.zeros((2, 12), dtype=np.float32)
    path = write_dat(
        tmp_path / "pairs.dat", pairs, content_type=22,
        flags=FLAG_DEMODULATED | FLAG_PHASE | FLAG_INTERLEAVED, gauge=GAUGE, format_version=0,
    )
    with File(path) as f:
        assert f.mode is Mode.PHASE_AMPLITUDE
        assert f.gauge_length_m == 9.6


def test_header_round_trips_the_gauge():
    h = Header(line_size=6, trig_frequency=50, sample_size=4, sample_rate=SAMPLE_RATE, flags=0,
               content_type=PHASE, gauge_length_m=GAUGE)
    assert h.gauge_length_m == 9.6
    buf = bytes(build_header_bytes(h))
    assert buf[120:124] == struct.pack("<f", GAUGE)
    assert parse_header(buf).gauge_length_m == 9.6
    assert "gauge_length_m=9.6" in repr(h)
    assert "gauge_length_m" not in repr(replace_header(h, gauge_length_m=0.0))
    for bad in (float("nan"), -1.0, float("inf")):
        assert replace_header(h, gauge_length_m=bad).gauge_length_m == 0.0


def test_an_export_keeps_the_gauge_of_a_phase_only(tmp_path):
    data = phase_samples(4, 6)
    src = write_dat(tmp_path / "src.dat", data, content_type=PHASE, gauge=GAUGE)
    with File(src) as f:
        f.export_dat(tmp_path / "copy.dat", start=1)
        h = f.header
    assert _bytes_120(tmp_path / "copy.dat") == struct.pack("<f", GAUGE)
    with File(tmp_path / "copy.dat") as g:
        assert g.gauge_length_m == 9.6
        np.testing.assert_array_equal(g.read_all().view(np.uint32), data[1:].view(np.uint32))
    # The amplitude child of a split pair inherits its parent's header: no
    # gauge there, as the acquisition's writers do.
    amplitude = np.ones((2, 6), dtype=np.float32)
    export_dat(tmp_path / "amp.dat", amplitude, header=replace_header(h, content_type=16))
    assert _bytes_120(tmp_path / "amp.dat") == bytes(4)
    # A legacy copy (content type 0) never had one.
    export_dat(tmp_path / "legacy.dat", data, header=replace_header(h, content_type=0))
    assert _bytes_120(tmp_path / "legacy.dat") == bytes(4)


def test_export_takes_a_gauge_for_a_phase_only(tmp_path):
    data = phase_samples(2, 6)
    export_dat(tmp_path / "p.dat", data, sample_rate=SAMPLE_RATE, trig_frequency=50,
               content_type=PHASE, gauge_length_m=GAUGE)
    with File(tmp_path / "p.dat") as f:
        assert f.gauge_length_m == 9.6
    with pytest.raises(ValueError, match="phase content type only"):
        export_dat(tmp_path / "r.dat", np.zeros((2, 6), np.int16), sample_rate=SAMPLE_RATE,
                   trig_frequency=50, content_type=RAW, gauge_length_m=GAUGE)
    with pytest.raises(ValueError, match="phase content type only"):
        export_dat(tmp_path / "r.dat", data, sample_rate=SAMPLE_RATE, trig_frequency=50,
                   gauge_length_m=GAUGE)
    with pytest.raises(ValueError, match="length above zero"):
        export_dat(tmp_path / "p.dat", data, sample_rate=SAMPLE_RATE, trig_frequency=50,
                   content_type=PHASE, gauge_length_m=0.0)


def test_an_hdf5_gauge_that_is_not_a_length_is_absent(tmp_path):
    pytest.importorskip("h5py")
    for i, bad in enumerate([0.0, -2.0, float("nan"), "9.6"]):
        path = write_h5(tmp_path / f"bad{i}.h5", phase_samples(2, 6), content_type=PHASE,
                        extra={"gauge_length_m": bad})
        with File(path) as f:
            assert f.gauge_length_m is None, bad

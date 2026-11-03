"""
Files of data format v1 written by the acquisition's own writers.

``tests/assets/v1/`` holds small recordings written by Audace-Lib (the
library the recorder and the console write with): split pairs
(``phase/`` + ``amplitude/``, ``arctan/`` + ``magnitude/``,
``i_component/`` + ``q_component/``) as DAT, HDF5 (``.h5``), SEG-Y and
TDMS, a raw capture, an inexact trigger cadence, zone-variance archive
exports (27, 28), and legacy files (content type 0; a pre-v1 file carrying
an interleaved value). ``expected.json`` says what the writer put in each
(header fields, the SHA-256 of the samples); ``generate.rs`` is the
generator. A v1 file carries no FLOAT / UNSIGNED / INTERLEAVED flag: this
SDK must type and classify it from the content type alone, and keep every
field through an export (E1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from invisensing import File, Mode, export_dat, content

ASSETS = Path(__file__).resolve().parent / "assets" / "v1"
EXPECTED = ASSETS / "expected.json"

pytestmark = pytest.mark.skipif(not EXPECTED.exists(), reason="tests/assets/v1 not present")


def _entries():
    if not EXPECTED.exists():
        return []
    return json.loads(EXPECTED.read_text())["files"]


def _id(e):
    return e["path"]


def _dtype(e) -> np.dtype:
    enc = e["encoding"]
    if enc["float"]:
        return np.dtype(f"<f{enc['bytes']}")
    return np.dtype(f"<{'u' if enc['unsigned'] else 'i'}{enc['bytes']}")


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()).hexdigest()


def _check_header(f: File, e: dict, *, exported: bool = False) -> None:
    h = f.header
    assert f.content_type == e["content_type"]
    assert f.content_type_name == e["content_type_name"]
    assert f.format_version == e["format_version"]
    assert f.flags == e["flags"]
    assert f.line_size == e["line_size"]
    assert f.sample_size == e["sample_size"]
    assert f.positions_per_line == e["positions_per_line"]
    assert f.num_lines == e["num_lines"]
    assert h.duration_s == e["duration_s"]
    assert h.spatial_step_m == e["spatial_step_m"]
    assert h.temporal_step_s == e["temporal_step_s"]
    assert f.card == e["card"]
    assert h.sample_rate == e["sample_rate"]
    assert h.trig_frequency == e["trig_frequency"]
    assert f.trigger_delay == e["trigger_delay"]
    if e["format"] in ("dat", "h5", "tdms") or exported:
        assert h.acquisition_ns == e["acquisition_ns"]
    if e["format"] in ("dat", "h5") or (e["format"] == "tdms" and e["trig_frequency"] * e["trig_period_ticks"] == e["trig_clock_hz"]):
        assert (h.trig_clock_hz, h.trig_period_ticks) == (e["trig_clock_hz"], e["trig_period_ticks"])
    assert f.dtype == _dtype(e)
    assert f.mode is Mode(e["mode"])


@pytest.mark.parametrize("e", _entries(), ids=_id)
def test_every_file_reads_as_its_writer_wrote_it(e):
    with File(ASSETS / e["path"]) as f:
        _check_header(f, e)
        data = f.read_all()
        assert data.dtype == _dtype(e)
        assert _sha(data) == e["data_sha256"]
        if e.get("compression"):
            assert f.compression == e["compression"]


@pytest.mark.parametrize(
    "e", [e for e in _entries() if e["format_version"] == 1], ids=_id
)
def test_a_v1_file_is_typed_by_its_content_type_alone(e):
    with File(ASSETS / e["path"]) as f:
        # No flag but AC / HIZ in a v1 file …
        assert f.flags & ~0xC == 0
        # … and still the right type, quantity, unit and geometry.
        info = content.info(f.content_type)
        assert f.quantity == info.quantity and f.unit == info.unit
        assert not f.is_interleaved
        assert f.spatial_step_m == e["spatial_step_m"]
        assert f.temporal_step_s == e["temporal_step_s"]
        assert f.duration_s == (e["duration_s"] or None)
        assert f.distance == pytest.approx(e["positions_per_line"] * e["spatial_step_m"])


@pytest.mark.parametrize(
    "e", [e for e in _entries() if "sibling" in e], ids=_id
)
def test_a_split_child_finds_its_sibling(e):
    with File(ASSETS / e["path"]) as f:
        assert f.sibling_path == ASSETS / e["sibling"]
        with File(f.sibling_path) as g:
            assert g.num_lines == f.num_lines and g.card == f.card
            assert g.header.acquisition_ns == f.header.acquisition_ns


def test_the_split_pair_rebuilds_the_stream():
    """phase/ and amplitude/ of one recording are the two members of the
    stream (here of a pre-v1 file that still holds the pair)."""
    with File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat") as p, \
         File(ASSETS / "split_dat/amplitude/2026-11-03_10-00-00.dat") as a, \
         File(ASSETS / "legacy/v0_ct8_phase_amplitude.dat") as pair:
        assert pair.mode is Mode.PHASE_AMPLITUDE
        buf = pair.read_all()
        np.testing.assert_array_equal(pair.get_phase(buf), p.get_phase(n=p.num_lines))
        np.testing.assert_array_equal(pair.get_amplitude(buf), a.get_amplitude(n=a.num_lines))
        assert pair.channels(buf).keys() == {"phase", "amplitude"}


def test_the_card_magnitude_in_volts_and_the_arctan_in_radians():
    with File(ASSETS / "arctan_magnitude_h5/magnitude/2026-11-03_10-00-00.h5") as m:
        mag = m.get_magnitude(n=m.num_lines)
        assert mag.dtype == np.uint16 and mag.max() > 32_767
        m.rewind()
        np.testing.assert_allclose(m.get_magnitude_volts(n=m.num_lines), mag * (1.0 / 32768), rtol=1e-6)
    with File(ASSETS / "arctan_magnitude_dat/arctan/2026-11-03_10-00-00.dat") as a:
        rad = a.get_arctan_radians(n=a.num_lines)
        assert rad.dtype == np.float32 and np.abs(rad).max() <= np.pi


def test_the_archive_exports_give_the_variance():
    with File(ASSETS / "archive/viz_variance.h5") as v, File(ASSETS / "archive/viz_logvar.h5") as lv:
        assert v.mode is Mode.VARIANCE and lv.mode is Mode.LOG_VARIANCE
        assert v.source_content_type == lv.source_content_type == 19
        assert v.unit == "source_unit_squared" and lv.unit == "dimensionless"
        assert v.spatial_step_m == 10.0 and v.temporal_step_s == 0.25
        assert v.distance == 50.0
        variance = v.get_variance(n=v.num_lines)
        np.testing.assert_allclose(lv.get_variance(n=lv.num_lines), variance, rtol=1e-5, atol=1e-6)
        lv.rewind()
        v.rewind()
        np.testing.assert_allclose(v.get_log_variance(n=v.num_lines), lv.get_log_variance(n=lv.num_lines), rtol=1e-6)
    with File(ASSETS / "archive/viz_logvar.dat") as d:
        assert d.source_content_type is None  # only HDF5 carries it


@pytest.mark.parametrize(
    "e", [e for e in _entries() if e["path"] != "inexact/2026-11-03_10-00-00.h5"
          and not e["path"].endswith("_zstd.h5")], ids=_id
)
def test_an_export_keeps_the_whole_header(tmp_path, e):
    """E1: File.export_dat copies every field; a v1 source stays v1, a
    legacy one is a legacy copy; the samples are bit-identical."""
    out = tmp_path / "copy.dat"
    with File(ASSETS / e["path"]) as f:
        f.export_dat(out)
        assert f.lines_left == f.num_lines  # the cursor is put back
        source = f.read_all()
    with File(out) as g:
        _check_header(g, {**e, "format": "dat"} if e["format"] == "dat" else e, exported=True)
        if e["format"] in ("sgy", "tdms") and e["format_version"] == 1:
            # The DAT gets the duration of its lines on the cadence.
            assert g.duration_s == (e["duration_s"] or None)
        np.testing.assert_array_equal(g.read_all(), source)


def test_an_export_of_a_range_moves_the_anchor(tmp_path):
    out = tmp_path / "part.dat"
    with File(ASSETS / "split_h5/phase/2026-11-03_10-00-00.h5") as f:
        f.export_dat(out, start=50, count=60)
        expected = f.line_time_ns(50)
        f.seek_lines(50)
        rows = f.read_lines(60)
    with File(out) as g:
        assert g.num_lines == 60
        assert g.header.acquisition_ns == expected
        assert g.duration_s == 1  # 60 lines at 50 Hz
        np.testing.assert_array_equal(g.read_all(), rows)


def test_an_inexact_cadence_is_not_written_to_dat(tmp_path):
    """A DAT never carries an inexact trigger frequency (the acquisition
    refuses it too): keep such a recording in HDF5."""
    with File(ASSETS / "inexact/2026-11-03_10-00-00.h5") as f:
        assert not f.trig_frequency_is_exact
        assert f.temporal_step_s == 3333 / 250_000_000
        with pytest.raises(ValueError, match="inexact"):
            f.export_dat(tmp_path / "x.dat")
        with pytest.raises(ValueError, match="inexact"):
            export_dat(tmp_path / "y.dat", f.read_lines(2), header=f.header)

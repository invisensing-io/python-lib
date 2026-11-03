"""
What the samples ARE: the header's ``content_type`` (and, beside it, the code
of the acquisition card).

The content type is the reference for every file that carries one; a capture
written before the byte existed (``content_type == 0``, ``"undefined"``)
keeps the flag-based reading it always had. The SDK's registry must be the
producer's: ``tests/assets/content_types_v1.json`` is a byte-identical copy
of the machine-readable registry the acquisition publishes, pinned by its
SHA-256 (the same pin sits in the producer's and the console's tests, so
changing one copy fails the others until all are updated). The copy is kept
out of the source distribution; the tests that need it skip without it.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Header,
    Mode,
    content,
    export_dat,
    parse_header,
    FLAG_AC,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_HIZ,
    FLAG_INTERLEAVED,
    FLAG_MOZART,
    FLAG_PHASE,
    FLAG_RESERVED,
    FLAG_SCHUBERT,
    FLAG_UNSIGNED,
    FORMAT_VERSION,
    HEADER_SIZE,
)
from invisensing._core import (
    CARD_CODES,
    CONTENT_TYPE_NAMES,
    CONTENT_TYPES,
    HEADER_LAYOUT,
    build_header_bytes,
    card_code,
)


def _header(**kw) -> Header:
    base = dict(line_size=4, trig_frequency=1_000, sample_size=2, sample_rate=125_000_000, flags=0)
    base.update(kw)
    return Header(**base)


# ── The registry, against the producer's ───────────────────────────────────

#: The producer's registry, revision 2 (content types 0..28). To follow a
#: new revision: copy the producer's `tests/fixtures/content_types_v1.json`
#: here byte for byte, put its SHA-256 below, and add the new rows to
#: `src/content.rs`.
REGISTRY_FIXTURE = Path(__file__).resolve().parent / "assets" / "content_types_v1.json"
REGISTRY_SHA256 = "f349d176e376bb2ee0e16f1a9eae9420bafcdd3cbefbf092afe081ebb78db074"


@pytest.fixture(scope="module")
def registry() -> dict:
    if not REGISTRY_FIXTURE.exists():
        pytest.skip("tests/assets/content_types_v1.json not present (installing from sdist)")
    return json.loads(REGISTRY_FIXTURE.read_text())


def test_the_registry_copy_is_the_pinned_one(registry):
    assert hashlib.sha256(REGISTRY_FIXTURE.read_bytes()).hexdigest() == REGISTRY_SHA256


#: Columns that are the producer's prose or its own identifiers.
_NOT_COMPARED = {"chain", "notes", "rust_variant"}


def test_every_row_is_the_producer_s(registry):
    """A content type added upstream must be added here too — or every file
    carrying it reads as ``unknown(<n>)`` and is decoded from its flags."""
    theirs = {e["value"]: {k: v for k, v in e.items() if k not in _NOT_COMPARED}
              for e in registry["content_types"]}
    ours = {row["value"]: row for row in CONTENT_TYPES}
    assert sorted(ours) == sorted(theirs)
    for value, row in theirs.items():
        assert ours[value] == row, value
    assert CONTENT_TYPE_NAMES == {v: r["tag"] for v, r in theirs.items()}
    beyond = max(theirs) + 1
    assert _header(content_type=beyond).content_type_name == f"unknown({beyond})"
    assert registry["unrecognized_marker"] == 255
    assert content.info(255) is None


def test_the_helpers_give_the_producer_s_sets(registry):
    sets = registry["sets"]
    values = range(256)
    assert [v for v in values if content.is_phase(v)] == sets["is_phase"]
    assert [v for v in values if content.is_amplitude(v)] == sets["is_amplitude"]
    assert [v for v in values if content.is_stream_only(v)] == sets["stream_only"]
    assert [v for v in values if content.is_legacy(v)] == sets["legacy"]
    assert [v for v in values if content.sample_type(v) == "from_flags"] == sets["sample_type_from_flags"]
    for e in registry["content_types"]:
        v = e["value"]
        assert content.split_children(v) == (tuple(e["split_children"]) if e["split_children"] else None)
        assert content.method_of(v) == e["method"]
        assert content.is_file_value(v) == ("file" in e["file_or_stream"].split("+"))


def test_the_header_layout_is_the_producer_s(registry):
    assert registry["header_size"] == HEADER_SIZE
    assert registry["format_version"] == FORMAT_VERSION
    assert registry["endianness"] == "little"
    theirs = {f["name"]: (f["offset"], f["size"], f["type"]) for f in registry["header"]}
    ours = {name: (offset, size, ty) for name, offset, size, ty in HEADER_LAYOUT}
    assert ours == theirs


def test_the_flag_bits_are_the_producer_s(registry):
    """Same bits; the 0x020 bit is named neutrally here (FLAG_RESERVED)."""
    ours = [FLAG_DEMODULATED, FLAG_FLOAT, FLAG_AC, FLAG_HIZ, FLAG_SCHUBERT,
            FLAG_RESERVED, FLAG_MOZART, FLAG_PHASE, FLAG_INTERLEAVED, FLAG_UNSIGNED]
    assert [f["bit"] for f in registry["flags"]] == ours


@pytest.mark.parametrize("ct", [1, 2, 3, 5, 12, 13, 16, 19, 24, 26, 27, 28])
def test_export_writes_the_producer_s_file_flags(registry, tmp_path, ct):
    """The flag rules of a v1 file: `kept` bits stay, `zero` and
    `deprecated` ones go, `from_flags_only` ones stay only on the values
    typed by their flags (12)."""
    every = sum(f["bit"] for f in registry["flags"])
    dtype = {"i16": np.int16, "u16": np.uint16, "f32": np.float32,
             "from_flags": np.float32}[content.sample_type(ct)]
    path = tmp_path / "f.dat"
    export_dat(path, np.zeros((2, 4), dtype), sample_rate=1_000, trig_frequency=1_000,
               content_type=ct, flags=every)
    with File(path) as f:
        written = f.flags
    for flag in registry["flags"]:
        rule, bit = flag["v1_files"], flag["bit"]
        if rule == "kept" or (rule == "from_flags_only" and ct == 12 and bit == FLAG_FLOAT):
            assert written & bit, flag
        else:
            assert not written & bit, flag


def test_the_light_speed_is_the_producer_s(registry):
    c = registry["light_speed_in_fiber_m_s"]
    for rate in (250_000_000, 500_000_000, 62_500_000):
        h = _header(sample_rate=rate, content_type=1)
        assert h.derived_spatial_step_m == c / 2 / rate


def test_the_card_codes_are_the_producer_s(registry):
    assert list(CARD_CODES) == [c["code"] for c in registry["cards"]]
    for c in registry["cards"]:
        assert card_code(c["code"]) == c["code"]
        # A tag written by a two-card run before data format v1 reads as its
        # code (the tags are never spelled in this SDK's sources).
        for tag in c["legacy_tags"]:
            assert card_code(tag) == c["code"]


@pytest.mark.parametrize("value", [29, 42, 254, 255])
def test_an_unknown_value_is_not_called_undefined(value):
    """``undefined`` means "written before the byte existed": a value from a
    newer producer is something else, and says which number it was."""
    assert _header(content_type=value).content_type_name == f"unknown({value})"


def test_the_v1_names():
    assert CONTENT_TYPE_NAMES[10] == "sw_phase_sac"
    assert CONTENT_TYPE_NAMES[11] == "sw_amplitude_sac"
    assert CONTENT_TYPE_NAMES[19] == "dui_coherent_phase"
    assert CONTENT_TYPE_NAMES[27] == "viz_variance"
    assert CONTENT_TYPE_NAMES[28] == "viz_logvar"
    assert content.from_tag("viz_logvar") == 28
    assert content.member_of(22, "amplitude") == (1, "f32")
    assert content.member_of(7, "amplitude") == (1, "u16")
    assert content.to_physical(3).multiply_by == math.pi / 32768
    assert content.to_physical(28).expr == "10^v - 1"
    assert content.to_physical(27) is None


def test_the_dat_header_carries_the_content_type(tmp_path):
    path = tmp_path / "hw_i.dat"
    h = _header(flags=FLAG_DEMODULATED, content_type=13)
    path.write_bytes(build_header_bytes(h) + np.zeros((2, 4), np.int16).tobytes())
    with File(path) as f:
        assert f.content_type == 13
        assert f.content_type_name == "hw_i"
        assert f.header.content_type_name == "hw_i"
        assert f.mode is Mode.I
        assert f.quantity == "i" and f.unit == "count"


# ── HDF5 / TDMS: the number and the tag beside it ─────────────────────────


def _h5(path: Path, data=None, **attrs) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=np.zeros((2, 4), np.int16) if data is None else data)
        base = {"line_size": 4, "trig_frequency": 1_000, "sample_size": 2,
                "sample_rate": 125_000_000, "flags": FLAG_DEMODULATED}
        for k, v in {**base, **attrs}.items():
            fp.attrs[k] = v
    return path


def test_hdf5_names_a_value_the_sdk_does_not_know_by_its_own_tag(tmp_path):
    path = _h5(tmp_path / "new.h5", content_type=np.uint32(40), content_type_name="sw_future")
    with pytest.warns(UserWarning, match="content type 40 is not known"):
        f = File(path)
    with f:
        assert f.content_type == 40
        assert f.content_type_name == "sw_future"
        # The header alone knows only the number.
        assert f.header.content_type_name == "unknown(40)"
        # Nothing is assumed about what it is: read from the flags.
        assert f.quantity is None and f.mode is Mode.RAW
        assert f.dtype == np.int16


def test_hdf5_tag_stands_in_for_a_missing_number(tmp_path):
    path = _h5(tmp_path / "tag_only.h5", content_type_name="hw_amplitude")
    with File(path) as f:
        assert f.content_type == 5
        assert f.content_type_name == "hw_amplitude"


def test_hdf5_number_wins_over_a_disagreeing_tag(tmp_path):
    path = _h5(tmp_path / "both.h5", data=np.zeros((2, 4), np.float32), sample_size=4,
               flags=FLAG_DEMODULATED | FLAG_FLOAT | FLAG_PHASE,
               content_type=np.uint32(2), content_type_name="hw_q")
    with File(path) as f:
        assert f.content_type == 2
        assert f.content_type_name == "hw_phase"


def test_a_sample_size_the_content_type_contradicts_reads_from_the_flags(tmp_path):
    """The retired software PHASE method wrote f32 pairs under value 7
    (whose members are i16 + u16): read from its flags, with a warning."""
    pairs = np.zeros((2, 8), np.float32)
    path = tmp_path / "phase_method.dat"
    h = _header(line_size=8, sample_size=4, content_type=7,
                flags=FLAG_DEMODULATED | FLAG_FLOAT | FLAG_INTERLEAVED)
    path.write_bytes(build_header_bytes(h) + pairs.tobytes())
    with pytest.warns(UserWarning, match="carries 2-byte samples, but the header says 4"):
        f = File(path)
    with f:
        assert f.dtype == np.float32
        assert f.positions_per_line == 4
        assert f.mode is Mode.IQ  # what its flags say, as before
        assert f.quantity is None


def test_tdms_names_a_value_the_sdk_does_not_know_by_its_own_tag(tmp_path):
    pytest.importorskip("nptdms")
    from nptdms import TdmsWriter, ChannelObject, RootObject

    props = {"line_size": 4, "trig_frequency": 1_000, "sample_size": 2,
             "sample_rate": 125_000_000, "flags": FLAG_DEMODULATED,
             "content_type": 41, "content_type_name": "sw_future"}
    path = tmp_path / "new.tdms"
    with TdmsWriter(str(path)) as w:
        w.write_segment([RootObject(properties=props),
                         ChannelObject("AudaceGroup", "acoustic_data", np.zeros(8, np.int16))])
    with pytest.warns(UserWarning, match="content type 41"):
        f = File(path)
    with f:
        assert f.content_type == 41
        assert f.content_type_name == "sw_future"


# ── is_mozart: the content type decides; legacy files keep the flags ──────

_DEMOD_PHASE = FLAG_DEMODULATED | FLAG_FLOAT | FLAG_PHASE


@pytest.mark.parametrize(
    "flags,content_type,expected",
    [
        # Legacy captures (0): exactly the flag-based answer of 1.2.x.
        (0, 0, None),
        (FLAG_DEMODULATED, 0, False),
        (_DEMOD_PHASE | FLAG_MOZART, 0, True),
        (FLAG_MOZART, 0, None),
        # A content type answers alone, whatever the MOZART bit says.
        (_DEMOD_PHASE | FLAG_MOZART, 9, True),
        (_DEMOD_PHASE, 9, True),
        (0, 23, True),
        (_DEMOD_PHASE | FLAG_MOZART, 4, False),
        (_DEMOD_PHASE, 2, False),
        (0, 19, False),
        (FLAG_DEMODULATED, 13, False),
        (0, 12, False),
        (FLAG_MOZART, 1, None),
        (0, 1, None),
        # A value this SDK does not know: no answer.
        (_DEMOD_PHASE | FLAG_MOZART, 200, None),
    ],
)
def test_is_mozart(tmp_path, flags, content_type, expected):
    size = 4 if content_type in (2, 4, 9, 19, 23) else 2
    h = _header(flags=flags, content_type=content_type, sample_size=size)
    assert h.is_mozart is expected
    path = tmp_path / "m.dat"
    path.write_bytes(build_header_bytes(h) + bytes(4 * size))
    with warnings.catch_warnings():
        # Only the unknown value (200) warns.
        warnings.simplefilter("ignore" if content_type == 200 else "error")
        with File(path) as f:
            assert f.is_mozart is expected


# ── The flag-described predicates follow the content type ─────────────────


@pytest.mark.parametrize(
    "ct,size,mode,dtype,float_,unsigned,phase,demod",
    [
        (1, 2, Mode.RAW, np.int16, False, False, False, False),
        (2, 4, Mode.PHASE, np.float32, True, False, True, True),
        (3, 2, Mode.ARCTAN, np.int16, False, False, False, True),
        (5, 2, Mode.AMPLITUDE, np.uint16, False, True, False, True),
        (11, 4, Mode.AMPLITUDE, np.float32, True, False, False, True),
        (13, 2, Mode.I, np.int16, False, False, False, True),
        (14, 2, Mode.Q, np.int16, False, False, False, True),
        (16, 4, Mode.AMPLITUDE, np.float32, True, False, False, True),
        (19, 4, Mode.PHASE, np.float32, True, False, True, True),
        (23, 4, Mode.PHASE, np.float32, True, False, True, True),
        (24, 2, Mode.CUSTOM, np.int16, False, False, False, False),
        (25, 2, Mode.CUSTOM, np.uint16, False, True, False, False),
        (26, 4, Mode.CUSTOM, np.float32, True, False, False, False),
        (27, 4, Mode.VARIANCE, np.float32, True, False, False, False),
        (28, 4, Mode.LOG_VARIANCE, np.float32, True, False, False, False),
    ],
)
def test_a_v1_file_is_described_by_its_content_type_alone(
    tmp_path, ct, size, mode, dtype, float_, unsigned, phase, demod
):
    """A file of data format v1 writes FLOAT / UNSIGNED / PHASE /
    DEMODULATED / INTERLEAVED as 0: the content type says it all."""
    h = _header(content_type=ct, sample_size=size, format_version=1, flags=FLAG_AC)
    path = tmp_path / "v1.dat"
    path.write_bytes(build_header_bytes(h) + np.zeros((3, 4), dtype).tobytes())
    with File(path) as f:
        assert f.mode is mode
        assert f.dtype == dtype and f.read_all().dtype == dtype
        assert bool(f.is_float) is float_
        assert bool(f.is_unsigned) is unsigned
        assert bool(f.is_phase) is phase
        assert bool(f.is_demodulated) is demod
        assert not f.is_interleaved
        assert f.positions_per_line == 4
        assert f.is_ac


def test_a_v0_f32_hw_amplitude_is_the_f32_magnitude():
    """Before v1 the demodulation labelled its f32 magnitude 5: read as 16."""
    h = _header(content_type=5, sample_size=4, flags=FLAG_DEMODULATED | FLAG_FLOAT)
    assert h.normalized_content_type == 16
    assert h.sample_encoding == (4, True, False)
    assert not h.sample_size_contradicts_content_type
    assert Mode.from_header(h) is Mode.AMPLITUDE
    v1 = _header(content_type=5, sample_size=4, format_version=1)
    assert v1.normalized_content_type == 5 and v1.sample_size_contradicts_content_type


# ── The card code (bytes 65..81, every run since data format v1) ──────────


def _raw_header(card_bytes: bytes) -> bytes:
    buf = bytearray(build_header_bytes(_header(content_type=1, trigger_delay=250)))
    buf[65:65 + len(card_bytes)] = card_bytes
    return bytes(buf)


@pytest.mark.parametrize("code", ["p18", "p21", "s125", "s500"])
def test_the_card_code_is_read(tmp_path, code):
    path = tmp_path / "card.dat"
    path.write_bytes(_raw_header(code.encode()) + np.zeros((1, 4), np.int16).tobytes())
    with File(path) as f:
        assert f.card == code
        assert f.header.card == code
        # The fields around the slot are untouched by it.
        assert f.content_type == 1 and f.trigger_delay == 250
        assert f"card='{code}'" in repr(f.header)


def test_a_tag_written_before_v1_reads_as_its_code(registry, tmp_path):
    for c in registry["cards"]:
        for tag in c["legacy_tags"]:
            path = tmp_path / f"{c['code']}.dat"
            path.write_bytes(_raw_header(tag.encode()) + np.zeros((1, 4), np.int16).tobytes())
            with File(path) as f:
                assert f.card == c["code"]


@pytest.mark.parametrize(
    "slot",
    [
        b"",                    # an older single-card capture: zero region
        b"card_500mhz",         # neither a code nor a tag written before v1
        b"0123456789abcdef",    # 16 bytes, no NUL
        b"p18/x",
        b"P18",
        b"\xffjunk",            # not ASCII: another producer's bytes
    ],
)
def test_anything_else_in_the_slot_is_no_card(slot):
    assert parse_header(_raw_header(slot)).card is None


def test_the_card_code_round_trips_and_older_headers_keep_their_repr(registry):
    h = _header(card="p21")
    assert parse_header(build_header_bytes(h)).card == "p21"
    # A tag written before v1 given to the constructor is stored as its code.
    tag = next(t for c in registry["cards"] for t in c["legacy_tags"])
    assert _header(card=tag).card in CARD_CODES
    assert _header().card is None
    assert _header(card="").card is None
    assert "card" not in repr(_header())
    assert parse_header(build_header_bytes(_header())).card is None
    with pytest.raises(ValueError, match="card must be one of"):
        _header(card="card-7")


def test_hdf5_carries_the_card_code(tmp_path):
    with File(_h5(tmp_path / "c.h5", card="s500")) as f:
        assert f.card == "s500"
    with File(_h5(tmp_path / "none.h5", card="")) as f:
        assert f.card is None
    with File(_h5(tmp_path / "older.h5")) as f:
        assert f.card is None


# ── The data format v1 fields ─────────────────────────────────────────────


def test_the_v1_fields_round_trip_through_the_dat_header():
    h = _header(content_type=19, sample_size=4, sample_rate=62_500_000,
                format_version=1, duration_s=7, spatial_step_m=1.6, temporal_step_s=0.001)
    raw = build_header_bytes(h)
    assert raw[108] == 1 and int.from_bytes(raw[104:108], "little") == 7
    assert np.frombuffer(raw[112:120], "<f4").tolist() == [np.float32(1.6), np.float32(0.001)]
    back = parse_header(raw)
    assert back.format_version == 1
    assert back.duration_s == 7
    # Stored as f32, read back as the exact value it stands for (the
    # derived one), as HDF5 stores it.
    assert back.spatial_step_m == 1.6
    assert back.temporal_step_s == 0.001


def test_a_step_that_is_not_the_derived_one_is_kept():
    """An archive export's zone spacing is its own (not 1e8 / sample_rate)."""
    h = _header(content_type=27, sample_size=4, format_version=1,
                spatial_step_m=10.0, temporal_step_s=0.25)
    back = parse_header(build_header_bytes(h))
    assert back.spatial_step_m == 10.0 and back.temporal_step_s == 0.25
    assert back.derived_spatial_step_m == 0.0  # zone variances: not derivable


def test_an_older_header_has_no_v1_fields():
    h = parse_header(build_header_bytes(_header()))
    assert (h.format_version, h.duration_s, h.spatial_step_m, h.temporal_step_s) == (0, 0, 0.0, 0.0)
    assert "format_version" not in repr(h) and "duration_s" not in repr(h)


def test_export_dat_writes_a_v1_file_from_a_content_type(tmp_path):
    path = tmp_path / "phase.dat"
    data = np.linspace(-1, 1, 3000 * 4, dtype=np.float32).reshape(3000, 4)
    export_dat(path, data, sample_rate=250_000_000, trig_frequency=1_000,
               content_type=17, card="p21", flags=FLAG_AC | FLAG_DEMODULATED | FLAG_FLOAT)
    with File(path) as f:
        assert f.format_version == 1
        assert f.content_type_name == "dui_phase"
        assert f.flags == FLAG_AC  # only what the content type does not say
        assert f.duration_s == 3
        assert f.spatial_step_m == 0.4 and f.temporal_step_s == 0.001
        assert f.card == "p21"
        np.testing.assert_array_equal(f.read_all(), data)
    with pytest.raises(ValueError, match="stored as float32"):
        export_dat(path, data.astype(np.int16), sample_rate=1, trig_frequency=1, content_type=17)
    with pytest.raises(ValueError, match="sample_rate and trig_frequency"):
        export_dat(path, data)


def test_a_file_of_content_type_0_is_exported_as_a_legacy_copy(tmp_path):
    """export_dat of a legacy header keeps its flags as they are."""
    path = tmp_path / "legacy.dat"
    export_dat(path, np.zeros((2, 4), np.int16), sample_rate=1_000, trig_frequency=1_000,
               flags=FLAG_DEMODULATED | FLAG_HIZ | FLAG_INTERLEAVED | FLAG_UNSIGNED)
    with File(path) as f:
        assert f.format_version == 0 and f.content_type == 0
        assert f.flags == FLAG_DEMODULATED | FLAG_HIZ | FLAG_INTERLEAVED | FLAG_UNSIGNED
        assert f.mode is Mode.ARCTAN_MAGNITUDE
        assert f.duration_s is None

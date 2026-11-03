"""
What the samples ARE: the header's ``content_type`` (and, beside it, the
acquisition card's tag).

The content type is the reference for every new file; a capture written
before the byte existed (``content_type == 0``, ``"undefined"``) keeps the
flag-based reading it always had. The SDK's table must name every value the
producer writes — ``test_the_table_is_the_producer_s`` reads the producer's
own enum (Audace-Lib ``src/types.rs``) when that tree sits next to this one.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Header,
    parse_header,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_MOZART,
    FLAG_PHASE,
)
from invisensing._core import CONTENT_TYPE_NAMES, build_header_bytes


def _header(**kw) -> Header:
    base = dict(line_size=4, trig_frequency=1_000, sample_size=2, sample_rate=125_000_000, flags=0)
    base.update(kw)
    return Header(**base)


# ── The table, against the producer's enum ─────────────────────────────────

_AUDACE_LIB_TYPES = Path(__file__).resolve().parent.parent.parent / "Audace-Lib" / "src" / "types.rs"


def _producer_content_types() -> dict:
    """``{value: tag}`` of Audace-Lib's ``ContentType``: the enum's
    ``Variant = n`` lines and ``as_tag``'s ``Self::Variant => "tag"`` arms."""
    src = _AUDACE_LIB_TYPES.read_text()
    enum = src[src.index("pub enum ContentType"):]
    enum = enum[: enum.index("\n}")]
    values = {name: int(v) for name, v in re.findall(r"^\s*(\w+)\s*=\s*(\d+)\s*,", enum, re.M)}
    impl = src[src.index("impl ContentType"):]
    as_tag = impl[impl.index("pub fn as_tag(self)"):]
    as_tag = as_tag[: as_tag.index("\n  }")]
    tags = dict(re.findall(r'Self::(\w+)\s*=>\s*"([a-z0-9_]+)"', as_tag))
    assert values and set(values) == set(tags), "ContentType / as_tag moved? fix the parser"
    return {values[name]: tags[name] for name in values}


@pytest.mark.skipif(
    not _AUDACE_LIB_TYPES.exists(),
    reason="Audace-Lib is not checked out next to python-lib",
)
def test_the_table_is_the_producer_s():
    """A content type added to Audace-Lib must be named here too — or every
    file carrying it reads as ``unknown(<n>)``."""
    producer = _producer_content_types()
    assert CONTENT_TYPE_NAMES == producer
    for value, tag in producer.items():
        assert _header(content_type=value).content_type_name == tag
    beyond = max(producer) + 1
    assert _header(content_type=beyond).content_type_name == f"unknown({beyond})"


def test_every_value_the_producers_write_today_is_named():
    expected = {
        0: "undefined", 1: "raw", 2: "hw_phase", 3: "hw_arctan", 4: "hw_arctan_dui",
        5: "hw_amplitude", 6: "interleaved_hw_iq", 7: "interleaved_hw_arctan_amplitude",
        8: "interleaved_hw_phase_amplitude", 9: "sw_mozart", 10: "sw_phase",
        11: "sw_amplitude", 12: "sw_custom", 13: "hw_i", 14: "hw_q",
    }
    assert CONTENT_TYPE_NAMES == expected


@pytest.mark.parametrize("value", [15, 42, 255])
def test_an_unknown_value_is_not_called_undefined(value):
    """``undefined`` means "written before the byte existed": a value from a
    newer producer is something else, and says which number it was."""
    assert _header(content_type=value).content_type_name == f"unknown({value})"


def test_the_dat_header_carries_the_content_type(tmp_path):
    path = tmp_path / "hw_i.dat"
    h = _header(flags=FLAG_DEMODULATED, content_type=13)
    path.write_bytes(build_header_bytes(h) + np.zeros((2, 4), np.int16).tobytes())
    with File(path) as f:
        assert f.content_type == 13
        assert f.content_type_name == "hw_i"
        assert f.header.content_type_name == "hw_i"


# ── HDF5 / TDMS: the number and the tag beside it ─────────────────────────


def _h5(path: Path, **attrs) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=np.zeros((2, 4), np.int16))
        base = {"line_size": 4, "trig_frequency": 1_000, "sample_size": 2,
                "sample_rate": 125_000_000, "flags": FLAG_DEMODULATED}
        for k, v in {**base, **attrs}.items():
            fp.attrs[k] = v
    return path


def test_hdf5_names_a_value_the_sdk_does_not_know_by_its_own_tag(tmp_path):
    path = _h5(tmp_path / "new.h5", content_type=np.uint32(40), content_type_name="sw_future")
    with File(path) as f:
        assert f.content_type == 40
        assert f.content_type_name == "sw_future"
        # The header alone knows only the number.
        assert f.header.content_type_name == "unknown(40)"


def test_hdf5_tag_stands_in_for_a_missing_number(tmp_path):
    path = _h5(tmp_path / "tag_only.h5", content_type_name="hw_amplitude")
    with File(path) as f:
        assert f.content_type == 5
        assert f.content_type_name == "hw_amplitude"


def test_hdf5_number_wins_over_a_disagreeing_tag(tmp_path):
    path = _h5(tmp_path / "both.h5", content_type=np.uint32(2), content_type_name="hw_q")
    with File(path) as f:
        assert f.content_type == 2
        assert f.content_type_name == "hw_phase"


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
    with File(path) as f:
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
        (_DEMOD_PHASE | FLAG_MOZART, 4, False),
        (_DEMOD_PHASE, 2, False),
        (FLAG_DEMODULATED, 13, False),
        (0, 12, False),
        (FLAG_MOZART, 1, None),
        (0, 1, None),
        # A value this SDK does not know: no answer.
        (_DEMOD_PHASE | FLAG_MOZART, 200, None),
    ],
)
def test_is_mozart(tmp_path, flags, content_type, expected):
    h = _header(flags=flags, content_type=content_type)
    assert h.is_mozart is expected
    path = tmp_path / "m.dat"
    path.write_bytes(build_header_bytes(h) + np.zeros((1, 4), np.int16).tobytes())
    with File(path) as f:
        assert f.is_mozart is expected


# ── The card tag (bytes 65..81, dual-card runs) ───────────────────────────


def _raw_header(card_bytes: bytes) -> bytes:
    buf = bytearray(build_header_bytes(_header(content_type=1, trigger_delay=250)))
    buf[65:65 + len(card_bytes)] = card_bytes
    return bytes(buf)


def test_the_card_tag_is_read(tmp_path):
    path = tmp_path / "card.dat"
    path.write_bytes(_raw_header(b"pcie7818") + np.zeros((1, 4), np.int16).tobytes())
    with File(path) as f:
        assert f.card == "pcie7818"
        assert f.header.card == "pcie7818"
        # The fields around the slot are untouched by it.
        assert f.content_type == 1 and f.trigger_delay == 250
        assert "card='pcie7818'" in repr(f.header)


@pytest.mark.parametrize(
    "slot,expected",
    [
        (b"", None),                          # single card, legacy zero region
        (b"spectrum_500mhz", "spectrum_500mhz"),
        (b"0123456789abcdef", "0123456789abcdef"),  # 16 bytes, no NUL
        (b"pcie/7818", None),                 # a path separator: not a tag
        (b"\xffjunk", None),                  # not ASCII: another producer's bytes
    ],
)
def test_the_card_tag_follows_audace_lib_s_rule(slot, expected):
    assert parse_header(_raw_header(slot)).card == expected


def test_the_card_tag_round_trips_and_older_headers_keep_their_repr():
    h = _header(card="pcie7821")
    assert parse_header(build_header_bytes(h)).card == "pcie7821"
    assert _header().card is None
    assert "card" not in repr(_header())
    assert parse_header(build_header_bytes(_header())).card is None


def test_hdf5_carries_no_card_tag(tmp_path):
    with File(_h5(tmp_path / "c.h5")) as f:
        assert f.card is None

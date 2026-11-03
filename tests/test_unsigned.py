"""
A single-column ``UNSIGNED`` file reads as unsigned.

When the FileWriter splits an interleaved arctan/magnitude stream, the
``magnitude/`` child keeps the ``UNSIGNED`` bit (Audace-Lib
``header::interleaved_split``: ``DEMODULATED | UNSIGNED``, content type
``hw_amplitude``) and carries ``√(I²+Q²)`` as ``u16``. Read as ``int16``,
every magnitude above 32767 came back negative (40000 → −25536). Every
format must give the same ``uint16`` numbers: the DAT reader, an HDF5
written with an ``int16`` dataset and ``unsigned = 1`` (every file so far)
or with a ``uint16`` dataset (newer writers), TDMS and SEG-Y.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Header,
    Mode,
    FLAG_DEMODULATED,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
)
from invisensing._core import build_header_bytes

#: Magnitudes on both sides of 32767 — the values int16 got wrong.
MAGNITUDES = np.array(
    [[40_000, 65_535, 32_768, 0, 32_767, 1],
     [65_535, 40_000, 10, 32_768, 50_000, 2]],
    dtype=np.uint16,
)
ROWS, COLS = MAGNITUDES.shape
FLAGS = FLAG_DEMODULATED | FLAG_UNSIGNED
HW_AMPLITUDE = 5


def _header(flags=FLAGS, line_size=COLS) -> Header:
    return Header(
        line_size=line_size, trig_frequency=1_000, sample_size=2,
        sample_rate=125_000_000, flags=flags, range=1_000, content_type=HW_AMPLITUDE,
    )


def _dat(path: Path) -> Path:
    path.write_bytes(build_header_bytes(_header()) + MAGNITUDES.tobytes())
    return path


def _h5(path: Path, dataset_dtype) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as fp:
        # Same bits, stored as the writer types them.
        fp.create_dataset("acoustic_data", data=MAGNITUDES.view(dataset_dtype))
        for k, v in {"line_size": COLS, "trig_frequency": 1_000, "sample_size": 2,
                     "sample_rate": 125_000_000, "flags": FLAGS, "range": 1_000,
                     "content_type": HW_AMPLITUDE, "content_type_name": "hw_amplitude",
                     "interleaved": 0, "unsigned": 1}.items():
            fp.attrs[k] = np.uint32(v) if not isinstance(v, str) else v
    return path


def _tdms(path: Path) -> Path:
    pytest.importorskip("nptdms")
    from nptdms import TdmsWriter, ChannelObject, RootObject

    props = {"line_size": COLS, "trig_frequency": 1_000, "sample_size": 2,
             "sample_rate": 125_000_000, "flags": FLAGS, "range": 1_000,
             "content_type": HW_AMPLITUDE}
    # Audace-Lib's TDMS writer stores every 2-byte file as an i16 channel.
    with TdmsWriter(str(path)) as w:
        w.write_segment([RootObject(properties=props),
                         ChannelObject("AudaceGroup", "acoustic_data",
                                       MAGNITUDES.view(np.int16).ravel())])
    return path


def _segy(path: Path) -> Path:
    segyio = pytest.importorskip("segyio")
    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.SIGNED_SHORT_2_BYTE)
    spec.samples = list(range(COLS))
    spec.tracecount = ROWS
    with segyio.create(str(path), spec) as s:
        for i in range(ROWS):
            s.trace[i] = MAGNITUDES[i].view(np.int16)
        s.text[0] = segyio.tools.create_text_header({
            4: "SAMPLE RATE: 125000000 Hz",
            5: "TRIGGER FREQUENCY: 1000 Hz",
            8: f"SAMPLES PER TRACE: {COLS}",
            21: f"HEADER FLAGS (raw bits): 0x{FLAGS:08X}",
        })
    return path


WRITERS = {
    "dat": lambda d: _dat(d / "magnitude.dat"),
    "hdf5-int16": lambda d: _h5(d / "magnitude_i16.hdf5", np.int16),
    "hdf5-uint16": lambda d: _h5(d / "magnitude_u16.hdf5", np.uint16),
    "tdms": lambda d: _tdms(d / "magnitude.tdms"),
    "segy": lambda d: _segy(d / "magnitude.sgy"),
}


@pytest.mark.parametrize("fmt", sorted(WRITERS))
def test_a_split_magnitude_child_reads_as_uint16(tmp_path, fmt):
    path = WRITERS[fmt](tmp_path)
    with File(path) as f:
        assert f.mode is Mode.RAW
        assert f.is_unsigned and not f.is_interleaved
        assert f.dtype == np.uint16
        data = f.read_all()
        assert data.dtype == np.uint16
        np.testing.assert_array_equal(data, MAGNITUDES)
        assert data[0, 0] == 40_000 and data[0, 1] == 65_535 and data[0, 2] == 32_768
        f.rewind()
        assert next(iter(f)).dtype == np.uint16
        # The empty read of an exhausted file has the same dtype.
        f.read_all()
        assert f.read_all().dtype == np.uint16


def test_the_dat_reader_returns_uint16_without_the_facade(tmp_path):
    from invisensing import DatReader

    r = DatReader(str(_dat(tmp_path / "m.dat")))
    block = r.read_lines(ROWS)
    assert block.dtype == np.uint16
    np.testing.assert_array_equal(block, MAGNITUDES)
    r.close()


def test_an_interleaved_arctan_magnitude_file_is_unchanged(tmp_path):
    """INTERLEAVED | UNSIGNED: the pair stays int16 (the arctan lane is
    signed); only the magnitude lane is unsigned, through get_magnitude."""
    pair = np.empty((ROWS, 2 * COLS), np.int16)
    arctan = np.array([[-32768, 32767, -1, 0, 5, -5]] * ROWS, np.int16)
    pair[:, 0::2] = arctan
    pair[:, 1::2] = MAGNITUDES.view(np.int16)
    path = tmp_path / "pair.dat"
    h = _header(flags=FLAGS | FLAG_INTERLEAVED, line_size=2 * COLS)
    path.write_bytes(build_header_bytes(h) + pair.tobytes())
    with File(path) as f:
        assert f.mode is Mode.ARCTAN_MAGNITUDE
        assert f.dtype == np.int16
        buf = f.read_all()
        assert buf.dtype == np.int16
        np.testing.assert_array_equal(f.get_arctan(buf), arctan)
        np.testing.assert_array_equal(f.get_magnitude(buf), MAGNITUDES)


def test_an_interleaved_hdf5_stored_as_uint16_still_deinterleaves(tmp_path):
    """Should a writer store the pair as u16, the kernels still get int16."""
    h5py = pytest.importorskip("h5py")
    pair = np.empty((ROWS, 2 * COLS), np.uint16)
    pair[:, 0::2] = 7
    pair[:, 1::2] = MAGNITUDES
    path = tmp_path / "pair_u16.hdf5"
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=pair)
        for k, v in {"line_size": 2 * COLS, "trig_frequency": 1_000, "sample_size": 2,
                     "sample_rate": 125_000_000, "flags": FLAGS | FLAG_INTERLEAVED}.items():
            fp.attrs[k] = np.uint32(v)
    with File(path) as f:
        buf = f.read_all()
        assert buf.dtype == np.int16
        np.testing.assert_array_equal(f.get_magnitude(buf), MAGNITUDES)

"""Synthetic recordings that carry a gauge length (or none), in every format.

The fields are the ones Audace-Lib writes since a5fccd5 (DATA_FORMAT.md
§3.4): DAT header bytes 120..124 (f32), HDF5 root attribute and TDMS
property ``gauge_length_m`` (f64, absent when there is none), SEG-Y line
``C25 GAUGE LENGTH: <m> METRES``. DAT headers are built byte by byte; the
other formats with the library their backend reads them with.

The fibre (Audace-Lib a288414, DATA_FORMAT.md §3.5) on request: the speed
of light in the fibre at DAT bytes 124..128 (u32), HDF5 attribute and TDMS
property ``light_speed_in_fiber_m_s`` beside the exact ``fiber_group_index``
(f64), SEG-Y line ``C26 FIBER GROUP INDEX: <n> (LIGHT SPEED <v> M/S)``; the
steps then follow the speed. Without it (the default) the files are legacy:
no speed, steps at 2·10⁸ m/s.

Not a test module: ``test_gauge_length.py`` and ``test_nanostrain.py``
import it.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

#: 62.5 MHz: 1.6 m between positions (the p21 Phase tap at downsampling 4)
#: at the legacy speed of light in the fibre.
SAMPLE_RATE = 62_500_000
#: The speed of light in the fibre of a file that does not state one, m/s.
LEGACY_SPEED = 200_000_000
#: Six positions of 1.6 m: the f64 `6 * 1e8 / SAMPLE_RATE` is 9.6 exactly
#: as Python spells it, and the DAT stores `float32(9.6)`.
GAUGE = 9.6
TRIG = 50
ANCHOR_NS = 1_790_000_000_123_456_789

#: Every class of f32 a multiply can meet: zeros, subnormals, the extremes,
#: infinities, quiet NaNs with payloads.
SPECIAL_F32 = np.array(
    [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007F_FFFF,
     0x7F7F_FFFF, 0xFF7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000,
     0xFFC0_1234, 0x3F80_0000],
    dtype=np.uint32,
).view(np.float32)


def phase_samples(rows: int, positions: int, seed: int = 0) -> np.ndarray:
    """Phases in radians (a few dozen rad either way), salted with the
    special values."""
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((rows, positions)) * 30.0).astype(np.float32)
    flat = data.reshape(-1)
    flat[: min(flat.size, SPECIAL_F32.size)] = SPECIAL_F32[: flat.size]
    return data


def pitch(sample_rate: int = SAMPLE_RATE, light_speed: int = 0) -> float:
    """Metres between two positions, ``v / (2 · rate)`` (Audace-Lib
    ``fibre::sample_pitch_m``) at ``light_speed`` (0: the legacy speed, the
    f64 ``1e8 / rate``)."""
    return (light_speed or LEGACY_SPEED) / (2 * sample_rate) if sample_rate else 0.0


def dat_header(
    *,
    content_type: int,
    line_size: int,
    sample_size: int,
    flags: int = 0,
    gauge: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    format_version: int = 1,
    light_speed: int = 0,
) -> bytes:
    """A 128-byte DataHeader, as a producer of 2026-11 writes it (with the
    speed of light in the fibre at bytes 124..128 when ``light_speed``)."""
    buf = bytearray(128)
    struct.pack_into("<q", buf, 0, ANCHOR_NS)
    struct.pack_into("<8i", buf, 32, line_size, TRIG, sample_size, sample_rate, flags, 1000, 100, 1)
    buf[64] = content_type
    buf[65:68] = b"p21"
    buf[108] = format_version
    if format_version:
        struct.pack_into("<ff", buf, 112, pitch(sample_rate, light_speed), 1.0 / TRIG)
    struct.pack_into("<f", buf, 120, gauge)
    struct.pack_into("<I", buf, 124, light_speed)
    return bytes(buf)


def write_dat(path: Path, data: np.ndarray, **header) -> Path:
    path.write_bytes(
        dat_header(line_size=data.shape[1], sample_size=data.dtype.itemsize, **header)
        + np.ascontiguousarray(data).astype(data.dtype.newbyteorder("<")).tobytes()
    )
    return path


def write_h5(
    path: Path,
    data: np.ndarray,
    *,
    content_type: int,
    gauge: Optional[float] = None,
    flags: int = 0,
    sample_size: Optional[int] = None,
    extra: Optional[dict] = None,
    light_speed: int = 0,
    group_index: Optional[float] = None,
) -> Path:
    import h5py

    from invisensing._core import CONTENT_TYPE_NAMES

    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=data)
        a = fp.attrs
        a["line_size"] = data.shape[1]
        a["trig_frequency"] = TRIG
        a["sample_size"] = sample_size or data.dtype.itemsize
        a["sample_rate"] = SAMPLE_RATE
        a["flags"] = flags
        a["range"] = 1000
        a["pulse_width"] = 100
        a["num_channels"] = 1
        a["timestamp"] = "2026-11-12 10:00:00.000+01:00"
        a["timestamp_ns"] = ANCHOR_NS
        a["content_type"] = np.uint32(content_type)
        a["content_type_name"] = CONTENT_TYPE_NAMES[content_type]
        a["format_version"] = 1
        a["card"] = "p21"
        a["spatial_step_m"] = pitch(SAMPLE_RATE, light_speed)
        a["temporal_step_s"] = 1.0 / TRIG
        if gauge is not None:
            a["gauge_length_m"] = np.float64(gauge)
        if light_speed:
            a["light_speed_in_fiber_m_s"] = np.uint32(light_speed)
            a["fiber_group_index"] = np.float64(group_index)
        for k, v in (extra or {}).items():
            a[k] = v
    return path


def write_tdms(
    path: Path,
    data: np.ndarray,
    *,
    content_type: int,
    gauge: Optional[float] = None,
    light_speed: int = 0,
    group_index: Optional[float] = None,
) -> Path:
    from nptdms import ChannelObject, RootObject, TdmsWriter

    from invisensing._core import CONTENT_TYPE_NAMES

    props = {
        "line_size": np.int32(data.shape[1]),
        "trig_frequency": np.int32(TRIG),
        "sample_size": np.int32(data.dtype.itemsize),
        "sample_rate": np.int32(SAMPLE_RATE),
        "flags": np.int32(0),
        "range": np.int32(1000),
        "pulse_width": np.int32(100),
        "num_channels": np.int32(1),
        "timestamp": "2026-11-12 10:00:00.000+01:00",
        "content_type": np.int32(content_type),
        "content_type_name": CONTENT_TYPE_NAMES[content_type],
        "format_version": np.int32(1),
        "card": "p21",
        "spatial_step_m": np.float64(pitch(SAMPLE_RATE, light_speed)),
        "temporal_step_s": np.float64(1.0 / TRIG),
    }
    if gauge is not None:
        props["gauge_length_m"] = np.float64(gauge)
    if light_speed:
        props["light_speed_in_fiber_m_s"] = np.uint32(light_speed)
        props["fiber_group_index"] = np.float64(group_index)
    with TdmsWriter(str(path)) as w:
        w.write_segment([
            RootObject(properties=props),
            ChannelObject("Audace", "acoustic_data", np.ascontiguousarray(data).reshape(-1)),
        ])
    return path


def segy_text_lines(
    *,
    content_type: int,
    positions: int,
    traces: int,
    gauge: Optional[float],
    light_speed: int = 0,
    group_index: Optional[float] = None,
) -> list:
    """The textual header Audace-Lib writes (`textual_header_lines`)."""
    from invisensing._core import CONTENT_TYPE_NAMES

    lines = [
        "C 1 AUDACE SEG-Y FILE - DISTRIBUTED ACOUSTIC SENSING DATA",
        "C 2 GENERATED BY INVISENSING AUDACE SYSTEM",
        "C 3 TIMESTAMP: 2026-11-12 10:00:00.000+01:00",
        f"C 4 SAMPLE RATE: {SAMPLE_RATE} Hz",
        f"C 5 TRIGGER FREQUENCY: {TRIG} Hz",
        "C 6 PULSE WIDTH: 100",
        "C 7 VOLTAGE RANGE: 1000 mV",
        f"C 8 SAMPLES PER TRACE: {positions}",
        f"C 9 NUMBER OF TRACES: {traces}",
        "C10 BYTES PER SAMPLE: 4",
        f"C11 FIBER POSITIONS PER TRACE: {positions}",
        "C12 FIBER OPTIC DAS DATA FORMAT:",
        "C13 - EACH TRACE IS ONE TRIGGER LINE, ITS SAMPLES RUN ALONG THE FIBER",
        f"C14 - SAMPLE SPACING ALONG THE FIBER: {pitch(SAMPLE_RATE, light_speed)!r} METERS",
        "C15 - TRACE SPACING IN TIME: 0.02 S",
        f"C16 - CONTENT-TYPE: {CONTENT_TYPE_NAMES[content_type]} ({content_type})",
        "C17 - QUANTITY: phase UNIT: rad",
        "C18 - FORMAT VERSION: 1",
        "C19 - SENSING RANGE: 8 METERS",
        "C20 - DURATION: 0 S",
        "C21 HEADER FLAGS (raw bits): 0x00000000",
        "C22 TRIGGER DELAY: 0 METRES",
        f"C23 TIMESTAMP NS: {ANCHOR_NS}",
        "C24 CARD: p21",
    ]
    if gauge is not None:
        lines.append(f"C25 GAUGE LENGTH: {gauge!r} METRES")
    if light_speed:
        n = len(lines) + 1
        lines.append(f"C{n:>2} FIBER GROUP INDEX: {group_index!r} (LIGHT SPEED {light_speed} M/S)")
    return lines


def write_sgy(
    path: Path,
    data: np.ndarray,
    *,
    content_type: int,
    gauge: Optional[float] = None,
    light_speed: int = 0,
    group_index: Optional[float] = None,
) -> Path:
    import segyio

    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE)
    spec.samples = list(range(data.shape[1]))
    spec.tracecount = data.shape[0]
    with segyio.create(str(path), spec) as f:
        for i in range(data.shape[0]):
            f.trace[i] = data[i]
    lines = segy_text_lines(
        content_type=content_type, positions=data.shape[1], traces=data.shape[0], gauge=gauge,
        light_speed=light_speed, group_index=group_index,
    )
    text = "".join(line.ljust(80)[:80] for line in lines).ljust(3200)[:3200]
    raw = bytearray(path.read_bytes())
    raw[:3200] = text.encode("ascii")
    path.write_bytes(bytes(raw))
    return path


#: Writer of each format, by suffix.
WRITERS = {
    ".dat": lambda p, d, ct, g: write_dat(p, d, content_type=ct, gauge=g or 0.0),
    ".h5": lambda p, d, ct, g: write_h5(p, d, content_type=ct, gauge=g),
    ".tdms": lambda p, d, ct, g: write_tdms(p, d, content_type=ct, gauge=g),
    ".sgy": lambda p, d, ct, g: write_sgy(p, d, content_type=ct, gauge=g),
}

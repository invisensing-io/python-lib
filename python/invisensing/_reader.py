"""
Modern :class:`File` reader for Audace DAS acquisition files.

The public API is **stable** from v1.0.0 onwards. Anything not exported
via :data:`__all__` is internal and may change without notice.

Usage::

    from invisensing import File, Mode

    with File("acquisition.dat") as f:
        if f.mode is Mode.IQ:
            i = f.get_i()
            q = f.get_q()
        elif f.mode is Mode.PHASE:
            phase = f.get_phase()
        else:
            samples = f.read_lines(1000)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np

from . import _core
from ._core import (
    DatReader,
    Header,
    parse_header,
    build_header_bytes,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_AC,
    FLAG_HIZ,
    FLAG_SCHUBERT,
    FLAG_SPECTRUM,
    FLAG_MOZART,
    FLAG_PHASE,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    HEADER_SIZE,
)

__all__ = [
    "File",
    "Mode",
    "Header",
    "DatReader",
    "parse_header",
    "export_dat",
    "FLAG_DEMODULATED",
    "FLAG_FLOAT",
    "FLAG_AC",
    "FLAG_HIZ",
    "FLAG_SCHUBERT",
    "FLAG_SPECTRUM",
    "FLAG_MOZART",
    "FLAG_PHASE",
    "FLAG_INTERLEAVED",
    "FLAG_UNSIGNED",
    "HEADER_SIZE",
]


# Speed-of-light in single-mode silica fibre at 1550 nm (n ≈ 1.45).
# Same value the Audace producer (Cardcontrol) uses for its own
# distance / spatial-resolution computations.
_LIGHT_SPEED_IN_FIBER = 206_856_796  # m/s


# ── Mode enum ───────────────────────────────────────────────────────────────


class Mode(Enum):
    """
    High-level acquisition mode of a file. Derived from the header
    flags so the user can dispatch with one ``match`` statement instead
    of bit-twiddling::

        match file.mode:
            case Mode.RAW:               samples = file.read_lines(n)
            case Mode.IQ:                i, q    = file.get_i(), file.get_q()
            case Mode.ARCTAN_MAGNITUDE:  a, m    = file.get_arctan(), file.get_magnitude()
            case Mode.PHASE:             phase   = file.get_phase()
    """

    #: Plain ADC samples — no on-board DSP, no INTERLEAVED. Read with
    #: :meth:`File.read_lines`.
    RAW = "raw"
    #: PCIe7821 IQ demodulation — INTERLEAVED ``[I, Q, I, Q, …]``,
    #: signed ``int16``. Use :meth:`File.get_i` / :meth:`File.get_q` /
    #: :meth:`File.get_iq`.
    IQ = "iq"
    #: PCIe7821 Arctan/Magnitude — INTERLEAVED + UNSIGNED:
    #: ``[arctan(Q/I) (i16), √(I²+Q²) (u16), …]``. Use
    #: :meth:`File.get_arctan` / :meth:`File.get_magnitude`.
    ARCTAN_MAGNITUDE = "arctan_magnitude"
    #: PCIe7821 Phase — single ``float32`` radians per spatial position,
    #: post DSP (fading suppression + spatial differential + detrend).
    #: Use :meth:`File.get_phase` (or :meth:`File.read_lines`).
    PHASE = "phase"

    @classmethod
    def from_flags(cls, flags: int) -> "Mode":
        """Decode the high-level mode from a raw header flag mask."""
        if flags & FLAG_PHASE:
            return cls.PHASE
        if flags & FLAG_INTERLEAVED:
            if flags & FLAG_UNSIGNED:
                return cls.ARCTAN_MAGNITUDE
            return cls.IQ
        return cls.RAW


# ── Bool-callable flag wrapper ──────────────────────────────────────────────


class _Flag(int):
    """
    Bool-like ``int`` subclass that is also callable.

    Used as the return type of the ``is_*`` properties so both the
    modern style (``file.is_interleaved``) and the legacy method style
    (``file.is_interleaved()``) yield the same boolean. Truthiness,
    equality with ``True``/``False``, and ``not`` all behave like a
    plain ``bool``.

    Internal — never raised in :data:`__all__`.
    """

    __slots__ = ()

    def __new__(cls, value: object) -> "_Flag":
        return int.__new__(cls, 1 if value else 0)

    def __call__(self) -> bool:
        return bool(int(self))

    def __bool__(self) -> bool:
        return bool(int(self))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "True" if self else "False"


# ── File facade ─────────────────────────────────────────────────────────────


class File:
    """
    Unified reader for Audace DAS acquisition files.

    Detects the on-disk format from the path's suffix (``.dat``,
    ``.h5``/``.hdf5``, ``.tdms``, ``.sgy``/``.segy``) and delegates the
    bytes-to-numpy work to a format-specific backend. The public API is
    identical across formats: same properties, same channel extractors,
    same numpy output shape.

    Use it as a context manager so the underlying file handle is closed
    cleanly even if an exception is raised::

        with File("acquisition.dat") as f:
            ...

    Parameters
    ----------
    path
        Path to the file. Suffix is used to pick the backend unless
        ``format`` is supplied explicitly.
    format
        Optional override (``"dat"``, ``"hdf5"``, ``"tdms"``, ``"segy"``).
        Use when the file has no recognised suffix.

    Raises
    ------
    FileNotFoundError
        If the path doesn't exist.
    ValueError
        If the format can't be auto-detected (and no override was given)
        or if the file isn't a valid Audace acquisition (zero
        ``line_size`` / ``sample_size``).
    ImportError
        If the chosen backend needs an optional third-party library
        (``h5py`` / ``npTDMS`` / ``segyio``) that isn't installed. The
        message names the right ``pip install invisensing[<extra>]``.
    """

    # Suffix → backend class name (looked up in ``_formats``). Adding a
    # new format means adding an entry here and a class in ``_formats``.
    _BACKENDS = {
        ".dat": "_DatBackend",
        ".h5": "_Hdf5Backend",
        ".hdf5": "_Hdf5Backend",
        ".tdms": "_TdmsBackend",
        ".sgy": "_SegyBackend",
        ".segy": "_SegyBackend",
    }

    def __init__(self, path: Union[str, Path], *, format: Optional[str] = None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no such file: '{path}'")
        suffix = (format or path.suffix).lower()
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix
        backend_name = self._BACKENDS.get(suffix)
        if backend_name is None:
            raise ValueError(
                f"unsupported file format '{suffix}' for {path.name!r}. "
                f"Supported suffixes: {sorted(self._BACKENDS.keys())}. "
                "Pass format='dat' (or another supported format) if the suffix is missing."
            )

        # Import the format wrapper lazily so a DAT-only install doesn't
        # need to have h5py / npTDMS / segyio available.
        from . import _formats

        backend_cls = getattr(_formats, backend_name)
        self._backend = backend_cls(path)
        self._path = path
        self._closed = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying file handle. Safe to call multiple times."""
        if self._closed:
            return
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()
        self._closed = True

    def __enter__(self) -> "File":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        # Best-effort cleanup. The backend's own __del__ takes care of
        # the underlying handle; this just suppresses double-close
        # warnings if the user already called close() explicitly.
        try:
            self.close()
        except Exception:
            pass

    # ── Iteration ──────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate one pulse at a time. Equivalent to ``while
        f.lines_left: yield f.read_lines(1)[0]``. Useful for streaming
        processing over large files.
        """
        while self.lines_left > 0:
            yield self.read_lines(1)[0]

    # ── Metadata (properties) ──────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Path to the source file."""
        return self._path

    @property
    def header(self) -> Header:
        """Parsed :class:`Header` (line_size, flags, rates, timestamp…)."""
        return self._backend.header

    @property
    def mode(self) -> Mode:
        """High-level acquisition mode — see :class:`Mode`."""
        return Mode.from_flags(self._backend.header.flags)

    @property
    def line_size(self) -> int:
        """Wire-side samples per pulse (doubled when ``is_interleaved``)."""
        return self._backend.header.line_size

    @property
    def positions_per_line(self) -> int:
        """Spatial fibre positions per pulse — ``line_size / 2`` if INTERLEAVED."""
        return self._backend.header.positions_per_line

    @property
    def sample_size(self) -> int:
        """Bytes per sample (2 for i16, 4 for i32/f32)."""
        return self._backend.header.sample_size

    @property
    def sample_rate(self) -> int:
        """Wire-side sample rate in Hz."""
        return self._backend.header.sample_rate

    @property
    def trig_frequency(self) -> int:
        """Trigger pulse repetition rate in Hz."""
        return self._backend.header.trig_frequency

    @property
    def num_lines(self) -> int:
        """Total trigger pulses recorded in the file."""
        return self._backend.num_lines

    @property
    def lines_left(self) -> int:
        """Pulses remaining behind the current read cursor."""
        return self._backend.lines_left

    @property
    def timestamp(self) -> str:
        """Acquisition start timestamp (free-form string from the producer)."""
        return self._backend.header.timestamp

    @property
    def flags(self) -> int:
        """Raw header flag bitmask (see ``FLAG_*`` constants)."""
        return self._backend.header.flags

    @property
    def range(self) -> float:
        """Voltage range in volts (header field is mV)."""
        return self._backend.header.range / 1000

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype matching ``(sample_size, FLOAT)``."""
        return _dtype_for(self._backend.header)

    @property
    def shape(self) -> tuple:
        """Total ``(num_lines, line_size)`` of the wire payload."""
        return (self.num_lines, self.line_size)

    @property
    def spatial_shape(self) -> tuple:
        """``(num_lines, positions_per_line)`` — shape returned by the channel extractors."""
        return (self.num_lines, self.positions_per_line)

    @property
    def duration(self) -> float:
        """Total duration of the recording in seconds."""
        return self.num_lines / max(self.trig_frequency, 1)

    @property
    def distance(self) -> float:
        """Round-trip distance (m) the pulse covers in the fibre."""
        positions = max(self.positions_per_line, 1)
        return (_LIGHT_SPEED_IN_FIBER * positions) / (2 * max(self.sample_rate, 1))

    # ── Flag predicates (properties — bool-callable for legacy compat) ──

    @property
    def is_demodulated(self) -> _Flag:
        """True if the on-board DSP was used (PCIe7821 demod modes)."""
        return _Flag(self._backend.header.is_demodulated)

    @property
    def is_interleaved(self) -> _Flag:
        """True if the wire carries 2 samples per spatial position."""
        return _Flag(self._backend.header.is_interleaved)

    @property
    def is_float(self) -> _Flag:
        """True if samples are IEEE-754 floats."""
        return _Flag(self._backend.header.is_float)

    @property
    def is_phase(self) -> _Flag:
        """True if the file carries unwrapped DAS phase (rad)."""
        return _Flag(self._backend.header.is_phase)

    @property
    def is_unsigned(self) -> _Flag:
        """True if the odd lane of an INTERLEAVED pair is u16 (magnitude)."""
        return _Flag(self._backend.header.is_unsigned)

    @property
    def is_ac(self) -> _Flag:
        """True if the ADC was AC-coupled during acquisition."""
        return _Flag(self._backend.header.is_ac)

    @property
    def is_hiz(self) -> _Flag:
        """True if the ADC ran in high-impedance mode."""
        return _Flag(self._backend.header.is_hiz)

    # ── Legacy aliases (kept stable forever) ───────────────────────────

    def is_acquisition_ac(self) -> bool:
        """Alias of :attr:`is_ac` kept for backwards compatibility."""
        return bool(self.is_ac)

    def is_acquisition_hiz(self) -> bool:
        """Alias of :attr:`is_hiz` kept for backwards compatibility."""
        return bool(self.is_hiz)

    # ── Streaming reads ────────────────────────────────────────────────

    def read_lines(self, n: int = 1) -> np.ndarray:
        """
        Read up to ``n`` consecutive pulses (rows). Returns a
        ``(rows, line_size)`` typed numpy array — dtype is
        :attr:`dtype`.

        Short-reads are tolerated: when fewer than ``n`` pulses remain,
        returns what's left. Raises :class:`OSError` only when the
        reader is already exhausted; raises :class:`ValueError` if
        ``n <= 0``.
        """
        return self._backend.read_lines(n)

    def read_all(self) -> np.ndarray:
        """
        Read every remaining pulse in one allocation. Use only when the
        file fits comfortably in RAM (``num_lines × line_size ×
        sample_size`` bytes). For large captures, prefer
        :meth:`read_lines` in a loop or iterate the file directly.
        """
        if self.lines_left == 0:
            return np.empty((0, self.line_size), dtype=self.dtype)
        return self.read_lines(self.lines_left)

    def rewind(self) -> None:
        """Reset the read cursor back to the start of the sample stream."""
        self._backend.rewind()

    # ── Channel extractors ─────────────────────────────────────────────
    #
    # Each extractor accepts an optional pre-read array. Passing ``None``
    # reads ``n`` new pulses from the file; passing an array re-uses the
    # buffer the user already has (e.g. across multiple extractors in
    # the same loop iteration so the file cursor isn't double-advanced).

    def get_i(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Extract the **I** lane of an INTERLEAVED I/Q file (PCIe7821
        :attr:`Mode.IQ`).

        Returns a ``(rows, positions_per_line)`` ``int16`` array.
        Raises :class:`ValueError` if the file's mode is not
        :attr:`Mode.IQ`.
        """
        self._require_mode("get_i", Mode.IQ)
        return _core.split_pair_i16_primary(self._ensure(data, n))

    def get_q(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """Extract the **Q** lane of an INTERLEAVED I/Q file."""
        self._require_mode("get_q", Mode.IQ)
        return _core.split_pair_i16_secondary(self._ensure(data, n))

    def get_iq(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Combined ``complex64`` view of an INTERLEAVED I/Q file:
        ``result[r, j] = I[r, j] + j·Q[r, j]``. Returns
        ``(rows, positions_per_line)`` ``complex64``.
        """
        self._require_mode("get_iq", Mode.IQ)
        return _core.split_pair_to_complex_i16(self._ensure(data, n))

    def get_arctan(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Extract the **arctan(Q/I)** lane of an INTERLEAVED arctan/√
        file (PCIe7821 :attr:`Mode.ARCTAN_MAGNITUDE`). Returns
        ``(rows, positions_per_line)`` ``int16`` — the fixed-point
        scale is ``32767 ↔ +π`` (vendor convention).
        """
        self._require_mode("get_arctan", Mode.ARCTAN_MAGNITUDE)
        return _core.split_pair_i16_primary(self._ensure(data, n))

    def get_magnitude(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Extract the **√(I²+Q²)** magnitude lane of an INTERLEAVED
        arctan/√ file, reinterpreted as ``u16`` (the wire buffer is
        typed ``i16`` end-to-end but the vendor sends magnitudes as
        unsigned). Returns ``(rows, positions_per_line)`` ``uint16``.
        """
        self._require_mode("get_magnitude", Mode.ARCTAN_MAGNITUDE)
        return _core.split_pair_unsigned(self._ensure(data, n))

    def get_phase(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Phase samples (``float32`` radians) for PCIe7821
        :attr:`Mode.PHASE` files. One sample per spatial position, no
        INTERLEAVED pair. Returns ``(rows, positions_per_line)``
        ``float32``.

        For consistency with the other extractors this method exists
        even though Phase files are already one sample per position
        — passing ``data`` returns it unchanged; passing nothing reads
        ``n`` new pulses.
        """
        self._require_mode("get_phase", Mode.PHASE)
        if data is not None:
            return np.ascontiguousarray(data)
        return self.read_lines(n)

    # ── Physical-unit extractors (float32, scaled to the underlying physics) ──
    #
    # I, Q and the magnitude are physically continuous voltages on the
    # photodetector, stored as fixed-point integers on the wire (i16 for
    # I/Q, u16 for the magnitude). Arctan is a continuous angle stored
    # as i16 with the vendor convention ``32767 ↔ +π``. The methods below
    # return ``float32`` numpy arrays in the natural physical unit.
    #
    # The volt scaling uses ``self.range`` (from the ADC voltage range
    # field of the header). The radian scaling for arctan uses
    # ``π / 32768`` — vendor-specified and not user-configurable.

    def get_i_volts(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> np.ndarray:
        """
        Like :meth:`get_i` but returns the I lane as a ``float32``
        numpy array scaled to **volts**: ``f32 = i16 × range / 32768``.
        Use this when you want physically-meaningful values without
        having to remember the ADC scaling.
        """
        self._require_mode("get_i_volts", Mode.IQ)
        return _to_volts_i16(_core.split_pair_i16_primary(self._ensure(data, n)), self.range)

    def get_q_volts(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> np.ndarray:
        """Like :meth:`get_q` but returns the Q lane as ``float32`` volts."""
        self._require_mode("get_q_volts", Mode.IQ)
        return _to_volts_i16(_core.split_pair_i16_secondary(self._ensure(data, n)), self.range)

    def get_iq_volts(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> np.ndarray:
        """
        Like :meth:`get_iq` but returns a ``complex64`` array whose
        real / imaginary parts are in **volts**. Convenient for
        envelope / phase computations that need physical units::

            iq = f.get_iq_volts()
            envelope_v = np.abs(iq)         # volts
            phase_rad  = np.angle(iq)       # radians (wrapped)
        """
        self._require_mode("get_iq_volts", Mode.IQ)
        buf = self._ensure(data, n)
        i = _core.split_pair_i16_primary(buf).astype(np.float32) * (self.range / 32768.0)
        q = _core.split_pair_i16_secondary(buf).astype(np.float32) * (self.range / 32768.0)
        out = np.empty(i.shape, dtype=np.complex64)
        out.real = i
        out.imag = q
        return out

    def get_arctan_radians(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Like :meth:`get_arctan` but returns the arctan lane as a
        ``float32`` numpy array in **radians**, scaled with the vendor
        convention ``32767 ↔ +π`` (``f32 = i16 × π / 32768``). Range
        is ``[-π, +π]``, the raw wrapped output of the FPGA arctan2.
        """
        self._require_mode("get_arctan_radians", Mode.ARCTAN_MAGNITUDE)
        arr = _core.split_pair_i16_primary(self._ensure(data, n))
        return arr.astype(np.float32) * np.float32(np.pi / 32768.0)

    def get_magnitude_volts(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Like :meth:`get_magnitude` but returns the magnitude lane as a
        ``float32`` numpy array in **volts**:
        ``f32 = u16 × range / 32768``.

        The LSB scaling matches the I/Q lanes (the FPGA computes
        ``√(I²+Q²)`` from the same i16 I/Q codes and keeps the same
        per-LSB voltage). The output is always ≥ 0.
        """
        self._require_mode("get_magnitude_volts", Mode.ARCTAN_MAGNITUDE)
        arr = _core.split_pair_unsigned(self._ensure(data, n))
        return arr.astype(np.float32) * np.float32(self.range / 32768.0)

    def channels(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> dict:
        """
        Return *all* channels of the file in one call, as a dict whose
        keys depend on the :attr:`mode`::

            Mode.RAW              -> {"samples": ndarray}
            Mode.IQ               -> {"i": …, "q": …, "iq": complex64}
            Mode.ARCTAN_MAGNITUDE -> {"arctan": …, "magnitude": …}
            Mode.PHASE            -> {"phase": …}

        Convenience for ad-hoc scripts: pass a pre-read buffer if you
        already have one, otherwise the method reads ``n`` pulses.
        """
        m = self.mode
        if m is Mode.RAW:
            buf = self._ensure(data, n)
            return {"samples": buf}
        if m is Mode.IQ:
            buf = self._ensure(data, n)
            return {
                "i": _core.split_pair_i16_primary(buf),
                "q": _core.split_pair_i16_secondary(buf),
                "iq": _core.split_pair_to_complex_i16(buf),
            }
        if m is Mode.ARCTAN_MAGNITUDE:
            buf = self._ensure(data, n)
            return {
                "arctan": _core.split_pair_i16_primary(buf),
                "magnitude": _core.split_pair_unsigned(buf),
            }
        # PHASE
        buf = self._ensure(data, n)
        return {"phase": buf}

    # ── Helpers ────────────────────────────────────────────────────────

    def _ensure(self, data: Optional[np.ndarray], n: int) -> np.ndarray:
        if data is None:
            return self.read_lines(n)
        return np.ascontiguousarray(data)

    def _require_mode(self, method: str, expected: Mode) -> None:
        if self.mode is expected:
            return
        raise ValueError(
            f"{method}: file mode is {self.mode.value!r}, but this extractor "
            f"only applies to {expected.value!r}. "
            f"Inspect file.mode to dispatch — see the Mode enum docs."
        )

    # ── Introspection ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"File({str(self._path)!r}, mode={self.mode.value}, "
            f"shape={self.shape}, sample_rate={self.sample_rate} Hz, "
            f"trig_frequency={self.trig_frequency} Hz, "
            f"duration={self.duration:.2f}s, distance={self.distance:.2f}m)"
        )


# ── dtype helper ────────────────────────────────────────────────────────────


def _dtype_for(h: Header) -> np.dtype:
    """numpy dtype matching ``(sample_size, FLOAT)``."""
    if h.is_float:
        return np.dtype(f"float{h.sample_size * 8}")
    return np.dtype(f"int{h.sample_size * 8}")


def _to_volts_i16(arr: np.ndarray, range_v: float) -> np.ndarray:
    """
    Scale an ``int16`` ADC-code array to ``float32`` volts using
    ``f32 = i16 × range_v / 32768``. ``range_v`` is taken from the
    file's header (``File.range``, in volts).

    Returns an owned ``float32`` array; the input ``arr`` is left
    unchanged.
    """
    return arr.astype(np.float32) * np.float32(range_v / 32768.0)


# ── Write path ──────────────────────────────────────────────────────────────


def export_dat(
    path: Union[str, Path],
    data: np.ndarray,
    *,
    sample_rate: int,
    trig_frequency: int,
    range_v: float = 0.0,
    timestamp: str = "",
    flags: int = 0,
    pulse_width: int = 0,
    num_channels: int = 1,
) -> None:
    """
    Write a 2-D numpy array out as an Audace ``.dat`` file (128-byte
    header + raw samples). The array's ``shape[1]`` becomes
    ``line_size`` and its dtype determines ``sample_size`` plus the
    ``FLOAT`` flag.

    Parameters
    ----------
    path
        Output file path. Overwritten if it exists.
    data
        ``(num_lines, line_size)`` 2-D array. Any
        contiguous-or-not C-order layout is accepted; the data is
        densified on write.
    sample_rate
        Wire-side sample rate in Hz.
    trig_frequency
        Trigger pulse rate in Hz.
    range_v
        ADC voltage range in volts (stored as mV in the header).
    timestamp
        Free-form acquisition timestamp (≤ 32 bytes UTF-8).
    flags
        Raw header flag mask. The ``FLOAT`` bit is set automatically
        when ``data.dtype.kind == 'f'``.
    pulse_width
        Trigger pulse width in ns.
    num_channels
        Number of active socket channels in the original acquisition.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D.
    OSError
        If the write fails.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (rows × line_size), got {data.ndim}-D")
    if data.dtype.kind == "f":
        flags = flags | FLAG_FLOAT
    else:
        flags = flags & ~FLAG_FLOAT
    header = Header(
        line_size=int(data.shape[1]),
        trig_frequency=int(trig_frequency),
        sample_size=int(data.dtype.itemsize),
        sample_rate=int(sample_rate),
        flags=int(flags),
        range=int(round(range_v * 1000)),
        pulse_width=int(pulse_width),
        num_channels=int(num_channels),
        timestamp=str(timestamp),
    )
    bytes_hdr = build_header_bytes(header)
    with open(path, "wb") as fp:
        fp.write(bytes_hdr)
        fp.write(np.ascontiguousarray(data).tobytes())

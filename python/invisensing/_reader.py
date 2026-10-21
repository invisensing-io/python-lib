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

The ``with`` block is not decoration: leaving it closes the underlying
file descriptor there and then. Without it, call ``f.close()``. A closed
file still describes itself (header, mode, shape, dtype…); only reading
data from it raises.
"""

from __future__ import annotations

import operator
import warnings
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
    TIME_FLAG_MAP_V1,
    TIME_FLAG_CORRECTED,
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
    "TIME_FLAG_MAP_V1",
    "TIME_FLAG_CORRECTED",
]


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_FS_PER_NS = 1_000_000

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

    Use it as a context manager. It is the only usage whose file
    descriptor is released at a point *you* control rather than
    whenever the garbage collector gets to the object::

        with File("acquisition.dat") as f:
            ...

    Outside a ``with`` block, call :meth:`close` yourself; it is
    idempotent. A :class:`File` that is neither closed nor used as a
    context manager emits a :class:`ResourceWarning` when it is
    collected — silent by default, shown under ``python -W default``,
    ``python -X dev`` and pytest. This matters on a long-running
    process walking a data tree: one leaked descriptor per file ends in
    ``EMFILE``, and on an NFS/CIFS mount an open handle also blocks the
    file from being deleted or replaced.

    Closing releases the *descriptor*, and nothing else. The header was
    read at open and is kept, so a closed :class:`File` still describes
    the recording — :attr:`header`, :attr:`mode`, :attr:`num_lines`,
    :attr:`sample_rate`, :attr:`dtype`, :attr:`duration`, the ``is_*``
    predicates and the rest all keep answering, and asking them is not
    an error::

        with File("acquisition.dat") as f:
            data = f.read_all()
        dt = 1.0 / f.trig_frequency_real_hz  # fine: metadata, not I/O

    numpy arrays already returned by :meth:`read_lines` or the channel
    extractors own their memory and remain valid and readable too.
    Only an operation that would have to touch the file again —
    :meth:`read_lines`, :meth:`read_all`, :meth:`rewind`,
    :meth:`seek_lines`, iterating the file, and :attr:`lines_left`,
    which is the cursor of a handle that no longer exists — raises
    :class:`ValueError`.

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

    # HDF5's `trig_frequency_real_hz` attribute when it describes the file
    # (set at open). A class-level default so that `__repr__` never trips on
    # an instance whose construction stopped half way.
    _real_hz_attr: Optional[float] = None
    # The file's time attributes (HDF5 written from the card's time map),
    # set at open; ``None`` for every other file.
    _time_axis = None

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
        # Metadata is read once, here, and kept on the `File` itself.
        # Describing a file needs the descriptor exactly once — at open.
        # Everything downstream (mode, dtype, shape, duration, the flag
        # predicates) is arithmetic on these two values, so it keeps
        # answering after close(); see `close()` for why that matters.
        self._header = self._backend.header
        self._num_lines = int(self._backend.num_lines)
        # HDF5's `trig_frequency_real_hz` attribute, kept only when it can
        # describe this file (same guard as the header's rational). Every
        # other backend has none.
        self._real_hz_attr = _consistent_rate(
            getattr(self._backend, "trig_frequency_real_hz_attr", None),
            self._header.trig_frequency,
        )
        # The card's time map as the recorder wrote it (HDF5 only).
        self._time_axis = getattr(self._backend, "time_axis", None)

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has run. Never resets."""
        return bool(getattr(self, "_closed", True))

    def close(self) -> None:
        """
        Release the underlying OS file handle, immediately.

        Idempotent: calling it a second time (or after a ``with``
        block) is a no-op and never raises.

        The reference to the backend is dropped *before* the handle is
        closed, so the descriptor goes away at this call and not
        whenever the garbage collector happens to reclaim this object.

        Closing gives up the *descriptor*, not the *knowledge*. What was
        already read stays readable: every metadata accessor
        (:attr:`header`, :attr:`mode`, :attr:`num_lines`, :attr:`dtype`,
        :attr:`shape`, :attr:`duration`, :attr:`distance`, the ``is_*``
        predicates…) keeps answering from the values captured at open,
        and numpy arrays already returned by :meth:`read_lines` (and the
        channel extractors) own their memory and stay valid. Only what
        genuinely needs the file back — :meth:`read_lines`,
        :meth:`read_all`, :meth:`rewind`, :meth:`seek_lines`, iteration
        and :attr:`lines_left` (the cursor of a handle that no longer
        exists) — raises :class:`ValueError`.

        One caveat, and it is the only case where the descriptor does
        *not* go away at this call: closing a ``.dat`` while another
        thread is inside :meth:`read_lines` on the *same* object. The
        native reader is mutably borrowed for the length of that read,
        so the release is deferred until it finishes; this call still
        returns without raising, and a :class:`ResourceWarning` says so.
        Do not share one :class:`File` across threads.
        """
        if getattr(self, "_closed", True):
            return
        # Drop our reference FIRST. `File._backend` is the only one, so
        # this is what actually releases the handle; the explicit
        # `close()` below just makes the moment deterministic instead of
        # leaving it to the refcount. Order also makes the method
        # idempotent even if the backend's close were to raise.
        backend, self._backend = getattr(self, "_backend", None), None
        self._closed = True
        if backend is not None:
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> "File":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        # Nothing here may raise: at interpreter shutdown module globals
        # (including `warnings`) may already be None, hence the
        # systematic getattr-with-default and the blanket guards.
        try:
            if not getattr(self, "_closed", True):
                try:
                    warnings.warn(
                        f"invisensing.File({getattr(self, '_path', '?')}) was not "
                        "closed explicitly; use a `with` block or call .close()",
                        ResourceWarning,
                        stacklevel=2,
                        source=self,
                    )
                except Exception:
                    pass
            self.close()
        except Exception:
            pass

    def _require_open(self):
        """Return the live backend, or explain that the file is closed.

        Guards *data* access only — anything that has to touch the file
        again. Metadata read at open is answered from :attr:`header` and
        never comes through here: raising on a question the object can
        already answer would turn a correct script into a crash for no
        gain. Without this guard a genuine use-after-close would surface
        as ``AttributeError: 'NoneType' object has no attribute
        'read_lines'``, which says nothing about what the caller did
        wrong.
        """
        backend = getattr(self, "_backend", None)
        if backend is None or getattr(self, "_closed", True):
            raise ValueError(
                f"operation on a closed File({getattr(self, '_path', '?')})"
            )
        return backend

    # ── Iteration ──────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate one pulse at a time. Equivalent to ``while
        f.lines_left: yield f.read_lines(1)[0]``. Useful for streaming
        processing over large files.

        The backend is resolved once, at the first ``next()``, and the
        loop then talks to it directly: at one pulse per iteration the
        two per-pulse ``_require_open()`` checks were measurable (see
        ``tests/test_performance.py::test_iteration_overhead_per_pulse``).
        Closing the file mid-iteration is still caught — the backend
        raises :class:`ValueError` on the very next read — it is just
        reported by the backend instead of by the facade.
        """
        backend = self._require_open()
        while backend.lines_left > 0:
            yield backend.read_lines(1)[0]

    # ── Metadata (properties) ──────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Path to the source file."""
        return self._path

    @property
    def header(self) -> Header:
        """Parsed :class:`Header` (line_size, flags, rates, timestamp…)."""
        return self._header

    @property
    def mode(self) -> Mode:
        """High-level acquisition mode — see :class:`Mode`."""
        return Mode.from_flags(self._header.flags)

    @property
    def line_size(self) -> int:
        """Wire-side samples per pulse (doubled when ``is_interleaved``)."""
        return self._header.line_size

    @property
    def positions_per_line(self) -> int:
        """Spatial fibre positions per pulse — ``line_size / 2`` if INTERLEAVED."""
        return self._header.positions_per_line

    @property
    def sample_size(self) -> int:
        """Bytes per sample (2 for i16, 4 for i32/f32)."""
        return self._header.sample_size

    @property
    def sample_rate(self) -> int:
        """Wire-side sample rate in Hz."""
        return self._header.sample_rate

    @property
    def trig_frequency(self) -> int:
        """**Requested** trigger pulse repetition rate in Hz (an integer).

        The card fires at exactly this rate only when it divides the card's
        clock; otherwise it fires slightly faster — see
        :attr:`trig_frequency_real_hz`, which is the rate to compute times
        and durations with. This integer stays the configured value: use it
        to size blocks or to label the acquisition.
        """
        return self._header.trig_frequency

    @property
    def trig_clock_hz(self) -> Optional[int]:
        """Clock the card counts its trigger period in, Hz — or ``None``.

        With :attr:`trig_period_ticks`, the exact rate the card fires at:
        ``trig_clock_hz / trig_period_ticks``. ``None`` when the file does
        not carry it (recordings made before it existed, an external
        trigger, TDMS and SEG-Y files) **or** carries one that does not
        describe this file's :attr:`trig_frequency` (more than 1 % and more
        than 1 Hz away from it: left behind by a header whose frequency was
        rewritten — it is ignored). The raw stored values are in
        ``header.trig_clock_hz`` / ``header.trig_period_ticks``.
        """
        t = self._header.trig_timing
        return None if t is None else int(t[0])

    @property
    def trig_period_ticks(self) -> Optional[int]:
        """Trigger period in ticks of :attr:`trig_clock_hz` — or ``None``
        (same rule as :attr:`trig_clock_hz`; both are ``None`` together)."""
        t = self._header.trig_timing
        return None if t is None else int(t[1])

    @property
    def trig_frequency_real_hz(self) -> float:
        """The trigger rate the card **really** fired at, in Hz.

        * ``trig_clock_hz / trig_period_ticks`` when the file carries the
          card's rational (``75_000`` requested on a 250 MHz clock →
          ``250e6 / 3333 = 75007.5008`` Hz);
        * else HDF5's ``trig_frequency_real_hz`` attribute, when the file
          carries it alone;
        * else ``float(trig_frequency)`` — older recordings and external
          triggers, where the requested rate is the only one known.

        Use it for anything derived from the line cadence: time axes,
        durations, the sampling rate of a spectrum along time.
        """
        t = self._header.trig_timing
        if t is not None:
            return t[0] / t[1]
        if self._real_hz_attr is not None:
            return self._real_hz_attr
        return float(self._header.trig_frequency)

    @property
    def trig_frequency_is_exact(self) -> bool:
        """``True`` when the card fired exactly at :attr:`trig_frequency`,
        and when the real rate is unknown (then nothing says otherwise)."""
        if self._header.trig_timing is not None:
            return bool(self._header.trig_frequency_is_exact)
        if self._real_hz_attr is not None:
            return self._real_hz_attr == float(self._header.trig_frequency)
        return True

    def line_time_ns(self, i: int, *, nominal: bool = False) -> int:
        """Acquisition time of line (trigger pulse) ``i``, 0-based, in
        nanoseconds since the Unix epoch.

        * A file stamped from the card's **time map** (HDF5 with
          ``time_correction = "cardclock-v1"``, see
          :attr:`time_correction`): the time on the file's axis — from the
          last node ``k`` at or before ``i``, ``time_map_ns[k] + ⌊(i −
          time_map_line[k]) · time_map_period_fs[k] / 10⁶⌋`` (a period 0
          counts on the card's rational). The **corrected** time: the card's
          oscillator drift and its lost shots taken out, exactly the
          nanosecond the acquisition's live modules gave this line.
        * Any other file: ``header.acquisition_ns + ⌊i · trig_period_ticks ·
          10⁹ / trig_clock_hz⌋`` when it carries the card's rational, else
          ``header.acquisition_ns + ⌊i · 10⁹ / trig_frequency⌋`` — the
          integer 1.3.0 gave. A DAT, SEG-Y or TDMS whose header says
          :attr:`Header.time_corrected` starts at the corrected time of its
          first line and counts from there.

        ``nominal=True`` gives the count instead, as 1.3.0 computed it: from
        the file's ``timestamp_nominal_ns`` when it carries one, else from
        its start time.

        Exact integer arithmetic. When the file has no anchor
        (``header.acquisition_ns == 0``: legacy ASCII timestamp) the result
        is the offset from the first line. ``i`` may exceed
        :attr:`num_lines` (the last segment extrapolates). Raises
        :class:`ValueError` for a negative ``i`` and :class:`TypeError` for
        a non-integer.
        """
        i = operator.index(i)
        if i < 0:
            raise ValueError(f"line index must be >= 0 (got {i})")
        axis = self._time_axis
        if nominal:
            if axis is not None and axis.nominal_start_ns is not None:
                return _clamp_i64(axis.nominal_start_ns + self._header.lines_to_ns(i))
            return self._header.line_time_ns(i)
        if axis is not None and axis.corrected:
            return _axis_time_ns(axis.lines, axis.ns, axis.period_fs, self._header.line_timing, i)
        return self._header.line_time_ns(i)

    def times_ns(
        self, start: int = 0, count: Optional[int] = None, *, nominal: bool = False
    ) -> np.ndarray:
        """The times of lines ``start .. start + count`` (default: to the
        end of the file), ns since the Unix epoch, as an ``int64`` numpy
        vector — :meth:`line_time_ns` of each line, the same integers,
        computed in the native core. Metadata only: no sample is read, and
        it works on a closed file.
        """
        start = operator.index(start)
        if start < 0:
            raise ValueError(f"start must be >= 0 (got {start})")
        if count is None:
            count = max(0, self._num_lines - start)
        count = operator.index(count)
        if count < 0:
            raise ValueError(f"count must be >= 0 (got {count})")
        timing = self._header.line_timing
        axis = self._time_axis
        if not nominal and axis is not None and axis.corrected:
            return _core.time_axis_ns(axis.lines, axis.ns, axis.period_fs, timing, start, count)
        if nominal and axis is not None and axis.nominal_start_ns is not None:
            anchor = axis.nominal_start_ns
        else:
            anchor = self._header.acquisition_ns
        # One node of period 0 at line 0: `anchor + ⌊i · ticks · 10⁹ / clock⌋`,
        # exactly `Header.line_time_ns`.
        return _core.time_axis_ns([0], [anchor], [0], timing, start, count)

    @property
    def time_correction(self) -> Optional[str]:
        """How the file's times were stamped, or ``None`` when it does not
        say (files written before the card's time map, DAT/SEG-Y/TDMS
        without the flag).

        * ``"cardclock-v1"``: from the card's time map — ``timestamp_ns``
          and :meth:`line_time_ns` are the **corrected** times (the card's
          oscillator drift and lost shots taken out); the count is
          ``line_time_ns(i, nominal=True)``. A DAT/SEG-Y/TDMS says so with
          :attr:`Header.time_corrected` (its start time only).
        * ``"none"``: timed by the count (no time map for that run).
        """
        axis = self._time_axis
        if axis is not None:
            return axis.correction
        if self._header.time_corrected:
            return "cardclock-v1"
        return None

    @property
    def time_correction_state(self) -> Optional[str]:
        """The least favourable state of the correction over the file
        (``"tracking"``, ``"prior"``, ``"slewing"``, ``"suspect"``,
        ``"provisional"``…), or ``None`` when the file does not say."""
        axis = self._time_axis
        return None if axis is None else axis.state

    @property
    def time_uncertainty_ns(self) -> Optional[int]:
        """Largest estimated uncertainty (1 σ) of the corrected time over
        the file, ns — the precision of the TIME, not of the cadence —
        or ``None``."""
        axis = self._time_axis
        return None if axis is None else axis.uncertainty_ns

    @property
    def trig_frequency_measured_hz(self) -> Optional[float]:
        """The card's cadence as the acquisition measured it against the
        host clock (true seconds), Hz, or ``None`` when the file does not
        carry it.

        Not the rate to compute a spectrum with: :attr:`trig_frequency_real_hz`
        (the programmed cadence) stays the sampling rate of the data along
        time; this one says how far the card's oscillator was off it.
        """
        axis = self._time_axis
        return None if axis is None else axis.measured_hz

    @property
    def num_lines(self) -> int:
        """Total trigger pulses recorded in the file."""
        return self._num_lines

    @property
    def lines_left(self) -> int:
        """Pulses remaining behind the current read cursor.

        This one *does* need the file open: it describes the position of
        a live read cursor, not a property of the recording. On a closed
        file it raises :class:`ValueError` — use :attr:`num_lines`, which
        keeps answering, to describe what the file contains.
        """
        return self._require_open().lines_left

    @property
    def timestamp(self) -> str:
        """Acquisition start timestamp (free-form string from the producer)."""
        return self._header.timestamp

    @property
    def flags(self) -> int:
        """Raw header flag bitmask (see ``FLAG_*`` constants)."""
        return self._header.flags

    @property
    def range(self) -> float:
        """Voltage range in volts (header field is mV)."""
        return self._header.range / 1000

    @property
    def trigger_delay(self) -> int:
        """Trigger pre-roll in **metres of fibre**.

        ``0`` means no delay — which is also what captures written before the
        field existed report, and for them it says the same thing.
        """
        return self._header.trigger_delay

    @property
    def is_mozart(self) -> Optional[bool]:
        """Which demodulation pipeline ran, or ``None`` if none did.

        ``True`` = Mozart, ``False`` = Schubert, ``None`` on raw data. The
        distinction is carried by the *absence* of the ``MOZART`` bit rather
        than by a bit of its own, so on a raw capture "not Mozart" would read
        as "Schubert" and name a pipeline that never executed. Treat ``None``
        as "not applicable", never as a default.
        """
        return self._header.is_mozart

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype matching ``(sample_size, FLOAT)``."""
        return _dtype_for(self._header)

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
        """Total duration of the recording in seconds:
        ``num_lines / trig_frequency_real_hz``.

        On a recording without a real rate (older files, external trigger)
        this is ``num_lines / trig_frequency``, exactly as before. A file
        stamped from the card's time map (:attr:`time_correction` =
        ``"cardclock-v1"``) is measured on its corrected times instead:
        ``num_lines`` mean line periods, the first line to the last
        (``timestamp_last_ns − timestamp_ns``) spread over ``num_lines − 1``
        periods — lost shots inside the file included.
        """
        axis = self._time_axis
        n = self.num_lines
        if axis is not None and axis.corrected and axis.last_ns is not None and n > 1:
            first = self.line_time_ns(0)
            return (axis.last_ns - first) / 1e9 * n / (n - 1)
        if self._header.trig_timing is None and self._real_hz_attr is None:
            return self.num_lines / max(self.trig_frequency, 1)
        return self.num_lines / self.trig_frequency_real_hz

    @property
    def distance(self) -> float:
        """Round-trip distance (m) the pulse covers in the fibre."""
        positions = max(self.positions_per_line, 1)
        return (_LIGHT_SPEED_IN_FIBER * positions) / (2 * max(self.sample_rate, 1))

    # ── Flag predicates (properties — bool-callable for legacy compat) ──

    @property
    def is_demodulated(self) -> _Flag:
        """True if the on-board DSP was used (PCIe7821 demod modes)."""
        return _Flag(self._header.is_demodulated)

    @property
    def is_interleaved(self) -> _Flag:
        """True if the wire carries 2 samples per spatial position."""
        return _Flag(self._header.is_interleaved)

    @property
    def is_float(self) -> _Flag:
        """True if samples are IEEE-754 floats."""
        return _Flag(self._header.is_float)

    @property
    def is_phase(self) -> _Flag:
        """True if the file carries unwrapped DAS phase (rad)."""
        return _Flag(self._header.is_phase)

    @property
    def is_unsigned(self) -> _Flag:
        """True if the odd lane of an INTERLEAVED pair is u16 (magnitude)."""
        return _Flag(self._header.is_unsigned)

    @property
    def is_ac(self) -> _Flag:
        """True if the ADC was AC-coupled during acquisition."""
        return _Flag(self._header.is_ac)

    @property
    def is_hiz(self) -> _Flag:
        """True if the ADC ran in high-impedance mode."""
        return _Flag(self._header.is_hiz)

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
        return self._require_open().read_lines(n)

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
        self._require_open().rewind()

    def seek_lines(self, n: int) -> None:
        """Move the read cursor to pulse ``n`` (0-based) without reading the
        skipped pulses.

        For ``.dat``/``.hdf5`` this is an O(1) seek; the next
        :meth:`read_lines` continues from ``n``. ``n`` is clamped to
        :attr:`num_lines`. Lets a viewer jump to ``--start-time`` on a
        multi-GB file without streaming past the skipped data.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        self._require_open().seek(int(n))

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
        # Debuggers call this on objects in any state, including one whose
        # construction failed — never let it raise.
        path = str(getattr(self, "_path", "?"))
        if getattr(self, "_header", None) is None:
            # Construction failed before the header was read.
            return f"File({path!r}, unopened)"
        state = ", closed" if self.closed else ""
        real = (
            ""
            if self.trig_frequency_is_exact
            else f" (real {self.trig_frequency_real_hz:.4f} Hz)"
        )
        return (
            f"File({path!r}, mode={self.mode.value}, "
            f"shape={self.shape}, sample_rate={self.sample_rate} Hz, "
            f"trig_frequency={self.trig_frequency} Hz{real}, "
            f"duration={self.duration:.2f}s, distance={self.distance:.2f}m"
            f"{state})"
        )


# ── dtype helper ────────────────────────────────────────────────────────────


def _dtype_for(h: Header) -> np.dtype:
    """numpy dtype matching ``(sample_size, FLOAT)``."""
    if h.is_float:
        return np.dtype(f"float{h.sample_size * 8}")
    return np.dtype(f"int{h.sample_size * 8}")


def _clamp_i64(v: int) -> int:
    return max(_I64_MIN, min(_I64_MAX, v))


def _axis_time_ns(lines, ns, period_fs, timing, line: int) -> int:
    """Time of file line ``line`` on a time axis — the formula of the
    acquisition's ``audace::time_map::axis_time_of``, in Python integers:
    from the last node ``k`` with ``lines[k] <= line`` (the first node for a
    line before every node), ``ns[k] + ⌊(line − lines[k]) · period_fs[k] /
    10⁶⌋``, or ``⌊(line − lines[k]) · ticks · 10⁹ / clock⌋`` for a node of
    period 0 (``timing = (clock, ticks)``; 0 without one). Floor division,
    negative offsets included; saturated to the ``int64`` range."""
    k = 0
    lo, hi = 0, len(lines)
    while lo < hi:  # last index with lines[k] <= line
        mid = (lo + hi) // 2
        if lines[mid] <= line:
            lo = mid + 1
        else:
            hi = mid
    k = max(lo - 1, 0)
    d = line - lines[k]
    p = period_fs[k]
    if p == 0:
        if timing is None or timing[0] <= 0:
            dt = 0
        else:
            clock, ticks = timing
            dt = (d * ticks * 1_000_000_000) // clock
    else:
        dt = (d * p) // _FS_PER_NS
    return _clamp_i64(ns[k] + dt)


def _consistent_rate(rate: Optional[float], trig_frequency: int) -> Optional[float]:
    """``rate`` when it can be the real rate of a recording requested at
    ``trig_frequency`` — within 1 % or within 1 Hz of it, the guard the
    header's rational goes through — else ``None``."""
    if rate is None or trig_frequency <= 0:
        return None
    diff = abs(rate - trig_frequency)
    return rate if diff * 100 < trig_frequency or diff < 1 else None


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
        Trigger pulse rate in Hz, a whole number. The file carries no card
        rational (header bytes 88..96 are left zero), so a reader times it
        on this integer: a DAT cannot record a rate that is not a whole
        number of hertz — the Audace acquisition refuses DAT for such rates
        and records them in HDF5.
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

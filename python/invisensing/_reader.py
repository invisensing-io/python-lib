"""
Modern :class:`File` reader for Audace DAS acquisition files.

The public API is **stable** from v1.0.0 onwards. Anything not exported
via :data:`__all__` is internal and may change without notice.

Usage::

    from invisensing import File, Mode

    with File("phase/2026-11-03_10-00-00.h5") as f:
        if f.mode is Mode.PHASE:
            phase = f.get_phase()          # float32, rad
        elif f.mode is Mode.AMPLITUDE:
            amplitude = f.get_amplitude()
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
from . import _formats
from . import content as _content
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
    FLAG_RESERVED,
    FLAG_MOZART,
    FLAG_PHASE,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    FORMAT_VERSION,
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
    "FLAG_RESERVED",
    "FLAG_MOZART",
    "FLAG_PHASE",
    "FLAG_INTERLEAVED",
    "FLAG_UNSIGNED",
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "TIME_FLAG_MAP_V1",
    "TIME_FLAG_CORRECTED",
]


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_FS_PER_NS = 1_000_000

# Speed of light in the fibre (m/s): 2·10⁸, the one value used everywhere in
# Audace. It is the legacy C++ CardControl's ``LIGHT_SPEED_IN_FIBER``, and the
# producer (Cardcontrol) still turns the operator's metres into the card's
# sample count with it, so a file's distances only come back right with the
# same value.
_LIGHT_SPEED_IN_FIBER = 200_000_000  # m/s


# ── Mode enum ───────────────────────────────────────────────────────────────


class Mode(Enum):
    """
    What a file holds, and therefore which extractor reads it. Derived from
    the file's **content type** (:attr:`File.content_type`) — from its
    header flags, exactly as before, on a file written before the content
    type existed (``content_type == 0``) and on the few files written
    before data format v1 whose content type does not describe them
    (``Header.mode_from_flags``) — so a script can dispatch on it instead of
    bit-twiddling::

        mode = file.mode
        if mode is Mode.RAW:
            samples = file.read_lines(n)
        elif mode is Mode.PHASE:
            phase = file.get_phase()
        elif mode is Mode.AMPLITUDE:
            amp = file.get_amplitude()
        elif mode is Mode.ARCTAN:
            rad = file.get_arctan_radians()
        elif mode in (Mode.I, Mode.Q):
            column = file.read_lines(n)
        elif mode is Mode.IQ:
            i, q = file.get_i(), file.get_q()
        elif mode is Mode.ARCTAN_MAGNITUDE:
            a, m = file.get_arctan(), file.get_magnitude()
        elif mode is Mode.PHASE_AMPLITUDE:
            p, m = file.get_phase(), file.get_amplitude()

    Since data format v1 every recording holds one quantity per file: an
    interleaved stream is split into two files (``phase/`` and
    ``amplitude/``, ``arctan/`` and ``magnitude/``, ``i_component/`` and
    ``q_component/``; see :attr:`File.sibling_path`). The paired modes
    (``IQ``, ``ARCTAN_MAGNITUDE``, ``PHASE_AMPLITUDE``) are the files
    written before, which may still hold both.
    """

    #: Raw ADC samples (``int16``, content type ``raw``). On a file written
    #: before the content type existed, every single column the flags do
    #: not name otherwise. Read with :meth:`File.read_lines`.
    RAW = "raw"
    #: I/Q pairs ``[I, Q, I, Q, …]``, signed ``int16`` (a file written
    #: before data format v1). Use :meth:`File.get_i` / :meth:`File.get_q`
    #: / :meth:`File.get_iq`.
    IQ = "iq"
    #: Arctan/magnitude pairs ``[arctan(Q/I) (i16), √(I²+Q²) (u16), …]`` (a
    #: file written before data format v1). Use :meth:`File.get_arctan` /
    #: :meth:`File.get_magnitude`.
    ARCTAN_MAGNITUDE = "arctan_magnitude"
    #: One phase per position, ``float32`` radians — on-board or software
    #: demodulation (:attr:`File.content_type_name` names the chain). Use
    #: :meth:`File.get_phase` (or :meth:`File.read_lines`).
    PHASE = "phase"
    #: Phase/amplitude pairs ``[phase, amplitude, …]``, ``float32`` (a
    #: demodulation that kept the amplitude, in a file written before data
    #: format v1). Use :meth:`File.get_phase` / :meth:`File.get_amplitude`.
    PHASE_AMPLITUDE = "phase_amplitude"
    #: The card's wrapped ``arctan(Q/I)``, ``int16`` with ``32767 ↔ +π``
    #: (the ``arctan/`` file of a split arctan/magnitude recording). Use
    #: :meth:`File.get_arctan` / :meth:`File.get_arctan_radians`.
    ARCTAN = "arctan"
    #: An amplitude (magnitude): the card's ``√(I²+Q²)`` as ``uint16`` or
    #: ``float32``, or a software front end's (``sw_amplitude_sac``). Use
    #: :meth:`File.get_amplitude` / :meth:`File.get_magnitude`.
    AMPLITUDE = "amplitude"
    #: The card's I, ``int16`` (the ``i_component/`` file). Use :meth:`File.get_i`.
    I = "i"  # noqa: E741
    #: The card's Q, ``int16`` (the ``q_component/`` file). Use :meth:`File.get_q`.
    Q = "q"
    #: A script's output (Sandbox), typed by its content type (or, for
    #: ``sw_custom``, by its flags). Read with :meth:`File.read_lines`. A
    #: ``sw_custom`` file of a Sandbox older than data format v1 kept its
    #: input's flags and takes its mode from them instead (``PHASE``,
    #: ``IQ``, ``ARCTAN_MAGNITUDE`` or ``RAW``), as every reader before v1 did.
    CUSTOM = "custom"
    #: Zone variances of the console's archives (``viz_variance``,
    #: ``float32``, in the square of the source column's unit). Use
    #: :meth:`File.get_variance`.
    VARIANCE = "variance"
    #: The same variances on the console's Log scale, ``log10(1 +
    #: variance)`` (``viz_logvar``). Use :meth:`File.get_log_variance`, or
    #: :meth:`File.get_variance` for the variance itself.
    LOG_VARIANCE = "log_variance"

    @classmethod
    def from_flags(cls, flags: int) -> "Mode":
        """Decode the mode from a raw header flag mask — the rule of a file
        written before the content type existed (content type 0)."""
        if flags & FLAG_PHASE:
            return cls.PHASE
        if flags & FLAG_INTERLEAVED:
            if flags & FLAG_UNSIGNED:
                return cls.ARCTAN_MAGNITUDE
            return cls.IQ
        return cls.RAW

    @classmethod
    def from_header(cls, h: Header) -> "Mode":
        """The mode of a file of header ``h``: from its content type; from
        its flags (:meth:`from_flags`) for content type 0, a value this SDK
        does not know, a ``sample_size`` the content type contradicts, a
        ``sw_custom`` file of a Sandbox older than data format v1, which
        kept its input's flags, and a file written before v1 under a
        stream-only content type without the INTERLEAVED flag — an archive
        export of the console, which kept its source's header over one
        column per zone (``Header.mode_from_flags``)."""
        if h.mode_from_flags:
            return cls.from_flags(h.flags)
        ct = h.normalized_content_type
        info = _content.info(ct)
        if info.members is not None:
            return _PAIR_MODES[info.members[0].quantity]
        return _QUANTITY_MODES[info.quantity]


#: The mode of each single-column quantity of the registry.
_QUANTITY_MODES = {
    "raw": Mode.RAW,
    "phase": Mode.PHASE,
    "arctan": Mode.ARCTAN,
    "amplitude": Mode.AMPLITUDE,
    "i": Mode.I,
    "q": Mode.Q,
    "custom": Mode.CUSTOM,
    "variance": Mode.VARIANCE,
    "log_variance": Mode.LOG_VARIANCE,
}

#: The mode of each pair, by the quantity of its member 0.
_PAIR_MODES = {
    "i": Mode.IQ,
    "arctan": Mode.ARCTAN_MAGNITUDE,
    "phase": Mode.PHASE_AMPLITUDE,
}


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
    # Read at open from the backend (HDF5 / TDMS attributes, DAT size);
    # class-level defaults for the same reason as `_real_hz_attr`.
    _content_type_name_attr: Optional[str] = None
    _compression: Optional[str] = None
    _quantization_step_rad: Optional[float] = None
    _quantization_max_error_rad: Optional[float] = None
    _source_content_type: Optional[int] = None
    _trailing_bytes: int = 0

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
        # The writer's content-type tag (HDF5 / TDMS), the compression and
        # the lossy bound (HDF5), and the bytes past the last whole line
        # (DAT). Absent from a backend: unknown / none.
        self._content_type_name_attr = getattr(self._backend, "content_type_name_attr", None)
        self._compression = getattr(self._backend, "compression", None)
        self._quantization_step_rad = getattr(self._backend, "quantization_step_rad", None)
        self._quantization_max_error_rad = getattr(
            self._backend, "quantization_max_error_rad", None
        )
        self._source_content_type = getattr(self._backend, "source_content_type", None)
        self._trailing_bytes = int(getattr(self._backend, "trailing_bytes", 0) or 0)
        _warn_about_the_header(path, self._header)
        if self._trailing_bytes:
            warnings.warn(
                f"{path}: {self._trailing_bytes} byte(s) after the last whole line "
                f"(line {self._num_lines}) — the file was cut short in the middle "
                "of a line; that partial line is not read (File.trailing_bytes).",
                stacklevel=2,
            )

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
        """What the file holds — see :class:`Mode`: from the content type,
        or from the flags on a file written before it existed."""
        return Mode.from_header(self._header)

    @property
    def line_size(self) -> int:
        """Wire-side samples per pulse (doubled when ``is_interleaved``)."""
        return self._header.line_size

    @property
    def positions_per_line(self) -> int:
        """Spatial fibre positions per pulse — ``line_size / 2`` for a file of
        pairs (:attr:`is_interleaved`), ``line_size`` otherwise."""
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
        describe this file's :attr:`trig_frequency` (further from it than
        a card's own quantization puts it — one tick of ``trig_clock_hz``
        at 10 MHz, plus 1 Hz: left behind by a header whose frequency was
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
          integer 1.3.0 gave. A DAT whose header says
          :attr:`Header.time_corrected` starts at the corrected time of its
          first line and counts from there. TDMS and SEG-Y carry no absolute
          time (only the display string :attr:`timestamp`) and no time
          flags: ``header.acquisition_ns`` is ``0`` and the result is the
          offset from the first line.

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

        ``count`` is clamped to the lines the file holds: the vector never
        runs past :attr:`num_lines` (use :meth:`line_time_ns` to extrapolate
        beyond the last line). Raises :class:`MemoryError` when the vector
        cannot be allocated (a file that declares more lines than memory
        can time).
        """
        start = operator.index(start)
        if start < 0:
            raise ValueError(f"start must be >= 0 (got {start})")
        available = max(0, self._num_lines - start)
        if count is None:
            count = available
        count = operator.index(count)
        if count < 0:
            raise ValueError(f"count must be >= 0 (got {count})")
        count = min(count, available)
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
        say (files written before the card's time map, a DAT without the
        flag, and every TDMS and SEG-Y: neither format carries an absolute
        time or the time flags).

        * ``"cardclock-v1"``: from the card's time map — ``timestamp_ns``
          and :meth:`line_time_ns` are the **corrected** times (the card's
          oscillator drift and lost shots taken out); the count is
          ``line_time_ns(i, nominal=True)``. A DAT says so with
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
        """Total trigger pulses recorded in the file (whole lines only, see
        :attr:`trailing_bytes`)."""
        return self._num_lines

    @property
    def trailing_bytes(self) -> int:
        """Bytes after the last whole line of a ``.dat`` — ``0`` for a
        complete file. Non-zero means the file was cut short in the middle
        of a line (a recording stopped mid-write, a partial copy): those
        bytes are not read, and opening the file emits a warning. Always
        ``0`` for HDF5, TDMS and SEG-Y, which store whole lines."""
        return self._trailing_bytes

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
        """Acquisition start time as the file spells it — a string whose form
        depends on the format:

        * **DAT**: the header's anchor in nanoseconds since the Unix epoch
          (UTC), as a decimal string (``"1790000000000000000"``); a capture
          written before the binary anchor carries its free-form ASCII
          string instead.
        * **HDF5**: the ``timestamp`` attribute, the node's local time with
          its UTC offset, to the millisecond
          (``"2026-09-23 14:03:12.123+02:00"``) or the legacy string.
        * **TDMS**: the ``timestamp`` property, same form as HDF5.
        * **SEG-Y**: the ``TIMESTAMP`` line of the textual header, same form.

        To compute with, use ``header.acquisition_ns`` (the exact anchor, ns,
        UTC; ``0`` when the file has none — always for TDMS and SEG-Y, which
        carry only this display string) and :meth:`line_time_ns`.
        """
        return self._header.timestamp

    @property
    def content_type(self) -> int:
        """What the samples are: the header's ``content_type`` byte (``0`` on
        a capture written before the byte existed). See
        :attr:`content_type_name`."""
        return self._header.content_type

    @property
    def content_type_name(self) -> str:
        """The lowercase tag of :attr:`content_type` (``"hw_phase"``,
        ``"sw_mozart"``, ``"hw_amplitude"``…): ``"undefined"`` for a capture
        written before the byte existed (``0``). A value this SDK does not
        know (a newer producer) is named by the file's own tag when it
        carries one (HDF5 / TDMS ``content_type_name``), else
        ``"unknown(<n>)"``."""
        name = self._header.content_type_name
        if name.startswith("unknown(") and self._content_type_name_attr:
            return self._content_type_name_attr
        return name

    @property
    def card(self) -> Optional[str]:
        """The code of the acquisition card that recorded the file —
        ``"p18"``, ``"p21"``, ``"s125"`` or ``"s500"`` — or ``None`` when the
        file does not say. Every file of data format v1 carries it (DAT
        header, HDF5 / TDMS ``card``, SEG-Y ``CARD:`` line); before v1 only
        DAT headers did, as a tag read as the same code: every run's from
        2026-07-15 to 2026-10-13, then a two-card run's."""
        return self._header.card

    @property
    def format_version(self) -> int:
        """The data format version the file was written in: ``1`` for data
        format v1 (content type authoritative, steps and duration in the
        header), ``0`` for a file written before it."""
        return self._header.format_version

    @property
    def duration_s(self) -> Optional[int]:
        """The duration the writer recorded when it closed the file, in whole
        seconds rounded down — or ``None`` when the file does not say (a
        file written before data format v1, one cut short by a crash, one
        under a second). :attr:`duration` is the duration measured from the
        lines, to the nanosecond."""
        return self._header.duration_s or None

    @property
    def spatial_step_m(self) -> Optional[float]:
        """Metres of fibre between two adjacent positions (columns).

        The file's own value when it carries one (data format v1; the zone
        spacing of an archive export); else derived from the sample rate,
        ``1e8 / sample_rate`` (light at 2·10⁸ m/s in the fibre, halved for
        the round trip) for samples that are the acquisition's own; else
        ``None`` (a script's output, an archive export of an older file).
        """
        h = self._header
        step = h.spatial_step_m or h.derived_spatial_step_m
        return step or None

    @property
    def temporal_step_s(self) -> Optional[float]:
        """Seconds between two lines: the file's own value when it carries
        one (data format v1), else the line period of the header's cadence
        (``Header.derived_temporal_step_s``: ``trig_period_ticks /
        trig_clock_hz`` on the card's rational when the header carries a
        consistent one, else ``1 / trig_frequency``), else ``None``.
        Informative — times come from :meth:`line_time_ns`."""
        h = self._header
        step = h.temporal_step_s or h.derived_temporal_step_s
        return step or None

    @property
    def quantity(self) -> Optional[str]:
        """What the samples are, from the content type: ``"raw"``,
        ``"phase"``, ``"arctan"``, ``"amplitude"``, ``"i"``, ``"q"``,
        ``"custom"``, ``"variance"``, ``"log_variance"`` — or both members
        of a file of pairs (``"phase+amplitude"``). ``None`` when the
        content type does not say and :attr:`mode` comes from the flags
        (``Header.mode_from_flags``): content type 0, a value this SDK does
        not know, a file written before data format v1 whose content type
        does not describe it."""
        return _join_members(self._header, "quantity")

    @property
    def unit(self) -> Optional[str]:
        """The unit of the samples, from the content type: ``"rad"``,
        ``"count"`` (the card's integer scale), ``"adc_count"``,
        ``"front_end_count"``, ``"source_unit_squared"``,
        ``"dimensionless"``, ``"script"`` — or ``"rad+count"`` for a file of
        pairs; ``None`` when the content type does not say (see
        :attr:`quantity`)."""
        return _join_members(self._header, "unit")

    @property
    def source_content_type(self) -> Optional[int]:
        """An archive export of zone variances (content type 27 or 28, HDF5):
        the content type of the column the variance was computed from;
        ``None`` otherwise."""
        return self._source_content_type

    @property
    def sibling_path(self) -> Optional[Path]:
        """The other file of a split pair, when it is on disk: the
        ``amplitude/`` file of a ``phase/`` file (and back), ``magnitude/``
        of ``arctan/``, ``q_component/`` of ``i_component/``. Since data
        format v1 a recording writes each member of an interleaved stream to
        its own file, same name, sibling directories. ``None`` for any other
        file or when the sibling is missing."""
        return _sibling_path(self._path, self._header)

    @property
    def compression(self) -> Optional[str]:
        """The FileWriter ``compression`` an HDF5 was written with:
        ``"NONE"``, ``"ZSTD"`` (lossless) or ``"ZSTD_LOSSY_1E4"`` (phase
        rounded, see :attr:`quantization_step_rad`); a filter pipeline no
        Audace writer makes is named by its filters. ``None`` for DAT, TDMS
        and SEG-Y, which are never compressed."""
        return self._compression

    @property
    def quantization_step_rad(self) -> Optional[float]:
        """``ZSTD_LOSSY_1E4`` only: the grid every phase sample was rounded
        to, in radians (``2⁻¹³``), or ``None`` when the samples are exact
        (any other file)."""
        return self._quantization_step_rad

    @property
    def quantization_max_error_rad(self) -> Optional[float]:
        """``ZSTD_LOSSY_1E4`` only: the largest difference, in radians,
        between a stored phase sample and the acquired one (half the step,
        ≤ 6.1·10⁻⁵ rad), or ``None`` when the samples are exact."""
        return self._quantization_max_error_rad

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
        """Whether the samples are the Mozart software phase (Mozart or
        Mozart+), or ``None`` when the question does not apply.

        * A file that carries a content type (:attr:`content_type` ≠ 0) is
          answered by it alone: ``True`` for ``sw_mozart`` and
          ``sw_mozart_plus``, ``None`` for ``raw`` (no demodulation ran) and
          for a content type this SDK does not know, ``False`` for every
          other one.
        * A capture written before the content type existed (``0``) keeps
          the flag-based answer: ``None`` when the ``DEMODULATED`` bit is
          clear, else whether the ``MOZART`` bit is set. So do the files
          written before data format v1 whose content type does not
          describe them: a ``sw_custom`` file of an older Sandbox and an
          archive export of the console under a stream-only value without
          the INTERLEAVED flag (see :meth:`Mode.from_header`).

        ``False`` only says "not Mozart" — it names no other pipeline (the
        ``SCHUBERT`` flag is deprecated). Treat ``None`` as "not
        applicable", never as a default.
        """
        return self._header.is_mozart

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype :meth:`read_lines` returns: the content type's sample
        type (``int16``, ``uint16`` — a magnitude —, ``float32``); for a file
        written before the content type existed (and the files whose mode
        comes from the flags, see :meth:`Mode.from_header`),
        ``(sample_size, FLOAT)``, signed integers except a single-column
        ``UNSIGNED`` file (``uint16``) and 1-byte samples (``uint8``)."""
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
        """Length of fibre (m) one line covers, measured one way along the fibre.

        ``positions_per_line · spatial_step_m`` when the file carries its
        spatial step (data format v1; the zone spacing of an archive
        export); otherwise ``positions_per_line · c / (2 · sample_rate)``
        with ``c = 2·10⁸ m/s``: a sample spans ``1 / sample_rate`` of the
        pulse's round trip, hence the 2 — the result is the distance along
        the fibre, not the path the light travels. Both give the same
        length for the acquisition's own samples.
        """
        positions = max(self.positions_per_line, 1)
        step = self._header.spatial_step_m
        if step > 0 and step != self._header.derived_spatial_step_m:
            return positions * step
        return (_LIGHT_SPEED_IN_FIBER * positions) / (2 * max(self.sample_rate, 1))

    # ── Flag predicates (properties — bool-callable for legacy compat) ──

    # From the content type (data format v1 writes these flags as 0); from
    # the header flags, as before, on a file written before it existed.

    @property
    def is_demodulated(self) -> _Flag:
        """True if the samples come out of a demodulation, on board or in
        software (phase, arctan, amplitude, I or Q)."""
        return _Flag(self._header.is_demodulated)

    @property
    def is_interleaved(self) -> _Flag:
        """True if the file carries 2 samples per spatial position (a file
        of pairs written before data format v1)."""
        return _Flag(self._header.is_interleaved)

    @property
    def is_float(self) -> _Flag:
        """True if samples are IEEE-754 floats."""
        return _Flag(self._header.is_float)

    @property
    def is_phase(self) -> _Flag:
        """True if the file carries a phase in radians (alone, or as the
        first member of phase/amplitude pairs)."""
        return _Flag(self._header.is_phase)

    @property
    def is_unsigned(self) -> _Flag:
        """True if (some of) the integer samples are unsigned: a ``uint16``
        magnitude column, or the magnitude lane of arctan/magnitude pairs."""
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

    # ── Export ─────────────────────────────────────────────────────────

    def export_dat(
        self,
        path: Union[str, Path],
        *,
        start: int = 0,
        count: Optional[int] = None,
        chunk_lines: int = 65_536,
    ) -> None:
        """Write lines ``start .. start + count`` (default: to the end) to a
        ``.dat``, with this file's whole header: content type, card, steps,
        trigger pre-roll, the card's rational, time flags (see
        :func:`export_dat`). The first line's time is :meth:`line_time_ns`
        of ``start`` — the corrected time on a file stamped from the card's
        time map. The data is copied ``chunk_lines`` lines at a time; the
        read cursor is put back where it was.
        """
        start = operator.index(start)
        if not 0 <= start <= self._num_lines:
            raise ValueError(f"start must be within 0..{self._num_lines} (got {start})")
        available = self._num_lines - start
        count = available if count is None else min(operator.index(count), available)
        if count < 0:
            raise ValueError(f"count must be >= 0 (got {count})")
        backend = self._require_open()
        h = self._header
        if h.acquisition_ns:
            h = _formats.replace_header(h, acquisition_ns=self.line_time_ns(start))
        header = _file_header(h, self.dtype, self.line_size, count)
        cursor = self._num_lines - backend.lines_left

        def blocks():
            backend.seek(start)
            left = count
            while left > 0:
                block = backend.read_lines(min(chunk_lines, left))
                left -= block.shape[0]
                yield block

        try:
            _write_dat(path, header, blocks())
        finally:
            backend.seek(cursor)

    # ── Channel extractors ─────────────────────────────────────────────
    #
    # Each extractor accepts an optional pre-read array. Passing ``None``
    # reads ``n`` new pulses from the file; passing an array re-uses the
    # buffer the user already has (e.g. across multiple extractors in
    # the same loop iteration so the file cursor isn't double-advanced).
    #
    # A file of data format v1 holds one quantity: the extractor of that
    # quantity returns its column. A file of pairs (written before v1)
    # returns the member asked for.

    def get_i(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        The card's **I**: the ``i_component/`` file (:attr:`Mode.I`) or the
        I lane of I/Q pairs (:attr:`Mode.IQ`).

        Returns a ``(rows, positions_per_line)`` ``int16`` array.
        Raises :class:`ValueError` for any other mode.
        """
        return self._i_of(self._require_mode("get_i", Mode.I, Mode.IQ), data, n)

    def get_q(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """The card's **Q**: the ``q_component/`` file (:attr:`Mode.Q`) or
        the Q lane of I/Q pairs (:attr:`Mode.IQ`)."""
        return self._q_of(self._require_mode("get_q", Mode.Q, Mode.IQ), data, n)

    def get_iq(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Combined ``complex64`` view of I/Q pairs (:attr:`Mode.IQ`):
        ``result[r, j] = I[r, j] + j·Q[r, j]``. Returns
        ``(rows, positions_per_line)`` ``complex64``. A recording of data
        format v1 keeps I and Q in two files: read :meth:`get_i` from one and
        :meth:`get_q` from its :attr:`sibling_path`.
        """
        self._require_mode("get_iq", Mode.IQ)
        return _core.split_pair_to_complex_i16(self._ensure(data, n))

    def get_arctan(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        The card's wrapped **arctan(Q/I)**: the ``arctan/`` file
        (:attr:`Mode.ARCTAN`) or the arctan lane of arctan/magnitude pairs
        (:attr:`Mode.ARCTAN_MAGNITUDE`). Returns ``(rows,
        positions_per_line)`` ``int16`` — the fixed-point scale is
        ``32767 ↔ +π`` (:meth:`get_arctan_radians` converts).
        """
        mode = self._require_mode("get_arctan", Mode.ARCTAN, Mode.ARCTAN_MAGNITUDE)
        return self._arctan_of(mode, data, n)

    def get_amplitude(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        The **amplitude**: the column of an amplitude file
        (:attr:`Mode.AMPLITUDE` — the card's magnitude as ``uint16``
        (``magnitude/`` of an arctan/magnitude recording) or ``float32``
        (``amplitude/`` of a demodulation that kept it), or a software front
        end's ``float32``), or the amplitude member of a file of pairs
        (:attr:`Mode.ARCTAN_MAGNITUDE`: ``uint16``;
        :attr:`Mode.PHASE_AMPLITUDE`: ``float32``). Returns ``(rows,
        positions_per_line)``.
        """
        mode = self._require_mode("get_amplitude", *_AMPLITUDE_MODES)
        return self._amplitude_of(mode, data, n)

    def get_magnitude(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        The **√(I²+Q²)** magnitude — :meth:`get_amplitude` under the name
        the card's arctan/magnitude tap gives it: the ``uint16`` lane of
        arctan/magnitude pairs (the wire buffer is typed ``i16`` end to end
        but the card sends magnitudes as unsigned), or the column of an
        amplitude file (:attr:`Mode.AMPLITUDE`), or the amplitude member of
        phase/amplitude pairs.
        """
        mode = self._require_mode("get_magnitude", *_AMPLITUDE_MODES)
        return self._amplitude_of(mode, data, n)

    def get_phase(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Phase samples (``float32`` radians): the column of a phase file
        (:attr:`Mode.PHASE` — the card's phase or a software
        demodulation's, :attr:`content_type_name` says which) or the phase
        member of phase/amplitude pairs (:attr:`Mode.PHASE_AMPLITUDE`).
        Returns ``(rows, positions_per_line)`` ``float32``.

        On a phase file passing ``data`` returns it unchanged; passing
        nothing reads ``n`` new pulses.
        """
        if self._require_mode("get_phase", Mode.PHASE, Mode.PHASE_AMPLITUDE) is Mode.PHASE:
            return self._column(data, n)
        return _core.split_pair_f32_primary(self._f32_pairs(data, n))

    def get_variance(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Zone **variances** of a console archive export, ``float32``, in the
        square of the source column's unit (:attr:`source_content_type`):
        the column of a ``viz_variance`` file (:attr:`Mode.VARIANCE`), or a
        ``viz_logvar`` file (:attr:`Mode.LOG_VARIANCE`) converted back,
        ``10^v − 1``.
        """
        if self._require_mode("get_variance", Mode.VARIANCE, Mode.LOG_VARIANCE) is Mode.VARIANCE:
            return self._column(data, n)
        v = np.asarray(self._column(data, n), dtype=np.float64)
        return (np.power(10.0, v) - 1.0).astype(np.float32)

    def get_log_variance(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Zone variances on the console's **Log scale**, ``log10(1 +
        max(variance, 0))``, ``float32``: the column of a ``viz_logvar`` file
        (:attr:`Mode.LOG_VARIANCE`), or a ``viz_variance`` file
        (:attr:`Mode.VARIANCE`) put on that scale.
        """
        if self._require_mode("get_log_variance", Mode.LOG_VARIANCE, Mode.VARIANCE) is Mode.LOG_VARIANCE:
            return self._column(data, n)
        v = np.asarray(self._column(data, n), dtype=np.float64)
        return np.log10(1.0 + np.maximum(v, 0.0)).astype(np.float32)

    # ── Physical-unit extractors (float32, scaled to the underlying physics) ──
    #
    # I, Q and the card's magnitude are physically continuous voltages on
    # the photodetector, stored as fixed-point integers (i16 for I/Q, u16
    # for the magnitude). Arctan is a continuous angle stored as i16 with
    # the card's convention ``32767 ↔ +π``. The methods below return
    # ``float32`` numpy arrays in the natural physical unit.
    #
    # The volt scaling uses ``self.range`` (from the ADC voltage range
    # field of the header). The radian scaling for arctan uses
    # ``π / 32768`` — fixed by the card, not user-configurable.

    def get_i_volts(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> np.ndarray:
        """
        Like :meth:`get_i` but returns I as a ``float32`` numpy array scaled
        to **volts**: ``f32 = i16 × range / 32768``. Use this when you want
        physically-meaningful values without having to remember the ADC
        scaling.
        """
        mode = self._require_mode("get_i_volts", Mode.I, Mode.IQ)
        return _to_volts_i16(self._i_of(mode, data, n), self.range)

    def get_q_volts(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> np.ndarray:
        """Like :meth:`get_q` but returns Q as ``float32`` volts."""
        mode = self._require_mode("get_q_volts", Mode.Q, Mode.IQ)
        return _to_volts_i16(self._q_of(mode, data, n), self.range)

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
        Like :meth:`get_arctan` but returns the arctan as a ``float32`` numpy
        array in **radians**, scaled with the card's convention
        ``32767 ↔ +π`` (``f32 = i16 × π / 32768``). Range is ``[-π, +π]``,
        the raw wrapped output of the card's arctan2.
        """
        mode = self._require_mode("get_arctan_radians", Mode.ARCTAN, Mode.ARCTAN_MAGNITUDE)
        arr = self._arctan_of(mode, data, n)
        return arr.astype(np.float32) * np.float32(np.pi / 32768.0)

    def get_magnitude_volts(
        self, data: Optional[np.ndarray] = None, *, n: int = 1
    ) -> np.ndarray:
        """
        Like :meth:`get_magnitude` but returns the card's magnitude as a
        ``float32`` numpy array in **volts**: ``f32 = magnitude × range /
        32768``.

        The LSB scaling matches the I/Q lanes (the card computes
        ``√(I²+Q²)`` from the same i16 I/Q codes and keeps the same
        per-LSB voltage). The output is always ≥ 0. Only for the card's
        magnitude (unit ``count``): a software front end's amplitude
        (``sw_amplitude_sac``) is on another scale and is refused.
        """
        mode = self._require_mode("get_magnitude_volts", *_AMPLITUDE_MODES)
        if mode is not Mode.ARCTAN_MAGNITUDE and _amplitude_unit(self._header) != "count":
            raise ValueError(
                f"get_magnitude_volts: the amplitude of {self.content_type_name!r} is not "
                "the card's magnitude (it is not on the ADC's scale); use get_amplitude()"
            )
        arr = self._amplitude_of(mode, data, n)
        return arr.astype(np.float32) * np.float32(self.range / 32768.0)

    def channels(self, data: Optional[np.ndarray] = None, *, n: int = 1) -> dict:
        """
        Return *all* channels of the file in one call, as a dict whose
        keys depend on the :attr:`mode`::

            Mode.RAW              -> {"samples": ndarray}
            Mode.CUSTOM           -> {"samples": …}
            Mode.PHASE            -> {"phase": …}
            Mode.AMPLITUDE        -> {"amplitude": …}
            Mode.ARCTAN           -> {"arctan": …}
            Mode.I / Mode.Q       -> {"i": …} / {"q": …}
            Mode.VARIANCE         -> {"variance": …}
            Mode.LOG_VARIANCE     -> {"log_variance": …}
            Mode.IQ               -> {"i": …, "q": …, "iq": complex64}
            Mode.ARCTAN_MAGNITUDE -> {"arctan": …, "magnitude": …}
            Mode.PHASE_AMPLITUDE  -> {"phase": …, "amplitude": …}

        Convenience for ad-hoc scripts: pass a pre-read buffer if you
        already have one, otherwise the method reads ``n`` pulses.
        """
        m = self.mode
        buf = self._ensure(data, n)
        if m is Mode.IQ:
            return {
                "i": _core.split_pair_i16_primary(buf),
                "q": _core.split_pair_i16_secondary(buf),
                "iq": _core.split_pair_to_complex_i16(buf),
            }
        if m is Mode.ARCTAN_MAGNITUDE:
            return {
                "arctan": _core.split_pair_i16_primary(buf),
                "magnitude": _core.split_pair_unsigned(buf),
            }
        if m is Mode.PHASE_AMPLITUDE:
            pairs = self._f32_pairs(buf, n)
            return {
                "phase": _core.split_pair_f32_primary(pairs),
                "amplitude": _core.split_pair_f32_secondary(pairs),
            }
        key = {Mode.RAW: "samples", Mode.CUSTOM: "samples"}.get(m, m.value)
        return {key: buf}

    # ── Helpers ────────────────────────────────────────────────────────

    def _ensure(self, data: Optional[np.ndarray], n: int) -> np.ndarray:
        if data is None:
            return self.read_lines(n)
        return np.ascontiguousarray(data)

    def _column(self, data: Optional[np.ndarray], n: int) -> np.ndarray:
        """The single column of a file of one quantity, as read."""
        return self._ensure(data, n)

    # The member of a mode already checked by the public extractor.

    def _i_of(self, mode: Mode, data, n: int) -> np.ndarray:
        if mode is Mode.I:
            return self._column(data, n)
        return _core.split_pair_i16_primary(self._ensure(data, n))

    def _q_of(self, mode: Mode, data, n: int) -> np.ndarray:
        if mode is Mode.Q:
            return self._column(data, n)
        return _core.split_pair_i16_secondary(self._ensure(data, n))

    def _arctan_of(self, mode: Mode, data, n: int) -> np.ndarray:
        if mode is Mode.ARCTAN:
            return self._column(data, n)
        return _core.split_pair_i16_primary(self._ensure(data, n))

    def _amplitude_of(self, mode: Mode, data, n: int) -> np.ndarray:
        if mode is Mode.AMPLITUDE:
            return self._column(data, n)
        if mode is Mode.ARCTAN_MAGNITUDE:
            return _core.split_pair_unsigned(self._ensure(data, n))
        return _core.split_pair_f32_secondary(self._f32_pairs(data, n))

    def _f32_pairs(self, data: Optional[np.ndarray], n: int) -> np.ndarray:
        """Phase/amplitude pairs as ``float32`` (what the pair kernels take)."""
        return np.ascontiguousarray(self._ensure(data, n), dtype=np.float32)

    def _require_mode(self, method: str, *expected: Mode) -> Mode:
        mode = self.mode
        if mode in expected:
            return mode
        wanted = " or ".join(repr(m.value) for m in expected)
        raise ValueError(
            f"{method}: file mode is {mode.value!r}, but this extractor "
            f"only applies to {wanted}. "
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


# ── Content-type helpers ────────────────────────────────────────────────────

#: The modes :meth:`File.get_amplitude` reads.
_AMPLITUDE_MODES = (Mode.AMPLITUDE, Mode.ARCTAN_MAGNITUDE, Mode.PHASE_AMPLITUDE)


def _warn_about_the_header(path: Path, h: Header) -> None:
    """Say so, once per file opened, when the header cannot be read as what
    it says: a content type this SDK does not know (its samples are decoded
    from ``sample_size`` and the flags, its quantity is unknown), a
    ``sample_size`` the content type contradicts (read from the flags), a
    newer data format version (the fields this SDK knows are read)."""
    if h.content_type != 0 and _content.info(h.content_type) is None:
        warnings.warn(
            f"{path}: content type {h.content_type} is not known to this "
            "invisensing (a newer producer?): its samples are decoded from "
            "sample_size and the flags, and nothing is assumed about what "
            "they are. Upgrade invisensing.",
            stacklevel=3,
        )
    elif h.sample_size_contradicts_content_type:
        warnings.warn(
            f"{path}: content type {h.content_type_name} ({h.content_type}) "
            f"carries {_content.sample_size(h.normalized_content_type)}-byte "
            f"samples, but the header says {h.sample_size}: read from the flags",
            stacklevel=3,
        )
    if h.format_version > FORMAT_VERSION:
        warnings.warn(
            f"{path}: data format version {h.format_version} is newer than "
            f"this invisensing reads in full ({FORMAT_VERSION}): the fields it "
            "knows are read. Upgrade invisensing.",
            stacklevel=3,
        )


def _join_members(h: Header, field: str) -> Optional[str]:
    """A registry column (``quantity`` / ``unit``) of a header's content
    type: the value, or ``m0+m1`` for a pair; ``None`` when the content type
    does not say what the samples are and the mode comes from the flags
    (``Header.mode_from_flags``: 0, unknown, contradicted, a pre-v1
    ``sw_custom`` file, a pre-v1 unpaired file of a stream-only value)."""
    ct = h.normalized_content_type
    info = _content.info(ct)
    if h.mode_from_flags:
        return None
    if info.members is not None:
        return "+".join(getattr(m, field) for m in info.members)
    return getattr(info, field)


def _amplitude_unit(h: Header) -> Optional[str]:
    """The unit of a header's amplitude (the column, or the amplitude member
    of a pair); ``"count"`` for a file described by its flags
    (``Header.mode_from_flags``: written before the content type existed,
    or before v1 under a content type that does not describe it — always
    the card's magnitude)."""
    ct = h.normalized_content_type
    info = _content.info(ct)
    if h.mode_from_flags:
        return "count"
    if info.members is not None:
        k = _content.member_of(ct, "amplitude")
        return None if k is None else info.members[k[0]].unit
    return info.unit


#: Extensions a recording's sibling may carry (an HDF5 written before data
#: format v1 is ``.hdf5``, after it ``.h5``).
_SIBLING_SUFFIXES = {".h5": (".h5", ".hdf5"), ".hdf5": (".hdf5", ".h5")}


def _sibling_path(path: Path, h: Header) -> Optional[Path]:
    """The other file of the split pair ``path`` belongs to (see
    :attr:`File.sibling_path`)."""
    ct = h.normalized_content_type
    here = path.parent.name
    for info in _content.REGISTRY.values():
        for k, member in enumerate(info.members or ()):
            if member.child != ct or member.subdir != here:
                continue
            other = info.members[1 - k].subdir
            for suffix in _SIBLING_SUFFIXES.get(path.suffix.lower(), (path.suffix,)):
                candidate = path.parent.parent / other / (path.stem + suffix)
                if candidate.is_file():
                    return candidate
            return None
    return None


# ── dtype helper ────────────────────────────────────────────────────────────


def _dtype_for(h: Header) -> np.dtype:
    """numpy dtype the readers return for header ``h`` (see
    :func:`invisensing._formats.wire_dtype`, the one rule)."""
    from ._formats import wire_dtype

    return wire_dtype(h)


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


#: The highest trigger frequency an Audace header carries, Hz.
_MAX_TRIG_FREQUENCY_HZ = 10_000_000
#: The slowest clock a card counts its trigger period in: 125 MHz (the
#: other cards count in 250 MHz).
_SLOWEST_TRIGGER_CLOCK_HZ = 125_000_000


def _consistent_rate(rate: Optional[float], trig_frequency: int) -> Optional[float]:
    """``rate`` when it can be the real rate of a recording requested at
    ``trig_frequency`` (``f``), else ``None``.

    The guard the header's rational goes through (``consistent_rational`` in
    the native core, ``TriggerTiming::consistent_with`` in the acquisition):
    ``|rate − f| < f / q + 1`` Hz, ``q`` being the period, in ticks, of the
    fastest trigger (10 MHz) on the card's clock. A rate stored alone
    carries no clock, so ``q`` is that of the slowest trigger clock of any
    card (125 MHz: ``q = 12``, a card's floor divider is off by less than
    1/12, 8.3 %), which keeps every rate a card can fire at."""
    if rate is None or trig_frequency <= 0:
        return None
    q = _SLOWEST_TRIGGER_CLOCK_HZ // _MAX_TRIG_FREQUENCY_HZ
    diff = abs(rate - trig_frequency)
    return rate if diff * q < trig_frequency + q else None


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

#: The flags a file of data format v1 keeps (AC and HIZ: acquisition
#: hardware facts); everything the content type says is written 0.
_FILE_KEPT_FLAGS = FLAG_AC | FLAG_HIZ

#: numpy dtype of each sample type the registry fixes.
_SAMPLE_DTYPES = {"i16": np.dtype("int16"), "u16": np.dtype("uint16"), "f32": np.dtype("float32")}


def _file_header(h: Header, dtype: np.dtype, line_size: int, lines: int) -> Header:
    """The header a DAT of ``lines`` lines of ``line_size`` samples of
    ``dtype`` is written with, from header ``h`` — the
    acquisition's rule for a conversion (``to_file_header``):

    * a header whose content type is a single-column value of the
      registry becomes a **data format v1** header: ``format_version`` 1,
      the content type (an ``hw_amplitude`` with 4-byte samples, an older
      demodulation's, is ``hw_amplitude_f32``), the file flags (AC and HIZ;
      FLOAT and UNSIGNED too for ``sw_custom``, which they type), the steps
      kept or derived, the card code, no stream-only time fields, and
      ``duration_s`` of the lines written. ``dtype`` must be the content
      type's sample type;
    * any other header — content type 0, pairs (``sw_custom`` with
      INTERLEAVED too), a value this SDK does not know or whose sample size
      it contradicts — is written as a **legacy copy**: content type, flags
      (FLOAT set from ``dtype``, as :func:`export_dat` always did), version,
      steps, card and duration as they are.
    """
    if not h.trig_frequency_is_exact:
        raise ValueError(
            f"the card fired at {h.trig_frequency_real_hz:.4f} Hz, not the "
            f"{h.trig_frequency} Hz requested: a DAT never carries an inexact "
            "trigger frequency (the acquisition refuses it too) — keep this "
            "recording in HDF5"
        )
    dtype = np.dtype(dtype)
    geometry = dict(line_size=int(line_size), sample_size=int(dtype.itemsize))
    ct = h.normalized_content_type
    info = _content.info(ct)
    typed = (
        ct != 0
        and info is not None
        and info.members is None
        and not h.sample_size_contradicts_content_type
        # Pairs under 12 (an older Sandbox's): no v1 file holds them.
        and not h.is_interleaved
    )
    if not typed:
        flags = h.flags | FLAG_FLOAT if dtype.kind == "f" else h.flags & ~FLAG_FLOAT
        return _formats.replace_header(h, flags=flags, **geometry)
    if info.sample_type == "from_flags":
        # sw_custom: typed by its FLOAT / UNSIGNED flags, from the array.
        flags = h.flags & _FILE_KEPT_FLAGS
        if dtype.kind == "f":
            flags |= FLAG_FLOAT
        elif dtype.kind == "u":
            flags |= FLAG_UNSIGNED
    else:
        want = _SAMPLE_DTYPES[info.sample_type]
        if dtype != want:
            raise ValueError(
                f"content type {info.tag} ({ct}) is stored as {want}, "
                f"but the data is {dtype}"
            )
        flags = h.flags & _FILE_KEPT_FLAGS
    out = _formats.replace_header(
        h,
        content_type=ct,
        flags=flags,
        format_version=FORMAT_VERSION,
        spatial_step_m=h.spatial_step_m or h.derived_spatial_step_m,
        temporal_step_s=h.temporal_step_s or h.derived_temporal_step_s,
        line_offset=0,
        time_flags=h.time_flags & ~TIME_FLAG_MAP_V1,
        **geometry,
    )
    return _formats.replace_header(out, duration_s=_duration_s(out, int(lines)))


def _duration_s(h: Header, lines: int) -> int:
    """Whole seconds of ``lines`` lines, rounded down, saturated to u32 —
    exact on the cadence (``Header.lines_to_ns``); from ``temporal_step_s``
    without one; 0 when neither is known."""
    if h.line_timing is not None:
        secs = max(h.lines_to_ns(lines), 0) // 1_000_000_000
    elif h.temporal_step_s > 0:
        secs = int(lines * h.temporal_step_s)
    else:
        secs = 0
    return min(secs, 0xFFFF_FFFF)


def _write_dat(path: Union[str, Path], header: Header, blocks) -> None:
    """Write a DAT: the 128-byte header, then each block of lines,
    little-endian."""
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        for block in blocks:
            arr = np.ascontiguousarray(block)
            fp.write(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())


def export_dat(
    path: Union[str, Path],
    data: np.ndarray,
    *,
    sample_rate: Optional[int] = None,
    trig_frequency: Optional[int] = None,
    range_v: Optional[float] = None,
    timestamp: Optional[str] = None,
    flags: Optional[int] = None,
    pulse_width: Optional[int] = None,
    num_channels: Optional[int] = None,
    header: Optional[Header] = None,
    content_type: Optional[int] = None,
    card: Optional[str] = None,
    acquisition_ns: Optional[int] = None,
) -> None:
    """
    Write a 2-D numpy array out as an Audace ``.dat`` file (128-byte
    header + raw samples). The array's ``shape[1]`` becomes
    ``line_size`` and its dtype ``sample_size``.

    Two ways to describe the samples:

    * ``header=`` (e.g. ``f.header`` of the file the data was read from):
      **every field** is copied — timestamp, card, trigger pre-roll, the
      card's trigger rational, time flags, content type, steps — as the
      acquisition's conversions do. A content type of the registry gives a
      file of data format v1 (``format_version`` 1, the file flags, the
      steps filled, ``duration_s`` of the lines written; ``data`` must be of
      the content type's sample type: ``f.read_lines()`` of a v1 file is);
      content type 0, pairs and unknown values give a legacy copy, as they
      are. :meth:`File.export_dat` does this for a range of a file's lines.
    * the keyword fields below, as in earlier versions: a file of content
      type 0 described by ``flags``, or, with ``content_type=``, a file of
      data format v1.

    Keyword fields given with ``header=`` override its values. As in earlier
    versions, every integer field goes through ``int()`` (``250e6``, a
    numpy scalar, ``100.0`` are taken; a fraction is dropped) and
    ``timestamp`` and ``card`` through ``str()``.

    Parameters
    ----------
    path
        Output file path. Overwritten if it exists.
    data
        ``(num_lines, line_size)`` 2-D array. Any
        contiguous-or-not C-order layout is accepted; the data is
        densified on write.
    sample_rate
        Rate of the positions along the fibre in Hz (required without
        ``header``).
    trig_frequency
        Trigger pulse rate in Hz, a whole number (required without
        ``header``). Without ``header`` the file carries no card rational
        (header bytes 88..96 are left zero), so a reader times it on this
        integer: a DAT cannot record a rate that is not a whole number of
        hertz — the Audace acquisition refuses DAT for such rates and
        records them in HDF5.
    range_v
        ADC voltage range in volts (stored as mV in the header).
    timestamp
        Free-form acquisition timestamp (≤ 32 bytes UTF-8), for a file
        without ``acquisition_ns``.
    flags
        Raw header flag mask. On a file of content type 0 the ``FLOAT`` bit
        is set automatically when ``data.dtype.kind == 'f'``; a file of data
        format v1 keeps only ``AC`` and ``HIZ``.
    pulse_width
        Trigger pulse width in ns.
    num_channels
        Number of active socket channels in the original acquisition.
    header
        A :class:`Header` to copy (see above).
    content_type
        The content type of the samples (see :mod:`invisensing.content`).
    card
        The card code (``"p18"``, ``"p21"``, ``"s125"``, ``"s500"``).
    acquisition_ns
        Time of the first line, ns since the Unix epoch (UTC): with
        ``header=``, the time of the first line written when it is not the
        source's first (``f.line_time_ns(start)``).

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, if a required field is missing, or if the
        data is not of the content type's sample type.
    OSError
        If the write fails.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (rows × line_size), got {data.ndim}-D")

    def as_int(value):
        # As 1.3.0 did: ``int(...)`` takes ``250e6``, numpy scalars and
        # ``100.0`` (and truncates a fraction), which ``Header`` refuses.
        return None if value is None else int(value)

    overrides = {
        "sample_rate": as_int(sample_rate),
        "trig_frequency": as_int(trig_frequency),
        "range": None if range_v is None else int(round(float(range_v) * 1000)),
        "timestamp": None if timestamp is None else str(timestamp),
        "flags": as_int(flags),
        "pulse_width": as_int(pulse_width),
        "num_channels": as_int(num_channels),
        "content_type": as_int(content_type),
        "card": None if card is None else str(card),
        "acquisition_ns": as_int(acquisition_ns),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    content_type = overrides.get("content_type")
    if header is None:
        if sample_rate is None or trig_frequency is None:
            raise ValueError("export_dat needs sample_rate and trig_frequency (or header=)")
        info = _content.info(content_type) if content_type else None
        want = _SAMPLE_DTYPES.get(info.sample_type) if info is not None else None
        if want is not None and data.dtype != want:
            raise ValueError(
                f"content type {info.tag} ({content_type}) is stored as {want}, "
                f"but the data is {data.dtype}"
            )
        defaults = dict(range=0, timestamp="", flags=0, pulse_width=0, num_channels=1)
        header = Header(
            line_size=int(data.shape[1]),
            sample_size=int(data.dtype.itemsize),
            **{**defaults, **overrides},
        )
    elif overrides:
        header = _formats.replace_header(header, **overrides)
    _write_dat(path, _file_header(header, data.dtype, data.shape[1], data.shape[0]), [data])

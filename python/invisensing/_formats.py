"""
Per-format backends behind :class:`invisensing.File`.

Each backend wraps a third-party reader (h5py / npTDMS / segyio) for the
metadata + bytes-to-numpy step, then funnels the result through the same
:class:`~invisensing._core.Header` object so the channel extractors are
format-agnostic.

The DAT backend skips third-party libs entirely — the Rust core's
:class:`~invisensing._core.DatReader` is the fast path for what is by
far the most common file in an Audace deployment.

Every backend exposes the same lifecycle contract:

* ``close()`` releases the underlying OS handle, is **idempotent**, and
  never raises — calling it twice is a no-op.
* after ``close()`` any read (``read_lines`` / ``rewind`` / ``seek`` /
  ``lines_left``, the cursor of a handle that is gone) raises
  :class:`ValueError`, never ``AttributeError`` and never a segfault
  from the native side.
* ``header`` and ``num_lines`` are captured at construction and keep
  answering after ``close()``: they cost one descriptor, once, at open.
  :class:`invisensing.File` copies them at open and serves its metadata
  from that copy, so a closed file still describes itself.
* ``__del__`` only delegates to ``close()`` as a last-resort safety net;
  it is not the intended release path (see :class:`invisensing.File`).

Arrays handed back by ``read_lines`` **own their memory** — none of the
backends returns a view onto the still-open file — so a caller may keep
reading its numpy arrays after the file has been closed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from ._core import (
    CONTENT_TYPE_NAMES,
    DatReader,
    Header,
    FLAG_FLOAT,
    TIME_FLAG_CORRECTED,
    card_code,
)


#: Message of the :class:`ValueError` every backend raises when it is
#: read after ``close()``.
_CLOSED_MSG = "operation on a closed file"

#: ``content_type`` value of each tag this SDK knows (the inverse of
#: ``_core.CONTENT_TYPE_NAMES``).
_CONTENT_TYPE_BY_NAME = {name: value for value, name in CONTENT_TYPE_NAMES.items()}


#: Every field of :class:`Header`, in constructor order.
HEADER_FIELDS = (
    "line_size", "trig_frequency", "sample_size", "sample_rate", "flags",
    "range", "pulse_width", "num_channels", "timestamp", "content_type",
    "acquisition_ns", "trigger_delay", "trig_clock_hz", "trig_period_ticks",
    "line_offset", "time_flags", "card", "duration_s", "format_version",
    "spatial_step_m", "temporal_step_s",
)


def replace_header(h: Header, **changes) -> Header:
    """A copy of ``h`` with ``changes`` applied (``Header`` is immutable)."""
    unknown = set(changes) - set(HEADER_FIELDS)
    if unknown:
        raise TypeError(f"unknown Header field(s): {sorted(unknown)}")
    fields = {name: getattr(h, name) for name in HEADER_FIELDS}
    fields.update(changes)
    return Header(**fields)


def wire_dtype(h: Header) -> np.dtype:
    """The numpy dtype :meth:`read_lines` returns for a file of header ``h``
    — the native DAT reader's rule (``Header.sample_encoding``), which every
    backend follows:

    * the content type's sample type when it fixes one: ``int16``,
      ``uint16`` (5, 25: a magnitude, a script's ``u16``) or ``float32`` —
      a v1 file carries no FLOAT / UNSIGNED flag;
    * content type 0, 12, an unknown value, or a ``sample_size`` the
      content type contradicts — the flags, exactly as before: ``FLOAT``:
      ``float32`` / ``float64``; a single column with ``UNSIGNED``: the
      unsigned integer of ``sample_size``; otherwise the signed integer;
    * 1 byte: ``uint8``.

    An interleaved pair stays signed: the ``u16`` magnitude lane of an
    arctan/magnitude pair is reinterpreted by :meth:`File.get_magnitude`.
    """
    size, is_float, unsigned = h.sample_encoding
    if is_float and size in (4, 8):
        return np.dtype(f"float{size * 8}")
    if size == 1:
        return np.dtype("uint8")
    if unsigned:
        return np.dtype(f"uint{size * 8}")
    return np.dtype(f"int{size * 8}")


def _as_wire_dtype(arr: np.ndarray, h: Header) -> np.ndarray:
    """``arr`` with the dtype :func:`wire_dtype` names, when it holds the same
    integers under the other signedness (an HDF5 written ``int16`` with
    ``unsigned = 1`` before writers stored ``uint16``, a TDMS channel that is
    always ``int16``): the bits are kept, reinterpreted. Any other array is
    returned unchanged."""
    want = wire_dtype(h)
    have = arr.dtype
    if (
        have != want
        and have.kind in "iu"
        and want.kind in "iu"
        and have.itemsize == want.itemsize
    ):
        return arr.view(want)
    return arr


def _card(value) -> Optional[str]:
    """The card code a file's ``card`` attribute / property / text line
    stands for (a tag written before data format v1 is mapped), or
    ``None``."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if not isinstance(value, str):
        return None
    return card_code(value.strip())


def _step(value: Optional[float]) -> float:
    """A step attribute for ``Header``: the value :func:`_positive_float_attr`
    returned (which keeps only a finite, strictly positive number), or
    ``0.0`` (absent) for its ``None``. It filters nothing itself."""
    return value if value is not None else 0.0


def _content_type(number: Optional[int], name: Optional[str]) -> int:
    """The ``content_type`` byte from a file's number and tag attributes:
    the number when there is one (it is the canonical value), else the value
    of a tag this SDK knows, else ``0`` (a file written before either
    existed)."""
    if number is not None and 0 <= number <= 0xFF:
        return number
    if name is not None:
        return _CONTENT_TYPE_BY_NAME.get(name, 0)
    return 0


# ── DAT — native Rust ──────────────────────────────────────────────────────


class _DatBackend:
    """Wraps the Rust DatReader so the same `File` facade applies."""

    def __init__(self, path: Path):
        self._inner = DatReader(str(path))
        # Cached at construction so metadata stays answerable after
        # close() — the file is gone, the numbers describing it are not.
        self._header = self._inner.header
        self._num_lines = int(self._inner.num_lines)
        # Bytes past the last whole line (a file cut short); `File` warns.
        self.trailing_bytes = int(getattr(self._inner, "trailing_bytes", 0))

    def _require_open(self):
        inner = getattr(self, "_inner", None)
        if inner is None:
            raise ValueError(_CLOSED_MSG)
        return inner

    @property
    def header(self) -> Header:
        return self._header

    @property
    def num_lines(self) -> int:
        return self._num_lines

    @property
    def lines_left(self) -> int:
        # Cursor state, not metadata: it only means something while the
        # handle exists, so it raises once the file is closed.
        return int(self._require_open().lines_left)

    def read_lines(self, n: int) -> np.ndarray:
        return self._require_open().read_lines(n)

    def rewind(self) -> None:
        self._require_open().rewind()

    def seek(self, n: int) -> None:
        self._require_open().seek_lines(int(n))

    def close(self) -> None:
        """Close the Rust reader. Idempotent, never raises.

        It does not *swallow* either. Two different things used to land
        in the same blanket ``except``:

        * ``DatReader`` has no ``close`` — an extension built before the
          method existed. Dropping our reference is then the whole
          release path, and it has just happened. Nothing to report.
        * ``close()`` exists and failed. The one way that happens is a
          concurrent ``read_lines`` on the same reader: PyO3 holds a
          mutable borrow of the pyclass for the length of the read (the
          GIL is released inside it, so another thread really can get
          here) and raises ``RuntimeError('Already borrowed')``. The
          reader survives, the descriptor is released when that read
          finishes — later, not now. Silence there is what makes
          "``closed`` is True but the file is still open on the NAS"
          impossible to diagnose, so we warn instead.

        Either way we never raise: callers close in ``finally`` blocks
        and in ``__del__``.
        """
        inner = getattr(self, "_inner", None)
        # Drop our reference FIRST: the close is then idempotent even if
        # the native call below were to fail.
        self._inner = None
        if inner is None:
            return
        close_fn = getattr(inner, "close", None)
        if close_fn is None:
            return
        try:
            close_fn()
        except Exception as exc:
            # `warnings` may already be None during interpreter shutdown.
            try:
                warnings.warn(
                    f"invisensing: the native DatReader.close() failed "
                    f"({exc!r}); the file descriptor will only be released "
                    "when the in-flight read finishes. Do not close a File "
                    "while another thread is reading it.",
                    ResourceWarning,
                    stacklevel=2,
                )
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── HDF5 — via h5py ────────────────────────────────────────────────────────
#
# Audace HDF5 files (``.h5`` since data format v1, ``.hdf5`` before) store
# the samples as a (num_traces, samples_per_trace) dataset named
# ``acoustic_data``, typed by the content type (``int16``, ``uint16`` or
# ``float32``; for content type 0 and 12, by ``sample_size`` and the
# ``FLOAT`` / ``UNSIGNED`` flags). All header fields are mirrored as scalar
# attributes. Only the metadata is read at
# construction; the samples are fetched per ``read_lines`` slice, so a
# multi-GB capture does not have to fit in RAM. Each slice is a fresh,
# owning numpy array — h5py materialises it on the spot — so closing
# the file does not invalidate arrays already handed to the caller.


_ZSTD_FILTER_ID = "32015"

_U32_MAX = 0xFFFF_FFFF
_I64_MAX = 0x7FFF_FFFF_FFFF_FFFF


def _int_attr(attrs, name: str, lo: int, hi: int) -> int:
    """An integer attribute within ``[lo, hi]``, or ``0`` when it is absent,
    not an integer, or out of range.

    For the optional attributes whose ``0`` means "unknown": a value the
    native :class:`Header` cannot hold (a negative clock, a period past
    ``u32``) or a non-integer is the same "unknown", not a reason to refuse
    a recording whose samples are fine.
    """
    value = attrs.get(name)
    if value is None:
        return 0
    try:
        arr = np.asarray(value)
        if arr.size != 1 or arr.dtype.kind not in "iu":
            return 0
        v = int(arr.reshape(()))
    except (TypeError, ValueError):
        return 0
    return v if lo <= v <= hi else 0


def _positive_float_attr(attrs, name: str) -> Optional[float]:
    """A finite, strictly positive numeric attribute, or ``None``."""
    value = attrs.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size != 1 or arr.dtype.kind not in "iuf":
            return None
        v = float(arr.reshape(()))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _float_attr(attrs, name: str) -> Optional[float]:
    """A finite numeric scalar attribute (any sign), or ``None``."""
    value = attrs.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size != 1 or arr.dtype.kind not in "iuf":
            return None
        v = float(arr.reshape(()))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _opt_int_attr(attrs, name: str) -> Optional[int]:
    """An integer scalar attribute as a Python ``int`` (exact), or ``None``."""
    value = attrs.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size != 1 or arr.dtype.kind not in "iu":
            return None
        return int(arr.reshape(()))
    except (TypeError, ValueError):
        return None


def _str_attr(attrs, name: str) -> Optional[str]:
    """A string attribute (h5py hands variable-length UTF-8 back as ``str``
    or ``bytes`` depending on its version), or ``None``."""
    value = attrs.get(name)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else None


def _int_list_attr(attrs, name: str) -> Optional[list]:
    """A 1-D integer array attribute as a list of exact Python ints."""
    value = attrs.get(name)
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.dtype.kind not in "iu":
        return None
    return [int(v) for v in arr.reshape(-1).tolist()]


#: ``time_correction`` of a file timed by the card's time map.
CORRECTION_CARDCLOCK_V1 = "cardclock-v1"
#: ``time_correction`` of a file timed by the count (no time map).
CORRECTION_NONE = "none"

#: The longest line period a time-axis node may carry, fs (1 000 s; a card's
#: is 5 µs to 1 ms) — ``audace::time_map::MAX_PERIOD_FS``. An axis beyond it
#: is forged or corrupt, and not trusted.
_MAX_PERIOD_FS = 10**18


class _TimeAxis:
    """What an HDF5 says about its time (Audace FileWriter >= the card time
    map): the attributes written by ``audace::formats::write_time_attrs``.

    ``lines`` / ``ns`` / ``period_fs`` are the axis's nodes (attributes
    ``time_map_line`` / ``time_map_ns`` / ``time_map_period_fs``): from node
    ``k`` on, the time of file line ``i`` is ``ns[k] + ⌊(i − lines[k]) ·
    period_fs[k] / 10⁶⌋`` — or, for a period 0, the file's own cadence. The
    first node may sit before line 0 (the segment the file starts in). Empty
    when the file carries no axis.
    """

    __slots__ = (
        "correction", "lines", "ns", "period_fs", "start_ns", "nominal_start_ns",
        "last_ns", "measured_hz", "error_ppm", "uncertainty_ns", "prior_ppm",
        "state", "synchronized", "missing_shots",
    )

    def __init__(self, attrs):
        self.correction = _str_attr(attrs, "time_correction")
        lines = _int_list_attr(attrs, "time_map_line")
        ns = _int_list_attr(attrs, "time_map_ns")
        period = _int_list_attr(attrs, "time_map_period_fs")
        ok = (
            lines is not None and ns is not None and period is not None
            and 0 < len(lines) == len(ns) == len(period)
            and all(a < b for a, b in zip(lines, lines[1:]))
            and all(0 <= p <= _MAX_PERIOD_FS for p in period)
        )
        self.lines = lines if ok else []
        self.ns = ns if ok else []
        self.period_fs = period if ok else []
        self.start_ns = _opt_int_attr(attrs, "timestamp_ns")
        self.nominal_start_ns = _opt_int_attr(attrs, "timestamp_nominal_ns")
        self.last_ns = _opt_int_attr(attrs, "timestamp_last_ns")
        self.measured_hz = _positive_float_attr(attrs, "trig_frequency_measured_hz")
        self.error_ppm = _float_attr(attrs, "trig_frequency_error_ppm")
        self.uncertainty_ns = _opt_int_attr(attrs, "time_uncertainty_ns")
        self.prior_ppm = _float_attr(attrs, "time_prior_ppm")
        self.state = _str_attr(attrs, "time_correction_state")
        sync = _opt_int_attr(attrs, "time_reference_synchronized")
        self.synchronized = None if sync is None else bool(sync)
        self.missing_shots = _opt_int_attr(attrs, "missing_shots") or 0

    @property
    def corrected(self) -> bool:
        """The file is timed by the card's time map and carries its axis."""
        return self.correction == CORRECTION_CARDCLOCK_V1 and bool(self.lines)


def _hdf5_compression(filters: set, quantization_step_rad: Optional[float]) -> str:
    """The FileWriter ``compression`` setting an HDF5 was written with, from
    its dataset's filter pipeline: ``"NONE"`` (no filter), ``"ZSTD"`` (zstd,
    filter 32015, lossless) or ``"ZSTD_LOSSY_1E4"`` (the same pipeline with
    the phase rounded, said by ``quantization_step_rad``). A pipeline no
    Audace writer makes is reported by its filter names (``"gzip+shuffle"``).
    """
    if _ZSTD_FILTER_ID in filters:
        return "ZSTD_LOSSY_1E4" if quantization_step_rad is not None else "ZSTD"
    if not filters:
        return "NONE"
    return "+".join(sorted(filters))


def _read_time_axis(attrs) -> Optional[_TimeAxis]:
    """The file's time attributes, ``None`` for a file that has none
    (written before them, or converted from such a file)."""
    if _str_attr(attrs, "time_correction") not in (CORRECTION_NONE, CORRECTION_CARDCLOCK_V1):
        return None
    return _TimeAxis(attrs)


def _filter_ids(dataset) -> set:
    """Filter names / ids of an h5py dataset's pipeline, as strings (a filter
    h5py has no name for, such as zstd, appears as its numeric id)."""
    try:
        return {str(k) for k in dataset._filters}
    except Exception:
        return set()


class _Hdf5Backend:
    def __init__(self, path: Path):
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            # `h5py` is part of the default install — only hit this branch
            # if the user did `pip install --no-deps`. Tell them how to
            # recover without hiding what actually went wrong.
            raise ImportError(
                "h5py is missing. It is normally installed automatically with "
                "`pip install invisensing`. If you installed with --no-deps, "
                "run `pip install h5py` (or `pip install invisensing[hdf5]`)."
            ) from exc
        # Recordings written with the FileWriter's `compression = ZSTD` store
        # `acoustic_data` through the registered HDF5 zstd filter (32015),
        # which `hdf5plugin` registers with h5py when imported. It is a
        # dependency of this package; if it is missing anyway, such a file
        # must say what to install instead of failing deep inside libhdf5.
        try:
            import hdf5plugin  # type: ignore  # noqa: F401
        except ImportError:
            hdf5plugin = None
        self._file = h5py.File(str(path), "r")
        self._dataset = self._file["acoustic_data"]
        if hdf5plugin is None and _ZSTD_FILTER_ID in _filter_ids(self._dataset):
            self._file.close()
            raise ImportError(
                f"{path} is compressed with zstd (HDF5 filter 32015, Audace "
                "FileWriter compression = ZSTD) and `hdf5plugin` is not "
                "installed: run `pip install hdf5plugin`."
            )
        # Cached so the shape stays answerable once the handle is gone.
        self._num_lines = int(self._dataset.shape[0])
        self._cursor = 0
        filters = _filter_ids(self._dataset)

        attrs = self._file.attrs
        # The card's time map, when the recorder stamped the file from it
        # (``time_correction`` = ``"cardclock-v1"``): then ``timestamp_ns`` is
        # the CORRECTED time of the first line and the axis gives every
        # line's. ``None`` on older files.
        self.time_axis = _read_time_axis(attrs)
        corrected = self.time_axis is not None and self.time_axis.correction == CORRECTION_CARDCLOCK_V1
        self._header = Header(
            line_size=int(attrs["line_size"]),
            trig_frequency=int(attrs["trig_frequency"]),
            sample_size=int(attrs["sample_size"]),
            sample_rate=int(attrs["sample_rate"]),
            flags=int(attrs["flags"]),
            range=int(attrs.get("range", 0)),
            pulse_width=int(attrs.get("pulse_width", 0)),
            num_channels=int(attrs.get("num_channels", 1)),
            timestamp=str(attrs.get("timestamp", "")),
            # The exact acquisition anchor (i64 ns, UTC), written beside the
            # display string whenever the capture has one. It is what
            # `File.line_time_ns` counts from; absent on legacy captures,
            # where 0 says "no anchor" exactly as it does for a DAT.
            acquisition_ns=_int_attr(attrs, "timestamp_ns", 1, _I64_MAX),
            # Absent on files written before the attribute existed; 0 is also
            # the right answer for a capture taken with no pre-roll.
            trigger_delay=int(attrs.get("trigger_delay", 0)),
            # Same story: the Rust writer emits it as of the pre-roll fix, and
            # 0 (Undefined) is what a pre-fix file honestly is. Without this,
            # the same capture reported a content type through the DAT backend
            # and `undefined` through the HDF5 one, for no reason a caller
            # could see. The number is canonical; the tag beside it
            # (`content_type_name`) stands in when the number is missing.
            content_type=_content_type(
                _opt_int_attr(attrs, "content_type"),
                _str_attr(attrs, "content_type_name"),
            ),
            # The card's trigger rational (since the real-frequency
            # recordings). Absent — 0, unknown — on older files and for an
            # external trigger; the File falls back on `trig_frequency`.
            trig_clock_hz=_int_attr(attrs, "trig_clock_hz", 0, _U32_MAX),
            trig_period_ticks=_int_attr(attrs, "trig_period_ticks", 0, _U32_MAX),
            # What the acquisition library reads back for such a file: its
            # start time is the corrected one (kept through a DAT export).
            time_flags=TIME_FLAG_CORRECTED if corrected else 0,
            # Data format v1 (absent — 0, unknown — on every older file): the
            # card code, the version, the duration of the traces written and
            # the steps, stored as f64 (exact).
            card=_card(_str_attr(attrs, "card")),
            format_version=_int_attr(attrs, "format_version", 0, 0xFF),
            duration_s=_int_attr(attrs, "duration_s", 0, _U32_MAX),
            spatial_step_m=_step(_positive_float_attr(attrs, "spatial_step_m")),
            temporal_step_s=_step(_positive_float_attr(attrs, "temporal_step_s")),
        )
        # `clock / ticks` as a float, for MATLAB / HDFView users. Normally
        # redundant with the rational above (same writer, same value); kept
        # for a file that carries it alone.
        self.trig_frequency_real_hz_attr = _positive_float_attr(
            attrs, "trig_frequency_real_hz"
        )
        # The writer's tag of the content type: what names a value this SDK
        # does not know (a newer producer).
        self.content_type_name_attr = _str_attr(attrs, "content_type_name")
        # An archive export (zone variances, content type 27 or 28): the
        # content type of the column the variance was computed from.
        self.source_content_type = _opt_int_attr(attrs, "source_content_type")
        # ZSTD_LOSSY_1E4: every phase sample was rounded to a multiple of the
        # step, at most the error away from what was acquired (Audace-Lib
        # `write_hdf5_attrs`). Absent = the samples are exact.
        self.quantization_step_rad = _positive_float_attr(attrs, "quantization_step_rad")
        self.quantization_max_error_rad = _positive_float_attr(
            attrs, "quantization_max_error_rad"
        )
        self.compression = _hdf5_compression(filters, self.quantization_step_rad)

    def _require_open(self):
        dataset = getattr(self, "_dataset", None)
        if dataset is None:
            raise ValueError(_CLOSED_MSG)
        return dataset

    @property
    def header(self) -> Header:
        return self._header

    @property
    def num_lines(self) -> int:
        return self._num_lines

    @property
    def lines_left(self) -> int:
        # Cursor state: meaningless once the handle is gone, so it says
        # so rather than answering with a frozen number.
        self._require_open()
        return max(0, self._num_lines - self._cursor)

    def read_lines(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        dataset = self._require_open()
        if self.lines_left == 0:
            raise OSError("end of file")
        n = min(n, self.lines_left)
        out = np.asarray(dataset[self._cursor : self._cursor + n])
        self._cursor += n
        # h5py returns the dataset dtype as-is, which matches the header's
        # sample encoding — the Audace HDF5 writer types it that way. The one
        # reinterpretation: an unsigned column (the `magnitude/` child of a
        # split arctan/magnitude stream) is read as unsigned whether the
        # writer stored `int16` + `unsigned = 1` (every file before u16
        # datasets) or `uint16` — the same bits either way.
        return _as_wire_dtype(out, self._header)

    def rewind(self) -> None:
        self._require_open()
        self._cursor = 0

    def seek(self, n: int) -> None:
        self._require_open()
        self._cursor = max(0, min(int(n), self._num_lines))

    def close(self) -> None:
        """Close the h5py handle. Idempotent, never raises.

        Note this releases the *file descriptor* only: arrays already
        returned by :meth:`read_lines` are owning copies and stay valid.
        """
        f = getattr(self, "_file", None)
        # Drop the references FIRST so a second call is a no-op even if
        # h5py raises on the way out.
        self._file = None
        self._dataset = None
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        # Last-resort net; `File.close()` is the intended path.
        try:
            self.close()
        except Exception:
            pass


# ── TDMS — via npTDMS ──────────────────────────────────────────────────────
#
# Audace TDMS files write a single channel ``acoustic_data`` (lines
# flattened row-major) under one group — ``Audace`` for the files of data
# format v1, ``AudaceGroup`` in older ones. All header fields travel as
# file-level properties.


class _TdmsBackend:
    GROUP_NAMES = ("Audace", "AudaceGroup")
    CHANNEL_NAME = "acoustic_data"

    def __init__(self, path: Path):
        try:
            from nptdms import TdmsFile  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "npTDMS is missing. It is normally installed automatically with "
                "`pip install invisensing`. If you installed with --no-deps, "
                "run `pip install npTDMS` (or `pip install invisensing[tdms]`)."
            ) from exc
        # `open` keeps the file handle and reads metadata only — sample
        # data is fetched lazily per `read_data(offset, length)`, so a
        # multi-GB TDMS no longer has to fit in RAM.
        self._tdms = TdmsFile.open(str(path))
        props = self._tdms.properties
        line_size = int(props["line_size"])
        sample_size = int(props["sample_size"])
        self._header = Header(
            line_size=line_size,
            trig_frequency=int(props["trig_frequency"]),
            sample_size=sample_size,
            sample_rate=int(props["sample_rate"]),
            flags=int(props["flags"]),
            range=int(props.get("range", 0)),
            pulse_width=int(props.get("pulse_width", 0)),
            num_channels=int(props.get("num_channels", 1)),
            timestamp=str(props.get("timestamp", "")),
            trigger_delay=int(props.get("trigger_delay", 0)),
            content_type=_content_type(
                _opt_int_attr(props, "content_type"),
                _str_attr(props, "content_type_name"),
            ),
            # Written by the data format v1 writer (absent before): the exact
            # anchor, the card's rational when the frequency is exact, the
            # card code, the version, the duration and the steps.
            acquisition_ns=_int_attr(props, "timestamp_ns", 1, _I64_MAX),
            trig_clock_hz=_int_attr(props, "trig_clock_hz", 0, _U32_MAX),
            trig_period_ticks=_int_attr(props, "trig_period_ticks", 0, _U32_MAX),
            card=_card(_str_attr(props, "card")),
            format_version=_int_attr(props, "format_version", 0, 0xFF),
            duration_s=_int_attr(props, "duration_s", 0, _U32_MAX),
            spatial_step_m=_step(_positive_float_attr(props, "spatial_step_m")),
            temporal_step_s=_step(_positive_float_attr(props, "temporal_step_s")),
        )
        # The writer's tag, which names a value this SDK does not know.
        self.content_type_name_attr = _str_attr(props, "content_type_name")
        self._line_size = line_size
        self._channel = self._find_channel()
        self._num_lines = int(len(self._channel)) // line_size
        self._cursor = 0

    def _find_channel(self):
        """Locate `acoustic_data` in whichever group holds it (the Audace
        groups first), else the first channel of the first group."""
        groups = sorted(self._tdms.groups(), key=lambda g: g.name not in self.GROUP_NAMES)
        for group in groups:
            for channel in group.channels():
                if channel.name == self.CHANNEL_NAME:
                    return channel
        return groups[0].channels()[0]

    def _require_open(self):
        channel = getattr(self, "_channel", None)
        if channel is None:
            raise ValueError(_CLOSED_MSG)
        return channel

    @property
    def header(self) -> Header:
        return self._header

    @property
    def num_lines(self) -> int:
        return self._num_lines

    @property
    def lines_left(self) -> int:
        self._require_open()
        return max(0, self._num_lines - self._cursor)

    def read_lines(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        channel = self._require_open()
        if self.lines_left == 0:
            raise OSError("end of file")
        n = min(n, self.lines_left)
        offset = self._cursor * self._line_size
        length = n * self._line_size
        # `read_data` returns a freshly allocated array (npTDMS decodes
        # the segment into it), so the reshape below is a view onto
        # memory we own — closing the file cannot invalidate it.
        flat = np.asarray(channel.read_data(offset, length))
        self._cursor += n
        # The TDMS channel of an older file is `int16` for every 2-byte file:
        # an unsigned column is reinterpreted as `uint16`, like the DAT
        # reader (the v1 writer stores `uint16` already).
        return _as_wire_dtype(
            flat[: n * self._line_size].reshape(n, self._line_size), self._header
        )

    def rewind(self) -> None:
        self._require_open()
        self._cursor = 0

    def seek(self, n: int) -> None:
        self._require_open()
        self._cursor = max(0, min(int(n), self._num_lines))

    def close(self) -> None:
        """Close the npTDMS handle. Idempotent, never raises."""
        tdms = getattr(self, "_tdms", None)
        self._tdms = None
        # The channel object holds the reader that owns the fd; drop it
        # too, otherwise the handle survives the close.
        self._channel = None
        if tdms is not None:
            try:
                tdms.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── SEG-Y — via segyio ─────────────────────────────────────────────────────
#
# Audace SEG-Y files write one trace per line (trigger), whose samples run
# along the fibre: ``samples_per_trace = line_size``. Header fields that the
# SEG-Y binary header cannot hold are written as ``KEY: value`` lines of the
# 3200-byte textual header — ``C 4 SAMPLE RATE: NNN Hz``, ``C16 -
# CONTENT-TYPE: dui_coherent_phase (19)``, ``C21 HEADER FLAGS (raw bits):
# 0xNNNNNNNN``… — and parsed back into a ``Header`` here.
#
# The textual header is read from the file's first 3200 bytes, not through
# segyio: segyio converts it from EBCDIC unconditionally, and the Audace
# writers write it in ASCII (an ASCII header came back as garbage, every
# field lost and every float sample read as an integer). Both encodings are
# accepted: a first byte ``C`` is ASCII, ``0xC3`` is EBCDIC ``C``; anything
# else is read as ASCII, with a warning.


#: The textual header: 40 lines of 80 characters.
_SEGY_TEXT_LEN = 3200


def _decode_segy_text(raw: bytes) -> str:
    """The textual header as text, whichever encoding it was written in
    (ASCII, or EBCDIC code page 037); unprintable bytes become spaces."""
    first = raw[:1]
    if first == b"C":
        text = raw.decode("ascii", errors="replace")
    elif first == b"\xc3":
        text = raw.decode("cp037", errors="replace")
    else:
        warnings.warn(
            "SEG-Y textual header starts with neither ASCII nor EBCDIC 'C' "
            f"(first byte {raw[:1]!r}): read as ASCII",
            stacklevel=3,
        )
        text = raw.decode("ascii", errors="replace")
    return "".join(c if c.isprintable() and c.isascii() else " " for c in text)


def _segy_text_fields(text: str) -> dict:
    """``{KEY: value}`` of the textual header's ``Cnn KEY: value`` lines: the
    ``Cnn`` prefix and a leading ``- `` removed, keys upper-case and
    trimmed (a later line wins). Lines without a colon (the notes) are not
    fields."""
    out = {}
    for line in _split_segy_lines(text):
        body = line[4:].strip()
        if body.startswith("- "):
            body = body[2:]
        key, sep, value = body.partition(":")
        if sep:
            out[key.strip().upper()] = value.strip()
    return out


def _leading_int(value: Optional[str], base: int = 10) -> Optional[int]:
    """The integer a value starts with (``"2000 Hz"`` → 2000, ``"-3"`` →
    -3; ``base=16`` reads ``"0x00000021"``), or ``None``."""
    if not value:
        return None
    v = value.strip()
    if base == 16:
        if v[:2].lower() != "0x":
            return None
        digits = ""
        for ch in v[2:]:
            if ch not in "0123456789abcdefABCDEF":
                break
            digits += ch
        return int(digits, 16) if digits else None
    end = 0
    while end < len(v) and (v[end].isdigit() or (end == 0 and v[end] == "-")):
        end += 1
    try:
        return int(v[:end])
    except ValueError:
        return None


def _leading_float(value: Optional[str]) -> Optional[float]:
    """The number a value starts with (``"0.4 METERS"`` → 0.4), or ``None``."""
    if not value or not value.split():
        return None
    try:
        return float(value.split()[0])
    except ValueError:
        return None


def _segy_content_type(value: Optional[str]) -> int:
    """``<tag> (<value>)`` since data format v1, ``<tag>`` alone before: the
    number when there is one, else the value of a known tag, else 0."""
    if not value:
        return 0
    head, sep, rest = value.partition("(")
    if sep:
        n = _leading_int(rest)
        if n is not None and 0 <= n <= 0xFF:
            return n
    tag = head.split()[0] if head.split() else ""
    return _CONTENT_TYPE_BY_NAME.get(tag, 0)


class _SegyBackend:
    def __init__(self, path: Path):
        try:
            import segyio  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "segyio is missing. It is normally installed automatically with "
                "`pip install invisensing`. If you installed with --no-deps, "
                "run `pip install segyio` (or `pip install invisensing[segy]`)."
            ) from exc
        with open(path, "rb") as fp:
            raw_text = fp.read(_SEGY_TEXT_LEN)
        fields = _segy_text_fields(_decode_segy_text(raw_text))
        # `ignore_geometry=True` skips the inline/crossline scan — our
        # files are arbitrary trace lists, not 3-D survey volumes.
        self._segy = segyio.open(str(path), ignore_geometry=True)

        # Bytes per sample and float-ness come from segyio's format code.
        # Enum member names follow segyio's SEG-Y revision wording.
        fmt = segyio.SegySampleFormat
        fmt_to_storage = {
            int(fmt.SIGNED_CHAR_1_BYTE): (1, False),
            int(fmt.UNSIGNED_CHAR_1_BYTE): (1, False),
            int(fmt.SIGNED_SHORT_2_BYTE): (2, False),
            int(fmt.UNSIGNED_SHORT_2_BYTE): (2, False),
            int(fmt.SIGNED_INTEGER_4_BYTE): (4, False),
            int(fmt.UNSIGNED_INTEGER_4_BYTE): (4, False),
            int(fmt.SIGNED_INTEGER_8_BYTE): (8, False),
            int(fmt.UNSIGNED_INTEGER_8_BYTE): (8, False),
            int(fmt.IEEE_FLOAT_4_BYTE): (4, True),
            int(fmt.IEEE_FLOAT_8_BYTE): (8, True),
            int(fmt.IBM_FLOAT_4_BYTE): (4, True),
        }
        # `SegyFile.format` is an object that converts to the numeric code
        # (segyio 1.9); looking it up as-is never matched, and every
        # integer SEG-Y was read as 4-byte samples.
        try:
            fmt_code = int(self._segy.format)
        except (TypeError, ValueError):
            fmt_code = None
        sample_size, stored_float = fmt_to_storage.get(fmt_code, (4, False))
        # u16 samples are stored as 4-byte integers (format code 2) and the
        # text says `BYTES PER SAMPLE: 2`: narrowed back to the 2-byte layout.
        if not stored_float and sample_size == 4 and _leading_int(fields.get("BYTES PER SAMPLE")) == 2:
            sample_size = 2

        samples_per_trace = _leading_int(fields.get("SAMPLES PER TRACE"))
        if samples_per_trace is None:
            samples_per_trace = self._segy.samples.size
        sample_rate = _leading_int(fields.get("SAMPLE RATE"))
        if sample_rate is None and len(self._segy.samples) > 1:
            dt_s = float(self._segy.samples[1] - self._segy.samples[0])
            sample_rate = int(round(1.0 / dt_s)) if dt_s > 0 else 0
        flags = _leading_int(fields.get("HEADER FLAGS (RAW BITS)"), base=16)
        if flags is None:
            # A file without the flags line: the format code is the only
            # description of the samples, as before.
            flags = FLAG_FLOAT if stored_float else 0
        # The raw bits of an i32: a top bit set reads as negative.
        flags = int(flags) & 0xFFFF_FFFF
        if flags >= 1 << 31:
            flags -= 1 << 32
        acquisition_ns = _leading_int(fields.get("TIMESTAMP NS")) or 0
        if not 0 < acquisition_ns <= _I64_MAX:
            acquisition_ns = 0

        def u32(key: str) -> int:
            v = _leading_int(fields.get(key))
            return v if v is not None and 0 <= v <= _U32_MAX else 0

        def step(key: str) -> float:
            v = _leading_float(fields.get(key))
            return v if v is not None and np.isfinite(v) and v > 0 else 0.0

        self._header = Header(
            line_size=int(samples_per_trace),
            trig_frequency=int(_leading_int(fields.get("TRIGGER FREQUENCY")) or 0),
            sample_size=int(sample_size),
            sample_rate=int(sample_rate or 0),
            flags=flags,
            range=int(_leading_int(fields.get("VOLTAGE RANGE")) or 0),
            pulse_width=int(_leading_int(fields.get("PULSE WIDTH")) or 0),
            num_channels=1,
            timestamp=fields.get("TIMESTAMP") or "",
            content_type=_segy_content_type(fields.get("CONTENT-TYPE")),
            acquisition_ns=acquisition_ns,
            trigger_delay=u32("TRIGGER DELAY"),
            card=_card(fields.get("CARD")),
            format_version=min(u32("FORMAT VERSION"), 0xFF),
            duration_s=u32("DURATION"),
            spatial_step_m=step("SAMPLE SPACING ALONG THE FIBER"),
            temporal_step_s=step("TRACE SPACING IN TIME"),
        )

        # Traces are read lazily, one chunk per `read_lines`, via segyio's
        # `trace.raw[start:stop]` -- a multi-GB SEG-Y no longer has to fit
        # in RAM. We only cast to the integer dtype on the (small) chunk.
        self._sample_size = sample_size
        self._num_lines = int(self._segy.tracecount)
        self._cursor = 0

    def _require_open(self):
        segy = getattr(self, "_segy", None)
        if segy is None:
            raise ValueError(_CLOSED_MSG)
        return segy

    @property
    def header(self) -> Header:
        return self._header

    @property
    def num_lines(self) -> int:
        return self._num_lines

    @property
    def lines_left(self) -> int:
        self._require_open()
        return max(0, self._num_lines - self._cursor)

    def read_lines(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        self._require_open()
        if self.lines_left == 0:
            raise OSError("end of file")
        n = min(n, self.lines_left)
        # `trace.raw[start:stop]` -> (n, samples_per_trace) float32.
        # A *slice* of `trace.raw` allocates its own array (unlike
        # iterating `trace.raw`, which recycles one buffer). Verified in
        # tests/test_close.py: the array survives `close()` unchanged.
        block = np.asarray(self._segy.trace.raw[self._cursor : self._cursor + n])
        self._cursor += n
        want = wire_dtype(self._header)
        if want.kind in "iu" and self._sample_size in (2, 4, 8):
            if want.kind == "u":
                # Through int64 (exact for every 2- and 4-byte code), then
                # modular: a u16 magnitude written with a signed format code
                # (-25536.0) and one written unsigned (40000.0) both come
                # back as 40000.
                block = block.astype(np.int64).astype(want)
            else:
                block = block.astype(want)
        return np.ascontiguousarray(block)

    def rewind(self) -> None:
        self._require_open()
        self._cursor = 0

    def seek(self, n: int) -> None:
        self._require_open()
        self._cursor = max(0, min(int(n), self._num_lines))

    def close(self) -> None:
        """Close the segyio handle. Idempotent, never raises."""
        segy = getattr(self, "_segy", None)
        self._segy = None
        if segy is not None:
            try:
                segy.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _split_segy_lines(text: str):
    # The textual header is 40 lines × 80 chars, usually without newlines.
    # Split fixed-width unless the text carries explicit newlines.
    if "\n" in text:
        return text.split("\n")
    return [text[i : i + 80] for i in range(0, len(text), 80)]

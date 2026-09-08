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
    DatReader,
    Header,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    FLAG_PHASE,
)


#: Message of the :class:`ValueError` every backend raises when it is
#: read after ``close()``.
_CLOSED_MSG = "operation on a closed file"


# ── DAT — native Rust ──────────────────────────────────────────────────────


class _DatBackend:
    """Wraps the Rust DatReader so the same `File` facade applies."""

    def __init__(self, path: Path):
        self._inner = DatReader(str(path))
        # Cached at construction so metadata stays answerable after
        # close() — the file is gone, the numbers describing it are not.
        self._header = self._inner.header
        self._num_lines = int(self._inner.num_lines)

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
# Audace HDF5 files store the samples as a (num_traces,
# samples_per_trace) dataset named ``acoustic_data``, typed by
# ``bytes_per_sample`` + the ``FLOAT`` attribute. All header fields are
# mirrored as scalar attributes. Only the metadata is read at
# construction; the samples are fetched per ``read_lines`` slice, so a
# multi-GB capture does not have to fit in RAM. Each slice is a fresh,
# owning numpy array — h5py materialises it on the spot — so closing
# the file does not invalidate arrays already handed to the caller.


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
        self._file = h5py.File(str(path), "r")
        self._dataset = self._file["acoustic_data"]
        # Cached so the shape stays answerable once the handle is gone.
        self._num_lines = int(self._dataset.shape[0])
        self._cursor = 0

        attrs = self._file.attrs
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
            # Absent on files written before the attribute existed; 0 is also
            # the right answer for a capture taken with no pre-roll.
            trigger_delay=int(attrs.get("trigger_delay", 0)),
        )

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
        # h5py returns the dataset dtype as-is, which already matches
        # (sample_size, FLOAT) — the Audace HDF5 writer types it that
        # way. We deliberately do not reinterpret here so callers see
        # the same numbers HDF5 saved.
        return out

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
# Audace TDMS files write a single channel ``acoustic_data`` under the
# group ``AudaceGroup``. All header fields travel as file-level
# properties.


class _TdmsBackend:
    GROUP_NAME = "AudaceGroup"
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
        )
        self._line_size = line_size
        self._channel = self._find_channel()
        self._num_lines = int(len(self._channel)) // line_size
        self._cursor = 0

    def _find_channel(self):
        """Locate `acoustic_data` whichever group nptdms reports."""
        groups = self._tdms.groups()
        group = next((g for g in groups if g.name == self.GROUP_NAME), groups[0])
        channels = group.channels()
        return next((c for c in channels if c.name == self.CHANNEL_NAME), channels[0])

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
        return flat[: n * self._line_size].reshape(n, self._line_size)

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
# Audace SEG-Y files write one trace per fibre position, with
# ``samples_per_trace = line_size`` (wire-side, doubled in INTERLEAVED).
# Header fields that aren't expressible in the SEG-Y binary header are
# encoded as discrete lines in the EBCDIC textual header — e.g.
# ``C21 HEADER FLAGS (raw bits): 0xNNNNNNNN``, ``C4 SAMPLE RATE: NNN Hz``.
# We parse those back into a ``Header`` so the channel extractors keep
# working.


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
        # `ignore_geometry=True` skips the inline/crossline scan — our
        # files are arbitrary trace lists, not 3-D survey volumes.
        self._segy = segyio.open(str(path), ignore_geometry=True)
        text = self._segy.text[0].decode("ascii", errors="replace")

        flags = _scan_segy_text_int(text, "HEADER FLAGS", base=16)
        sample_rate = _scan_segy_text_int(text, "SAMPLE RATE")
        trig_freq = _scan_segy_text_int(text, "TRIGGER FREQUENCY")
        pulse_width = _scan_segy_text_int(text, "PULSE WIDTH")
        trigger_delay = _scan_segy_text_int(text, "TRIGGER DELAY")
        voltage_mv = _scan_segy_text_int(text, "VOLTAGE RANGE")
        samples_per_trace = _scan_segy_text_int(text, "SAMPLES PER TRACE")
        timestamp = _scan_segy_text_str(text, "TIMESTAMP")

        # bytes per sample comes from segyio's format code. Enum member
        # names follow segyio's SEG-Y revision wording (SIGNED_*_BYTE).
        fmt_to_bytes = {
            segyio.SegySampleFormat.SIGNED_CHAR_1_BYTE: 1,
            segyio.SegySampleFormat.UNSIGNED_CHAR_1_BYTE: 1,
            segyio.SegySampleFormat.SIGNED_SHORT_2_BYTE: 2,
            segyio.SegySampleFormat.UNSIGNED_SHORT_2_BYTE: 2,
            segyio.SegySampleFormat.SIGNED_INTEGER_4_BYTE: 4,
            segyio.SegySampleFormat.UNSIGNED_INTEGER_4_BYTE: 4,
            segyio.SegySampleFormat.SIGNED_INTEGER_8_BYTE: 8,
            segyio.SegySampleFormat.UNSIGNED_INTEGER_8_BYTE: 8,
            segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE: 4,
            segyio.SegySampleFormat.IEEE_FLOAT_8_BYTE: 8,
            segyio.SegySampleFormat.IBM_FLOAT_4_BYTE: 4,
        }
        sample_size = fmt_to_bytes.get(self._segy.format, 4)

        # Sanity: when the textual lookup fails, fall back to the binary
        # header / segyio's geometry so we still return a usable Header.
        if samples_per_trace is None:
            samples_per_trace = self._segy.samples.size
        if sample_rate is None and len(self._segy.samples) > 1:
            dt_s = float(self._segy.samples[1] - self._segy.samples[0])
            sample_rate = int(round(1.0 / dt_s)) if dt_s > 0 else 0

        self._header = Header(
            line_size=int(samples_per_trace),
            trig_frequency=int(trig_freq or 0),
            sample_size=int(sample_size),
            sample_rate=int(sample_rate or 0),
            flags=int(flags or 0),
            range=int(voltage_mv or 0),
            pulse_width=int(pulse_width or 0),
            num_channels=1,
            timestamp=timestamp or "",
            trigger_delay=int(trigger_delay or 0),
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
        if not self._header.is_float and self._sample_size in (2, 4, 8):
            block = block.astype(f"int{self._sample_size * 8}")
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


# ── SEG-Y textual-header scanners ──────────────────────────────────────────


def _scan_segy_text_int(text: str, marker: str, base: int = 10) -> Optional[int]:
    """
    Find the first line containing ``marker`` and parse the first
    integer token after the marker. ``base=16`` recognises ``0xNN``
    notation. Returns ``None`` if the marker / digits are not found —
    callers fall back to defaults rather than raising.
    """
    for line in _split_segy_lines(text):
        idx = line.find(marker)
        if idx < 0:
            continue
        rest = line[idx + len(marker):]
        if base == 16:
            # The writer emits ``0x{:08X}`` — anchor on the ``0x`` prefix.
            # Grabbing the first loose hex-letter run instead would parse
            # prose around the value (e.g. the ``a`` of "(raw bits)") and
            # return garbage flags for every SEG-Y produced by Filewriter.
            pos = rest.find("0x")
            if pos < 0:
                pos = rest.find("0X")
            if pos < 0:
                continue
            token = ""
            for ch in rest[pos + 2:]:
                if ch in "0123456789abcdefABCDEF":
                    token += ch
                else:
                    break
            if not token:
                continue
            try:
                return int(token, 16)
            except ValueError:
                continue
        else:
            # base 10: first run of decimal digits after the marker.
            token = ""
            in_token = False
            for ch in rest:
                if ch.isdigit():
                    token += ch
                    in_token = True
                elif in_token:
                    break
            if not token:
                continue
            try:
                return int(token, base)
            except ValueError:
                continue
    return None


def _scan_segy_text_str(text: str, marker: str) -> Optional[str]:
    for line in _split_segy_lines(text):
        idx = line.find(marker)
        if idx < 0:
            continue
        # Skip the marker and any "  :  " separator.
        rest = line[idx + len(marker):].lstrip(": ").rstrip()
        return rest or None
    return None


def _split_segy_lines(text: str):
    # The textual header is 40 lines × 80 chars, but segyio may return it
    # without explicit newlines. Split fixed-width to be safe.
    if "\n" in text:
        return text.split("\n")
    return [text[i : i + 80] for i in range(0, len(text), 80)]

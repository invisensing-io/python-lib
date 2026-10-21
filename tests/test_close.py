"""
Descriptor-lifecycle tests for :class:`invisensing.File`.

These are the tests that actually *prove* point 77. The defect they pin
down is narrow and easy to mis-diagnose: on CPython the backend used to
be released when the ``File`` object itself was reclaimed, so a throwaway
``File(p)`` inside a loop looked clean. What leaked was the file that was
*explicitly* closed — ``close()`` and ``__exit__`` marked ``_closed`` and
returned without dropping ``self._backend``, so the descriptor survived
for as long as the caller kept the (closed!) ``File`` around. Every test
below therefore keeps the ``File`` alive across the assertion; that is
the whole point, and a version of these tests that let it go out of scope
would pass on the broken code.

The measurement is the number of entries in ``/proc/self/fd``, taken
before and after, and compared for exact equality.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from invisensing import File, export_dat, FLAG_FLOAT
from invisensing._core import DatReader


# ── Fixtures: one small file per on-disk format ────────────────────────────

ROWS = 8
POSITIONS = 6
SAMPLE_RATE = 100_000_000
TRIG_FREQUENCY = 1_000


def _payload() -> np.ndarray:
    """Deterministic int16 payload — values chosen to include negatives."""
    return (
        np.arange(ROWS * POSITIONS, dtype=np.int16).reshape(ROWS, POSITIONS) * 7 - 100
    )


def _write_dat(path: Path) -> np.ndarray:
    data = _payload()
    export_dat(
        path,
        data,
        sample_rate=SAMPLE_RATE,
        trig_frequency=TRIG_FREQUENCY,
        range_v=1.0,
        timestamp="2026-01-01_close",
    )
    return data


def _write_hdf5(path: Path) -> np.ndarray:
    h5py = pytest.importorskip("h5py")
    data = _payload()
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=data)
        fp.attrs["line_size"] = POSITIONS
        fp.attrs["trig_frequency"] = TRIG_FREQUENCY
        fp.attrs["sample_size"] = 2
        fp.attrs["sample_rate"] = SAMPLE_RATE
        fp.attrs["flags"] = 0
        fp.attrs["range"] = 1_000
        fp.attrs["pulse_width"] = 4
        fp.attrs["num_channels"] = 1
        fp.attrs["timestamp"] = "2026-01-01_close"
    return data


def _write_tdms(path: Path) -> np.ndarray:
    pytest.importorskip("nptdms")
    from nptdms import TdmsWriter, ChannelObject, RootObject

    data = _payload()
    root_props = {
        "line_size": POSITIONS,
        "trig_frequency": TRIG_FREQUENCY,
        "sample_size": 2,
        "sample_rate": SAMPLE_RATE,
        "flags": 0,
        "range": 1_000,
        "pulse_width": 4,
        "num_channels": 1,
        "timestamp": "2026-01-01_close",
    }
    with TdmsWriter(str(path)) as w:
        w.write_segment(
            [
                RootObject(properties=root_props),
                ChannelObject("AudaceGroup", "acoustic_data", data.flatten()),
            ]
        )
    return data


def _write_segy(path: Path) -> np.ndarray:
    segyio = pytest.importorskip("segyio")

    # float32 payload: the SEG-Y sample formats segyio can create are
    # float-shaped, and the FLOAT flag in the textual header is what tells
    # the SDK not to re-cast. Same shape Filewriter's SEG-Y writer emits.
    data = _payload().astype(np.float32)
    spec = segyio.spec()
    spec.format = int(segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE)
    spec.samples = list(range(POSITIONS))
    spec.tracecount = ROWS
    with segyio.create(str(path), spec) as f:
        for i in range(ROWS):
            f.trace[i] = data[i]
        f.text[0] = segyio.tools.create_text_header(
            {
                1: "AUDACE DAS DATA",
                3: "TIMESTAMP: 2026-01-01_close",
                4: f"SAMPLE RATE: {SAMPLE_RATE} Hz",
                5: f"TRIGGER FREQUENCY: {TRIG_FREQUENCY} Hz",
                8: f"SAMPLES PER TRACE: {POSITIONS}",
                21: f"HEADER FLAGS (raw bits): 0x{FLAG_FLOAT:08X}",
            }
        )
    return data


_WRITERS = {
    "dat": ("close.dat", _write_dat),
    "hdf5": ("close.h5", _write_hdf5),
    "tdms": ("close.tdms", _write_tdms),
    "segy": ("close.sgy", _write_segy),
}


@pytest.fixture(params=sorted(_WRITERS))
def sample_file(request, tmp_path):
    """(path, expected_array) for each supported on-disk format."""
    name, writer = _WRITERS[request.param]
    path = tmp_path / name
    expected = writer(path)
    return path, expected


@pytest.fixture
def dat_file(tmp_path):
    path = tmp_path / "one.dat"
    expected = _write_dat(path)
    return path, expected


# ── Descriptor accounting ──────────────────────────────────────────────────

_FD_DIR = "/proc/self/fd"

requires_proc_fd = pytest.mark.skipif(
    not os.path.isdir(_FD_DIR),
    reason="descriptor counting needs /proc/self/fd (Linux)",
)


def _open_fds() -> int:
    return len(os.listdir(_FD_DIR))


def _warm_up(path: Path) -> None:
    """Open and drop one file so lazy imports / library-internal handles
    are already in place before the baseline is taken."""
    f = File(path)
    f.read_lines(1)
    f.close()
    del f
    gc.collect()


@requires_proc_fd
def test_close_releases_the_descriptor_while_the_object_is_still_alive(sample_file):
    """`close()` must free the handle at the call, not at collection time.

    On the pre-fix code the assertion below fails by exactly one
    descriptor: `close()` set `_closed` and left `_backend` in place, so
    the handle lived on with the object.
    """
    path, _ = sample_file
    _warm_up(path)

    base = _open_fds()
    f = File(path)
    assert _open_fds() > base, "opening a file should cost at least one descriptor"
    f.close()
    assert _open_fds() == base
    assert f.closed
    del f


@requires_proc_fd
def test_with_block_releases_the_descriptor_at_its_exit(sample_file):
    """Leaving the `with` must free the handle even though the name `f`
    is still bound in the enclosing scope — that promise is the whole
    reason to write `with File(...)`."""
    path, _ = sample_file
    _warm_up(path)

    base = _open_fds()
    with File(path) as f:
        assert _open_fds() > base
    assert _open_fds() == base
    assert f.closed


@requires_proc_fd
def test_two_hundred_closed_files_leak_nothing(sample_file, tmp_path):
    """The fiche's headline check: open then close 200 files, keeping all
    the (closed) objects, and the descriptor count must come back exactly
    to where it started. Pre-fix it grows by 200.

    The 200 paths are **distinct copies**, not the same path 200 times.
    h5py de-duplicates the descriptor between two opens of one path, so
    on a single path the pre-fix HDF5 leak reads as +1 instead of +200
    and a regression that only leaks the dataset handle would slip
    through. Distinct paths are also the real shape of the failure: a
    server walking a data tree opens different files, not one file over
    and over.
    """
    path, _ = sample_file
    _warm_up(path)

    copies = tmp_path / "many"
    copies.mkdir()
    paths = []
    for i in range(200):
        target = copies / f"{i:03d}{path.suffix}"
        shutil.copyfile(path, target)
        paths.append(target)

    base = _open_fds()
    kept = []
    for target in paths:
        handle = File(target)
        handle.close()
        kept.append(handle)
    assert _open_fds() == base
    assert len(kept) == 200


# ── Idempotence and use-after-close ────────────────────────────────────────


def test_double_close_is_a_noop(sample_file):
    path, _ = sample_file
    f = File(path)
    f.close()
    f.close()
    f.close()
    assert f.closed


def test_close_after_a_with_block_is_a_noop(sample_file):
    path, _ = sample_file
    with File(path) as f:
        pass
    f.close()
    assert f.closed


def test_reads_after_close_raise_value_error(sample_file):
    """Not AttributeError('NoneType' has no attribute …), and — for .dat —
    not a native crash either."""
    path, _ = sample_file
    f = File(path)
    f.close()

    with pytest.raises(ValueError, match="closed"):
        f.read_lines(1)
    with pytest.raises(ValueError, match="closed"):
        f.read_all()
    with pytest.raises(ValueError, match="closed"):
        f.lines_left
    with pytest.raises(ValueError, match="closed"):
        f.rewind()
    with pytest.raises(ValueError, match="closed"):
        f.seek_lines(1)
    with pytest.raises(ValueError, match="closed"):
        list(f)


#: Every metadata accessor of `File`. None of them touches the file: the
#: header is read once, at open, and these are arithmetic on it. They are
#: listed exhaustively on purpose — a new accessor added behind
#: `_require_open()` by mistake fails here instead of at a customer site.
_METADATA_ACCESSORS = {
    "header": lambda f: f.header.line_size,
    "mode": lambda f: f.mode,
    "line_size": lambda f: f.line_size,
    "positions_per_line": lambda f: f.positions_per_line,
    "sample_size": lambda f: f.sample_size,
    "sample_rate": lambda f: f.sample_rate,
    "trig_frequency": lambda f: f.trig_frequency,
    "trig_clock_hz": lambda f: f.trig_clock_hz,
    "trig_period_ticks": lambda f: f.trig_period_ticks,
    "trig_frequency_real_hz": lambda f: f.trig_frequency_real_hz,
    "trig_frequency_is_exact": lambda f: f.trig_frequency_is_exact,
    "line_time_ns": lambda f: f.line_time_ns(ROWS - 1),
    "num_lines": lambda f: f.num_lines,
    "timestamp": lambda f: f.timestamp,
    "flags": lambda f: f.flags,
    "range": lambda f: f.range,
    "trigger_delay": lambda f: f.trigger_delay,
    "dtype": lambda f: f.dtype,
    "shape": lambda f: f.shape,
    "spatial_shape": lambda f: f.spatial_shape,
    "duration": lambda f: f.duration,
    "distance": lambda f: f.distance,
    "is_mozart": lambda f: f.is_mozart,
    "is_demodulated": lambda f: bool(f.is_demodulated),
    "is_interleaved": lambda f: bool(f.is_interleaved),
    "is_float": lambda f: bool(f.is_float),
    "is_phase": lambda f: bool(f.is_phase),
    "is_unsigned": lambda f: bool(f.is_unsigned),
    "is_ac": lambda f: bool(f.is_ac),
    "is_hiz": lambda f: bool(f.is_hiz),
    "is_acquisition_ac": lambda f: f.is_acquisition_ac(),
    "is_acquisition_hiz": lambda f: f.is_acquisition_hiz(),
}


def test_metadata_after_close_keeps_answering(sample_file):
    """Closing gives up the descriptor, not the knowledge.

    Reading a header field of a file you have already closed is a normal,
    common thing to do — ``with File(p) as f: data = f.read_all()`` then
    ``dt = 1.0 / f.trig_frequency`` outside the block. Nothing about it
    needs the file back: the header was parsed at open and lives on the
    object. Raising there would turn a correct script into a crash on the
    one tool the end user runs himself, so every accessor below must
    return the *same value* after close as before.
    """
    path, _ = sample_file
    f = File(path)
    before = {name: get(f) for name, get in _METADATA_ACCESSORS.items()}
    f.close()
    after = {name: get(f) for name, get in _METADATA_ACCESSORS.items()}
    assert after == before


def test_the_documented_after_close_idiom_works(sample_file):
    """The exact motif from the class docstring, end to end."""
    path, _ = sample_file
    with File(path) as f:
        data = f.read_all()
    # Outside the block: the file is closed and still describes itself.
    assert f.closed
    dt = 1.0 / f.trig_frequency
    axis = np.arange(f.num_lines) * dt
    assert axis.shape == (data.shape[0],)
    assert f.dtype == data.dtype
    assert f.shape == data.shape


def test_cursor_state_after_close_raises_value_error(sample_file):
    """`lines_left` is the one number that is *not* metadata: it is the
    position of a read cursor on a handle that no longer exists. It must
    say so rather than answer with a frozen value that would send a
    ``while f.lines_left:`` loop into a read that raises."""
    path, _ = sample_file
    f = File(path)
    f.close()
    with pytest.raises(ValueError, match="closed"):
        f.lines_left


def test_extractors_accept_a_preread_buffer_after_close(sample_file):
    """Channel extraction is arithmetic on an array the caller already
    holds. Once the buffer is read, the file is not needed any more —
    closing it early (the right thing to do on a NAS) must not disarm the
    extractors."""
    path, expected = sample_file
    f = File(path)
    buf = f.read_all()
    f.close()

    channels = f.channels(buf)
    assert set(channels) == {"samples"}  # fixtures are RAW files
    np.testing.assert_array_equal(channels["samples"], buf)
    np.testing.assert_allclose(
        buf.astype(np.float64), expected.astype(np.float64), rtol=0, atol=0
    )


def test_repr_of_a_closed_file_still_describes_it(sample_file):
    path, _ = sample_file
    f = File(path)
    live = repr(f)
    f.close()
    closed = repr(f)
    assert "closed" in closed
    assert f"shape={f.shape}" in closed
    # Same description, plus the closed marker.
    assert closed.replace(", closed)", ")") == live


def test_path_and_repr_still_work_after_close(sample_file):
    """`repr` is what a debugger calls; it must not blow up on a closed
    file, and `path` is metadata the File owns itself."""
    path, _ = sample_file
    f = File(path)
    f.close()
    assert f.path == path
    assert "closed" in repr(f)
    assert str(path) in repr(f)


# ── The trap the fiche flags: arrays must not be views on the file ─────────


def test_arrays_read_before_close_survive_the_close(sample_file):
    """If a backend handed back a view onto the still-open file, closing
    would turn correct data into garbage — silently. Read, close, and
    compare against the values written."""
    path, expected = sample_file
    f = File(path)
    got = f.read_all()
    snapshot = got.copy()
    f.close()
    gc.collect()

    np.testing.assert_array_equal(got, snapshot)
    np.testing.assert_allclose(
        got.astype(np.float64), expected.astype(np.float64), rtol=0, atol=0
    )


def test_two_successive_reads_do_not_share_a_buffer(sample_file):
    """A backend recycling one internal buffer between reads would let the
    second read overwrite the array the caller still holds."""
    path, expected = sample_file
    with File(path) as f:
        first = f.read_lines(2)
        first_snapshot = first.copy()
        f.read_lines(2)
        np.testing.assert_array_equal(first, first_snapshot)
        np.testing.assert_allclose(
            first.astype(np.float64), expected[:2].astype(np.float64), rtol=0, atol=0
        )


# ── ResourceWarning ────────────────────────────────────────────────────────


def test_resource_warning_when_the_file_is_never_closed(dat_file):
    path, _ = dat_file
    with pytest.warns(ResourceWarning, match="was not closed explicitly"):
        File(path)  # dropped immediately -> __del__ runs here
        gc.collect()


def test_no_resource_warning_after_a_with_block(dat_file):
    path, _ = dat_file
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with File(path) as f:
            f.read_lines(1)
        del f
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


def test_no_resource_warning_after_an_explicit_close(dat_file):
    path, _ = dat_file
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = File(path)
        f.close()
        del f
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


def test_del_does_not_raise_when_the_open_failed(tmp_path):
    """A File whose construction blew up must still be collectable in
    silence — `__del__` runs on a half-built object."""
    bad = tmp_path / "not-audace.dat"
    bad.write_bytes(b"\x00" * 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Exception):
            File(bad)
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


# ── Native side: DatReader ─────────────────────────────────────────────────


@requires_proc_fd
def test_dat_reader_close_releases_the_native_descriptor(dat_file):
    path, _ = dat_file
    r = DatReader(str(path))
    r.read_lines(1)
    del r
    gc.collect()

    base = _open_fds()
    reader = DatReader(str(path))
    assert _open_fds() == base + 1
    reader.close()
    assert _open_fds() == base
    assert reader.is_closed


def test_dat_reader_close_is_idempotent_and_reads_raise(dat_file):
    path, _ = dat_file
    reader = DatReader(str(path))
    reader.close()
    reader.close()
    assert reader.is_closed

    with pytest.raises(ValueError, match="closed"):
        reader.read_lines(1)
    with pytest.raises(ValueError, match="closed"):
        reader.rewind()
    with pytest.raises(ValueError, match="closed"):
        reader.seek_lines(0)


def test_dat_reader_keeps_describing_the_file_after_close(dat_file):
    """Closing gives up the handle, not the knowledge: a viewer that
    closed the file still has to render its header."""
    path, _ = dat_file
    reader = DatReader(str(path))
    header = reader.header
    num_lines = reader.num_lines
    reader.close()
    assert reader.header.line_size == header.line_size
    assert reader.num_lines == num_lines
    assert reader.path == str(path)


# ── The native close can fail, and it must not do so in silence ────────────


class _FakeReader:
    """Stands in for the Rust ``DatReader``. Its ``close()`` raises the way
    PyO3 does when another thread is inside ``read_lines``: the pyclass is
    mutably borrowed for the whole read (the GIL is released inside it, so a
    second thread really can reach ``close()``), and the borrow guard raises
    ``RuntimeError('Already borrowed')``. Reproducing that for real needs a
    multi-hundred-megabyte read racing a thread; the failure mode it puts on
    `_DatBackend.close()` is exactly this, so it is injected here instead."""

    def __init__(self, header, num_lines):
        self.header = header
        self.num_lines = num_lines
        self.lines_left = num_lines
        self.closed_attempts = 0

    def close(self):
        self.closed_attempts += 1
        raise RuntimeError("Already borrowed")


def test_backend_close_warns_when_the_native_close_fails(dat_file):
    """A blanket ``except Exception: pass`` here meant ``File.closed`` could
    be True while the descriptor was still open — the NAS refuses to replace
    the file and nothing anywhere says why. Warn, and still never raise."""
    from invisensing import _formats

    path, _ = dat_file
    backend = _formats._DatBackend(path)
    fake = _FakeReader(backend.header, backend.num_lines)
    backend._inner = fake

    with pytest.warns(ResourceWarning, match="native DatReader.close"):
        backend.close()          # must not raise

    assert fake.closed_attempts == 1
    assert backend._inner is None
    # Still idempotent, and the second call has nothing left to warn about.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.close()
    assert not caught


def test_backend_close_is_silent_on_an_extension_without_close(dat_file):
    """The legitimate reason the old blanket ``except`` existed: a ``.so``
    built before ``DatReader.close()`` existed. Dropping the reference is
    then the whole release path — nothing to report."""
    from invisensing import _formats

    path, _ = dat_file
    backend = _formats._DatBackend(path)
    inner = backend._inner
    backend._inner = _OldReader(inner)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.close()
    assert not caught
    assert backend._inner is None


class _OldReader:
    """A reader with no ``close`` method, like a pre-1.2.1 extension."""

    def __init__(self, inner):
        self.header = inner.header
        self.num_lines = inner.num_lines
        self.lines_left = inner.lines_left


# ── Iteration ──────────────────────────────────────────────────────────────


def test_iteration_stops_with_value_error_if_the_file_is_closed_midway(
    sample_file,
):
    """`__iter__` resolves the backend once and then talks to it directly.
    Closing halfway through must still be caught on the very next pulse —
    reported by the backend now, but still a ValueError, never a crash and
    never silent truncation."""
    path, _ = sample_file
    f = File(path)
    it = iter(f)
    first = next(it)
    assert first.shape[0] > 0
    f.close()
    with pytest.raises(ValueError, match="closed"):
        next(it)


def test_iteration_yields_every_pulse_in_order(sample_file):
    path, expected = sample_file
    with File(path) as f:
        pulses = list(f)
    assert len(pulses) == expected.shape[0]
    np.testing.assert_allclose(
        np.stack(pulses).astype(np.float64),
        expected.astype(np.float64),
        rtol=0,
        atol=0,
    )

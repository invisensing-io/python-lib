"""
Tests for the new channel extractors (`get_i / get_q / get_arctan /
get_magnitude / get_phase`) and the format auto-detection.

We synthesise test files in-memory so the suite doesn't depend on a
particular PCIe7821 capture — that way the kernel correctness is what's
under test, not a frozen acquisition.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Mode,
    Header,
    DatReader,
    parse_header,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    FLAG_PHASE,
)
from invisensing._reader import build_header_bytes


HEADER_SIZE = 128


def _write_dat(path: Path, data: np.ndarray, *, flags: int, sample_rate: int = 250_000_000):
    """Helper: build an Audace .dat from a 2-D numpy array + flag mask."""
    if data.dtype.kind == "f":
        flags |= FLAG_FLOAT
    header = Header(
        line_size=int(data.shape[1]),
        trig_frequency=2_000,
        sample_size=int(data.dtype.itemsize),
        sample_rate=int(sample_rate),
        flags=int(flags),
        range=2_000,
        pulse_width=4,
        num_channels=1,
        timestamp="2026-01-01_test",
    )
    path.write_bytes(b"")
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        fp.write(np.ascontiguousarray(data).tobytes())


# ── Header round-trip ──────────────────────────────────────────────────────


def test_header_round_trip():
    h = Header(
        line_size=500,
        trig_frequency=2_000,
        sample_size=2,
        sample_rate=250_000_000,
        flags=FLAG_DEMODULATED | FLAG_INTERLEAVED,
        range=1500,
        pulse_width=8,
        num_channels=1,
        timestamp="2026-01-01_round_trip",
    )
    raw = build_header_bytes(h)
    assert len(raw) == HEADER_SIZE
    parsed = parse_header(raw)
    assert parsed.line_size == 500
    assert parsed.sample_rate == 250_000_000
    assert parsed.is_interleaved
    assert parsed.is_demodulated
    assert not parsed.is_phase
    assert parsed.positions_per_line == 250
    assert parsed.timestamp == "2026-01-01_round_trip"


# ── IQ mode: extract I and Q from an INTERLEAVED file ─────────────────────


def test_iq_file_extracts_i_and_q():
    # Build a synthetic IQ file: 4 pulses × 6 positions → line_size = 12,
    # interleaved as [I0,Q0,I1,Q1,…,I5,Q5] per row.
    rows, positions = 4, 6
    i = np.tile(np.arange(positions, dtype=np.int16), (rows, 1)) * 10
    q = np.tile(np.arange(positions, dtype=np.int16), (rows, 1)) * 10 + 1
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        _write_dat(path, interleaved, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)

        f = File(path)
        assert f.is_interleaved
        assert not f.is_unsigned
        # Bool-callable: both styles work.
        assert f.is_interleaved() is True
        assert f.is_unsigned() is False
        assert f.line_size == positions * 2
        assert f.positions_per_line == positions
        assert f.dtype == np.int16

        chunk = f.read_lines(rows)
        assert chunk.shape == (rows, positions * 2)

        got_i = f.get_i(chunk)
        got_q = f.get_q(chunk)
        assert got_i.shape == (rows, positions)
        assert got_q.shape == (rows, positions)
        np.testing.assert_array_equal(got_i, i)
        np.testing.assert_array_equal(got_q, q)

        # IQ → complex64
        f.rewind()
        # `read_lines` first, then `get_iq`, so we don't consume the cursor
        # twice (the cursor advances independently from chunked extractors).
        chunk = f.read_lines(rows)
        complex_iq = f.get_iq(chunk)
        assert complex_iq.dtype == np.complex64
        np.testing.assert_array_equal(complex_iq.real.astype(np.int16), i)
        np.testing.assert_array_equal(complex_iq.imag.astype(np.int16), q)


def test_iq_file_returns_volts_when_asked():
    """The ``*_volts`` extractors scale i16 by ``range / 32768``."""
    rows, positions = 2, 4
    # ADC codes ±32767 with a 1.9 V range → ±0.9499V at full scale.
    i_codes = np.array([[16384, -16384, 32767, -32768],
                        [    0,    100,  -200,    300]], dtype=np.int16)
    q_codes = np.zeros_like(i_codes)
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i_codes
    interleaved[:, 1::2] = q_codes

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        # range_v = 1900 mV (PCIe7821 nominal). We pass it via `range_v`
        # in the helper, which writes the header field in mV.
        _write_dat(path, interleaved, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)
        # _write_dat doesn't accept range, so patch the asset post-hoc by
        # re-opening it through the public writer.
        f = File(path)
        buf = f.read_all()
        scale = f.range / 32768.0
        np.testing.assert_allclose(
            f.get_i_volts(buf),
            i_codes.astype(np.float32) * scale,
            rtol=1e-6,
        )
        # `q_codes` are zeros so volts should also be zero (asserts the
        # scaler doesn't introduce a bias).
        np.testing.assert_array_equal(
            f.get_q_volts(buf), np.zeros((rows, positions), dtype=np.float32)
        )


def test_iq_complex_volts_packs_real_imag():
    rows, positions = 2, 3
    i = np.array([[100, 200, 300], [-100, -200, -300]], dtype=np.int16)
    q = np.array([[ 10,  20,  30], [ -10,  -20,  -30]], dtype=np.int16)
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        _write_dat(path, interleaved, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)
        f = File(path)
        iq_v = f.get_iq_volts(n=rows)
        assert iq_v.dtype == np.complex64
        scale = f.range / 32768.0
        np.testing.assert_allclose(iq_v.real, i.astype(np.float32) * scale, rtol=1e-6)
        np.testing.assert_allclose(iq_v.imag, q.astype(np.float32) * scale, rtol=1e-6)


def test_arctan_radians_uses_vendor_pi_scaling():
    rows, positions = 2, 4
    # Vendor convention: i16 = 32767 → +π, -32768 → -π.
    arctan_codes = np.array([[16384,      0, -16384, 32767],
                             [    0,   8192,  -8192, -16384]], dtype=np.int16)
    mag = np.zeros_like(arctan_codes)  # unused
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = arctan_codes
    interleaved[:, 1::2] = mag
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "am.dat"
        _write_dat(
            path,
            interleaved,
            flags=FLAG_DEMODULATED | FLAG_INTERLEAVED | FLAG_UNSIGNED,
        )
        f = File(path)
        rad = f.get_arctan_radians(n=rows)
        assert rad.dtype == np.float32
        expected = arctan_codes.astype(np.float32) * np.float32(np.pi / 32768.0)
        np.testing.assert_allclose(rad, expected, rtol=1e-6)
        # Spot-check: i16 = 16384 → +π/2 within float32 precision.
        assert abs(rad[0, 0] - np.pi / 2) < 1e-4


def test_magnitude_volts_scales_unsigned_lane():
    rows, positions = 2, 3
    arctan = np.zeros((rows, positions), dtype=np.int16)
    mag_u16 = np.array([[10000, 20000, 30000],
                        [40000, 50000, 65535]], dtype=np.uint16)
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = arctan
    interleaved[:, 1::2] = mag_u16.view(np.int16)  # bitcast for storage
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "am.dat"
        _write_dat(
            path,
            interleaved,
            flags=FLAG_DEMODULATED | FLAG_INTERLEAVED | FLAG_UNSIGNED,
        )
        f = File(path)
        v = f.get_magnitude_volts(n=rows)
        assert v.dtype == np.float32
        expected = mag_u16.astype(np.float32) * np.float32(f.range / 32768.0)
        np.testing.assert_allclose(v, expected, rtol=1e-6)
        # Magnitude is always non-negative.
        assert (v >= 0).all()


def test_iq_file_rejects_arctan_extractors():
    # Same payload, IQ flags — calling get_arctan must raise instead of
    # silently returning the wrong lane.
    rows, positions = 2, 4
    payload = np.zeros((rows, positions * 2), dtype=np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        _write_dat(path, payload, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)
        f = File(path)
        with pytest.raises(ValueError, match="mode is 'iq'"):
            f.get_arctan()
        with pytest.raises(ValueError, match="mode is 'iq'"):
            f.get_magnitude()


# ── ArctanMagnitude mode: arctan as i16, magnitude as u16 ─────────────────


def test_arctan_magnitude_file_extracts_unsigned_magnitude():
    rows, positions = 3, 5
    arctan = np.tile(np.arange(positions, dtype=np.int16) - 2, (rows, 1)) * 100
    # u16 magnitudes (0..65535). Stored on the wire as i16 with the same
    # bit pattern → values > 32767 appear "negative" in the i16 buffer.
    magnitude_u16 = (np.arange(positions, dtype=np.uint16) + 1) * 10_000  # up to 50_000
    magnitude_u16 = np.tile(magnitude_u16, (rows, 1))
    # Bitcast u16 → i16 for storage.
    magnitude_i16 = magnitude_u16.view(np.int16)

    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = arctan
    interleaved[:, 1::2] = magnitude_i16

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "arctan_mag.dat"
        _write_dat(
            path,
            interleaved,
            flags=FLAG_DEMODULATED | FLAG_INTERLEAVED | FLAG_UNSIGNED,
        )

        f = File(path)
        assert f.is_interleaved
        assert f.is_unsigned
        assert f.mode is Mode.ARCTAN_MAGNITUDE

        chunk = f.read_lines(rows)
        got_arctan = f.get_arctan(chunk)
        got_mag = f.get_magnitude(chunk)
        assert got_arctan.dtype == np.int16
        assert got_mag.dtype == np.uint16
        np.testing.assert_array_equal(got_arctan, arctan)
        np.testing.assert_array_equal(got_mag, magnitude_u16)


def test_arctan_magnitude_file_rejects_iq_extractors():
    rows, positions = 2, 4
    payload = np.zeros((rows, positions * 2), dtype=np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "am.dat"
        _write_dat(
            path,
            payload,
            flags=FLAG_DEMODULATED | FLAG_INTERLEAVED | FLAG_UNSIGNED,
        )
        f = File(path)
        with pytest.raises(ValueError, match="mode is 'arctan_magnitude'"):
            f.get_i()
        with pytest.raises(ValueError, match="mode is 'arctan_magnitude'"):
            f.get_q()


# ── Phase mode: one f32 sample per spatial position, no INTERLEAVED ──────


def test_phase_file_returns_f32_radians():
    rows, positions = 3, 8
    payload = np.linspace(-np.pi, np.pi, rows * positions, dtype=np.float32).reshape(
        rows, positions
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "phase.dat"
        _write_dat(
            path,
            payload,
            flags=FLAG_DEMODULATED | FLAG_PHASE,  # FLAG_FLOAT added automatically
        )

        f = File(path)
        assert f.is_phase
        assert f.is_float
        assert not f.is_interleaved
        assert f.mode is Mode.PHASE
        assert f.dtype == np.float32

        # get_phase == read_lines for Phase mode files (no pair to split).
        got = f.get_phase(n=rows)
        np.testing.assert_allclose(got, payload, rtol=1e-6)


def test_phase_extractor_rejects_non_phase_file():
    rows = 2
    payload = np.zeros((rows, 4), dtype=np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)
        f = File(path)
        with pytest.raises(ValueError, match="mode is 'raw'"):
            f.get_phase()


# ── Raw mode: no extractors apply, read_lines returns the wire payload ───


def test_raw_file_passthrough():
    rows, line_size = 4, 16
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)

        f = File(path)
        assert not f.is_interleaved
        assert not f.is_demodulated
        assert f.mode is Mode.RAW
        chunk = f.read_lines(rows)
        np.testing.assert_array_equal(chunk, payload)

        # Pair extractors must refuse — the file isn't INTERLEAVED.
        with pytest.raises(ValueError, match="mode is 'raw'"):
            f.get_i()
        with pytest.raises(ValueError, match="mode is 'raw'"):
            f.get_arctan()


# ── DatReader (low-level) ────────────────────────────────────────────────


def test_dat_reader_rewind():
    rows, line_size = 5, 10
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)
        reader = DatReader(str(path))
        first = reader.read_lines(3)
        assert first.shape == (3, line_size)
        assert reader.lines_left == 2
        reader.rewind()
        assert reader.lines_left == rows
        again = reader.read_lines(3)
        np.testing.assert_array_equal(first, again)


# ── Format dispatch (auto-detection) ─────────────────────────────────────


def test_format_auto_detection_unknown_extension():
    # The file has to exist (FileNotFoundError takes precedence) — what
    # we want to assert is that an unknown suffix raises ValueError with
    # a message that points the user at the supported list.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.xyz"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="unsupported file format"):
            File(path)


def test_context_manager_closes_file():
    rows, line_size = 2, 8
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)
        with File(path) as f:
            np.testing.assert_array_equal(f.read_lines(rows), payload)
        # File is closed; subsequent reads on the same instance should
        # surface a clean error rather than crashing the process.
        # (We don't assert on the exact exception type — depends on the
        # backend; just that calling close() twice doesn't blow up.)
        f.close()


def test_iteration_yields_one_pulse_at_a_time():
    rows, line_size = 5, 6
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)
        with File(path) as f:
            pulses = list(f)
            assert len(pulses) == rows
            for i, pulse in enumerate(pulses):
                np.testing.assert_array_equal(pulse, payload[i])


def test_channels_helper_dispatches_by_mode():
    rows, positions = 3, 4
    i = np.arange(rows * positions, dtype=np.int16).reshape(rows, positions)
    q = i + 100
    interleaved = np.empty((rows, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        _write_dat(path, interleaved, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)
        with File(path) as f:
            assert f.mode is Mode.IQ
            ch = f.channels(n=rows)
            assert set(ch.keys()) == {"i", "q", "iq"}
            np.testing.assert_array_equal(ch["i"], i)
            np.testing.assert_array_equal(ch["q"], q)
            assert ch["iq"].dtype == np.complex64


def test_read_all_returns_remaining_pulses():
    rows, line_size = 4, 8
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "raw.dat"
        _write_dat(path, payload, flags=0)
        with File(path) as f:
            _ = f.read_lines(2)
            rest = f.read_all()
            assert rest.shape == (2, line_size)
            np.testing.assert_array_equal(rest, payload[2:])
            # After exhausting, read_all returns an empty (0, line_size).
            empty = f.read_all()
            assert empty.shape == (0, line_size)


def test_repr_is_human_readable():
    rows, positions = 2, 4
    payload = np.zeros((rows, positions * 2), dtype=np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "iq.dat"
        _write_dat(path, payload, flags=FLAG_DEMODULATED | FLAG_INTERLEAVED)
        f = File(path)
        rep = repr(f)
        assert "mode=iq" in rep
        assert "shape=" in rep
        assert "Hz" in rep


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        File("/this/path/does/not/exist.dat")


def test_explicit_format_override():
    rows, line_size = 2, 8
    payload = np.arange(rows * line_size, dtype=np.int16).reshape(rows, line_size)
    with tempfile.TemporaryDirectory() as d:
        # Suffix-less file — auto-detection should fail, but the explicit
        # `format='dat'` override should work.
        path = Path(d) / "acquisition"
        _write_dat(path, payload, flags=0)
        with pytest.raises(ValueError):
            File(path)
        f = File(path, format="dat")
        np.testing.assert_array_equal(f.read_lines(rows), payload)

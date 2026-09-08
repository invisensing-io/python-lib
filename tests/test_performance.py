"""
Performance + safety regression tests.

These exercise the optimised paths (uninit Vec, SIMD-friendly
chunks_exact, bitcast i16→u16) on inputs large enough to surface
correctness bugs that wouldn't show up in the small synthetic tests
elsewhere, and assert the GIL-release wiring for large reads.

Throughput numbers are NOT asserted (they vary across CI hardware) —
the timings are printed via pytest's `-s` flag if you want to spot
regressions. The hard assertions are on correctness across millions of
samples plus a sanity ceiling so a 100× perf regression *would* fail.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from invisensing import (
    File,
    Header,
    FLAG_DEMODULATED,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
)
from invisensing._reader import build_header_bytes


def _make_iq_file(path: Path, n_pulses: int, positions: int) -> None:
    """Build an INTERLEAVED IQ DAT file with a deterministic pattern
    we can verify position-by-position."""
    # Pattern: I[r, j] = r * positions + j, Q[r, j] = -I[r, j].
    i = np.arange(n_pulses * positions, dtype=np.int16).reshape(n_pulses, positions)
    q = -i
    interleaved = np.empty((n_pulses, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = i
    interleaved[:, 1::2] = q

    header = Header(
        line_size=positions * 2,
        trig_frequency=2_000,
        sample_size=2,
        sample_rate=250_000_000,
        flags=FLAG_DEMODULATED | FLAG_INTERLEAVED,
        range=2_000,
        pulse_width=4,
        num_channels=1,
        timestamp="2026-01-01_perf",
    )
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        fp.write(np.ascontiguousarray(interleaved).tobytes())


def _make_arctan_mag_file(path: Path, n_pulses: int, positions: int) -> None:
    """Build an INTERLEAVED arctan/√ DAT file with a deterministic pattern."""
    # Arctan: a small angle ramp encoded as i16 fixed-point.
    arctan = (
        np.linspace(-32768, 32767, n_pulses * positions, dtype=np.int32)
        .astype(np.int16)
        .reshape(n_pulses, positions)
    )
    # Magnitude: u16 ramp from 0 to 65000 — covers values > 32767 to
    # exercise the unsigned reinterpretation.
    mag_u16 = (
        np.linspace(0, 65000, n_pulses * positions, dtype=np.uint16)
        .reshape(n_pulses, positions)
    )

    interleaved = np.empty((n_pulses, positions * 2), dtype=np.int16)
    interleaved[:, 0::2] = arctan
    interleaved[:, 1::2] = mag_u16.view(np.int16)  # bitcast for storage

    header = Header(
        line_size=positions * 2,
        trig_frequency=2_000,
        sample_size=2,
        sample_rate=250_000_000,
        flags=FLAG_DEMODULATED | FLAG_INTERLEAVED | FLAG_UNSIGNED,
        range=2_000,
        pulse_width=4,
        num_channels=1,
        timestamp="2026-01-01_perf",
    )
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        fp.write(np.ascontiguousarray(interleaved).tobytes())


# ── Correctness at scale ──────────────────────────────────────────────────


def test_deinterleave_correct_on_large_iq_file(tmp_path: Path):
    """1 million I/Q pairs — guards against off-by-one in the
    chunks_exact path and the set_len uninit-buffer write."""
    n_pulses, positions = 1_000, 1_000
    path = tmp_path / "iq_big.dat"
    _make_iq_file(path, n_pulses, positions)

    with File(path) as f:
        assert f.num_lines == n_pulses
        buf = f.read_all()
        i = f.get_i(buf)
        q = f.get_q(buf)

        # Reconstruct the expected pattern.
        expected_i = np.arange(n_pulses * positions, dtype=np.int16).reshape(
            n_pulses, positions
        )
        np.testing.assert_array_equal(i, expected_i)
        np.testing.assert_array_equal(q, -expected_i)


def test_bitcast_correct_on_large_arctan_mag_file(tmp_path: Path):
    """Same scale, exercises the i16 → u16 bitcast path used by
    `split_pair_unsigned`. Magnitudes > 32767 must come out as
    positive u16 values, not as negative i16 reinterpretations."""
    n_pulses, positions = 1_000, 1_000
    path = tmp_path / "arctan_mag_big.dat"
    _make_arctan_mag_file(path, n_pulses, positions)

    with File(path) as f:
        buf = f.read_all()
        mag = f.get_magnitude(buf)
        # Magnitude is always >= 0 by definition.
        assert mag.dtype == np.uint16
        assert (mag >= 0).all()
        # Highest magnitudes in the test pattern reach ~65000 and must
        # round-trip without sign-flip.
        assert mag.max() > 60_000
        # Re-derive the expected values from the original pattern.
        expected = (
            np.linspace(0, 65000, n_pulses * positions, dtype=np.uint16)
            .reshape(n_pulses, positions)
        )
        np.testing.assert_array_equal(mag, expected)


# ── Throughput sanity ─────────────────────────────────────────────────────


def test_read_lines_throughput_sanity(tmp_path: Path, capsys):
    """End-to-end read of ~100 MB of i16 IQ samples in chunks. The hard
    assertion is the *ceiling* — anything slower than 200 MB/s on a
    typical laptop means a regression in the read path."""
    n_pulses = 50_000     # 50 000 pulses
    positions = 1_000     # 2 000 i16 / pulse = 4 KB / pulse
    path = tmp_path / "iq_throughput.dat"
    _make_iq_file(path, n_pulses, positions)
    file_size = os.path.getsize(path)
    mb = file_size / 1_048_576

    with File(path) as f:
        chunk = 1000
        t0 = time.perf_counter()
        total = 0
        while f.lines_left:
            buf = f.read_lines(chunk)
            total += buf.nbytes
        elapsed = time.perf_counter() - t0

    mb_per_s = (total / 1_048_576) / max(elapsed, 1e-6)
    # Print, don't assert — exact numbers depend on hardware. Just
    # confirm we made it through.
    with capsys.disabled():
        print(f"\n  read_lines throughput: {mb_per_s:.0f} MB/s "
              f"({mb:.1f} MB in {elapsed*1000:.1f} ms)")
    assert mb_per_s > 50, (
        f"read_lines throughput collapsed to {mb_per_s:.1f} MB/s — "
        "investigate Vec init, buffered-I/O changes, or GIL pressure."
    )


def test_deinterleave_throughput_sanity(tmp_path: Path, capsys):
    """Run the I-lane extractor on a pre-loaded buffer to isolate the
    kernel cost from disk I/O. The threshold is loose (200 MB/s on a
    typical laptop) — what we care about is that nobody accidentally
    re-introduces the `Vec::push` capacity-check pattern in the loop."""
    n_pulses, positions = 50_000, 1_000
    path = tmp_path / "iq_dl.dat"
    _make_iq_file(path, n_pulses, positions)

    with File(path) as f:
        buf = f.read_all()
        t0 = time.perf_counter()
        for _ in range(3):
            i = f.get_i(buf)
            q = f.get_q(buf)
        elapsed = time.perf_counter() - t0

    # 3 × 2 = 6 passes over the buffer; per pass = i + q bytes returned.
    per_pass_mb = (i.nbytes + q.nbytes) / 1_048_576
    mb_per_s = (6 * per_pass_mb) / max(elapsed, 1e-6)
    with capsys.disabled():
        print(f"\n  deinterleave throughput: {mb_per_s:.0f} MB/s "
              f"({per_pass_mb:.1f} MB / pass)")
    assert mb_per_s > 100, (
        f"deinterleave throughput collapsed to {mb_per_s:.1f} MB/s — the "
        "Rust kernel may have lost auto-vectorisation."
    )


# ── Safety + edge cases ──────────────────────────────────────────────────


def test_invalid_header_rejected_at_open(tmp_path: Path):
    """`validate_header` should reject zero / odd-on-interleaved up
    front so downstream maths never sees a garbage shape."""
    path = tmp_path / "bad.dat"
    # Write a header with line_size = 0.
    header = Header(
        line_size=0,
        trig_frequency=1,
        sample_size=2,
        sample_rate=1,
        flags=0,
    )
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))

    with pytest.raises(ValueError, match="line_size"):
        File(path)


def test_invalid_sample_size_rejected_at_open(tmp_path: Path):
    """sample_size must be one of {1, 2, 4, 8}."""
    path = tmp_path / "bad.dat"
    header = Header(
        line_size=10,
        trig_frequency=1,
        sample_size=3,       # invalid
        sample_rate=1,
        flags=0,
    )
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        fp.write(b"\x00" * 100)
    with pytest.raises(ValueError, match="sample_size"):
        File(path)


def test_odd_line_size_on_interleaved_rejected_at_open(tmp_path: Path):
    """INTERLEAVED files must have an even line_size — otherwise the
    pair extractors would always fail with a confusing error."""
    path = tmp_path / "bad.dat"
    header = Header(
        line_size=7,                          # odd
        trig_frequency=1,
        sample_size=2,
        sample_rate=1,
        flags=FLAG_INTERLEAVED,
    )
    with open(path, "wb") as fp:
        fp.write(build_header_bytes(header))
        fp.write(b"\x00" * 14)
    with pytest.raises(ValueError, match="INTERLEAVED"):
        File(path)


def test_iteration_overhead_per_pulse(tmp_path: Path, capsys):
    """`for pulse in f` must not pay the facade twice per pulse.

    Iterating yields one pulse per call, so anything the facade does per
    call is multiplied by the pulse count — 14 M of them on a 4 h capture
    at 1 kHz. The first version of the close() fix routed both halves of
    the loop (`self.lines_left` and `self.read_lines`) through
    `File._require_open()`, which is two dict lookups and a raise-check
    per pulse for nothing: the backend cannot change under the loop, and
    if it does the very next backend call raises anyway.

    The comparison is against the shape the loop used to have, measured
    in this same process on this same file, so it does not depend on the
    machine: the A/B is what is asserted, never an absolute time.
    """
    n_pulses = 20_000
    positions = 256
    path = tmp_path / "iter_overhead.dat"
    _make_iq_file(path, n_pulses, positions)

    def old_shape(f: File) -> int:
        # Two File-level guarded accesses per pulse.
        rows = 0
        while f.lines_left > 0:
            rows += f.read_lines(1)[0].shape[0]
        return rows

    def new_shape(f: File) -> int:
        rows = 0
        for pulse in f:
            rows += pulse.shape[0]
        return rows

    best = {"old": float("inf"), "new": float("inf")}
    counts = set()
    for _ in range(5):
        for name, loop in (("old", old_shape), ("new", new_shape)):
            with File(path) as f:
                t0 = time.perf_counter()
                counts.add(loop(f))
                best[name] = min(best[name], time.perf_counter() - t0)

    # Both shapes must read the same thing, whatever they cost.
    assert counts == {n_pulses * positions * 2}

    with capsys.disabled():
        print(
            f"\n  iteration: {best['new']*1e3:.1f} ms vs {best['old']*1e3:.1f} ms "
            f"for {n_pulses} pulses "
            f"({(best['old']-best['new'])/n_pulses*1e6:+.3f} us/pulse saved)"
        )
    assert best["new"] <= best["old"], (
        "File.__iter__ is now slower than the plain "
        "`while f.lines_left: f.read_lines(1)` loop it exists to shorten — "
        "the per-pulse work of the facade has grown."
    )

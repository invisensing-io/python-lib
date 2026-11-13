"""
The phase to strain and strain-rate kernels of the native core
(``src/strain.rs``).

Strain: ``factor = NANOSTRAIN_PER_RAD_METRE / gauge_length_m`` once in
float64, rounded once to float32 for float32 samples, then one IEEE multiply
per sample (numpy's ``x * np.float32(factor)`` is that multiply).

Strain rate: ``factor = NANOSTRAIN_PER_RAD_METRE / gauge_length_m ×
trig_frequency_hz`` the same way, then per sample one IEEE subtraction of the
previous line and one multiply (numpy's ``(x[n] - x[n-1]) * factor``).

Every SIMD kernel this CPU runs must give the scalar kernel's bits — on every
length, from unaligned starts, on special values — and leave the amplitude
lanes of pairs untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from invisensing._core import (
    NANOSTRAIN_PER_RAD_METRE,
    nanostrain_factor,
    phase_to_nanostrain,
    phase_to_strain_rate,
    simd_level,
    simd_levels,
    strain_rate_factor,
)

LEVELS = simd_levels()
GAUGES = [0.4, 1.6, 9.6, 10.0, 20.4, 102.4]


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """The numpy reference multiplies NaNs and overflows on purpose."""
    with np.errstate(all="ignore"):
        yield


def _random_bits_f32(rng, n: int) -> np.ndarray:
    """Every bit pattern class: random finite values, random bit patterns
    (NaN payloads, subnormals, infinities), and the special values."""
    finite = (rng.standard_normal(n) * 40.0).astype(np.float32)
    bits = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    pick = rng.integers(0, 3, n)
    out = np.where(pick == 0, finite, bits).astype(np.float32)
    special = np.array(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 1.17e-38, 3.4e38, -3.4e38],
        dtype=np.float32,
    )
    k = min(n, special.size)
    out[:k] = special[:k]
    return out


def _random_bits_f64(rng, n: int) -> np.ndarray:
    finite = rng.standard_normal(n) * 40.0
    bits = rng.integers(0, np.iinfo(np.uint64).max, n, dtype=np.uint64, endpoint=True).view(np.float64)
    pick = rng.integers(0, 3, n)
    out = np.where(pick == 0, finite, bits)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, 2.2e-308, 1.7e308])
    k = min(n, special.size)
    out[:k] = special[:k]
    return out


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def test_the_constant_and_the_factor():
    assert NANOSTRAIN_PER_RAD_METRE == 110.37
    for g in GAUGES:
        assert nanostrain_factor(g) == float(np.float32(NANOSTRAIN_PER_RAD_METRE / g))
    for bad in (0.0, -1.0, float("nan"), float("inf"), 1e-40):
        with pytest.raises(ValueError, match="not a gauge length"):
            nanostrain_factor(bad)


def test_the_tiers_this_cpu_runs():
    assert LEVELS[0] == "scalar"
    assert set(LEVELS) <= {"scalar", "avx2", "avx512"}
    assert simd_level() in LEVELS
    flags = ""
    try:
        with open("/proc/cpuinfo") as fp:
            flags = fp.read()
    except OSError:
        pass
    if " avx512f" in flags:
        assert LEVELS == ["scalar", "avx2", "avx512"]
    elif " avx2" in flags:
        assert LEVELS == ["scalar", "avx2"]


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("pairs", [False, True])
def test_every_tier_gives_the_reference_bits_on_every_length(level, pairs):
    rng = np.random.default_rng(1)
    for g in GAUGES:
        f32 = np.float32(NANOSTRAIN_PER_RAD_METRE / g)
        f64 = NANOSTRAIN_PER_RAD_METRE / g
        for n in range(0, 202, 2 if pairs else 1):
            for offset in range(3):  # unaligned starts
                buf = _random_bits_f32(rng, n + offset)
                block = buf[offset:]
                want = block.copy()
                if pairs:
                    want[0::2] = block[0::2] * f32
                else:
                    want = block * f32
                ref = block.copy()
                assert phase_to_nanostrain(ref, g, interleaved=pairs, simd="scalar") == "scalar"
                assert np.array_equal(_bits(ref), _bits(want)), (g, n, offset)
                assert phase_to_nanostrain(block, g, interleaved=pairs, simd=level) == level
                assert np.array_equal(_bits(block), _bits(ref)), (level, g, n, offset)

                buf64 = _random_bits_f64(rng, n + offset)
                block64 = buf64[offset:]
                want64 = block64.copy()
                if pairs:
                    want64[0::2] = block64[0::2] * f64
                else:
                    want64 = block64 * f64
                phase_to_nanostrain(block64, g, interleaved=pairs, simd=level)
                # NaN payloads: numpy's float64 multiply keeps them as the
                # hardware does; compare the bits of everything else.
                nan = np.isnan(want64)
                assert np.array_equal(np.isnan(block64), nan)
                assert np.array_equal(_bits(block64[~nan]), _bits(want64[~nan])), (level, g, n)


@pytest.mark.parametrize("level", LEVELS)
def test_every_tier_gives_the_scalar_bits_on_large_arrays(level):
    rng = np.random.default_rng(2)
    for shape, pairs in [(((1 << 20) + 7,), False), ((1025, 2 * 517), True)]:
        base = _random_bits_f32(rng, int(np.prod(shape))).reshape(shape)
        ref = base.copy()
        phase_to_nanostrain(ref, 9.6, interleaved=pairs, simd="scalar")
        got = base.copy()
        phase_to_nanostrain(got, 9.6, interleaved=pairs, simd=level)
        assert np.array_equal(_bits(got), _bits(ref))
        if pairs:
            assert np.array_equal(_bits(got[:, 1::2]), _bits(base[:, 1::2]))


def test_the_amplitude_lanes_keep_their_bits():
    for level in LEVELS:
        a = np.zeros((3, 40), dtype=np.float32)
        a[:, 1::2] = np.array([0x7F800001, 0xFFC01234, 0x00000001], dtype=np.uint32).view(np.float32)[:, None]
        a[:, 0::2] = 1.0
        before = a.copy()
        phase_to_nanostrain(a, 10.0, interleaved=True, simd=level)
        assert np.array_equal(_bits(a[:, 1::2]), _bits(before[:, 1::2])), level
        assert np.all(a[:, 0::2] == np.float32(NANOSTRAIN_PER_RAD_METRE / 10.0))


def test_what_the_kernel_refuses():
    a = np.ones((4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="C-contiguous"):
        phase_to_nanostrain(a[:, ::2], 9.6)
    with pytest.raises(ValueError, match="even last dimension"):
        phase_to_nanostrain(np.ones((2, 5), np.float32), 9.6, interleaved=True)
    with pytest.raises(TypeError, match="float32 or float64"):
        phase_to_nanostrain(np.ones(4, np.int16), 9.6)
    with pytest.raises(TypeError, match="float32 or float64"):
        phase_to_nanostrain(np.ones(4, np.dtype(">f4")), 9.6)
    with pytest.raises(ValueError, match="not a gauge length"):
        phase_to_nanostrain(a, 0.0)
    with pytest.raises(ValueError, match="simd must be"):
        phase_to_nanostrain(a, 9.6, simd="sse4")
    ro = np.ones(4, np.float32)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        phase_to_nanostrain(ro, 9.6)
    assert np.all(a == 1.0), "nothing converted by a refusal"


# ── Strain rate ────────────────────────────────────────────────────────────

#: Line rates: exact, the inexact 75 kHz of a 250 MHz clock, the DAS ceiling.
RATES = [50.0, 2_000.0, 250e6 / 3_333, 100_160.0]


def _rate_reference(block: np.ndarray, prev: np.ndarray, factor, pairs: bool) -> np.ndarray:
    """numpy's `(x[n] − x[n−1]) × factor`, line −1 being `prev`."""
    before = np.concatenate([prev[None, :], block[:-1]]) if block.shape[0] else block
    want = block.copy()
    lanes = slice(0, None, 2) if pairs else slice(None)
    want[:, lanes] = (block[:, lanes] - before[:, lanes]) * factor
    return want


def test_the_strain_rate_factor():
    for g in GAUGES:
        for hz in RATES:
            assert strain_rate_factor(g, hz) == float(np.float32(NANOSTRAIN_PER_RAD_METRE / g * hz))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not a line rate"):
            strain_rate_factor(9.6, bad)
    with pytest.raises(ValueError, match="not a gauge length"):
        strain_rate_factor(float("nan"), 2_000.0)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("pairs", [False, True])
def test_every_tier_gives_the_reference_strain_rate_on_every_length(level, pairs):
    rng = np.random.default_rng(4)
    for g, hz in [(1.6, 2_000.0), (9.6, 250e6 / 3_333), (20.4, 100_160.0)]:
        f32 = np.float32(NANOSTRAIN_PER_RAD_METRE / g * hz)
        f64 = NANOSTRAIN_PER_RAD_METRE / g * hz
        for line in range(2 if pairs else 1, 202, 2 if pairs else 1):
            for rows in (0, 1, 3):
                for offset in range(3):  # unaligned starts of both buffers
                    buf = _random_bits_f32(rng, rows * line + offset)
                    block = buf[offset:].reshape(rows, line)
                    pbuf = _random_bits_f32(rng, line + offset)
                    prev = pbuf[offset:]
                    want = _rate_reference(block, prev, f32, pairs)
                    want_prev = block[-1].copy() if rows else prev.copy()
                    ref, ref_prev = block.copy(), prev.copy()
                    used = phase_to_strain_rate(ref, ref_prev, g, hz, interleaved=pairs, simd="scalar")
                    assert used == "scalar"
                    assert np.array_equal(_bits(ref), _bits(want)), (g, line, rows, offset)
                    assert np.array_equal(_bits(ref_prev), _bits(want_prev))
                    assert phase_to_strain_rate(block, prev, g, hz, interleaved=pairs, simd=level) == level
                    assert np.array_equal(_bits(block), _bits(ref)), (level, g, line, rows, offset)
                    assert np.array_equal(_bits(prev), _bits(ref_prev)), (level, g, line, rows)

                    buf64 = _random_bits_f64(rng, rows * line + offset)
                    block64 = buf64[offset:].reshape(rows, line)
                    prev64 = _random_bits_f64(rng, line + offset)[offset:]
                    want64 = _rate_reference(block64, prev64, f64, pairs)
                    ref64, ref_prev64 = block64.copy(), prev64.copy()
                    phase_to_strain_rate(ref64, ref_prev64, g, hz, interleaved=pairs, simd="scalar")
                    phase_to_strain_rate(block64, prev64, g, hz, interleaved=pairs, simd=level)
                    assert np.array_equal(_bits(block64), _bits(ref64)), (level, g, line, rows)
                    assert np.array_equal(_bits(prev64), _bits(ref_prev64))
                    # numpy against the scalar kernel: NaN payloads aside
                    # (see the strain test), the same bits.
                    nan = np.isnan(want64)
                    assert np.array_equal(np.isnan(ref64), nan)
                    assert np.array_equal(_bits(ref64[~nan]), _bits(want64[~nan])), (g, line, rows)


@pytest.mark.parametrize("level", LEVELS)
def test_every_tier_gives_the_scalar_strain_rate_on_large_arrays(level):
    rng = np.random.default_rng(5)
    for (rows, line), pairs in [((1025, 1031), False), ((517, 2 * 1001), True), ((2, (1 << 20) + 7), False)]:
        base = _random_bits_f32(rng, rows * line).reshape(rows, line)
        prev = _random_bits_f32(rng, line)
        ref, ref_prev = base.copy(), prev.copy()
        phase_to_strain_rate(ref, ref_prev, 9.6, 2_000.0, interleaved=pairs, simd="scalar")
        got, got_prev = base.copy(), prev.copy()
        phase_to_strain_rate(got, got_prev, 9.6, 2_000.0, interleaved=pairs, simd=level)
        assert np.array_equal(_bits(got), _bits(ref))
        assert np.array_equal(_bits(got_prev), _bits(ref_prev))
        assert np.array_equal(_bits(got_prev), _bits(base[-1]))
        if pairs:
            assert np.array_equal(_bits(got[:, 1::2]), _bits(base[:, 1::2]))


@pytest.mark.parametrize("level", LEVELS)
def test_blocks_carry_the_previous_line(level):
    """Any split into blocks, the carried line passed along, gives the bits
    of one block: the chunk boundaries never show."""
    rng = np.random.default_rng(6)
    rows, line = 37, 70
    src = _random_bits_f32(rng, rows * line).reshape(rows, line)
    for pairs in (False, True):
        whole, wprev = src.copy(), np.full(line, np.nan, np.float32)
        phase_to_strain_rate(whole, wprev, 9.6, 2_000.0, interleaved=pairs, simd="scalar")
        for size in range(1, rows + 1):
            prev = np.full(line, np.nan, np.float32)
            parts = []
            for r in range(0, rows, size):
                b = src[r:r + size].copy()
                phase_to_strain_rate(b, prev, 9.6, 2_000.0, interleaved=pairs, simd=level)
                parts.append(b)
            assert np.array_equal(_bits(np.concatenate(parts)), _bits(whole)), (pairs, size)
            assert np.array_equal(_bits(prev), _bits(src[-1]))
        lanes = slice(0, None, 2) if pairs else slice(None)
        assert np.isnan(whole[0, lanes]).all(), "after a NaN line, the phase lanes are NaN"
        if pairs:
            assert np.array_equal(_bits(whole[0, 1::2]), _bits(src[0, 1::2]))


def test_the_amplitude_lanes_keep_their_bits_in_the_strain_rate():
    for level in LEVELS:
        a = np.zeros((3, 40), dtype=np.float32)
        a[:, 1::2] = np.array([0x7F800001, 0xFFC01234, 0x00000001], dtype=np.uint32).view(np.float32)[:, None]
        a[:, 0::2] = np.arange(3, dtype=np.float32)[:, None]
        before = a.copy()
        prev = np.full(40, -1.0, np.float32)
        phase_to_strain_rate(a, prev, 10.0, 2_000.0, interleaved=True, simd=level)
        assert np.array_equal(_bits(a[:, 1::2]), _bits(before[:, 1::2])), level
        assert np.all(a[:, 0::2] == np.float32(NANOSTRAIN_PER_RAD_METRE / 10.0 * 2_000.0))
        assert np.array_equal(_bits(prev), _bits(before[-1]))


def test_what_the_strain_rate_kernel_refuses():
    a = np.ones((4, 6), dtype=np.float32)
    p = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="C-contiguous"):
        phase_to_strain_rate(a[:, ::2], p[:3], 9.6, 50.0)
    with pytest.raises(ValueError, match="one line of the array"):
        phase_to_strain_rate(a, np.zeros(5, np.float32), 9.6, 50.0)
    with pytest.raises(ValueError, match="one line of the array"):
        phase_to_strain_rate(a, np.zeros(0, np.float32), 9.6, 50.0)
    with pytest.raises(ValueError, match="even line length"):
        phase_to_strain_rate(np.ones((2, 5), np.float32), np.zeros(5, np.float32), 9.6, 50.0,
                             interleaved=True)
    with pytest.raises(TypeError, match="previous must be a native float32"):
        phase_to_strain_rate(a, np.zeros(6), 9.6, 50.0)
    with pytest.raises(TypeError, match="float32 or float64"):
        phase_to_strain_rate(np.ones((2, 6), np.int16), p, 9.6, 50.0)
    with pytest.raises(ValueError, match="not a gauge length"):
        phase_to_strain_rate(a, p, 0.0, 50.0)
    with pytest.raises(ValueError, match="not a line rate"):
        phase_to_strain_rate(a, p, 9.6, 0.0)
    with pytest.raises(ValueError, match="simd must be"):
        phase_to_strain_rate(a, p, 9.6, 50.0, simd="sse4")
    ro = np.ones(6, np.float32)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        phase_to_strain_rate(a, ro, 9.6, 50.0)
    with pytest.raises(ValueError):  # the carried line inside the block
        phase_to_strain_rate(a, a[0], 9.6, 50.0)
    assert np.all(a == 1.0) and np.all(p == 0.0), "nothing converted by a refusal"


def _level_in_subprocess(value: str) -> str:
    env = dict(os.environ, INVISENSING_SIMD=value)
    out = subprocess.run(
        [sys.executable, "-c", "from invisensing._core import simd_level; print(simd_level())"],
        env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_the_environment_caps_the_tier():
    best = LEVELS[-1]
    assert _level_in_subprocess("scalar") == "scalar"
    assert _level_in_subprocess("AVX2") == ("avx2" if "avx2" in LEVELS else "scalar")
    assert _level_in_subprocess("avx512") == best
    assert _level_in_subprocess("not-a-tier") == best

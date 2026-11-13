"""
The phase to strain kernels of the native core (``src/strain.rs``).

The reference: ``factor = NANOSTRAIN_PER_RAD_METRE / gauge_length_m`` once in
float64, rounded once to float32 for float32 samples, then one IEEE multiply
per sample (numpy's ``x * np.float32(factor)`` is that multiply). Every SIMD
kernel this CPU runs must give the scalar kernel's bits — on every length,
from unaligned starts, on special values — and leave the amplitude lanes
of pairs untouched.
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
    simd_level,
    simd_levels,
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
        assert np.all(a[:, 0::2] == np.float32(11.5))


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

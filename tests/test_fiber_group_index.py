"""
The fibre group index: the speed of light in the fibre every metre of a file
is labelled with.

The acquisition derives one integer speed from the fibre group index ``n``
it is configured with, ``v = round(c / n)`` m/s, half away from zero, and
stamps it in every file (Audace-Lib a288414, DATA_FORMAT.md §3.5): DAT
header bytes 124..128 (u32), HDF5 root attributes and TDMS properties
``light_speed_in_fiber_m_s`` (u32) and ``fiber_group_index`` (f64, the index
exactly as configured), SEG-Y line ``FIBER GROUP INDEX: n (LIGHT SPEED v
M/S)``. A file recorded before the field carries none and was acquired at
the legacy 2·10⁸ m/s (index 1.49896229): it reads exactly as it always did.
``File(path, fiber_group_index=n)`` labels a recording at another index —
the labels change, never the samples.

The numbers are the study's (benchmarks/refractive-index-2026-11-17): at
1.4682, v = 204 190 477 m/s and the pitch at 250 MS/s is 0.408381 m.
"""

from __future__ import annotations

import math
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

import invisensing
from invisensing import (
    DEFAULT_FIBER_GROUP_INDEX,
    LEGACY_FIBER_GROUP_INDEX,
    LEGACY_LIGHT_SPEED_IN_FIBER_M_S,
    NANOSTRAIN_PER_RAD_METRE,
    SPEED_OF_LIGHT_M_S,
    File,
    Header,
    export_dat,
    group_index_of_light_speed,
    light_speed_of_group_index,
    parse_header,
)
from invisensing._core import build_header_bytes

from gauge_files import SAMPLE_RATE, WRITERS, dat_header, phase_samples, write_dat, write_h5, write_sgy, write_tdms

ASSETS = Path(__file__).resolve().parent / "assets" / "v1"
FIBRE = ASSETS / "fibre"

#: v(1.4682): SMF-28e+ at 1550 nm, the acquisition's default index.
SMF28 = 204_190_477
LEGACY = LEGACY_LIGHT_SPEED_IN_FIBER_M_S
#: v(1.4675): an index its speed alone does not give back exactly.
V1475 = light_speed_of_group_index(1.4675)
RATES = (62_500_000, 125_000_000, 250_000_000, 500_000_000, 1_000_000_000)
#: The pitch at 1.4682 at those rates (the study's table, 6 decimals).
SMF28_PITCH = (1.633524, 0.816762, 0.408381, 0.204190, 0.102095)
#: Every format, with what its reader gets for the index: a DAT carries the
#: speed only.
FORMATS = (".dat", ".h5", ".tdms", ".sgy")
PHASE = 19  # dui_coherent_phase


def _phase_header(**kw) -> Header:
    base = dict(line_size=6, trig_frequency=50, sample_size=4, sample_rate=250_000_000,
                flags=0, content_type=PHASE, format_version=1)
    base.update(kw)
    return Header(**base)


# ── The speed of an index, and back ─────────────────────────────────────────


def test_the_speed_of_an_index():
    assert SPEED_OF_LIGHT_M_S == 299_792_458
    assert DEFAULT_FIBER_GROUP_INDEX == 1.4682
    # The exactness the whole chain rests on: c / 1.49896229 is 2e8 in f64.
    assert SPEED_OF_LIGHT_M_S / LEGACY_FIBER_GROUP_INDEX == 2e8
    assert light_speed_of_group_index(LEGACY_FIBER_GROUP_INDEX) == LEGACY == 200_000_000
    assert light_speed_of_group_index(1.4682) == SMF28
    assert light_speed_of_group_index(1.0) == SPEED_OF_LIGHT_M_S
    assert light_speed_of_group_index(100.0) == 2_997_925
    # And back: the two indexes the acquisition writes come back exactly.
    assert group_index_of_light_speed(SMF28) == 1.4682
    assert group_index_of_light_speed(LEGACY) == LEGACY_FIBER_GROUP_INDEX
    assert group_index_of_light_speed(0) == LEGACY_FIBER_GROUP_INDEX  # absent
    assert group_index_of_light_speed(V1475) == SPEED_OF_LIGHT_M_S / V1475


@pytest.mark.parametrize("n", [1.0, 1.0000001, 1.4620, 1.4675, 1.4676, 1.4682, LEGACY_FIBER_GROUP_INDEX,
                               1.5, 2.0, 3.3, 14.68, 99.99, 100.0])
def test_the_speed_is_rounded_half_away_from_zero(n):
    """``round(c / n)`` as Rust's ``f64::round``: in Python ``math.floor(x +
    0.5)`` for a positive x, never ``round()`` (halves to even)."""
    assert light_speed_of_group_index(n) == math.floor(SPEED_OF_LIGHT_M_S / n + 0.5)


BAD = [0.0, -1.4682, 0.999_999_9, 100.000_001, 1e9, math.nan, math.inf, -math.inf]
GOOD = [1.0, 1.0000001, 1.4682, LEGACY_FIBER_GROUP_INDEX, 1.6, 3.0, 99.99, 100.0, np.float64(1.4675), 2]


@pytest.mark.parametrize("n", BAD)
def test_an_index_outside_1_to_100_is_refused(n, tmp_path):
    """The only rule on n (user decision 2026-11-17): below 1, above 100,
    NaN or infinite is refused — with the same ValueError everywhere."""
    with pytest.raises(ValueError, match="fibre group index .* must be a number from 1 to 100"):
        light_speed_of_group_index(n)
    with pytest.raises(ValueError, match="must be a number from 1 to 100"):
        File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat", fiber_group_index=n)
    with pytest.raises(ValueError, match="must be a number from 1 to 100"):
        export_dat(tmp_path / "x.dat", np.zeros((2, 4), np.float32), sample_rate=250_000_000,
                   trig_frequency=50, content_type=PHASE, fiber_group_index=n)
    with pytest.raises(ValueError, match="must be a number from 1 to 100"):
        _phase_header().with_fiber_group_index(n)
    assert not (tmp_path / "x.dat").exists()


@pytest.mark.parametrize("n", GOOD)
def test_any_index_from_1_to_100_is_accepted(n):
    with File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat", fiber_group_index=n) as f:
        assert f.fiber_group_index == float(n)
        assert f.light_speed_in_fiber_m_s == light_speed_of_group_index(float(n))


@pytest.mark.parametrize("n", ["1.4682", True, [1.4682]])
def test_an_index_that_is_not_a_number_is_a_type_error(n):
    with pytest.raises(TypeError, match="fiber_group_index must be a number"):
        File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat", fiber_group_index=n)


# ── The pitch and the gauge at the file's speed ────────────────────────────


def test_the_pitch_at_the_default_index_is_the_study_s():
    for rate, want in zip(RATES, SMF28_PITCH):
        h = _phase_header(sample_rate=rate, light_speed_in_fiber_m_s=SMF28)
        assert h.light_speed_m_s == SMF28
        assert h.derived_spatial_step_m == SMF28 / (2 * rate)
        assert abs(h.derived_spatial_step_m - want) < 5e-7, rate
    assert abs(_phase_header(light_speed_in_fiber_m_s=SMF28).derived_spatial_step_m - 0.408381) < 1e-6


#: Odd rates too: a decimated stream, a third of a hertz.
GRID_RATES = RATES + (31_250_000, 48_828_125, 3, 1_000_003)


@pytest.mark.parametrize("rate", GRID_RATES)
def test_a_file_without_a_speed_reads_as_before_bit_for_bit(rate, tmp_path):
    """No speed (every recording made before the field) and the legacy speed
    stamped explicitly give the f64s the reader computed before the index
    existed: the pitch ``1e8 / rate``, the gauge ``k · 1e8 / rate`` read back
    from its f32 — and the old distance, ``2e8 · positions / (2 · rate)``."""
    for speed in (0, LEGACY):
        raw = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4,
                                      sample_rate=rate, light_speed=speed))
        assert raw.light_speed_m_s == LEGACY
        assert raw.derived_spatial_step_m == 1e8 / rate
        for k in (1, 2, 5, 6, 12, 13, 24, 25, 49, 50, 98, 100, 1_001, 65_536):
            g32 = np.float32(k * 1e8 / rate)
            h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4,
                                        sample_rate=rate, gauge=float(g32), light_speed=speed))
            # The reader of 1.6.0, term for term: k = round(g · rate / 1e8),
            # exact = k · 1e8 / rate when it rounds to the stored f32.
            old_k = math.floor(float(g32) * rate / 100_000_000.0 + 0.5)
            old = old_k * 100_000_000.0 / rate
            assert h.gauge_length_m == (old if np.float32(old) == g32 else float(g32)), (rate, k)
    # The distance, through a file.
    positions = 1_001
    path = tmp_path / "raw.dat"
    path.write_bytes(
        dat_header(content_type=1, line_size=positions, sample_size=2, sample_rate=rate, format_version=0)
        + np.zeros((2, positions), np.int16).tobytes()
    )
    with File(path) as f:
        assert f.light_speed_in_fiber_m_s == LEGACY
        assert f.fiber_group_index == LEGACY_FIBER_GROUP_INDEX
        assert f.distance == (200_000_000 * positions) / (2 * rate)


def test_the_gauge_at_the_default_index_reads_back_exactly():
    """A producer stamps the gauge it applied, ``k`` whole positions; at
    1.4682 a 10 m gauge is 6 / 12 / 24 / 49 / 98 positions at the five rates
    (13 / 25 / 50 / 100 at the legacy speed). The DAT's f32 reads back as the
    exact ``k · v / (2 · rate)`` the other formats store."""
    for rate, k in zip(RATES, (6, 12, 24, 49, 98)):
        exact = k * SMF28 / (2 * rate)
        h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, sample_rate=rate,
                                    gauge=float(np.float32(exact)), light_speed=SMF28))
        assert h.gauge_length_m == exact, rate
        assert abs(exact - 10.0) < 0.21
    for rate, k in zip(RATES, (6, 13, 25, 50, 100)):
        h = parse_header(dat_header(content_type=PHASE, line_size=6, sample_size=4, sample_rate=rate,
                                    gauge=float(np.float32(k * 1e8 / rate))))
        assert h.gauge_length_m == k * 1e8 / rate, rate


# ── Every format carries it ────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", FORMATS)
@pytest.mark.parametrize("speed,index", [(SMF28, 1.4682), (V1475, 1.4675), (LEGACY, LEGACY_FIBER_GROUP_INDEX), (0, None)])
def test_every_format_round_trips_the_fibre(tmp_path, suffix, speed, index):
    """DAT bytes 124..128, the HDF5 attributes, the TDMS properties, the
    SEG-Y line: the speed comes back, the index exactly where the format
    stores it (a DAT: the index of its speed), the steps and the distance
    follow the speed; a file without it is the legacy one."""
    data = phase_samples(4, 6)
    path = tmp_path / f"f{suffix}"
    if suffix == ".dat":
        write_dat(path, data, content_type=PHASE, light_speed=speed)
    else:
        writer = {".h5": write_h5, ".tdms": write_tdms, ".sgy": write_sgy}[suffix]
        writer(path, data, content_type=PHASE, light_speed=speed, group_index=index)
    v = speed or LEGACY
    with File(path) as f:
        assert f.header.light_speed_in_fiber_m_s == speed
        assert f.light_speed_in_fiber_m_s == v
        if suffix == ".dat" or index is None:
            assert f.header.fiber_group_index == 0.0
            assert f.fiber_group_index == group_index_of_light_speed(v)
        else:
            assert f.header.fiber_group_index == index
            assert f.fiber_group_index == index
        assert f.spatial_step_m == v / (2 * SAMPLE_RATE)
        assert f.distance == 6 * v / (2 * SAMPLE_RATE)
        np.testing.assert_array_equal(f.read_all().view(np.uint32), data.view(np.uint32))
        # And through an export to DAT: the speed, at bytes 124..128.
        out = tmp_path / "copy.dat"
        f.export_dat(out)
    raw = out.read_bytes()
    assert raw[124:128] == struct.pack("<I", speed)
    with File(out) as g:
        assert g.light_speed_in_fiber_m_s == v
        assert g.fiber_group_index == group_index_of_light_speed(v)
        assert g.spatial_step_m == v / (2 * SAMPLE_RATE)


def test_the_header_bytes_round_trip():
    for speed in (0, 1, LEGACY, SMF28, V1475, SPEED_OF_LIGHT_M_S, 0xFFFF_FFFF):
        h = _phase_header(light_speed_in_fiber_m_s=speed, fiber_group_index=1.4675)
        raw = build_header_bytes(h)
        assert raw[124:128] == struct.pack("<I", speed)
        back = parse_header(raw)
        assert back.light_speed_in_fiber_m_s == speed
        # A DAT stores the speed only.
        assert back.fiber_group_index == 0.0
    # Everything before byte 124 is what it was without a speed.
    plain = build_header_bytes(_phase_header())
    stamped = build_header_bytes(_phase_header(light_speed_in_fiber_m_s=LEGACY))
    assert plain[:124] == stamped[:124] and plain[124:] == bytes(4)


def test_the_header_s_index_is_believed_only_when_it_gives_its_speed():
    assert _phase_header(light_speed_in_fiber_m_s=V1475, fiber_group_index=1.4675).group_index == 1.4675
    assert _phase_header(light_speed_in_fiber_m_s=V1475, fiber_group_index=1.5).group_index == SPEED_OF_LIGHT_M_S / V1475
    assert _phase_header(fiber_group_index=1.4682).group_index == LEGACY_FIBER_GROUP_INDEX
    assert _phase_header().group_index == LEGACY_FIBER_GROUP_INDEX


def test_a_speed_no_index_gives_is_read_as_the_legacy_one(tmp_path):
    path = tmp_path / "odd.dat"
    write_dat(path, phase_samples(2, 6), content_type=PHASE, light_speed=1)
    with pytest.warns(UserWarning, match="1 m/s, is not the speed of a fibre group index"):
        f = File(path)
    with f:
        assert f.header.light_speed_in_fiber_m_s == 1
        assert f.light_speed_in_fiber_m_s == LEGACY
        assert f.fiber_group_index == LEGACY_FIBER_GROUP_INDEX
    # No warning for an absent or a valid speed.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for speed in (0, SMF28):
            write_dat(tmp_path / f"{speed}.dat", phase_samples(2, 6), content_type=PHASE, light_speed=speed)
            File(tmp_path / f"{speed}.dat").close()


def test_the_representation_names_the_index_only_when_the_file_carries_one(tmp_path):
    assert "light_speed" not in repr(_phase_header())
    assert "fiber_group_index" not in repr(_phase_header())
    assert "light_speed_in_fiber_m_s=204190477" in repr(_phase_header(light_speed_in_fiber_m_s=SMF28))
    with File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat") as f:
        assert "fiber_group_index" not in repr(f)
    with File(FIBRE / "1.4682/2026-11-03_10-00-00.h5") as f:
        assert "fiber_group_index=1.4682" in repr(f)


# ── The acquisition's own files (tests/assets/v1/fibre) ─────────────────────


@pytest.mark.parametrize("suffix", (".dat", ".h5", ".tdms", ".sgy"))
def test_the_acquisition_s_files_at_the_default_index(suffix):
    """Written by Audace-Lib a288414 at 1.4682: a phase of 6 positions at
    250 MS/s with a 24-position gauge, the pre-roll of 1 224 samples at
    125 MS/s."""
    with File(FIBRE / f"1.4682/2026-11-03_10-00-00{suffix}") as f:
        assert f.light_speed_in_fiber_m_s == SMF28
        assert f.fiber_group_index == 1.4682
        assert f.spatial_step_m == SMF28 / (2 * 250_000_000)
        assert abs(f.spatial_step_m - 0.408381) < 1e-6
        assert f.gauge_length_m == 24 * SMF28 / (2 * 250_000_000)
        assert f.distance == pytest.approx(6 * 0.408380954)
        assert f.trigger_delay == 999  # ⌊1224 · v / 2.5e8⌋, 1 250 samples at 2e8
        assert f.nanostrain_per_rad == np.float32(NANOSTRAIN_PER_RAD_METRE / f.gauge_length_m)


@pytest.mark.parametrize("suffix", (".dat", ".h5", ".tdms", ".sgy"))
def test_an_index_its_speed_does_not_give_back(suffix):
    """1.4675 is stored exactly where the format stores an index (HDF5,
    TDMS, SEG-Y); a DAT gives the index of its speed."""
    with File(FIBRE / f"1.4675/2026-11-03_10-00-00{suffix}") as f:
        assert f.light_speed_in_fiber_m_s == V1475
        want = SPEED_OF_LIGHT_M_S / V1475 if suffix == ".dat" else 1.4675
        assert f.fiber_group_index == want
        assert f.spatial_step_m == V1475 / (2 * 250_000_000)


def test_the_hdf5_attributes_are_the_writer_s():
    import h5py

    with h5py.File(FIBRE / "1.4682/2026-11-03_10-00-00.h5", "r") as fp:
        assert fp.attrs["light_speed_in_fiber_m_s"] == SMF28
        assert fp.attrs["light_speed_in_fiber_m_s"].dtype == np.uint32
        assert fp.attrs["fiber_group_index"] == 1.4682
    with h5py.File(ASSETS / "split_h5/phase/2026-11-03_10-00-00.h5", "r") as fp:
        assert "light_speed_in_fiber_m_s" not in fp.attrs
        assert "fiber_group_index" not in fp.attrs


# ── Relabelling: File(path, fiber_group_index=n) ────────────────────────────


@pytest.mark.parametrize("suffix", ("dat", "h5"))
def test_an_old_recording_relabelled_at_the_default_index(suffix):
    """A recording made before the field (legacy 2e8 m/s: 62.5 MS/s, 1.6 m,
    a 6-position gauge of 9.6 m, a 120 m pre-roll) labelled at 1.4682: every
    metre is 2.0952 % longer, the samples are the same bits."""
    path = ASSETS / f"split_{suffix}/phase/2026-11-03_10-00-00.{suffix}"
    with File(path) as plain:
        samples = plain.read_all()
        assert plain.light_speed_in_fiber_m_s == LEGACY
        assert (plain.spatial_step_m, plain.gauge_length_m, plain.trigger_delay) == (1.6, 9.6, 120)
    with File(path, fiber_group_index=1.4682) as f:
        assert f.light_speed_in_fiber_m_s == SMF28
        assert f.fiber_group_index == 1.4682
        assert f.recorded_fiber_group_index == LEGACY_FIBER_GROUP_INDEX
        assert f.recorded_light_speed_in_fiber_m_s == LEGACY
        assert f.spatial_step_m == SMF28 / (2 * 62_500_000)
        assert abs(f.spatial_step_m - 1.633524) < 1e-6
        assert f.gauge_length_m == 6 * SMF28 / (2 * 62_500_000)
        assert f.distance == pytest.approx(6 * f.spatial_step_m)
        assert f.trigger_delay == 123  # 120 m × 1.020952, to the metre
        # The relative relabelling of the study: × 1.49896229 / 1.4682.
        assert f.distance / plain.distance == pytest.approx(LEGACY_FIBER_GROUP_INDEX / 1.4682, rel=3e-9)
        # The strain factor follows the gauge (a length); the constant does not.
        assert f.nanostrain_per_rad == np.float32(NANOSTRAIN_PER_RAD_METRE / f.gauge_length_m)
        assert invisensing.NANOSTRAIN_PER_RAD_METRE == 110.37
        np.testing.assert_array_equal(f.read_all().view(np.uint32), samples.view(np.uint32))
        # Everything that is not a metre is untouched.
        for field in ("line_size", "sample_rate", "trig_frequency", "content_type", "acquisition_ns",
                      "trig_clock_hz", "trig_period_ticks", "temporal_step_s", "duration_s", "card"):
            assert getattr(f.header, field) == getattr(plain.header, field), field


def test_relabelling_at_the_file_s_own_index_changes_nothing_but_the_index():
    path = FIBRE / "1.4682/2026-11-03_10-00-00.dat"
    with File(path) as plain, File(path, fiber_group_index=1.4682) as same:
        for field in ("light_speed_m_s", "spatial_step_m", "gauge_length_m", "trigger_delay"):
            assert getattr(same.header, field) == getattr(plain.header, field), field
        assert same.header.fiber_group_index == 1.4682 and plain.header.fiber_group_index == 0.0
        assert same.distance == plain.distance


def test_a_new_recording_relabelled_at_the_legacy_index():
    with File(FIBRE / "1.4682/2026-11-03_10-00-00.h5", fiber_group_index=LEGACY_FIBER_GROUP_INDEX) as f:
        assert f.light_speed_in_fiber_m_s == LEGACY
        assert f.spatial_step_m == 0.4
        assert f.gauge_length_m == 24 * 1e8 / 250_000_000
        assert f.trigger_delay == 978  # 999 m × 2e8 / v, to the metre
        assert f.recorded_fiber_group_index == 1.4682


def test_a_zone_spacing_is_scaled():
    """An archive export's zone spacing is not the pitch: it is scaled by
    v' / v."""
    with File(ASSETS / "archive/viz_variance.h5", fiber_group_index=1.4682) as f:
        assert f.spatial_step_m == 10.0 * SMF28 / LEGACY
        assert f.distance == 5 * f.spatial_step_m


def test_a_relabelled_export_carries_the_new_index(tmp_path):
    out = tmp_path / "relabelled.dat"
    with File(ASSETS / "split_h5/phase/2026-11-03_10-00-00.h5", fiber_group_index=1.4682) as f:
        f.export_dat(out)
        step, gauge = f.spatial_step_m, f.gauge_length_m
    assert out.read_bytes()[124:128] == struct.pack("<I", SMF28)
    with File(out) as g:
        assert g.fiber_group_index == 1.4682
        assert (g.spatial_step_m, g.gauge_length_m) == (step, gauge)


def test_export_dat_stamps_an_index(tmp_path):
    """``export_dat(…, fiber_group_index=n)``: a new file carries the speed
    of n and its steps follow it; a gauge given in metres is at that
    index. With ``header=`` the header is relabelled first."""
    data = phase_samples(3, 6)
    gauge = 24 * SMF28 / (2 * 250_000_000)
    path = tmp_path / "new.dat"
    export_dat(path, data, sample_rate=250_000_000, trig_frequency=50, content_type=PHASE,
               gauge_length_m=gauge, fiber_group_index=1.4682)
    with File(path) as f:
        assert f.header.light_speed_in_fiber_m_s == SMF28
        assert f.spatial_step_m == SMF28 / (2 * 250_000_000)
        assert f.gauge_length_m == gauge
    with File(ASSETS / "split_dat/phase/2026-11-03_10-00-00.dat") as src:
        export_dat(tmp_path / "re.dat", src.read_all(), header=src.header, fiber_group_index=1.4682)
    with File(tmp_path / "re.dat") as g:
        assert g.light_speed_in_fiber_m_s == SMF28
        assert g.gauge_length_m == 6 * SMF28 / (2 * 62_500_000)
    # Without the keyword the header's speed is kept (here: none).
    export_dat(tmp_path / "plain.dat", data, sample_rate=250_000_000, trig_frequency=50, content_type=PHASE)
    assert (tmp_path / "plain.dat").read_bytes()[124:128] == bytes(4)


def test_the_strain_of_a_relabelled_phase_uses_its_gauge():
    path = ASSETS / "split_h5/phase/2026-11-03_10-00-00.h5"
    with File(path, phase_unit="nanostrain", fiber_group_index=1.4682) as f:
        strain = f.read_all()
    with File(path) as f:
        rad = f.read_all()
    factor = np.float32(NANOSTRAIN_PER_RAD_METRE / (6 * SMF28 / (2 * 62_500_000)))
    np.testing.assert_array_equal(strain.view(np.uint32), (rad * factor).view(np.uint32))


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_a_relabelled_file_of_every_format(tmp_path, suffix):
    path = tmp_path / f"g{suffix}"
    WRITERS[suffix](path, phase_samples(2, 6), PHASE, 9.6)
    with File(path, fiber_group_index=1.4682) as f:
        assert f.gauge_length_m == 6 * SMF28 / (2 * SAMPLE_RATE)
        assert f.spatial_step_m == SMF28 / (2 * SAMPLE_RATE)

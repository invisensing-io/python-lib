"""
Invisensing — Python SDK for the Audace DAS environment.

Read acquisition files produced by the Audace FileWriter and extract
the channel of interest (phase, amplitude, arctan, I, Q, raw samples)
regardless of the on-disk format (DAT, HDF5, TDMS, SEG-Y). What a file
holds comes from its content type (``invisensing.content``, Audace data
format v1); files written before it are read from their header flags.

The hot path (header parsing, bytes → numpy conversion, de-interleave)
lives in the native ``invisensing._core`` Rust extension. The Python
side is a thin facade that dispatches per format.

Public API
----------

The recommended modern usage::

    from invisensing import File, Mode

    with File("phase/2026-11-03_10-00-00.h5") as f:
        if f.mode is Mode.PHASE:
            phase = f.get_phase()
        elif f.mode is Mode.AMPLITUDE:
            amp = f.get_amplitude()
        elif f.mode is Mode.ARCTAN:
            rad = f.get_arctan_radians()
        elif f.mode is Mode.IQ:
            i, q = f.get_i(), f.get_q()
        elif f.mode is Mode.ARCTAN_MAGNITUDE:
            a, m = f.get_arctan(), f.get_magnitude()
        elif f.mode is Mode.RAW:
            data = f.read_lines(1000)

A phase can be read as strain, in nm/m, when the file records the gauge
length it was measured over (``File.gauge_length_m``)::

    with File("phase/2026-11-03_10-00-00.h5", phase_unit="nanostrain") as f:
        strain = f.get_phase()   # phase_rad / gauge_length_m * NANOSTRAIN_PER_RAD_METRE

``NANOSTRAIN_PER_RAD_METRE`` = 110.37 nm/m per rad/m, from
``ΔΦ = 2β(n + Cε)·ε·L`` at λ = 1550.12 nm and n = 1.46 (the core's phase
index, not the fibre group index below: it does not follow it).

Or as strain rate, in nm/m/s: the strain increment from the previous line
times the real trigger frequency (NaN on line 0 and after a gap of the
time axis)::

    with File("phase/2026-11-03_10-00-00.h5", phase_unit="strain_rate") as f:
        rate = f.get_phase()     # (phase[n] - phase[n-1]) / gauge_length_m
                                 #   * NANOSTRAIN_PER_RAD_METRE * trig_frequency_real_hz

Every metre (``File.distance``, ``spatial_step_m``, ``gauge_length_m``) is
labelled at the speed of light in the fibre the file was acquired at,
``round(c / n)`` of the fibre group index the acquisition was configured
with (``File.light_speed_in_fiber_m_s``, ``File.fiber_group_index``); a file
recorded before the acquisition wrote it reads at the legacy 2·10⁸ m/s
(index 1.49896229), as it always did. An old recording whose fibre is known
can be labelled at its index::

    with File("old/2026-06-01_10-00-00.dat", fiber_group_index=1.4682) as f:
        f.distance             # metres along an SMF-28 fibre at 1550 nm

The legacy SDK is preserved verbatim::

    import invisensing.File as iFile
    file = iFile.File("acquisition.dat")
    data = file.get_lines(100)
    iFile.export("out.dat", data, ...)

Both paths share the same Rust-backed implementation — the legacy
module is a thin shim that delegates to the modern :class:`File`.

API stability
-------------

Everything in :data:`__all__` is **stable** from v1.0.0 onwards.
Adding new optional arguments, methods, properties, or formats is
allowed (additive changes); renaming or removing anything in
:data:`__all__` is not. (One exception, in 1.4.0: the constant of the
header bit ``0x020``, never set by any producer, is now ``FLAG_RESERVED``.)
"""

from __future__ import annotations

# Import the canonical class + helpers from the implementation module.
# These bindings become attributes of the ``invisensing`` package so
# both ``from invisensing import File`` and ``invisensing.File`` (used
# as an attribute, not as a submodule import) return the class.
from ._reader import (
    File,
    Mode,
    export_dat,
    Header,
    DatReader,
    parse_header,
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
    NANOSTRAIN_PER_RAD_METRE,
    TIME_FLAG_MAP_V1,
    TIME_FLAG_CORRECTED,
    SPEED_OF_LIGHT_M_S,
    DEFAULT_FIBER_GROUP_INDEX,
    LEGACY_FIBER_GROUP_INDEX,
    LEGACY_LIGHT_SPEED_IN_FIBER_M_S,
    FIBER_GROUP_INDEX_MIN,
    FIBER_GROUP_INDEX_MAX,
    group_index_of_light_speed,
    light_speed_of_group_index,
)
from . import content
from ._analog import AnalogTrigger
from ._start_clock import HostClockSync, StartTimeGuard
from . import _legacy as _legacy_impl

# Kept in step with pyproject.toml and Cargo.toml by
# tests/test_version.py — on an isolated site this string is the only
# way to tell which SDK is installed, so it must not drift again.
__version__ = "1.7.0"

__all__ = [
    "File",
    "Mode",
    "AnalogTrigger",
    "HostClockSync",
    "StartTimeGuard",
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
    "NANOSTRAIN_PER_RAD_METRE",
    "TIME_FLAG_MAP_V1",
    "TIME_FLAG_CORRECTED",
    "SPEED_OF_LIGHT_M_S",
    "DEFAULT_FIBER_GROUP_INDEX",
    "LEGACY_FIBER_GROUP_INDEX",
    "LEGACY_LIGHT_SPEED_IN_FIBER_M_S",
    "FIBER_GROUP_INDEX_MIN",
    "FIBER_GROUP_INDEX_MAX",
    "group_index_of_light_speed",
    "light_speed_of_group_index",
    "content",
    "__version__",
]


# ── Legacy ``invisensing.File`` shim ──────────────────────────────────────
#
# The original public API was:
#
#     import invisensing.File as iFile
#     file = iFile.File("…")
#     iFile.export(…)
#
# i.e. ``invisensing.File`` had to be a *module* containing the class
# and the helpers. The modern API binds ``invisensing.File`` to the
# class itself (via ``from ._reader import File`` above). To keep the
# legacy idiom working we:
#
# 1. attach ``File.File = File`` (self-reference) so ``iFile.File`` is
#    the class regardless of whether ``iFile`` is the class or a module
#    in disguise.
# 2. attach the legacy module-level helpers (``export``, ``hflags``,
#    ``Constants``, ``h``, ``HEADER_SIZE``, ``HEADER_FORMAT``) as
#    class-level attributes on ``File`` so the same ``iFile.<name>``
#    lookups work.
# 3. register the class in ``sys.modules['invisensing.File']`` so the
#    ``import invisensing.File`` statement doesn't trip on a missing
#    submodule file. (Python is fine with non-Module objects in
#    ``sys.modules`` — the import statement only does an attribute
#    lookup on the parent package afterwards.)
#
# Net effect:
#
#     from invisensing import File          # → the class
#     import invisensing.File as iFile      # → also the class (same object)
#     iFile.File("…")                       # → constructs the class
#     iFile.export(…)                       # → legacy export helper
#     iFile.hflags.H_AC                     # → legacy enum
#
# The class-level attachments do not interfere with the modern API
# because the class instances don't shadow class attributes by accident
# (the instance attributes are ``_backend``, ``_path``, ``_closed``,
# ``_header`` and ``_num_lines`` — all underscored, none of them a
# legacy name).


def _install_legacy_shim():
    # Self-reference so ``iFile.File`` resolves to the class.
    File.File = File
    # Legacy module-level helpers, attached as class attributes.
    File.export = staticmethod(_legacy_impl.export)
    File.Constants = _legacy_impl.Constants
    File.h = _legacy_impl.h
    File.hflags = _legacy_impl.hflags
    File.HEADER_SIZE = _legacy_impl.HEADER_SIZE
    File.HEADER_FORMAT = _legacy_impl.HEADER_FORMAT

    # Register in sys.modules so ``import invisensing.File`` is a
    # no-op lookup (the actual ``iFile`` binding comes from the
    # subsequent ``invisensing.File`` attribute access — which is the
    # class).
    import sys as _sys
    _sys.modules["invisensing.File"] = File


_install_legacy_shim()

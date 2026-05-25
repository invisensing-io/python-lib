"""
Invisensing — Python SDK for the Audace DAS environment.

Read acquisition files produced by the Audace FileWriter and extract
the channel of interest (I, Q, arctan, magnitude, phase) regardless of
the on-disk format (DAT, HDF5, TDMS, SEG-Y).

The hot path (header parsing, bytes → numpy conversion, de-interleave)
lives in the native ``invisensing._core`` Rust extension. The Python
side is a thin facade that dispatches per format.

Public API
----------

The recommended modern usage::

    from invisensing import File, Mode

    with File("acquisition.dat") as f:
        match f.mode:
            case Mode.IQ:               i, q  = f.get_i(), f.get_q()
            case Mode.ARCTAN_MAGNITUDE: a, m  = f.get_arctan(), f.get_magnitude()
            case Mode.PHASE:            phase = f.get_phase()
            case Mode.RAW:              data  = f.read_lines(1000)

The legacy SDK is preserved verbatim::

    import invisensing.File as iFile
    file = iFile.File("acquisition.dat")
    data = file.get_lines(100)
    iFile.export("out.dat", data, ...)

Both paths share the same Rust-backed implementation — the legacy
module is a thin shim that delegates to the modern :class:`File`.

API stability
-------------

Everything in :data:`__all__` is **stable** from v0.2.0 onwards.
Adding new optional arguments, methods, properties, or formats is
allowed (additive changes); renaming or removing anything in
:data:`__all__` is not.
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
    FLAG_SPECTRUM,
    FLAG_MOZART,
    FLAG_PHASE,
    FLAG_INTERLEAVED,
    FLAG_UNSIGNED,
    HEADER_SIZE,
)
from . import _legacy as _legacy_impl

__version__ = "0.2.0"

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
    "FLAG_SPECTRUM",
    "FLAG_MOZART",
    "FLAG_PHASE",
    "FLAG_INTERLEAVED",
    "FLAG_UNSIGNED",
    "HEADER_SIZE",
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
# (the instance attributes are ``_backend``, ``_path``, ``_closed``).


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

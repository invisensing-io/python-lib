"""
Legacy ``invisensing.File`` module shim.

The original SDK exposed the reader as ``invisensing.File.File`` (a
module named ``File`` containing a class ``File``) plus module-level
``invisensing.File.export``, ``hflags``, ``Constants``. All of those
are kept here for backwards compatibility — new code should
``from invisensing import File`` directly (which re-exports the same
class from this module).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

# Bring the canonical class + helpers in unchanged. Importing here makes
# ``invisensing.File.File`` resolve to exactly the same class as
# ``invisensing.File``, so legacy ``iFile.File(...)`` calls hit the
# Rust-backed reader without any extra hops. The boolean accessors
# ``is_demodulated()`` / ``is_ac()`` / ``is_hiz()`` and the legacy
# aliases ``is_acquisition_ac()`` / ``is_acquisition_hiz()`` live on the
# class itself (see ``_reader.py``).
from ._reader import (
    File,
    export_dat,
    FLAG_AC,
    FLAG_DEMODULATED,
    FLAG_FLOAT,
    FLAG_HIZ,
)

__all__ = [
    "File",
    "Constants",
    "h",
    "hflags",
    "export",
    "HEADER_SIZE",
    "HEADER_FORMAT",
]

HEADER_SIZE = 128
# Kept verbatim for legacy callers that imported the struct format. The
# real header is now parsed in Rust (``invisensing._core.parse_header``)
# but the wire layout is unchanged.
HEADER_FORMAT = "32siiiiii72x"


class Constants:
    """Constants used in the application — kept for backwards compat."""

    # Speed-of-light in single-mode silica fibre at 1550 nm (n ≈ 1.45).
    LIGHT_SPEED_IN_FIBER = 206_856_796  # m/s


class h(IntEnum):
    """Legacy header-field indices into the old ``struct.unpack`` result."""

    TIMESTAMP = 0
    LINE_SIZE = 1
    TRIG_FREQUENCY = 2
    SAMPLE_SIZE = 3
    SAMPLE_RATE = 4
    FLAGS = 5
    RANGE = 6


class hflags(IntEnum):
    """
    Legacy header-flag enum mirrored from the original Python SDK.

    Values are aligned with the canonical Audace ``header_flags``
    constants — same wire bits, just exposed under the legacy ``H_*``
    names.
    """

    H_DEMODULATED = FLAG_DEMODULATED  # 0x1
    H_FLOAT = FLAG_FLOAT              # 0x2
    H_AC = FLAG_AC                    # 0x4
    H_HIZ = FLAG_HIZ                  # 0x8


def export(
    filename,
    data: np.ndarray,
    timestamp,
    trigger_frequency: int,
    sample_rate: int,
    range: float,
    is_demodulated: bool = True,
    is_ac: bool = True,
    is_hiz: bool = True,
):
    """
    Legacy export shim. Builds the flag set from the bool arguments the
    old API used and delegates to :func:`invisensing.export_dat`.

    ``timestamp`` may be ``bytes`` (legacy) or ``str`` (new). The
    ``pulse_width`` / ``num_channels`` fields did not exist in the
    original SDK and are written as zeros so the byte layout matches
    the historical format exactly (useful for round-trip tests that
    ``filecmp`` the input and output).
    """
    if isinstance(timestamp, (bytes, bytearray)):
        timestamp_str = timestamp.decode("utf-8", errors="replace").rstrip("\x00 ")
    else:
        timestamp_str = str(timestamp)

    flags = 0
    if is_demodulated:
        flags |= FLAG_DEMODULATED
    if is_ac:
        flags |= FLAG_AC
    if is_hiz:
        flags |= FLAG_HIZ

    export_dat(
        filename,
        data,
        sample_rate=int(sample_rate),
        trig_frequency=int(trigger_frequency),
        range_v=float(range),
        timestamp=timestamp_str,
        flags=flags,
        pulse_width=0,    # legacy: field didn't exist, kept as zero padding
        num_channels=0,   # ditto — preserves the original byte layout
    )


# ── Legacy ``get_*`` method aliases on the canonical File class ───────────
#
# The original ``File`` class exposed ``get_lines``, ``get_data_type``,
# ``get_line_size``, etc.  The modern class uses properties
# (``f.line_size``, ``f.dtype``…) and ``read_lines`` for the streaming
# read. We install the legacy method names as thin delegations so
# existing scripts that imported the class via the legacy submodule
# keep working without any refactor.
#
# Idempotent — re-importing the module is a no-op.


def _install_legacy_aliases(cls):
    if getattr(cls, "_legacy_get_aliases_installed", False):
        return cls

    cls.get_distance = lambda self: self.distance
    cls.get_line_size = lambda self: self.line_size
    cls.get_trigger_frequency = lambda self: self.trig_frequency
    cls.get_sample_rate = lambda self: self.sample_rate
    cls.get_range = lambda self: self.range
    cls.get_num_lines = lambda self: self.num_lines
    cls.get_lines_left = lambda self: self.lines_left
    cls.get_duration = lambda self: self.duration

    # ``get_data_type`` returned a numpy type class (``np.float32``),
    # not a dtype instance. ``dtype.type`` matches the historical
    # return value.
    cls.get_data_type = lambda self: self.dtype.type

    # ``get_timestamp`` returned raw 32-byte bytes; re-encode + NUL-pad
    # so callers that round-trip back through ``export`` get the exact
    # same wire bytes.
    cls.get_timestamp = lambda self: self.timestamp.encode("utf-8").ljust(32, b"\x00")

    # ``get_lines(n)`` is just the legacy name for ``read_lines(n)``.
    cls.get_lines = lambda self, n=1: self.read_lines(n)

    cls._legacy_get_aliases_installed = True
    return cls


_install_legacy_aliases(File)

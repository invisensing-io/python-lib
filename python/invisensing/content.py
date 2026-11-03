"""
The ``content_type`` registry of the Audace data format v1.

A file's ``content_type`` (header byte 64; HDF5 attribute and TDMS property
``content_type``; SEG-Y textual line ``CONTENT-TYPE``) says what its samples
are: one value stands for one **quantity**, one **sample type** and one
producing chain. In a file of data format v1 the sample type comes from the
content type alone — its FLOAT / UNSIGNED / INTERLEAVED flags are written 0.
Two values are not typed: ``0`` (a file written before the byte existed,
described by its flags as it always was) and ``12`` (a script output typed
by its flags).

Stream-only values (6, 7, 8, 15, 20, 21, 22) carry two values per position
(``[m0, m1, m0, m1, …]``). Recordings split them into two single-column
files, one per member, in the member's sub-directory (``phase/`` and
``amplitude/``, ``arctan/`` and ``magnitude/``, ``i_component/`` and
``q_component/``); files written before data format v1 may still hold the
pair.

These helpers mirror the acquisition's own (same names, same answers) and
are checked against the registry the acquisition publishes.

Usage::

    from invisensing import File, content

    with File("phase/2026-11-03_10-00-00.h5") as f:
        content.quantity_of(f.content_type)   # "phase"
        content.unit(f.content_type)          # "rad"
        content.method_of(f.content_type)     # "DUI_COHERENT"
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from ._core import CARD_CODES, CONTENT_TYPES as _ROWS, FORMAT_VERSION, card_code

__all__ = [
    "ContentInfo",
    "Member",
    "Conversion",
    "REGISTRY",
    "CARD_CODES",
    "FORMAT_VERSION",
    "card_code",
    "info",
    "from_tag",
    "quantity_of",
    "members",
    "member_of",
    "sample_type",
    "sample_size",
    "is_phase",
    "is_amplitude",
    "is_stream_only",
    "is_file_value",
    "is_legacy",
    "split_children",
    "method_of",
    "unit",
    "to_physical",
]


class Member(NamedTuple):
    """One of the two values a stream-only content type carries per position."""

    #: ``"phase"``, ``"amplitude"``, ``"arctan"``, ``"i"``, ``"q"``.
    quantity: str
    #: ``"i16"``, ``"u16"`` or ``"f32"``.
    sample_type: str
    unit: str
    #: Sub-directory the member's file is written to.
    subdir: str
    #: Content type of that file.
    child: int


class Conversion(NamedTuple):
    """How a stored value ``v`` becomes a physical one (``expr``, in ``to``)."""

    to: str
    expr: str
    #: The factor, when the conversion is one (``π / 32768`` for arctan).
    multiply_by: Optional[float]
    #: ``True`` for a nominal, uncalibrated conversion (raw ADC → mV).
    informative: bool


class ContentInfo(NamedTuple):
    """One row of the registry."""

    value: int
    tag: str
    #: ``raw``, ``phase``, ``arctan``, ``amplitude``, ``i``, ``q``, ``custom``,
    #: ``variance``, ``log_variance``; ``None`` for 0 and for a stream-only
    #: value (see ``members``).
    quantity: Optional[str]
    #: ``"i16"``, ``"u16"``, ``"f32"``, ``"from_flags"`` (0 and 12); ``None``
    #: for a stream-only value.
    sample_type: Optional[str]
    #: Bytes per sample (per member for a stream-only value); ``None`` for 0
    #: and 12.
    sample_size: Optional[int]
    unit: Optional[str]
    conversion: Optional[Conversion]
    #: The demodulation method that produced it (``"DUI_COHERENT"``…), or
    #: ``None`` for a card value and the values no method produces.
    method: Optional[str]
    #: ``"file"``, ``"stream"`` or ``"file+stream"``.
    file_or_stream: str
    members: Optional[tuple]
    #: ``"legacy"``, ``"current"``, ``"frozen"`` or ``"new"``.
    status: str


def _row(d: dict) -> ContentInfo:
    conv = d["conversion"]
    mems = d["members"]
    return ContentInfo(
        value=d["value"],
        tag=d["tag"],
        quantity=d["quantity"],
        sample_type=d["sample_type"],
        sample_size=d["sample_size"],
        unit=d["unit"],
        conversion=None if conv is None else Conversion(**conv),
        method=d["method"],
        file_or_stream=d["file_or_stream"],
        members=None if mems is None else tuple(Member(**m) for m in mems),
        status=d["status"],
    )


#: ``{value: ContentInfo}`` for every value this SDK knows.
REGISTRY: dict = {d["value"]: _row(d) for d in _ROWS}

_BY_TAG = {i.tag: v for v, i in REGISTRY.items()}


def info(ct: int) -> Optional[ContentInfo]:
    """The registry row of ``ct``; ``None`` for a value this SDK does not
    know (a newer producer)."""
    return REGISTRY.get(int(ct))


def from_tag(tag: str) -> Optional[int]:
    """The value of a tag (``"dui_coherent_phase"`` → 19), or ``None``."""
    return _BY_TAG.get(tag)


def quantity_of(ct: int) -> Optional[str]:
    """What a single column of ``ct`` is; ``None`` for 0 (the flags say),
    for a stream-only value (see :func:`members`) and an unknown one."""
    i = info(ct)
    return None if i is None else i.quantity


def members(ct: int) -> Optional[tuple]:
    """The two :class:`Member` of a stream-only value, member 0 first;
    ``None`` for a single column."""
    i = info(ct)
    return None if i is None else i.members


def member_of(ct: int, quantity: str) -> Optional[tuple]:
    """``(index, sample_type)`` of the member of ``ct`` carrying
    ``quantity`` (``member_of(22, "amplitude") == (1, "f32")``), or ``None``."""
    for k, m in enumerate(members(ct) or ()):
        if m.quantity == quantity:
            return k, m.sample_type
    return None


def sample_type(ct: int) -> Optional[str]:
    """``"i16"``, ``"u16"``, ``"f32"``, or ``"from_flags"`` for 0 and 12;
    ``None`` for a stream-only value and an unknown one."""
    i = info(ct)
    return None if i is None else i.sample_type


def sample_size(ct: int) -> Optional[int]:
    """Bytes per sample (per member for a stream-only value); ``None`` for 0,
    12 and an unknown value."""
    i = info(ct)
    return None if i is None else i.sample_size


def is_phase(ct: int) -> bool:
    """A single column of phase, in radians (2, 4, 9, 10, 17, 18, 19, 23)."""
    return quantity_of(ct) == "phase"


def is_amplitude(ct: int) -> bool:
    """A single column of amplitude (5, 11, 16)."""
    return quantity_of(ct) == "amplitude"


def is_stream_only(ct: int) -> bool:
    """A value that carries pairs (6, 7, 8, 15, 20, 21, 22): split into two
    files by every recording since data format v1."""
    return members(ct) is not None


def is_file_value(ct: int) -> bool:
    """Every known value but the stream-only ones."""
    return info(ct) is not None and not is_stream_only(ct)


def is_legacy(ct: int) -> bool:
    """0, 4, 8 and 12: never written by a producer of data format v1."""
    i = info(ct)
    return i is not None and i.status in ("legacy", "frozen")


def split_children(ct: int) -> Optional[tuple]:
    """The content types of the two files a stream-only value is split into
    (``split_children(7) == (3, 5)``), or ``None``."""
    m = members(ct)
    return None if m is None else (m[0].child, m[1].child)


def method_of(ct: int) -> Optional[str]:
    """The demodulation method that produced ``ct``, or ``None``."""
    i = info(ct)
    return None if i is None else i.method


def unit(ct: int) -> Optional[str]:
    """The unit of a single column of ``ct`` (``"rad"``, ``"count"``…)."""
    i = info(ct)
    return None if i is None else i.unit


def to_physical(ct: int) -> Optional[Conversion]:
    """How a stored value of ``ct`` becomes a physical one, or ``None``:
    ``hw_arctan`` (3) to radians (``v · π / 32768``), ``raw`` (1) to mV
    (informative), ``viz_logvar`` (28) back to the variance
    (``10^v − 1``)."""
    i = info(ct)
    return None if i is None else i.conversion

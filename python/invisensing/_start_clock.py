"""The host clock and the time guard at a run's START (HDF5 root attributes).

Since Audace 2026-11-22 (decision D4, Audace-Lib ``docs/DATA_FORMAT.md``
§8.7) every HDF5 file the FileWriter writes records whether the node's clock
was synchronised when the console sent the run's START
(``host_clock_synchronized``) and how that START passed the console's time
guard (``start_time_guard``: a FORCED START went past a guard an Admin
lifted). ``File.host_clock_synchronized`` and ``File.start_time_guard`` read
them as the two enums below; DAT, SEG-Y and TDMS files carry neither (the
manifest beside them does).
"""

from __future__ import annotations

import enum
from typing import Optional


class HostClockSync(str, enum.Enum):
    """The node's clock when the console sent the run's START.

    A ``str`` enum: a member compares equal to the tag the file stores
    (``File.host_clock_synchronized == "synchronized"``).
    """

    #: Nobody said: the console did not read it, an element of the chain
    #: predates the attribute, or the file stores a value this SDK does not
    #: know. Never read as synchronised.
    UNKNOWN = "unknown"
    #: The host clock monitor published, less than 120 s before, a
    #: synchronised clock.
    SYNCHRONIZED = "synchronized"
    #: It published one that is not synchronised.
    NOT_SYNCHRONIZED = "not_synchronized"

    @classmethod
    def from_tag(cls, tag: Optional[str]) -> Optional["HostClockSync"]:
        """The member a file's tag names: ``None`` for no tag, ``UNKNOWN``
        for a tag this SDK does not know."""
        if tag is None:
            return None
        try:
            return cls(tag)
        except ValueError:
            return cls.UNKNOWN

    @property
    def synchronized(self) -> Optional[bool]:
        """``True`` / ``False``, or ``None`` when unknown."""
        if self is HostClockSync.UNKNOWN:
            return None
        return self is HostClockSync.SYNCHRONIZED


class StartTimeGuard(str, enum.Enum):
    """How the run's START passed the console's time guard.

    A ``str`` enum: a member compares equal to the tag the file stores.
    """

    #: Nobody said (see :attr:`HostClockSync.UNKNOWN`).
    UNKNOWN = "unknown"
    #: The guard was open: the host clock synchronised, the proof made.
    OPEN = "open"
    #: The guard does not apply: a Lab node in offline mode with no time
    #: source configured. The node's clock dates the files.
    WAIVED = "waived"
    #: Past the guard an Admin lifted until the node's next shutdown: a
    #: forced START.
    LIFTED = "lifted"
    #: Past the guard an Admin lifted permanently: a forced START too.
    LIFTED_PERMANENT = "lifted_permanent"

    @classmethod
    def from_tag(cls, tag: Optional[str]) -> Optional["StartTimeGuard"]:
        """The member a file's tag names: ``None`` for no tag, ``UNKNOWN``
        for a tag this SDK does not know."""
        if tag is None:
            return None
        try:
            return cls(tag)
        except ValueError:
            return cls.UNKNOWN

    @property
    def forced(self) -> Optional[bool]:
        """``True`` for a START forced past a lifted guard, ``False`` for
        one the guard let through (open or waived), ``None`` when unknown."""
        if self is StartTimeGuard.UNKNOWN:
            return None
        return self in (StartTimeGuard.LIFTED, StartTimeGuard.LIFTED_PERMANENT)

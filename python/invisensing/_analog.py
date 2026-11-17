"""The analog-trigger metadata of a recording (HDF5 root attributes).

Audace's analog-trigger recording mode records the fibre (CH1) when a
second, analog input of the same card (CH2) crosses a threshold. CH2 itself
is never written; every file of the mode carries, as HDF5 root attributes,
what triggered it (Audace-Lib ``docs/DATA_FORMAT.md`` §8.3). This module
holds the record :class:`AnalogTrigger` that ``File.analog_trigger``
returns; the attributes are parsed by the HDF5 backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: ``File.analog_trigger.option`` of option A: one file of the file duration
#: per trigger, from trigger + offset.
RISING_EDGE = "rising_edge"
#: ``File.analog_trigger.option`` of option B: recording while CH2 stays
#: above the threshold.
ABOVE_THRESHOLD = "above_threshold"


@dataclass(frozen=True)
class AnalogTrigger:
    """What triggered an analog-trigger recording, as its file records it.

    Lines are CARD lines: ``line − File.first_line`` is the index of the
    trigger line in this file (:attr:`File.analog_trigger_index`; negative
    when the trigger came before this file, e.g. a later part of a long
    event or a positive offset). Times are UTC nanoseconds on the axis of
    the file's ``timestamp_ns``.
    """

    #: ``"rising_edge"`` (A) or ``"above_threshold"`` (B).
    option: str
    #: The threshold, mV at the card input: a line triggers when
    #: ``|v| ≥ threshold`` on one of its CH2 samples.
    threshold_mv: float
    #: The threshold in ADC counts actually compared.
    threshold_counts: Optional[int]
    #: CH2's full scale used for the conversion (nominal, uncalibrated).
    full_scale_mv: Optional[float]
    full_scale_counts: Optional[int]
    #: Signed offset of the recording start from the trigger line, ns.
    offset_ns: int
    #: B only: below-threshold runs up to this many ms were bridged.
    hold_ms: Optional[int]
    #: 1-based event number within the run, and file number within the
    #: event (always 1 for A).
    event: Optional[int]
    part: Optional[int]
    #: Card line of the trigger line, its first sample at or above the
    #: threshold, and that sample's offset in the line, ns.
    line: Optional[int]
    sample: Optional[int]
    sample_offset_ns: Optional[float]
    #: Time of the trigger line, UTC ns.
    time_ns: Optional[int]
    #: ``max |v|`` of the trigger line, mV.
    peak_mv: Optional[float]
    #: The event's first line, and ``line − start_line``.
    start_line: Optional[int]
    pre_lines: Optional[int]
    #: Why the event starts later than trigger + offset: ``""`` (it does
    #: not), ``"run_start"``, ``"previous_event"`` or ``"ring"``.
    pre_truncated: str
    #: A only: crossing lines inside the file after the trigger, ignored.
    crossings_ignored: Optional[int]
    #: The event's last line and why it ended (``"length"``, ``"gate"``,
    #: ``"stop"``, ``"error"``): only on the event's last file.
    end_line: Optional[int]
    end_reason: Optional[str]

    @property
    def is_last_part(self) -> bool:
        """This file is the event's last (it says how the event ended)."""
        return self.end_reason is not None

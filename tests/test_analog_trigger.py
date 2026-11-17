"""Sub-second files and the analog-trigger metadata (invisensing 1.7.0).

Audace writes files shorter than one second (file durations with decimals,
e.g. 0.1 s, and every file of the analog-trigger mode). Their header
``duration_s`` floors to whole seconds and reads 0 ("unknown"): the
duration is measured from the lines (``File.duration``), and HDF5 files
also carry ``duration_exact_s``, ``first_line`` and, in the analog-trigger
mode, the ``analog_trigger_*`` attributes (Audace-Lib DATA_FORMAT.md §8.3).
The fixtures are written the way the FileWriter writes them.
"""

from __future__ import annotations

import numpy as np
import pytest

import invisensing
from invisensing import AnalogTrigger, File

from gauge_files import TRIG, write_dat, write_h5

RAW = 1  # content type `raw` (i16 ADC counts)


def _raw(rows: int, positions: int = 16) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(-2000, 2000, size=(rows, positions), dtype=np.int16)


def test_a_tenth_of_a_second_dat_reads_its_duration_from_the_lines(tmp_path):
    rows = TRIG // 10  # 0.1 s at 50 Hz: 5 lines
    path = write_dat(tmp_path / "17-11-26_14h07m03s_250.dat", _raw(rows), content_type=RAW)
    with File(path) as f:
        assert f.num_lines == rows
        assert f.duration_s is None, "a file under one second says nothing in duration_s"
        assert f.duration == pytest.approx(0.1)
        # DAT carries no extra attributes.
        assert f.duration_exact_s is None
        assert f.first_line is None
        assert f.analog_trigger is None
        assert f.analog_trigger_index is None


def test_a_tenth_of_a_second_hdf5_reads_its_exact_duration(tmp_path):
    rows = TRIG // 10
    path = write_h5(
        tmp_path / "17-11-26_14h07m03s_250.h5",
        _raw(rows),
        content_type=RAW,
        extra={
            "duration_s": np.uint32(0),
            "duration_exact_s": np.float64(0.1),
            "first_line": np.int64(123_456),
        },
    )
    with File(path) as f:
        assert f.duration_s is None
        assert f.duration == pytest.approx(0.1)
        assert f.duration_exact_s == pytest.approx(0.1)
        assert f.first_line == 123_456
        assert f.analog_trigger is None
        np.testing.assert_array_equal(f.read_lines(rows), _raw(rows))


def _analog_attrs(option: str, *, last: bool) -> dict:
    a = {
        "recording_mode": "analog_trigger",
        "analog_trigger_option": option,
        "analog_trigger_threshold_mv": np.float64(100.0),
        "analog_trigger_threshold_counts": np.int32(3450),
        "analog_trigger_full_scale_mv": np.float64(950.0),
        "analog_trigger_full_scale_counts": np.uint32(32768),
        "analog_trigger_offset_ns": np.int64(-2_000_000_000),
        "analog_trigger_event": np.uint32(3),
        "analog_trigger_part": np.uint32(1),
        "analog_trigger_line": np.int64(1_000_100),
        "analog_trigger_sample": np.uint32(4021),
        "analog_trigger_time_ns": np.int64(1_790_000_002_123_456_789),
        "analog_trigger_sample_offset_ns": np.float64(16084.0),
        "analog_trigger_peak_mv": np.float64(412.5),
        "analog_trigger_start_line": np.int64(1_000_000),
        "analog_trigger_pre_lines": np.int64(100),
        "analog_trigger_pre_truncated": "",
        "first_line": np.int64(1_000_000),
        "duration_exact_s": np.float64(0.1),
        "duration_s": np.uint32(0),
    }
    if option == "rising_edge":
        a["analog_trigger_crossings_ignored"] = np.uint32(2)
    else:
        a["analog_trigger_hold_ms"] = np.uint32(20)
    if last:
        a["analog_trigger_end_line"] = np.int64(1_000_004)
        a["analog_trigger_end_reason"] = "length" if option == "rising_edge" else "gate"
    return a


@pytest.mark.parametrize("option", ["rising_edge", "above_threshold"])
def test_an_analog_trigger_file_says_what_triggered_it(tmp_path, option):
    rows = TRIG // 10
    path = write_h5(
        tmp_path / "17-11-26_14h07m03s_250.h5",
        _raw(rows),
        content_type=RAW,
        extra=_analog_attrs(option, last=True),
    )
    with File(path) as f:
        at = f.analog_trigger
    # Metadata stays readable once closed, like the rest of the header.
    assert isinstance(at, AnalogTrigger)
    assert at.option == option
    assert at.threshold_mv == 100.0
    assert at.threshold_counts == 3450
    assert at.full_scale_mv == 950.0
    assert at.full_scale_counts == 32768
    assert at.offset_ns == -2_000_000_000
    assert at.event == 3 and at.part == 1
    assert at.line == 1_000_100
    assert at.sample == 4021
    assert at.time_ns == 1_790_000_002_123_456_789
    assert at.sample_offset_ns == 16084.0
    assert at.peak_mv == 412.5
    assert at.start_line == 1_000_000 and at.pre_lines == 100
    assert at.pre_truncated == ""
    assert at.is_last_part
    if option == "rising_edge":
        assert at.crossings_ignored == 2 and at.hold_ms is None
        assert at.end_reason == "length"
    else:
        assert at.hold_ms == 20 and at.crossings_ignored is None
        assert at.end_reason == "gate"
    assert f.first_line == 1_000_000
    assert f.analog_trigger_index == 100
    assert f.duration_s is None
    assert f.duration_exact_s == pytest.approx(0.1)


def test_an_earlier_part_of_an_event_has_no_end(tmp_path):
    attrs = _analog_attrs("above_threshold", last=False)
    attrs["analog_trigger_part"] = np.uint32(2)
    attrs["first_line"] = np.int64(1_000_200)
    path = write_h5(tmp_path / "part2.h5", _raw(5), content_type=RAW, extra=attrs)
    with File(path) as f:
        at = f.analog_trigger
        assert at.part == 2
        assert not at.is_last_part
        assert at.end_line is None and at.end_reason is None
        # The trigger came before this part.
        assert f.analog_trigger_index == -100


def test_a_file_of_another_mode_has_no_analog_trigger(tmp_path):
    # Attributes alone do not make the mode: `recording_mode` does.
    attrs = _analog_attrs("rising_edge", last=True)
    del attrs["recording_mode"]
    path = write_h5(tmp_path / "plain.h5", _raw(5), content_type=RAW, extra=attrs)
    with File(path) as f:
        assert f.analog_trigger is None
        assert f.analog_trigger_index is None
        assert f.first_line == 1_000_000


def test_the_record_is_public_api():
    assert "AnalogTrigger" in invisensing.__all__
    assert invisensing.AnalogTrigger is AnalogTrigger

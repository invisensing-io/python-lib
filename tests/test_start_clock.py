"""
The host clock and the time guard at a run's START (Audace decision D4).

Since 2026-11-22 every HDF5 file the FileWriter writes records, as string
root attributes, whether the node's clock was synchronised when the console
sent the run's START (``host_clock_synchronized``: ``synchronized``,
``not_synchronized``, ``unknown``) and how that START passed the console's
time guard (``start_time_guard``: ``open``, ``waived``, ``lifted``,
``lifted_permanent`` — a forced START — or ``unknown``), Audace-Lib
``docs/DATA_FORMAT.md`` §8.7.

What these tests pin down:

* every tag reads as its enum member, which compares equal to the tag;
* a tag this SDK does not know reads ``UNKNOWN``, never synchronised;
* a file without them (older HDF5, DAT, SEG-Y, TDMS) answers ``None``;
* the answer survives ``close()`` like the rest of the metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import invisensing
from invisensing import File, HostClockSync, StartTimeGuard

h5py = pytest.importorskip("h5py")

ANCHOR = 1_760_000_000_000_000_000


def _write_hdf5(path: Path, **tags) -> Path:
    payload = np.arange(10 * 8, dtype=np.int16).reshape(10, 8)
    with h5py.File(str(path), "w") as fp:
        fp.create_dataset("acoustic_data", data=payload)
        for k, v in {
            "line_size": 8, "trig_frequency": 2_000, "sample_size": 2,
            "sample_rate": 100_000_000, "flags": 0, "num_channels": 1,
        }.items():
            fp.attrs[k] = np.uint32(v)
        fp.attrs["timestamp_ns"] = np.int64(ANCHOR)
        for k, v in tags.items():
            fp.attrs[k] = v
    return path


@pytest.mark.parametrize(
    "tag, member, synchronized",
    [
        ("synchronized", HostClockSync.SYNCHRONIZED, True),
        ("not_synchronized", HostClockSync.NOT_SYNCHRONIZED, False),
        ("unknown", HostClockSync.UNKNOWN, None),
    ],
)
def test_the_host_clock_at_start_reads_as_its_member(tmp_path, tag, member, synchronized):
    path = _write_hdf5(tmp_path / "f.h5", host_clock_synchronized=tag, start_time_guard="open")
    with File(path) as f:
        got = f.host_clock_synchronized
        assert got is member
        assert got == tag, "a str enum: equal to the stored tag"
        assert got.synchronized is synchronized


@pytest.mark.parametrize(
    "tag, member, forced",
    [
        ("open", StartTimeGuard.OPEN, False),
        ("waived", StartTimeGuard.WAIVED, False),
        ("lifted", StartTimeGuard.LIFTED, True),
        ("lifted_permanent", StartTimeGuard.LIFTED_PERMANENT, True),
        ("unknown", StartTimeGuard.UNKNOWN, None),
    ],
)
def test_the_time_guard_at_start_reads_as_its_member(tmp_path, tag, member, forced):
    path = _write_hdf5(tmp_path / "f.h5", host_clock_synchronized="not_synchronized", start_time_guard=tag)
    with File(path) as f:
        assert f.start_time_guard is member
        assert f.start_time_guard == tag
        assert f.start_time_guard.forced is forced


def test_a_tag_this_sdk_does_not_know_reads_unknown(tmp_path):
    # A newer writer's value, stored as bytes by some HDF5 tools.
    path = _write_hdf5(tmp_path / "f.h5", host_clock_synchronized=b"drifting", start_time_guard="bypassed")
    with File(path) as f:
        assert f.host_clock_synchronized is HostClockSync.UNKNOWN
        assert f.host_clock_synchronized.synchronized is None, "never synchronised"
        assert f.start_time_guard is StartTimeGuard.UNKNOWN


def test_a_file_without_them_answers_none(tmp_path):
    older = _write_hdf5(tmp_path / "older.h5")
    with File(older) as f:
        assert f.host_clock_synchronized is None
        assert f.start_time_guard is None
        # DAT keeps neither (the save or run manifest beside it does).
        f.export_dat(tmp_path / "copy.dat")
    with File(tmp_path / "copy.dat") as d:
        assert d.host_clock_synchronized is None and d.start_time_guard is None


def test_they_survive_close_and_are_exported(tmp_path):
    path = _write_hdf5(tmp_path / "f.h5", host_clock_synchronized="synchronized", start_time_guard="lifted")
    f = File(path)
    f.close()
    assert f.host_clock_synchronized is HostClockSync.SYNCHRONIZED
    assert f.start_time_guard.forced is True
    assert "HostClockSync" in invisensing.__all__ and "StartTimeGuard" in invisensing.__all__
    assert HostClockSync.from_tag(None) is None and StartTimeGuard.from_tag(None) is None

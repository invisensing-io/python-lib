# Invisensing — Python SDK for the Audace DAS environment

`invisensing` is the official Python library for reading **Distributed
Acoustic Sensing (DAS)** acquisition files produced by the Audace
platform. It handles every file format the platform writes (`.dat`,
`.h5` / `.hdf5`, `.tdms`, `.sgy`) through a single uniform API, and exposes
the channels each file actually contains — `phase` / `amplitude` /
`arctan` / `I` / `Q` / raw samples — in one method call, with the right
dtype and optional physical-unit scaling. What a file holds comes from its
**content type** (data format v1, see below). A file written before the
content type existed (content type 0) is read from its header flags as
1.3.0 read it, except that:

- a single column flagged `UNSIGNED` reads as unsigned integers (`uint16`
  for 2-byte samples) instead of signed;
- `File.distance` uses the producer's 2·10⁸ m/s in the fibre: distances
  are 3.3 % shorter than 1.3.0's;
- a SEG-Y file's textual header (ASCII or EBCDIC) is read, and its integer
  samples at their own width;
- `File.dtype` is the dtype `read_lines()` returns.

A file written between June 2026 and data format v1 that carries a
content type is classified by it — except a `sw_custom` file of an older
Sandbox and a console archive export under a paired content type without
the `INTERLEAVED` flag, which are read from their flags.
[CHANGELOG.md](https://github.com/invisensing-io/python-lib/blob/master/CHANGELOG.md)
lists every change since 1.3.0.

The library is **format-agnostic but mode-faithful**: it never
silently re-derives one demodulation product from another. The
on-board DAQ DSP chain (fading suppression, spatial differential,
detrend filter) is not reproducible in software from an earlier tap,
so the channel you can extract is the one the acquisition wrote. Inspect
``file.mode`` to dispatch; the SDK raises a clear `ValueError` if you
call an extractor that doesn't apply.

The hot path (header parsing, bytes → numpy conversion, de-interleave)
is implemented in **Rust** and exposed through PyO3, so reading a
multi-GB capture stays fast even from Python.

## Installation

```bash
pip install invisensing
```

That's it — every supported file format (DAT, HDF5, TDMS, SEG-Y) works
out of the box. The default install pulls in `numpy`, `h5py`,
`hdf5plugin`, `npTDMS`, and `segyio` so any file written by the Audace
FileWriter — HDF5 compressed with zstd included — can be opened
without further setup.

Wheels are built from a Rust extension; `pip` handles the native build
transparently. No Rust toolchain is required for end users.

> **Note** — the format extras (`pip install invisensing[hdf5]` etc.)
> are still recognised for backwards compatibility with older
> installation guides, but they are no-ops now: every backend is part
> of the default install.

## Quick start

```python
from invisensing import File, Mode

with File("phase/2026-11-03_10-00-00.h5") as f:
    print(f)                       # File('…', mode=phase, shape=…, …)
    print(f.content_type_name)     # "dui_coherent_phase": what the samples are
    print(f.duration, "s")
    print(f.distance, "m of fibre")

    # The Mode enum lets you dispatch cleanly on what the file contains.
    mode = f.mode
    if mode is Mode.RAW:
        data = f.read_lines(1000)                  # (1000, line_size) ADC codes
    elif mode is Mode.PHASE:
        phase = f.get_phase()                      # (n, positions) f32 radians
    elif mode is Mode.AMPLITUDE:
        amplitude = f.get_amplitude()              # (n, positions) u16 or f32
    elif mode is Mode.ARCTAN:
        rad = f.get_arctan_radians()               # (n, positions) f32 radians
    elif mode in (Mode.I, Mode.Q):
        column = f.read_lines(1000)                # (n, positions) i16
    # Files written before data format v1 may hold pairs:
    elif mode is Mode.IQ:
        iq = f.get_iq()                            # (n, positions) complex64
    elif mode is Mode.ARCTAN_MAGNITUDE:
        arctan, magnitude = f.get_arctan(), f.get_magnitude()
    elif mode is Mode.PHASE_AMPLITUDE:
        phase, amplitude = f.get_phase(), f.get_amplitude()
```

The examples use `if` / `elif`: the SDK supports Python 3.9, which has no
`match` statement.

## Streaming large files

`read_lines(n)` advances a cursor; iterating the file yields one pulse
at a time:

```python
with File("long_capture.h5") as f:
    while f.lines_left:
        chunk = f.read_lines(10_000)
        process(chunk)

# Or, one pulse at a time:
with File("long_capture.dat") as f:
    for pulse in f:
        process(pulse)
```

For ad-hoc scripts on small files, `read_all()` returns everything in
one allocation.

## Output dtypes — raw codes vs. physical units

I, Q, arctan, and √(I²+Q²) are **physically continuous quantities**
(volts for I/Q/magnitude, radians for arctan) that the card encodes as
fixed-point integers. The SDK offers two flavours of
extractor for each lane:

- **Default extractors** (`get_i`, `get_q`, `get_arctan`,
  `get_magnitude`) return the **wire dtype** — fast, no copy beyond
  the de-interleave, no precision loss on round-trip writes.
- **Physical-unit extractors** (`get_i_volts`, `get_q_volts`,
  `get_iq_volts`, `get_arctan_radians`, `get_magnitude_volts`)
  return ``float32`` numpy arrays in the natural physical unit
  (volts or radians). Use these when you start doing DSP — they save
  you from remembering the scaling constants.

The full mapping per mode:

| Mode | Method | dtype | Unit | Scaling applied |
|---|---|---|---|---|
| **Raw** | `read_lines()` | `int16` | ADC code | — |
| **I** / **Q** | `get_i()` / `get_q()` | `int16` | ADC code | — |
| **I** / **Q** | `get_i_volts()` / `get_q_volts()` | **`float32`** | **V** | `i16 × range / 32768` |
| **Arctan** | `get_arctan()` | `int16` | fixed-point | `32767 ↔ +π` |
| **Arctan** | `get_arctan_radians()` | **`float32`** | **rad** | `i16 × π / 32768` |
| **Amplitude** | `get_amplitude()` | `uint16` (card) / `float32` | count | — |
| **Amplitude** | `get_magnitude_volts()` | **`float32`** | **V** | card magnitude only: `× range / 32768` |
| **Phase** | `read_lines()` / `get_phase()` | `float32` | **rad** | — |
| **Variance** / **LogVariance** | `get_variance()` / `get_log_variance()` | `float32` | source unit² / — | `10^v − 1` ↔ `log10(1 + v)` |
| **IQ** | `read_lines()` | `int16` | ADC code | wire layout `[I, Q, I, Q, …]` |
| **IQ** | `get_i()` / `get_q()` | `int16` | ADC code | de-interleave only |
| **IQ** | `get_iq()` | `complex64` | ADC code | `I + j·Q` packed |
| **IQ** | `get_i_volts()` / `get_q_volts()` | **`float32`** | **V** | `i16 × range / 32768` |
| **IQ** | `get_iq_volts()` | **`complex64`** | **V** | real/imag both in volts |
| **ArctanMagnitude** | `read_lines()` | `int16` | mixed | wire layout `[atan, √, atan, √, …]` |
| **ArctanMagnitude** | `get_arctan()` | `int16` | fixed-point | `32767 ↔ +π` |
| **ArctanMagnitude** | `get_arctan_radians()` | **`float32`** | **rad** | `i16 × π / 32768` |
| **ArctanMagnitude** | `get_magnitude()` | `uint16` | ADC code | bitcast from wire i16 |
| **ArctanMagnitude** | `get_magnitude_volts()` | **`float32`** | **V** | `u16 × range / 32768` |
| **PhaseAmplitude** | `get_phase()` / `get_amplitude()` | `float32` | rad / count | de-interleave only |

A phase is always floating-point radians — the acquisition converts it
before writing, so consumers never need to know the card's scaling.

`f.dtype` always reflects what `read_lines()` will return: the content
type's sample type (`int16`, `uint16`, `float32`); on a file written before
the content type existed (and the older files read from their flags),
`sample_size` and the `FLOAT` / `UNSIGNED` flags.

### When to pick which

```python
with File("iq.dat") as f:
    # Quick QC / dumping — keep the wire codes:
    i = f.get_i()                      # int16, fast
    plt.plot(i[0])                     # ADC codes on Y axis

    # Real DSP — work in volts:
    iq_v = f.get_iq_volts()            # complex64, in volts
    envelope_v = np.abs(iq_v)          # volts
    phase_rad  = np.angle(iq_v)        # radians, wrapped

with File("arctan_mag.dat") as f:
    rad = f.get_arctan_radians()       # float32, ±π
    vol = f.get_magnitude_volts()      # float32, ≥ 0 volts
```

Reading a 4 GB DAT capture stays a 4 GB numpy array if you use the
default extractors; the `*_volts` / `*_radians` family allocates a new
`float32` array (so 2× the raw size for i16 / 2× for u16, 8× for the
complex64 in `get_iq_volts`).

## Channel extractors — what they mean per mode

`file.mode` comes from the file's content type (from its header flags
on a file written before the content type existed, on a `sw_custom`
file of a Sandbox older than data format v1, which kept its input's
flags, and on a console archive export written before v1 under a paired
content type without the `INTERLEAVED` flag, which kept its source's
header over one column per zone: `f.header.mode_from_flags`). Since data format v1
a recording holds **one quantity per file**: an interleaved product of the
card or of the demodulation is split into two files, same name, sibling
directories (`phase/` + `amplitude/`, `arctan/` + `magnitude/`,
`i_component/` + `q_component/`; `f.sibling_path` finds the other one).
The paired modes are the files written before.

| Mode | Layout (per pulse) | Extractor(s) | Dtype out |
|---|---|---|---|
| `Mode.RAW` | `[s0, s1, …, sN-1]` | `read_lines()` | `int16` |
| `Mode.PHASE` | `[φ0, φ1, …]` | `get_phase()` | `float32` (radians) |
| `Mode.AMPLITUDE` | `[a0, a1, …]` | `get_amplitude()` (= `get_magnitude()`) | `uint16` (card) / `float32` |
| `Mode.ARCTAN` | `[atan0, atan1, …]` | `get_arctan()`, `get_arctan_radians()` | `int16` / `float32` |
| `Mode.I`, `Mode.Q` | `[I0, I1, …]` | `get_i()` / `get_q()` | `int16` |
| `Mode.CUSTOM` | a script's output | `read_lines()` | per content type |
| `Mode.VARIANCE`, `Mode.LOG_VARIANCE` | zone variances (archive exports) | `get_variance()`, `get_log_variance()` | `float32` |
| `Mode.IQ` | `[I0, Q0, I1, Q1, …]` | `get_i()`, `get_q()`, `get_iq()` | `int16` / `int16` / `complex64` |
| `Mode.ARCTAN_MAGNITUDE` | `[atan0, √0, atan1, √1, …]` | `get_arctan()`, `get_magnitude()` | `int16` / `uint16` |
| `Mode.PHASE_AMPLITUDE` | `[φ0, a0, φ1, a1, …]` | `get_phase()`, `get_amplitude()` | `float32` |

Each extractor reads from the file *or* from a buffer you've already
read — pass `data=` to reuse a buffer across extractors so the cursor
doesn't double-advance:

```python
with File("iq.dat") as f:
    chunk = f.read_lines(1000)
    i = f.get_i(chunk)                # reuses the buffer
    q = f.get_q(chunk)                # same — no extra file read
    # equivalent: f.channels(chunk) returns {"i": …, "q": …, "iq": …}
```

Calling the wrong extractor for the file's mode raises a clear error
that names both the actual and the expected mode — no silent garbage:

```python
>>> with File("phase.dat") as f:
...     f.get_i()
ValueError: get_i: file mode is 'phase', but this extractor only
applies to 'i' or 'iq'. Inspect file.mode to dispatch — see the Mode enum docs.
```

## Format auto-detection

The format is picked from the file suffix:

| Suffix | Backend |
|---|---|
| `.dat` | Native Rust (no third-party dep) |
| `.h5`, `.hdf5` | `h5py` |
| `.tdms` | `npTDMS` |
| `.sgy`, `.segy` | `segyio` |

If your file has no suffix or an unusual one, force the backend:

```python
File("acquisition_data", format="dat")
```

## Metadata at your fingertips

```python
with File("acquisition.dat") as f:
    f.line_size                 # samples per pulse on the wire
    f.positions_per_line        # spatial positions per pulse (= line_size/2 if INTERLEAVED)
    f.sample_size               # bytes per sample
    f.sample_rate               # Hz
    f.trig_frequency            # Hz, the REQUESTED trigger rate (int)
    f.trig_frequency_real_hz    # Hz, the rate the card really fired at (float)
    f.trig_frequency_is_exact   # True when both are equal (or the real one is unknown)
    f.trig_clock_hz             # card rational, or None when unknown —
    f.trig_period_ticks         #   real rate = trig_clock_hz / trig_period_ticks
    f.line_time_ns(i)           # acquisition time of pulse i, int ns (Unix epoch)
    f.num_lines                 # total pulses recorded
    f.lines_left                # remaining behind the cursor
    f.duration                  # seconds (= num_lines / trig_frequency_real_hz)
    f.distance                  # metres of fibre covered by one pulse
    f.range                     # ADC voltage range (V)
    f.timestamp                 # start time as the format spells it (see below)
    f.content_type_name         # what the samples are: "dui_coherent_phase", "hw_amplitude"…
    f.quantity, f.unit          # "phase", "rad" — from the content type
    f.card                      # card code "p18" / "p21" / "s125" / "s500", or None
    f.format_version            # 1 for data format v1, 0 before
    f.duration_s                # whole seconds the writer recorded (None: not said)
    f.spatial_step_m            # metres between two positions
    f.temporal_step_s           # seconds between two lines
    f.sibling_path              # the other file of a split pair, or None
    f.compression               # HDF5: "NONE" / "ZSTD" / "ZSTD_LOSSY_1E4"; None otherwise
    f.quantization_max_error_rad  # ZSTD_LOSSY_1E4 bound (rad), else None
    f.trailing_bytes            # DAT cut mid-line: bytes not read (0 normally)
    f.dtype                     # numpy dtype of the wire samples
    f.shape                     # (num_lines, line_size)
    f.spatial_shape             # (num_lines, positions_per_line)

    # Flag inspection — works as a property or as a method call.
    f.is_demodulated            # True / False
    f.is_interleaved
    f.is_float
    f.is_phase
    f.is_unsigned               # arctan/√ pair, or the split magnitude/ column
    f.is_ac                     # ADC AC-coupled
    f.is_hiz                    # ADC high-impedance

    # Full parsed header (byte-exact wire fields):
    f.header.num_channels       # active socket channels (1 or 2)
    f.header.content_type       # on-disk content descriptor byte (see below)
    f.header.content_type_name  # lowercase tag, e.g. "sw_mozart" / "raw" / "undefined"
    f.header.card               # card code (DAT bytes 65..81), or None
    f.header.acquisition_ns     # precise acquisition time, i64 ns Unix (0 if legacy ASCII)
```

### `content_type` — what the samples *are*

The header carries a `content_type` byte (offset 64; HDF5 attribute, TDMS
property, SEG-Y text line `CONTENT-TYPE`) describing the *meaning* of the
samples. One value stands for one quantity, one sample type and one
producing chain: `dui_coherent_phase` (19) is an `f32` phase in radians out
of the Phase+ demodulation, `hw_amplitude` (5) the card's `u16` magnitude,
`viz_variance` (27) zone variances exported from the console's archives…
It is the reference for every file that carries it. Legacy captures (which
had no such byte) decode as `undefined` (0) and keep the flag-based reading
they always had.

```python
from invisensing import File, content

with File("acquisition.h5") as f:
    content.quantity_of(f.content_type)   # "phase"
    content.unit(f.content_type)          # "rad"
    content.method_of(f.content_type)     # "DUI_COHERENT"
    content.to_physical(3)                # arctan → rad: v · π / 32768
```

`invisensing.content` is the registry (values 0–32) with the
acquisition's helpers (`quantity_of`, `members`, `member_of`,
`sample_type`, `is_phase`, `is_stream_only`, `split_children`,
`method_of`, `unit`, `to_physical`…); it is checked against the registry
the acquisition publishes. Since 1.5.0 it knows the phases computed on the
software front end by DUI and DUI_STATEFUL (`dui_arctan_phase` 29,
`dui_stateful_arctan_phase` 30, `Mode.PHASE`) and their pairs with the
amplitude kept (31, 32); the software front end's amplitude
(`sw_amplitude_sac`, 11) has no single `method_of` any more, as several
methods write it. A value this SDK does not know reads
`unknown(<n>)` — never `undefined` — unless the file names it itself (HDF5
/ TDMS `content_type_name`, used by `f.content_type_name`), and its samples
are decoded from `sample_size` and the flags, with a warning.

`f.is_mozart` follows the content type (`True` for `sw_mozart` and
`sw_mozart_plus`; `None` for `raw`); on a legacy file it keeps reading the
`MOZART` flag. The `SCHUBERT` flag is deprecated: nothing is read from it,
and `False` means "not Mozart", not "Schubert".

A magnitude column — the `magnitude/` child of a split arctan/magnitude
stream, `hw_amplitude` — reads as `uint16` in every format (an HDF5 stored
as `int16` with `unsigned = 1`, or as `uint16`).

The acquisition timestamp is also surfaced as a precise `i64` nanosecond Unix
instant via `f.header.acquisition_ns` (it is `0` when the file used the legacy
ASCII timestamp string, which is still readable through `f.timestamp`). For
HDF5 it comes from the `timestamp_ns` attribute (since 1.3.0; earlier
versions reported `0` for HDF5); in a TDMS from the `timestamp_ns` property
and in a SEG-Y from the `TIMESTAMP NS` text line of data format v1. Older
TDMS and SEG-Y files carry no absolute time — only a display string
(`f.timestamp`, the node's local time with its UTC offset) — and neither
format carries the time flags.

### Real trigger frequency (since 1.3.0)

A card fires on a divided clock, so it hits the requested trigger rate
exactly only when that rate divides its clock. A card on a 250 MHz
clock asked for 75 kHz programs a period of `floor(250e6 / 75000) =
3333` ticks and fires at `250e6 / 3333 = 75007.5008` Hz (+100 ppm):
counting time with the requested 75 000 Hz drifts by 0.36 s per hour.
Recordings made with an inexact rate (HDF5 only — the acquisition refuses
DAT, TDMS and SEG-Y for them) carry the card's exact rational:

```python
with File("acquisition.h5") as f:
    f.trig_frequency            # 75000           — what was requested
    f.trig_frequency_real_hz    # 75007.50075...  — what the card did
    f.trig_frequency_is_exact   # False
    f.trig_clock_hz, f.trig_period_ticks   # (250000000, 3333)

    # Time axes and durations: use the real rate…
    t_s = np.arange(f.num_lines) / f.trig_frequency_real_hz
    # …or the exact integer timeline the acquisition stamped:
    t0 = f.line_time_ns(0)      # header.acquisition_ns
    t1 = f.line_time_ns(75_000) # t0 + 999_900_000 (not 1 s)
```

`line_time_ns(i)` is `acquisition_ns + ⌊i · ticks · 10⁹ / clock⌋` in exact
integer arithmetic — the same integer the Audace acquisition computes. On
an exact rate (2 000 Hz, 10 kHz, 200 kHz…) and on every file without the
rational (older recordings, external trigger, DAT/TDMS/SEG-Y), the
properties fall back on `trig_frequency` and every time and duration is
exactly what 1.2.x computed. Where the rational is stored: HDF5 attributes
`trig_clock_hz` (u32), `trig_period_ticks` (u32) and
`trig_frequency_real_hz` (f64); in the 128-byte header, bytes 88..92 and
92..96 (u32 LE, `0` = unknown). A rational further from `trig_frequency`
than a card's own quantization puts it — `f / q + 1` Hz, `q = ⌊clock /
10 MHz⌋` ticks (25 on a 250 MHz clock: up to 4 % near 10 MHz) — does not
describe the file and is ignored.

### Corrected line times (since 1.4.0)

The Audace acquisition corrects the drift of the card's oscillator and the
shots it lost (its *time map*); the recorder writes the axis of each HDF5
file into its attributes, and `line_time_ns(i)` follows it:

```python
with File("acquisition.h5") as f:
    f.time_correction           # "cardclock-v1" | "none" | None (older file)
    f.line_time_ns(1000)        # corrected time of line 1000, exact int ns
    f.line_time_ns(1000, nominal=True)  # the count, as 1.3.0 gave it
    t = f.times_ns()            # every line, int64 numpy vector
    f.trig_frequency_measured_hz, f.time_uncertainty_ns
```

On a corrected file `timestamp_ns` is the corrected start and
`timestamp_nominal_ns` the count. Spectra and filters keep
`trig_frequency_real_hz` as their sampling rate. Files without the
attributes are timed exactly as before.

## Data format v1 (since 1.4.0)

Recordings made since the acquisition's data format v1 describe themselves
in their header (all formats; `0` / absent on older files):

- the **content type** fixes the sample type: a v1 file writes its
  `FLOAT`, `UNSIGNED`, `INTERLEAVED`, `DEMODULATED`, `PHASE` and `MOZART`
  flags as 0 (only `AC` and `HIZ` stay), and this SDK types and classifies
  it from the content type alone (`f.dtype`, `f.mode`, the `is_*`
  properties). Files with content type 0 are read from their flags, as
  before;
- every file holds **one quantity** (a pair is split into two files,
  `f.sibling_path`);
- `f.format_version` (1), `f.duration_s` (written when the file closed),
  `f.spatial_step_m` and `f.temporal_step_s` (DAT bytes 104..120; HDF5 /
  TDMS attributes in f64; SEG-Y text lines);
- `f.card`: the code of the card that recorded it — `p18`, `p21`, `s125`
  or `s500` — on every file (before v1 only DAT headers carried it, as a
  tag read as the same code: every run's from 2026-07-15 to 2026-10-13,
  then a two-card run's);
- new HDF5 files are named `.h5`; `.hdf5` is read as always;
- SEG-Y textual headers are read from the file's bytes, in ASCII (what the
  acquisition writes) or EBCDIC;
- the archive exports of zone variances: `viz_variance` (27) and
  `viz_logvar` (28, the console's Log scale `log10(1 + variance)`), with
  `f.get_variance()` / `f.get_log_variance()` and, in HDF5,
  `f.source_content_type` (the column the variance was computed from).

The header bit `0x020`, never set by any producer, is now
`FLAG_RESERVED`.

## Writing files

```python
from invisensing import File, export_dat
import numpy as np

samples = np.random.randn(10_000, 512).astype("float32")
export_dat(
    "out.dat",
    samples,
    sample_rate=250_000_000,
    trig_frequency=2_000,
    range_v=1.0,
    timestamp="2026-05-25_12:34:56",
)

# A file of data format v1: say what the samples are.
export_dat("phase.dat", samples, sample_rate=250_000_000, trig_frequency=2_000,
           content_type=17, card="p21")

# Lines 1000..2000 of a recording, with its whole header (content type,
# card, steps, trigger pre-roll, the card's rational, start time of line 1000):
with File("phase/2026-11-03_10-00-00.h5") as f:
    f.export_dat("excerpt.dat", start=1000, count=1000)
```

`export_dat` writes the 128-byte Audace header (matching the C-side
wire format byte-for-byte) followed by the array's raw samples. Without a
content type the `FLOAT` flag is set automatically when
`data.dtype.kind == 'f'`. With `header=f.header` it copies every field of
the source, as the acquisition's conversions do: a content type of the
registry gives a v1 file (its flags, steps, card code, and the duration of
the lines written), anything else a legacy copy. A recording whose card
fired at an inexact rate is refused (keep it in HDF5).

## API stability

Everything exported via `from invisensing import *` is part of the
**stable public API** starting from version 1.0.0. We follow semantic
versioning:

- **Patch releases** (0.2.x): bug fixes, performance improvements.
- **Minor releases** (0.x.0): additive changes only (new methods, new
  formats, new optional arguments). Existing code keeps working
  without changes.
- **Major releases** (x.0.0): breaking changes — announced via the
  changelog with a migration guide.

## Backwards compatibility with the legacy SDK

The original `invisensing.File` module is preserved unchanged:

```python
import invisensing.File as iFile

file = iFile.File("acquisition.dat")
print(file.get_line_size(), file.get_trigger_frequency())
print(file.is_demodulated(), file.is_acquisition_ac())

while file.get_lines_left() > 0:
    data = file.get_lines(5)
    # process …

iFile.export("out.dat", data, file.get_timestamp(),
             file.get_trigger_frequency(), file.get_sample_rate(),
             file.get_range())
```

The legacy class delegates to the same Rust-backed implementation as
the modern `File`, so legacy scripts run at the new speed without any
modification.

## Performance & safety

### Performance

The lib is built to be the fastest credible way to get DAS samples
into a numpy array in Python.

- **DAT reads** stream through a `BufReader<File>`; the inner
  `read_exact` runs with the **GIL released** so a second Python
  thread can do work while the OS is blocked on disk.
- **No zero-init pass**: the typed-conversion path allocates an
  uninitialised `Vec<T>` (via `Vec::with_capacity` + `set_len`) and
  fills it with one `copy_nonoverlapping`. Saves the 4 GB of pointless
  DRAM writes a `vec![0; n]` would do on a 4 GB acquisition.
- **De-interleave kernels** read the contiguous numpy buffer via
  `as_slice()` and iterate with `chunks_exact(2)` — LLVM
  auto-vectorises this into SSE/AVX gather-extract instructions on
  x86_64. The output `Vec` is also `set_len`'d, no per-element
  capacity check.
- **HDF5 / TDMS / SEG-Y** loading is handled by their respective
  Python libraries (themselves C-backed). The de-interleave step
  always runs through the same Rust kernels, regardless of the source
  format — channel extraction perf doesn't depend on the file format.
- **Zero allocation on the hot path** in the channel extractors: each
  call returns a single freshly-allocated numpy array; no intermediate
  scratch buffers, no per-row copies.

Measured on a recent laptop (single-threaded, release build):

| Operation | Throughput |
|---|---|
| `read_lines()` from a `.dat` (i16 IQ, 100 MB chunk) | ≈ **3.9 GB/s** |
| `get_i()` + `get_q()` on a pre-loaded buffer | ≈ **3.0 GB/s** |
| `get_iq()` → `complex64` packing | ≈ same order of magnitude |

Reproduce with `pytest tests/test_performance.py -v -s`. The throughput
sanity tests also fail loudly (assertion) if a regression cuts perf
below the floor — `> 50 MB/s` for `read_lines`, `> 100 MB/s` for the
kernels — so a future refactor can't silently re-introduce the
`Vec::push` / `vec![0; n]` anti-patterns.

### Safety

- **Strict header validation at open time**: a `File` constructor
  rejects up front any header with `line_size <= 0`, `sample_size`
  not in `{1, 2, 4, 8}`, or an INTERLEAVED file with an odd
  `line_size`. The downstream maths can therefore assume positive
  sizes everywhere.
- **Bounded `unsafe`**: the whole extension contains three small
  `unsafe` blocks (uninit `Vec` via `set_len`, the byte→typed
  primitive copy, and the i16→u16 bitcast for the magnitude lane).
  Each one has a one-line invariant in a `// SAFETY:` comment and is
  exercised by the 1-million-sample correctness tests in
  `tests/test_performance.py`.
- **Clear Python exceptions** instead of crashes: every error path
  (missing file, malformed header, short read, wrong mode for the
  extractor) raises a typed `OSError` / `ValueError` / `FileNotFoundError`
  / `ImportError` with a message that names the field at fault.
- **No silent type coercion**: the wire dtype is preserved by default;
  explicit `*_volts` / `*_radians` methods do the physical scaling
  when you want it. No implicit `astype(float64)` surprise on a 4 GB
  array.
- **Thread safety**: each `File` owns its own backend; sharing a
  single `File` across Python threads is not supported (the file
  cursor isn't synchronised). Open one `File` per thread for parallel
  reads — they don't compete for any internal state.
- **No `mmap`**: deliberate. Memory-mapped files turn I/O errors into
  SIGBUS (segfault) instead of Python exceptions, which is unsafe
  when reading user-supplied paths in a long-running lab process.

## Examples

See [`assets/basic_usage.py`](assets/basic_usage.py) for a runnable
end-to-end script.

## Building from source

```bash
git clone <repo>
cd python-lib
pip install maturin
maturin develop --release        # local dev install
maturin build  --release         # build a wheel for distribution
```

The Rust crate lives in [`src/lib.rs`](src/lib.rs) and the Python
facade in [`python/invisensing/`](python/invisensing/).

## License

MIT — © 2024-2026 Invisensing. See [LICENSE](LICENSE) for the full text.

# User guide

This page walks through the SDK from end to end. For a focused tour
see the [Quickstart on the home page](index.md#30-second-tour); for
hands-on code see the [Examples](examples.md).

---

## Opening a file

The `File` class is the single entry point. It auto-detects the
format from the file suffix:

```python
from invisensing import File

f = File("acquisition.dat")    # DAT, native Rust reader
f = File("acquisition.h5")     # HDF5 via h5py
f = File("acquisition.tdms")   # TDMS via npTDMS
f = File("acquisition.sgy")    # SEG-Y via segyio
```

If your file has no recognised suffix, pass `format=` explicitly:

```python
f = File("acquisition_data", format="dat")
```

Use it as a context manager — the underlying file handle is closed
cleanly even when an exception is raised:

```python
with File("acquisition.dat") as f:
    ...
# file descriptor released here, not whenever `f` is collected
```

Outside a `with` block, call `f.close()` yourself. It is idempotent
(closing twice, or closing after a `with`, is a no-op) and it releases
the descriptor at the call, not at collection time:

```python
f = File("acquisition.dat")
try:
    header = f.header
finally:
    f.close()
```

A `File` that is neither closed nor used as a context manager emits a
`ResourceWarning` when it is finally collected. Like the one CPython's
built-in `open()` emits, it is silent by default and shows up under
`python -W default`, `python -X dev` and pytest. It matters on a
long-running process that walks a data tree — a labelling server, a
dashboard backend: one leaked descriptor per file ends in `EMFILE`, and
on an NFS/CIFS mount an open handle also blocks the file from being
deleted or replaced.

#### What a closed file still knows

Closing gives up the **descriptor**, not what was read from it. The
header is parsed once, at open, and stays on the object, so a closed
`File` still describes its recording:

```python
with File("acquisition.dat") as f:
    data = f.read_all()

# outside the block: the handle is gone, the description is not
dt   = 1.0 / f.trig_frequency_real_hz  # fine
axis = np.arange(f.num_lines) * dt      # fine
f.mode, f.dtype, f.shape, f.is_interleaved, repr(f)   # all fine
```

That is deliberate: asking a file you have already closed what its
sample rate was is a normal thing to do, not a mistake, and raising
there would break correct scripts for nothing.

What does raise `ValueError` after `close()` is anything that would have
to touch the file again — `read_lines()`, `read_all()`, `rewind()`,
`seek_lines()`, iterating the file — plus `lines_left`, which is the
position of a read cursor on a handle that no longer exists (use
`num_lines` to describe the file itself). numpy arrays already returned
by `read_lines()` or the channel extractors own their memory and stay
valid — closing never invalidates data you already hold, and the channel
extractors still accept a buffer you read before closing:

```python
f = File("acquisition.dat")
buf = f.read_lines(1000)
f.close()                       # release the handle as early as you can
i, q = f.get_i(buf), f.get_q(buf)   # arithmetic on your array, no file
```

One caveat: do not share a `File` across threads. Closing a `.dat`
while another thread is inside `read_lines()` on the same object cannot
release the descriptor immediately — the native reader is borrowed for
the length of that read — so the release is deferred to the end of that
read. `close()` still returns without raising and emits a
`ResourceWarning` saying so, rather than reporting a file as closed
while its handle is still open.

### Inspecting metadata

Every `File` exposes the parsed header as plain Python
properties — no struct unpacking, no flag bit-twiddling:

```python
with File("acquisition.dat") as f:
    f.line_size              # samples per pulse on the wire
    f.positions_per_line     # spatial fibre positions per pulse
    f.sample_size            # bytes per sample (2, 4, 8…)
    f.sample_rate            # Hz
    f.trig_frequency         # Hz, the REQUESTED trigger rate (int)
    f.trig_frequency_real_hz # Hz, the rate the card really fired at (float)
    f.trig_frequency_is_exact  # True when both are equal (or unknown)
    f.trig_clock_hz          # card rational (int), or None when unknown
    f.trig_period_ticks      # real rate = trig_clock_hz / trig_period_ticks
    f.line_time_ns(i)        # acquisition time of pulse i, int ns since epoch
    f.num_lines              # total trigger pulses recorded
    f.lines_left             # pulses behind the read cursor
    f.duration               # seconds (= num_lines / trig_frequency_real_hz)
    f.distance               # metres of fibre covered per pulse
    f.range                  # ADC voltage range (V)
    f.timestamp              # producer-side timestamp string
    f.dtype                  # numpy dtype of the wire samples
    f.shape                  # (num_lines, line_size)
    f.spatial_shape          # (num_lines, positions_per_line)

    # Flag predicates — properties, work in if/while idioms.
    if f.is_demodulated and f.is_interleaved:
        ...

    f.is_demodulated, f.is_interleaved, f.is_float, f.is_phase
    f.is_unsigned, f.is_ac, f.is_hiz
```

### Requested vs. real trigger frequency

A card fires on a divided clock: it hits the requested rate exactly only
when that rate divides its clock. A 250 MHz card asked for 75 kHz fires
every 3333 ticks, at `250e6 / 3333 = 75007.5008` Hz (+100 ppm); timing
the lines on 75 000 Hz drifts by 0.36 s per hour. Since 1.3.0 the SDK
reads the card's exact rational from the recordings that carry it (HDF5
attributes `trig_clock_hz`, `trig_period_ticks`, `trig_frequency_real_hz`;
header bytes 88..96 of a `.dat`):

| Property | Meaning |
|---|---|
| `trig_frequency` | the **requested** rate, an `int` (unchanged) |
| `trig_frequency_real_hz` | the rate the card fired at, a `float` — `trig_clock_hz / trig_period_ticks`, else HDF5's `trig_frequency_real_hz`, else `float(trig_frequency)` |
| `trig_frequency_is_exact` | `True` when real == requested, and when the real rate is unknown |
| `trig_clock_hz`, `trig_period_ticks` | the card's rational, or `None` (older files, external trigger, TDMS, SEG-Y) |
| `line_time_ns(i)` | `header.acquisition_ns + ⌊i · ticks · 10⁹ / clock⌋` (else `⌊i · 10⁹ / trig_frequency⌋`), an exact `int` |
| `duration` | `num_lines / trig_frequency_real_hz` |

Compute time axes and durations on `trig_frequency_real_hz` (or with
`line_time_ns`); keep `trig_frequency` for sizing (pulses per block) and
labels. At an exact rate and on every file without the rational, the
results are bit-for-bit those of 1.2.x. A rational more than 1 % (and
more than 1 Hz) away from `trig_frequency` cannot describe the file and
is ignored (`trig_clock_hz` is then `None`; the stored values stay in
`f.header.trig_clock_hz` / `f.header.trig_period_ticks`).

---

## Acquisition modes (`Mode` enum)

A PCIe7821 acquisition can run in one of four on-board DSP modes.
Each produces a different file content. The `Mode` enum lets you
dispatch on what's in the file with one `match` statement:

```python
from invisensing import Mode

with File("acquisition.dat") as f:
    match f.mode:
        case Mode.RAW:               ...
        case Mode.IQ:                ...
        case Mode.ARCTAN_MAGNITUDE:  ...
        case Mode.PHASE:             ...
```

| `Mode` | Wire layout per pulse | What the FPGA writes |
|---|---|---|
| `RAW` | `[s₀, s₁, …, s_{N-1}]` | Raw ADC codes, no DSP |
| `IQ` | `[I₀, Q₀, I₁, Q₁, …]` | I/Q pair after NCO + LPF, interleaved |
| `ARCTAN_MAGNITUDE` | `[arctan₀, √₀, arctan₁, √₁, …]` | `arctan(Q/I)` + `√(I²+Q²)`, interleaved |
| `PHASE` | `[φ₀, φ₁, …, φ_{N-1}]` | Unwrapped DAS phase, post fading-suppression + gauge differential + detrend |

!!! warning "The SDK is mode-faithful, not mode-translating"
    The library **never** silently re-derives one product from
    another. The on-board DSP chain (fading suppression, spatial
    differential, detrend filter) is not reproducible in software
    from an earlier tap, so the channel you can extract is the one
    the FPGA wrote. Calling the wrong extractor raises a
    `ValueError` that names both modes:

    ```python
    >>> with File("phase.dat") as f:
    ...     f.get_i()
    ValueError: get_i: file mode is 'phase', but this extractor
    only applies to 'iq'.
    ```

---

## Extracting channels

Every extractor has the same shape: `(rows, positions_per_line)`,
typed with the most natural dtype for the lane. Each one accepts an
optional pre-read buffer so you can reuse it across extractors
without double-advancing the cursor:

```python
with File("iq.dat") as f:
    chunk = f.read_lines(1000)
    i = f.get_i(chunk)            # reuses the buffer
    q = f.get_q(chunk)            # same — no extra file read
    iq = f.get_iq(chunk)          # same again
```

### Default extractors — wire dtype

Fast, no copy beyond the de-interleave, no precision loss on
round-trip writes.

| `Mode` | Method | dtype | Notes |
|---|---|---|---|
| `RAW` | `read_lines()` | `int16` | ADC codes |
| `IQ` | `get_i()` / `get_q()` | `int16` | One lane each |
| `IQ` | `get_iq()` | `complex64` | `I + j·Q` packed |
| `ARCTAN_MAGNITUDE` | `get_arctan()` | `int16` | Fixed-point: `32767 ↔ +π` |
| `ARCTAN_MAGNITUDE` | `get_magnitude()` | `uint16` | Bitcast from wire i16 |
| `PHASE` | `get_phase()` | `float32` | Radians (already converted) |

### Physical-unit extractors — `float32`

When you start doing DSP, you typically want volts and radians, not
fixed-point codes. The `_volts` / `_radians` family does the
scaling for you:

| `Mode` | Method | dtype | Unit | Scaling |
|---|---|---|---|---|
| `IQ` | `get_i_volts()` / `get_q_volts()` | `float32` | V | `i16 × range / 32768` |
| `IQ` | `get_iq_volts()` | `complex64` | V | real/imag both in volts |
| `ARCTAN_MAGNITUDE` | `get_arctan_radians()` | `float32` | rad | `i16 × π / 32768` |
| `ARCTAN_MAGNITUDE` | `get_magnitude_volts()` | `float32` | V | `u16 × range / 32768` |

```python
with File("iq.dat") as f:
    iq_v = f.get_iq_volts()       # complex64, volts
    envelope_v = np.abs(iq_v)     # volts
    phase_rad  = np.angle(iq_v)   # radians, wrapped
```

!!! info "Memory cost of physical units"
    Default extractors keep the wire dtype — a 4 GB i16 capture
    becomes a 4 GB numpy array. The `_volts` / `_radians` family
    allocates a new `float32` buffer (2× the size for i16/u16
    inputs, 8× for the `complex64` `get_iq_volts`). For
    multi-GB captures, prefer the wire dtype + an explicit
    `.astype(np.float32) * scale` inside your processing loop.

### Get every channel in one call

For ad-hoc scripts, `channels()` returns a `dict` keyed by mode:

```python
with File("iq.dat") as f:
    ch = f.channels()
    # {"i": int16, "q": int16, "iq": complex64}

with File("arctan_mag.dat") as f:
    ch = f.channels()
    # {"arctan": int16, "magnitude": uint16}
```

---

## Streaming large files

`read_lines(n)` advances a cursor in the file. Iterate to process
captures larger than memory:

```python
with File("long_capture.h5") as f:
    while f.lines_left:
        chunk = f.read_lines(10_000)
        process(chunk)
```

For one-pulse-at-a-time processing, iterate the file directly:

```python
with File("long_capture.dat") as f:
    for pulse in f:               # yields (line_size,) arrays
        process(pulse)
```

For small files, `read_all()` returns everything in one allocation:

```python
with File("small.dat") as f:
    data = f.read_all()           # (num_lines, line_size)
```

`rewind()` resets the cursor:

```python
with File("small.dat") as f:
    a = f.read_all()
    f.rewind()
    b = f.read_all()
    assert (a == b).all()
```

---

## File formats — what each backend does

The `File` facade dispatches to a format-specific backend. All four
funnel through the same `Header` so the channel extractors above are
format-agnostic.

### DAT — native Rust

128-byte header + raw little-endian samples. Read by the native
`invisensing._core.DatReader` — no third-party dep, no Python
overhead in the hot loop. The most common format and the fastest
path.

### HDF5 — via `h5py`

Audace HDF5 files store the samples as a `(num_traces,
samples_per_trace)` dataset named `acoustic_data`, typed by
`bytes_per_sample` and the `FLOAT` attribute. Every header field is
mirrored as a scalar attribute. Lines are read slice by slice: a
multi-GB capture never has to fit in RAM.

Recordings written with the FileWriter's `compression = ZSTD` store the
dataset in chunks through the standard pipeline byte shuffle + zstd
(HDF5 filter 32015). `invisensing` imports `hdf5plugin`, a dependency,
so they open like any other file; outside it, run `import hdf5plugin`
before opening one with `h5py`, or the file does not open.

### TDMS — via `npTDMS`

Single channel `acoustic_data` under the group `AudaceGroup`. Header
fields travel as file-level properties.

### SEG-Y — via `segyio`

One trace per fibre position, `samples_per_trace = line_size`
(wire-side, doubled for `INTERLEAVED`). Header fields not expressible
in the SEG-Y binary header are encoded as discrete lines in the
EBCDIC textual header (e.g. `C21 HEADER FLAGS (raw bits):
0xNNNNNNNN`). The SDK parses those back into a `Header`.

---

## Writing files

`export_dat()` writes a 2-D numpy array out as an Audace `.dat` file
(128-byte header + raw samples):

```python
from invisensing import export_dat
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
```

The `FLOAT` flag is set automatically when `data.dtype.kind == 'f'`.

---

## Performance & safety notes

### Performance

- DAT reads stream through a `BufReader<File>`; the inner
  `read_exact` runs with the **GIL released**.
- The typed-conversion path allocates an uninitialised `Vec<T>` and
  fills it with one `copy_nonoverlapping`. Skips the 4 GB of
  pointless DRAM writes a `vec![0; n]` would do on a 4 GB read.
- De-interleave kernels read via `as_slice()` and iterate with
  `chunks_exact(2)` — LLVM auto-vectorises into SSE/AVX
  gather-extract instructions on x86_64.
- HDF5 / TDMS / SEG-Y loading is C-backed by their respective
  libraries; the de-interleave step always runs through the same
  Rust kernels, so channel-extraction perf doesn't depend on the
  source format.

### Safety

- **Strict header validation** at open: `line_size > 0`,
  `sample_size ∈ {1, 2, 4, 8}`, `INTERLEAVED ⇒ even line_size`.
  Downstream maths can assume positive sizes everywhere.
- **Bounded `unsafe`**: three small blocks in the Rust extension,
  each one documented and exercised by 1-million-sample
  correctness tests.
- **No `mmap`**: deliberate. Memory-mapped files turn I/O errors
  into SIGBUS (segfault) instead of Python exceptions, which is
  unacceptable in a long-running lab process.
- **Thread safety**: each `File` owns its own backend; sharing a
  single `File` across Python threads is not supported. Open one
  per thread for parallel reads — they don't compete for any
  internal state.

---

## Legacy SDK compatibility

The original `invisensing.File` module is preserved unchanged:

```python
import invisensing.File as iFile

file = iFile.File("acquisition.dat")
file.get_line_size(), file.get_trigger_frequency()
file.is_demodulated(), file.is_acquisition_ac()

while file.get_lines_left() > 0:
    data = file.get_lines(5)
    process(data)

iFile.export("out.dat", data, file.get_timestamp(),
             file.get_trigger_frequency(), file.get_sample_rate(),
             file.get_range())
```

The legacy class delegates to the same Rust-backed implementation as
the modern `File`, so legacy scripts run at the new speed without any
modification.

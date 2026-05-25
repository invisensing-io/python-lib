# Invisensing — Python SDK for the Audace DAS environment

`invisensing` is the official Python library for reading **Distributed
Acoustic Sensing (DAS)** acquisition files produced by the Audace
platform. It handles every file format the platform writes (`.dat`,
`.hdf5`, `.tdms`, `.sgy`) through a single uniform API, and exposes
the channels each file actually contains — `I` / `Q` / `arctan` /
`magnitude` / `phase` — in one method call, with the right dtype and
optional physical-unit scaling.

The library is **format-agnostic but mode-faithful**: it never
silently re-derives one demodulation product from another. The
on-board PCIe7821 DSP chain (fading suppression, spatial differential,
detrend filter) is not reproducible in software from an earlier tap,
so the channel you can extract is the one the FPGA wrote. Inspect
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
`npTDMS`, and `segyio` so any file written by the Audace FileWriter
can be opened without further setup.

Wheels are built from a Rust extension; `pip` handles the native build
transparently. No Rust toolchain is required for end users.

> **Note** — the format extras (`pip install invisensing[hdf5]` etc.)
> are still recognised for backwards compatibility with older
> installation guides, but they are no-ops now: every backend is part
> of the default install.

## Quick start

```python
from invisensing import File, Mode

with File("acquisition.dat") as f:
    print(f)                       # File('…', mode=iq, shape=…, …)
    print(f.duration, "s")
    print(f.distance, "m of fibre")

    # The Mode enum lets you dispatch cleanly on what the file contains.
    match f.mode:
        case Mode.RAW:
            data = f.read_lines(1000)              # (1000, line_size) ADC codes
        case Mode.IQ:
            i = f.get_i()                          # (n, positions) i16
            q = f.get_q()                          # (n, positions) i16
            iq = f.get_iq()                        # (n, positions) complex64
        case Mode.ARCTAN_MAGNITUDE:
            arctan = f.get_arctan()                # (n, positions) i16
            magnitude = f.get_magnitude()          # (n, positions) u16
        case Mode.PHASE:
            phase = f.get_phase()                  # (n, positions) f32 radians
```

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
(volts for I/Q/magnitude, radians for arctan) that the FPGA encodes as
fixed-point integers on the wire. The SDK offers two flavours of
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
| **Phase** | `read_lines()` / `get_phase()` | `float32` | **rad** | already converted by Cardcontrol (`i32 × π/32768`) |

Phase mode is the only one whose payload is already floating-point on
the wire — the conversion happens in the acquisition driver so
consumers never need to know the vendor's `π/32768` scaling.

`f.dtype` always reflects what `read_lines()` will return, derived
from `sample_size` and the `FLOAT` flag in the header.

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

`PCIe7821` on-board DSP can emit four different products. The SDK
maps each to the right extractor:

| Mode | Wire layout (per pulse) | Extractor(s) | Dtype out |
|---|---|---|---|
| `Mode.RAW` | `[s0, s1, …, sN-1]` | `read_lines()` | `int16` |
| `Mode.IQ` | `[I0, Q0, I1, Q1, …]` | `get_i()`, `get_q()`, `get_iq()` | `int16` / `int16` / `complex64` |
| `Mode.ARCTAN_MAGNITUDE` | `[atan0, √0, atan1, √1, …]` | `get_arctan()`, `get_magnitude()` | `int16` / `uint16` |
| `Mode.PHASE` | `[φ0, φ1, …, φN-1]` | `get_phase()` | `float32` (radians) |

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
applies to 'iq'. Inspect file.mode to dispatch — see the Mode enum docs.
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
    f.trig_frequency            # Hz
    f.num_lines                 # total pulses recorded
    f.lines_left                # remaining behind the cursor
    f.duration                  # seconds (= num_lines / trig_frequency)
    f.distance                  # metres of fibre covered by one pulse
    f.range                     # ADC voltage range (V)
    f.timestamp                 # producer-side timestamp string
    f.dtype                     # numpy dtype of the wire samples
    f.shape                     # (num_lines, line_size)
    f.spatial_shape             # (num_lines, positions_per_line)

    # Flag inspection — works as a property or as a method call.
    f.is_demodulated            # True / False
    f.is_interleaved
    f.is_float
    f.is_phase
    f.is_unsigned               # arctan/√ files only
    f.is_ac                     # ADC AC-coupled
    f.is_hiz                    # ADC high-impedance
```

## Writing files

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

`export_dat` writes the 128-byte Audace header (matching the C-side
wire format byte-for-byte) followed by the array's raw samples. The
`FLOAT` flag is set automatically when `data.dtype.kind == 'f'`.

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

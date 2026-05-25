# Invisensing Python SDK

`invisensing` is the official Python library for reading
**Distributed Acoustic Sensing (DAS)** acquisition files produced by
the Invisensing Audace platform. It handles every file format the
platform writes (`.dat`, `.hdf5`, `.tdms`, `.sgy`) through a single
uniform API, and exposes the channels each file actually contains —
**I / Q / arctan / magnitude / phase** — in one method call, with the
right `dtype` and optional physical-unit scaling.

The hot path (header parsing, bytes → numpy conversion,
de-interleave) is implemented in **Rust** and exposed through PyO3, so
reading a multi-GB capture stays fast even from Python.

[Install](#installation){ .md-button .md-button--primary }
[Quickstart](#30-second-tour){ .md-button }
[GitHub](https://github.com/invisensing-io/python-lib){ .md-button }

---

## Installation

```bash
pip install invisensing
```

That's it. Every supported file format (DAT, HDF5, TDMS, SEG-Y) works
out of the box. The default install pulls in `numpy`, `h5py`,
`npTDMS`, and `segyio`. Pre-built wheels ship for **CPython
3.9–3.14** on:

- Linux x86_64 + aarch64
- macOS arm64 (Apple Silicon)
- Windows x86_64

Intel-Mac users install via the same command — `pip` falls back to
compiling from the source distribution, which needs a Rust toolchain
(`rustup` one-liner).

> The format extras (`pip install invisensing[hdf5]` etc.) are still
> recognised for backwards compatibility but are no-ops now — every
> backend is part of the default install.

---

## 30-second tour

```python
from invisensing import File, Mode

with File("acquisition.dat") as f:
    print(f)                       # File('…', mode=iq, shape=…, …)
    print(f"{f.duration:.1f}s of recording, {f.distance:.0f}m of fibre")

    match f.mode:
        case Mode.RAW:
            data = f.read_lines(1000)              # (1000, line_size) ADC codes
        case Mode.IQ:
            i = f.get_i()                          # int16
            q = f.get_q()
            iq = f.get_iq_volts()                  # complex64, in volts
        case Mode.ARCTAN_MAGNITUDE:
            arctan = f.get_arctan_radians()        # float32 radians
            magnitude = f.get_magnitude_volts()    # float32 volts
        case Mode.PHASE:
            phase = f.get_phase()                  # float32 radians
```

That's the whole API surface for 95% of use cases. The
[**Guide**](guide.md) walks through each piece in detail; the
[**Examples**](examples.md) page shows complete DAS workflows; the
[**API Reference**](api.md) is the exhaustive docstring listing.

---

## Why a Rust core?

DAS captures get big fast — a single second of dual-channel IQ data at
250 MSps is ~1 GB. Parsing the header, copying bytes into a numpy
array, and de-interleaving the I/Q pair lanes are all bandwidth-bound
loops. We measured (on a recent laptop, single-threaded, release build):

| Operation | Throughput |
|---|---|
| `read_lines()` from `.dat` (i16 IQ, 100 MB chunk) | ≈ **3.9 GB/s** |
| `get_i()` + `get_q()` on a pre-loaded buffer | ≈ **3.0 GB/s** |

Implementing those loops in pure Python would land in the 100–200
MB/s range. The Rust extension uses `chunks_exact(2)` (auto-vectorised
by LLVM on SSE/AVX) and skips the zero-init pass on multi-GB
allocations. The Python side is a thin facade for ergonomics.

The `read_lines` call **releases the GIL** during the actual disk
I/O — a second Python thread can do work while the first is blocked
on a multi-GB read.

---

## Safety

- **Strict header validation** at open time: any `line_size <= 0`,
  unsupported `sample_size`, or odd `line_size` on an `INTERLEAVED`
  file is rejected up front.
- **Clear Python exceptions** instead of segfaults — every error path
  (missing file, malformed header, short read, wrong mode) raises a
  typed `OSError` / `ValueError` / `FileNotFoundError` with a
  message that names the field at fault.
- **No silent type coercion** — the wire dtype is preserved by
  default; explicit `_volts` / `_radians` methods scale to physical
  units when you want them. No accidental `astype(float64)` on a 4
  GB array.
- **Thread safety**: each `File` owns its own backend; open one per
  thread for parallel reads.

---

## API stability

Everything exported via `from invisensing import *` is part of the
**stable public API** starting from version `1.0.0`. We follow
[semantic versioning](https://semver.org):

- **Patch releases** (1.0.x): bug fixes, performance improvements.
- **Minor releases** (1.x.0): additive changes only (new methods,
  new formats, new optional arguments). Existing code keeps working
  without changes.
- **Major releases** (x.0.0): breaking changes — announced via the
  changelog with a migration guide.

---

## Quick links

- [Source code on GitHub](https://github.com/invisensing-io/python-lib)
- [PyPI package page](https://pypi.org/project/invisensing/)
- [Release notes](https://github.com/invisensing-io/python-lib/releases)
- [Issue tracker](https://github.com/invisensing-io/python-lib/issues)

## License

MIT — © 2024-2026 Invisensing.

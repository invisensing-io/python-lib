# API reference

Generated directly from the docstrings of the `invisensing` package
(version always matches the wheel you're reading these docs from).

The stable public symbols (those listed in `invisensing.__all__`) are
documented below. Anything not listed there is internal and may
change without notice.

---

## File

The unified reader. Auto-detects the on-disk format from the path
suffix and exposes the channels each file actually contains.

::: invisensing.File

---

## Mode

High-level acquisition mode of a file. Use this enum to dispatch on
what the file contains:

```python
from invisensing import File, Mode

with File("acquisition.dat") as f:
    match f.mode:
        case Mode.RAW:               ...
        case Mode.IQ:                ...
        case Mode.ARCTAN_MAGNITUDE:  ...
        case Mode.PHASE:             ...
```

::: invisensing.Mode

---

## Header

Parsed 128-byte Audace `DataHeader`. The format-specific backends
(HDF5 / TDMS / SEG-Y) reconstruct one of these from their native
metadata so the channel extractors take a uniform path regardless of
the source file.

You rarely need to instantiate one directly — `File.header` returns
it for you. The constructor is exposed for advanced users (e.g.
synthesising files in tests).

::: invisensing.Header

---

## DatReader

Low-level streaming reader for `.dat` files. The `File` class wraps
this for the common path; use it directly if you need access to the
raw cursor (rare).

::: invisensing.DatReader

---

## Functions

### `parse_header`

::: invisensing.parse_header

### `export_dat`

::: invisensing.export_dat

---

## Header-flag constants

The raw flag bits used by the Audace wire-format header. You typically
don't need these — `File.is_*` properties and `Mode.from_flags()`
cover the common cases. They're exposed for callers who need to
inspect or build a `Header` by hand.

| Constant | Hex | Meaning |
|---|---|---|
| `FLAG_DEMODULATED` | `0x001` | Samples are the output of on-board DSP |
| `FLAG_FLOAT` | `0x002` | Samples are IEEE-754 floats |
| `FLAG_AC` | `0x004` | ADC was AC-coupled |
| `FLAG_HIZ` | `0x008` | ADC ran in high-impedance mode |
| `FLAG_SCHUBERT` | `0x010` | (reserved, internal) |
| `FLAG_SPECTRUM` | `0x020` | (reserved, internal) |
| `FLAG_MOZART` | `0x040` | Software-Mozart pipeline marker |
| `FLAG_PHASE` | `0x080` | Carries unwrapped DAS phase (rad) |
| `FLAG_INTERLEAVED` | `0x100` | Wire carries 2 samples per spatial position |
| `FLAG_UNSIGNED` | `0x200` | Odd lane of an INTERLEAVED pair is `u16` |
| `HEADER_SIZE` | `128` | Size of the wire header in bytes |

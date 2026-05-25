# Examples

Runnable snippets for the most common DAS workflows. All examples
import:

```python
import numpy as np
import matplotlib.pyplot as plt
from invisensing import File, Mode
```

---

## 1. Inspect a capture without reading the samples

Open, print the metadata, close — no sample bytes touched:

```python
with File("capture.dat") as f:
    print(f)
    # File('capture.dat', mode=iq, shape=(10000, 974),
    #      sample_rate=250000000 Hz, trig_frequency=1000 Hz,
    #      duration=10.00s, distance=507.21m)

    print(f"Mode:               {f.mode.value}")
    print(f"Pulses:             {f.num_lines:,}")
    print(f"Positions per pulse:{f.positions_per_line:,}")
    print(f"Sample rate:        {f.sample_rate / 1e6:.0f} MSps")
    print(f"Trigger rate:       {f.trig_frequency} Hz")
    print(f"Duration:           {f.duration:.2f} s")
    print(f"Round-trip distance:{f.distance:.1f} m of fibre")
    print(f"Range:              ±{f.range:.2f} V")
    print(f"Timestamp:          {f.timestamp}")
```

Useful as a quick sanity check before launching a long batch
processing job.

---

## 2. Plot the first pulse of an IQ capture

```python
with File("iq.dat") as f:
    assert f.mode is Mode.IQ
    buf = f.read_lines(1)             # first pulse only

    i = f.get_i(buf)[0]               # (positions_per_line,)
    q = f.get_q(buf)[0]
    position_m = np.arange(f.positions_per_line) * (f.distance / f.positions_per_line)

    fig, (ax_iq, ax_iv) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_iq.plot(position_m, i, label="I", lw=0.8)
    ax_iq.plot(position_m, q, label="Q", lw=0.8)
    ax_iq.set_ylabel("ADC code")
    ax_iq.legend()

    # Same data in volts using the physical-unit extractor.
    iq_v = f.get_iq_volts(buf)[0]
    ax_iv.plot(position_m, np.abs(iq_v) * 1000, label="|I+jQ| [mV]", color="C2")
    ax_iv.set_ylabel("Envelope (mV)")
    ax_iv.set_xlabel("Along-fibre distance (m)")
    ax_iv.legend()
    plt.tight_layout()
    plt.show()
```

---

## 3. Wrapped phase from an IQ capture

```python
with File("iq.dat") as f:
    assert f.mode is Mode.IQ

    # Stream 1 second at a time, accumulate the wrapped phase trace.
    n_per_block = f.trig_frequency               # 1 second of pulses
    phase_blocks = []

    while f.lines_left:
        chunk = f.read_lines(n_per_block)
        iq = f.get_iq_volts(chunk)               # complex64, volts
        wrapped = np.angle(iq).astype(np.float32)
        phase_blocks.append(wrapped)

    phase_rad = np.concatenate(phase_blocks)     # (num_lines, positions)
    print("Wrapped phase shape:", phase_rad.shape)
```

This gives you the *wrapped* (`±π`) phase computed in software from
I/Q. For unwrapped + fading-suppressed + gauge-differential phase,
acquire in `Mode.PHASE` directly instead — the FPGA does the heavy
lifting.

---

## 4. Magnitude waterfall from ArctanMagnitude

```python
with File("arctan_mag.dat") as f:
    assert f.mode is Mode.ARCTAN_MAGNITUDE
    mag_v = f.get_magnitude_volts(n=f.num_lines)   # (num_lines, positions), V

    t = np.arange(mag_v.shape[0]) / f.trig_frequency
    x_m = np.arange(mag_v.shape[1]) * (f.distance / mag_v.shape[1])

    plt.figure(figsize=(10, 6))
    plt.imshow(
        mag_v * 1000,                      # mV
        aspect="auto", origin="lower",
        extent=[x_m[0], x_m[-1], t[0], t[-1]],
        cmap="viridis",
    )
    plt.xlabel("Along-fibre distance (m)")
    plt.ylabel("Time (s)")
    plt.colorbar(label="|envelope| (mV)")
    plt.title("Magnitude waterfall")
    plt.show()
```

---

## 5. Process a multi-GB capture without OOMing

The naive `read_all()` would load 20 GB at once. Streaming
chunk-by-chunk keeps the peak memory bounded:

```python
with File("multi_gb.dat") as f:
    chunk_size = 5_000               # ~5s of pulses at 1 kHz triggers
    running_var = np.zeros(f.positions_per_line, dtype=np.float64)
    n_seen = 0

    while f.lines_left:
        chunk = f.read_lines(chunk_size)
        # Spatial mean & variance, accumulated online.
        for pulse in chunk:
            n_seen += 1
            delta = pulse - running_var
            running_var += delta / n_seen

    # `running_var` is the per-position mean across the whole capture
    # — never held more than `chunk_size × positions_per_line × 2 B`
    # of pulses in RAM at once.
```

Combine with multiprocessing for embarrassingly-parallel passes — open
one `File` per worker (each has its own internal cursor; sharing a
single `File` across processes is not supported).

---

## 6. Dispatch in a generic processing pipeline

When you write a function that should accept any DAS file regardless
of mode, dispatch on `f.mode` and return a uniform `(rows,
positions)` `float32` array:

```python
def physical_amplitude(f: File) -> np.ndarray:
    """Return the per-position amplitude (V) for any mode."""
    match f.mode:
        case Mode.RAW:
            # ADC codes → volts via the range field.
            return f.read_lines(f.lines_left).astype(np.float32) * (f.range / 32768)
        case Mode.IQ:
            return np.abs(f.get_iq_volts())
        case Mode.ARCTAN_MAGNITUDE:
            return f.get_magnitude_volts()
        case Mode.PHASE:
            # Phase has no amplitude — return zeros to keep the
            # caller's shape contract.
            return np.zeros(f.spatial_shape, dtype=np.float32)
        case _:
            raise ValueError(f"Unsupported mode: {f.mode}")

with File("any.dat") as f:
    amp = physical_amplitude(f)
    print(amp.shape, amp.dtype)
```

---

## 7. Read once, dispatch many times

For exploratory analysis you often want to look at multiple channels
of the same buffer. Pass the buffer through to keep the read cursor
intact:

```python
with File("iq.dat") as f:
    buf = f.read_lines(1000)

    # All four lanes from the same bytes — no extra disk read.
    i   = f.get_i(buf)
    q   = f.get_q(buf)
    iq  = f.get_iq(buf)
    iv  = f.get_i_volts(buf)
    iqv = f.get_iq_volts(buf)
```

Each call is a thin Rust loop over `buf`; the file handle is never
touched.

---

## 8. Convert between formats

The SDK can read every Audace format but only writes `.dat`. To
re-export an HDF5 capture as a DAT (e.g. for a downstream tool that
only accepts DAT), use `export_dat`:

```python
from invisensing import File, export_dat, Mode

with File("acquisition.h5") as f:
    if f.mode is Mode.PHASE:
        samples = f.read_all()
        export_dat(
            "acquisition.dat",
            samples,
            sample_rate=f.sample_rate,
            trig_frequency=f.trig_frequency,
            range_v=f.range,
            timestamp=f.timestamp,
            flags=f.flags,           # preserve PHASE / FLOAT / DEMODULATED
        )
```

Round-trip is byte-perfect for the common cases — the 128-byte
header is reconstructed and the samples are written verbatim.

---

## 9. Drop-in replacement for the legacy SDK

Old scripts using `invisensing.File` keep working without changes:

```python
import invisensing.File as iFile

file = iFile.File("acquisition.dat")
print(file.get_line_size(), file.get_sample_rate())

while file.get_lines_left() > 0:
    data = file.get_lines(5)
    # … existing processing logic …
```

The legacy class is a thin wrapper around the modern `File` — same
Rust-backed implementation, same throughput, no migration needed.

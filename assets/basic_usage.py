"""
Invisensing SDK — basic usage example.

Demonstrates the recommended modern API:
- Open any recording via the unified ``File`` facade (DAT, HDF5, TDMS, SEG-Y).
- Inspect what it holds (``mode``, from the file's content type).
- Extract the right channel using ``Mode`` dispatch.
- Stream large files with the iterator / ``read_lines`` loop.
"""

from invisensing import File, Mode


def process(pulse):
    """Replace with your DSP. Here we just print the shape."""
    print(f"  pulse: shape={pulse.shape}, dtype={pulse.dtype}")


def main(path: str) -> None:
    with File(path) as f:
        print(f)  # File('…', mode=…, shape=(N, line_size), …)
        print(f"  content type: {f.content_type_name}, card: {f.card}, "
              f"step: {f.spatial_step_m} m × {f.temporal_step_s} s")

        # Dispatch on what the file holds (if / elif: Python 3.9 has no
        # match statement). Since data format v1 each file holds one
        # quantity (a split pair's other half is f.sibling_path); the
        # paired modes are files written before it.
        mode = f.mode
        if mode is Mode.RAW:
            print("Raw ADC samples — streaming 1000 pulses at a time.")
            while f.lines_left:
                chunk = f.read_lines(1000)
                process(chunk)

        elif mode is Mode.PHASE:
            print("Phase — radians, one sample per spatial position.")
            phase = f.get_phase(n=f.num_lines)
            print(f"  phase: {phase.shape}, dtype={phase.dtype}")

        elif mode is Mode.AMPLITUDE:
            amplitude = f.get_amplitude(n=f.num_lines)
            print(f"  amplitude: {amplitude.shape}, dtype={amplitude.dtype}")
            print(f"  phase file beside it: {f.sibling_path}")

        elif mode is Mode.ARCTAN:
            rad = f.get_arctan_radians(n=f.num_lines)
            print(f"  arctan: {rad.shape} rad, magnitude file: {f.sibling_path}")

        elif mode in (Mode.I, Mode.Q):
            print(f"  {mode.value}: {f.read_lines(1000).shape}, other: {f.sibling_path}")

        elif mode is Mode.IQ:
            print("I/Q pairs — extracting I and Q on each pulse.")
            for pulse in f:
                process(pulse)        # pulse is wire layout [I, Q, I, Q, …]
            # Or, in one shot for a small file:
            #   data = f.read_all()
            #   i = f.get_i(data)
            #   q = f.get_q(data)

        elif mode is Mode.ARCTAN_MAGNITUDE:
            print("Arctan/magnitude pairs — extracting both lanes.")
            buf = f.read_all()
            arctan = f.get_arctan(buf)
            magnitude = f.get_magnitude(buf)
            print(f"  arctan: {arctan.shape}, dtype={arctan.dtype}")
            print(f"  magnitude: {magnitude.shape}, dtype={magnitude.dtype}")

        elif mode is Mode.PHASE_AMPLITUDE:
            buf = f.read_all()
            print(f"  phase: {f.get_phase(buf).shape}, amplitude: {f.get_amplitude(buf).shape}")

        elif mode in (Mode.VARIANCE, Mode.LOG_VARIANCE):
            variance = f.get_variance(n=f.num_lines)
            print(f"  zone variances: {variance.shape}, every {f.spatial_step_m} m")

        else:
            print(f"  {mode.value}: read with f.read_lines()")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "acquisition.h5"
    main(path)

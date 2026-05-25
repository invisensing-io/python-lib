"""
Invisensing SDK — basic usage example.

Demonstrates the recommended modern API:
- Open any FileWriter format via the unified ``File`` facade.
- Inspect the high-level mode (Raw / IQ / ArctanMag / Phase).
- Extract the right channel automatically using ``Mode`` dispatch.
- Stream large files with the iterator / ``read_lines`` loop.
"""

from invisensing import File, Mode


def process(pulse):
    """Replace with your DSP. Here we just print the shape."""
    print(f"  pulse: shape={pulse.shape}, dtype={pulse.dtype}")


def main(path: str) -> None:
    with File(path) as f:
        print(f)  # File('…', mode=…, shape=(N, line_size), …)

        # Dispatch on the high-level mode — works for every PCIe7821
        # demodulation product plus plain Raw.
        match f.mode:
            case Mode.RAW:
                print("Raw ADC samples — streaming 1000 pulses at a time.")
                while f.lines_left:
                    chunk = f.read_lines(1000)
                    process(chunk)

            case Mode.IQ:
                print("IQ demod — extracting I and Q on each pulse.")
                for pulse in f:
                    process(pulse)        # pulse is wire layout [I, Q, I, Q, …]
                # Or, in one shot for a small file:
                #   data = f.read_all()
                #   i = f.get_i(data)
                #   q = f.get_q(data)

            case Mode.ARCTAN_MAGNITUDE:
                print("Arctan/Magnitude — extracting both lanes.")
                buf = f.read_all()
                arctan = f.get_arctan(buf)
                magnitude = f.get_magnitude(buf)
                print(f"  arctan: {arctan.shape}, dtype={arctan.dtype}")
                print(f"  magnitude: {magnitude.shape}, dtype={magnitude.dtype}")

            case Mode.PHASE:
                print("Phase — radians, one sample per spatial position.")
                phase = f.get_phase(n=f.num_lines)
                print(f"  phase: {phase.shape}, dtype={phase.dtype}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "acquisition.dat"
    main(path)

//! Invisensing Python SDK — native Rust core.
//!
//! Bound into Python as `invisensing._core` (see `Cargo.toml`'s `lib.name`).
//! Exposes:
//!
//! - [`DatReader`]: streaming reader for the Audace `.dat` wire format
//!   (128-byte header + interleaved samples). The Python wrapper imports
//!   this class as the foundation for the `File` factory.
//! - [`parse_header`]: standalone helper for the 128-byte header, reused
//!   by the HDF5 / TDMS / SEG-Y wrappers (which round-trip the same
//!   fields as file attributes).
//! - De-interleave kernels [`split_pair_i16_primary`],
//!   [`split_pair_i16_secondary`], [`split_pair_unsigned`],
//!   [`split_pair_to_complex_i16`], [`split_pair_f32_primary`],
//!   [`split_pair_f32_secondary`].
//!
//! All hot kernels operate on contiguous-memory views (`as_slice`) and use
//! `chunks_exact(2)` so LLVM can auto-vectorise the stride-2 reads. The
//! large-buffer `read_lines` path releases the GIL during the actual file
//! I/O so a second Python thread can run while the OS is blocked on disk.
//!
//! ## Safety
//!
//! Three small `unsafe` blocks total — each one bounded, documented and
//! exercised by the test suite:
//! - [`bytes_to_vec`] uses `set_len` + `copy_nonoverlapping` to avoid the
//!   zero-init pass on multi-GB reads.
//! - The transfer is `Vec<u8>` → `Vec<T>` for POD primitives only; the
//!   trait bound + the call sites enforce that statically.
//!
//! The 128-byte Audace wire-format header and the `header_flags`
//! bit values are documented in the public Invisensing SDK reference;
//! this crate ships its own copy of the layout so it stays standalone.

use numpy::{
    PyArray2,
    ndarray::{Array2, ArrayView2},
};
use pyo3::{
    exceptions::{PyIOError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::PathBuf,
};

// ── Wire format constants ───────────────────────────────────────────────────
//
// Mirrors the Audace wire-format header definition so this crate stays
// standalone (no upstream dependency — the python-lib ships independently).
// The header layout is a stable public ABI; keep these constants aligned
// with any future revision of the producer.

const HEADER_SIZE: usize = 128;

/// Byte offsets inside the 128-byte fixed-layout header.
mod offsets {
    pub const TIMESTAMP: usize = 0; // 32 bytes, NUL-padded UTF-8
    pub const LINE_SIZE: usize = 32; // i32 LE
    pub const TRIG_FREQUENCY: usize = 36;
    pub const SAMPLE_SIZE: usize = 40;
    pub const SAMPLE_RATE: usize = 44;
    pub const FLAGS: usize = 48;
    pub const RANGE: usize = 52;
    pub const PULSE_WIDTH: usize = 56;
    pub const NUM_CHANNELS: usize = 60;
}

/// Flag bits — must match the Audace wire-format `header_flags`
/// specification. Producer files (Cardcontrol, FileWriter) use the
/// exact same numeric values.
pub mod flag_bits {
    pub const DEMODULATED: i32 = 0x001;
    pub const FLOAT: i32 = 0x002;
    pub const AC: i32 = 0x004;
    pub const HIZ: i32 = 0x008;
    pub const SCHUBERT: i32 = 0x010;
    pub const SPECTRUM: i32 = 0x020;
    pub const MOZART: i32 = 0x040;
    pub const PHASE: i32 = 0x080;
    pub const INTERLEAVED: i32 = 0x100;
    pub const UNSIGNED: i32 = 0x200;
}

fn read_i32_le(buf: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

// ── Parsed header (Python-exposed) ──────────────────────────────────────────

/// 128-byte Audace `DataHeader` parsed into Python-friendly fields.
///
/// Exposed as `invisensing._core.Header`. The HDF5 / TDMS / SEG-Y wrappers
/// reconstruct one of these from format-native metadata so the channel
/// extractors take a single uniform path.
#[pyclass(get_all, module = "invisensing._core")]
#[derive(Clone, Debug)]
pub struct Header {
    pub timestamp: String,
    pub line_size: i32,
    pub trig_frequency: i32,
    pub sample_size: i32,
    pub sample_rate: i32,
    pub flags: i32,
    pub range: i32,
    pub pulse_width: i32,
    pub num_channels: i32,
}

#[pymethods]
impl Header {
    /// Build a header from individual fields. Used by the format wrappers
    /// (HDF5/TDMS/SEG-Y) to feed the same channel extractors as the DAT
    /// path without going through the binary wire format.
    #[new]
    #[pyo3(signature = (
        line_size,
        trig_frequency,
        sample_size,
        sample_rate,
        flags,
        range = 0,
        pulse_width = 0,
        num_channels = 1,
        timestamp = String::new(),
    ))]
    fn new(
        line_size: i32,
        trig_frequency: i32,
        sample_size: i32,
        sample_rate: i32,
        flags: i32,
        range: i32,
        pulse_width: i32,
        num_channels: i32,
        timestamp: String,
    ) -> Self {
        Self {
            timestamp,
            line_size,
            trig_frequency,
            sample_size,
            sample_rate,
            flags,
            range,
            pulse_width,
            num_channels,
        }
    }

    /// True if the `INTERLEAVED` bit is set — the wire carries 2 samples
    /// per spatial position (I/Q or arctan/√).
    #[getter]
    fn is_interleaved(&self) -> bool {
        self.flags & flag_bits::INTERLEAVED != 0
    }

    /// True if samples are IEEE-754 floats.
    #[getter]
    fn is_float(&self) -> bool {
        self.flags & flag_bits::FLOAT != 0
    }

    /// True if the odd-index lane of an INTERLEAVED pair is unsigned
    /// (PCIe7821 ArctanMagnitude mode: `√(I²+Q²)` as `u16`).
    #[getter]
    fn is_unsigned(&self) -> bool {
        self.flags & flag_bits::UNSIGNED != 0
    }

    #[getter]
    fn is_demodulated(&self) -> bool {
        self.flags & flag_bits::DEMODULATED != 0
    }

    #[getter]
    fn is_phase(&self) -> bool {
        self.flags & flag_bits::PHASE != 0
    }

    #[getter]
    fn is_ac(&self) -> bool {
        self.flags & flag_bits::AC != 0
    }

    #[getter]
    fn is_hiz(&self) -> bool {
        self.flags & flag_bits::HIZ != 0
    }

    /// Physical spatial positions per pulse — `line_size / 2` when
    /// INTERLEAVED, `line_size` otherwise. Mirrors
    /// the Audace `positions_per_line` helper from the producer.
    #[getter]
    fn positions_per_line(&self) -> i32 {
        if self.is_interleaved() {
            self.line_size / 2
        } else {
            self.line_size
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Header(line_size={}, sample_size={}, sample_rate={}, trig_frequency={}, \
             flags=0x{:X}, positions_per_line={}, interleaved={}, float={}, unsigned={})",
            self.line_size,
            self.sample_size,
            self.sample_rate,
            self.trig_frequency,
            self.flags,
            self.positions_per_line(),
            self.is_interleaved(),
            self.is_float(),
            self.is_unsigned(),
        )
    }
}

/// Parse a 128-byte `.dat` header. Useful for callers that already hold
/// the bytes (memory-mapped files, in-memory buffers).
#[pyfunction]
fn parse_header(bytes: &[u8]) -> PyResult<Header> {
    if bytes.len() < HEADER_SIZE {
        return Err(PyValueError::new_err(format!(
            "header must be at least {HEADER_SIZE} bytes (got {})",
            bytes.len()
        )));
    }
    let ts_raw = &bytes[offsets::TIMESTAMP..offsets::TIMESTAMP + 32];
    // Trim trailing NULs / spaces, then lossily decode — the field is
    // user-supplied so we never want to reject a file because of a
    // mangled byte.
    let end = ts_raw
        .iter()
        .rposition(|&b| b != 0 && b != b' ')
        .map(|i| i + 1)
        .unwrap_or(0);
    let timestamp = String::from_utf8_lossy(&ts_raw[..end]).into_owned();

    Ok(Header {
        timestamp,
        line_size: read_i32_le(bytes, offsets::LINE_SIZE),
        trig_frequency: read_i32_le(bytes, offsets::TRIG_FREQUENCY),
        sample_size: read_i32_le(bytes, offsets::SAMPLE_SIZE),
        sample_rate: read_i32_le(bytes, offsets::SAMPLE_RATE),
        flags: read_i32_le(bytes, offsets::FLAGS),
        range: read_i32_le(bytes, offsets::RANGE),
        pulse_width: read_i32_le(bytes, offsets::PULSE_WIDTH),
        num_channels: read_i32_le(bytes, offsets::NUM_CHANNELS),
    })
}

// ── DAT reader (Python-exposed) ─────────────────────────────────────────────

/// Streaming reader for an Audace `.dat` file. Holds an open `BufReader`
/// positioned past the 128-byte header, advanced one chunk per
/// `read_lines` call so the user can iterate large acquisitions without
/// loading the whole file.
///
/// Owns a parsed [`Header`]; the Python wrapper accesses fields via the
/// `header` property.
///
/// **Thread safety**: the reader holds a mutable file cursor, so it is
/// not safe to share between Python threads. The Python facade ensures
/// each `File` has its own `DatReader`. PyO3 holds the GIL across method
/// calls except during the `read_exact` itself (where we release it
/// explicitly via `Python::allow_threads`) — concurrent reads from the
/// same instance are still serialised by the GIL even on a release-build
/// extension, but `read_exact` runs without it so a second Python thread
/// can make progress while the first is blocked on disk I/O.
#[pyclass(module = "invisensing._core")]
pub struct DatReader {
    path: PathBuf,
    reader: BufReader<File>,
    header: Header,
    /// Total pulses in the file (file_size − HEADER) / (line_size × sample_size).
    num_lines: u64,
    /// Pulses remaining after the current read cursor.
    lines_left: u64,
}

#[pymethods]
impl DatReader {
    /// Open a `.dat` file, parse the header, and prepare for streaming
    /// reads. Raises `IOError` if the file can't be opened and
    /// `ValueError` on a malformed header (zero / negative `line_size`,
    /// `sample_size`, or an unsupported `sample_size` value).
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let file = File::open(&path).map_err(|e| {
            PyIOError::new_err(format!("could not open '{}': {e}", path.display()))
        })?;
        let file_size = file
            .metadata()
            .map_err(|e| PyIOError::new_err(format!("stat failed: {e}")))?
            .len();
        let mut reader = BufReader::new(file);
        let mut buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut buf).map_err(|e| {
            PyIOError::new_err(format!("could not read 128-byte header: {e}"))
        })?;
        let header = parse_header(&buf)?;
        validate_header(&header)?;

        let payload = file_size.saturating_sub(HEADER_SIZE as u64);
        let bytes_per_line = (header.line_size as u64) * (header.sample_size as u64);
        // Guarded by validate_header (line_size and sample_size are > 0)
        // but keep the saturating arithmetic anyway so a header field
        // crafted to overflow `i32 × i32` lands at u64::MAX safely.
        let num_lines = if bytes_per_line == 0 {
            0
        } else {
            payload / bytes_per_line
        };

        Ok(Self {
            path,
            reader,
            header,
            num_lines,
            lines_left: num_lines,
        })
    }

    #[getter]
    fn header(&self) -> Header {
        self.header.clone()
    }

    #[getter]
    fn num_lines(&self) -> u64 {
        self.num_lines
    }

    #[getter]
    fn lines_left(&self) -> u64 {
        self.lines_left
    }

    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// Rewind back to the start of the sample stream (just past the
    /// header). Useful to make multiple passes over the same reader
    /// without re-opening the file.
    fn rewind(&mut self) -> PyResult<()> {
        self.reader
            .seek(SeekFrom::Start(HEADER_SIZE as u64))
            .map_err(|e| PyIOError::new_err(format!("rewind failed: {e}")))?;
        self.lines_left = self.num_lines;
        Ok(())
    }

    /// Read up to `n` pulses of samples and return them as a `(rows,
    /// line_size)` numpy array. The dtype is picked from
    /// `(sample_size, FLOAT)` — matches the FileWriter convention so the
    /// numbers come out signed (negative ADC codes stay negative).
    ///
    /// The actual `read_exact` runs with the GIL released so other
    /// Python threads can do work while the OS blocks on disk I/O.
    ///
    /// - Raises `ValueError` if `n <= 0`.
    /// - Raises `IOError` if the file is already exhausted.
    /// - Caps `n` at `lines_left` (does not raise on a short read).
    #[pyo3(signature = (n = 1))]
    fn read_lines<'py>(&mut self, py: Python<'py>, n: i64) -> PyResult<Bound<'py, PyAny>> {
        if n <= 0 {
            return Err(PyValueError::new_err("n must be a positive integer"));
        }
        if self.lines_left == 0 {
            return Err(PyIOError::new_err("end of file"));
        }
        let n = (n as u64).min(self.lines_left);
        let line_size = self.header.line_size as usize;
        let sample_size = self.header.sample_size as usize;
        let total_bytes = (n as usize)
            .checked_mul(line_size)
            .and_then(|s| s.checked_mul(sample_size))
            .ok_or_else(|| PyValueError::new_err(
                "read request overflows usize (file claims more samples than addressable memory)",
            ))?;

        // Allocate uninitialised so we don't pay the zero-init cost on
        // multi-GB reads — the next call (`read_exact`) overwrites every
        // byte before we hand them out, so the uninitialised state is
        // never observable.
        let mut bytes: Vec<u8> = Vec::with_capacity(total_bytes);
        // SAFETY: capacity == total_bytes; we extend the logical length
        // *before* the I/O so `read_exact` writes into initialised
        // memory (read_exact requires `&mut [u8]`). If the read fails,
        // the Vec is dropped before any consumer sees its contents.
        unsafe {
            bytes.set_len(total_bytes);
        }

        // Release the GIL while the OS is potentially blocked on disk.
        // `allow_threads` re-acquires it before returning.
        let read_result = py.allow_threads(|| self.reader.read_exact(&mut bytes));
        read_result.map_err(|e| {
            PyIOError::new_err(format!(
                "short read (wanted {total_bytes} bytes for {n} lines): {e}"
            ))
        })?;
        self.lines_left -= n;

        bytes_to_typed_array(
            py,
            bytes,
            self.header.sample_size,
            self.header.flags,
            n as usize,
            line_size,
        )
    }
}

/// Reject malformed headers up front so downstream maths can assume
/// positive sizes and a known sample format. Centralises the rules so
/// the modern API and the legacy SDK both report the same errors.
fn validate_header(h: &Header) -> PyResult<()> {
    if h.line_size <= 0 {
        return Err(PyValueError::new_err(format!(
            "invalid header: line_size = {} (must be > 0)",
            h.line_size
        )));
    }
    if h.sample_size <= 0 {
        return Err(PyValueError::new_err(format!(
            "invalid header: sample_size = {} (must be > 0)",
            h.sample_size
        )));
    }
    if !matches!(h.sample_size, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err(format!(
            "invalid header: sample_size = {} (must be 1, 2, 4 or 8)",
            h.sample_size
        )));
    }
    // INTERLEAVED line_size must be even — otherwise the pair extractors
    // would always trip on the same input. Catch it once here rather
    // than on every read.
    if h.is_interleaved() && h.line_size % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "invalid header: INTERLEAVED files require an even line_size (got {})",
            h.line_size
        )));
    }
    Ok(())
}

/// Convert a raw byte buffer of `rows × line_size` samples into a typed
/// 2-D numpy array. Dispatches on `(sample_size, FLOAT)`. Always returns
/// the **signed** type when integer (matches FileWriter / Cardcontrol
/// conventions — the UNSIGNED bit is a per-lane reinterpretation handled
/// downstream by `split_pair_unsigned`, not a whole-array type change).
///
/// Takes ownership of `bytes` so the dispatch can move it directly into
/// the typed `Vec<T>` without an intermediate copy.
fn bytes_to_typed_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    sample_size: i32,
    flags: i32,
    rows: usize,
    line_size: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let is_float = flags & flag_bits::FLOAT != 0;
    let shape = (rows, line_size);

    let any: Bound<'py, PyAny> = match (sample_size, is_float) {
        (2, _) => {
            let arr = Array2::<i16>::from_shape_vec(shape, bytes_to_vec::<i16>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (4, true) => {
            let arr = Array2::<f32>::from_shape_vec(shape, bytes_to_vec::<f32>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (4, false) => {
            let arr = Array2::<i32>::from_shape_vec(shape, bytes_to_vec::<i32>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (8, true) => {
            let arr = Array2::<f64>::from_shape_vec(shape, bytes_to_vec::<f64>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (8, false) => {
            let arr = Array2::<i64>::from_shape_vec(shape, bytes_to_vec::<i64>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (1, _) => {
            // u8 path: no reinterpret needed, hand the bytes straight to ndarray.
            let arr = Array2::<u8>::from_shape_vec(shape, bytes)
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (s, f) => {
            return Err(PyValueError::new_err(format!(
                "unsupported sample format: sample_size={s} bytes, float={f}"
            )));
        }
    };
    Ok(any)
}

/// Reinterpret a byte slice as a `Vec<T>` for plain-old-data primitives.
///
/// We always copy (numpy owns the destination memory; the input may not
/// even be `T`-aligned, so a zero-copy view isn't safe). What we do
/// avoid is the *zero-init* pass — `vec![T::default(); n]` would write
/// `n × size_of::<T>` zeros that the `copy_nonoverlapping` then
/// immediately overwrites. On a 4 GB read that's 4 GB of wasted DRAM
/// traffic; with `set_len` + uninit capacity we skip it.
fn bytes_to_vec<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let elem = std::mem::size_of::<T>();
    debug_assert!(elem > 0);
    let n = bytes.len() / elem;
    let mut out: Vec<T> = Vec::with_capacity(n);
    // SAFETY: capacity == n; `copy_nonoverlapping` initialises exactly
    // n × size_of::<T>() bytes (the destination of the copy) before we
    // expose the length. `T` is restricted to plain-old-data primitives
    // by the call sites in `bytes_to_typed_array` (i16/i32/i64/f32/f64).
    // The trailing partial element (if `bytes.len() % elem != 0`) is
    // intentionally dropped.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * elem,
        );
        out.set_len(n);
    }
    out
}

// ── De-interleave kernels ───────────────────────────────────────────────────
//
// All four families take a `(rows, line_size)` array of an INTERLEAVED
// payload and return the requested half. `line_size` must be even.
//
// **Performance notes.**
// 1. The input is read via `as_slice()` so we get a single contiguous
//    `&[T]` rather than going through ndarray's strided iterator. The
//    Python facade always reads C-contiguous buffers from numpy.
// 2. The output is pre-allocated with `Vec::with_capacity(out_len)`
//    plus `set_len` so the inner loop has no capacity-check branch.
// 3. The lane pick is `chunks_exact(2).map(|c| c[lane])` — LLVM
//    auto-vectorises this on SSE/AVX (gather + extract).
// 4. We never invoke the slow strided fallback in production; if a
//    caller hands us a non-contiguous numpy view we make an explicit
//    contiguous copy via `view.to_owned()` so the fast path runs.

/// Pick the even-indexed (primary) lane of an INTERLEAVED `i16` array.
/// PCIe7821 mapping: I in `Iq`, `arctan(Q/I)` in `ArctanMagnitude`.
#[pyfunction]
fn split_pair_i16_primary<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, i16>,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let out = split_lane_into_vec::<i16>(view, 0, rows * out_cols);
    let arr = Array2::from_shape_vec((rows, out_cols), out)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Pick the odd-indexed (secondary) lane of an INTERLEAVED `i16` array.
/// PCIe7821 mapping: Q in `Iq` (use the unsigned variant for magnitude
/// in `ArctanMagnitude`).
#[pyfunction]
fn split_pair_i16_secondary<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, i16>,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let out = split_lane_into_vec::<i16>(view, 1, rows * out_cols);
    let arr = Array2::from_shape_vec((rows, out_cols), out)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Pick the odd-indexed lane of an INTERLEAVED `i16` array and
/// **reinterpret the bits as `u16`**. PCIe7821 mapping:
/// `√(I²+Q²)` magnitude in `ArctanMagnitude` (vendor sends magnitude as
/// unsigned 16-bit despite the buffer being typed `i16` end-to-end).
#[pyfunction]
fn split_pair_unsigned<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, i16>,
) -> PyResult<Bound<'py, PyArray2<u16>>> {
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let total = rows * out_cols;

    // Run the same fast path as the i16 kernels, then bit-cast the
    // result to u16. `i16` and `u16` have identical layout so the
    // transmute is sound.
    let i16_vec = split_lane_into_vec::<i16>(view, 1, total);
    // SAFETY: i16 and u16 have the same size and alignment; the bit
    // pattern is preserved (this is exactly the wire reinterpretation
    // the vendor doc calls for).
    let u16_vec: Vec<u16> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(i16_vec);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut u16, v.len(), v.capacity())
    };

    let arr = Array2::from_shape_vec((rows, out_cols), u16_vec)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Build a `complex64` array from an INTERLEAVED `i16` I,Q payload:
/// `result[r, j] = I[r, j] + j·Q[r, j]`.
#[pyfunction]
fn split_pair_to_complex_i16<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, i16>,
) -> PyResult<Bound<'py, PyArray2<numpy::Complex32>>> {
    use numpy::Complex32;
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let total = rows * out_cols;
    let mut out: Vec<Complex32> = Vec::with_capacity(total);
    // SAFETY: same uninit + fill pattern as `split_lane_into_vec`; we
    // overwrite every element below before exposing the length.
    unsafe {
        out.set_len(total);
    }

    // Hot loop in 64-bit chunks (one I,Q pair = 32 bits = 4 bytes;
    // numpy gives us a contiguous slice for typical inputs).
    if let Some(input) = view.as_slice() {
        for (pair, dst) in input.chunks_exact(2).zip(out.iter_mut()) {
            *dst = Complex32::new(pair[0] as f32, pair[1] as f32);
        }
    } else {
        // Strided fallback — copy through a contiguous buffer once.
        let owned = view.to_owned();
        let slice = owned.as_slice().expect("to_owned is always contiguous");
        for (pair, dst) in slice.chunks_exact(2).zip(out.iter_mut()) {
            *dst = Complex32::new(pair[0] as f32, pair[1] as f32);
        }
    }

    let arr = Array2::from_shape_vec((rows, out_cols), out)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// f32 even-lane picker — provided for parity with the i16 kernels.
#[pyfunction]
fn split_pair_f32_primary<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let out = split_lane_into_vec::<f32>(view, 0, rows * out_cols);
    let arr = Array2::from_shape_vec((rows, out_cols), out)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pyfunction]
fn split_pair_f32_secondary<'py>(
    py: Python<'py>,
    interleaved: numpy::PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = interleaved.as_array();
    let (rows, cols) = view.dim();
    check_pair_shape(cols)?;
    let out_cols = cols / 2;
    let out = split_lane_into_vec::<f32>(view, 1, rows * out_cols);
    let arr = Array2::from_shape_vec((rows, out_cols), out)
        .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Centralised shape check so every kernel returns the same error
/// message — easier to grep for in user-facing reports.
fn check_pair_shape(cols: usize) -> PyResult<()> {
    if cols % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "interleaved line_size must be even (got {cols})"
        )));
    }
    Ok(())
}

/// Generic lane picker. Reads from `view` and writes the requested
/// lane into a freshly-allocated, capacity-pre-set `Vec<T>`. The
/// `chunks_exact(2)` + indexing is what LLVM auto-vectorises into
/// SSE / AVX gather-extract instructions on x86_64.
///
/// Falls back to a one-time `to_owned()` copy if the input ArrayView
/// isn't contiguous — this only happens when the Python caller passes
/// a numpy slice / non-C-contiguous view; the production read path
/// always feeds C-contiguous arrays from `read_lines`.
fn split_lane_into_vec<T: Copy + numpy::Element>(
    view: ArrayView2<'_, T>,
    lane: usize,
    total: usize,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(total);
    // SAFETY: capacity == total; the loop below writes to every index
    // 0..total before we expose the slice via `Array2::from_shape_vec`.
    unsafe {
        out.set_len(total);
    }

    if let Some(input) = view.as_slice() {
        for (pair, dst) in input.chunks_exact(2).zip(out.iter_mut()) {
            *dst = pair[lane];
        }
    } else {
        // Strided / non-standard layout: pay the copy once.
        let owned = view.to_owned();
        let slice = owned.as_slice().expect("to_owned is always contiguous");
        for (pair, dst) in slice.chunks_exact(2).zip(out.iter_mut()) {
            *dst = pair[lane];
        }
    }
    out
}

// ── Header serializer (write path) ──────────────────────────────────────────

/// Build a 128-byte header buffer from a [`Header`]. Used by the Python
/// `export` helper so the `.dat` round-trip matches the C side byte for
/// byte. Returns a `bytes` object.
#[pyfunction]
fn build_header_bytes<'py>(py: Python<'py>, header: &Header) -> PyResult<Bound<'py, PyBytes>> {
    let mut buf = [0u8; HEADER_SIZE];

    // timestamp: 32 bytes, NUL-padded UTF-8.
    let ts = header.timestamp.as_bytes();
    let n = ts.len().min(32);
    buf[..n].copy_from_slice(&ts[..n]);

    let write = |buf: &mut [u8; HEADER_SIZE], offset: usize, v: i32| {
        buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
    };
    write(&mut buf, offsets::LINE_SIZE, header.line_size);
    write(&mut buf, offsets::TRIG_FREQUENCY, header.trig_frequency);
    write(&mut buf, offsets::SAMPLE_SIZE, header.sample_size);
    write(&mut buf, offsets::SAMPLE_RATE, header.sample_rate);
    write(&mut buf, offsets::FLAGS, header.flags);
    write(&mut buf, offsets::RANGE, header.range);
    write(&mut buf, offsets::PULSE_WIDTH, header.pulse_width);
    write(&mut buf, offsets::NUM_CHANNELS, header.num_channels);

    Ok(PyBytes::new(py, &buf))
}

// ── Module init ─────────────────────────────────────────────────────────────

/// Module entry point — Python imports this as `invisensing._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Header>()?;
    m.add_class::<DatReader>()?;
    m.add_function(wrap_pyfunction!(parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(build_header_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_i16_primary, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_i16_secondary, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_unsigned, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_to_complex_i16, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_f32_primary, m)?)?;
    m.add_function(wrap_pyfunction!(split_pair_f32_secondary, m)?)?;

    // Expose the header-flag bits as module-level constants (i.e.
    // `invisensing._core.FLAG_INTERLEAVED`) so the Python wrapper and any
    // downstream consumer can bit-test without reaching into the audace
    // crate.
    m.add("FLAG_DEMODULATED", flag_bits::DEMODULATED)?;
    m.add("FLAG_FLOAT", flag_bits::FLOAT)?;
    m.add("FLAG_AC", flag_bits::AC)?;
    m.add("FLAG_HIZ", flag_bits::HIZ)?;
    m.add("FLAG_SCHUBERT", flag_bits::SCHUBERT)?;
    m.add("FLAG_SPECTRUM", flag_bits::SPECTRUM)?;
    m.add("FLAG_MOZART", flag_bits::MOZART)?;
    m.add("FLAG_PHASE", flag_bits::PHASE)?;
    m.add("FLAG_INTERLEAVED", flag_bits::INTERLEAVED)?;
    m.add("FLAG_UNSIGNED", flag_bits::UNSIGNED)?;
    m.add("HEADER_SIZE", HEADER_SIZE)?;

    Ok(())
}

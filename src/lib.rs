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
    pub const CONTENT_TYPE: usize = 64; // u8 (carved out of the old reserved region)
    /// u32 LE metres of fibre, carved out of the reserved region past the
    /// 16-byte card tag (which occupies 65..81). Zero on captures written
    /// before the field existed — and zero is also the right value for
    /// "no delay".
    pub const TRIGGER_DELAY: usize = 84;
    /// u32 LE, the clock the card's trigger period is counted in (Hz). With
    /// [`TRIG_PERIOD_TICKS`], the cadence the card really fires at:
    /// `trig_clock_hz / trig_period_ticks`. Both zero when unknown (captures
    /// written before the fields existed, external trigger).
    pub const TRIG_CLOCK_HZ: usize = 88;
    /// u32 LE, the trigger period in ticks of [`TRIG_CLOCK_HZ`].
    pub const TRIG_PERIOD_TICKS: usize = 92;
}

const NS_PER_S: u128 = 1_000_000_000;

/// The card's trigger rational `(clock_hz, period_ticks)` when it can
/// describe a stream whose integer `trig_frequency` is the given one: both
/// terms non-zero, and `clock / ticks` within 1 % (or within 1 Hz) of
/// `trig_frequency`. Mirrors the producer's guard
/// (`audace::trig_timing::TriggerTiming::consistent_with`) term for term, so
/// a file is timed identically by the acquisition and by this SDK.
///
/// A legitimate card rational is off its requested integer by less than one
/// tick in a period (< 0.08 % over 1 kHz–200 kHz). A rational that is not
/// within the guard was left behind in the reserved bytes of a header whose
/// `trig_frequency` was rewritten, or written by a producer that used those
/// bytes for something else: it describes another stream and is ignored.
fn consistent_rational(clock_hz: u32, period_ticks: u32, trig_frequency: i32) -> Option<(u32, u32)> {
    if clock_hz == 0 || period_ticks == 0 || trig_frequency <= 0 {
        return None;
    }
    // Compare `clock / ticks` with `f` in clock units: `clock` vs `f · ticks`.
    let ticks = period_ticks as u128;
    let nominal = trig_frequency as u128 * ticks;
    let diff = (clock_hz as u128).abs_diff(nominal);
    if diff * 100 < nominal || diff < ticks {
        Some((clock_hz, period_ticks))
    } else {
        None
    }
}

/// `⌊lines · ticks · 10⁹ / clock⌋` ns in exact integer arithmetic (u128: no
/// overflow for any `u64` count), saturated to `i64::MAX`. `clock > 0`.
fn rational_lines_to_ns(clock_hz: u32, period_ticks: u32, lines: u64) -> i64 {
    let ns = lines as u128 * period_ticks as u128 * NS_PER_S / clock_hz as u128;
    ns.min(i64::MAX as u128) as i64
}

/// Lowercase tag for a `content_type` byte (mirrors `audace::types::ContentType`).
pub fn content_type_tag(v: u8) -> &'static str {
    match v {
        1 => "raw",
        2 => "hw_phase",
        3 => "hw_arctan",
        4 => "hw_arctan_dui",
        5 => "hw_amplitude",
        6 => "interleaved_hw_iq",
        7 => "interleaved_hw_arctan_amplitude",
        8 => "interleaved_hw_phase_amplitude",
        9 => "sw_mozart",
        10 => "sw_phase",
        11 => "sw_amplitude",
        _ => "undefined",
    }
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

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
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
    /// `content_type` descriptor byte (DataHeader byte 64). `0` = Undefined
    /// (legacy capture). Use `content_type_name` for the lowercase tag.
    pub content_type: u8,
    /// Precise acquisition time in i64 nanoseconds (DataHeader timestamp bytes
    /// [0..8]). `0` when the file uses the legacy ASCII timestamp string.
    pub acquisition_ns: i64,
    /// Trigger pre-roll in **metres of fibre** (DataHeader bytes 84..88).
    ///
    /// Metres rather than a sample count: a count is only meaningful against
    /// the rate it was counted at, and that is not always this header's
    /// `sample_rate` (a decimating tap counts its delay in ADC clocks while
    /// publishing the decimated rate). The producer converts. `0` means no
    /// pre-roll, and is also what captures written before the field hold.
    pub trigger_delay: u32,
    /// Clock the card counts its trigger period in, Hz (DataHeader bytes
    /// 88..92; HDF5 attribute `trig_clock_hz`). Raw value as stored: `0`
    /// when unknown. Use `trig_timing` for the value the timing helpers act
    /// on (it drops a rational that does not describe `trig_frequency`).
    pub trig_clock_hz: u32,
    /// Trigger period in ticks of `trig_clock_hz` (DataHeader bytes 92..96;
    /// HDF5 attribute `trig_period_ticks`). Raw value as stored: `0` when
    /// unknown.
    pub trig_period_ticks: u32,
}

impl Header {
    /// The consistent card rational, see [`consistent_rational`].
    fn rational(&self) -> Option<(u32, u32)> {
        consistent_rational(self.trig_clock_hz, self.trig_period_ticks, self.trig_frequency)
    }

    /// The cadence lines are timed with: the card's rational when known and
    /// consistent, else the nominal `trig_frequency / 1`; `None` without any
    /// cadence (`trig_frequency <= 0` and no rational).
    fn line_rational(&self) -> Option<(u32, u32)> {
        self.rational().or_else(|| {
            u32::try_from(self.trig_frequency)
                .ok()
                .filter(|&f| f > 0)
                .map(|f| (f, 1))
        })
    }
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
        content_type = 0,
        acquisition_ns = 0,
        trigger_delay = 0,
        trig_clock_hz = 0,
        trig_period_ticks = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
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
        content_type: u8,
        acquisition_ns: i64,
        trigger_delay: u32,
        trig_clock_hz: u32,
        trig_period_ticks: u32,
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
            content_type,
            acquisition_ns,
            trigger_delay,
            trig_clock_hz,
            trig_period_ticks,
        }
    }

    /// The card's trigger rational `(trig_clock_hz, trig_period_ticks)`, or
    /// `None` when the header does not carry one (both fields `0`: a capture
    /// written before the fields existed, an external trigger) or carries
    /// one that does not describe this header — `clock / ticks` more than
    /// 1 % (and more than 1 Hz) away from `trig_frequency`, a rational left
    /// behind by a header whose frequency was rewritten. Such a rational is
    /// ignored by every timing helper.
    #[getter]
    fn trig_timing(&self) -> Option<(u32, u32)> {
        self.rational()
    }

    /// The trigger rate the card really fires at, Hz:
    /// `trig_clock_hz / trig_period_ticks` when `trig_timing` is known, else
    /// `float(trig_frequency)` (the requested rate, which is then the only
    /// one known).
    #[getter]
    fn trig_frequency_real_hz(&self) -> f64 {
        match self.rational() {
            Some((clock, ticks)) => clock as f64 / ticks as f64,
            None => self.trig_frequency as f64,
        }
    }

    /// `True` when the card fires exactly at the requested `trig_frequency`
    /// (`trig_clock_hz == trig_frequency × trig_period_ticks`), and when the
    /// real rate is unknown.
    #[getter]
    fn trig_frequency_is_exact(&self) -> bool {
        match self.rational() {
            Some((clock, ticks)) => {
                clock as u64 == self.trig_frequency as u64 * ticks as u64
            }
            None => true,
        }
    }

    /// Time of `lines` trigger periods, ns: `⌊lines · ticks · 10⁹ / clock⌋`
    /// on the card's rational, or `⌊lines · 10⁹ / trig_frequency⌋` without
    /// it — the same integer at an exact frequency. Saturates at `i64::MAX`;
    /// `0` when the header has no cadence at all (`trig_frequency <= 0`).
    fn lines_to_ns(&self, lines: u64) -> i64 {
        self.line_rational()
            .map_or(0, |(clock, ticks)| rational_lines_to_ns(clock, ticks, lines))
    }

    /// Acquisition time of trigger line `index` (0-based), ns since the Unix
    /// epoch: `acquisition_ns + lines_to_ns(index)`. Relative to the start
    /// of the file when the header carries no anchor (`acquisition_ns == 0`).
    fn line_time_ns(&self, index: u64) -> i64 {
        self.acquisition_ns.saturating_add(self.lines_to_ns(index))
    }

    /// Lowercase `content_type` tag (e.g. `"interleaved_hw_phase_amplitude"`,
    /// `"undefined"` for legacy captures).
    #[getter]
    fn content_type_name(&self) -> &'static str {
        content_type_tag(self.content_type)
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

    /// Which demodulation pipeline produced these samples, or `None` when the
    /// question does not apply.
    ///
    /// `True` = Mozart (GPU phase pipeline), `False` = Schubert (the legacy
    /// CPU path). **`None` on raw data**: the distinction is carried by the
    /// absence of the `MOZART` bit, not by a bit of its own, so on a raw ADC
    /// capture — where no demodulation ran at all — "not Mozart" would read as
    /// "Schubert" and name a pipeline that never executed. Callers must treat
    /// `None` as "not applicable", not as a default.
    #[getter]
    fn is_mozart(&self) -> Option<bool> {
        if self.flags & flag_bits::DEMODULATED == 0 {
            return None;
        }
        Some(self.flags & flag_bits::MOZART != 0)
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
        // The trigger rational only when the header carries one, so the
        // representation of every older header is unchanged.
        let timing = if self.trig_clock_hz != 0 || self.trig_period_ticks != 0 {
            format!(
                ", trig_clock_hz={}, trig_period_ticks={}",
                self.trig_clock_hz, self.trig_period_ticks
            )
        } else {
            String::new()
        };
        format!(
            "Header(line_size={}, sample_size={}, sample_rate={}, trig_frequency={}{}, \
             flags=0x{:X}, positions_per_line={}, interleaved={}, float={}, unsigned={})",
            self.line_size,
            self.sample_size,
            self.sample_rate,
            self.trig_frequency,
            timing,
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
    // Timestamp: the current format is an i64-ns value in bytes [0..8] with
    // [8..32] left zero; legacy captures carried an ASCII string that always
    // spilled past byte 8. Disambiguate on the [8..32] region. For the i64
    // form we surface the raw nanoseconds in `acquisition_ns` and a decimal-ns
    // `timestamp` string (no chrono dependency here); legacy files keep their
    // free-form ASCII timestamp and `acquisition_ns = 0`.
    // Plausible window for an i64-ns timestamp (≈ 2014..2103) — guards against a
    // very short legacy ASCII string whose [8..32] also happens to be zero.
    const ACQ_NS_MIN: i64 = 1_400_000_000_000_000_000;
    const ACQ_NS_MAX: i64 = 4_200_000_000_000_000_000;
    let i64_ns = if ts_raw[8..32].iter().all(|&b| b == 0) {
        let ns = i64::from_le_bytes(ts_raw[0..8].try_into().unwrap());
        if (ACQ_NS_MIN..=ACQ_NS_MAX).contains(&ns) { Some(ns) } else { None }
    } else {
        None
    };
    let (timestamp, acquisition_ns) =
        if let Some(ns) = i64_ns {
            (ns.to_string(), ns)
        } else {
            // Trim trailing NULs / spaces, then lossily decode — the field is
            // user-supplied so we never reject a file over a mangled byte.
            let end = ts_raw
                .iter()
                .rposition(|&b| b != 0 && b != b' ')
                .map(|i| i + 1)
                .unwrap_or(0);
            (String::from_utf8_lossy(&ts_raw[..end]).into_owned(), 0i64)
        };

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
        content_type: bytes[offsets::CONTENT_TYPE],
        acquisition_ns,
        trigger_delay: read_u32_le(bytes, offsets::TRIGGER_DELAY),
        trig_clock_hz: read_u32_le(bytes, offsets::TRIG_CLOCK_HZ),
        trig_period_ticks: read_u32_le(bytes, offsets::TRIG_PERIOD_TICKS),
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
///
/// **Lifetime**: [`DatReader::close`] drops the `BufReader` — and with it
/// the OS descriptor — at a point the caller chooses, instead of leaving
/// it to whenever Python reclaims the object. It is idempotent, and any
/// later read raises `ValueError` rather than reading from a handle that
/// is no longer there. The parsed header and `num_lines` are kept, so a
/// closed reader can still describe the file it read.
#[pyclass(module = "invisensing._core")]
pub struct DatReader {
    path: PathBuf,
    /// `None` once [`DatReader::close`] has run.
    reader: Option<BufReader<File>>,
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
            reader: Some(reader),
            header,
            num_lines,
            lines_left: num_lines,
        })
    }

    /// Release the OS file descriptor. Idempotent — closing twice is a
    /// no-op, matching `File.close()` on the Python side. Every later
    /// read (`read_lines`, `rewind`, `seek_lines`) raises `ValueError`;
    /// `header`, `num_lines` and `path` keep answering.
    fn close(&mut self) {
        self.reader = None;
    }

    /// `True` once `close()` has run.
    #[getter]
    fn is_closed(&self) -> bool {
        self.reader.is_none()
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
        self.reader_mut()?
            .seek(SeekFrom::Start(HEADER_SIZE as u64))
            .map_err(|e| PyIOError::new_err(format!("rewind failed: {e}")))?;
        self.lines_left = self.num_lines;
        Ok(())
    }

    /// Move the read cursor to pulse `n` (0-based) without reading the
    /// skipped samples — an O(1) `seek` past the header plus `n` lines.
    /// `n` is clamped to `num_lines`; `lines_left` is updated to match.
    /// Subsequent `read_lines` continues from `n`.
    #[pyo3(signature = (n))]
    fn seek_lines(&mut self, n: u64) -> PyResult<()> {
        let n = n.min(self.num_lines);
        let bytes_per_line = (self.header.line_size as u64) * (self.header.sample_size as u64);
        let offset = HEADER_SIZE as u64 + n.saturating_mul(bytes_per_line);
        self.reader_mut()?
            .seek(SeekFrom::Start(offset))
            .map_err(|e| PyIOError::new_err(format!("seek failed: {e}")))?;
        self.lines_left = self.num_lines - n;
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
        if self.reader.is_none() {
            return Err(PyValueError::new_err("read on a closed file"));
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
        let reader = self.reader_mut()?;
        let read_result = py.allow_threads(|| reader.read_exact(&mut bytes));
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

impl DatReader {
    /// The live reader, or the error a caller gets for touching a file
    /// it has already closed. Kept out of `#[pymethods]` — it returns a
    /// Rust reference, which is not a Python signature.
    fn reader_mut(&mut self) -> PyResult<&mut BufReader<File>> {
        self.reader
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("read on a closed file"))
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

    // Timestamp: symmetric with `parse_header`. When the Header carries a
    // precise i64-ns acquisition time, write it as little-endian i64 in [0..8]
    // (leaving [8..32] zero, the new format). Otherwise fall back to the legacy
    // NUL-padded UTF-8 string. Writing the decimal-ns string here would round-
    // trip back as a *legacy* timestamp (acquisition_ns lost) — so don't.
    if header.acquisition_ns != 0 {
        buf[0..8].copy_from_slice(&header.acquisition_ns.to_le_bytes());
    } else {
        let ts = header.timestamp.as_bytes();
        let n = ts.len().min(32);
        buf[..n].copy_from_slice(&ts[..n]);
    }

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
    // content_type descriptor (byte 64).
    buf[offsets::CONTENT_TYPE] = header.content_type;
    // Trigger pre-roll (bytes 84..88), u32 LE.
    buf[offsets::TRIGGER_DELAY..offsets::TRIGGER_DELAY + 4]
        .copy_from_slice(&header.trigger_delay.to_le_bytes());
    // Trigger rational (bytes 88..96), u32 LE each, written as stored: zero
    // on every header that does not carry one.
    buf[offsets::TRIG_CLOCK_HZ..offsets::TRIG_CLOCK_HZ + 4]
        .copy_from_slice(&header.trig_clock_hz.to_le_bytes());
    buf[offsets::TRIG_PERIOD_TICKS..offsets::TRIG_PERIOD_TICKS + 4]
        .copy_from_slice(&header.trig_period_ticks.to_le_bytes());

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

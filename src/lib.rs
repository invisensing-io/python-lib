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
//! - [`time_axis_ns`]: the time of a run of lines on a file's time axis
//!   (the card's time map, HDF5 `time_map_*` attributes) — exact integers,
//!   the ones the acquisition stamped.
//! - The `content_type` registry of the Audace data format v1 and the card
//!   codes ([`content`]), exported as `CONTENT_TYPES`, `CARD_CODES` and
//!   [`card_code`]: a file's content type is what fixes the sample type of a
//!   v1 file (its FLOAT / UNSIGNED / INTERLEAVED flags are written 0);
//!   content type 0 (files written before the byte existed) keeps the
//!   flag-based reading.
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
//! One small `unsafe` block — bounded, documented and exercised by the test
//! suite: [`bytes_to_vec`] copies a `Vec<u8>` into a `Vec<T>` for POD
//! primitives only (`copy_nonoverlapping`, then `set_len` once every
//! element is initialised); the call sites in [`bytes_to_typed_array`]
//! restrict `T` to `u16`/`i16`/`u32`/`i32`/`u64`/`i64`/`f32`/`f64`. Every
//! other buffer is built initialised (`vec![0; n]`, `collect()`).
//!
//! The 128-byte Audace wire-format header and the `header_flags`
//! bit values are documented in the public Invisensing SDK reference;
//! this crate ships its own copy of the layout so it stays standalone.

use numpy::{
    PyArray1, PyArray2,
    ndarray::{Array2, ArrayView2},
};
use pyo3::{
    exceptions::{PyIOError, PyMemoryError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::PathBuf,
};

mod content;

use content::SampleType;

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
    /// The code of the card that produced the data (`p18`, `p21`, `s125`,
    /// `s500`), ASCII, NUL-padded, 16 bytes (65..81): stamped on every run
    /// since data format v1. Before v1 the slot held a variant tag on every
    /// run from 2026-07-15 to 2026-10-13, then on two-card runs only, all
    /// zero otherwise (see [`crate::content::card_code`]).
    pub const CARD_TAG: usize = 65;
    pub const CARD_TAG_LEN: usize = 16;
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
    /// i32 LE: line `m` of a wire stream is line `m + line_offset` of the
    /// card's time map (`−D` at the output of Demodulation, 0 elsewhere and
    /// in every file). Zero on captures written before the field existed.
    pub const LINE_OFFSET: usize = 96;
    /// u32 LE bit set, see [`time_flag_bits`]. Zero on older captures.
    pub const TIME_FLAGS: usize = 100;
    /// Data format v1 (zero = absent on every older capture): u32 LE, the
    /// file's duration in whole seconds, rounded down, written at close.
    pub const DURATION_S: usize = 104;
    /// u8: 1 for a header of data format v1, 0 before.
    pub const FORMAT_VERSION: usize = 108;
    /// f32 LE: metres of fibre between two adjacent positions (columns).
    pub const SPATIAL_STEP_M: usize = 112;
    /// f32 LE: seconds between two lines.
    pub const TEMPORAL_STEP_S: usize = 116;
}

/// The data format version this SDK reads in full.
const FORMAT_VERSION: u8 = 1;

/// The header layout, as `(name, offset, size, type)`: what
/// `tests/test_content_type.py` compares with the producer's registry.
const HEADER_LAYOUT: [(&str, usize, usize, &str); 20] = [
    ("timestamp", offsets::TIMESTAMP, 32, "bytes"),
    ("line_size", offsets::LINE_SIZE, 4, "i32"),
    ("trig_frequency", offsets::TRIG_FREQUENCY, 4, "i32"),
    ("sample_size", offsets::SAMPLE_SIZE, 4, "i32"),
    ("sample_rate", offsets::SAMPLE_RATE, 4, "i32"),
    ("flags", offsets::FLAGS, 4, "i32"),
    ("range", offsets::RANGE, 4, "i32"),
    ("pulse_width", offsets::PULSE_WIDTH, 4, "i32"),
    ("num_channels", offsets::NUM_CHANNELS, 4, "i32"),
    ("content_type", offsets::CONTENT_TYPE, 1, "u8"),
    ("card", offsets::CARD_TAG, offsets::CARD_TAG_LEN, "ascii"),
    ("trigger_delay", offsets::TRIGGER_DELAY, 4, "u32"),
    ("trig_clock_hz", offsets::TRIG_CLOCK_HZ, 4, "u32"),
    ("trig_period_ticks", offsets::TRIG_PERIOD_TICKS, 4, "u32"),
    ("line_offset", offsets::LINE_OFFSET, 4, "i32"),
    ("time_flags", offsets::TIME_FLAGS, 4, "u32"),
    ("duration_s", offsets::DURATION_S, 4, "u32"),
    ("format_version", offsets::FORMAT_VERSION, 1, "u8"),
    ("spatial_step_m", offsets::SPATIAL_STEP_M, 4, "f32"),
    ("temporal_step_s", offsets::TEMPORAL_STEP_S, 4, "f32"),
];

/// Bits of the header's `time_flags` (bytes 100..104) — must match the
/// Audace wire-format `header::time_flags`.
pub mod time_flag_bits {
    /// The acquisition published a time map for this run (wire streams).
    pub const TIME_MAP_V1: u32 = 1 << 0;
    /// Files: the header's start time is the CORRECTED time of the file's
    /// first line (the card's time map), not the count `anchor + lines`.
    pub const TIME_CORRECTED: u32 = 1 << 1;
}

const NS_PER_S: u128 = 1_000_000_000;

/// The highest trigger frequency an Audace header carries, Hz (the
/// acquisition's `audace::trig_timing::MAX_TRIG_FREQUENCY_HZ`).
const MAX_TRIG_FREQUENCY_HZ: u128 = 10_000_000;

/// Below this many ticks per period at [`MAX_TRIG_FREQUENCY_HZ`] (a clock
/// under 30 MHz) a clock is no card's trigger clock: within 1 Hz only.
const MIN_TICKS_AT_MAX_FREQUENCY: u128 = 3;

/// The card's trigger rational `(clock_hz, period_ticks)` when it can
/// describe a stream whose integer `trig_frequency` (`f`) is the given one:
/// both terms non-zero, and `|clock / ticks − f| < f / q + 1 Hz`, where
/// `q = ⌊clock / 10 MHz⌋` is the period, in ticks of this clock, of the
/// fastest trigger a header carries (25 on 250 MHz, 12 on 125 MHz); within
/// 1 Hz only when `q < 3` (a reduced fraction, not a card's clock). Mirrors
/// the producer's guard (`audace::trig_timing::TriggerTiming::consistent_with`)
/// term for term, so a file is timed identically by the acquisition and by
/// this SDK; both check the shared table `tests/assets/trig_guard_v1.txt`.
///
/// A card fires on `ticks = ⌊clock / f⌋` and is off its requested integer by
/// less than one tick in a period: less than `1/ticks ≤ 1/q` (+40 000 ppm at
/// most up to 10 MHz on 250 MHz). The `1 Hz` covers a rate rounded to an
/// integer (archive rows). A rational beyond that was left behind in the
/// reserved bytes of a header whose `trig_frequency` was rewritten (a
/// decimation, a row rate: ×2 is already 50 % off), or written by a producer
/// that used those bytes for something else: it describes another stream
/// and is ignored.
fn consistent_rational(clock_hz: u32, period_ticks: u32, trig_frequency: i32) -> Option<(u32, u32)> {
    if clock_hz == 0 || period_ticks == 0 || trig_frequency <= 0 {
        return None;
    }
    // Compare `clock / ticks` with `f` in clock units: `clock` vs `f · ticks`.
    let ticks = period_ticks as u128;
    let nominal = trig_frequency as u128 * ticks;
    let diff = (clock_hz as u128).abs_diff(nominal);
    let q = clock_hz as u128 / MAX_TRIG_FREQUENCY_HZ;
    let consistent = if q < MIN_TICKS_AT_MAX_FREQUENCY {
        // Within 1 Hz: |clock/ticks − f| < 1.
        diff < ticks
    } else {
        // |clock/ticks − f| < f/q + 1, times q·ticks.
        diff * q < nominal + q * ticks
    };
    consistent.then_some((clock_hz, period_ticks))
}

/// `⌊lines · ticks · 10⁹ / clock⌋` ns in exact integer arithmetic (u128: no
/// overflow for any `u64` count), saturated to `i64::MAX`. `clock > 0`.
fn rational_lines_to_ns(clock_hz: u32, period_ticks: u32, lines: u64) -> i64 {
    let ns = lines as u128 * period_ticks as u128 * NS_PER_S / clock_hz as u128;
    ns.min(i64::MAX as u128) as i64
}

/// Lowercase tag of a `content_type` byte this SDK knows — the registry's
/// (see [`content`]). `None` for a value the SDK does not know (a newer
/// producer).
pub fn known_content_type_tag(v: u8) -> Option<&'static str> {
    content::info(v).map(|i| i.tag)
}

/// Lowercase tag for a `content_type` byte: [`known_content_type_tag`], or
/// `unknown(<n>)` for a value this SDK does not know. Never `"undefined"`
/// for such a value: `undefined` (0) says the capture predates the byte,
/// which a value written by a newer producer does not.
pub fn content_type_tag(v: u8) -> String {
    match known_content_type_tag(v) {
        Some(tag) => tag.to_string(),
        None => format!("unknown({v})"),
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
    /// Reserved: never set by any producer, written 0 in v1 files.
    pub const RESERVED: i32 = 0x020;
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
///
/// The fields are the header as stored. The `is_*` predicates, the
/// positions and the sample encoding follow the **content type** when it
/// fixes them (data format v1: a v1 file writes its FLOAT, UNSIGNED,
/// INTERLEAVED, DEMODULATED and PHASE flags as 0); for content type 0 (a
/// capture written before the byte existed), an unassigned value or a
/// `sample_size` the content type contradicts, they follow the flags
/// exactly as before.
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
    /// `content_type` descriptor byte (DataHeader byte 64): what the samples
    /// are. `0` = Undefined (a capture written before the byte existed).
    /// Use `content_type_name` for the lowercase tag.
    pub content_type: u8,
    /// The code of the card that produced the data — `"p18"`, `"p21"`,
    /// `"s125"` or `"s500"` — or `None` when the file does not say (an older
    /// single-card capture, a file converted without it). Stamped on every
    /// run since data format v1 (DAT bytes 65..81, HDF5 / TDMS `card`, SEG-Y
    /// `CARD:` line); the tag a run wrote there before v1 (every run from
    /// 2026-07-15 to 2026-10-13, two-card runs only afterwards) reads as its
    /// code.
    pub card: Option<String>,
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
    /// DataHeader bytes 96..100: line `m` of a wire stream is line
    /// `m + line_offset` of the card's time map. `0` in files and on every
    /// older capture.
    pub line_offset: i32,
    /// DataHeader bytes 100..104, bits `TIME_FLAG_MAP_V1` /
    /// `TIME_FLAG_CORRECTED`. For an HDF5, `TIME_FLAG_CORRECTED` when its
    /// `time_correction` attribute is `"cardclock-v1"`. `0` on older files,
    /// and always `0` for TDMS and SEG-Y, which carry no time flags.
    pub time_flags: u32,
    /// Data format v1 — duration of the file in whole seconds, rounded down,
    /// written when the file was closed (DAT bytes 104..108, attribute
    /// `duration_s`). `0` = unknown: a file written before v1, a file under
    /// one second, a recording cut by a crash.
    pub duration_s: u32,
    /// Data format version (DAT byte 108, attribute `format_version`): `1`
    /// for a file of data format v1, `0` before.
    pub format_version: u8,
    /// Metres of fibre between two adjacent positions (columns), as the file
    /// stores it (DAT f32 bytes 112..116, read back exactly as the f64 the
    /// other formats store); `0.0` when absent.
    pub spatial_step_m: f64,
    /// Seconds between two lines, as the file stores it (DAT f32 bytes
    /// 116..120, same rule); `0.0` when absent. Informative: times come from
    /// the anchor, the card's rational and the time map, never from it.
    pub temporal_step_s: f64,
}

/// The light speed in the fibre, halved for the round trip: one sample at
/// `f` Hz spans `1e8 / f` metres of fibre (`audace::LIGHT_SPEED_IN_FIBER_M_S`
/// = 2e8 m/s).
const METRES_PER_SAMPLE_HZ: f64 = 100_000_000.0;

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

    /// The content type a reader applies: the header's, except value 5 with
    /// 4-byte samples in a header written before v1 — the f32 magnitude the
    /// demodulation wrote under 5 then, which is 16 (Audace-Lib
    /// `header::normalized_content_type`).
    fn normalized_ct(&self) -> u8 {
        if self.format_version == 0 && self.content_type == 5 && self.sample_size == 4 {
            16
        } else {
            self.content_type
        }
    }

    fn info(&self) -> Option<&'static content::ContentInfo> {
        content::info(self.normalized_ct())
    }

    /// `true` when the content type fixes a sample size the header does not
    /// have: such a file is read from the flags, like content type 0.
    fn contradicts(&self) -> bool {
        let ct = self.normalized_ct();
        !content::layout_from_flags(ct)
            && content::info(ct).and_then(|i| i.sample_size).map(i32::from) != Some(self.sample_size)
    }

    /// Whether what the samples ARE comes from the flags: content type 0, an
    /// unassigned value, a contradicted `sample_size`.
    fn quantity_from_flags(&self) -> bool {
        self.content_type == 0 || self.info().is_none() || self.contradicts()
    }

    /// `(bytes, float, unsigned)`: how every sample is stored — Audace-Lib's
    /// `header::sample_encoding`. From the content type whenever it fixes it
    /// and `sample_size` agrees; else from `sample_size` and the FLOAT /
    /// UNSIGNED / INTERLEAVED flags, exactly as before v1. `unsigned` means
    /// every sample of the array (a single `u16` column); a pair whose
    /// members differ stays signed, its odd member reinterpreted by the
    /// extractor.
    fn encoding(&self) -> (i32, bool, bool) {
        let bytes = self.sample_size;
        if let Some(info) = self.info().filter(|i| i.sample_size.map(i32::from) == Some(bytes)) {
            let typed = match (info.sample_type, info.members) {
                (Some(SampleType::FromFlags), _) | (None, None) => None,
                (Some(t), _) => Some((t, t)),
                (None, Some([a, b])) => Some((a.sample_type, b.sample_type)),
            };
            if let Some((a, b)) = typed {
                return (bytes, a == SampleType::F32, a == SampleType::U16 && b == SampleType::U16);
            }
        }
        let f = self.flags;
        (
            bytes,
            f & flag_bits::FLOAT != 0,
            f & flag_bits::UNSIGNED != 0 && f & (flag_bits::FLOAT | flag_bits::INTERLEAVED) == 0,
        )
    }

    fn interleaved(&self) -> bool {
        let ct = self.normalized_ct();
        content::is_stream_only(ct)
            || (content::layout_from_flags(ct) && self.flags & flag_bits::INTERLEAVED != 0)
    }

    /// The steps the header's other fields give, f64 (Audace-Lib
    /// `derived_steps`): `1e8 / sample_rate` for a value whose positions are
    /// the producer's samples (not a script's output, not a zone variance,
    /// not an unassigned value), the real line period from the cadence; 0
    /// each when it cannot be derived.
    fn derived_steps(&self) -> (f64, f64) {
        let ct = self.content_type;
        let quantity = content::info(ct).and_then(|i| i.quantity.or(i.members.map(|m| m[0].quantity)));
        let derivable = content::info(ct).is_some()
            && !matches!(quantity, Some("custom" | "variance" | "log_variance"));
        let spatial = if derivable && self.sample_rate > 0 {
            METRES_PER_SAMPLE_HZ / self.sample_rate as f64
        } else {
            0.0
        };
        let temporal = self
            .line_rational()
            .map_or(0.0, |(clock, ticks)| ticks as f64 / clock as f64);
        (spatial, temporal)
    }
}

/// A step stored as f32 (DAT), in f64: the derived value when the field is
/// that very value rounded to f32 (`0.4` m, not `0.4000000059604645`), the
/// field otherwise; 0 when absent — Audace-Lib `exact_spatial_step_m` /
/// `exact_temporal_step_s`, so a DAT and an HDF5 of one recording agree.
fn exact_step(field: f32, derived: f64) -> f64 {
    if !(field.is_finite() && field > 0.0) {
        0.0
    } else if derived > 0.0 && (derived as f32) == field {
        derived
    } else {
        f64::from(field)
    }
}

/// The card code of a value given to `Header(card=…)`: a code, a tag written
/// before v1 (mapped), `None` / `""` for none; anything else is refused.
fn card_argument(card: Option<String>) -> PyResult<Option<String>> {
    match card.as_deref() {
        None | Some("") => Ok(None),
        Some(s) => content::card_code(s).map(|c| Some(c.to_string())).ok_or_else(|| {
            PyValueError::new_err(format!(
                "card must be one of {:?} (or None), got {s:?}",
                content::CARD_CODES
            ))
        }),
    }
}

#[pymethods]
impl Header {
    /// Build a header from individual fields. Used by the format wrappers
    /// (HDF5/TDMS/SEG-Y) to feed the same channel extractors as the DAT
    /// path without going through the binary wire format. `card` takes a
    /// card code (`"p18"`, `"p21"`, `"s125"`, `"s500"`) or `None`.
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
        line_offset = 0,
        time_flags = 0,
        card = None,
        duration_s = 0,
        format_version = 0,
        spatial_step_m = 0.0,
        temporal_step_s = 0.0,
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
        line_offset: i32,
        time_flags: u32,
        card: Option<String>,
        duration_s: u32,
        format_version: u8,
        spatial_step_m: f64,
        temporal_step_s: f64,
    ) -> PyResult<Self> {
        let finite_or_zero = |v: f64| if v.is_finite() && v > 0.0 { v } else { 0.0 };
        Ok(Self {
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
            card: card_argument(card)?,
            acquisition_ns,
            trigger_delay,
            trig_clock_hz,
            trig_period_ticks,
            line_offset,
            time_flags,
            duration_s,
            format_version,
            spatial_step_m: finite_or_zero(spatial_step_m),
            temporal_step_s: finite_or_zero(temporal_step_s),
        })
    }

    /// `True` when the header's start time is the corrected time of the
    /// file's first line (`time_flags & TIME_FLAG_CORRECTED`): a DAT stamped
    /// from the card's time map, or an HDF5 whose `time_correction` is
    /// `"cardclock-v1"`. Always `False` for TDMS and SEG-Y: neither format
    /// carries the time flags.
    #[getter]
    fn time_corrected(&self) -> bool {
        self.time_flags & time_flag_bits::TIME_CORRECTED != 0
    }

    /// The cadence lines are timed with, `(clock, ticks)`: the card's
    /// rational when known and consistent, else `(trig_frequency, 1)`;
    /// `None` without any cadence.
    #[getter]
    fn line_timing(&self) -> Option<(u32, u32)> {
        self.line_rational()
    }

    /// The card's trigger rational `(trig_clock_hz, trig_period_ticks)`, or
    /// `None` when the header does not carry one (both fields `0`: a capture
    /// written before the fields existed, an external trigger) or carries
    /// one that does not describe this header — `clock / ticks` further from
    /// `trig_frequency` than any card's quantization puts it (see
    /// `consistent_rational`), a rational left behind by a header whose
    /// frequency was rewritten. Such a rational is ignored by every timing
    /// helper.
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

    /// Lowercase `content_type` tag (e.g. `"dui_coherent_phase"`),
    /// `"undefined"` for a capture written before the byte existed (0), and
    /// `"unknown(<n>)"` for a value this SDK does not know (a newer producer).
    #[getter]
    fn content_type_name(&self) -> String {
        content_type_tag(self.content_type)
    }

    /// The content type a reader applies: `content_type`, except value 5
    /// with 4-byte samples in a header written before data format v1 (the
    /// f32 magnitude the demodulation labelled 5 then), which is 16
    /// (`hw_amplitude_f32`).
    #[getter]
    fn normalized_content_type(&self) -> u8 {
        self.normalized_ct()
    }

    /// `True` when the content type fixes a sample size this header does not
    /// have (a file of the retired software PHASE method, say): the samples
    /// are then read from the flags, as for content type 0.
    #[getter]
    fn sample_size_contradicts_content_type(&self) -> bool {
        self.contradicts()
    }

    /// `(bytes, float, unsigned)`: how each sample is stored — from the
    /// content type when it fixes it, else from `sample_size` and the
    /// flags (content type 0, 12, an unassigned value, a contradicted
    /// `sample_size`). `unsigned` is `True` when every sample of the array
    /// is unsigned (a single `u16` column).
    #[getter]
    fn sample_encoding(&self) -> (i32, bool, bool) {
        self.encoding()
    }

    /// True when the samples come in pairs per position: a stream-only
    /// (interleaved) content type, or the `INTERLEAVED` flag for content
    /// type 0, 12 and an unassigned value. Never for a v1 file.
    #[getter]
    fn is_interleaved(&self) -> bool {
        self.interleaved()
    }

    /// True if samples are IEEE-754 floats (from the content type when it
    /// fixes the sample type, else the `FLOAT` flag).
    #[getter]
    fn is_float(&self) -> bool {
        self.encoding().1
    }

    /// True when (some of) the integer samples are unsigned: a `u16` column
    /// (content types 5, 25), the magnitude member of an arctan/magnitude
    /// pair (7) — or, read from the flags (content type 0, 12, an unassigned
    /// value), the `UNSIGNED` bit: on an INTERLEAVED pair the odd lane is
    /// `u16`; on a single column every integer sample is (read as `uint16`).
    #[getter]
    fn is_unsigned(&self) -> bool {
        let ct = self.normalized_ct();
        match self.info() {
            Some(i) if !content::layout_from_flags(ct) && !self.contradicts() => {
                i.sample_type == Some(SampleType::U16)
                    || i.members.is_some_and(|m| m.iter().any(|m| m.sample_type == SampleType::U16))
            }
            _ => self.flags & flag_bits::UNSIGNED != 0,
        }
    }

    /// True if the samples come out of a demodulation (on board or in
    /// software): phase, arctan, amplitude, I or Q, or a pair of them. From
    /// the content type; from the `DEMODULATED` flag for content type 0, an
    /// unassigned value or a contradicted `sample_size`.
    #[getter]
    fn is_demodulated(&self) -> bool {
        if self.quantity_from_flags() {
            return self.flags & flag_bits::DEMODULATED != 0;
        }
        let i = self.info().expect("quantity_from_flags covers an unknown value");
        i.members.is_some()
            || matches!(i.quantity, Some("phase" | "arctan" | "amplitude" | "i" | "q"))
    }

    /// True if the samples are (or include, as member 0 of a pair) a phase in
    /// radians. From the content type; from the `PHASE` flag for content
    /// type 0, an unassigned value or a contradicted `sample_size`.
    #[getter]
    fn is_phase(&self) -> bool {
        if self.quantity_from_flags() {
            return self.flags & flag_bits::PHASE != 0;
        }
        let i = self.info().expect("quantity_from_flags covers an unknown value");
        i.quantity == Some("phase") || i.members.is_some_and(|m| m[0].quantity == "phase")
    }

    #[getter]
    fn is_ac(&self) -> bool {
        self.flags & flag_bits::AC != 0
    }

    #[getter]
    fn is_hiz(&self) -> bool {
        self.flags & flag_bits::HIZ != 0
    }

    /// Whether the samples are the Mozart software phase (Mozart or
    /// Mozart+), or `None` when the question does not apply or cannot be
    /// answered.
    ///
    /// * A file that carries a `content_type` (non-zero) is answered by it
    ///   alone — the flags are not consulted: `True` for `sw_mozart` and
    ///   `sw_mozart_plus`, `None` for `raw` (no demodulation ran) and for a
    ///   value this SDK does not know, `False` for every other content type.
    /// * A capture written before `content_type` existed (`0`) keeps the
    ///   flag-based answer: `None` when `DEMODULATED` is clear, else whether
    ///   the `MOZART` bit is set.
    ///
    /// `False` only says "not Mozart"; it names no other pipeline (the
    /// `SCHUBERT` flag is deprecated). Treat `None` as "not applicable",
    /// never as a default.
    #[getter]
    fn is_mozart(&self) -> Option<bool> {
        match self.content_type {
            0 => (self.flags & flag_bits::DEMODULATED != 0)
                .then_some(self.flags & flag_bits::MOZART != 0),
            ct => match content::info(ct) {
                None => None,
                Some(i) if i.quantity == Some("raw") => None,
                Some(i) => Some(matches!(i.method, Some("MOZART" | "MOZART_PLUS"))),
            },
        }
    }

    /// Physical spatial positions per pulse — `line_size / 2` for interleaved
    /// pairs (see `is_interleaved`), `line_size` otherwise. Mirrors the
    /// producer's `positions_per_line`.
    #[getter]
    fn positions_per_line(&self) -> i32 {
        if self.interleaved() {
            self.line_size / 2
        } else {
            self.line_size
        }
    }

    /// The spatial step the header's other fields give, m: `1e8 /
    /// sample_rate` for a content type whose positions are the producer's
    /// samples (not a script's output, a zone variance or an unassigned
    /// value), else `0.0`. What a reader uses when `spatial_step_m` is
    /// absent.
    #[getter]
    fn derived_spatial_step_m(&self) -> f64 {
        self.derived_steps().0
    }

    /// The line period the cadence gives, s: `trig_period_ticks /
    /// trig_clock_hz` on a consistent rational, else `1 / trig_frequency`;
    /// `0.0` without a cadence.
    #[getter]
    fn derived_temporal_step_s(&self) -> f64 {
        self.derived_steps().1
    }

    fn __repr__(&self) -> String {
        // The trigger rational, the time-map fields, the card and the data
        // format v1 fields only when the header carries them, so the
        // representation of every older header is unchanged.
        let mut timing = if self.trig_clock_hz != 0 || self.trig_period_ticks != 0 {
            format!(
                ", trig_clock_hz={}, trig_period_ticks={}",
                self.trig_clock_hz, self.trig_period_ticks
            )
        } else {
            String::new()
        };
        if self.line_offset != 0 || self.time_flags != 0 {
            timing.push_str(&format!(
                ", line_offset={}, time_flags=0x{:X}",
                self.line_offset, self.time_flags
            ));
        }
        if let Some(card) = &self.card {
            timing.push_str(&format!(", card='{card}'"));
        }
        if self.content_type != 0 {
            timing.push_str(&format!(", content_type={}", self.content_type_name()));
        }
        if self.format_version != 0 {
            timing.push_str(&format!(", format_version={}", self.format_version));
        }
        if self.duration_s != 0 {
            timing.push_str(&format!(", duration_s={}", self.duration_s));
        }
        if self.spatial_step_m != 0.0 || self.temporal_step_s != 0.0 {
            timing.push_str(&format!(
                ", spatial_step_m={}, temporal_step_s={}",
                self.spatial_step_m, self.temporal_step_s
            ));
        }
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
    let read_f32 = |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    let mut h = Header {
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
        card: content::card_code_of_slot(&bytes[offsets::CARD_TAG..offsets::CARD_TAG + offsets::CARD_TAG_LEN])
            .map(str::to_string),
        acquisition_ns,
        trigger_delay: read_u32_le(bytes, offsets::TRIGGER_DELAY),
        trig_clock_hz: read_u32_le(bytes, offsets::TRIG_CLOCK_HZ),
        trig_period_ticks: read_u32_le(bytes, offsets::TRIG_PERIOD_TICKS),
        line_offset: read_i32_le(bytes, offsets::LINE_OFFSET),
        time_flags: read_u32_le(bytes, offsets::TIME_FLAGS),
        duration_s: read_u32_le(bytes, offsets::DURATION_S),
        format_version: bytes[offsets::FORMAT_VERSION],
        spatial_step_m: 0.0,
        temporal_step_s: 0.0,
    };
    // The steps are f32 here: read back exactly as the other formats store
    // them (f64), see `exact_step`.
    let (spatial, temporal) = h.derived_steps();
    h.spatial_step_m = exact_step(read_f32(offsets::SPATIAL_STEP_M), spatial);
    h.temporal_step_s = exact_step(read_f32(offsets::TEMPORAL_STEP_S), temporal);
    Ok(h)
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
    /// Bytes past the last whole line: a file cut short mid-line. Never read.
    trailing_bytes: u64,
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
        let num_lines = payload.checked_div(bytes_per_line).unwrap_or(0);
        let trailing_bytes = payload.checked_rem(bytes_per_line).unwrap_or(payload);

        Ok(Self {
            path,
            reader: Some(reader),
            header,
            num_lines,
            trailing_bytes,
            lines_left: num_lines,
        })
    }

    /// Bytes past the last whole line — `0` for a complete file. A non-zero
    /// value is a file cut short in the middle of a line (a recording that
    /// stopped mid-write, a partial copy); those bytes are never read.
    #[getter]
    fn trailing_bytes(&self) -> u64 {
        self.trailing_bytes
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
    /// line_size)` numpy array. The dtype is the header's sample encoding
    /// (`Header.sample_encoding`: the content type's sample type, or
    /// `(sample_size, FLOAT, UNSIGNED, INTERLEAVED)` for content type 0, 12
    /// and an unassigned value) — see [`bytes_to_typed_array`]: signed
    /// integers (negative ADC codes stay negative) except a single unsigned
    /// column, which is unsigned.
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

        // Initialised buffer: `read_exact` needs a `&mut [u8]`, which must
        // never cover uninitialised memory. `vec![0; n]` is a zeroed
        // allocation (calloc: fresh pages from the OS come zeroed), so a
        // multi-GB read does not pay a separate zero-fill pass.
        let mut bytes: Vec<u8> = vec![0u8; total_bytes];

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

        let (sample_size, is_float, unsigned) = self.header.encoding();
        bytes_to_typed_array(py, bytes, sample_size, is_float, unsigned, n as usize, line_size)
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
/// 2-D numpy array. Dispatches on the header's sample encoding
/// (`Header::encoding`: `sample_size`, float, unsigned): the **signed**
/// type when integer (FileWriter / Cardcontrol conventions), except a
/// single unsigned column (content type 5, 25, or a legacy single column
/// with UNSIGNED), whose samples are the unsigned type of the same width.
/// On a pair whose members differ (arctan `i16` + magnitude `u16`) the
/// array stays signed: the odd lane is reinterpreted downstream by
/// `split_pair_unsigned`, not by a whole-array type change.
///
/// Takes ownership of `bytes` so the dispatch can move it directly into
/// the typed `Vec<T>` without an intermediate copy.
fn bytes_to_typed_array<'py>(
    py: Python<'py>,
    bytes: Vec<u8>,
    sample_size: i32,
    is_float: bool,
    unsigned: bool,
    rows: usize,
    line_size: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = (rows, line_size);
    macro_rules! typed {
        ($t:ty) => {{
            let arr = Array2::<$t>::from_shape_vec(shape, bytes_to_vec::<$t>(&bytes))
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }};
    }

    let any: Bound<'py, PyAny> = match (sample_size, is_float, unsigned) {
        (2, _, false) => typed!(i16),
        (2, _, true) => typed!(u16),
        (4, true, _) => typed!(f32),
        (4, false, false) => typed!(i32),
        (4, false, true) => typed!(u32),
        (8, true, _) => typed!(f64),
        (8, false, false) => typed!(i64),
        (8, false, true) => typed!(u64),
        (1, _, _) => {
            // u8 path: no reinterpret needed, hand the bytes straight to ndarray.
            let arr = Array2::<u8>::from_shape_vec(shape, bytes)
                .map_err(|e| PyValueError::new_err(format!("shape mismatch: {e}")))?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        (s, f, _) => {
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
    // by the call sites in `bytes_to_typed_array` (u16/i16/u32/i32/u64/
    // i64/f32/f64).
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
// 2. The output is collected from an exact-size iterator
//    (`chunks_exact(2).map(..)`), so it is allocated once and the inner
//    loop has no capacity-check branch — and no element is ever exposed
//    uninitialised.
// 3. The lane pick is `chunks_exact(2).map(|c| c[lane])` — LLVM
//    auto-vectorises this on SSE/AVX (gather + extract).
// 4. We never invoke the slow strided fallback in production; if a
//    caller hands us a non-contiguous numpy view we make an explicit
//    contiguous copy via `view.to_owned()` so the fast path runs.

/// Pick the even-indexed (primary) lane of an INTERLEAVED `i16` array:
/// member 0 — I of an I/Q pair, `arctan(Q/I)` of an arctan/magnitude pair.
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

/// Pick the odd-indexed (secondary) lane of an INTERLEAVED `i16` array:
/// member 1 — Q of an I/Q pair (use the unsigned variant for the magnitude
/// of an arctan/magnitude pair).
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
/// **reinterpret the bits as `u16`**: the `√(I²+Q²)` magnitude of an
/// arctan/magnitude pair (the card sends the magnitude as unsigned 16-bit
/// although the buffer is typed `i16` end to end).
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

    // The odd lane, each `i16` reinterpreted bit for bit as `u16` (`as`
    // between same-width integers keeps the bit pattern) — exactly the wire
    // reinterpretation the vendor doc calls for.
    let u16_vec: Vec<u16> = lane_values(view, 1, total, |v: i16| v as u16);

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
    // One I,Q pair (two contiguous i16) per output element; numpy gives
    // us a contiguous slice for typical inputs.
    let pair = |p: &[i16]| Complex32::new(p[0] as f32, p[1] as f32);
    let out: Vec<Complex32> = match view.as_slice() {
        Some(input) => input.chunks_exact(2).map(pair).collect(),
        // Strided fallback — copy through a contiguous buffer once.
        None => {
            let owned = view.to_owned();
            let slice = owned.as_slice().expect("to_owned is always contiguous");
            slice.chunks_exact(2).map(pair).collect()
        }
    };
    debug_assert_eq!(out.len(), total);

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

/// Generic lane picker. Reads from `view` and collects the requested lane
/// into a fresh `Vec<T>`: `chunks_exact(2)` is an exact-size iterator, so
/// `collect` allocates once and the loop has no capacity check — what LLVM
/// auto-vectorises into SSE / AVX gather-extract on x86_64.
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
    lane_values(view, lane, total, |v: T| v)
}

/// [`split_lane_into_vec`] with a per-element conversion `f` (a bit-cast).
fn lane_values<T: Copy + numpy::Element, U>(
    view: ArrayView2<'_, T>,
    lane: usize,
    total: usize,
    f: impl Fn(T) -> U,
) -> Vec<U> {
    let out: Vec<U> = match view.as_slice() {
        Some(input) => input.chunks_exact(2).map(|pair| f(pair[lane])).collect(),
        None => {
            // Strided / non-standard layout: pay the copy once.
            let owned = view.to_owned();
            let slice = owned.as_slice().expect("to_owned is always contiguous");
            slice.chunks_exact(2).map(|pair| f(pair[lane])).collect()
        }
    };
    debug_assert_eq!(out.len(), total);
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
    // Card code (bytes 65..81): ASCII, NUL-padded (Audace-Lib
    // `set_card_code`); zero when the header carries none. `Header.card` only
    // ever holds a code.
    if let Some(card) = header.card.as_deref().and_then(content::card_code) {
        let code = card.as_bytes();
        buf[offsets::CARD_TAG..offsets::CARD_TAG + code.len()].copy_from_slice(code);
    }
    // Trigger pre-roll (bytes 84..88), u32 LE.
    buf[offsets::TRIGGER_DELAY..offsets::TRIGGER_DELAY + 4]
        .copy_from_slice(&header.trigger_delay.to_le_bytes());
    // Trigger rational (bytes 88..96), u32 LE each, written as stored: zero
    // on every header that does not carry one.
    buf[offsets::TRIG_CLOCK_HZ..offsets::TRIG_CLOCK_HZ + 4]
        .copy_from_slice(&header.trig_clock_hz.to_le_bytes());
    buf[offsets::TRIG_PERIOD_TICKS..offsets::TRIG_PERIOD_TICKS + 4]
        .copy_from_slice(&header.trig_period_ticks.to_le_bytes());
    // Time-map fields (bytes 96..104), written as stored: zero on every
    // header that does not carry them.
    buf[offsets::LINE_OFFSET..offsets::LINE_OFFSET + 4]
        .copy_from_slice(&header.line_offset.to_le_bytes());
    buf[offsets::TIME_FLAGS..offsets::TIME_FLAGS + 4]
        .copy_from_slice(&header.time_flags.to_le_bytes());
    // Data format v1 (bytes 104..120), zero when absent: duration, version,
    // and the steps as f32 (a step read from this layout comes back as the
    // same f32, see `exact_step`).
    buf[offsets::DURATION_S..offsets::DURATION_S + 4].copy_from_slice(&header.duration_s.to_le_bytes());
    buf[offsets::FORMAT_VERSION] = header.format_version;
    let step = |v: f64| if v.is_finite() && v > 0.0 { v as f32 } else { 0.0 };
    buf[offsets::SPATIAL_STEP_M..offsets::SPATIAL_STEP_M + 4]
        .copy_from_slice(&step(header.spatial_step_m).to_le_bytes());
    buf[offsets::TEMPORAL_STEP_S..offsets::TEMPORAL_STEP_S + 4]
        .copy_from_slice(&step(header.temporal_step_s).to_le_bytes());

    Ok(PyBytes::new(py, &buf))
}

// ── Time axis (the card's time map, in a file) ──────────────────────────────

const FS_PER_NS: i128 = 1_000_000;

/// The longest line period a time-axis node may carry, fs (1 000 s; a card's
/// is 5 µs to 1 ms) — Audace-Lib's `time_map::MAX_PERIOD_FS`. Beyond it an
/// axis is forged or corrupt, and refused.
const MAX_PERIOD_FS: u64 = 1_000_000_000_000_000_000;

/// Time of file line `line` on a time axis: from node `k` on (the last node
/// whose line is `<= line`, the first one for a line before every node),
/// `ns[k] + ⌊(line − lines[k]) · period_fs[k] / 10⁶⌋`, or, for a node of
/// period 0, `ns[k] + ⌊(line − lines[k]) · ticks · 10⁹ / clock⌋` on the
/// file's cadence (0 without one). Floor for a negative offset too;
/// saturated to the `i64` range. The very integers of the acquisition's
/// `audace::time_map::axis_time_of` (shared fixtures).
fn axis_eval(lines: &[i64], ns: &[i64], period_fs: &[u64], timing: Option<(u32, u32)>, line: i64) -> i64 {
    let k = lines.partition_point(|&l| l <= line).saturating_sub(1);
    let d = line as i128 - lines[k] as i128;
    // Checked products: |d| < 2⁶⁴, so a period within MAX_PERIOD_FS (and any
    // u32 cadence) stays far inside i128 — and anything else saturates
    // instead of panicking.
    let sat = |p: Option<i128>| p.unwrap_or(if d < 0 { i128::MIN / 2 } else { i128::MAX / 2 });
    let dt = if period_fs[k] == 0 {
        match timing {
            Some((clock, ticks)) if clock > 0 => {
                sat(d.checked_mul(ticks as i128 * NS_PER_S as i128)).div_euclid(clock as i128)
            }
            _ => 0,
        }
    } else {
        sat(d.checked_mul(period_fs[k] as i128)).div_euclid(FS_PER_NS)
    };
    (ns[k] as i128).saturating_add(dt).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// The times of `count` consecutive file lines from `first` on, ns, as an
/// `int64` numpy vector, on the time axis `(lines, ns, period_fs)` (see
/// [`axis_eval`]) with the file's cadence `timing = (clock, ticks)` for
/// nodes of period 0. Exact integer arithmetic (i128): a time above 2⁵³
/// keeps every nanosecond. Metadata only — no sample is read. Raises
/// `MemoryError` when `count` times cannot be allocated.
#[pyfunction]
#[pyo3(signature = (lines, ns, period_fs, timing, first, count))]
fn time_axis_ns<'py>(
    py: Python<'py>,
    lines: Vec<i64>,
    ns: Vec<i64>,
    period_fs: Vec<u64>,
    timing: Option<(u32, u32)>,
    first: i64,
    count: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if lines.is_empty() || lines.len() != ns.len() || ns.len() != period_fs.len() {
        return Err(PyValueError::new_err(format!(
            "a time axis needs as many lines, times and periods, at least one \
             (got {}, {}, {})",
            lines.len(),
            ns.len(),
            period_fs.len()
        )));
    }
    if lines.windows(2).any(|w| w[0] >= w[1]) {
        return Err(PyValueError::new_err("time axis lines must be strictly increasing"));
    }
    if period_fs.iter().any(|&p| p > MAX_PERIOD_FS) {
        return Err(PyValueError::new_err("a time axis period beyond 1 000 s per line: forged or corrupt"));
    }
    // The vector is reserved up front, fallibly: a count no memory can hold
    // (a forged HDF5 that declares 10¹² lines) raises `MemoryError` instead
    // of aborting the interpreter on a failed allocation.
    let mut out: Vec<i64> = Vec::new();
    out.try_reserve_exact(count).map_err(|e| {
        PyMemoryError::new_err(format!("cannot hold the times of {count} lines: {e}"))
    })?;
    py.allow_threads(|| {
        out.extend((0..count).map(|j| {
            let line = first.saturating_add(i64::try_from(j).unwrap_or(i64::MAX));
            axis_eval(&lines, &ns, &period_fs, timing, line)
        }))
    });
    Ok(PyArray1::from_vec(py, out))
}

// ── Module init ─────────────────────────────────────────────────────────────

/// Module entry point — Python imports this as `invisensing._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Header>()?;
    m.add_class::<DatReader>()?;
    m.add_function(wrap_pyfunction!(parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(build_header_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(time_axis_ns, m)?)?;
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
    // Deprecated: no reader decides anything on this bit (the content type
    // is the reference); kept so existing imports keep working.
    m.add("FLAG_SCHUBERT", flag_bits::SCHUBERT)?;
    m.add("FLAG_RESERVED", flag_bits::RESERVED)?;
    m.add("FLAG_MOZART", flag_bits::MOZART)?;
    m.add("FLAG_PHASE", flag_bits::PHASE)?;
    m.add("FLAG_INTERLEAVED", flag_bits::INTERLEAVED)?;
    m.add("FLAG_UNSIGNED", flag_bits::UNSIGNED)?;
    m.add("HEADER_SIZE", HEADER_SIZE)?;
    m.add("TIME_FLAG_MAP_V1", time_flag_bits::TIME_MAP_V1)?;
    m.add("TIME_FLAG_CORRECTED", time_flag_bits::TIME_CORRECTED)?;

    // `{content_type value: tag}` for every value this SDK knows, so the
    // format backends can map a file's tag attribute back to its value.
    let names = PyDict::new(m.py());
    for v in 0..=u8::MAX {
        if let Some(tag) = known_content_type_tag(v) {
            names.set_item(v, tag)?;
        }
    }
    m.add("CONTENT_TYPE_NAMES", names)?;
    m.add("CONTENT_TYPES", content_types(m.py())?)?;
    m.add("FORMAT_VERSION", FORMAT_VERSION)?;
    m.add("CARD_CODES", content::CARD_CODES.to_vec())?;
    // `[("fnv1a64:<hex>", code)]`: the digests the former card variant tags
    // are recognised by, written as the registry copy writes them.
    let digests: Vec<(String, &str)> = content::LEGACY_TAG_DIGESTS
        .iter()
        .map(|(d, code)| (format!("fnv1a64:{d:016x}"), *code))
        .collect();
    m.add("LEGACY_CARD_DIGESTS", digests)?;
    m.add_function(wrap_pyfunction!(card_code, m)?)?;
    let layout = PyList::empty(m.py());
    for (name, offset, size, ty) in HEADER_LAYOUT {
        layout.append((name, offset, size, ty))?;
    }
    m.add("HEADER_LAYOUT", layout)?;

    Ok(())
}

/// The registry as a list of dicts, one per value, keyed like the
/// producer's machine-readable registry (`value`, `tag`, `quantity`,
/// `sample_type`, `sample_size`, `unit`, `conversion`, `method`,
/// `file_or_stream`, `members`, `split_children`, `status`).
fn content_types(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let rows = PyList::empty(py);
    for i in content::REGISTRY.iter() {
        let d = PyDict::new(py);
        d.set_item("value", i.value)?;
        d.set_item("tag", i.tag)?;
        d.set_item("quantity", i.quantity)?;
        d.set_item("sample_type", i.sample_type.map(SampleType::as_str))?;
        d.set_item("sample_size", i.sample_size)?;
        d.set_item("unit", i.unit)?;
        match i.conversion {
            Some(c) => {
                let cd = PyDict::new(py);
                cd.set_item("to", c.to)?;
                cd.set_item("expr", c.expr)?;
                cd.set_item("multiply_by", c.multiply_by)?;
                cd.set_item("informative", c.informative)?;
                d.set_item("conversion", cd)?;
            }
            None => d.set_item("conversion", py.None())?,
        }
        d.set_item("method", i.method)?;
        d.set_item("file_or_stream", i.placement.as_str())?;
        match i.members {
            Some(members) => {
                let ms = PyList::empty(py);
                for m in members {
                    let md = PyDict::new(py);
                    md.set_item("quantity", m.quantity)?;
                    md.set_item("sample_type", m.sample_type.as_str())?;
                    md.set_item("unit", m.unit)?;
                    md.set_item("subdir", m.subdir)?;
                    md.set_item("child", m.child)?;
                    ms.append(md)?;
                }
                d.set_item("members", ms)?;
                d.set_item("split_children", vec![u32::from(members[0].child), u32::from(members[1].child)])?;
            }
            None => {
                d.set_item("members", py.None())?;
                d.set_item("split_children", py.None())?;
            }
        }
        d.set_item("status", i.status)?;
        rows.append(d)?;
    }
    Ok(rows)
}

/// The card code a header slot or a file's `card` attribute stands for:
/// `"p18"`, `"p21"`, `"s125"` or `"s500"` as is, a variant tag written before
/// data format v1 mapped to its code, `None` for anything else (empty,
/// another producer's bytes).
#[pyfunction]
fn card_code(value: &str) -> Option<&'static str> {
    content::card_code(value)
}

//! The `content_type` registry of the Audace data format v1 and the card
//! codes, as this SDK knows them.
//!
//! One row per value: what the samples are (quantity), how they are stored
//! (sample type, size), their unit, the chain that produced them and, for a
//! stream-only (interleaved) value, its two members and their split
//! children. `tests/test_content_type.py` checks every row against the
//! machine-readable registry the producer publishes
//! (`tests/assets/content_types_v1.json`, pinned by its SHA-256), so a value
//! added upstream fails the suite until it is added here.
//!
//! The registry is append-only: a value is never renumbered, reused or given
//! another sample type.

/// How a column's samples are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleType {
    I16,
    U16,
    F32,
    /// Not fixed by the value (0 and 12): from `sample_size` and the
    /// FLOAT / UNSIGNED flags, as before data format v1.
    FromFlags,
}

impl SampleType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::F32 => "f32",
            Self::FromFlags => "from_flags",
        }
    }
}

/// A conversion of the stored value into a physical one (`expr` in `v`).
#[derive(Clone, Copy, Debug)]
pub struct Conversion {
    pub to: &'static str,
    pub expr: &'static str,
    /// The factor, when the conversion is one.
    pub multiply_by: Option<f64>,
    /// `true` when the conversion is nominal and uncalibrated (raw ADC → mV).
    pub informative: bool,
}

/// One of the two values a stream-only content type carries per position.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub quantity: &'static str,
    pub sample_type: SampleType,
    pub unit: &'static str,
    /// Sub-directory the member's file is written to when the stream is split.
    pub subdir: &'static str,
    /// Content type of that file.
    pub child: u8,
}

/// Where a value may appear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    File,
    Stream,
    FileAndStream,
}

impl Placement {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Stream => "stream",
            Self::FileAndStream => "file+stream",
        }
    }
}

/// One registry row.
#[derive(Clone, Copy, Debug)]
pub struct ContentInfo {
    pub value: u8,
    pub tag: &'static str,
    /// `None` for 0 (from the flags) and for a stream-only value (see
    /// `members`).
    pub quantity: Option<&'static str>,
    /// `None` for a stream-only value.
    pub sample_type: Option<SampleType>,
    /// Bytes per sample (per member for a stream-only value); `None` for 0
    /// and 12.
    pub sample_size: Option<u8>,
    pub unit: Option<&'static str>,
    pub conversion: Option<Conversion>,
    /// The demodulation method that produced it (the acquisition's
    /// `DemodMethod` name), `None` for card values and the values no method
    /// produces.
    pub method: Option<&'static str>,
    pub placement: Placement,
    /// The two members of a stream-only value, member 0 first on the wire.
    pub members: Option<[Member; 2]>,
    /// `legacy`, `current`, `frozen` or `new`.
    pub status: &'static str,
}

use Placement::{File, FileAndStream, Stream};
use SampleType::{FromFlags, F32, I16, U16};

#[allow(clippy::too_many_arguments)] // one registry row, column by column
const fn single(
    value: u8,
    tag: &'static str,
    quantity: &'static str,
    sample_type: SampleType,
    unit: &'static str,
    method: Option<&'static str>,
    placement: Placement,
    status: &'static str,
) -> ContentInfo {
    let sample_size = match sample_type {
        I16 | U16 => Some(2),
        F32 => Some(4),
        FromFlags => None,
    };
    ContentInfo {
        value,
        tag,
        quantity: Some(quantity),
        sample_type: Some(sample_type),
        sample_size,
        unit: Some(unit),
        conversion: None,
        method,
        placement,
        members: None,
        status,
    }
}

const fn member(quantity: &'static str, sample_type: SampleType, unit: &'static str, subdir: &'static str, child: u8) -> Member {
    Member { quantity, sample_type, unit, subdir, child }
}

const fn pair(value: u8, tag: &'static str, sample_size: u8, method: Option<&'static str>, status: &'static str, members: [Member; 2]) -> ContentInfo {
    ContentInfo {
        value,
        tag,
        quantity: None,
        sample_type: None,
        sample_size: Some(sample_size),
        unit: None,
        conversion: None,
        method,
        placement: Stream,
        members: Some(members),
        status,
    }
}

const fn with_conversion(mut info: ContentInfo, conversion: Conversion) -> ContentInfo {
    info.conversion = Some(conversion);
    info
}

/// π / 32768: the card's arctan scale (`32767 ↔ +π`).
pub const ARCTAN_RAD_PER_COUNT: f64 = std::f64::consts::PI / 32768.0;

const PHASE_DUI: Member = member("phase", F32, "rad", "phase", 17);
const AMPLITUDE_F32: Member = member("amplitude", F32, "count", "amplitude", 16);

/// The registry, indexed by value (`REGISTRY[v].value == v`): revision 2 of
/// the producer's table (values 0 to 28).
pub static REGISTRY: [ContentInfo; 29] = [
    ContentInfo {
        value: 0,
        tag: "undefined",
        quantity: None,
        sample_type: Some(FromFlags),
        sample_size: None,
        unit: None,
        conversion: None,
        method: None,
        placement: File,
        members: None,
        status: "legacy",
    },
    with_conversion(
        single(1, "raw", "raw", I16, "adc_count", None, FileAndStream, "current"),
        Conversion { to: "mV", expr: "v * range / 32768", multiply_by: None, informative: true },
    ),
    single(2, "hw_phase", "phase", F32, "rad", None, FileAndStream, "current"),
    with_conversion(
        single(3, "hw_arctan", "arctan", I16, "count", None, File, "current"),
        Conversion { to: "rad", expr: "v * pi / 32768", multiply_by: Some(ARCTAN_RAD_PER_COUNT), informative: false },
    ),
    single(4, "hw_arctan_dui", "phase", F32, "rad", None, File, "frozen"),
    single(5, "hw_amplitude", "amplitude", U16, "count", None, File, "current"),
    pair(6, "interleaved_hw_iq", 2, None, "current", [
        member("i", I16, "count", "i_component", 13),
        member("q", I16, "count", "q_component", 14),
    ]),
    pair(7, "interleaved_hw_arctan_amplitude", 2, None, "current", [
        member("arctan", I16, "count", "arctan", 3),
        member("amplitude", U16, "count", "magnitude", 5),
    ]),
    pair(8, "interleaved_hw_phase_amplitude", 4, None, "frozen", [
        member("phase", F32, "rad", "phase", 4),
        AMPLITUDE_F32,
    ]),
    single(9, "sw_mozart", "phase", F32, "rad", Some("MOZART"), FileAndStream, "current"),
    single(10, "sw_phase_sac", "phase", F32, "rad", Some("PHASE_SAC"), FileAndStream, "new"),
    single(11, "sw_amplitude_sac", "amplitude", F32, "front_end_count", Some("PHASE_SAC"), File, "new"),
    single(12, "sw_custom", "custom", FromFlags, "script", None, FileAndStream, "frozen"),
    single(13, "hw_i", "i", I16, "count", None, File, "current"),
    single(14, "hw_q", "q", I16, "count", None, File, "current"),
    pair(15, "interleaved_phase_sac_amplitude", 4, Some("PHASE_SAC"), "new", [
        member("phase", F32, "rad", "phase", 10),
        member("amplitude", F32, "front_end_count", "amplitude", 11),
    ]),
    single(16, "hw_amplitude_f32", "amplitude", F32, "count", None, FileAndStream, "new"),
    single(17, "dui_phase", "phase", F32, "rad", Some("DUI"), FileAndStream, "new"),
    single(18, "dui_stateful_phase", "phase", F32, "rad", Some("DUI_STATEFUL"), FileAndStream, "new"),
    single(19, "dui_coherent_phase", "phase", F32, "rad", Some("DUI_COHERENT"), FileAndStream, "new"),
    pair(20, "interleaved_dui_phase_amplitude", 4, Some("DUI"), "new", [PHASE_DUI, AMPLITUDE_F32]),
    pair(21, "interleaved_dui_stateful_phase_amplitude", 4, Some("DUI_STATEFUL"), "new", [
        member("phase", F32, "rad", "phase", 18),
        AMPLITUDE_F32,
    ]),
    pair(22, "interleaved_dui_coherent_phase_amplitude", 4, Some("DUI_COHERENT"), "new", [
        member("phase", F32, "rad", "phase", 19),
        AMPLITUDE_F32,
    ]),
    single(23, "sw_mozart_plus", "phase", F32, "rad", Some("MOZART_PLUS"), FileAndStream, "new"),
    single(24, "sw_custom_i16", "custom", I16, "script", None, FileAndStream, "new"),
    single(25, "sw_custom_u16", "custom", U16, "script", None, FileAndStream, "new"),
    single(26, "sw_custom_f32", "custom", F32, "script", None, FileAndStream, "new"),
    single(27, "viz_variance", "variance", F32, "source_unit_squared", None, File, "new"),
    with_conversion(
        single(28, "viz_logvar", "log_variance", F32, "dimensionless", None, File, "new"),
        Conversion { to: "source_unit_squared", expr: "10^v - 1", multiply_by: None, informative: false },
    ),
];

/// The registry row of a content type byte; `None` for an unassigned value.
pub fn info(v: u8) -> Option<&'static ContentInfo> {
    REGISTRY.get(v as usize)
}

/// Stream-only (interleaved) values: 6, 7, 8, 15, 20, 21, 22.
pub fn is_stream_only(v: u8) -> bool {
    info(v).is_some_and(|i| i.members.is_some())
}

/// The values whose sample type and layout come from `sample_size` and the
/// flags, as before v1: 0, 12, and an unassigned value.
pub fn layout_from_flags(v: u8) -> bool {
    matches!(info(v), None | Some(ContentInfo { sample_type: Some(FromFlags), .. }))
}

// ── Card codes ──────────────────────────────────────────────────────────────

/// The card codes a header's card slot (bytes 65..81) or a file's `card`
/// attribute holds since data format v1. They are the only names this SDK
/// uses for the acquisition cards.
pub const CARD_CODES: [&str; 4] = ["p18", "p21", "s125", "s500"];

/// 64-bit FNV-1a.
const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// Before data format v1 the slot held a card variant tag, written only by
/// two-card runs; two such tags exist, one per code below. They are
/// recognised by their 64-bit FNV-1a digest rather than spelled out, so
/// that this public source names the cards by their codes only (the
/// registry copy `tests/assets/content_types_v1.json` carries the same
/// digests in `cards[].legacy_tags`, and `tests/test_content_type.py`
/// checks both against each other and against the producer's file).
pub const LEGACY_TAG_DIGESTS: [(u64, &str); 2] = [(0x39ef_1277_a031_18a8, "p18"), (0x39f9_1b77_a039_7c78, "p21")];

/// The card code a slot or attribute value stands for: a code is taken as
/// is, a tag written before v1 is mapped to its code, anything else
/// (empty, another producer's bytes) is `None` — Audace-Lib's
/// `header::card_code`.
pub fn card_code(value: &str) -> Option<&'static str> {
    if let Some(code) = CARD_CODES.iter().find(|c| **c == value) {
        return Some(code);
    }
    let digest = fnv1a64(value.as_bytes());
    LEGACY_TAG_DIGESTS.iter().find(|(d, _)| *d == digest).map(|(_, code)| *code)
}

/// The card code of a header's 16-byte slot (bytes 65..81): the bytes up to
/// the first NUL, through [`card_code`].
pub fn card_code_of_slot(slot: &[u8]) -> Option<&'static str> {
    let end = slot.iter().position(|&b| b == 0).unwrap_or(slot.len());
    std::str::from_utf8(&slot[..end]).ok().and_then(card_code)
}


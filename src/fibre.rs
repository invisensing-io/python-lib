//! The sensing fibre's group index and the speed of light it gives: the
//! reader's mirror of Audace-Lib `src/fibre.rs` (the producer's one
//! derivation), term for term and tested against the same numbers.
//!
//! A DAS measures the round-trip time of a pulse envelope, which travels at
//! the GROUP velocity: the index that turns time into distance is the fibre's
//! group index `n`, and light travels at `v = c / n`. The acquisition derives
//! ONE integer speed, `v = round(c / n)` m/s, half away from zero, and stamps
//! it into every header (bytes 124..128; HDF5 / TDMS `light_speed_in_fiber_m_s`
//! beside the exact `fiber_group_index`; SEG-Y line `FIBER GROUP INDEX`). A
//! file without it was acquired at the legacy 2·10⁸ m/s (index 1.49896229).
//!
//! The f64 forms below scale both operands of the legacy `1e8 / rate` by a
//! power of two, which IEEE 754 does exactly: at v = 2·10⁸ they give the very
//! floats every reader computed before the index existed.
//!
//! The strain constant ([`crate::strain::NANOSTRAIN_PER_RAD_METRE`]) uses the
//! core's PHASE index and the photo-elastic term, not this group index: it
//! does not follow it.

/// Speed of light in vacuum, m/s (exact, SI definition).
pub const SPEED_OF_LIGHT_M_S: u32 = 299_792_458;

/// The acquisition's default group index: Corning SMF-28e+ (ITU-T G.652.D)
/// at 1550 nm (Audace-Lib `fibre::DEFAULT_GROUP_INDEX`).
pub const DEFAULT_GROUP_INDEX: f64 = 1.4682;

/// The index the platform assumed before it was a parameter: `c / 2·10⁸`,
/// exact in decimal. What a file without a speed was acquired at.
pub const LEGACY_GROUP_INDEX: f64 = 1.498_962_29;

/// The speed of light in the fibre a file without a speed (bytes 124..128 =
/// 0, no attribute, no SEG-Y line) was acquired at, m/s.
pub const LEGACY_SPEED_M_S: u32 = 200_000_000;

/// Lowest accepted group index (inclusive): the only rule on n is 1 ≤ n ≤ 100
/// and finite.
pub const MIN_GROUP_INDEX: f64 = 1.0;
/// Highest accepted group index (inclusive).
pub const MAX_GROUP_INDEX: f64 = 100.0;

/// The speed of [`MAX_GROUP_INDEX`]: the lowest speed a header may carry.
pub const MIN_SPEED_M_S: u32 = 2_997_925;
/// The speed of [`MIN_GROUP_INDEX`]: the highest speed a header may carry.
pub const MAX_SPEED_M_S: u32 = SPEED_OF_LIGHT_M_S;

/// `true` when `n` is an accepted group index: finite and within
/// [`MIN_GROUP_INDEX`]`..=`[`MAX_GROUP_INDEX`].
pub fn group_index_is_valid(n: f64) -> bool {
    n.is_finite() && (MIN_GROUP_INDEX..=MAX_GROUP_INDEX).contains(&n)
}

/// The speed of light in a fibre of group index `n`, whole m/s: `round(c /
/// n)`, half away from zero (`f64::round`); `None` for an index outside the
/// accepted range. `speed_m_s(1.4682) == Some(204_190_477)`,
/// `speed_m_s(LEGACY_GROUP_INDEX) == Some(200_000_000)`.
pub fn speed_m_s(n: f64) -> Option<u32> {
    // c / n lies in [c / 100, c] for an accepted n: it fits a u32.
    group_index_is_valid(n).then(|| (f64::from(SPEED_OF_LIGHT_M_S) / n).round() as u32)
}

/// `true` when `v` is a speed a header may carry (the speed of an accepted
/// index). 0 (absent) is not a speed: readers map it to [`LEGACY_SPEED_M_S`].
pub fn speed_is_valid(v: u32) -> bool {
    (MIN_SPEED_M_S..=MAX_SPEED_M_S).contains(&v)
}

/// The speed every metre conversion of a header uses: the stored one when it
/// is a speed an accepted index gives, [`LEGACY_SPEED_M_S`] otherwise (0:
/// every file written before the field; anything else: a header no producer
/// writes). Audace-Lib `header::light_speed_m_s`.
pub fn effective_speed_m_s(stored: u32) -> u32 {
    if speed_is_valid(stored) {
        stored
    } else {
        LEGACY_SPEED_M_S
    }
}

/// The group index a speed stands for, for a reader that only has the speed
/// (a DAT): [`LEGACY_GROUP_INDEX`] for [`LEGACY_SPEED_M_S`] (and for 0) and
/// [`DEFAULT_GROUP_INDEX`] for its speed — the two indexes the platform
/// writes, given back exactly — and `c / v` otherwise. Audace-Lib
/// `fibre::group_index_of_speed`.
pub fn group_index_of_speed(v: u32) -> f64 {
    let v = if v == 0 { LEGACY_SPEED_M_S } else { v };
    if v == LEGACY_SPEED_M_S {
        LEGACY_GROUP_INDEX
    } else if speed_m_s(DEFAULT_GROUP_INDEX) == Some(v) {
        DEFAULT_GROUP_INDEX
    } else {
        f64::from(SPEED_OF_LIGHT_M_S) / f64::from(v)
    }
}

/// The group index of a header: its stored exact index when the file stores
/// one (HDF5, TDMS, SEG-Y) and it gives the header's speed, else the index of
/// the speed ([`group_index_of_speed`]). Audace-Lib `header::fiber_group_index`.
pub fn effective_group_index(stored_index: f64, stored_speed: u32) -> f64 {
    let v = effective_speed_m_s(stored_speed);
    if speed_m_s(stored_index) == Some(v) {
        stored_index
    } else {
        group_index_of_speed(v)
    }
}

/// Metres of fibre between two adjacent positions sampled at `rate_hz`:
/// `v / (2 · rate)`, f64; 0 for a rate that is not positive. At v = 2·10⁸ it
/// is the f64 `1e8 / rate`.
pub fn sample_pitch_m(v: u32, rate_hz: f64) -> f64 {
    if rate_hz > 0.0 { f64::from(v) / (2.0 * rate_hz) } else { 0.0 }
}

/// Metres of fibre spanned by `positions` positions at `rate_hz`, in one
/// expression: `positions · v / (2 · rate)`, f64; 0 without positions or
/// rate. At v = 2·10⁸ it is the f64 `positions · 1e8 / rate`.
pub fn metres_of_positions(positions: f64, v: u32, rate_hz: f64) -> f64 {
    if positions == 0.0 || rate_hz <= 0.0 {
        0.0
    } else {
        positions * f64::from(v) / (2.0 * rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_index_gives_the_legacy_speed_bit_for_bit() {
        assert_eq!(f64::from(SPEED_OF_LIGHT_M_S) / LEGACY_GROUP_INDEX, 2e8);
        assert_eq!(speed_m_s(LEGACY_GROUP_INDEX), Some(LEGACY_SPEED_M_S));
        // An f32 index cannot carry it.
        let n32 = LEGACY_GROUP_INDEX as f32;
        assert_ne!((f64::from(SPEED_OF_LIGHT_M_S) / f64::from(n32)).round(), 2e8);
    }

    #[test]
    fn the_default_index_is_smf28_at_1550_nm() {
        assert_eq!(speed_m_s(DEFAULT_GROUP_INDEX), Some(204_190_477));
        assert_eq!(group_index_of_speed(204_190_477), DEFAULT_GROUP_INDEX);
        assert_eq!(group_index_of_speed(LEGACY_SPEED_M_S), LEGACY_GROUP_INDEX);
        assert_eq!(group_index_of_speed(0), LEGACY_GROUP_INDEX);
        let v = speed_m_s(1.462).unwrap();
        assert_eq!(group_index_of_speed(v), f64::from(SPEED_OF_LIGHT_M_S) / f64::from(v));
        assert_eq!(speed_m_s(MIN_GROUP_INDEX), Some(MAX_SPEED_M_S));
        assert_eq!(speed_m_s(MAX_GROUP_INDEX), Some(MIN_SPEED_M_S));
    }

    #[test]
    fn only_below_one_above_one_hundred_or_not_a_number_is_refused() {
        for ok in [1.0, 1.0000001, 1.4682, LEGACY_GROUP_INDEX, 1.6, 3.0, 14.68, 99.99, 100.0] {
            assert!(speed_m_s(ok).is_some(), "{ok}");
        }
        for bad in [0.0, -1.4682, 0.999_999_9, 100.000_001, 1e9, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(speed_m_s(bad).is_none(), "{bad}");
        }
        assert!(!speed_is_valid(0) && !speed_is_valid(MIN_SPEED_M_S - 1));
        assert!(!speed_is_valid(MAX_SPEED_M_S + 1));
        assert_eq!(effective_speed_m_s(0), LEGACY_SPEED_M_S);
        assert_eq!(effective_speed_m_s(1), LEGACY_SPEED_M_S);
        assert_eq!(effective_speed_m_s(204_190_477), 204_190_477);
    }

    #[test]
    fn the_stored_index_is_believed_only_when_it_gives_the_stored_speed() {
        let v = speed_m_s(1.4675).unwrap();
        assert_eq!(effective_group_index(1.4675, v), 1.4675);
        assert_eq!(effective_group_index(0.0, v), f64::from(SPEED_OF_LIGHT_M_S) / f64::from(v));
        assert_eq!(effective_group_index(1.5, v), effective_group_index(0.0, v));
        assert_eq!(effective_group_index(0.0, 0), LEGACY_GROUP_INDEX);
        assert_eq!(effective_group_index(1.4682, 0), LEGACY_GROUP_INDEX, "no speed: legacy");
    }

    #[test]
    fn the_pitch_at_the_legacy_speed_is_the_old_f64() {
        for rate in [62_500_000i32, 125_000_000, 250_000_000, 500_000_000, 1_000_000_000, 48_828_125, 3] {
            let r = f64::from(rate);
            assert_eq!(sample_pitch_m(LEGACY_SPEED_M_S, r), 100_000_000.0 / r, "{rate}");
            for k in [1.0, 6.0, 13.0, 25.0, 100.0, 4097.0] {
                assert_eq!(metres_of_positions(k, LEGACY_SPEED_M_S, r), k * 100_000_000.0 / r, "{rate} {k}");
            }
        }
        assert_eq!(sample_pitch_m(LEGACY_SPEED_M_S, 0.0), 0.0);
        let v = speed_m_s(DEFAULT_GROUP_INDEX).unwrap();
        for (rate, want) in [(62.5e6, 1.633_524), (125e6, 0.816_762), (250e6, 0.408_381), (500e6, 0.204_190), (1e9, 0.102_095)] {
            assert!((sample_pitch_m(v, rate) - want).abs() < 1e-6, "{rate}");
        }
    }
}

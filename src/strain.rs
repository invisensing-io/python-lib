//! Phase (radians) to strain (nm/m): the factor and the in-place kernels.
//!
//! A differential phase measured over a gauge of `g` metres is a strain:
//!
//! ```text
//! strain [nm/m] = phase [rad] / g [m] × NANOSTRAIN_PER_RAD_METRE
//! ```
//!
//! **The reference** (what every tier computes, bit for bit): the factor
//! `NANOSTRAIN_PER_RAD_METRE / g` is computed once in f64 and, for `f32`
//! samples, rounded once to f32 ([`factor_f32`]); every sample then takes
//! one IEEE-754 multiply by it, in its own precision (`f32 × f32` or
//! `f64 × f64`, round to nearest even). A correctly rounded multiply gives
//! the same bits whatever the lane width, so the SIMD tiers below are
//! bit-identical to the scalar loop: no reciprocal approximation, no
//! reassociation, no FMA. NaN, ±∞, ±0 and subnormals go through the same
//! instruction (the MXCSR is shared by the scalar and the vector unit).
//!
//! **Pairs** (`even_only`): a file of phase/amplitude pairs holds
//! `[phase, amplitude, phase, amplitude, …]`; only the even lanes (the
//! phases) are multiplied. The odd lanes (the amplitudes) are never written
//! by the scalar loop and are left bit-exact by the vector tiers (AVX-512
//! merge-masking, AVX2 blend of the untouched input), signalling NaNs
//! included. Parity is that of the index in the slice: a `(rows, line_size)`
//! C-contiguous array with an even `line_size` starts every row on an even
//! index.
//!
//! **Tiers**: AVX-512F and AVX2 kernels with explicit intrinsics on x86_64,
//! chosen at run time ([`active_tier`]: detected once, cached), and the
//! scalar loop everywhere else (LLVM may vectorise it with the target's
//! baseline SIMD — SSE2 on x86_64, NEON on aarch64 — which changes nothing
//! to its results). `INVISENSING_SIMD=scalar|avx2|avx512` caps the tier for
//! the process (read once; a tier the CPU lacks is clamped to the best one
//! below it, an unknown value is ignored).

use std::sync::OnceLock;

/// nm/m of strain per rad/m of phase over the gauge:
/// `strain_nm_per_m = phase_rad / gauge_length_m × NANOSTRAIN_PER_RAD_METRE`.
///
/// **PROVISIONAL.** This value (115) is an approximation; the final constant
/// must be fixed here before the PyPI release, and it stays a fixed constant
/// of the SDK (not a parameter a caller can change).
pub const NANOSTRAIN_PER_RAD_METRE: f64 = 115.0;

/// Samples below which the Python binding keeps the GIL: releasing it costs
/// more than converting a line.
pub const GIL_RELEASE_MIN_SAMPLES: usize = 1 << 16;

/// `NANOSTRAIN_PER_RAD_METRE / gauge_length_m` in f64: the factor of `f64`
/// samples. `None` when the gauge is not a length (zero, negative, not
/// finite) or the factor is not finite.
pub fn factor_f64(gauge_length_m: f64) -> Option<f64> {
    if !(gauge_length_m.is_finite() && gauge_length_m > 0.0) {
        return None;
    }
    let f = NANOSTRAIN_PER_RAD_METRE / gauge_length_m;
    f.is_finite().then_some(f)
}

/// [`factor_f64`] rounded once to f32: the factor of `f32` samples. `None`
/// as [`factor_f64`], and when the rounded factor is not finite.
pub fn factor_f32(gauge_length_m: f64) -> Option<f32> {
    let f = factor_f64(gauge_length_m)? as f32;
    f.is_finite().then_some(f)
}

/// A conversion kernel, ordered from the plainest to the widest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Scalar,
    Avx2,
    Avx512,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Scalar, Tier::Avx2, Tier::Avx512];

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
        }
    }

    pub fn parse(name: &str) -> Option<Tier> {
        match name.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Tier::Scalar),
            "avx2" => Some(Tier::Avx2),
            "avx512" | "avx512f" => Some(Tier::Avx512),
            _ => None,
        }
    }
}

/// The widest tier this CPU runs (detected once, cached).
pub fn cpu_tier() -> Tier {
    static CPU: OnceLock<Tier> = OnceLock::new();
    *CPU.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Tier {
    if std::arch::is_x86_feature_detected!("avx512f") {
        Tier::Avx512
    } else if std::arch::is_x86_feature_detected!("avx2") {
        Tier::Avx2
    } else {
        Tier::Scalar
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> Tier {
    Tier::Scalar
}

/// `requested`, or the widest tier below it this CPU runs.
pub fn clamp(requested: Tier) -> Tier {
    requested.min(cpu_tier())
}

/// The tier conversions use: [`cpu_tier`], capped by `INVISENSING_SIMD`
/// when it names a tier (read once, cached).
pub fn active_tier() -> Tier {
    static ACTIVE: OnceLock<Tier> = OnceLock::new();
    *ACTIVE.get_or_init(|| {
        std::env::var("INVISENSING_SIMD")
            .ok()
            .and_then(|v| Tier::parse(&v))
            .map_or_else(cpu_tier, clamp)
    })
}

/// Multiply `data` (every sample, or its even lanes) by `factor` in place
/// with `tier`, clamped to what this CPU runs. Returns the tier used.
pub fn scale_f32(tier: Tier, data: &mut [f32], factor: f32, even_only: bool) -> Tier {
    let tier = clamp(tier);
    match tier {
        Tier::Scalar => scalar_f32(data, factor, even_only),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX2 only when the CPU runs it.
        Tier::Avx2 => unsafe { x86::avx2_f32(data, factor, even_only) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX-512 only when the CPU runs AVX-512F.
        Tier::Avx512 => unsafe { x86::avx512_f32(data, factor, even_only) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => scalar_f32(data, factor, even_only),
    }
    tier
}

/// [`scale_f32`] for `f64` samples.
pub fn scale_f64(tier: Tier, data: &mut [f64], factor: f64, even_only: bool) -> Tier {
    let tier = clamp(tier);
    match tier {
        Tier::Scalar => scalar_f64(data, factor, even_only),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX2 only when the CPU runs it.
        Tier::Avx2 => unsafe { x86::avx2_f64(data, factor, even_only) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX-512 only when the CPU runs AVX-512F.
        Tier::Avx512 => unsafe { x86::avx512_f64(data, factor, even_only) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => scalar_f64(data, factor, even_only),
    }
    tier
}

/// The reference: one multiply per converted sample, in index order.
fn scalar_f32(data: &mut [f32], factor: f32, even_only: bool) {
    if even_only {
        data.iter_mut().step_by(2).for_each(|x| *x *= factor);
    } else {
        data.iter_mut().for_each(|x| *x *= factor);
    }
}

fn scalar_f64(data: &mut [f64], factor: f64, even_only: bool) {
    if even_only {
        data.iter_mut().step_by(2).for_each(|x| *x *= factor);
    } else {
        data.iter_mut().for_each(|x| *x *= factor);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! Explicit-intrinsic kernels. Each loads, multiplies and stores whole
    //! vectors (unaligned loads and stores: any start address), then hands
    //! the tail — fewer samples than a vector, starting on an even index
    //! since every vector width is even — to the scalar reference.

    use std::arch::x86_64::*;

    use super::{scalar_f32, scalar_f64};

    /// Even lanes of 16 f32: bits 0, 2, …, 14.
    const EVEN_F32X16: __mmask16 = 0x5555;
    /// Even lanes of 8 f64: bits 0, 2, 4, 6.
    const EVEN_F64X8: __mmask8 = 0x55;
    /// `_mm256_blend_ps` mask taking lanes 0, 2, 4, 6 from the product.
    const EVEN_F32X8: i32 = 0b0101_0101;
    /// `_mm256_blend_pd` mask taking lanes 0, 2 from the product.
    const EVEN_F64X4: i32 = 0b0101;

    /// # Safety
    /// The CPU must run AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512_f32(data: &mut [f32], factor: f32, even_only: bool) {
        let f = _mm512_set1_ps(factor);
        let mut chunks = data.chunks_exact_mut(16);
        for c in &mut chunks {
            let p = c.as_mut_ptr();
            // SAFETY: `c` holds exactly 16 f32; loadu/storeu take any alignment.
            unsafe {
                let v = _mm512_loadu_ps(p);
                let r = if even_only {
                    _mm512_mask_mul_ps(v, EVEN_F32X16, v, f)
                } else {
                    _mm512_mul_ps(v, f)
                };
                _mm512_storeu_ps(p, r);
            }
        }
        scalar_f32(chunks.into_remainder(), factor, even_only);
    }

    /// # Safety
    /// The CPU must run AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_f32(data: &mut [f32], factor: f32, even_only: bool) {
        let f = _mm256_set1_ps(factor);
        let mut chunks = data.chunks_exact_mut(8);
        for c in &mut chunks {
            let p = c.as_mut_ptr();
            // SAFETY: `c` holds exactly 8 f32; loadu/storeu take any alignment.
            unsafe {
                let v = _mm256_loadu_ps(p);
                let m = _mm256_mul_ps(v, f);
                let r = if even_only { _mm256_blend_ps::<EVEN_F32X8>(v, m) } else { m };
                _mm256_storeu_ps(p, r);
            }
        }
        scalar_f32(chunks.into_remainder(), factor, even_only);
    }

    /// # Safety
    /// The CPU must run AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn avx512_f64(data: &mut [f64], factor: f64, even_only: bool) {
        let f = _mm512_set1_pd(factor);
        let mut chunks = data.chunks_exact_mut(8);
        for c in &mut chunks {
            let p = c.as_mut_ptr();
            // SAFETY: `c` holds exactly 8 f64; loadu/storeu take any alignment.
            unsafe {
                let v = _mm512_loadu_pd(p);
                let r = if even_only {
                    _mm512_mask_mul_pd(v, EVEN_F64X8, v, f)
                } else {
                    _mm512_mul_pd(v, f)
                };
                _mm512_storeu_pd(p, r);
            }
        }
        scalar_f64(chunks.into_remainder(), factor, even_only);
    }

    /// # Safety
    /// The CPU must run AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_f64(data: &mut [f64], factor: f64, even_only: bool) {
        let f = _mm256_set1_pd(factor);
        let mut chunks = data.chunks_exact_mut(4);
        for c in &mut chunks {
            let p = c.as_mut_ptr();
            // SAFETY: `c` holds exactly 4 f64; loadu/storeu take any alignment.
            unsafe {
                let v = _mm256_loadu_pd(p);
                let m = _mm256_mul_pd(v, f);
                let r = if even_only { _mm256_blend_pd::<EVEN_F64X4>(v, m) } else { m };
                _mm256_storeu_pd(p, r);
            }
        }
        scalar_f64(chunks.into_remainder(), factor, even_only);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic generator (xorshift64*): no dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
    }

    /// Every class of f32 a multiply can meet: zeros, subnormals, the
    /// extremes, infinities, quiet and signalling NaNs with payloads.
    const SPECIAL_F32: [u32; 16] = [
        0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007f_ffff, 0x0080_0000,
        0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0001,
        0x7f80_0001, 0xff80_1234, 0x3f80_0000, 0x4049_0fdb,
    ];

    const SPECIAL_F64: [u64; 14] = [
        0, 1 << 63, 1, (1 << 63) | 1, 0x000f_ffff_ffff_ffff, 0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff, 0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000, 0xfff8_0000_0000_0001, 0x7ff0_0000_0000_0001,
        0x3ff0_0000_0000_0000, 0x4009_21fb_5444_2d18,
    ];

    /// Random bit patterns (every exponent, NaNs included), salted with the
    /// special values.
    fn samples_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match rng.next() % 4 {
                0 => f32::from_bits(SPECIAL_F32[i % SPECIAL_F32.len()]),
                1 => f32::from_bits(rng.next() as u32),
                _ => ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0,
            })
            .collect()
    }

    fn samples_f64(rng: &mut Rng, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match rng.next() % 4 {
                0 => f64::from_bits(SPECIAL_F64[i % SPECIAL_F64.len()]),
                1 => f64::from_bits(rng.next()),
                _ => ((rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 20.0,
            })
            .collect()
    }

    fn bits32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits64(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Factors of real gauges, and ones that push products to overflow and
    /// to subnormals.
    fn factors() -> Vec<f64> {
        let mut f: Vec<f64> = [0.4, 1.6, 2.0, 9.6, 10.0, 20.4, 100.0]
            .iter()
            .map(|&g| factor_f64(g).unwrap())
            .collect();
        f.extend([1.0, 3.0e38, 1.0e-39, 1.0e300, 1.0e-310, 0.0, -2.5]);
        f
    }

    #[test]
    fn the_factor_is_computed_once_in_f64_then_rounded_once() {
        assert_eq!(NANOSTRAIN_PER_RAD_METRE, 115.0);
        assert_eq!(factor_f64(10.0), Some(11.5));
        assert_eq!(factor_f32(9.6), Some((115.0f64 / 9.6) as f32));
        assert_eq!(factor_f32(1.6), Some((115.0f64 / 1.6) as f32));
        for g in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(factor_f64(g), None, "{g}");
            assert_eq!(factor_f32(g), None, "{g}");
        }
        // Finite in f64, beyond f32.
        assert!(factor_f64(1e-37).is_some());
        assert_eq!(factor_f32(1e-37), None);
        assert_eq!(factor_f64(f64::from_bits(1)), None, "the smallest subnormal");
    }

    #[test]
    fn the_scalar_reference_is_one_multiply_per_converted_sample() {
        let mut v = vec![1.0f32, 2.0, -3.0, 4.0, f32::NAN, 6.0];
        scalar_f32(&mut v, 0.5, false);
        assert_eq!(&v[..4], &[0.5, 1.0, -1.5, 2.0]);
        assert!(v[4].is_nan());
        let mut v = vec![1.0f32, 2.0, -3.0, 4.0, 5.0];
        scalar_f32(&mut v, 0.5, true);
        assert_eq!(v, [0.5, 2.0, -1.5, 4.0, 2.5]);
    }

    #[test]
    fn every_tier_gives_the_scalar_bits_on_every_length_and_offset() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for &factor in &factors() {
            let f32f = factor as f32;
            for len in 0..=200 {
                let base = samples_f32(&mut rng, len + 3);
                let base64 = samples_f64(&mut rng, len + 3);
                // Unaligned starts: offsets 0..3 into the buffer.
                for off in 0..3 {
                    for even_only in [false, true] {
                        let mut want = base[off..off + len].to_vec();
                        scalar_f32(&mut want, f32f, even_only);
                        let mut want64 = base64[off..off + len].to_vec();
                        scalar_f64(&mut want64, factor, even_only);
                        for tier in Tier::ALL {
                            let mut got = base.clone();
                            let used = scale_f32(tier, &mut got[off..off + len], f32f, even_only);
                            assert_eq!(used, clamp(tier));
                            assert_eq!(bits32(&got[off..off + len]), bits32(&want), "{tier:?} f32 {len} {off} {even_only} {factor}");
                            assert_eq!(bits32(&got[..off]), bits32(&base[..off]), "before the slice");
                            let mut got64 = base64.clone();
                            scale_f64(tier, &mut got64[off..off + len], factor, even_only);
                            assert_eq!(bits64(&got64[off..off + len]), bits64(&want64), "{tier:?} f64 {len} {off} {even_only} {factor}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_tier_gives_the_scalar_bits_on_a_large_buffer() {
        let mut rng = Rng(42);
        let n = (1 << 20) + 13;
        let base = samples_f32(&mut rng, n);
        let base64 = samples_f64(&mut rng, n);
        let factor = factor_f64(9.6).unwrap();
        for even_only in [false, true] {
            let mut want = base.clone();
            scalar_f32(&mut want, factor as f32, even_only);
            let mut want64 = base64.clone();
            scalar_f64(&mut want64, factor, even_only);
            for tier in Tier::ALL {
                let mut got = base.clone();
                scale_f32(tier, &mut got, factor as f32, even_only);
                assert_eq!(bits32(&got), bits32(&want), "{tier:?} {even_only}");
                let mut got64 = base64.clone();
                scale_f64(tier, &mut got64, factor, even_only);
                assert_eq!(bits64(&got64), bits64(&want64), "{tier:?} {even_only}");
            }
        }
    }

    #[test]
    fn the_odd_lanes_of_pairs_keep_their_bits_signalling_nans_included() {
        let mut rng = Rng(7);
        for tier in Tier::ALL {
            let mut v = samples_f32(&mut rng, 67);
            for x in v.iter_mut().skip(1).step_by(2) {
                *x = f32::from_bits(0x7f80_0001); // a signalling NaN
            }
            let before = bits32(&v);
            scale_f32(tier, &mut v, 11.5, true);
            for i in (1..v.len()).step_by(2) {
                assert_eq!(v[i].to_bits(), before[i], "{tier:?} lane {i}");
            }
        }
    }

    #[test]
    fn the_cpu_tier_is_what_the_cpu_reports() {
        #[cfg(target_arch = "x86_64")]
        {
            let want = if std::arch::is_x86_feature_detected!("avx512f") {
                Tier::Avx512
            } else if std::arch::is_x86_feature_detected!("avx2") {
                Tier::Avx2
            } else {
                Tier::Scalar
            };
            assert_eq!(cpu_tier(), want);
        }
        assert!(active_tier() <= cpu_tier());
        assert_eq!(clamp(Tier::Scalar), Tier::Scalar);
        assert_eq!(Tier::parse(" AVX512 "), Some(Tier::Avx512));
        assert_eq!(Tier::parse("sse"), None);
        for t in Tier::ALL {
            assert_eq!(Tier::parse(t.as_str()), Some(t));
        }
    }
}

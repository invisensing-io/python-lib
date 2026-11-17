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
//!
//! # Strain rate (nm/m/s)
//!
//! A phase file holds the accumulated (unwrapped) phase of every line. The
//! strain rate of line `n` is the strain increment from line `n − 1` times
//! the line rate:
//!
//! ```text
//! strain_rate[n] [nm/m/s] = (phase[n] − phase[n−1]) [rad] × NANOSTRAIN_PER_RAD_METRE / g [m] × f_trig [Hz]
//! ```
//!
//! **The reference** ([`diff_scale_f32`], [`diff_scale_f64`]): the factor
//! `NANOSTRAIN_PER_RAD_METRE / g × f_trig` is computed once in f64
//! ([`rate_factor_f64`]: the strain factor [`factor_f64`] times `f_trig`)
//! and, for `f32` samples, rounded once to f32 ([`rate_factor_f32`]); every
//! converted sample then takes ONE IEEE-754 subtraction in its own
//! precision, `x[n] − x[n−1]`, and ONE multiply of that difference by the
//! factor. The order is fixed: no FMA (nothing here is a multiply-add),
//! no reassociation, no `x[n]·f − x[n−1]·f`. Every tier executes the same
//! two correctly rounded operations per lane, in the same operand order,
//! so they are bit-identical to the scalar loop, NaN payloads included.
//!
//! The kernels work on a block of whole lines and a `prev` line: row 0 of
//! the block is differenced against `prev`, row `n` against row `n − 1` as
//! it was stored, and on return `prev` holds the block's last line as it
//! was stored — the line the next block of the same file is differenced
//! against. Reading a file in blocks of any size therefore gives the bits
//! of reading it at once. Pairs keep their amplitude lanes bit-exact (they
//! are copied to `prev`, never computed). The caller chooses what precedes
//! a file's first line: a `prev` of NaN makes its phase lanes NaN.

use std::sync::OnceLock;

/// nm/m of strain per rad/m of phase over the gauge:
/// `strain_nm_per_m = phase_rad / gauge_length_m × NANOSTRAIN_PER_RAD_METRE`.
///
/// From the DAS phase-to-strain relation `ΔΦ = 2β(n + Cε)·ε·L`, with
/// `β = 2π/λ`, `λ` the optical wavelength, `n` the refractive index of the
/// fibre core, `Cε` the photo-elastic coefficient and `L` the gauge length:
/// `ε = ΔΦ / (2β(n + Cε)·L)`, so the constant is `λ / (4π(n + Cε))`, in nm
/// per rad. At λ = 1550.12 nm and n = 1.46 it is 110.37 nm/m per rad/m
/// (n + Cε ≈ 1.1177). It is a fixed constant of the SDK, not a parameter a
/// caller can change.
///
/// Its `n` is the core's PHASE (effective) index together with the
/// photo-elastic term, not the GROUP index that turns time of flight into
/// distance (a file's `fiber_group_index`, [`crate::fibre`]): it does not
/// follow that index. A file's gauge length does (it is a distance), so the
/// factor `NANOSTRAIN_PER_RAD_METRE / gauge_length_m` uses the gauge the
/// file states at its own index.
pub const NANOSTRAIN_PER_RAD_METRE: f64 = 110.37;

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

/// The factor a phase increment (rad) of a file whose gauge is
/// `gauge_length_m` metres and whose lines come at `trig_frequency_hz` is
/// multiplied by to read in nm/m/s: [`factor_f64`] × `trig_frequency_hz`,
/// in f64. `None` when the gauge is not a length (see [`factor_f64`]), the
/// line rate is not a rate (zero, negative, not finite) or the factor is
/// not finite.
pub fn rate_factor_f64(gauge_length_m: f64, trig_frequency_hz: f64) -> Option<f64> {
    if !(trig_frequency_hz.is_finite() && trig_frequency_hz > 0.0) {
        return None;
    }
    let f = factor_f64(gauge_length_m)? * trig_frequency_hz;
    f.is_finite().then_some(f)
}

/// [`rate_factor_f64`] rounded once to f32: the factor of `f32` samples.
/// `None` as [`rate_factor_f64`], and when the rounded factor is not finite.
pub fn rate_factor_f32(gauge_length_m: f64, trig_frequency_hz: f64) -> Option<f32> {
    let f = rate_factor_f64(gauge_length_m, trig_frequency_hz)? as f32;
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

/// Strain rate in place, with `tier` clamped to what this CPU runs: `data`
/// holds whole lines of `prev.len()` samples; every line `n` (every sample,
/// or its even lanes) becomes `(line[n] − line[n−1]) × factor`, line `−1`
/// being `prev`, and on return `prev` holds `data`'s last line as it was
/// stored. Returns the tier used.
///
/// # Panics
/// When `prev` is empty while `data` is not, or `data` is not a whole number
/// of lines (the binding checks both before calling).
pub fn diff_scale_f32(tier: Tier, data: &mut [f32], prev: &mut [f32], factor: f32, even_only: bool) -> Tier {
    check_lines(data.len(), prev.len());
    let tier = clamp(tier);
    match tier {
        Tier::Scalar => diff_scalar_f32(data, prev, factor, even_only),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX2 only when the CPU runs it.
        Tier::Avx2 => unsafe { x86::diff_avx2_f32(data, prev, factor, even_only) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX-512 only when the CPU runs AVX-512F.
        Tier::Avx512 => unsafe { x86::diff_avx512_f32(data, prev, factor, even_only) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => diff_scalar_f32(data, prev, factor, even_only),
    }
    tier
}

/// [`diff_scale_f32`] for `f64` samples.
pub fn diff_scale_f64(tier: Tier, data: &mut [f64], prev: &mut [f64], factor: f64, even_only: bool) -> Tier {
    check_lines(data.len(), prev.len());
    let tier = clamp(tier);
    match tier {
        Tier::Scalar => diff_scalar_f64(data, prev, factor, even_only),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX2 only when the CPU runs it.
        Tier::Avx2 => unsafe { x86::diff_avx2_f64(data, prev, factor, even_only) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `clamp` returned AVX-512 only when the CPU runs AVX-512F.
        Tier::Avx512 => unsafe { x86::diff_avx512_f64(data, prev, factor, even_only) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => diff_scalar_f64(data, prev, factor, even_only),
    }
    tier
}

fn check_lines(samples: usize, line: usize) {
    assert!(
        samples == 0 || (line > 0 && samples.is_multiple_of(line)),
        "{samples} samples are not whole lines of {line}"
    );
}

/// The reference, one line: per converted lane one subtraction then one
/// multiply, `prev` taking every lane of the line as it was stored.
fn diff_line_scalar_f32(line: &mut [f32], prev: &mut [f32], factor: f32, even_only: bool) {
    if even_only {
        for (x, p) in line.chunks_mut(2).zip(prev.chunks_mut(2)) {
            let v = x[0];
            x[0] = (v - p[0]) * factor;
            p[0] = v;
            if let (Some(&a), Some(q)) = (x.get(1), p.get_mut(1)) {
                *q = a;
            }
        }
    } else {
        for (x, p) in line.iter_mut().zip(prev.iter_mut()) {
            let v = *x;
            *x = (v - *p) * factor;
            *p = v;
        }
    }
}

fn diff_line_scalar_f64(line: &mut [f64], prev: &mut [f64], factor: f64, even_only: bool) {
    if even_only {
        for (x, p) in line.chunks_mut(2).zip(prev.chunks_mut(2)) {
            let v = x[0];
            x[0] = (v - p[0]) * factor;
            p[0] = v;
            if let (Some(&a), Some(q)) = (x.get(1), p.get_mut(1)) {
                *q = a;
            }
        }
    } else {
        for (x, p) in line.iter_mut().zip(prev.iter_mut()) {
            let v = *x;
            *x = (v - *p) * factor;
            *p = v;
        }
    }
}

fn diff_scalar_f32(data: &mut [f32], prev: &mut [f32], factor: f32, even_only: bool) {
    let n = prev.len();
    if n == 0 {
        return;
    }
    for line in data.chunks_exact_mut(n) {
        diff_line_scalar_f32(line, prev, factor, even_only);
    }
}

fn diff_scalar_f64(data: &mut [f64], prev: &mut [f64], factor: f64, even_only: bool) {
    let n = prev.len();
    if n == 0 {
        return;
    }
    for line in data.chunks_exact_mut(n) {
        diff_line_scalar_f64(line, prev, factor, even_only);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! Explicit-intrinsic kernels. Each loads, multiplies and stores whole
    //! vectors (unaligned loads and stores: any start address), then hands
    //! the tail — fewer samples than a vector, starting on an even index
    //! since every vector width is even — to the scalar reference.

    use std::arch::x86_64::*;

    use super::{diff_line_scalar_f32, diff_line_scalar_f64, scalar_f32, scalar_f64};

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

    // ── Strain rate: `(x − prev) × factor`, `prev` taking `x` ──────────────
    //
    // Line by line: whole vectors of the line and of `prev` (unaligned loads
    // and stores), the stored samples written to `prev` and the converted
    // ones back to the line; the line's tail (fewer samples than a vector,
    // starting on an even index since every width is even) goes to the
    // scalar reference. The subtraction keeps `x` as its first operand and
    // the multiply the difference: the scalar loop's order, hence its NaN
    // payloads. Pairs: AVX-512 masks both operations to the even lanes and
    // keeps `x` in the odd ones; AVX2 computes every lane and blends the
    // untouched `x` back into the odd ones.

    /// # Safety
    /// The CPU must run AVX-512F; `data` is whole lines of `prev.len() > 0`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn diff_avx512_f32(data: &mut [f32], prev: &mut [f32], factor: f32, even_only: bool) {
        let f = _mm512_set1_ps(factor);
        for line in data.chunks_exact_mut(prev.len()) {
            let mut xs = line.chunks_exact_mut(16);
            let mut ps = prev.chunks_exact_mut(16);
            for (x, p) in (&mut xs).zip(&mut ps) {
                // SAFETY: `x` and `p` hold exactly 16 f32 each, in two
                // distinct buffers; loadu/storeu take any alignment.
                unsafe {
                    let a = _mm512_loadu_ps(x.as_ptr());
                    let b = _mm512_loadu_ps(p.as_ptr());
                    let r = if even_only {
                        let d = _mm512_mask_sub_ps(a, EVEN_F32X16, a, b);
                        _mm512_mask_mul_ps(a, EVEN_F32X16, d, f)
                    } else {
                        _mm512_mul_ps(_mm512_sub_ps(a, b), f)
                    };
                    _mm512_storeu_ps(p.as_mut_ptr(), a);
                    _mm512_storeu_ps(x.as_mut_ptr(), r);
                }
            }
            diff_line_scalar_f32(xs.into_remainder(), ps.into_remainder(), factor, even_only);
        }
    }

    /// # Safety
    /// The CPU must run AVX2; `data` is whole lines of `prev.len() > 0`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn diff_avx2_f32(data: &mut [f32], prev: &mut [f32], factor: f32, even_only: bool) {
        let f = _mm256_set1_ps(factor);
        for line in data.chunks_exact_mut(prev.len()) {
            let mut xs = line.chunks_exact_mut(8);
            let mut ps = prev.chunks_exact_mut(8);
            for (x, p) in (&mut xs).zip(&mut ps) {
                // SAFETY: `x` and `p` hold exactly 8 f32 each, in two
                // distinct buffers; loadu/storeu take any alignment.
                unsafe {
                    let a = _mm256_loadu_ps(x.as_ptr());
                    let b = _mm256_loadu_ps(p.as_ptr());
                    let m = _mm256_mul_ps(_mm256_sub_ps(a, b), f);
                    let r = if even_only { _mm256_blend_ps::<EVEN_F32X8>(a, m) } else { m };
                    _mm256_storeu_ps(p.as_mut_ptr(), a);
                    _mm256_storeu_ps(x.as_mut_ptr(), r);
                }
            }
            diff_line_scalar_f32(xs.into_remainder(), ps.into_remainder(), factor, even_only);
        }
    }

    /// # Safety
    /// The CPU must run AVX-512F; `data` is whole lines of `prev.len() > 0`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn diff_avx512_f64(data: &mut [f64], prev: &mut [f64], factor: f64, even_only: bool) {
        let f = _mm512_set1_pd(factor);
        for line in data.chunks_exact_mut(prev.len()) {
            let mut xs = line.chunks_exact_mut(8);
            let mut ps = prev.chunks_exact_mut(8);
            for (x, p) in (&mut xs).zip(&mut ps) {
                // SAFETY: `x` and `p` hold exactly 8 f64 each, in two
                // distinct buffers; loadu/storeu take any alignment.
                unsafe {
                    let a = _mm512_loadu_pd(x.as_ptr());
                    let b = _mm512_loadu_pd(p.as_ptr());
                    let r = if even_only {
                        let d = _mm512_mask_sub_pd(a, EVEN_F64X8, a, b);
                        _mm512_mask_mul_pd(a, EVEN_F64X8, d, f)
                    } else {
                        _mm512_mul_pd(_mm512_sub_pd(a, b), f)
                    };
                    _mm512_storeu_pd(p.as_mut_ptr(), a);
                    _mm512_storeu_pd(x.as_mut_ptr(), r);
                }
            }
            diff_line_scalar_f64(xs.into_remainder(), ps.into_remainder(), factor, even_only);
        }
    }

    /// # Safety
    /// The CPU must run AVX2; `data` is whole lines of `prev.len() > 0`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn diff_avx2_f64(data: &mut [f64], prev: &mut [f64], factor: f64, even_only: bool) {
        let f = _mm256_set1_pd(factor);
        for line in data.chunks_exact_mut(prev.len()) {
            let mut xs = line.chunks_exact_mut(4);
            let mut ps = prev.chunks_exact_mut(4);
            for (x, p) in (&mut xs).zip(&mut ps) {
                // SAFETY: `x` and `p` hold exactly 4 f64 each, in two
                // distinct buffers; loadu/storeu take any alignment.
                unsafe {
                    let a = _mm256_loadu_pd(x.as_ptr());
                    let b = _mm256_loadu_pd(p.as_ptr());
                    let m = _mm256_mul_pd(_mm256_sub_pd(a, b), f);
                    let r = if even_only { _mm256_blend_pd::<EVEN_F64X4>(a, m) } else { m };
                    _mm256_storeu_pd(p.as_mut_ptr(), a);
                    _mm256_storeu_pd(x.as_mut_ptr(), r);
                }
            }
            diff_line_scalar_f64(xs.into_remainder(), ps.into_remainder(), factor, even_only);
        }
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
        assert_eq!(NANOSTRAIN_PER_RAD_METRE, 110.37);
        assert_eq!(factor_f64(10.0), Some(110.37 / 10.0));
        assert_eq!(factor_f32(9.6), Some((110.37f64 / 9.6) as f32));
        assert_eq!(factor_f32(1.6), Some((110.37f64 / 1.6) as f32));
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

    // ── Strain rate ────────────────────────────────────────────────────────

    /// 75 kHz asked of a 250 MHz clock: the card fires at 250 MHz / 3 333.
    const INEXACT_HZ: f64 = 250e6 / 3_333.0;
    /// Line rates of real files (an inexact card rate among them).
    const RATES: [f64; 4] = [50.0, 2_000.0, INEXACT_HZ, 100_160.0];

    #[test]
    fn the_rate_factor_is_the_strain_factor_times_the_line_rate_computed_once() {
        assert_eq!(rate_factor_f64(10.0, 2_000.0), Some(110.37 / 10.0 * 2_000.0));
        assert_eq!(rate_factor_f32(9.6, 75_007.5), Some((110.37f64 / 9.6 * 75_007.5) as f32));
        for &hz in &RATES {
            for g in [0.4, 1.6, 9.6, 20.4] {
                assert_eq!(rate_factor_f64(g, hz), Some(factor_f64(g).unwrap() * hz));
                assert_eq!(rate_factor_f32(g, hz), Some((factor_f64(g).unwrap() * hz) as f32));
            }
        }
        for hz in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(rate_factor_f64(9.6, hz), None, "{hz}");
            assert_eq!(rate_factor_f32(9.6, hz), None, "{hz}");
        }
        for g in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(rate_factor_f64(g, 2_000.0), None, "{g}");
        }
        // Finite in f64, beyond f32; beyond f64.
        assert!(rate_factor_f64(1e-30, 1e10).is_some());
        assert_eq!(rate_factor_f32(1e-30, 1e10), None);
        assert_eq!(rate_factor_f64(1e-300, 1e300), None);
    }

    #[test]
    fn the_scalar_strain_rate_is_one_subtraction_then_one_multiply() {
        // Two lines of 3, after a line `prev`.
        let mut prev = vec![1.0f32, 2.0, 4.0];
        let mut v = vec![2.0f32, 2.0, 1.0, 5.0, 0.0, 1.0];
        diff_scalar_f32(&mut v, &mut prev, 0.5, false);
        assert_eq!(v, [0.5, 0.0, -1.5, 1.5, -1.0, 0.0]);
        assert_eq!(prev, [5.0, 0.0, 1.0], "the last line as stored");
        // Pairs: the phase lanes only; `prev` takes every lane.
        let mut prev = vec![1.0f32, 9.0];
        let mut v = vec![3.0f32, 7.0, 6.0, 8.0];
        diff_scalar_f32(&mut v, &mut prev, 2.0, true);
        assert_eq!(v, [4.0, 7.0, 6.0, 8.0]);
        assert_eq!(prev, [6.0, 8.0]);
        // A NaN line before the first: its phase lanes are NaN.
        let mut prev = vec![f32::NAN; 4];
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        diff_scalar_f32(&mut v, &mut prev, 2.0, true);
        assert!(v[0].is_nan() && v[2].is_nan());
        assert_eq!((v[1], v[3]), (2.0, 4.0));
        assert_eq!(prev, [1.0, 2.0, 3.0, 4.0]);
        // Not `x·f − p·f`: the difference is rounded once, then scaled.
        let (x, p, f) = (1.000_000_1f32, 1.0f32, 3.0f32);
        let mut prev = vec![p];
        let mut v = vec![x];
        diff_scalar_f32(&mut v, &mut prev, f, false);
        assert_eq!(v[0].to_bits(), ((x - p) * f).to_bits());
        // An empty block changes nothing.
        let mut prev = vec![7.0f64; 3];
        diff_scalar_f64(&mut [], &mut prev, 2.0, false);
        assert_eq!(prev, [7.0; 3]);
    }

    /// `data` and `prev` from `base` and `base_prev`, each starting `off`
    /// samples in (unaligned starts), run with `tier`; returns both.
    fn run_f32(tier: Tier, base: &[f32], base_prev: &[f32], off: usize, factor: f32, even: bool) -> (Vec<f32>, Vec<f32>) {
        let (mut d, mut p) = (base.to_vec(), base_prev.to_vec());
        let used = diff_scale_f32(tier, &mut d[off..], &mut p[off..], factor, even);
        assert_eq!(used, clamp(tier));
        assert_eq!(bits32(&d[..off]), bits32(&base[..off]), "before the block");
        assert_eq!(bits32(&p[..off]), bits32(&base_prev[..off]), "before prev");
        (d.split_off(off), p.split_off(off))
    }

    fn run_f64(tier: Tier, base: &[f64], base_prev: &[f64], off: usize, factor: f64, even: bool) -> (Vec<f64>, Vec<f64>) {
        let (mut d, mut p) = (base.to_vec(), base_prev.to_vec());
        diff_scale_f64(tier, &mut d[off..], &mut p[off..], factor, even);
        (d.split_off(off), p.split_off(off))
    }

    /// Rate factors of real files, and ones that push the products to
    /// overflow and to subnormals.
    fn rate_factors() -> Vec<f64> {
        let mut f: Vec<f64> = [(1.6, 2_000.0), (9.6, INEXACT_HZ), (20.4, 100_160.0)]
            .iter()
            .map(|&(g, hz)| rate_factor_f64(g, hz).unwrap())
            .collect();
        f.extend([1.0, 3.0e38, 1.0e-39, -2.5]);
        f
    }

    #[test]
    fn every_tier_gives_the_scalar_strain_rate_bits_on_every_length_and_offset() {
        let mut rng = Rng(0x51ed_270b_c3a1_9e4d);
        for &factor in &rate_factors() {
            let f32f = factor as f32;
            for line in 1..=200usize {
                for rows in [0usize, 1, 3] {
                    let n = rows * line;
                    let base = samples_f32(&mut rng, n + 3);
                    let base_prev = samples_f32(&mut rng, line + 3);
                    let base64 = samples_f64(&mut rng, n + 3);
                    let base_prev64 = samples_f64(&mut rng, line + 3);
                    for off in 0..3 {
                        for even in [false, true] {
                            if even && line % 2 == 1 {
                                continue; // pairs come in whole lines of pairs
                            }
                            let (want, want_p) = run_f32(Tier::Scalar, &base[..off + n], &base_prev[..off + line], off, f32f, even);
                            let (want64, want_p64) = run_f64(Tier::Scalar, &base64[..off + n], &base_prev64[..off + line], off, factor, even);
                            for tier in Tier::ALL {
                                let (got, got_p) = run_f32(tier, &base[..off + n], &base_prev[..off + line], off, f32f, even);
                                assert_eq!(bits32(&got), bits32(&want), "{tier:?} f32 line {line} rows {rows} off {off} even {even} f {factor}");
                                assert_eq!(bits32(&got_p), bits32(&want_p), "{tier:?} f32 prev line {line} rows {rows}");
                                let (got64, got_p64) = run_f64(tier, &base64[..off + n], &base_prev64[..off + line], off, factor, even);
                                assert_eq!(bits64(&got64), bits64(&want64), "{tier:?} f64 line {line} rows {rows} off {off} even {even} f {factor}");
                                assert_eq!(bits64(&got_p64), bits64(&want_p64), "{tier:?} f64 prev line {line} rows {rows}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_scalar_strain_rate_matches_the_formula_lane_by_lane() {
        let mut rng = Rng(3);
        for (line, even) in [(37usize, false), (38, true), (64, false)] {
            let rows = 5;
            let src = samples_f32(&mut rng, rows * line);
            let first = samples_f32(&mut rng, line);
            let f = rate_factor_f32(9.6, 2_000.0).unwrap();
            let (got, last) = run_f32(Tier::Scalar, &src, &first, 0, f, even);
            for r in 0..rows {
                for i in 0..line {
                    let x = src[r * line + i];
                    let p = if r == 0 { first[i] } else { src[(r - 1) * line + i] };
                    let want = if even && i % 2 == 1 { x } else { (x - p) * f };
                    assert_eq!(got[r * line + i].to_bits(), want.to_bits(), "row {r} lane {i}");
                }
            }
            assert_eq!(bits32(&last), bits32(&src[(rows - 1) * line..]));
        }
    }

    #[test]
    fn every_tier_gives_the_scalar_strain_rate_bits_on_a_large_buffer() {
        let mut rng = Rng(42);
        for (line, rows, even) in [(1031usize, 1017usize, false), (2 * 517, 1013, true), ((1 << 20) + 13, 2, false)] {
            let base = samples_f32(&mut rng, line * rows);
            let prev = samples_f32(&mut rng, line);
            let base64 = samples_f64(&mut rng, line * rows);
            let prev64 = samples_f64(&mut rng, line);
            let factor = rate_factor_f64(9.6, INEXACT_HZ).unwrap();
            let (want, want_p) = run_f32(Tier::Scalar, &base, &prev, 0, factor as f32, even);
            let (want64, want_p64) = run_f64(Tier::Scalar, &base64, &prev64, 0, factor, even);
            for tier in Tier::ALL {
                let (got, got_p) = run_f32(tier, &base, &prev, 0, factor as f32, even);
                assert_eq!(bits32(&got), bits32(&want), "{tier:?} {line}x{rows}");
                assert_eq!(bits32(&got_p), bits32(&want_p), "{tier:?} {line}x{rows} prev");
                let (got64, got_p64) = run_f64(tier, &base64, &prev64, 0, factor, even);
                assert_eq!(bits64(&got64), bits64(&want64), "{tier:?} f64 {line}x{rows}");
                assert_eq!(bits64(&got_p64), bits64(&want_p64), "{tier:?} f64 {line}x{rows} prev");
            }
        }
    }

    #[test]
    fn blocks_of_any_size_give_the_bits_of_one_block() {
        let mut rng = Rng(11);
        let factor = rate_factor_f32(9.6, 2_000.0).unwrap();
        for (line, even) in [(1usize, false), (17, false), (40, true), (130, true)] {
            let rows = 23;
            let src = samples_f32(&mut rng, rows * line);
            let first = vec![f32::NAN; line];
            let (whole, whole_p) = run_f32(Tier::Scalar, &src, &first, 0, factor, even);
            for tier in Tier::ALL {
                for block in 1..=rows {
                    let mut data = src.clone();
                    let mut prev = first.clone();
                    for chunk in data.chunks_mut(block * line) {
                        diff_scale_f32(tier, chunk, &mut prev, factor, even);
                    }
                    assert_eq!(bits32(&data), bits32(&whole), "{tier:?} line {line} blocks of {block}");
                    assert_eq!(bits32(&prev), bits32(&whole_p), "{tier:?} line {line} blocks of {block}: prev");
                }
            }
            // The first line follows a NaN line: its phase lanes are NaN.
            for (i, x) in whole[..line].iter().enumerate() {
                assert!(if even && i % 2 == 1 { x.to_bits() == src[i].to_bits() } else { x.is_nan() }, "lane {i}");
            }
        }
    }

    #[test]
    fn the_odd_lanes_of_pairs_keep_their_bits_in_the_strain_rate() {
        let mut rng = Rng(7);
        for tier in Tier::ALL {
            let line = 68;
            let mut v = samples_f32(&mut rng, 3 * line);
            for x in v.iter_mut().skip(1).step_by(2) {
                *x = f32::from_bits(0x7f80_0001); // a signalling NaN
            }
            let before = bits32(&v);
            let mut prev = samples_f32(&mut rng, line);
            diff_scale_f32(tier, &mut v, &mut prev, 11.5, true);
            for i in (1..v.len()).step_by(2) {
                assert_eq!(v[i].to_bits(), before[i], "{tier:?} lane {i}");
            }
            assert_eq!(bits32(&prev), before[2 * line..], "{tier:?}: prev is the last line as stored");
        }
    }

    #[test]
    #[should_panic(expected = "not whole lines")]
    fn a_block_that_is_not_whole_lines_is_refused() {
        diff_scale_f32(Tier::Scalar, &mut [1.0; 5], &mut [0.0; 2], 1.0, false);
    }

    #[test]
    #[should_panic(expected = "not whole lines")]
    fn a_block_without_a_line_length_is_refused() {
        diff_scale_f64(Tier::Scalar, &mut [1.0; 4], &mut [], 1.0, false);
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

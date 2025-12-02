use crate::{State, STATE_LEN};
use core::convert::TryFrom;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Ring dimension n.
pub(crate) const N: usize = 64;
/// Number of columns m.
pub(crate) const M: usize = 16;
/// Prime modulus p.
pub(crate) const P: u16 = 257;
/// A 128th root of unity modulo 257 (order exactly 128).
pub(crate) const OMEGA: u16 = 42;
/// Lightweight field element wrapper for `F_257`.
///
/// Provides modular addition/multiplication helpers without branching on feature
/// flags at call sites.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldElement(pub u16);

impl FieldElement {
    pub const P: u16 = P;
    /// R = 2^16 for Montgomery reduction.
    const R_BITS: u32 = 16;
    const R_MASK: u32 = (1u32 << Self::R_BITS) - 1;
    /// -P^{-1} mod R for P = 257, R = 2^16.
    const N_PRIME: u16 = 255;
    pub const BYTES: usize = 2;
    pub const MAX: u16 = P - 1;
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub const fn new(v: u16) -> Self {
        Self(v % P)
    }

    #[inline]
    pub const fn value(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn is_canonical(v: u16) -> bool {
        v < P
    }

    #[inline]
    pub const fn from_canonical(v: u16) -> Option<Self> {
        if Self::is_canonical(v) {
            Some(Self(v))
        } else {
            None
        }
    }

    #[allow(dead_code)] // handy for future algebraic ops
    pub fn pow(self, mut exp: u16) -> Self {
        let mut base = self;
        let mut acc = FieldElement::ONE;

        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    #[allow(dead_code)] // handy for future algebraic ops
    pub fn inv(self) -> Self {
        // Fermat's little theorem: a^(p-2) mod p for prime p.
        self.pow(P - 2)
    }

    /// Montgomery reduction for 16-bit modulus P = 257 with R = 2^16.
    #[inline]
    pub const fn montyred(x: u32) -> u16 {
        // m = (x * n') mod R
        let m = (x.wrapping_mul(Self::N_PRIME as u32)) & Self::R_MASK;

        // t = (x + m * P) / R
        let t = (x.wrapping_add(m * (Self::P as u32))) >> Self::R_BITS;
        let t16 = t as u16;

        if t16 >= Self::P {
            t16 - Self::P
        } else {
            t16
        }
    }
}

impl From<u16> for FieldElement {
    #[inline]
    fn from(v: u16) -> Self {
        FieldElement::new(v)
    }
}

impl From<FieldElement> for u16 {
    #[inline]
    fn from(fe: FieldElement) -> Self {
        fe.0
    }
}

impl Add for FieldElement {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 + rhs.0;
        if sum >= Self::P {
            FieldElement(sum - P)
        } else {
            FieldElement(sum)
        }
    }
}

impl AddAssign for FieldElement {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for FieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let prod = FieldElement::montyred(u32::from(self.0) * u32::from(rhs.0));
        FieldElement(prod)
    }
}

impl MulAssign for FieldElement {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sub for FieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 >= rhs.0 {
            FieldElement(self.0 - rhs.0)
        } else {
            FieldElement(self.0 + P - rhs.0)
        }
    }
}

impl SubAssign for FieldElement {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for FieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            FieldElement::ZERO
        } else {
            FieldElement(P - self.0)
        }
    }
}

const POW_TABLE: [u16; 128] = precompute_powers();
const TWIDDLE: [[u16; N]; N] = precompute_twiddle();

#[allow(clippy::cast_possible_truncation)] // indices are bounded by construction
const fn precompute_powers() -> [u16; 128] {
    let mut table = [1u16; 128];
    let mut i = 1;
    while i < 128 {
        table[i] = mul_mod257_const_int(table[i - 1], OMEGA);
        i += 1;
    }
    table
}

#[allow(clippy::cast_possible_truncation)] // i,k are always < 64
const fn precompute_twiddle() -> [[u16; N]; N] {
    let mut table = [[0u16; N]; N];
    let mut i = 0;
    while i < N {
        let factor = (2 * (i as u32)) + 1;
        let mut k = 0;
        while k < N {
            let exponent = factor * (k as u32);
            table[i][k] = pow_omega_const(exponent);
            k += 1;
        }
        i += 1;
    }
    table
}

#[allow(clippy::cast_possible_truncation)] // bounded by P = 257
const fn mul_mod257_const_int(a: u16, b: u16) -> u16 {
    ((a as u32 * b as u32) % (P as u32)) as u16
}

#[cfg(feature = "ark-ntt")]
#[inline]
fn mul_mod257_const(a: u16, b: u16) -> u16 {
    let prod = coeff_to_field(a) * coeff_to_field(b);
    field_to_coeff(&prod)
}

#[cfg(not(feature = "ark-ntt"))]
#[allow(dead_code)]
#[inline]
fn mul_mod257_const(a: u16, b: u16) -> u16 {
    mul_mod257_const_int(a, b)
}

#[cfg(feature = "ark-ntt")]
#[inline]
pub(crate) fn add_mod257(a: u16, b: u16) -> u16 {
    let sum = coeff_to_field(a) + coeff_to_field(b);
    field_to_coeff(&sum)
}

#[cfg(not(feature = "ark-ntt"))]
#[inline]
pub(crate) fn add_mod257(a: u16, b: u16) -> u16 {
    (FieldElement::from(a) + FieldElement::from(b)).value()
}

#[cfg(feature = "ark-ntt")]
#[inline]
pub(crate) fn mul_mod257(a: u16, b: u16) -> u16 {
    let prod = coeff_to_field(a) * coeff_to_field(b);
    field_to_coeff(&prod)
}

#[cfg(not(feature = "ark-ntt"))]
#[inline]
pub(crate) fn mul_mod257(a: u16, b: u16) -> u16 {
    (FieldElement::from(a) * FieldElement::from(b)).value()
}

/// Compute OMEGA^exp mod 257 using the precomputed table.
#[inline]
#[allow(dead_code)] // used in tests and for debugging
pub(crate) fn pow_omega(exp: u32) -> u16 {
    POW_TABLE[exp_index(exp)]
}

#[inline]
#[allow(dead_code)] // paired with pow_omega
fn exp_index(exp: u32) -> usize {
    usize::try_from(exp & 0x7F).expect("exp reduced to <128")
}

/// Compute OMEGA^exp mod 257 (const-friendly for build-time tables).
const fn pow_omega_const(exp: u32) -> u16 {
    let mut e = exp & 0x7F;
    let mut result: u16 = 1;
    let mut base: u16 = OMEGA;

    while e > 0 {
        if (e & 1) != 0 {
            result = mul_mod257_const_int(result, base);
        }
        base = mul_mod257_const_int(base, base);
        e >>= 1;
    }

    result
}

/// Compute F(x) for a 64-bit vector x ∈ {0,1}^64 using a precomputed twiddle matrix.
///
/// Formula (paper):
///   F(x)_i = Σ_{k=0}^{63} `x_k` · ω^{(2i+1)k}  (mod 257)
pub(crate) fn transform(bits: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];

    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc: u16 = 0;
        let twiddle_row = &TWIDDLE[i];

        for (bit, &omega_pow) in bits.iter().zip(twiddle_row.iter()) {
            debug_assert!(*bit <= 1, "transform bits must be 0/1");
            if *bit == 1 {
                acc = add_mod257(acc, omega_pow);
            }
        }

        *out_i = acc;
    }

    out
}

/// Encode 64 coefficients in `Z_257` into the 72-byte digest format.
///
/// Layout:
///   bytes[0..64):  low 8 bits of each coefficient `z_i`.
///   bytes[64..72): high bit (bit 8) of each `z_i`, packed LSB-first.
pub(crate) fn encode_state(coeffs: &[u16; N]) -> State {
    let mut bytes = [0u8; STATE_LEN];

    // Low 8 bits.
    for (i, coeff) in coeffs.iter().enumerate() {
        debug_assert!(*coeff <= 256, "coefficient must be in 0..=256");
        bytes[i] = (coeff & 0xFF) as u8;
    }

    // High bit (bit 8) packed into the last 8 bytes.
    for (i, coeff) in coeffs.iter().enumerate() {
        let hi = (coeff >> 8) & 1;
        let byte_index = 64 + (i / 8); // 64..71
        let bit_in_byte = i % 8;

        bytes[byte_index] |= (hi as u8) << bit_in_byte;
    }

    State(bytes)
}

// Optional arkworks-backed FFT/NTT helpers for benchmarking and experimentation.
// The SWIFFT core does not rely on these, but they provide a convenient,
// well-tested NTT for F_257 when the `ark-ntt` feature is enabled.
#[cfg(feature = "ark-ntt")]
mod ark_fft {
    use ark_ff::{fields::Fp64, MontBackend, MontConfig, PrimeField};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    #[derive(MontConfig)]
    #[modulus = "257"]
    #[generator = "3"] // generator of order 256 in F_257
    pub struct F257Config;

    pub type F257 = Fp64<MontBackend<F257Config, 1>>;

    /// Two-adicity for F_257 (257 - 1 = 2^8).
    pub const TWO_ADICITY: u64 = 8;

    /// In-place forward NTT using arkworks domains (requires power-of-two len ≤ 256).
    pub fn ntt_in_place(values: &mut [F257]) {
        let domain = GeneralEvaluationDomain::<F257>::new(values.len())
            .expect("NTT length must be compatible with two-adicity 8");
        domain.fft(values);
    }

    /// In-place inverse NTT using arkworks domains (requires power-of-two len ≤ 256).
    pub fn intt_in_place(values: &mut [F257]) {
        let domain = GeneralEvaluationDomain::<F257>::new(values.len())
            .expect("NTT length must be compatible with two-adicity 8");
        domain.ifft(values);
    }

    /// Convert a coefficient (0..=256) into the F_257 element type.
    #[inline]
    pub fn coeff_to_field(c: u16) -> F257 {
        F257::from(c as u64)
    }

    /// Convert an F_257 element back into a coefficient (0..=256).
    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn field_to_coeff(f: &F257) -> u16 {
        // into_bigint is canonical; still mask to be safe.
        let limbs = f.into_bigint().0;
        (limbs[0] % (super::P as u64)) as u16
    }
}

#[cfg(feature = "ark-ntt")]
pub use ark_fft::{
    coeff_to_field, field_to_coeff, intt_in_place, ntt_in_place, F257Config,
    F257, TWO_ADICITY,
};

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn field_element_add_sub_mul_basic() {
        let a = FieldElement::new(200);
        let b = FieldElement::new(100);
        assert_eq!((a + b).value(), 43); // 300 mod 257
        assert_eq!((a - b).value(), 100);
        assert_eq!((b - a).value(), 157); // wrap
        assert_eq!((a * b).value(), (200u32 * 100u32 % 257) as u16);
    }

    proptest! {
        #[test]
        fn field_element_pow_matches_naive(a in 0u16..=256, e in 0u16..=512) {
            let a_fe = FieldElement::new(a);
            let fast = a_fe.pow(e).value();
            let mut naive = 1u32;
            for _ in 0..e {
                naive = (naive * a as u32) % (P as u32);
            }
            prop_assert_eq!(fast, naive as u16);
        }

        #[test]
        fn montyred_matches_mod_reduce(x in 0u32..100_000u32) {
            let reduced = FieldElement::montyred(x);
            prop_assert!(reduced < P);
            prop_assert_eq!(reduced as u32, x % (P as u32));
        }

        #[test]
        fn neg_and_inv_properties(a in 1u16..=256) { // avoid zero for inverse
            let fe = FieldElement::new(a);
            prop_assert_eq!((fe + (-fe)).value(), 0);
            let inv = fe.inv();
            prop_assert_eq!((fe * inv).value(), 1);
        }
    }
}

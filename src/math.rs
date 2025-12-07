use crate::field_element::FieldElement;
#[cfg(test)]
use crate::State;

/// Ring dimension n.
pub(crate) const N: usize = 64;
/// Number of columns m.
pub(crate) const M: usize = 16;
/// A 128th root of unity modulo 257 (order exactly 128).
pub(crate) const OMEGA: u16 = 42;

/// Lazy runtime table for ω^k.
fn pow_table() -> &'static [FieldElement; 128] {
    use std::sync::OnceLock;

    static TABLE: OnceLock<[FieldElement; 128]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [FieldElement::ONE; 128];
        let omega = FieldElement::from(OMEGA);
        let mut i = 1;
        while i < 128 {
            table[i] = table[i - 1] * omega;
            i += 1;
        }
        table
    })
}

#[inline]
fn pow_table_get(exp: u32) -> FieldElement {
    pow_table()[exp_index(exp)]
}

/// Compute OMEGA^exp mod 257 using the precomputed table.
#[inline]
#[allow(dead_code)] // used in tests and for debugging
pub(crate) fn pow_omega(exp: u32) -> u16 {
    pow_table_get(exp).value()
}

#[inline]
#[allow(dead_code)] // paired with pow_omega
fn exp_index(exp: u32) -> usize {
    reduce_exp(exp)
}

#[inline]
const fn reduce_exp(exp: u32) -> usize {
    // Reduce modulo 128 (order of omega) before casting to usize.
    (exp & 0x7F) as usize
}

/// Lazy twiddle matrix in `FieldElement` form to avoid per-call conversions.
#[allow(clippy::cast_possible_truncation)] // i,k < N=64 so casts to u32 are safe
fn twiddle() -> &'static [[FieldElement; N]; N] {
    use std::sync::OnceLock;

    static TWIDDLE: OnceLock<[[FieldElement; N]; N]> = OnceLock::new();
    TWIDDLE.get_or_init(|| {
        let mut table = [[FieldElement::ZERO; N]; N];
        let mut i = 0;
        while i < N {
            let factor = (2 * (i as u32)) + 1;
            let mut k = 0;
            while k < N {
                let exponent = factor * (k as u32);
                table[i][k] = pow_table_get(exponent);
                k += 1;
            }
            i += 1;
        }
        table
    })
}

/// Compute F(x) for a 64-bit vector x ∈ {0,1}^64 using a precomputed twiddle matrix.
///
/// Formula (paper):
///   F(x)_i = Σ_{k=0}^{63} `x_k` · ω^{(2i+1)k}  (mod 257)
pub(crate) fn transform(bits: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];

    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = FieldElement::ZERO;
        let twiddle_row = &twiddle()[i];

        for (bit, &omega_pow) in bits.iter().zip(twiddle_row.iter()) {
            debug_assert!(*bit <= 1, "transform bits must be 0/1");
            if *bit == 1 {
                acc += omega_pow;
            }
        }

        *out_i = acc.value();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow_mod(base: u32, mut exp: u32, modulus: u32) -> u16 {
        let mut result: u32 = 1;
        let mut b = base % modulus;
        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * b) % modulus;
            }
            b = (b * b) % modulus;
            exp >>= 1;
        }
        result as u16
    }

    #[test]
    fn pow_omega_matches_naive_pow() {
        for exp in 0..256u32 {
            let expected = pow_mod(OMEGA as u32, exp, FieldElement::P as u32);
            assert_eq!(pow_omega(exp), expected, "exp {}", exp);
        }
    }

    fn naive_transform(bits: &[u8; N]) -> [u16; N] {
        let mut out = [0u16; N];

        for (i, out_i) in out.iter_mut().enumerate() {
            let mut acc = 0u32;
            let factor = 2 * (i as u32) + 1;

            for (k, &bit) in bits.iter().enumerate() {
                if bit & 1 == 0 {
                    continue;
                }
                let exponent = factor * (k as u32);
                let w = pow_mod(OMEGA as u32, exponent, FieldElement::P as u32)
                    as u32;
                acc = (acc + w) % (FieldElement::P as u32);
            }

            *out_i = acc as u16;
        }

        out
    }

    #[test]
    fn transform_matches_naive() {
        let mut bits = [0u8; N];
        for idx in [0usize, 1, 5, 13, 21, 63] {
            bits[idx] = 1;
        }

        let fast = transform(&bits);
        let expected = naive_transform(&bits);
        assert_eq!(fast, expected);
    }

    #[test]
    fn encode_state_packs_high_bits() {
        let mut coeffs = [0u16; N];
        coeffs[0] = 256;
        coeffs[7] = 256;
        coeffs[8] = 256;
        coeffs[10] = 255;
        coeffs[63] = 256;

        let mut encoded = State::default();
        encoded.encode(&coeffs);

        assert_eq!(encoded.0[0], 0);
        assert_eq!(encoded.0[10], 255);
        assert_eq!(encoded.0[64], 0b1000_0001);
        assert_eq!(encoded.0[65], 0b0000_0001);
        assert_eq!(encoded.0[71], 0b1000_0000);
        assert_eq!(&encoded.0[66..71], &[0, 0, 0, 0, 0]);
    }
}

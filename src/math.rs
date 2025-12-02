use crate::{State, STATE_LEN};
use core::convert::TryFrom;

/// Ring dimension n.
pub(crate) const N: usize = 64;
/// Number of columns m.
pub(crate) const M: usize = 16;
/// Prime modulus p.
pub(crate) const P: u16 = 257;
/// A 128th root of unity modulo 257 (order exactly 128).
pub(crate) const OMEGA: u16 = 42;

pub(crate) type Coeff = u16;

const POW_TABLE: [Coeff; 128] = precompute_powers();

const fn precompute_powers() -> [Coeff; 128] {
    let mut table = [1u16; 128];
    let mut i = 1;
    while i < 128 {
        table[i] = mul_mod257_const(table[i - 1], OMEGA);
        i += 1;
    }
    table
}

#[allow(clippy::cast_possible_truncation)] // bounded by P = 257
const fn mul_mod257_const(a: Coeff, b: Coeff) -> Coeff {
    ((a as u32 * b as u32) % (P as u32)) as Coeff
}

#[inline]
pub(crate) fn add_mod257(a: Coeff, b: Coeff) -> Coeff {
    let sum = a + b;
    if sum >= P {
        sum - P
    } else {
        sum
    }
}

#[inline]
pub(crate) fn mul_mod257(a: Coeff, b: Coeff) -> Coeff {
    mul_mod257_const(a, b)
}

/// Compute OMEGA^exp mod 257 using the precomputed table.
#[inline]
pub(crate) fn pow_omega(exp: u32) -> Coeff {
    let idx = (exp & 0x7F) as usize;
    POW_TABLE[idx]
}

/// Compute F(x) for a 64-bit vector x ∈ {0,1}^64.
///
/// Formula (paper):
///   F(x)_i = Σ_{k=0}^{63} `x_k` · ω^{(2i+1)k}  (mod 257)
pub(crate) fn transform(bits: &[u8; N]) -> [Coeff; N] {
    let mut out = [0u16; N];

    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc: Coeff = 0;
        let factor = 2 * u32::try_from(i).expect("i < N") + 1; // (2i + 1)

        for (k, &bit) in bits.iter().enumerate() {
            debug_assert!(bit <= 1, "transform bits must be 0/1");
            if bit & 1 == 0 {
                continue;
            }

            let exponent = factor * u32::try_from(k).expect("k < N");
            let w = pow_omega(exponent);
            acc = add_mod257(acc, w);
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
pub(crate) fn encode_state(coeffs: &[Coeff; N]) -> State {
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

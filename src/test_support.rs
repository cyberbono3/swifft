use crate::{
    field_element::FieldElement,
    math::{N, OMEGA},
};

/// Fast modular exponentiation helper used in tests.
pub(crate) fn pow_mod(mut base: u32, mut exp: u32, modulus: u32) -> u16 {
    base %= modulus;
    let mut result: u32 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }
    result as u16
}

/// Straightforward reference implementation of the column transform.
pub(crate) fn naive_transform(bits: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];

    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0u32;
        let factor = 2 * (i as u32) + 1;

        for (k, &bit) in bits.iter().enumerate() {
            if bit & 1 == 0 {
                continue;
            }
            let exponent = factor * (k as u32);
            let w =
                pow_mod(OMEGA as u32, exponent, FieldElement::P as u32) as u32;
            acc = (acc + w) % (FieldElement::P as u32);
        }

        *out_i = acc as u16;
    }

    out
}

use crate::{
    math::{add_mod257, encode_state, mul_mod257, transform, Coeff, M, N},
    Block, Key, State, BLOCK_LEN, STATE_LEN,
};

const MSG_LEN: usize = STATE_LEN + BLOCK_LEN;
type Message = [u8; MSG_LEN];

/// Core SWIFFT compression function.
///
/// Conceptually:
/// 1. Form a 1024-bit message from `state || block`.
/// 2. View it as 16 vectors `x_j` ∈ {0,1}^64 (columns).
/// 3. Apply the FFT-based map F on each `x_j`:
///    F(x)_i = `Σ_k` `x_k` · ω^{(2i+1)k} mod 257
/// 4. Combine columns:
///    `z_i` = `Σ_j` `a_{i,j}` · y_{j,i} mod 257
///    where a_{i,j} comes from the key.
/// 5. Encode the 64 coefficients `z_i` ∈ {0,…,256} into 72 bytes.
pub fn compress(key: &Key, state: &mut State, block: &Block) {
    // 1. Build the 128-byte message buffer.
    let msg = assemble_message(state, block);

    // 2–3. Compute y[j][i] = F(x_j)_i.
    //
    // Here:
    //   - j = column index   (0..15)
    //   - i = output index   (0..63)
    let mut y = [[Coeff::default(); N]; M];

    for (j, y_col) in y.iter_mut().enumerate() {
        let bits = extract_column_bits(&msg, j);
        *y_col = transform(&bits);
    }

    // 4. Linear combination across columns: z[i] = Σ_j a_{i,j} * y[j][i] mod 257.
    //
    // Key layout:
    //   key.0[j * N + i]  ≙  a_{i,j}
    let mut z = [Coeff::default(); N];

    for (i, z_i) in z.iter_mut().enumerate() {
        let mut acc: u16 = 0;

        for (j, y_col) in y.iter().enumerate() {
            let a_ij = u16::from(key.0[j * N + i]);
            let term = mul_mod257(a_ij, y_col[i]);
            acc = add_mod257(acc, term);
        }

        *z_i = acc; // In [0, 256].
    }

    // 5. Encode back into the 72-byte state buffer.
    *state = encode_state(&z);
}

fn assemble_message(state: &State, block: &Block) -> Message {
    let mut msg = [0u8; MSG_LEN];
    msg[..STATE_LEN].copy_from_slice(&state.0);
    msg[STATE_LEN..].copy_from_slice(&block.0);
    msg
}

fn extract_column_bits(msg: &Message, column: usize) -> [u8; N] {
    debug_assert!(column < M, "column index out of range");

    let mut bits = [0u8; N];
    for (k, bit) in bits.iter_mut().enumerate() {
        let bit_index = column * N + k; // 0..1023
        let byte_index = bit_index / 8;
        let bit_in_byte = bit_index % 8;

        // Little-endian within each byte: bit 0 is least-significant.
        let b = (msg[byte_index] >> bit_in_byte) & 1;
        *bit = b;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::compress;
    use crate::{
        math::{self, pow_omega, Coeff, M, N, OMEGA, P},
        Block, Key, State, BLOCK_LEN, KEY_LEN, STATE_LEN,
    };

    mod helpers {
        use super::*;

        pub fn pow_mod(mut base: u32, mut exp: u32, modulus: u32) -> Coeff {
            base %= modulus;
            let mut result: u32 = 1;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = (result * base) % modulus;
                }
                base = (base * base) % modulus;
                exp >>= 1;
            }
            result as Coeff
        }

        pub fn naive_transform(bits: &[u8; N]) -> [u16; N] {
            let mut out = [0u16; N];

            for (i, out_i) in out.iter_mut().enumerate() {
                let mut acc = 0u32;
                let factor = 2 * (i as u32) + 1;

                for (k, &bit) in bits.iter().enumerate() {
                    if bit & 1 == 0 {
                        continue;
                    }
                    let exponent = factor * (k as u32);
                    let w = pow_mod(OMEGA as u32, exponent, P as u32) as u32;
                    acc = (acc + w) % (P as u32);
                }

                *out_i = acc as u16;
            }

            out
        }

        pub fn naive_encode(coeffs: &[u16; N]) -> State {
            math::encode_state(coeffs)
        }

        pub fn reference_compress(
            key: &Key,
            state: &State,
            block: &Block,
        ) -> State {
            let msg = super::super::assemble_message(state, block);

            let mut y = [[0u16; N]; M];

            for (j, y_col) in y.iter_mut().enumerate() {
                let bits = super::super::extract_column_bits(&msg, j);
                *y_col = naive_transform(&bits);
            }

            let mut z = [0u16; N];

            for (i, z_i) in z.iter_mut().enumerate() {
                let mut acc = 0u32;

                for (j, y_col) in y.iter().enumerate() {
                    let a_ij = key.0[j * N + i] as u32;
                    let term = (a_ij * y_col[i] as u32) % (P as u32);
                    acc = (acc + term) % (P as u32);
                }

                *z_i = acc as u16;
            }

            naive_encode(&z)
        }
    }

    #[test]
    fn add_mod257_wraps() {
        assert_eq!(math::add_mod257(200, 100), 43);
        assert_eq!(math::add_mod257(256, 1), 0);
        assert_eq!(math::add_mod257(256, 256), 255);
    }

    #[test]
    fn mul_mod257_basic() {
        assert_eq!(math::mul_mod257(30, 30), 129);
        assert_eq!(math::mul_mod257(256, 2), 255);
        assert_eq!(math::mul_mod257(0, 123), 0);
    }

    #[test]
    fn pow_omega_respects_order() {
        assert_eq!(pow_omega(0), 1);
        assert_eq!(pow_omega(1), OMEGA);
        assert_eq!(pow_omega(128), 1);
        assert_eq!(pow_omega(129), OMEGA);
    }

    #[test]
    fn transform_matches_naive_reference() {
        let mut bits = [0u8; N];
        for idx in [0usize, 1, 5, 13, 21, 63] {
            bits[idx] = 1;
        }

        let fast = math::transform(&bits);
        let expected = helpers::naive_transform(&bits);
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

        let encoded = math::encode_state(&coeffs);

        assert_eq!(encoded.0[0], 0);
        assert_eq!(encoded.0[10], 255);
        assert_eq!(encoded.0[64], 0b1000_0001);
        assert_eq!(encoded.0[65], 0b0000_0001);
        assert_eq!(encoded.0[71], 0b1000_0000);
        assert_eq!(&encoded.0[66..71], &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn compress_with_zero_key_clears_state() {
        let key = Key([0u8; KEY_LEN]);
        let mut state = State([0xAA; STATE_LEN]);
        let block = Block([0x55; BLOCK_LEN]);

        compress(&key, &mut state, &block);

        assert_eq!(state.0, [0u8; STATE_LEN]);
    }

    #[test]
    fn compress_matches_reference_implementation() {
        let mut key = Key([0u8; KEY_LEN]);
        for (i, byte) in key.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(3).wrapping_add(5);
        }

        let mut state_input = State([0u8; STATE_LEN]);
        for (i, byte) in state_input.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7).wrapping_add(1);
        }

        let mut block = Block([0u8; BLOCK_LEN]);
        for (i, byte) in block.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(13).wrapping_add(3);
        }

        let mut fast = state_input.clone();
        compress(&key, &mut fast, &block);

        let expected = helpers::reference_compress(&key, &state_input, &block);
        assert_eq!(fast, expected);
    }
}

use crate::{
    fe,
    field_element::FieldElement,
    math::{transform, M, N},
    BLOCK_LEN, KEY_LEN, STATE_LEN,
};
use core::convert::TryFrom;

/// SWIFFT key: 1024-byte vector interpreted as coefficients in `Z_257`.
///
/// Logically this is a 16×64 matrix (a_{i,j}) with entries in {0,…,256},
/// stored as bytes 0..255 (mod 257).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key(pub [u8; KEY_LEN]);

impl Key {
    /// Borrow the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }

    /// Consume and return the inner array.
    #[must_use]
    pub const fn into_inner(self) -> [u8; KEY_LEN] {
        self.0
    }
}

impl Default for Key {
    fn default() -> Self {
        Self([0u8; KEY_LEN])
    }
}

impl From<[u8; KEY_LEN]> for Key {
    fn from(bytes: [u8; KEY_LEN]) -> Self {
        Self(bytes)
    }
}

impl TryFrom<&[u8]> for Key {
    type Error = core::array::TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        value.try_into().map(Self)
    }
}

/// 72-byte chaining state / digest encoding.
///
/// First 64 bytes: low 8 bits of each coefficient.
/// Last 8 bytes: one extra bit per coefficient (bit 8), packed LSB-first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State(pub [u8; STATE_LEN]);

impl State {
    /// Borrow the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; STATE_LEN] {
        &self.0
    }

    /// Consume and return the inner array.
    #[must_use]
    pub const fn into_inner(self) -> [u8; STATE_LEN] {
        self.0
    }

    pub(crate) fn encode(&mut self, coeffs: &[u16; N]) {
        self.0 = [0u8; STATE_LEN];

        // Low 8 bits.
        for (i, coeff) in coeffs.iter().enumerate() {
            debug_assert!(*coeff <= 256, "coefficient must be in 0..=256");
            self.0[i] = (coeff & 0xFF) as u8;
        }

        // High bit (bit 8) packed into the last 8 bytes.
        for (i, coeff) in coeffs.iter().enumerate() {
            let hi = (coeff >> 8) & 1;
            let byte_index = 64 + (i / 8); // 64..71
            let bit_in_byte = i % 8;

            self.0[byte_index] |= (hi as u8) << bit_in_byte;
        }
    }

    /// Core SWIFFT compression in-place on this state.
    pub fn compress(&mut self, key: &Key, block: &Block) {
        // 1. Build the 128-byte message buffer.
        let msg = self.assemble_message(block);

        // 2–3. Compute y[j][i] = F(x_j)_i.
        //
        // Here:
        //   - j = column index   (0..15)
        //   - i = output index   (0..63)
        let y: [[FieldElement; N]; M] = core::array::from_fn(|j| {
            let bits = msg.extract_column_bits(j);
            transform(&bits).map(FieldElement::from)
        });

        // 4. Linear combination across columns: z[i] = Σ_j a_{i,j} * y[j][i] mod 257.
        //
        // Key layout:
        //   key.0[j * N + i]  ≙  a_{i,j}
        let z: [FieldElement; N] = core::array::from_fn(|i| {
            y.iter()
                .enumerate()
                .fold(FieldElement::ZERO, |acc, (j, y_col)| {
                    let a_ij = fe!(u16::from(key.0[j * N + i]));
                    acc + a_ij * y_col[i]
                })
        });

        // 5. Encode back into the 72-byte state buffer.
        let z_u16: [u16; N] = z.map(FieldElement::value);
        self.encode(&z_u16);
    }

    pub(crate) fn assemble_message(&self, block: &Block) -> Message {
        let mut msg = [0u8; MSG_LEN];
        msg[..STATE_LEN].copy_from_slice(&self.0);
        msg[STATE_LEN..].copy_from_slice(&block.0);
        Message(msg)
    }
}

impl Default for State {
    fn default() -> Self {
        Self([0u8; STATE_LEN])
    }
}

impl From<[u8; STATE_LEN]> for State {
    fn from(bytes: [u8; STATE_LEN]) -> Self {
        Self(bytes)
    }
}

impl TryFrom<&[u8]> for State {
    type Error = core::array::TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        value.try_into().map(Self)
    }
}

/// 56-byte message block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block(pub [u8; BLOCK_LEN]);

impl Block {
    /// Borrow the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; BLOCK_LEN] {
        &self.0
    }

    /// Consume and return the inner array.
    #[must_use]
    pub const fn into_inner(self) -> [u8; BLOCK_LEN] {
        self.0
    }
}

impl Default for Block {
    fn default() -> Self {
        Self([0u8; BLOCK_LEN])
    }
}

impl From<[u8; BLOCK_LEN]> for Block {
    fn from(bytes: [u8; BLOCK_LEN]) -> Self {
        Self(bytes)
    }
}

impl TryFrom<&[u8]> for Block {
    type Error = core::array::TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        value.try_into().map(Self)
    }
}

/// Trait abstraction for SWIFFT-like compressors.
pub trait Compressor {
    fn compress(&self, state: &mut State, block: &Block);
}

impl Compressor for Key {
    fn compress(&self, state: &mut State, block: &Block) {
        compress(self, state, block);
    }
}

const MSG_LEN: usize = STATE_LEN + BLOCK_LEN;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Message([u8; MSG_LEN]);

impl Message {
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub const fn as_bytes(&self) -> &[u8; MSG_LEN] {
        &self.0
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn into_inner(self) -> [u8; MSG_LEN] {
        self.0
    }

    pub(crate) fn extract_column_bits(&self, column: usize) -> [u8; N] {
        debug_assert!(column < M, "column index out of range");

        let mut bits = [0u8; N];
        for (k, bit) in bits.iter_mut().enumerate() {
            let bit_index = column * N + k; // 0..1023
            let byte_index = bit_index / 8;
            let bit_in_byte = bit_index % 8;

            // Little-endian within each byte: bit 0 is least-significant.
            let b = (self.0[byte_index] >> bit_in_byte) & 1;
            *bit = b;
        }
        bits
    }
}

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
    state.compress(key, block);
}

#[cfg(test)]
mod tests {
    use super::compress;
    use crate::{
        field_element::FieldElement,
        math::{self, fe_add, fe_mul, pow_omega, M, N, OMEGA},
        Block, Key, State, BLOCK_LEN, KEY_LEN, STATE_LEN,
    };
    use core::convert::TryFrom;

    mod helpers {
        use super::*;

        pub fn pow_mod(mut base: u32, mut exp: u32, modulus: u32) -> u16 {
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
                    let w =
                        pow_mod(OMEGA as u32, exponent, FieldElement::P as u32)
                            as u32;
                    acc = (acc + w) % (FieldElement::P as u32);
                }

                *out_i = acc as u16;
            }

            out
        }

        pub fn naive_encode(coeffs: &[u16; N]) -> State {
            let mut state = State::default();
            state.encode(coeffs);
            state
        }

        pub fn reference_compress(
            key: &Key,
            state: &State,
            block: &Block,
        ) -> State {
            let msg = state.assemble_message(block);

            let mut y = [[0u16; N]; M];

            for (j, y_col) in y.iter_mut().enumerate() {
                let bits = msg.extract_column_bits(j);
                *y_col = naive_transform(&bits);
            }

            let mut z = [0u16; N];

            for (i, z_i) in z.iter_mut().enumerate() {
                let mut acc = 0u32;

                for (j, y_col) in y.iter().enumerate() {
                    let a_ij = key.0[j * N + i] as u32;
                    let term =
                        (a_ij * y_col[i] as u32) % (FieldElement::P as u32);
                    acc = (acc + term) % (FieldElement::P as u32);
                }

                *z_i = acc as u16;
            }

            naive_encode(&z)
        }
    }

    #[test]
    fn fe_add_wraps() {
        assert_eq!(fe_add(200, 100), 43);
        assert_eq!(fe_add(256, 1), 0);
        assert_eq!(fe_add(256, 256), 255);
    }

    #[test]
    fn fe_mul_basic() {
        assert_eq!(fe_mul(30, 30), 129);
        assert_eq!(fe_mul(256, 2), 255);
        assert_eq!(fe_mul(0, 123), 0);
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

        let mut encoded = State::default();
        encoded.encode(&coeffs);

        assert_eq!(encoded.0[0], 0);
        assert_eq!(encoded.0[10], 255);
        assert_eq!(encoded.0[64], 0b1000_0001);
        assert_eq!(encoded.0[65], 0b0000_0001);
        assert_eq!(encoded.0[71], 0b1000_0000);
        assert_eq!(&encoded.0[66..71], &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn encode_overwrites_previous_contents() {
        let mut state = State([0xAA; STATE_LEN]);
        let mut coeffs = [0u16; N];
        coeffs[1] = 1;
        coeffs[63] = 256;

        state.encode(&coeffs);

        assert_eq!(state.0[0], 0);
        assert_eq!(state.0[1], 1);
        assert_eq!(state.0[63], 0);
        assert_eq!(state.0[71], 0b1000_0000); // high bit of coeff 63.
                                              // Middle bytes that were previously 0xAA should be reset to 0.
        assert_eq!(&state.0[10..20], &[0u8; 10]);
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

    #[test]
    fn state_method_matches_free_compress() {
        let mut key = Key([0u8; KEY_LEN]);
        for (i, byte) in key.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(11).wrapping_add(7);
        }

        let mut state_a = State::default();
        for (i, byte) in state_a.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(5).wrapping_add(9);
        }

        let mut state_b = state_a.clone();

        let mut block = Block::default();
        for (i, byte) in block.0.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
        }

        state_a.compress(&key, &block);
        compress(&key, &mut state_b, &block);

        assert_eq!(state_a, state_b);
    }

    #[test]
    fn assemble_message_concatenates_state_and_block() {
        let mut state_bytes = [0u8; STATE_LEN];
        let mut block_bytes = [0u8; BLOCK_LEN];

        for (i, b) in state_bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        for (i, b) in block_bytes.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(100);
        }

        let state = State::from(state_bytes);
        let block = Block::from(block_bytes);

        let msg = state.assemble_message(&block);
        assert_eq!(&msg.as_bytes()[..STATE_LEN], state_bytes.as_slice());
        assert_eq!(&msg.as_bytes()[STATE_LEN..], block_bytes.as_slice());
    }

    #[test]
    fn extract_column_bits_reads_from_state_and_block_regions() {
        let mut state_bytes = [0u8; STATE_LEN];
        let mut block_bytes = [0u8; BLOCK_LEN];

        // Column 0: first 8 bytes (state region).
        state_bytes[0] = 0b1010_0001; // bits 0,5,7 set.
                                      // Column 9 starts at byte index 72 (state_len), in block region.
        block_bytes[0] = 0b0101_0010; // bits 1,4,6 set for column 9.

        let state = State::from(state_bytes);
        let block = Block::from(block_bytes);
        let msg = state.assemble_message(&block);

        let col0 = msg.extract_column_bits(0);
        assert_eq!(col0[0], 1);
        assert_eq!(col0[1], 0);
        assert_eq!(col0[5], 1);
        assert_eq!(col0[7], 1);
        assert_eq!(col0[6], 0);

        let col9 = msg.extract_column_bits(9);
        assert_eq!(col9[1], 1);
        assert_eq!(col9[4], 1);
        assert_eq!(col9[6], 1);
        assert_eq!(col9[0], 0);
        assert_eq!(col9[2], 0);
    }

    #[test]
    fn constructors_cover_common_paths() {
        let key_bytes = [7u8; KEY_LEN];
        let state_bytes = [3u8; STATE_LEN];
        let block_bytes = [5u8; BLOCK_LEN];

        let key_from_array = Key::from(key_bytes);
        let state_from_array = State::from(state_bytes);
        let block_from_array = Block::from(block_bytes);

        assert_eq!(key_from_array.as_bytes(), &key_bytes);
        assert_eq!(state_from_array.as_bytes(), &state_bytes);
        assert_eq!(block_from_array.as_bytes(), &block_bytes);

        let key_try = Key::try_from(key_bytes.as_slice()).unwrap();
        let state_try = State::try_from(state_bytes.as_slice()).unwrap();
        let block_try = Block::try_from(block_bytes.as_slice()).unwrap();

        assert_eq!(key_try.into_inner(), key_bytes);
        assert_eq!(state_try.into_inner(), state_bytes);
        assert_eq!(block_try.into_inner(), block_bytes);

        let short = [0u8; KEY_LEN - 1];
        assert!(Key::try_from(short.as_slice()).is_err());
    }
}

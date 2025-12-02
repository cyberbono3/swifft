#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::all, clippy::pedantic))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    not(test),
    allow(
        clippy::module_name_repetitions,
        clippy::missing_panics_doc,
        clippy::missing_errors_doc
    )
)]
//! SWIFFT compression function
//!
//! This is a clean-room implementation based on the public description
//! of SWIFFT (n = 64, m = 16, p = 257). The crate is `no_std`-ready; enable
//! the `std` feature (default) if you want standard library support.

pub const KEY_LEN: usize = 1024; // 16 × 64 coefficients
pub const STATE_LEN: usize = 72; // 64 low bytes + 8 high-bit bytes
pub const BLOCK_LEN: usize = 56; // 72 + 56 = 128 bytes = 1024 bits

/// SWIFFT key: 1024-byte vector interpreted as coefficients in `Z_257`.
///
/// Logically this is a 16×64 matrix (a_{i,j}) with entries in {0,…,256},
/// stored as bytes 0..255 (mod 257).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key(pub [u8; KEY_LEN]);

/// 72-byte chaining state / digest encoding.
///
/// First 64 bytes: low 8 bits of each coefficient.
/// Last 8 bytes: one extra bit per coefficient (bit 8), packed LSB-first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State(pub [u8; STATE_LEN]);

/// 56-byte message block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block(pub [u8; BLOCK_LEN]);

pub mod core;
pub(crate) mod math;

/// Trait abstraction for SWIFFT-like compressors.
pub trait Compressor {
    fn compress(&self, state: &mut State, block: &Block);
}

impl Compressor for Key {
    fn compress(&self, state: &mut State, block: &Block) {
        core::compress(self, state, block);
    }
}

/// Convenience re-export of the compression function.
///
/// This is the main entry point you’ll typically call:
///
/// ```ignore
/// use swifft_rs::{Key, State, Block, compress};
///
/// let key = Key([0u8; 1024]);
/// let mut state = State([0u8; 72]);
/// let block = Block([0u8; 56]);
///
/// compress(&key, &mut state, &block);
/// ```
pub use crate::core::compress;

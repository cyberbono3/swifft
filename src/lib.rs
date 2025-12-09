#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::all, clippy::pedantic))]
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
//! of SWIFFT (n = 64, m = 16, p = 257).

pub const KEY_LEN: usize = 1024; // 16 × 64 coefficients
pub const STATE_LEN: usize = 72; // 64 low bytes + 8 high-bit bytes
pub const BLOCK_LEN: usize = 56; // 72 + 56 = 128 bytes = 1024 bits

pub mod field_element;
pub(crate) mod math;
pub mod pattern;
pub mod state;
#[cfg(test)]
pub(crate) mod test_support;

pub use crate::state::{Block, Key, State};

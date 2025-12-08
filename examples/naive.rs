use std::fmt::Write;

use swifft::{Block, Key, State, BLOCK_LEN, KEY_LEN};

fn main() {
    let key = demo_key();
    let block = demo_block();

    let mut state = State::default();
    compress_and_print(&mut state, &key, &block);
}

fn demo_key() -> Key {
    Key::from(patterned::<KEY_LEN>(11, 7))
}

fn demo_block() -> Block {
    Block::from(patterned::<BLOCK_LEN>(5, 3))
}

fn compress_and_print(state: &mut State, key: &Key, block: &Block) {
    state.compress(key, block);
    println!("SWIFFT digest (hex): {}", to_hex(state.as_bytes()));
}

fn patterned<const LEN: usize>(multiplier: u8, addend: u8) -> [u8; LEN] {
    let mut bytes = [0u8; LEN];
    fill_pattern(&mut bytes, multiplier, addend);
    bytes
}

fn fill_pattern(bytes: &mut [u8], multiplier: u8, addend: u8) {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(multiplier).wrapping_add(addend);
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{:02x}", b);
    }
    out
}

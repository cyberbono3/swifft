use std::fmt::Write;

use swifft::{pattern::patterned_bytes, Block, Key, State, BLOCK_LEN, KEY_LEN};

fn main() {
    let key = demo_key();
    let block = demo_block();

    let mut state = State::default();
    compress_and_print(&mut state, &key, &block);
}

fn demo_key() -> Key {
    Key::from(patterned_bytes::<KEY_LEN>(11, 7))
}

fn demo_block() -> Block {
    Block::from(patterned_bytes::<BLOCK_LEN>(5, 3))
}

fn compress_and_print(state: &mut State, key: &Key, block: &Block) {
    state.compress(key, block);
    println!("SWIFFT digest (hex): {}", to_hex(state.as_bytes()));
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{:02x}", b);
    }
    out
}

PoC [SWIFFT](https://cseweb.ucsd.edu/~vlyubash/papers/swifftfse.pdf) implementation (WIP) ![Tests](https://github.com/cyberbono3/swifft/actions/workflows/tests.yml/badge.svg)

- Clean-room Rust port of the n=64, m=16, p=257 parameters.
- Requires `std` for now (no `no_std` support).
- Optional feature flags reserved for future optimizations: `parallel`, `simd`.

## Building
Use `cargo`, the standard Rust build tool, to build the library:

```bash
git clone https://github.com/cyberbono3/swifft.git
cd swifft
cargo build --release
```

## Usage

```rust
use swifft::{Block, Key, State, BLOCK_LEN, KEY_LEN};

// Keys are 1024 bytes interpreted modulo 257 (0..=256).
let key_bytes = [0u8; KEY_LEN];
let key = Key::from(key_bytes);

// State is a 72-byte chaining value; initialize however you like.
let mut state = State::default();

// Blocks are exactly 56 bytes; you can build them from a byte array or slice.
let block_bytes = [0u8; BLOCK_LEN];
let block = Block::from(block_bytes);

// Compress in-place, writing the new state back into `state`.
state.compress(&key, &block);
```

## Project structure

- `src/lib.rs`: crate entry, constants, and public re-exports (`Block`, `Key`, `State`).
- `src/state.rs`: byte newtypes and the SWIFFT compression implementation.
- `src/math.rs`: power tables, twiddle factors, and the transform used by compression.
- `src/field_element.rs`: modular arithmetic over `F_257` used throughout the math layer.

## Invariants & notes

- `Key` entries are interpreted modulo 257 (range 0..=256).
- `Block` is exactly 56 bytes; `State` is exactly 72 bytes (low bytes then packed high bits).
- The implementation is `#![forbid(unsafe_code)]` and keeps arithmetic in `u16/u32` to avoid panics.
- Output compatibility is tested against a slow reference model in-unit tests; add official vectors once available.

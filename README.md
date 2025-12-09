 # swifft
 
 [![Tests](https://github.com/cyberbono3/swifft/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberbono3/swifft/actions/workflows/tests.yml)

 PoC [SWIFFT](https://cseweb.ucsd.edu/~vlyubash/papers/swifftfse.pdf)
 ## Disclamer: Do not use it in production. This is the experimental implementation.

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

## Usage (TODO test it)
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
## Example
There is a runnable version of this snippet in `examples/naive.rs`:

```bash
cargo run --example naive
```
It prints the resulting 72-byte SWIFFT digest in hex.
```
SWIFFT digest (hex): b4064d836d08c690362a48021db5960912bbf24d1e5f2239b19aebb459bbc974e0355afa89bc40ae93ed0064886c33bd3c40d36f3d6bb3eb9dfddb37c39f4d2b0000000000040000
```

## Project structure

- `src/lib.rs`: crate entry, constants, and public re-exports (`Block`, `Key`, `State`).
- `src/state.rs`: byte newtypes and the SWIFFT compression implementation.
- `src/math.rs`: power tables, twiddle factors, and the transform used by compression.
- `src/field_element.rs`: modular arithmetic over `F_257` used throughout the math layer.
- `src/test_support.rs`: naive reference helpers shared by tests to keep them DRY.

## Invariants & notes

- `Key` entries are interpreted modulo 257 (range 0..=256).
- `Block` is exactly 56 bytes; `State` is exactly 72 bytes (low bytes then packed high bits).
- The implementation is `#![forbid(unsafe_code)]` and keeps arithmetic in `u16/u32` to avoid panics.
- Output compatibility is tested against a shared naive reference model in-unit tests; add official vectors once available.

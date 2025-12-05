PoC [SWIFFT](https://cseweb.ucsd.edu/~vlyubash/papers/swifftfse.pdf) implementation (WIP)

- Clean-room Rust port of the n=64, m=16, p=257 parameters.
- Requires `std` for now (no `no_std` support).
- Optional feature flags reserved for future optimizations: `parallel`, `simd`.

## Usage

```rust
use swifft::{Block, Compressor, Key, State};

let key = Key::default(); // coefficients are bytes mod 257
let mut state = State::default();
let block = Block::default();

// Inherent method on State
state.compress(&key, &block);

// Or via the `Compressor` trait on Key
key.compress(&mut state, &block);
```

## Invariants & notes

- `Key` entries are interpreted modulo 257 (range 0..=256).
- `Block` is exactly 56 bytes; `State` is exactly 72 bytes (low bytes then packed high bits).
- The implementation is `#![forbid(unsafe_code)]` and keeps arithmetic in `u16/u32` to avoid panics.
- Output compatibility is tested against a slow reference model in-unit tests; add official vectors once available.

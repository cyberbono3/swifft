use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId,
    Criterion,
};
use swifft::{pattern::patterned_bytes, Block, Key, State, BLOCK_LEN, KEY_LEN};

fn bench_compress(c: &mut Criterion) {
    let key = Key(patterned_bytes::<KEY_LEN>(11, 7));
    let block = Block(patterned_bytes::<BLOCK_LEN>(5, 3));

    c.bench_function("compress_single_block", |b| {
        b.iter_batched(
            State::default,
            |mut state| {
                state.compress(&key, &block);
                black_box(state);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_with_input(
        BenchmarkId::new("compress_batch", 16usize),
        &16usize,
        |b, &batch_size| {
            let blocks: Vec<Block> = (0..batch_size)
                .map(|i| {
                    // Small variation per block to avoid unrealistically uniform data.
                    Block(patterned_bytes::<BLOCK_LEN>(
                        5 + (i as u8 % 3),
                        3 + (i as u8 % 5),
                    ))
                })
                .collect();

            b.iter_batched(
                State::default,
                |mut state| {
                    for blk in &blocks {
                        state.compress(&key, blk);
                    }
                    black_box(state);
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);

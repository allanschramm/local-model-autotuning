//! Criterion isolated speedup measurement for the search-state module.
//! Phase 0 Sprint 0.3 wires `mark_visited` + atomic persist. Until then
//! the bench scaffold keeps the harness buildable.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_scaffold(c: &mut Criterion) {
    c.bench_function("state::scaffold", |b| {
        b.iter(|| criterion::black_box(99u64))
    });
}

criterion_group!(state_group, bench_scaffold);
criterion_main!(state_group);

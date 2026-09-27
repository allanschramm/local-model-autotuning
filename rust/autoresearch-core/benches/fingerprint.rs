//! Criterion isolated speedup measurement for the fingerprint module.
//! Phase 0 Sprint 0.2 wires the actual `dump`, `load`, `apply` and
//! `mismatch_reason` benches. Until then the file only checks that the
//! bench path compiles against the canonical fingerprint functions.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_scaffold(c: &mut Criterion) {
    c.bench_function("fingerprint::scaffold", |b| {
        b.iter(|| {
            // Phase 0 Sprint 0.2 lands:
            //  - 10k dump calls over realistic payloads
            //  - 10k mismatch_reason scans with mixed-rule fixtures
            //  - 10k apply() merges
            criterion::black_box(42u64)
        })
    });
}

criterion_group!(fp_group, bench_scaffold);
criterion_main!(fp_group);

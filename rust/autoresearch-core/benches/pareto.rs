//! Criterion isolated speedup measurement for the pareto module.
//! Phase 0 Sprint 0.4 lands fingerprint_hash, dominates, pareto_set, merge.

use autoresearch_core::pareto::{
    dominates, fingerprint_hash, merge, pareto_set, ObjectiveVector, Trial,
};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_dominates(c: &mut Criterion) {
    let a = ObjectiveVector {
        ctx: Some(2048.0),
        tps: Some(100.0),
        agentic: Some(0.6),
        coding: Some(0.6),
    };
    let b = ObjectiveVector {
        ctx: Some(1024.0),
        tps: Some(50.0),
        agentic: Some(0.3),
        coding: Some(0.3),
    };
    c.bench_function("pareto::dominates", |bch| {
        bch.iter(|| dominates(&a, &b));
    });
}

fn bench_fingerprint_hash(c: &mut Criterion) {
    let engine = serde_json::json!({"threads": 8, "ctx": 2048, "kv": "q4_0"});
    let sampler = serde_json::json!({"top_k": 40, "temperature": 0.7});
    c.bench_function("pareto::fingerprint_hash", |bch| {
        bch.iter(|| fingerprint_hash(&engine, &sampler));
    });
}

fn bench_pareto_set(c: &mut Criterion) {
    let input: Vec<ObjectiveVector> = (0..1000)
        .map(|i| ObjectiveVector {
            ctx: Some(1024.0 + (i as f64) % 8.0),
            tps: Some(30.0 + (i % 100) as f64),
            agentic: Some((i % 11) as f64 / 10.0),
            coding: Some((i % 13) as f64 / 12.0),
        })
        .collect();
    c.bench_function("pareto::pareto_set::1k", |bch| {
        bch.iter(|| pareto_set(&input));
    });
}

fn bench_merge(c: &mut Criterion) {
    let trials: Vec<Trial> = (0..500)
        .map(|i| Trial {
            fp: format!("{:08x}", i / 4),
            vector: ObjectiveVector {
                ctx: Some(1024.0 + (i as f64) % 8.0),
                tps: Some(30.0 + (i % 100) as f64),
                agentic: None,
                coding: None,
            },
        })
        .collect();
    c.bench_function("pareto::merge::500", |bch| {
        bch.iter(|| merge(&trials));
    });
}

criterion_group!(
    name = pareto_group;
    config = Criterion::default().sample_size(50);
    targets = bench_dominates, bench_fingerprint_hash, bench_pareto_set, bench_merge
);
criterion_main!(pareto_group);

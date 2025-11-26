//! Benchmarks for bE-TES calculation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qualia_core::{
    betes::{BETESCalculator, BETESInput},
    config::{BETESWeights, BETESSettingsV31},
};

fn bench_betes_calculation(c: &mut Criterion) {
    let calculator = BETESCalculator::new(
        BETESWeights::default(),
        Some(BETESSettingsV31::default()),
    );
    
    let input = BETESInput {
        raw_mutation_score: 0.85,
        raw_emt_gain: 0.15,
        raw_assertion_iq: 4.0,
        raw_behaviour_coverage: 0.9,
        raw_median_test_time_ms: 50.0,
        raw_flakiness_rate: 0.05,
    };
    
    c.bench_function("betes_calculation", |b| {
        b.iter(|| {
            calculator.calculate(black_box(&input)).unwrap();
        });
    });
}

fn bench_normalization(c: &mut Criterion) {
    let calculator = BETESCalculator::new(
        BETESWeights::default(),
        Some(BETESSettingsV31::default()),
    );
    
    c.bench_function("normalize_mutation_score", |b| {
        b.iter(|| {
            calculator.normalize_mutation_score(black_box(0.85));
        });
    });
    
    c.bench_function("normalize_speed_factor", |b| {
        b.iter(|| {
            calculator.normalize_speed_factor(black_box(200.0));
        });
    });
}

criterion_group!(benches, bench_betes_calculation, bench_normalization);
criterion_main!(benches);

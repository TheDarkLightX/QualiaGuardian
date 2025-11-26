//! Integration tests for the core quality metrics system

use qualia_core::{
    betes::{BETESCalculator, BETESInput},
    config::{QualityConfig, QualityMode, BETESWeights, BETESSettingsV31},
    tes::{QualityCalculator, QualityInput, BETESMetrics},
    types::{QualityScore, RiskClass, QualityGrade},
    Result,
};

#[test]
fn test_end_to_end_betes_calculation() {
    let config = QualityConfig::builder()
        .mode(QualityMode::BETESv31)
        .build()
        .unwrap();
    
    let calculator = QualityCalculator::new(config);
    
    let input = QualityInput {
        betes_metrics: Some(BETESMetrics {
            raw_mutation_score: 0.85,
            raw_emt_gain: 0.15,
            raw_assertion_iq: 4.0,
            raw_behaviour_coverage: 0.9,
            raw_median_test_time_ms: 50.0,
            raw_flakiness_rate: 0.05,
        }),
        test_suite_data: None,
        codebase_data: None,
        previous_score: None,
        project_path: None,
        project_language: None,
    };
    
    let (score, output) = calculator.calculate(&input).unwrap();
    
    // Verify score is bounded
    assert!((0.0..=1.0).contains(&score.value()));
    
    // Verify output type
    match output {
        qualia_core::tes::QualityOutput::BETES(components) => {
            assert_eq!(components.raw_mutation_score, 0.85);
            assert!((0.0..=1.0).contains(&components.betes_score));
        }
        _ => panic!("Expected BETES output"),
    }
}

#[test]
fn test_risk_classification_workflow() {
    let score = QualityScore::new(0.85).unwrap();
    let risk_class = RiskClass::from_score(score.value());
    
    assert_eq!(risk_class, RiskClass::Financial);
    assert!(risk_class.meets_requirement(score.value()));
    
    // Test that lower scores don't meet requirements
    let lower_score = QualityScore::new(0.75).unwrap();
    assert!(!RiskClass::Aerospace.meets_requirement(lower_score.value()));
    assert!(RiskClass::Enterprise.meets_requirement(lower_score.value()));
}

#[test]
fn test_grade_assignment() {
    let scores = vec![
        (0.95, QualityGrade::APlus),
        (0.85, QualityGrade::A),
        (0.75, QualityGrade::B),
        (0.65, QualityGrade::C),
        (0.50, QualityGrade::F),
    ];
    
    for (score_value, expected_grade) in scores {
        let grade = QualityGrade::from_score(score_value);
        assert_eq!(grade, expected_grade);
    }
}

#[test]
fn test_betes_with_different_weights() {
    let weights = BETESWeights {
        w_m: 2.0,
        w_e: 1.0,
        w_a: 1.0,
        w_b: 1.0,
        w_s: 1.0,
    };
    
    let calculator = BETESCalculator::new(
        weights,
        Some(BETESSettingsV31::default()),
    );
    
    let input = BETESInput {
        raw_mutation_score: 0.9,
        raw_emt_gain: 0.1,
        raw_assertion_iq: 3.0,
        raw_behaviour_coverage: 0.8,
        raw_median_test_time_ms: 100.0,
        raw_flakiness_rate: 0.1,
    };
    
    let (score, components) = calculator.calculate(&input).unwrap();
    
    // Higher weight on mutation score should give higher overall score
    // when mutation score is high
    assert!(score.value() > 0.7);
    assert_eq!(components.applied_weights.unwrap().w_m, 2.0);
}

#[test]
fn test_sigmoid_vs_minmax_normalization() {
    let input = BETESInput {
        raw_mutation_score: 0.775, // Center of sigmoid
        raw_emt_gain: 0.125, // Center of sigmoid
        raw_assertion_iq: 3.0,
        raw_behaviour_coverage: 0.8,
        raw_median_test_time_ms: 100.0,
        raw_flakiness_rate: 0.05,
    };
    
    // With sigmoid
    let calc_sigmoid = BETESCalculator::new(
        BETESWeights::default(),
        Some(BETESSettingsV31 {
            smooth_m: true,
            smooth_e: true,
            k_m: 10.0,
            k_e: 10.0,
        }),
    );
    
    // Without sigmoid
    let calc_minmax = BETESCalculator::new(
        BETESWeights::default(),
        None,
    );
    
    let (score_sigmoid, _) = calc_sigmoid.calculate(&input).unwrap();
    let (score_minmax, _) = calc_minmax.calculate(&input).unwrap();
    
    // Both should be valid scores
    assert!((0.0..=1.0).contains(&score_sigmoid.value()));
    assert!((0.0..=1.0).contains(&score_minmax.value()));
    
    // They may differ but both should be reasonable
    let diff = (score_sigmoid.value() - score_minmax.value()).abs();
    assert!(diff < 0.5); // Shouldn't differ by more than 0.5
}

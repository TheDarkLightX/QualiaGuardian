//! Analyze command implementation

use std::path::PathBuf;
use std::collections::HashMap;
use anyhow::Result;
use tracing::{info, warn, error};
use indicatif::{ProgressBar, ProgressStyle};
use qualia_core::{
    QualityConfig, QualityMode, QualityScore,
    config::{BETESWeights, BETESSettingsV31, OSQIWeights},
    tes::{QualityCalculator, QualityInput, BETESMetrics, QualityOutput},
};
use qualia_sensors::{
    SensorContext, SensorRegistry, SensorExecutor,
    create_default_registry,
};
use crate::{OutputFormat, output::format_quality_output};

/// Run the analyze command
pub async fn run(
    path: PathBuf,
    run_quality: bool,
    quality_mode: &str,
    risk_class: Option<&str>,
    sensors: Option<Vec<String>>,
    format: OutputFormat,
) -> Result<()> {
    // Parse quality mode
    let mode = match quality_mode {
        "etes_v2" => QualityMode::ETESv2,
        "betes_v3" => QualityMode::BETESv3,
        "betes_v3.1" => QualityMode::BETESv31,
        "osqi_v1" => QualityMode::OSQIv1,
        _ => {
            error!("Invalid quality mode: {}", quality_mode);
            return Err(anyhow::anyhow!("Invalid quality mode"));
        }
    };
    
    // Create quality configuration
    let config = QualityConfig {
        mode,
        risk_class: risk_class.and_then(|s| parse_risk_class(s)),
        betes_weights: BETESWeights::default(),
        betes_v3_1_settings: Some(BETESSettingsV31::default()),
        osqi_weights: OSQIWeights::default(),
        chs_thresholds_path: Some("config/chs_thresholds.yml".to_string()),
    };
    
    if run_quality {
        info!("Running quality analysis with mode: {:?}", mode);
        
        // Create progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")?
                .progress_chars("#>-")
        );
        
        // Collect sensor data
        pb.set_message("Collecting sensor data...");
        let sensor_data = collect_sensors(&path, sensors.as_deref()).await?;
        pb.inc(50);
        
        // Calculate quality score
        pb.set_message("Calculating quality score...");
        let input = prepare_quality_input(&sensor_data, &path);
        let calculator = QualityCalculator::new(config);
        let (score, output) = calculator.calculate(&input)?;
        pb.inc(50);
        pb.finish_with_message("Analysis complete!");
        
        // Format and display output
        format_quality_output(score, output, format)?;
        
        // Show risk class evaluation if specified
        if let Some(risk) = risk_class {
            evaluate_risk_class(score.value(), risk)?;
        }
    } else {
        // Just run sensors without quality calculation
        info!("Running sensor analysis...");
        let sensor_data = collect_sensors(&path, sensors.as_deref()).await?;
        
        // Display sensor results
        for (name, result) in sensor_data {
            match result {
                Ok(value) => {
                    info!("Sensor {}: {:?}", name, value);
                }
                Err(e) => {
                    error!("Sensor {} failed: {}", name, e);
                }
            }
        }
    }
    
    Ok(())
}

/// Collect data from sensors
async fn collect_sensors(
    project_path: &PathBuf,
    sensor_filter: Option<&[String]>,
) -> Result<HashMap<String, qualia_sensors::Result<serde_json::Value>>> {
    let registry = create_default_registry();
    let executor = SensorExecutor::new(registry);
    
    let context = SensorContext::new(
        project_path.to_string_lossy(),
        "rust", // TODO: Detect language
    );
    
    // Determine which sensors to run
    let all_sensors = vec![
        "mutation",
        "assertion_iq",
        "behavior_coverage",
        "speed",
        "flakiness",
        "chs",
        "security",
        "arch",
    ];
    
    let sensors_to_run = if let Some(filter) = sensor_filter {
        filter.iter()
            .map(|s| s.as_str())
            .filter(|s| all_sensors.contains(s))
            .collect::<Vec<_>>()
    } else {
        all_sensors
    };
    
    info!("Running sensors: {:?}", sensors_to_run);
    
    Ok(executor.execute_all(&context, &sensors_to_run).await)
}

/// Prepare input for quality calculation
fn prepare_quality_input(
    sensor_data: &HashMap<String, qualia_sensors::Result<serde_json::Value>>,
    project_path: &PathBuf,
) -> QualityInput {
    // Extract bE-TES metrics from sensor data
    let mut betes_metrics = BETESMetrics {
        raw_mutation_score: 0.0,
        raw_emt_gain: 0.0,
        raw_assertion_iq: 1.0,
        raw_behaviour_coverage: 0.0,
        raw_median_test_time_ms: 1000.0,
        raw_flakiness_rate: 0.0,
    };
    
    // Parse mutation sensor data
    if let Some(Ok(mutation_data)) = sensor_data.get("mutation") {
        if let Some(score) = mutation_data.get("mutation_score").and_then(|v| v.as_f64()) {
            betes_metrics.raw_mutation_score = score;
        }
    }
    
    // Parse assertion IQ data
    if let Some(Ok(aiq_data)) = sensor_data.get("assertion_iq") {
        if let Some(mean_iq) = aiq_data.get("mean_iq").and_then(|v| v.as_f64()) {
            betes_metrics.raw_assertion_iq = mean_iq;
        }
    }
    
    // Parse behavior coverage data
    if let Some(Ok(cov_data)) = sensor_data.get("behavior_coverage") {
        if let Some(ratio) = cov_data.get("coverage_ratio").and_then(|v| v.as_f64()) {
            betes_metrics.raw_behaviour_coverage = ratio;
        }
    }
    
    // Parse speed data
    if let Some(Ok(speed_data)) = sensor_data.get("speed") {
        if let Some(median) = speed_data.get("median_test_time_ms").and_then(|v| v.as_f64()) {
            betes_metrics.raw_median_test_time_ms = median;
        }
    }
    
    // Parse flakiness data
    if let Some(Ok(flake_data)) = sensor_data.get("flakiness") {
        if let Some(rate) = flake_data.get("flakiness_rate").and_then(|v| v.as_f64()) {
            betes_metrics.raw_flakiness_rate = rate;
        }
    }
    
    QualityInput {
        betes_metrics: Some(betes_metrics),
        test_suite_data: None,
        codebase_data: None,
        previous_score: None,
        project_path: Some(project_path.to_string_lossy().to_string()),
        project_language: Some("rust".to_string()),
    }
}

/// Parse risk class from string
fn parse_risk_class(s: &str) -> Option<qualia_core::types::RiskClass> {
    use qualia_core::types::RiskClass;
    
    match s.to_lowercase().as_str() {
        "aerospace" => Some(RiskClass::Aerospace),
        "medical" => Some(RiskClass::Medical),
        "financial" => Some(RiskClass::Financial),
        "enterprise" => Some(RiskClass::Enterprise),
        "standard" => Some(RiskClass::Standard),
        "prototype" => Some(RiskClass::Prototype),
        "experimental" => Some(RiskClass::Experimental),
        _ => None,
    }
}

/// Evaluate score against risk class
fn evaluate_risk_class(score: f64, risk_class: &str) -> Result<()> {
    use qualia_core::types::RiskClass;
    
    if let Some(risk) = parse_risk_class(risk_class) {
        let threshold = risk.min_score();
        
        if score >= threshold {
            info!("✅ PASS: Score {:.3} meets {} threshold {:.3}", score, risk_class, threshold);
        } else {
            warn!("❌ FAIL: Score {:.3} does not meet {} threshold {:.3}", score, risk_class, threshold);
        }
    } else {
        error!("Unknown risk class: {}", risk_class);
    }
    
    Ok(())
}
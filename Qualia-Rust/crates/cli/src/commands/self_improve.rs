//! Self-improve command implementation

use anyhow::Result;
use tracing::{info, warn};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use colored::Colorize;
use crossterm::style::Stylize;

use qualia_core::{
    QualityConfig, QualityScore,
    tes::calculate_quality,
};
use qualia_analytics::trends::RiskLevel;
use qualia_sensors::{
    SensorContext,
    mutation::MutationSensor,
    assertion_iq::AssertionIQSensor,
    behaviour_coverage::BehaviorCoverageSensor,
    speed::SpeedSensor,
    flakiness::FlakinessSensor,
    chs::CHSSensor,
    security::SecuritySensor,
    arch::ArchSensor,
};

use crate::OutputFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelfAnalysisResult {
    project: String,
    overall_score: f64,
    risk_class: RiskLevel,
    component_scores: ComponentScores,
    recommendations: Vec<Recommendation>,
    auto_fixable: Vec<AutoFix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentScores {
    mutation_score: f64,
    assertion_iq: f64,
    behavior_coverage: f64,
    speed_score: f64,
    stability_score: f64,
    code_health: f64,
    security_score: f64,
    architecture_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Recommendation {
    category: String,
    priority: Priority,
    description: String,
    impact: f64,
    effort: Effort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AutoFix {
    file: String,
    line: Option<usize>,
    issue: String,
    fix_description: String,
    code_diff: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Effort {
    Trivial,
    Small,
    Medium,
    Large,
}

/// Run self-improvement analysis
pub async fn run(
    dry_run: bool,
    max_improvements: usize,
    format: OutputFormat,
) -> Result<()> {
    info!("Running self-improvement analysis");
    info!("Dry run: {}", dry_run);
    info!("Max improvements: {}", max_improvements);
    
    // Find Guardian's own source directory
    let guardian_root = find_guardian_root()?;
    let src_dir = guardian_root.join("src");
    let test_dir = guardian_root.join("tests");
    
    if !src_dir.exists() {
        // Try crates structure
        let crates_dir = guardian_root.join("crates");
        if crates_dir.exists() {
            return analyze_workspace(&crates_dir, &test_dir, dry_run, max_improvements, format).await;
        } else {
            anyhow::bail!("Could not find Guardian source directory");
        }
    }
    
    // Analyze single crate
    analyze_crate(&src_dir, &test_dir, dry_run, max_improvements, format).await
}

async fn analyze_workspace(
    crates_dir: &Path,
    _test_dir: &Path,
    dry_run: bool,
    max_improvements: usize,
    format: OutputFormat,
) -> Result<()> {
    info!("Analyzing Guardian workspace at: {}", crates_dir.display());
    
    let mut all_results = Vec::new();
    
    // Analyze each crate
    for entry in std::fs::read_dir(crates_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            let crate_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            
            let src_dir = path.join("src");
            let test_dir = path.join("tests");
            
            if src_dir.exists() {
                info!("Analyzing crate: {}", crate_name);
                
                match analyze_single_crate(&src_dir, &test_dir, crate_name).await {
                    Ok(result) => all_results.push(result),
                    Err(e) => warn!("Failed to analyze crate {}: {}", crate_name, e),
                }
            }
        }
    }
    
    if all_results.is_empty() {
        anyhow::bail!("No crates analyzed successfully");
    }
    
    // Aggregate results
    let aggregate_result = aggregate_results(all_results);
    
    // Display results
    display_results(&aggregate_result, format)?;
    
    // Apply fixes if not dry run
    if !dry_run && !aggregate_result.auto_fixable.is_empty() {
        apply_auto_fixes(&aggregate_result.auto_fixable, max_improvements)?;
    }
    
    Ok(())
}

async fn analyze_crate(
    src_dir: &Path,
    test_dir: &Path,
    dry_run: bool,
    max_improvements: usize,
    format: OutputFormat,
) -> Result<()> {
    let result = analyze_single_crate(src_dir, test_dir, "guardian").await?;
    
    display_results(&result, format)?;
    
    if !dry_run && !result.auto_fixable.is_empty() {
        apply_auto_fixes(&result.auto_fixable, max_improvements)?;
    }
    
    Ok(())
}

async fn analyze_single_crate(
    src_dir: &Path,
    test_dir: &Path,
    crate_name: &str,
) -> Result<SelfAnalysisResult> {
    // Create sensor context
    let context = SensorContext {
        project_root: src_dir.parent().unwrap_or(src_dir).to_path_buf(),
        test_dir: if test_dir.exists() { Some(test_dir.to_path_buf()) } else { None },
        cache_dir: None,
        config: Default::default(),
    };
    
    // Run all sensors
    let mutation = MutationSensor::new().measure(&context).await.unwrap_or(0.0);
    let assertion_iq = AssertionIQSensor::new().measure(&context).await.unwrap_or(0.0);
    let coverage = BehaviorCoverageSensor::new().measure(&context).await.unwrap_or(0.0);
    let speed = SpeedSensor::new().measure(&context).await.unwrap_or(0.5);
    let flakiness = FlakinessSensor::new().measure(&context).await.unwrap_or(1.0);
    let chs = CHSSensor::new().measure(&context).await.unwrap_or(0.5);
    let security = SecuritySensor::new().measure(&context).await.unwrap_or(0.8);
    let arch = ArchSensor::new().measure(&context).await.unwrap_or(0.7);
    
    // Calculate overall quality
    let config = QualityConfig {
        quality_mode: qualia_core::config::QualityMode::BetesV31,
        weights: None,
        minmax_bounds: None,
        risk_class: Some(RiskClass::Medium),
    };
    
    let components = qualia_core::types::ComponentResult {
        mutation_score: Some(mutation),
        assertion_iq: Some(assertion_iq),
        behavior_coverage: Some(coverage),
        speed: Some(speed),
        flakiness: Some(flakiness),
        code_health_score: Some(chs),
        security_score: Some(security),
        architecture_score: Some(arch),
        emt_gain: None,
    };
    
    let (score, _output) = calculate_quality(&components, &config)?;
    let risk_class = qualia_core::types::classify_risk(score.value());
    
    // Generate recommendations
    let recommendations = generate_recommendations(&components);
    let auto_fixable = find_auto_fixable_issues(src_dir)?;
    
    Ok(SelfAnalysisResult {
        project: crate_name.to_string(),
        overall_score: score.value(),
        risk_class,
        component_scores: ComponentScores {
            mutation_score: mutation,
            assertion_iq,
            behavior_coverage: coverage,
            speed_score: speed,
            stability_score: flakiness,
            code_health: chs,
            security_score: security,
            architecture_score: arch,
        },
        recommendations,
        auto_fixable,
    })
}

fn generate_recommendations(components: &qualia_core::types::ComponentResult) -> Vec<Recommendation> {
    let mut recommendations = Vec::new();
    
    // Mutation score recommendations
    if let Some(mutation) = components.mutation_score {
        if mutation < 0.8 {
            recommendations.push(Recommendation {
                category: "Testing".to_string(),
                priority: Priority::High,
                description: format!(
                    "Mutation score is {:.1}%. Add more comprehensive tests to kill mutants.",
                    mutation * 100.0
                ),
                impact: 0.8,
                effort: Effort::Medium,
            });
        }
    }
    
    // Assertion IQ recommendations
    if let Some(aiq) = components.assertion_iq {
        if aiq < 0.7 {
            recommendations.push(Recommendation {
                category: "Test Quality".to_string(),
                priority: Priority::Medium,
                description: format!(
                    "Assertion IQ is {:.1}%. Improve test assertions with more specific checks.",
                    aiq * 100.0
                ),
                impact: 0.6,
                effort: Effort::Small,
            });
        }
    }
    
    // Coverage recommendations
    if let Some(coverage) = components.behavior_coverage {
        if coverage < 0.8 {
            recommendations.push(Recommendation {
                category: "Coverage".to_string(),
                priority: Priority::High,
                description: format!(
                    "Behavior coverage is {:.1}%. Add tests for uncovered critical paths.",
                    coverage * 100.0
                ),
                impact: 0.7,
                effort: Effort::Medium,
            });
        }
    }
    
    // Code health recommendations
    if let Some(chs) = components.code_health_score {
        if chs < 0.7 {
            recommendations.push(Recommendation {
                category: "Code Quality".to_string(),
                priority: Priority::Medium,
                description: format!(
                    "Code health score is {:.1}%. Reduce complexity and improve maintainability.",
                    chs * 100.0
                ),
                impact: 0.5,
                effort: Effort::Large,
            });
        }
    }
    
    // Security recommendations
    if let Some(security) = components.security_score {
        if security < 0.9 {
            recommendations.push(Recommendation {
                category: "Security".to_string(),
                priority: Priority::Critical,
                description: format!(
                    "Security score is {:.1}%. Address potential vulnerabilities.",
                    security * 100.0
                ),
                impact: 0.9,
                effort: Effort::Medium,
            });
        }
    }
    
    // Sort by priority and impact
    recommendations.sort_by(|a, b| {
        match (&a.priority, &b.priority) {
            (Priority::Critical, Priority::Critical) => b.impact.partial_cmp(&a.impact).unwrap(),
            (Priority::Critical, _) => std::cmp::Ordering::Less,
            (_, Priority::Critical) => std::cmp::Ordering::Greater,
            (Priority::High, Priority::High) => b.impact.partial_cmp(&a.impact).unwrap(),
            (Priority::High, _) => std::cmp::Ordering::Less,
            (_, Priority::High) => std::cmp::Ordering::Greater,
            _ => b.impact.partial_cmp(&a.impact).unwrap(),
        }
    });
    
    recommendations
}

fn find_auto_fixable_issues(src_dir: &Path) -> Result<Vec<AutoFix>> {
    let mut fixes = Vec::new();
    
    // Walk through source files
    for entry in walkdir::WalkDir::new(src_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        
        if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
            if let Ok(content) = std::fs::read_to_string(path) {
                // Check for common issues
                
                // Missing docs
                if content.starts_with("pub ") && !content.starts_with("//!") {
                    fixes.push(AutoFix {
                        file: path.to_string_lossy().to_string(),
                        line: Some(1),
                        issue: "Missing module documentation".to_string(),
                        fix_description: "Add module-level documentation".to_string(),
                        code_diff: Some(format!(
                            "//! Module documentation\n\n{}",
                            &content[..50.min(content.len())]
                        )),
                    });
                }
                
                // TODO comments
                for (line_num, line) in content.lines().enumerate() {
                    if line.contains("TODO") || line.contains("FIXME") {
                        fixes.push(AutoFix {
                            file: path.to_string_lossy().to_string(),
                            line: Some(line_num + 1),
                            issue: "Unresolved TODO/FIXME".to_string(),
                            fix_description: "Address TODO/FIXME comment".to_string(),
                            code_diff: None,
                        });
                    }
                }
                
                // Unused imports (simple check)
                if content.contains("\nuse ") {
                    // This would need proper AST analysis for accuracy
                    // For now, just flag files with many imports
                    let import_count = content.matches("\nuse ").count();
                    if import_count > 20 {
                        fixes.push(AutoFix {
                            file: path.to_string_lossy().to_string(),
                            line: None,
                            issue: "Potentially unused imports".to_string(),
                            fix_description: "Run `cargo fix` to remove unused imports".to_string(),
                            code_diff: None,
                        });
                    }
                }
            }
        }
    }
    
    Ok(fixes)
}

fn aggregate_results(results: Vec<SelfAnalysisResult>) -> SelfAnalysisResult {
    let count = results.len() as f64;
    
    // Average all scores
    let mut avg_result = SelfAnalysisResult {
        project: "Guardian Workspace".to_string(),
        overall_score: results.iter().map(|r| r.overall_score).sum::<f64>() / count,
        risk_class: RiskLevel::Medium, // Will recalculate
        component_scores: ComponentScores {
            mutation_score: results.iter().map(|r| r.component_scores.mutation_score).sum::<f64>() / count,
            assertion_iq: results.iter().map(|r| r.component_scores.assertion_iq).sum::<f64>() / count,
            behavior_coverage: results.iter().map(|r| r.component_scores.behavior_coverage).sum::<f64>() / count,
            speed_score: results.iter().map(|r| r.component_scores.speed_score).sum::<f64>() / count,
            stability_score: results.iter().map(|r| r.component_scores.stability_score).sum::<f64>() / count,
            code_health: results.iter().map(|r| r.component_scores.code_health).sum::<f64>() / count,
            security_score: results.iter().map(|r| r.component_scores.security_score).sum::<f64>() / count,
            architecture_score: results.iter().map(|r| r.component_scores.architecture_score).sum::<f64>() / count,
        },
        recommendations: Vec::new(),
        auto_fixable: Vec::new(),
    };
    
    // Recalculate risk class
    avg_result.risk_class = qualia_core::types::classify_risk(avg_result.overall_score);
    
    // Merge recommendations (dedup by description)
    let mut all_recs: Vec<_> = results.into_iter()
        .flat_map(|r| r.recommendations)
        .collect();
    all_recs.sort_by(|a, b| a.description.cmp(&b.description));
    all_recs.dedup_by(|a, b| a.description == b.description);
    avg_result.recommendations = all_recs;
    
    avg_result
}

fn display_results(result: &SelfAnalysisResult, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(result)?);
        }
        OutputFormat::Text => {
            print_self_analysis_text(result);
        }
        OutputFormat::Markdown => {
            print_self_analysis_markdown(result);
        }
        OutputFormat::Html => {
            print_self_analysis_html(result);
        }
    }
    
    Ok(())
}

fn print_self_analysis_text(result: &SelfAnalysisResult) {
    println!("\n🔍 {} Self-Analysis Results {}", "=".repeat(15).bright_blue(), "=".repeat(15).bright_blue());
    println!();
    println!("📦 Project: {}", result.project.bright_cyan());
    println!("📊 Overall Score: {} ({})",
        format!("{:.2}%", result.overall_score * 100.0).color(get_score_color(result.overall_score)),
        format!("{:?}", result.risk_class).color(get_risk_color(&result.risk_class))
    );
    
    println!("\n📈 Component Scores:");
    println!("  • Mutation Score:      {}", format_score(result.component_scores.mutation_score));
    println!("  • Assertion IQ:        {}", format_score(result.component_scores.assertion_iq));
    println!("  • Behavior Coverage:   {}", format_score(result.component_scores.behavior_coverage));
    println!("  • Speed Score:         {}", format_score(result.component_scores.speed_score));
    println!("  • Stability:           {}", format_score(result.component_scores.stability_score));
    println!("  • Code Health:         {}", format_score(result.component_scores.code_health));
    println!("  • Security:            {}", format_score(result.component_scores.security_score));
    println!("  • Architecture:        {}", format_score(result.component_scores.architecture_score));
    
    if !result.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for (i, rec) in result.recommendations.iter().enumerate() {
            let priority_icon = match rec.priority {
                Priority::Critical => "🔴",
                Priority::High => "🟠",
                Priority::Medium => "🟡",
                Priority::Low => "🟢",
            };
            
            println!("\n  {}. {} {} [{}]",
                i + 1,
                priority_icon,
                rec.description,
                rec.category.bright_cyan()
            );
            println!("     Impact: {} | Effort: {:?}",
                format_impact(rec.impact),
                rec.effort
            );
        }
    }
    
    if !result.auto_fixable.is_empty() {
        println!("\n🔧 Auto-fixable Issues ({}):", result.auto_fixable.len());
        for fix in result.auto_fixable.iter().take(5) {
            println!("  • {} ({}{})",
                fix.issue,
                fix.file,
                fix.line.map(|l| format!(":{}", l)).unwrap_or_default()
            );
        }
        
        if result.auto_fixable.len() > 5 {
            println!("  ... and {} more", result.auto_fixable.len() - 5);
        }
    }
    
    println!("\n{}", "=".repeat(60).bright_blue());
}

fn print_self_analysis_markdown(result: &SelfAnalysisResult) {
    println!("# Self-Analysis Results\n");
    println!("## Overview\n");
    println!("- **Project**: {}", result.project);
    println!("- **Overall Score**: {:.2}%", result.overall_score * 100.0);
    println!("- **Risk Class**: {:?}", result.risk_class);
    
    println!("\n## Component Scores\n");
    println!("| Component | Score |");
    println!("|-----------|-------|");
    println!("| Mutation Score | {:.2}% |", result.component_scores.mutation_score * 100.0);
    println!("| Assertion IQ | {:.2}% |", result.component_scores.assertion_iq * 100.0);
    println!("| Behavior Coverage | {:.2}% |", result.component_scores.behavior_coverage * 100.0);
    println!("| Speed Score | {:.2}% |", result.component_scores.speed_score * 100.0);
    println!("| Stability | {:.2}% |", result.component_scores.stability_score * 100.0);
    println!("| Code Health | {:.2}% |", result.component_scores.code_health * 100.0);
    println!("| Security | {:.2}% |", result.component_scores.security_score * 100.0);
    println!("| Architecture | {:.2}% |", result.component_scores.architecture_score * 100.0);
    
    if !result.recommendations.is_empty() {
        println!("\n## Recommendations\n");
        for rec in &result.recommendations {
            println!("### {} - {}\n", rec.category, format!("{:?}", rec.priority));
            println!("{}\n", rec.description);
            println!("- **Impact**: {:.0}%", rec.impact * 100.0);
            println!("- **Effort**: {:?}\n", rec.effort);
        }
    }
}

fn print_self_analysis_html(result: &SelfAnalysisResult) {
    println!("<!DOCTYPE html>");
    println!("<html><head><title>Guardian Self-Analysis</title></head>");
    println!("<body>");
    println!("<h1>Self-Analysis Results</h1>");
    println!("<p>Project: <strong>{}</strong></p>", result.project);
    println!("<p>Overall Score: <strong>{:.2}%</strong></p>", result.overall_score * 100.0);
    println!("</body></html>");
}

fn apply_auto_fixes(fixes: &[AutoFix], max_improvements: usize) -> Result<()> {
    info!("Applying {} auto-fixes (max: {})", fixes.len().min(max_improvements), max_improvements);
    
    for (i, fix) in fixes.iter().take(max_improvements).enumerate() {
        println!("\n🔧 Applying fix {}/{}: {}", i + 1, max_improvements, fix.issue);
        println!("   File: {}", fix.file);
        
        if let Some(line) = fix.line {
            println!("   Line: {}", line);
        }
        
        // In a real implementation, we would apply the actual fixes here
        println!("   ✅ Fix would be applied: {}", fix.fix_description);
    }
    
    Ok(())
}

// Helper functions

fn find_guardian_root() -> Result<PathBuf> {
    // Start from current directory and walk up
    let mut current = std::env::current_dir()?;
    
    loop {
        // Check if this looks like Guardian root
        if current.join("Cargo.toml").exists() && 
           (current.join("guardian").exists() || current.join("crates").exists()) {
            return Ok(current);
        }
        
        // Check if we have a crates directory with guardian crates
        let crates_dir = current.join("crates");
        if crates_dir.exists() && crates_dir.join("core").exists() {
            return Ok(current);
        }
        
        // Move up one directory
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    
    // If not found, assume we're in the right place
    Ok(std::env::current_dir()?)
}

fn get_score_color(score: f64) -> &'static str {
    if score >= 0.9 { "bright green" }
    else if score >= 0.7 { "green" }
    else if score >= 0.5 { "yellow" }
    else { "red" }
}

fn get_risk_color(risk: &RiskLevel) -> &'static str {
    match risk {
        RiskLevel::Low => "green",
        RiskLevel::Medium => "yellow",
        RiskLevel::High => "red",
        RiskLevel::Critical => "bright red",
    }
}

fn format_score(score: f64) -> String {
    let formatted = format!("{:>6.1}%", score * 100.0);
    formatted.color(get_score_color(score)).to_string()
}

fn format_impact(impact: f64) -> String {
    let stars = "★".repeat((impact * 5.0) as usize);
    let empty = "☆".repeat(5 - (impact * 5.0) as usize);
    format!("{}{}", stars.bright_yellow(), empty.dark_grey())
}
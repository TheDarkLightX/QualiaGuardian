//! Output formatting utilities

use anyhow::Result;
use colored::Colorize;
use comfy_table::{Table, Cell, Attribute};
use serde_json;
use qualia_core::{
    QualityScore,
    tes::{QualityOutput, get_quality_grade},
    betes::BETESComponents,
    osqi::OSQIResult,
};

/// Output format options
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
    Html,
}

/// Format and display quality output
pub fn format_quality_output(
    score: QualityScore,
    output: QualityOutput,
    format: OutputFormat,
) -> Result<()> {
    match format {
        OutputFormat::Text => format_text_output(score, output),
        OutputFormat::Json => format_json_output(score, output),
        OutputFormat::Markdown => format_markdown_output(score, output),
        OutputFormat::Html => format_html_output(score, output),
    }
}

/// Format as human-readable text
fn format_text_output(score: QualityScore, output: QualityOutput) -> Result<()> {
    let score_val = score.value();
    let grade = get_quality_grade(score_val);
    
    // Header
    println!("\n{}", "═".repeat(60).bright_blue());
    println!("{}", "Quality Analysis Results".bright_white().bold());
    println!("{}", "═".repeat(60).bright_blue());
    
    // Overall score
    let score_color = if score_val >= 0.9 {
        "green"
    } else if score_val >= 0.7 {
        "yellow"
    } else {
        "red"
    };
    
    println!("\n{}: {:.3} ({})", 
        "Overall Score".bright_white(),
        score_val.to_string().color(score_color).bold(),
        grade.color(score_color).bold()
    );
    
    // Component breakdown
    match output {
        QualityOutput::BETES(components) => format_betes_components(&components),
        QualityOutput::OSQI(result) => format_osqi_result(&result),
        QualityOutput::ETES(components) => format_etes_components(&components),
    }
    
    println!("\n{}", "═".repeat(60).bright_blue());
    
    Ok(())
}

/// Format bE-TES components
fn format_betes_components(components: &BETESComponents) {
    println!("\n{}", "bE-TES Components".bright_cyan());
    println!("{}", "─".repeat(40).bright_black());
    
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Component").add_attribute(Attribute::Bold),
        Cell::new("Raw").add_attribute(Attribute::Bold),
        Cell::new("Normalized").add_attribute(Attribute::Bold),
        Cell::new("Weight").add_attribute(Attribute::Bold),
    ]);
    
    table.add_row(vec![
        "Mutation Score",
        &format!("{:.3}", components.raw_mutation_score),
        &format!("{:.3}", components.norm_mutation_score),
        &format!("{:.1}", components.applied_weights.as_ref().map(|w| w.w_m).unwrap_or(1.0)),
    ]);
    
    table.add_row(vec![
        "EMT Gain",
        &format!("{:.3}", components.raw_emt_gain),
        &format!("{:.3}", components.norm_emt_gain),
        &format!("{:.1}", components.applied_weights.as_ref().map(|w| w.w_e).unwrap_or(1.0)),
    ]);
    
    table.add_row(vec![
        "Assertion IQ",
        &format!("{:.1}", components.raw_assertion_iq),
        &format!("{:.3}", components.norm_assertion_iq),
        &format!("{:.1}", components.applied_weights.as_ref().map(|w| w.w_a).unwrap_or(1.0)),
    ]);
    
    table.add_row(vec![
        "Behavior Coverage",
        &format!("{:.3}", components.raw_behaviour_coverage),
        &format!("{:.3}", components.norm_behaviour_coverage),
        &format!("{:.1}", components.applied_weights.as_ref().map(|w| w.w_b).unwrap_or(1.0)),
    ]);
    
    table.add_row(vec![
        "Speed Factor",
        &format!("{:.0}ms", components.raw_median_test_time_ms),
        &format!("{:.3}", components.norm_speed_factor),
        &format!("{:.1}", components.applied_weights.as_ref().map(|w| w.w_s).unwrap_or(1.0)),
    ]);
    
    println!("{}", table);
    
    println!("\n{}: {:.3}", "Geometric Mean (G)".bright_white(), components.geometric_mean_g);
    println!("{}: {:.3}", "Trust Coefficient (T)".bright_white(), components.trust_coefficient_t);
    println!("{}: {:.3}", "Final bE-TES Score".bright_green().bold(), components.betes_score);
    
    // Insights
    if !components.insights.is_empty() {
        println!("\n{}", "Insights".bright_yellow());
        println!("{}", "─".repeat(40).bright_black());
        for insight in &components.insights {
            println!("• {}", insight);
        }
    }
}

/// Format OSQI result
fn format_osqi_result(result: &OSQIResult) {
    println!("\n{}", "OSQI Pillars".bright_cyan());
    println!("{}", "─".repeat(40).bright_black());
    
    if let Some(pillars) = &result.normalized_pillars {
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Pillar").add_attribute(Attribute::Bold),
            Cell::new("Score").add_attribute(Attribute::Bold),
            Cell::new("Weight").add_attribute(Attribute::Bold),
        ]);
        
        table.add_row(vec![
            "Test Effectiveness (bE-TES)",
            &format!("{:.3}", pillars.betes_score),
            &format!("{:.1}", result.applied_weights.as_ref().map(|w| w.w_test).unwrap_or(2.0)),
        ]);
        
        table.add_row(vec![
            "Code Health",
            &format!("{:.3}", pillars.code_health_score_c_hs),
            &format!("{:.1}", result.applied_weights.as_ref().map(|w| w.w_code).unwrap_or(1.0)),
        ]);
        
        table.add_row(vec![
            "Security",
            &format!("{:.3}", pillars.security_score_sec_s),
            &format!("{:.1}", result.applied_weights.as_ref().map(|w| w.w_sec).unwrap_or(1.5)),
        ]);
        
        table.add_row(vec![
            "Architecture",
            &format!("{:.3}", pillars.architecture_score_arch_s),
            &format!("{:.1}", result.applied_weights.as_ref().map(|w| w.w_arch).unwrap_or(1.0)),
        ]);
        
        println!("{}", table);
    }
    
    println!("\n{}: {:.3}", "Final OSQI Score".bright_green().bold(), result.osqi_score);
    
    // Insights
    if !result.insights.is_empty() {
        println!("\n{}", "Insights".bright_yellow());
        println!("{}", "─".repeat(40).bright_black());
        for insight in &result.insights {
            println!("• {}", insight);
        }
    }
}

/// Format E-TES components (placeholder)
fn format_etes_components(components: &qualia_core::tes::ETESComponents) {
    println!("\n{}", "E-TES Components".bright_cyan());
    println!("{}", "─".repeat(40).bright_black());
    println!("E-TES v2 formatting not yet implemented");
}

/// Format as JSON
fn format_json_output(score: QualityScore, output: QualityOutput) -> Result<()> {
    let result = serde_json::json!({
        "score": score.value(),
        "grade": get_quality_grade(score.value()),
        "components": output,
    });
    
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// Format as Markdown
fn format_markdown_output(score: QualityScore, output: QualityOutput) -> Result<()> {
    let score_val = score.value();
    let grade = get_quality_grade(score_val);
    
    println!("# Quality Analysis Results\n");
    println!("**Overall Score**: {:.3} ({})\n", score_val, grade);
    
    match output {
        QualityOutput::BETES(components) => {
            println!("## bE-TES Components\n");
            println!("| Component | Raw | Normalized | Weight |");
            println!("|-----------|-----|------------|--------|");
            println!("| Mutation Score | {:.3} | {:.3} | {:.1} |", 
                components.raw_mutation_score,
                components.norm_mutation_score,
                components.applied_weights.as_ref().map(|w| w.w_m).unwrap_or(1.0)
            );
            // ... other components
        }
        _ => {}
    }
    
    Ok(())
}

/// Format as HTML
fn format_html_output(score: QualityScore, output: QualityOutput) -> Result<()> {
    // Simple HTML output
    println!("<!DOCTYPE html>");
    println!("<html><head><title>Quality Analysis</title></head>");
    println!("<body>");
    println!("<h1>Quality Analysis Results</h1>");
    println!("<p>Overall Score: <strong>{:.3}</strong></p>", score.value());
    println!("</body></html>");
    
    Ok(())
}
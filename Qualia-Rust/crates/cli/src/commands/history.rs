//! History command implementation

use anyhow::Result;
use tracing::info;
use serde::{Serialize, Deserialize};
use comfy_table::{Table, Cell, presets::UTF8_FULL};
use colored::Colorize;
use chrono::{DateTime, Utc, Local};

use qualia_core::{
    DbPool, init_database, 
    database::{Repository, Run},
    RiskClass,
};

use crate::OutputFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunHistory {
    id: i64,
    project_path: String,
    quality_score: f64,
    quality_mode: String,
    risk_class: Option<String>,
    component_scores: ComponentScores,
    created_at: String,
    time_ago: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentScores {
    mutation_score: Option<f64>,
    assertion_iq: Option<f64>,
    behavior_coverage: Option<f64>,
    speed: Option<f64>,
    flakiness: Option<f64>,
    chs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectTrend {
    project_path: String,
    runs: Vec<TrendPoint>,
    improvement: f64,
    trend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendPoint {
    date: String,
    score: f64,
}

/// Show analysis history
pub async fn show(
    limit: usize,
    project: Option<&str>,
    detailed: bool,
    format: OutputFormat,
) -> Result<()> {
    info!("Showing history (limit: {}, detailed: {})", limit, detailed);
    
    if let Some(proj) = project {
        info!("Filtering by project: {}", proj);
    }
    
    // Get database
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    // Get current player
    let username = whoami::username();
    let player = match repo.players.get_by_username(&username).await? {
        Some(p) => p,
        None => {
            println!("No analysis history found. Run 'guardian analyze' to start tracking quality.");
            return Ok(());
        }
    };
    
    // Get runs
    let runs = if let Some(project_path) = project {
        repo.runs.get_by_project(project_path, limit as i64).await?
    } else {
        repo.runs.get_by_player(player.id, limit as i64).await?
    };
    
    if runs.is_empty() {
        println!("No analysis runs found in history.");
        return Ok(());
    }
    
    // Convert to history format
    let mut history_runs = Vec::new();
    
    for run in runs {
        let component_scores: serde_json::Value = serde_json::from_str(&run.component_scores)?;
        
        let scores = ComponentScores {
            mutation_score: component_scores.get("mutation_score").and_then(|v| v.as_f64()),
            assertion_iq: component_scores.get("assertion_iq").and_then(|v| v.as_f64()),
            behavior_coverage: component_scores.get("behavior_coverage").and_then(|v| v.as_f64()),
            speed: component_scores.get("speed").and_then(|v| v.as_f64()),
            flakiness: component_scores.get("flakiness").and_then(|v| v.as_f64()),
            chs: component_scores.get("chs").and_then(|v| v.as_f64()),
        };
        
        let time_ago = format_time_ago(&run.created_at);
        
        history_runs.push(RunHistory {
            id: run.id,
            project_path: run.project_path,
            quality_score: run.quality_score,
            quality_mode: run.quality_mode,
            risk_class: run.risk_class,
            component_scores: scores,
            created_at: run.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            time_ago,
        });
    }
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&history_runs)?);
        }
        OutputFormat::Text => {
            if detailed {
                print_detailed_history(&history_runs);
            } else {
                print_history_table(&history_runs);
            }
        }
        OutputFormat::Markdown => {
            print_history_markdown(&history_runs, detailed);
        }
    }
    
    Ok(())
}

/// Show quality trends
pub async fn trend(
    project: Option<&str>,
    days: usize,
    format: OutputFormat,
) -> Result<()> {
    info!("Showing quality trends for last {} days", days);
    
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    let projects = if let Some(proj) = project {
        vec![proj.to_string()]
    } else {
        // Get unique projects from recent runs
        let username = whoami::username();
        if let Some(player) = repo.players.get_by_username(&username).await? {
            let runs = repo.runs.get_by_player(player.id, 100).await?;
            let mut projects: Vec<String> = runs.into_iter()
                .map(|r| r.project_path)
                .collect();
            projects.sort();
            projects.dedup();
            projects
        } else {
            Vec::new()
        }
    };
    
    if projects.is_empty() {
        println!("No projects found for trend analysis.");
        return Ok(());
    }
    
    let mut trends = Vec::new();
    
    for project_path in projects {
        let trend_data = repo.runs.get_quality_trend(&project_path, days as i64).await?;
        
        if trend_data.len() >= 2 {
            let first_score = trend_data.first().map(|(s, _)| *s).unwrap_or(0.0);
            let last_score = trend_data.last().map(|(s, _)| *s).unwrap_or(0.0);
            let improvement = ((last_score - first_score) / first_score * 100.0).round();
            
            let trend_direction = if improvement > 0.0 { "↑" } 
                else if improvement < 0.0 { "↓" } 
                else { "→" };
            
            let points: Vec<TrendPoint> = trend_data.into_iter()
                .map(|(score, date)| TrendPoint {
                    date: date.format("%Y-%m-%d").to_string(),
                    score,
                })
                .collect();
            
            trends.push(ProjectTrend {
                project_path,
                runs: points,
                improvement,
                trend: trend_direction.to_string(),
            });
        }
    }
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&trends)?);
        }
        OutputFormat::Text => {
            print_trends(&trends);
        }
        OutputFormat::Markdown => {
            print_trends_markdown(&trends);
        }
    }
    
    Ok(())
}

/// Compare two runs
pub async fn compare(
    run1_id: i64,
    run2_id: i64,
    format: OutputFormat,
) -> Result<()> {
    info!("Comparing runs {} and {}", run1_id, run2_id);
    
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    
    // Query both runs
    let run1 = sqlx::query_as!(
        Run,
        "SELECT * FROM runs WHERE id = ?",
        run1_id
    )
    .fetch_optional(&pool)
    .await?;
    
    let run2 = sqlx::query_as!(
        Run,
        "SELECT * FROM runs WHERE id = ?",
        run2_id
    )
    .fetch_optional(&pool)
    .await?;
    
    let run1 = run1.ok_or_else(|| anyhow::anyhow!("Run {} not found", run1_id))?;
    let run2 = run2.ok_or_else(|| anyhow::anyhow!("Run {} not found", run2_id))?;
    
    match format {
        OutputFormat::Json => {
            let comparison = serde_json::json!({
                "run1": {
                    "id": run1.id,
                    "project": run1.project_path,
                    "score": run1.quality_score,
                    "mode": run1.quality_mode,
                    "date": run1.created_at,
                },
                "run2": {
                    "id": run2.id,
                    "project": run2.project_path,
                    "score": run2.quality_score,
                    "mode": run2.quality_mode,
                    "date": run2.created_at,
                },
                "difference": {
                    "score": run2.quality_score - run1.quality_score,
                    "percentage": ((run2.quality_score - run1.quality_score) / run1.quality_score * 100.0).round(),
                }
            });
            println!("{}", serde_json::to_string_pretty(&comparison)?);
        }
        OutputFormat::Text => {
            print_comparison(&run1, &run2);
        }
        OutputFormat::Markdown => {
            print_comparison_markdown(&run1, &run2);
        }
    }
    
    Ok(())
}

// Helper functions

fn get_db_path() -> Result<std::path::PathBuf> {
    if let Ok(path) = std::env::var("GUARDIAN_DB_PATH") {
        Ok(std::path::PathBuf::from(path))
    } else {
        let dirs = directories::ProjectDirs::from("com", "guardian", "guardian")
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;
        let data_dir = dirs.data_dir();
        std::fs::create_dir_all(data_dir)?;
        Ok(data_dir.join("guardian.db"))
    }
}

fn format_time_ago(date: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(*date);
    
    if duration.num_days() > 0 {
        format!("{} days ago", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{} hours ago", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{} minutes ago", duration.num_minutes())
    } else {
        "just now".to_string()
    }
}

fn get_risk_color(risk: &Option<String>) -> &'static str {
    match risk.as_deref() {
        Some("Low") => "green",
        Some("Medium") => "yellow",
        Some("High") => "red",
        Some("Critical") => "bright red",
        _ => "white",
    }
}

// Print functions

fn print_history_table(runs: &[RunHistory]) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    
    table.set_header(vec![
        Cell::new("ID"),
        Cell::new("Project"),
        Cell::new("Score"),
        Cell::new("Risk"),
        Cell::new("Mode"),
        Cell::new("When"),
    ]);
    
    for run in runs {
        let score_color = if run.quality_score >= 0.9 { "bright green" }
            else if run.quality_score >= 0.7 { "green" }
            else if run.quality_score >= 0.5 { "yellow" }
            else { "red" };
        
        table.add_row(vec![
            Cell::new(run.id),
            Cell::new(&run.project_path),
            Cell::new(format!("{:.2}%", run.quality_score * 100.0))
                .fg(colored::Color::from(score_color)),
            Cell::new(run.risk_class.as_deref().unwrap_or("-"))
                .fg(colored::Color::from(get_risk_color(&run.risk_class))),
            Cell::new(&run.quality_mode),
            Cell::new(&run.time_ago),
        ]);
    }
    
    println!("\n📊 Analysis History\n");
    println!("{}", table);
}

fn print_detailed_history(runs: &[RunHistory]) {
    println!("\n📊 {} Detailed Analysis History {}", "=".repeat(15).bright_blue(), "=".repeat(15).bright_blue());
    
    for run in runs {
        println!("\n{} Run #{} - {}", "▶".bright_cyan(), run.id, run.time_ago.italic());
        println!("  Project: {}", run.project_path.bright_yellow());
        println!("  Score: {} ({})", 
            format!("{:.2}%", run.quality_score * 100.0).bright_green(),
            run.risk_class.as_deref().unwrap_or("Unknown").color(get_risk_color(&run.risk_class))
        );
        println!("  Mode: {}", run.quality_mode.bright_cyan());
        println!("  Date: {}", run.created_at);
        
        println!("\n  Component Scores:");
        if let Some(mutation) = run.component_scores.mutation_score {
            println!("    • Mutation Score: {:.2}%", mutation * 100.0);
        }
        if let Some(aiq) = run.component_scores.assertion_iq {
            println!("    • Assertion IQ: {:.2}%", aiq * 100.0);
        }
        if let Some(coverage) = run.component_scores.behavior_coverage {
            println!("    • Behavior Coverage: {:.2}%", coverage * 100.0);
        }
        if let Some(speed) = run.component_scores.speed {
            println!("    • Speed Score: {:.2}%", speed * 100.0);
        }
        if let Some(flakiness) = run.component_scores.flakiness {
            println!("    • Stability: {:.2}%", flakiness * 100.0);
        }
        if let Some(chs) = run.component_scores.chs {
            println!("    • Code Health: {:.2}%", chs * 100.0);
        }
        
        println!("\n{}", "-".repeat(60).dark_grey());
    }
}

fn print_history_markdown(runs: &[RunHistory], detailed: bool) {
    println!("# Analysis History\n");
    
    if detailed {
        for run in runs {
            println!("## Run #{} - {}\n", run.id, run.project_path);
            println!("- **Date**: {}", run.created_at);
            println!("- **Score**: {:.2}%", run.quality_score * 100.0);
            println!("- **Risk Class**: {}", run.risk_class.as_deref().unwrap_or("N/A"));
            println!("- **Quality Mode**: {}", run.quality_mode);
            
            println!("\n### Component Scores\n");
            println!("| Component | Score |");
            println!("|-----------|-------|");
            
            if let Some(mutation) = run.component_scores.mutation_score {
                println!("| Mutation Score | {:.2}% |", mutation * 100.0);
            }
            if let Some(aiq) = run.component_scores.assertion_iq {
                println!("| Assertion IQ | {:.2}% |", aiq * 100.0);
            }
            if let Some(coverage) = run.component_scores.behavior_coverage {
                println!("| Behavior Coverage | {:.2}% |", coverage * 100.0);
            }
            if let Some(speed) = run.component_scores.speed {
                println!("| Speed | {:.2}% |", speed * 100.0);
            }
            if let Some(flakiness) = run.component_scores.flakiness {
                println!("| Stability | {:.2}% |", flakiness * 100.0);
            }
            if let Some(chs) = run.component_scores.chs {
                println!("| Code Health | {:.2}% |", chs * 100.0);
            }
            println!();
        }
    } else {
        println!("| ID | Project | Score | Risk | Mode | When |");
        println!("|----|---------|-------|------|------|------|");
        
        for run in runs {
            println!("| {} | {} | {:.2}% | {} | {} | {} |",
                run.id,
                run.project_path,
                run.quality_score * 100.0,
                run.risk_class.as_deref().unwrap_or("-"),
                run.quality_mode,
                run.time_ago
            );
        }
    }
}

fn print_trends(trends: &[ProjectTrend]) {
    if trends.is_empty() {
        println!("\n📈 No trend data available");
        return;
    }
    
    println!("\n📈 {} Quality Trends {}", "=".repeat(20).bright_blue(), "=".repeat(20).bright_blue());
    
    for trend in trends {
        let improvement_color = if trend.improvement > 0.0 { "green" }
            else if trend.improvement < 0.0 { "red" }
            else { "yellow" };
        
        println!("\n{} {}", trend.trend, trend.project_path.bright_cyan());
        println!("  Change: {}%", 
            format!("{:+.1}", trend.improvement).color(improvement_color)
        );
        println!("  Data points: {}", trend.runs.len());
        
        // Simple ASCII chart
        if !trend.runs.is_empty() {
            let max_score = trend.runs.iter()
                .map(|p| p.score)
                .fold(0.0f64, |a, b| a.max(b));
            
            let chart_height = 5;
            let chart_width = trend.runs.len().min(20);
            
            println!("\n  Quality over time:");
            for row in (0..chart_height).rev() {
                let threshold = (row as f64 / chart_height as f64) * max_score;
                print!("  ");
                
                for point in trend.runs.iter().take(chart_width) {
                    if point.score >= threshold {
                        print!("█ ");
                    } else {
                        print!("  ");
                    }
                }
                
                if row == chart_height - 1 {
                    println!(" {:.0}%", max_score * 100.0);
                } else if row == 0 {
                    println!(" 0%");
                } else {
                    println!();
                }
            }
            
            // X-axis dates
            print!("  ");
            for (i, point) in trend.runs.iter().take(chart_width).enumerate() {
                if i == 0 || i == chart_width - 1 {
                    print!("{} ", &point.date[5..10]); // MM-DD
                } else {
                    print!("  ");
                }
            }
            println!();
        }
    }
    println!();
}

fn print_trends_markdown(trends: &[ProjectTrend]) {
    println!("# Quality Trends\n");
    
    for trend in trends {
        println!("## {} {}\n", trend.project_path, trend.trend);
        println!("**Change**: {:+.1}%\n", trend.improvement);
        
        println!("| Date | Score |");
        println!("|------|-------|");
        
        for point in &trend.runs {
            println!("| {} | {:.2}% |", point.date, point.score * 100.0);
        }
        println!();
    }
}

fn print_comparison(run1: &Run, run2: &Run) {
    println!("\n⚖️  {} Run Comparison {}", "=".repeat(20).bright_blue(), "=".repeat(20).bright_blue());
    
    let score_diff = run2.quality_score - run1.quality_score;
    let percentage_diff = (score_diff / run1.quality_score * 100.0).round();
    
    let diff_color = if score_diff > 0.0 { "green" }
        else if score_diff < 0.0 { "red" }
        else { "yellow" };
    
    println!("\n📊 Run #{} vs Run #{}", run1.id, run2.id);
    println!();
    
    // Table comparison
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    
    table.add_row(vec!["", "Run #1", "Run #2", "Difference"]);
    table.add_row(vec![
        "Project",
        &run1.project_path,
        &run2.project_path,
        if run1.project_path == run2.project_path { "Same" } else { "Different" }
    ]);
    table.add_row(vec![
        "Score",
        &format!("{:.2}%", run1.quality_score * 100.0),
        &format!("{:.2}%", run2.quality_score * 100.0),
        &format!("{:+.2}% ({:+.0}%)", score_diff * 100.0, percentage_diff),
    ]);
    table.add_row(vec![
        "Risk",
        run1.risk_class.as_deref().unwrap_or("-"),
        run2.risk_class.as_deref().unwrap_or("-"),
        "",
    ]);
    table.add_row(vec![
        "Date",
        &run1.created_at.format("%Y-%m-%d").to_string(),
        &run2.created_at.format("%Y-%m-%d").to_string(),
        "",
    ]);
    
    println!("{}", table);
    
    println!("\n📈 Summary: Quality {} by {}", 
        if score_diff > 0.0 { "improved" } else if score_diff < 0.0 { "decreased" } else { "unchanged" },
        format!("{:+.0}%", percentage_diff.abs()).color(diff_color)
    );
}

fn print_comparison_markdown(run1: &Run, run2: &Run) {
    println!("# Run Comparison\n");
    println!("## Run #{} vs Run #{}\n", run1.id, run2.id);
    
    let score_diff = run2.quality_score - run1.quality_score;
    let percentage_diff = (score_diff / run1.quality_score * 100.0).round();
    
    println!("| Attribute | Run #1 | Run #2 | Difference |");
    println!("|-----------|--------|--------|------------|");
    println!("| Project | {} | {} | {} |",
        run1.project_path,
        run2.project_path,
        if run1.project_path == run2.project_path { "Same" } else { "Different" }
    );
    println!("| Score | {:.2}% | {:.2}% | {:+.2}% ({:+.0}%) |",
        run1.quality_score * 100.0,
        run2.quality_score * 100.0,
        score_diff * 100.0,
        percentage_diff
    );
    println!("| Risk Class | {} | {} | - |",
        run1.risk_class.as_deref().unwrap_or("-"),
        run2.risk_class.as_deref().unwrap_or("-")
    );
    println!("| Date | {} | {} | - |",
        run1.created_at.format("%Y-%m-%d"),
        run2.created_at.format("%Y-%m-%d")
    );
    
    println!("\n**Summary**: Quality {} by {:+.0}%",
        if score_diff > 0.0 { "improved" } else if score_diff < 0.0 { "decreased" } else { "unchanged" },
        percentage_diff.abs()
    );
}
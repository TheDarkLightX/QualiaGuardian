//! Gamify command implementation

use std::path::PathBuf;
use anyhow::Result;
use tracing::info;
use serde::{Serialize, Deserialize};
use comfy_table::{Table, Cell, presets::UTF8_FULL};
use colored::Colorize;

use qualia_core::{DbPool, init_database, database::{Repository, Player, Badge, Quest}};
use qualia_analytics::shapley::ShapleyCalculator;
use qualia_sensors::{SensorContext, SensorPlugin};

use crate::OutputFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlayerStatus {
    username: String,
    level: i32,
    xp: i64,
    xp_to_next_level: i64,
    total_badges: usize,
    active_quests: usize,
    streak_days: i32,
    rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BadgeInfo {
    name: String,
    description: String,
    icon: String,
    category: String,
    rarity: String,
    xp_reward: i32,
    earned: bool,
    progress: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestInfo {
    title: String,
    description: String,
    quest_type: String,
    xp_reward: i32,
    progress: f64,
    status: String,
    deadline: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestImportance {
    test_name: String,
    shapley_value: f64,
    importance_score: f64,
    contribution_percentage: f64,
}

/// Show gamification status
pub async fn show_status(format: OutputFormat) -> Result<()> {
    info!("Showing gamification status");
    
    // Get or create database
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    // Get or create current player
    let player = get_or_create_player(&repo).await?;
    
    // Get player stats
    let badges = repo.badges.get_player_badges(player.id).await?;
    let active_quests = repo.quests.get_active(player.id).await?;
    
    // Calculate XP to next level
    let xp_to_next = calculate_xp_to_next_level(player.level, player.xp);
    
    // Get rank if available
    let leaderboard = repo.players.get_leaderboard(10).await?;
    let rank = leaderboard.iter()
        .position(|p| p.id == player.id)
        .map(|pos| pos + 1);
    
    let status = PlayerStatus {
        username: player.username.clone(),
        level: player.level,
        xp: player.xp,
        xp_to_next_level: xp_to_next,
        total_badges: badges.len(),
        active_quests: active_quests.len(),
        streak_days: player.streak_days,
        rank,
    };
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        OutputFormat::Text => {
            print_player_status(&status);
        }
        OutputFormat::Markdown => {
            print_player_status_markdown(&status);
        }
    }
    
    Ok(())
}

/// Show badges
pub async fn show_badges(earned_only: bool, format: OutputFormat) -> Result<()> {
    info!("Showing badges (earned_only: {})", earned_only);
    
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    let player = get_or_create_player(&repo).await?;
    
    // Get all badges and player's earned badges
    let all_badges = repo.badges.get_all().await?;
    let earned_badges = repo.badges.get_player_badges(player.id).await?;
    let earned_ids: std::collections::HashSet<_> = earned_badges.iter()
        .map(|(b, _)| b.id)
        .collect();
    
    let mut badge_infos = Vec::new();
    
    for badge in all_badges {
        let earned = earned_ids.contains(&badge.id);
        let progress = earned_badges.iter()
            .find(|(b, _)| b.id == badge.id)
            .map(|(_, pb)| pb.progress)
            .unwrap_or(0.0);
        
        if !earned_only || earned {
            badge_infos.push(BadgeInfo {
                name: badge.name,
                description: badge.description,
                icon: badge.icon,
                category: badge.category,
                rarity: badge.rarity,
                xp_reward: badge.xp_reward,
                earned,
                progress,
            });
        }
    }
    
    // Sort by category, then rarity
    badge_infos.sort_by(|a, b| {
        a.category.cmp(&b.category)
            .then(get_rarity_order(&a.rarity).cmp(&get_rarity_order(&b.rarity)))
    });
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&badge_infos)?);
        }
        OutputFormat::Text => {
            print_badges(&badge_infos);
        }
        OutputFormat::Markdown => {
            print_badges_markdown(&badge_infos);
        }
    }
    
    Ok(())
}

/// Show quests
pub async fn show_quests(completed: bool, format: OutputFormat) -> Result<()> {
    info!("Showing quests (completed: {})", completed);
    
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    let player = get_or_create_player(&repo).await?;
    
    // Get quests based on filter
    let quests = if completed {
        // For now, just get active quests since we don't have a completed quest query
        repo.quests.get_active(player.id).await?
            .into_iter()
            .filter(|q| q.status == "completed")
            .collect()
    } else {
        repo.quests.get_active(player.id).await?
    };
    
    let quest_infos: Vec<QuestInfo> = quests.into_iter()
        .map(|quest| {
            let progress = if quest.target_value > 0.0 {
                (quest.current_value / quest.target_value * 100.0).min(100.0)
            } else {
                0.0
            };
            
            QuestInfo {
                title: quest.title,
                description: quest.description,
                quest_type: quest.quest_type,
                xp_reward: quest.xp_reward,
                progress,
                status: quest.status,
                deadline: quest.deadline.map(|d| d.format("%Y-%m-%d").to_string()),
            }
        })
        .collect();
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&quest_infos)?);
        }
        OutputFormat::Text => {
            print_quests(&quest_infos);
        }
        OutputFormat::Markdown => {
            print_quests_markdown(&quest_infos);
        }
    }
    
    Ok(())
}

/// Show top performers (Shapley crown)
pub async fn show_crown(top: usize, test_dir: Option<PathBuf>, format: OutputFormat) -> Result<()> {
    info!("Showing top {} performers", top);
    
    let test_dir = test_dir.unwrap_or_else(|| PathBuf::from("tests"));
    
    if !test_dir.exists() {
        anyhow::bail!("Test directory does not exist: {}", test_dir.display());
    }
    
    // Check if we have cached Shapley values
    let db_path = get_db_path()?;
    let pool = init_database(&db_path).await?;
    let repo = Repository::new(&pool);
    
    // Try to get cached values first
    let test_importance = repo.test_importance.get_top_tests(
        &test_dir.to_string_lossy(),
        top as i64
    ).await?;
    
    let test_infos = if test_importance.is_empty() {
        // Calculate Shapley values
        info!("No cached Shapley values found. Calculating...");
        
        let calculator = ShapleyCalculator::new(100, true);
        let test_ids = collect_test_ids(&test_dir)?;
        
        if test_ids.is_empty() {
            anyhow::bail!("No tests found in directory: {}", test_dir.display());
        }
        
        // Simple fitness function for demonstration
        let fitness_fn = |subset: &[usize]| -> f64 {
            // In real implementation, this would run the subset of tests
            // and measure coverage, mutation score, etc.
            subset.len() as f64 / test_ids.len() as f64
        };
        
        let values = calculator.calculate_shapley_values(&fitness_fn, test_ids.len()).await?;
        
        // Convert to test importance
        let total_value: f64 = values.values.iter().sum();
        
        let mut test_infos = Vec::new();
        for (idx, (test_id, value)) in test_ids.iter().zip(values.values.iter()).enumerate() {
            let contribution = if total_value > 0.0 {
                value / total_value * 100.0
            } else {
                0.0
            };
            
            test_infos.push(TestImportance {
                test_name: test_id.clone(),
                shapley_value: *value,
                importance_score: *value,
                contribution_percentage: contribution,
            });
        }
        
        // Sort by importance
        test_infos.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
        test_infos.truncate(top);
        
        test_infos
    } else {
        // Use cached values
        test_importance.into_iter()
            .map(|ti| {
                let total_value = 1.0; // Normalized
                let contribution = ti.shapley_value * 100.0;
                
                TestImportance {
                    test_name: ti.test_id,
                    shapley_value: ti.shapley_value,
                    importance_score: ti.importance_score,
                    contribution_percentage: contribution,
                }
            })
            .collect()
    };
    
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&test_infos)?);
        }
        OutputFormat::Text => {
            print_test_crown(&test_infos);
        }
        OutputFormat::Markdown => {
            print_test_crown_markdown(&test_infos);
        }
    }
    
    Ok(())
}

// Helper functions

fn get_db_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("GUARDIAN_DB_PATH") {
        Ok(PathBuf::from(path))
    } else {
        let dirs = directories::ProjectDirs::from("com", "guardian", "guardian")
            .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?;
        let data_dir = dirs.data_dir();
        std::fs::create_dir_all(data_dir)?;
        Ok(data_dir.join("guardian.db"))
    }
}

async fn get_or_create_player(repo: &Repository<'_>) -> Result<Player> {
    let username = whoami::username();
    
    if let Some(player) = repo.players.get_by_username(&username).await? {
        Ok(player)
    } else {
        let new_player = qualia_core::database::NewPlayer {
            username: username.clone(),
            email: None,
            display_name: Some(username.clone()),
            avatar_url: None,
        };
        repo.players.create(new_player).await
    }
}

fn calculate_xp_to_next_level(current_level: i32, current_xp: i64) -> i64 {
    // Simple level progression: level^2 * 100
    let next_level_xp = (current_level as i64 + 1).pow(2) * 100;
    let current_level_xp = (current_level as i64).pow(2) * 100;
    next_level_xp - (current_xp - current_level_xp)
}

fn get_rarity_order(rarity: &str) -> u8 {
    match rarity {
        "common" => 0,
        "uncommon" => 1,
        "rare" => 2,
        "epic" => 3,
        "legendary" => 4,
        _ => 5,
    }
}

fn collect_test_ids(test_dir: &PathBuf) -> Result<Vec<String>> {
    use walkdir::WalkDir;
    
    let mut test_ids = Vec::new();
    
    for entry in WalkDir::new(test_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() && is_test_file(path) {
            if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                test_ids.push(name.to_string());
            }
        }
    }
    
    Ok(test_ids)
}

fn is_test_file(path: &std::path::Path) -> bool {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        name.starts_with("test_") || name.ends_with("_test") || 
        name.ends_with(".test.js") || name.ends_with(".test.ts") ||
        name.ends_with(".spec.js") || name.ends_with(".spec.ts")
    } else {
        false
    }
}

// Print functions

fn print_player_status(status: &PlayerStatus) {
    println!("\n🎮 {} Guardian Status {}", "=".repeat(20).bright_blue(), "=".repeat(20).bright_blue());
    println!();
    println!("👤 Player: {}", status.username.bright_cyan());
    println!("🎖️  Level: {} {}", 
        status.level.to_string().bright_yellow(),
        if let Some(rank) = status.rank {
            format!("(Rank #{})", rank).bright_magenta()
        } else {
            String::new()
        }
    );
    
    // XP progress bar
    let progress = status.xp as f64 / (status.xp + status.xp_to_next_level) as f64;
    let bar_width = 30;
    let filled = (progress * bar_width as f64) as usize;
    let empty = bar_width - filled;
    
    println!("✨ XP: {} / {} [{}{}]",
        status.xp.to_string().bright_green(),
        (status.xp + status.xp_to_next_level).to_string().bright_green(),
        "█".repeat(filled).bright_green(),
        "░".repeat(empty).dark_grey()
    );
    
    println!("🏆 Badges: {}", status.total_badges.to_string().bright_yellow());
    println!("📜 Active Quests: {}", status.active_quests.to_string().bright_cyan());
    println!("🔥 Streak: {} days", status.streak_days.to_string().bright_red());
    println!();
}

fn print_player_status_markdown(status: &PlayerStatus) {
    println!("# Guardian Status\n");
    println!("## Player Information\n");
    println!("- **Username**: {}", status.username);
    println!("- **Level**: {}", status.level);
    if let Some(rank) = status.rank {
        println!("- **Rank**: #{}", rank);
    }
    println!("- **XP**: {} / {}", status.xp, status.xp + status.xp_to_next_level);
    println!("- **Badges**: {}", status.total_badges);
    println!("- **Active Quests**: {}", status.active_quests);
    println!("- **Streak**: {} days", status.streak_days);
}

fn print_badges(badges: &[BadgeInfo]) {
    if badges.is_empty() {
        println!("\n🏆 No badges to display");
        return;
    }
    
    println!("\n🏆 {} Badges {}", "=".repeat(25).bright_blue(), "=".repeat(25).bright_blue());
    println!();
    
    let mut current_category = "";
    
    for badge in badges {
        if badge.category != current_category {
            current_category = &badge.category;
            println!("\n📁 {}", current_category.to_uppercase().bright_cyan());
            println!("{}", "-".repeat(60).dark_grey());
        }
        
        let rarity_color = match badge.rarity.as_str() {
            "common" => "white",
            "uncommon" => "green",
            "rare" => "blue",
            "epic" => "purple",
            "legendary" => "yellow",
            _ => "white",
        };
        
        let status = if badge.earned {
            "✅".to_string()
        } else if badge.progress > 0.0 {
            format!("{}%", (badge.progress * 100.0) as u32).bright_yellow().to_string()
        } else {
            "🔒".to_string()
        };
        
        println!("{} {} {} - {} (+{} XP)",
            status,
            badge.icon,
            badge.name.color(rarity_color).bold(),
            badge.description,
            badge.xp_reward
        );
    }
    println!();
}

fn print_badges_markdown(badges: &[BadgeInfo]) {
    println!("# Badges\n");
    
    if badges.is_empty() {
        println!("No badges to display.");
        return;
    }
    
    println!("| Status | Badge | Description | Category | Rarity | XP |");
    println!("|--------|-------|-------------|----------|--------|-----|");
    
    for badge in badges {
        let status = if badge.earned {
            "✅"
        } else if badge.progress > 0.0 {
            &format!("{}%", (badge.progress * 100.0) as u32)
        } else {
            "🔒"
        };
        
        println!("| {} | {} {} | {} | {} | {} | {} |",
            status,
            badge.icon,
            badge.name,
            badge.description,
            badge.category,
            badge.rarity,
            badge.xp_reward
        );
    }
}

fn print_quests(quests: &[QuestInfo]) {
    if quests.is_empty() {
        println!("\n📜 No quests to display");
        return;
    }
    
    println!("\n📜 {} Quests {}", "=".repeat(25).bright_blue(), "=".repeat(25).bright_blue());
    println!();
    
    for quest in quests {
        let status_icon = match quest.status.as_str() {
            "active" => "⚔️",
            "completed" => "✅",
            "failed" => "❌",
            _ => "❓",
        };
        
        println!("{} {} (+{} XP)",
            status_icon,
            quest.title.bright_cyan().bold(),
            quest.xp_reward
        );
        println!("   {}", quest.description.italic());
        println!("   Type: {} | Progress: {:.0}%",
            quest.quest_type.bright_yellow(),
            quest.progress
        );
        
        if let Some(deadline) = &quest.deadline {
            println!("   ⏰ Deadline: {}", deadline.bright_red());
        }
        println!();
    }
}

fn print_quests_markdown(quests: &[QuestInfo]) {
    println!("# Quests\n");
    
    if quests.is_empty() {
        println!("No quests to display.");
        return;
    }
    
    println!("| Status | Quest | Type | Progress | XP | Deadline |");
    println!("|--------|-------|------|----------|-----|----------|");
    
    for quest in quests {
        let status_icon = match quest.status.as_str() {
            "active" => "⚔️",
            "completed" => "✅",
            "failed" => "❌",
            _ => "❓",
        };
        
        println!("| {} | **{}**<br>{} | {} | {:.0}% | {} | {} |",
            status_icon,
            quest.title,
            quest.description,
            quest.quest_type,
            quest.progress,
            quest.xp_reward,
            quest.deadline.as_deref().unwrap_or("-")
        );
    }
}

fn print_test_crown(tests: &[TestImportance]) {
    if tests.is_empty() {
        println!("\n👑 No test importance data available");
        return;
    }
    
    println!("\n👑 {} Shapley Crown {}", "=".repeat(20).bright_yellow(), "=".repeat(20).bright_yellow());
    println!("\nTop {} Most Important Tests:\n", tests.len());
    
    for (idx, test) in tests.iter().enumerate() {
        let medal = match idx {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "🏅",
        };
        
        println!("{} #{} {} - {:.1}% contribution",
            medal,
            idx + 1,
            test.test_name.bright_cyan(),
            test.contribution_percentage
        );
        println!("   Shapley Value: {:.4} | Importance: {:.4}",
            test.shapley_value,
            test.importance_score
        );
        println!();
    }
}

fn print_test_crown_markdown(tests: &[TestImportance]) {
    println!("# Shapley Crown - Top Test Contributors\n");
    
    if tests.is_empty() {
        println!("No test importance data available.");
        return;
    }
    
    println!("| Rank | Test | Shapley Value | Importance | Contribution |");
    println!("|------|------|---------------|------------|--------------|");
    
    for (idx, test) in tests.iter().enumerate() {
        let medal = match idx {
            0 => "🥇",
            1 => "🥈", 
            2 => "🥉",
            _ => "🏅",
        };
        
        println!("| {} {} | {} | {:.4} | {:.4} | {:.1}% |",
            medal,
            idx + 1,
            test.test_name,
            test.shapley_value,
            test.importance_score,
            test.contribution_percentage
        );
    }
}
//! Database migrations

use sqlx::{Pool, Sqlite};
use anyhow::Result;

/// Run all database migrations
pub async fn run_migrations(pool: &Pool<Sqlite>) -> Result<()> {
    // Create tables
    create_players_table(pool).await?;
    create_runs_table(pool).await?;
    create_badges_table(pool).await?;
    create_player_badges_table(pool).await?;
    create_quests_table(pool).await?;
    create_agent_runs_table(pool).await?;
    create_evolution_results_table(pool).await?;
    create_test_importance_table(pool).await?;
    create_quality_trends_table(pool).await?;
    
    // Insert default badges
    insert_default_badges(pool).await?;
    
    Ok(())
}

async fn create_players_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT,
            display_name TEXT,
            avatar_url TEXT,
            xp INTEGER NOT NULL DEFAULT 0,
            level INTEGER NOT NULL DEFAULT 1,
            streak_days INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
        "#
    )
    .execute(pool)
    .await?;
    
    Ok(())
}

async fn create_runs_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            player_id INTEGER NOT NULL,
            project_path TEXT NOT NULL,
            quality_score REAL NOT NULL,
            quality_mode TEXT NOT NULL,
            risk_class TEXT,
            component_scores TEXT NOT NULL, -- JSON
            metadata TEXT, -- JSON
            created_at DATETIME NOT NULL,
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
        "#
    )
    .execute(pool)
    .await?;
    
    // Create indices
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_runs_player_id ON runs(player_id)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_runs_project_path ON runs(project_path)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
        .execute(pool)
        .await?;
    
    Ok(())
}

async fn create_badges_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS badges (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            icon TEXT NOT NULL,
            category TEXT NOT NULL,
            rarity TEXT NOT NULL,
            xp_reward INTEGER NOT NULL,
            criteria TEXT NOT NULL -- JSON
        )
        "#
    )
    .execute(pool)
    .await?;
    
    Ok(())
}

async fn create_player_badges_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS player_badges (
            id INTEGER PRIMARY KEY,
            player_id INTEGER NOT NULL,
            badge_id INTEGER NOT NULL,
            earned_at DATETIME NOT NULL,
            progress REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (player_id) REFERENCES players(id),
            FOREIGN KEY (badge_id) REFERENCES badges(id),
            UNIQUE(player_id, badge_id)
        )
        "#
    )
    .execute(pool)
    .await?;
    
    Ok(())
}

async fn create_quests_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS quests (
            id INTEGER PRIMARY KEY,
            player_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            quest_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            target_value REAL NOT NULL,
            current_value REAL NOT NULL DEFAULT 0.0,
            xp_reward INTEGER NOT NULL,
            deadline DATETIME,
            created_at DATETIME NOT NULL,
            completed_at DATETIME,
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
        "#
    )
    .execute(pool)
    .await?;
    
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_quests_player_status ON quests(player_id, status)")
        .execute(pool)
        .await?;
    
    Ok(())
}

async fn create_agent_runs_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            agent_type TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            impact_score REAL NOT NULL,
            xp_earned INTEGER NOT NULL,
            files_modified TEXT, -- JSON array
            metrics_before TEXT NOT NULL, -- JSON
            metrics_after TEXT NOT NULL, -- JSON
            created_at DATETIME NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
        "#
    )
    .execute(pool)
    .await?;
    
    Ok(())
}

async fn create_evolution_results_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS evolution_results (
            id INTEGER PRIMARY KEY,
            project_path TEXT NOT NULL,
            generations INTEGER NOT NULL,
            population_size INTEGER NOT NULL,
            pareto_front_size INTEGER NOT NULL,
            best_fitness TEXT NOT NULL, -- JSON
            optimized_suites TEXT NOT NULL, -- JSON
            improvement_metrics TEXT NOT NULL, -- JSON
            created_at DATETIME NOT NULL
        )
        "#
    )
    .execute(pool)
    .await?;
    
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_evolution_project_created ON evolution_results(project_path, created_at)")
        .execute(pool)
        .await?;
    
    Ok(())
}

async fn create_test_importance_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS test_importance (
            id INTEGER PRIMARY KEY,
            project_path TEXT NOT NULL,
            test_id TEXT NOT NULL,
            shapley_value REAL NOT NULL,
            importance_score REAL NOT NULL,
            confidence_lower REAL,
            confidence_upper REAL,
            calculated_at DATETIME NOT NULL
        )
        "#
    )
    .execute(pool)
    .await?;
    
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_test_importance_project ON test_importance(project_path, calculated_at)")
        .execute(pool)
        .await?;
    
    Ok(())
}

async fn create_quality_trends_table(pool: &Pool<Sqlite>) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS quality_trends (
            id INTEGER PRIMARY KEY,
            project_path TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            trend_direction TEXT NOT NULL,
            trend_slope REAL NOT NULL,
            prediction REAL,
            measured_at DATETIME NOT NULL
        )
        "#
    )
    .execute(pool)
    .await?;
    
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_quality_trends_project_metric ON quality_trends(project_path, metric_name, measured_at)")
        .execute(pool)
        .await?;
    
    Ok(())
}

async fn insert_default_badges(pool: &Pool<Sqlite>) -> Result<()> {
    let default_badges = vec![
        // Quality badges
        ("First Analysis", "Complete your first code analysis", "🔍", "quality", "common", 50, r#"{"analyses": 1}"#),
        ("Quality Champion", "Achieve A+ quality score", "🏆", "quality", "rare", 200, r#"{"min_score": 0.9}"#),
        ("Perfect Score", "Achieve 100% quality score", "💯", "quality", "legendary", 500, r#"{"min_score": 1.0}"#),
        
        // Evolution badges
        ("Evolution Explorer", "Run your first test evolution", "🧬", "evolution", "common", 100, r#"{"evolutions": 1}"#),
        ("Darwin Award", "Improve test suite by 50%", "🦕", "evolution", "epic", 300, r#"{"improvement": 0.5}"#),
        
        // Testing badges
        ("Mutation Master", "Achieve 90% mutation score", "🔬", "testing", "rare", 150, r#"{"mutation_score": 0.9}"#),
        ("Speed Demon", "Reduce test time by 30%", "⚡", "testing", "uncommon", 100, r#"{"speed_improvement": 0.3}"#),
        ("Coverage King", "Achieve 95% code coverage", "👑", "testing", "rare", 150, r#"{"coverage": 0.95}"#),
        
        // Consistency badges
        ("Daily Grind", "7-day analysis streak", "📅", "consistency", "uncommon", 100, r#"{"streak_days": 7}"#),
        ("Dedicated", "30-day analysis streak", "🔥", "consistency", "epic", 400, r#"{"streak_days": 30}"#),
        ("Obsessed", "100-day analysis streak", "💎", "consistency", "legendary", 1000, r#"{"streak_days": 100}"#),
        
        // Improvement badges
        ("Improver", "Improve quality score by 10%", "📈", "improvement", "common", 75, r#"{"improvement": 0.1}"#),
        ("Optimizer", "Fix 10 code issues", "🔧", "improvement", "uncommon", 125, r#"{"fixes": 10}"#),
        ("Refactor Hero", "Successfully refactor 5 modules", "♻️", "improvement", "rare", 200, r#"{"refactors": 5}"#),
    ];
    
    for (name, desc, icon, category, rarity, xp, criteria) in default_badges {
        sqlx::query(
            r#"
            INSERT OR IGNORE INTO badges (name, description, icon, category, rarity, xp_reward, criteria)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#
        )
        .bind(name)
        .bind(desc)
        .bind(icon)
        .bind(category)
        .bind(rarity)
        .bind(xp)
        .bind(criteria)
        .execute(pool)
        .await?;
    }
    
    Ok(())
}
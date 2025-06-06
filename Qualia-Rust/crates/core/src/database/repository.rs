//! Repository pattern implementation for database access

use super::models::*;
use super::DbPool;
use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, query_as};

/// Player repository
pub struct PlayerRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> PlayerRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Create a new player
    pub async fn create(&self, new_player: NewPlayer) -> Result<Player> {
        let now = Utc::now();
        let player = query_as!(
            Player,
            r#"
            INSERT INTO players (username, email, display_name, avatar_url, xp, level, streak_days, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, 0, 1, 0, ?5, ?6)
            RETURNING *
            "#,
            new_player.username,
            new_player.email,
            new_player.display_name,
            new_player.avatar_url,
            now,
            now
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(player)
    }
    
    /// Get player by ID
    pub async fn get_by_id(&self, id: i64) -> Result<Option<Player>> {
        let player = query_as!(
            Player,
            "SELECT * FROM players WHERE id = ?",
            id
        )
        .fetch_optional(self.pool)
        .await?;
        
        Ok(player)
    }
    
    /// Get player by username
    pub async fn get_by_username(&self, username: &str) -> Result<Option<Player>> {
        let player = query_as!(
            Player,
            "SELECT * FROM players WHERE username = ?",
            username
        )
        .fetch_optional(self.pool)
        .await?;
        
        Ok(player)
    }
    
    /// Update player XP and level
    pub async fn update_xp(&self, player_id: i64, xp_delta: i32) -> Result<Player> {
        let player = query_as!(
            Player,
            r#"
            UPDATE players 
            SET xp = xp + ?1,
                level = CAST((SQRT(xp + ?1) / 10) AS INTEGER) + 1,
                updated_at = ?2
            WHERE id = ?3
            RETURNING *
            "#,
            xp_delta,
            Utc::now(),
            player_id
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(player)
    }
    
    /// Get leaderboard
    pub async fn get_leaderboard(&self, limit: i64) -> Result<Vec<Player>> {
        let players = query_as!(
            Player,
            "SELECT * FROM players ORDER BY xp DESC LIMIT ?",
            limit
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(players)
    }
}

/// Run repository
pub struct RunRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> RunRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Create a new run
    pub async fn create(&self, new_run: NewRun) -> Result<Run> {
        let component_scores_json = serde_json::to_string(&new_run.component_scores)?;
        let metadata_json = new_run.metadata.map(|m| serde_json::to_string(&m)).transpose()?;
        
        let run = query_as!(
            Run,
            r#"
            INSERT INTO runs (player_id, project_path, quality_score, quality_mode, risk_class, component_scores, metadata, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            RETURNING *
            "#,
            new_run.player_id,
            new_run.project_path,
            new_run.quality_score,
            new_run.quality_mode,
            new_run.risk_class,
            component_scores_json,
            metadata_json,
            Utc::now()
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(run)
    }
    
    /// Get runs for a player
    pub async fn get_by_player(&self, player_id: i64, limit: i64) -> Result<Vec<Run>> {
        let runs = query_as!(
            Run,
            "SELECT * FROM runs WHERE player_id = ? ORDER BY created_at DESC LIMIT ?",
            player_id,
            limit
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(runs)
    }
    
    /// Get runs for a project
    pub async fn get_by_project(&self, project_path: &str, limit: i64) -> Result<Vec<Run>> {
        let runs = query_as!(
            Run,
            "SELECT * FROM runs WHERE project_path = ? ORDER BY created_at DESC LIMIT ?",
            project_path,
            limit
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(runs)
    }
    
    /// Get quality trend for a project
    pub async fn get_quality_trend(&self, project_path: &str, days: i64) -> Result<Vec<(f64, chrono::DateTime<Utc>)>> {
        #[derive(sqlx::FromRow)]
        struct QualityPoint {
            quality_score: f64,
            created_at: chrono::DateTime<Utc>,
        }
        
        let points = query_as!(
            QualityPoint,
            r#"
            SELECT quality_score, created_at 
            FROM runs 
            WHERE project_path = ? 
                AND created_at > datetime('now', '-' || ? || ' days')
            ORDER BY created_at ASC
            "#,
            project_path,
            days
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(points.into_iter().map(|p| (p.quality_score, p.created_at)).collect())
    }
}

/// Badge repository
pub struct BadgeRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> BadgeRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Get all badges
    pub async fn get_all(&self) -> Result<Vec<Badge>> {
        let badges = query_as!(Badge, "SELECT * FROM badges ORDER BY category, rarity")
            .fetch_all(self.pool)
            .await?;
        
        Ok(badges)
    }
    
    /// Get badges for a player
    pub async fn get_player_badges(&self, player_id: i64) -> Result<Vec<(Badge, PlayerBadge)>> {
        #[derive(sqlx::FromRow)]
        struct BadgeWithProgress {
            // Badge fields
            id: i64,
            name: String,
            description: String,
            icon: String,
            category: String,
            rarity: String,
            xp_reward: i32,
            criteria: String,
            // PlayerBadge fields
            pb_id: i64,
            pb_player_id: i64,
            pb_badge_id: i64,
            earned_at: chrono::DateTime<Utc>,
            progress: f64,
        }
        
        let results = sqlx::query_as::<_, BadgeWithProgress>(
            r#"
            SELECT 
                b.id, b.name, b.description, b.icon, b.category, b.rarity, b.xp_reward, b.criteria,
                pb.id as pb_id, pb.player_id as pb_player_id, pb.badge_id as pb_badge_id, 
                pb.earned_at, pb.progress
            FROM badges b
            INNER JOIN player_badges pb ON b.id = pb.badge_id
            WHERE pb.player_id = ?
            ORDER BY pb.earned_at DESC
            "#
        )
        .bind(player_id)
        .fetch_all(self.pool)
        .await?;
        
        let badges = results.into_iter().map(|r| {
            let badge = Badge {
                id: r.id,
                name: r.name,
                description: r.description,
                icon: r.icon,
                category: r.category,
                rarity: r.rarity,
                xp_reward: r.xp_reward,
                criteria: r.criteria,
            };
            let player_badge = PlayerBadge {
                id: r.pb_id,
                player_id: r.pb_player_id,
                badge_id: r.pb_badge_id,
                earned_at: r.earned_at,
                progress: r.progress,
            };
            (badge, player_badge)
        }).collect();
        
        Ok(badges)
    }
    
    /// Award badge to player
    pub async fn award_badge(&self, player_id: i64, badge_id: i64) -> Result<PlayerBadge> {
        let player_badge = query_as!(
            PlayerBadge,
            r#"
            INSERT INTO player_badges (player_id, badge_id, earned_at, progress)
            VALUES (?1, ?2, ?3, 1.0)
            RETURNING *
            "#,
            player_id,
            badge_id,
            Utc::now()
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(player_badge)
    }
}

/// Quest repository
pub struct QuestRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> QuestRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Create a new quest
    pub async fn create(&self, new_quest: NewQuest) -> Result<Quest> {
        let quest = query_as!(
            Quest,
            r#"
            INSERT INTO quests (player_id, title, description, quest_type, status, target_value, current_value, xp_reward, deadline, created_at)
            VALUES (?1, ?2, ?3, ?4, 'active', ?5, 0.0, ?6, ?7, ?8)
            RETURNING *
            "#,
            new_quest.player_id,
            new_quest.title,
            new_quest.description,
            new_quest.quest_type,
            new_quest.target_value,
            new_quest.xp_reward,
            new_quest.deadline,
            Utc::now()
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(quest)
    }
    
    /// Get active quests for a player
    pub async fn get_active(&self, player_id: i64) -> Result<Vec<Quest>> {
        let quests = query_as!(
            Quest,
            "SELECT * FROM quests WHERE player_id = ? AND status = 'active' ORDER BY created_at DESC",
            player_id
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(quests)
    }
    
    /// Update quest progress
    pub async fn update_progress(&self, quest_id: i64, progress: f64) -> Result<Quest> {
        let quest = query_as!(
            Quest,
            r#"
            UPDATE quests 
            SET current_value = ?1,
                status = CASE 
                    WHEN ?1 >= target_value THEN 'completed'
                    ELSE status
                END,
                completed_at = CASE
                    WHEN ?1 >= target_value THEN ?2
                    ELSE completed_at
                END
            WHERE id = ?3
            RETURNING *
            "#,
            progress,
            Utc::now(),
            quest_id
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(quest)
    }
}

/// Evolution results repository
pub struct EvolutionRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> EvolutionRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Save evolution result
    pub async fn create(&self, result: NewEvolutionResult) -> Result<EvolutionResult> {
        let best_fitness_json = serde_json::to_string(&result.best_fitness)?;
        let optimized_suites_json = serde_json::to_string(&result.optimized_suites)?;
        let improvement_metrics_json = serde_json::to_string(&result.improvement_metrics)?;
        
        let evolution_result = query_as!(
            EvolutionResult,
            r#"
            INSERT INTO evolution_results 
            (project_path, generations, population_size, pareto_front_size, best_fitness, optimized_suites, improvement_metrics, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            RETURNING *
            "#,
            result.project_path,
            result.generations,
            result.population_size,
            result.pareto_front_size,
            best_fitness_json,
            optimized_suites_json,
            improvement_metrics_json,
            Utc::now()
        )
        .fetch_one(self.pool)
        .await?;
        
        Ok(evolution_result)
    }
    
    /// Get latest evolution result for project
    pub async fn get_latest(&self, project_path: &str) -> Result<Option<EvolutionResult>> {
        let result = query_as!(
            EvolutionResult,
            "SELECT * FROM evolution_results WHERE project_path = ? ORDER BY created_at DESC LIMIT 1",
            project_path
        )
        .fetch_optional(self.pool)
        .await?;
        
        Ok(result)
    }
}

/// Test importance repository
pub struct TestImportanceRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> TestImportanceRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }
    
    /// Save test importance values
    pub async fn save_batch(&self, project_path: &str, importance_values: Vec<NewTestImportance>) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        
        for value in importance_values {
            query!(
                r#"
                INSERT INTO test_importance 
                (project_path, test_id, shapley_value, importance_score, confidence_lower, confidence_upper, calculated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                project_path,
                value.test_id,
                value.shapley_value,
                value.importance_score,
                value.confidence_lower,
                value.confidence_upper,
                Utc::now()
            )
            .execute(&mut *tx)
            .await?;
        }
        
        tx.commit().await?;
        Ok(())
    }
    
    /// Get most important tests
    pub async fn get_top_tests(&self, project_path: &str, limit: i64) -> Result<Vec<TestImportance>> {
        let tests = query_as!(
            TestImportance,
            r#"
            SELECT * FROM test_importance 
            WHERE project_path = ? 
                AND calculated_at = (
                    SELECT MAX(calculated_at) 
                    FROM test_importance 
                    WHERE project_path = ?
                )
            ORDER BY importance_score DESC
            LIMIT ?
            "#,
            project_path,
            project_path,
            limit
        )
        .fetch_all(self.pool)
        .await?;
        
        Ok(tests)
    }
}

/// Unified repository container
pub struct Repository<'a> {
    pub players: PlayerRepository<'a>,
    pub runs: RunRepository<'a>,
    pub badges: BadgeRepository<'a>,
    pub quests: QuestRepository<'a>,
    pub evolution: EvolutionRepository<'a>,
    pub test_importance: TestImportanceRepository<'a>,
}

impl<'a> Repository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self {
            players: PlayerRepository::new(pool),
            runs: RunRepository::new(pool),
            badges: BadgeRepository::new(pool),
            quests: QuestRepository::new(pool),
            evolution: EvolutionRepository::new(pool),
            test_importance: TestImportanceRepository::new(pool),
        }
    }
}
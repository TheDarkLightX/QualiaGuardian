//! Repository pattern implementation for database access (runtime queries)

use super::models::*;
use super::DbPool;
use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, query_as, Row};
use serde_json;

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
        let result = query(
            r#"
            INSERT INTO players (username, email, display_name, avatar_url, xp, level, streak_days, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, 0, 1, 0, ?5, ?6)
            "#
        )
        .bind(&new_player.username)
        .bind(&new_player.email)
        .bind(&new_player.display_name)
        .bind(&new_player.avatar_url)
        .bind(&now)
        .bind(&now)
        .execute(self.pool)
        .await?;
        
        let id = result.last_insert_rowid();
        
        Ok(Player {
            id,
            username: new_player.username,
            email: new_player.email,
            display_name: new_player.display_name,
            avatar_url: new_player.avatar_url,
            xp: 0,
            level: 1,
            streak_days: 0,
            created_at: now,
            updated_at: now,
        })
    }
    
    /// Get player by ID
    pub async fn get_by_id(&self, id: i64) -> Result<Option<Player>> {
        let player = query_as::<_, Player>("SELECT * FROM players WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool)
            .await?;
        
        Ok(player)
    }
    
    /// Get player by username
    pub async fn get_by_username(&self, username: &str) -> Result<Option<Player>> {
        let player = query_as::<_, Player>("SELECT * FROM players WHERE username = ?")
            .bind(username)
            .fetch_optional(self.pool)
            .await?;
        
        Ok(player)
    }
    
    /// Update player XP and level
    pub async fn update_xp(&self, player_id: i64, xp_delta: i32) -> Result<Player> {
        let now = Utc::now();
        query(
            r#"
            UPDATE players 
            SET xp = xp + ?1,
                level = CAST((SQRT(xp + ?1) / 10) AS INTEGER) + 1,
                updated_at = ?2
            WHERE id = ?3
            "#
        )
        .bind(xp_delta)
        .bind(&now)
        .bind(player_id)
        .execute(self.pool)
        .await?;
        
        self.get_by_id(player_id).await?.ok_or_else(|| anyhow::anyhow!("Player not found"))
    }
    
    /// Get leaderboard
    pub async fn get_leaderboard(&self, limit: i64) -> Result<Vec<Player>> {
        let players = query_as::<_, Player>("SELECT * FROM players ORDER BY xp DESC LIMIT ?")
            .bind(limit)
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
        let now = Utc::now();
        
        let result = query(
            r#"
            INSERT INTO runs (player_id, project_path, quality_score, quality_mode, risk_class, component_scores, metadata, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#
        )
        .bind(new_run.player_id)
        .bind(&new_run.project_path)
        .bind(new_run.quality_score)
        .bind(&new_run.quality_mode)
        .bind(&new_run.risk_class)
        .bind(&component_scores_json)
        .bind(&metadata_json)
        .bind(&now)
        .execute(self.pool)
        .await?;
        
        let id = result.last_insert_rowid();
        
        Ok(Run {
            id,
            player_id: new_run.player_id,
            project_path: new_run.project_path,
            quality_score: new_run.quality_score,
            quality_mode: new_run.quality_mode,
            risk_class: new_run.risk_class,
            component_scores: component_scores_json,
            metadata: metadata_json,
            created_at: now,
        })
    }
    
    /// Get runs for a player
    pub async fn get_by_player(&self, player_id: i64, limit: i64) -> Result<Vec<Run>> {
        let runs = query_as::<_, Run>(
            "SELECT * FROM runs WHERE player_id = ? ORDER BY created_at DESC LIMIT ?"
        )
        .bind(player_id)
        .bind(limit)
        .fetch_all(self.pool)
        .await?;
        
        Ok(runs)
    }
    
    /// Get runs for a project
    pub async fn get_by_project(&self, project_path: &str, limit: i64) -> Result<Vec<Run>> {
        let runs = query_as::<_, Run>(
            "SELECT * FROM runs WHERE project_path = ? ORDER BY created_at DESC LIMIT ?"
        )
        .bind(project_path)
        .bind(limit)
        .fetch_all(self.pool)
        .await?;
        
        Ok(runs)
    }
    
    /// Get quality trend for a project
    pub async fn get_quality_trend(&self, project_path: &str, days: i64) -> Result<Vec<(f64, DateTime<Utc>)>> {
        let rows = query(
            r#"
            SELECT quality_score, created_at 
            FROM runs 
            WHERE project_path = ? 
                AND created_at > datetime('now', '-' || ? || ' days')
            ORDER BY created_at ASC
            "#
        )
        .bind(project_path)
        .bind(days)
        .fetch_all(self.pool)
        .await?;
        
        let mut points = Vec::new();
        for row in rows {
            let score: f64 = row.get("quality_score");
            let created_at: DateTime<Utc> = row.get("created_at");
            points.push((score, created_at));
        }
        
        Ok(points)
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
        let badges = query_as::<_, Badge>("SELECT * FROM badges ORDER BY category, rarity")
            .fetch_all(self.pool)
            .await?;
        
        Ok(badges)
    }
    
    /// Get badges for a player
    pub async fn get_player_badges(&self, player_id: i64) -> Result<Vec<(Badge, PlayerBadge)>> {
        let rows = query(
            r#"
            SELECT 
                b.*, pb.*
            FROM badges b
            INNER JOIN player_badges pb ON b.id = pb.badge_id
            WHERE pb.player_id = ?
            ORDER BY pb.earned_at DESC
            "#
        )
        .bind(player_id)
        .fetch_all(self.pool)
        .await?;
        
        let mut results = Vec::new();
        for row in rows {
            let badge = Badge {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                icon: row.get("icon"),
                category: row.get("category"),
                rarity: row.get("rarity"),
                xp_reward: row.get("xp_reward"),
                criteria: row.get("criteria"),
            };
            
            let player_badge = PlayerBadge {
                id: row.get("id"),
                player_id: row.get("player_id"),
                badge_id: row.get("badge_id"),
                earned_at: row.get("earned_at"),
                progress: row.get("progress"),
            };
            
            results.push((badge, player_badge));
        }
        
        Ok(results)
    }
    
    /// Award badge to player
    pub async fn award_badge(&self, player_id: i64, badge_id: i64) -> Result<PlayerBadge> {
        let now = Utc::now();
        let result = query(
            r#"
            INSERT INTO player_badges (player_id, badge_id, earned_at, progress)
            VALUES (?1, ?2, ?3, 1.0)
            "#
        )
        .bind(player_id)
        .bind(badge_id)
        .bind(&now)
        .execute(self.pool)
        .await?;
        
        let id = result.last_insert_rowid();
        
        Ok(PlayerBadge {
            id,
            player_id,
            badge_id,
            earned_at: now,
            progress: 1.0,
        })
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
        let now = Utc::now();
        let result = query(
            r#"
            INSERT INTO quests (player_id, title, description, quest_type, status, target_value, current_value, xp_reward, deadline, created_at)
            VALUES (?1, ?2, ?3, ?4, 'active', ?5, 0.0, ?6, ?7, ?8)
            "#
        )
        .bind(new_quest.player_id)
        .bind(&new_quest.title)
        .bind(&new_quest.description)
        .bind(&new_quest.quest_type)
        .bind(new_quest.target_value)
        .bind(new_quest.xp_reward)
        .bind(&new_quest.deadline)
        .bind(&now)
        .execute(self.pool)
        .await?;
        
        let id = result.last_insert_rowid();
        
        Ok(Quest {
            id,
            player_id: new_quest.player_id,
            title: new_quest.title,
            description: new_quest.description,
            quest_type: new_quest.quest_type,
            status: "active".to_string(),
            target_value: new_quest.target_value,
            current_value: 0.0,
            xp_reward: new_quest.xp_reward,
            deadline: new_quest.deadline,
            created_at: now,
            completed_at: None,
        })
    }
    
    /// Get active quests for a player
    pub async fn get_active(&self, player_id: i64) -> Result<Vec<Quest>> {
        let quests = query_as::<_, Quest>(
            "SELECT * FROM quests WHERE player_id = ? AND status = 'active' ORDER BY created_at DESC"
        )
        .bind(player_id)
        .fetch_all(self.pool)
        .await?;
        
        Ok(quests)
    }
    
    /// Update quest progress
    pub async fn update_progress(&self, quest_id: i64, progress: f64) -> Result<Quest> {
        let now = Utc::now();
        query(
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
            "#
        )
        .bind(progress)
        .bind(&now)
        .bind(quest_id)
        .execute(self.pool)
        .await?;
        
        query_as::<_, Quest>("SELECT * FROM quests WHERE id = ?")
            .bind(quest_id)
            .fetch_one(self.pool)
            .await
            .map_err(Into::into)
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
        let now = Utc::now();
        
        let db_result = query(
            r#"
            INSERT INTO evolution_results 
            (project_path, generations, population_size, pareto_front_size, best_fitness, optimized_suites, improvement_metrics, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#
        )
        .bind(&result.project_path)
        .bind(result.generations)
        .bind(result.population_size)
        .bind(result.pareto_front_size)
        .bind(&best_fitness_json)
        .bind(&optimized_suites_json)
        .bind(&improvement_metrics_json)
        .bind(&now)
        .execute(self.pool)
        .await?;
        
        let id = db_result.last_insert_rowid();
        
        Ok(EvolutionResult {
            id,
            project_path: result.project_path,
            generations: result.generations,
            population_size: result.population_size,
            pareto_front_size: result.pareto_front_size,
            best_fitness: best_fitness_json,
            optimized_suites: optimized_suites_json,
            improvement_metrics: improvement_metrics_json,
            created_at: now,
        })
    }
    
    /// Get latest evolution result for project
    pub async fn get_latest(&self, project_path: &str) -> Result<Option<EvolutionResult>> {
        let result = query_as::<_, EvolutionResult>(
            "SELECT * FROM evolution_results WHERE project_path = ? ORDER BY created_at DESC LIMIT 1"
        )
        .bind(project_path)
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
        let now = Utc::now();
        
        for value in importance_values {
            query(
                r#"
                INSERT INTO test_importance 
                (project_path, test_id, shapley_value, importance_score, confidence_lower, confidence_upper, calculated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#
            )
            .bind(project_path)
            .bind(&value.test_id)
            .bind(value.shapley_value)
            .bind(value.importance_score)
            .bind(value.confidence_lower)
            .bind(value.confidence_upper)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }
        
        tx.commit().await?;
        Ok(())
    }
    
    /// Get most important tests
    pub async fn get_top_tests(&self, project_path: &str, limit: i64) -> Result<Vec<TestImportance>> {
        let tests = query_as::<_, TestImportance>(
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
            "#
        )
        .bind(project_path)
        .bind(project_path)
        .bind(limit)
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
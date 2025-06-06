//! Database models

use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use chrono::{DateTime, Utc};

/// Player/User model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Player {
    pub id: i64,
    pub username: String,
    pub email: Option<String>,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub xp: i64,
    pub level: i32,
    pub streak_days: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Analysis run record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Run {
    pub id: i64,
    pub player_id: i64,
    pub project_path: String,
    pub quality_score: f64,
    pub quality_mode: String,
    pub risk_class: Option<String>,
    pub component_scores: String, // JSON
    pub metadata: Option<String>, // JSON
    pub created_at: DateTime<Utc>,
}

/// Badge/Achievement model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Badge {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub category: String,
    pub rarity: String,
    pub xp_reward: i32,
    pub criteria: String, // JSON
}

/// Player badge association
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PlayerBadge {
    pub id: i64,
    pub player_id: i64,
    pub badge_id: i64,
    pub earned_at: DateTime<Utc>,
    pub progress: f64,
}

/// Quest/Goal model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Quest {
    pub id: i64,
    pub player_id: i64,
    pub title: String,
    pub description: String,
    pub quest_type: String,
    pub status: String,
    pub target_value: f64,
    pub current_value: f64,
    pub xp_reward: i32,
    pub deadline: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Agent run record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AgentRun {
    pub id: i64,
    pub run_id: i64,
    pub agent_type: String,
    pub action_taken: String,
    pub reasoning: String,
    pub impact_score: f64,
    pub xp_earned: i32,
    pub files_modified: Option<String>, // JSON array
    pub metrics_before: String, // JSON
    pub metrics_after: String, // JSON
    pub created_at: DateTime<Utc>,
}

/// Evolution result record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct EvolutionResult {
    pub id: i64,
    pub project_path: String,
    pub generations: i32,
    pub population_size: i32,
    pub pareto_front_size: i32,
    pub best_fitness: String, // JSON
    pub optimized_suites: String, // JSON
    pub improvement_metrics: String, // JSON
    pub created_at: DateTime<Utc>,
}

/// Test importance record (Shapley values)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TestImportance {
    pub id: i64,
    pub project_path: String,
    pub test_id: String,
    pub shapley_value: f64,
    pub importance_score: f64,
    pub confidence_lower: Option<f64>,
    pub confidence_upper: Option<f64>,
    pub calculated_at: DateTime<Utc>,
}

/// Quality trend record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct QualityTrend {
    pub id: i64,
    pub project_path: String,
    pub metric_name: String,
    pub value: f64,
    pub trend_direction: String,
    pub trend_slope: f64,
    pub prediction: Option<f64>,
    pub measured_at: DateTime<Utc>,
}

// Creation structs for inserting new records

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewPlayer {
    pub username: String,
    pub email: Option<String>,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRun {
    pub player_id: i64,
    pub project_path: String,
    pub quality_score: f64,
    pub quality_mode: String,
    pub risk_class: Option<String>,
    pub component_scores: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewQuest {
    pub player_id: i64,
    pub title: String,
    pub description: String,
    pub quest_type: String,
    pub target_value: f64,
    pub xp_reward: i32,
    pub deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewAgentRun {
    pub run_id: i64,
    pub agent_type: String,
    pub action_taken: String,
    pub reasoning: String,
    pub impact_score: f64,
    pub xp_earned: i32,
    pub files_modified: Option<Vec<String>>,
    pub metrics_before: serde_json::Value,
    pub metrics_after: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewEvolutionResult {
    pub project_path: String,
    pub generations: i32,
    pub population_size: i32,
    pub pareto_front_size: i32,
    pub best_fitness: serde_json::Value,
    pub optimized_suites: serde_json::Value,
    pub improvement_metrics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewTestImportance {
    pub project_path: String,
    pub test_id: String,
    pub shapley_value: f64,
    pub importance_score: f64,
    pub confidence_lower: Option<f64>,
    pub confidence_upper: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewQualityTrend {
    pub project_path: String,
    pub metric_name: String,
    pub value: f64,
    pub trend_direction: String,
    pub trend_slope: f64,
    pub prediction: Option<f64>,
}
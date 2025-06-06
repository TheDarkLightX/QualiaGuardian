//! Database layer for Guardian

use anyhow::Result;
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use std::path::PathBuf;

/// Database connection pool
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Create a new database connection
    pub async fn new(db_path: Option<PathBuf>) -> Result<Self> {
        let path = db_path.unwrap_or_else(|| {
            let mut path = directories::ProjectDirs::from("com", "guardian", "guardian")
                .map(|dirs| dirs.data_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            path.push("guardian.db");
            path
        });
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let db_url = format!("sqlite://{}", path.display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;
        
        Ok(Self { pool })
    }
    
    /// Run migrations
    pub async fn migrate(&self) -> Result<()> {
        // TODO: Add SQLx migrations
        Ok(())
    }
    
    /// Get the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
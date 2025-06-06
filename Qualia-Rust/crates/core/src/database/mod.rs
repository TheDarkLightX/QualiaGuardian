//! Database layer for persistence

pub mod models;
pub mod repository_runtime;
pub use repository_runtime as repository;
pub mod migrations;

pub use models::*;
pub use repository::*;

use sqlx::{Pool, Sqlite, SqlitePool};
use std::path::Path;
use anyhow::Result;

/// Database connection pool type
pub type DbPool = Pool<Sqlite>;

/// Initialize database connection
pub async fn init_database(db_path: impl AsRef<Path>) -> Result<DbPool> {
    let db_url = format!("sqlite://{}", db_path.as_ref().display());
    let pool = SqlitePool::connect(&db_url).await?;
    
    // Run migrations
    migrations::run_migrations(&pool).await?;
    
    Ok(pool)
}

/// Create in-memory database for testing
pub async fn create_test_db() -> Result<DbPool> {
    let pool = SqlitePool::connect("sqlite::memory:").await?;
    migrations::run_migrations(&pool).await?;
    Ok(pool)
}
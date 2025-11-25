//! Core quality metrics for `QualiaGuardian`
//! 
//! This crate provides the fundamental quality scoring systems:
//! - TES (Test Effectiveness Score)
//! - bE-TES (Bounded Evolutionary TES)
//! - OSQI (Overall Software Quality Index)

pub mod error;
pub mod types;
pub mod config;
pub mod tes;
pub mod betes;
pub mod osqi;
pub mod traits;
pub mod database;
pub mod verification;

// Re-export commonly used types
pub use error::{QualityError, Result};
pub use types::{QualityScore, RiskClass, QualityGrade};
pub use config::{QualityConfig, QualityMode};
pub use traits::QualityMetric;
pub use database::{DbPool, init_database, create_test_db};
pub use verification::{
    verify_bounded, verify_monotonicity, verify_continuity,
    verify_idempotency, verify_geometric_mean_properties,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
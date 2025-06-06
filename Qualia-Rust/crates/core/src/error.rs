//! Error types for quality metrics

use thiserror::Error;

/// Errors that can occur during quality metric calculation
#[derive(Debug, Error)]
pub enum QualityError {
    /// Score value is outside the valid range [0.0, 1.0]
    #[error("Score {0} is outside valid range [0.0, 1.0]")]
    ScoreOutOfBounds(f64),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Missing required input data
    #[error("Missing required input: {0}")]
    MissingInput(String),
    
    /// Invalid input value
    #[error("Invalid input value for {field}: {value}")]
    InvalidInput { field: String, value: String },
    
    /// Calculation error
    #[error("Calculation error: {0}")]
    Calculation(String),
    
    /// IO error (e.g., reading config files)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// YAML parsing error
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    
    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for quality operations
pub type Result<T> = std::result::Result<T, QualityError>;
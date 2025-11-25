//! Error types for quality metrics
//!
//! This module provides comprehensive error handling for the quality metrics system.
//! All errors use `thiserror` for automatic `Error` trait implementation and
//! `anyhow`-compatible error handling.

use thiserror::Error;

/// Errors that can occur during quality metric calculation
///
/// This enum provides detailed error information for debugging and user feedback.
/// All variants implement `std::error::Error` via `thiserror`.
#[derive(Debug, Error)]
pub enum QualityError {
    /// Score value is outside the valid range [0.0, 1.0]
    ///
    /// This error occurs when attempting to create a `QualityScore` with
    /// a value outside the valid range.
    #[error("Score {0} is outside valid range [0.0, 1.0]")]
    ScoreOutOfBounds(f64),
    
    /// Configuration error
    ///
    /// Used for errors in configuration parsing or validation.
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Missing required input data
    ///
    /// Indicates that a required field is missing from the input data.
    #[error("Missing required input: {0}")]
    MissingInput(String),
    
    /// Invalid input value
    ///
    /// Used when an input value is present but invalid (e.g., out of range).
    #[error("Invalid input value for {field}: {value}")]
    InvalidInput { 
        /// The field name that has an invalid value
        field: String, 
        /// The invalid value
        value: String 
    },
    
    /// Calculation error
    ///
    /// Used for errors during mathematical calculations (e.g., division by zero,
    /// overflow, NaN results).
    #[error("Calculation error: {0}")]
    Calculation(String),
    
    /// IO error (e.g., reading config files)
    ///
    /// Wraps `std::io::Error` for file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// YAML parsing error
    ///
    /// Wraps `serde_yaml::Error` for YAML configuration parsing.
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    
    /// JSON parsing error
    ///
    /// Wraps `serde_json::Error` for JSON data parsing.
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// Database error
    ///
    /// Used for database-related errors (e.g., connection failures, query errors).
    #[error("Database error: {0}")]
    Database(String),
    
    /// Validation error
    ///
    /// Used when data validation fails (e.g., invalid ranges, missing required fields).
    #[error("Validation error: {0}")]
    Validation(String),
}

/// Result type alias for quality operations
///
/// This is a convenience alias for `Result<T, QualityError>`.
pub type Result<T> = std::result::Result<T, QualityError>;

impl QualityError {
    /// Check if this error is recoverable
    ///
    /// Some errors (like invalid input) might be recoverable by clamping or
    /// using default values, while others (like IO errors) are not.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            QualityError::ScoreOutOfBounds(_) | QualityError::InvalidInput { .. }
        )
    }
    
    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            QualityError::ScoreOutOfBounds(val) => {
                format!("Quality score must be between 0.0 and 1.0, got {}", val)
            }
            QualityError::Config(msg) => format!("Configuration error: {}", msg),
            QualityError::MissingInput(field) => {
                format!("Missing required input: {}", field)
            }
            QualityError::InvalidInput { field, value } => {
                format!("Invalid value for {}: {}", field, value)
            }
            QualityError::Calculation(msg) => format!("Calculation error: {}", msg),
            QualityError::Io(e) => format!("File operation failed: {}", e),
            QualityError::Yaml(e) => format!("YAML parsing failed: {}", e),
            QualityError::Json(e) => format!("JSON parsing failed: {}", e),
            QualityError::Database(msg) => format!("Database error: {}", msg),
            QualityError::Validation(msg) => format!("Validation failed: {}", msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_messages() {
        let err = QualityError::ScoreOutOfBounds(1.5);
        assert!(err.to_string().contains("1.5"));
        assert!(err.is_recoverable());
        
        let err = QualityError::InvalidInput {
            field: "mutation_score".to_string(),
            value: "1.5".to_string(),
        };
        assert!(err.to_string().contains("mutation_score"));
        assert!(err.is_recoverable());
        
        let err = QualityError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found",
        ));
        assert!(!err.is_recoverable());
    }
    
    #[test]
    fn test_error_conversions() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let quality_err: QualityError = io_err.into();
        assert!(matches!(quality_err, QualityError::Io(_)));
    }
}

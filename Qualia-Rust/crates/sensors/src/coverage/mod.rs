//! Coverage parsing utilities

pub mod lcov;
pub mod coverage_py;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic coverage data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageData {
    /// File path to coverage mapping
    pub files: HashMap<String, FileCoverage>,
    /// Total line coverage percentage
    pub line_coverage: f64,
    /// Total branch coverage percentage
    pub branch_coverage: f64,
    /// Critical paths identified
    pub critical_paths: Vec<CriticalPath>,
}

/// Coverage data for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    /// Lines covered
    pub covered_lines: Vec<u32>,
    /// Lines not covered
    pub uncovered_lines: Vec<u32>,
    /// Total lines
    pub total_lines: u32,
    /// Branch coverage info
    pub branches: Vec<BranchCoverage>,
}

/// Branch coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCoverage {
    /// Line number
    pub line: u32,
    /// Branch ID
    pub branch_id: String,
    /// Whether the branch was taken
    pub taken: bool,
}

/// Critical path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    /// Path identifier
    pub id: String,
    /// Files involved in this path
    pub files: Vec<String>,
    /// Whether this path is covered
    pub covered: bool,
    /// Importance score (0-1)
    pub importance: f64,
}
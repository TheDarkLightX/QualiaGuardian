//! Behavior coverage sensor for critical path coverage

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Coverage data structures and parsers
mod coverage {
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
    
    pub mod lcov {
        use super::*;
        use crate::{Result, SensorError};
        use std::fs;
        use std::path::Path;

        /// LCOV parser for coverage data
        pub struct LcovParser;

        impl LcovParser {
            /// Parse an LCOV file
            pub fn parse_file(path: &Path) -> Result<CoverageData> {
                let content = fs::read_to_string(path)
                    .map_err(|e| SensorError::Io(e))?;
                
                Self::parse_content(&content)
            }
            
            /// Parse LCOV content
            pub fn parse_content(content: &str) -> Result<CoverageData> {
                let mut files = HashMap::new();
                let mut current_file: Option<String> = None;
                let mut current_coverage = FileCoverage {
                    covered_lines: Vec::new(),
                    uncovered_lines: Vec::new(),
                    total_lines: 0,
                    branches: Vec::new(),
                };
                
                for line in content.lines() {
                    let line = line.trim();
                    
                    if line.starts_with("SF:") {
                        // Source file
                        if let Some(file) = current_file.take() {
                            files.insert(file, current_coverage.clone());
                        }
                        current_file = Some(line[3..].to_string());
                        current_coverage = FileCoverage {
                            covered_lines: Vec::new(),
                            uncovered_lines: Vec::new(),
                            total_lines: 0,
                            branches: Vec::new(),
                        };
                    } else if line.starts_with("DA:") {
                        // Line coverage data
                        let parts: Vec<&str> = line[3..].split(',').collect();
                        if parts.len() >= 2 {
                            if let Ok(line_num) = parts[0].parse::<u32>() {
                                if let Ok(hit_count) = parts[1].parse::<u32>() {
                                    if hit_count > 0 {
                                        current_coverage.covered_lines.push(line_num);
                                    } else {
                                        current_coverage.uncovered_lines.push(line_num);
                                    }
                                }
                            }
                        }
                    } else if line.starts_with("BRDA:") {
                        // Branch coverage data
                        let parts: Vec<&str> = line[5..].split(',').collect();
                        if parts.len() >= 4 {
                            if let Ok(line_num) = parts[0].parse::<u32>() {
                                let branch_id = format!("{}-{}", parts[1], parts[2]);
                                let taken = parts[3] != "-" && parts[3] != "0";
                                current_coverage.branches.push(BranchCoverage {
                                    line: line_num,
                                    branch_id,
                                    taken,
                                });
                            }
                        }
                    } else if line.starts_with("LF:") {
                        // Total lines
                        if let Ok(total) = line[3..].parse::<u32>() {
                            current_coverage.total_lines = total;
                        }
                    }
                }
                
                // Don't forget the last file
                if let Some(file) = current_file {
                    files.insert(file, current_coverage);
                }
                
                // Calculate overall metrics
                let (line_coverage, branch_coverage) = Self::calculate_overall_coverage(&files);
                
                Ok(CoverageData {
                    files,
                    line_coverage,
                    branch_coverage,
                })
            }
            
            /// Calculate overall coverage percentages
            fn calculate_overall_coverage(files: &HashMap<String, FileCoverage>) -> (f64, f64) {
                let mut total_lines = 0u32;
                let mut covered_lines = 0u32;
                let mut total_branches = 0u32;
                let mut covered_branches = 0u32;
                
                for coverage in files.values() {
                    total_lines += coverage.covered_lines.len() as u32 + coverage.uncovered_lines.len() as u32;
                    covered_lines += coverage.covered_lines.len() as u32;
                    
                    total_branches += coverage.branches.len() as u32;
                    covered_branches += coverage.branches.iter().filter(|b| b.taken).count() as u32;
                }
                
                let line_coverage = if total_lines > 0 {
                    (covered_lines as f64 / total_lines as f64) * 100.0
                } else {
                    0.0
                };
                
                let branch_coverage = if total_branches > 0 {
                    (covered_branches as f64 / total_branches as f64) * 100.0
                } else {
                    0.0
                };
                
                (line_coverage, branch_coverage)
            }
        }
    }
}

/// Behavior coverage output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorCoverageOutput {
    /// Total critical behaviors identified
    pub total_critical_behaviors: usize,
    /// Number of covered critical behaviors
    pub covered_critical_behaviors: usize,
    /// Coverage ratio (0.0 to 1.0)
    pub coverage_ratio: f64,
    /// Line coverage percentage
    pub line_coverage: f64,
    /// Branch coverage percentage
    pub branch_coverage: f64,
    /// Coverage files found
    pub coverage_files: Vec<String>,
}

/// Behavior coverage sensor
#[derive(Debug)]
pub struct BehaviorCoverageSensor {
    /// Patterns to identify critical code
    critical_patterns: Vec<String>,
}

impl BehaviorCoverageSensor {
    pub fn new() -> Self {
        Self {
            critical_patterns: vec![
                "error".to_string(),
                "panic".to_string(),
                "auth".to_string(),
                "security".to_string(),
                "payment".to_string(),
                "database".to_string(),
                "api".to_string(),
            ],
        }
    }
    
    /// Find coverage files in the project
    async fn find_coverage_files(&self, project_path: &str) -> Result<Vec<PathBuf>> {
        let mut coverage_files = Vec::new();
        
        for entry in WalkDir::new(project_path)
            .max_depth(5)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".lcov") || 
                   name.ends_with(".info") || 
                   name == "lcov.info" ||
                   name == "coverage.lcov" {
                    coverage_files.push(path.to_path_buf());
                }
            }
        }
        
        Ok(coverage_files)
    }
    
    /// Identify critical behaviors from coverage data
    fn identify_critical_behaviors(&self, coverage: &coverage::CoverageData) -> (usize, usize) {
        let mut total_critical = 0;
        let mut covered_critical = 0;
        
        for (file_path, file_cov) in &coverage.files {
            // Check if file is critical based on patterns
            let is_critical_file = self.critical_patterns.iter()
                .any(|pattern| file_path.to_lowercase().contains(pattern));
            
            if is_critical_file {
                // Count critical lines (simplified - in real implementation would analyze AST)
                total_critical += file_cov.covered_lines.len() + file_cov.uncovered_lines.len();
                covered_critical += file_cov.covered_lines.len();
            }
        }
        
        // If no critical files found, use overall coverage as proxy
        if total_critical == 0 {
            total_critical = 100; // Default assumption
            covered_critical = (coverage.line_coverage as usize).min(100);
        }
        
        (total_critical, covered_critical)
    }
}

impl Default for BehaviorCoverageSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for BehaviorCoverageSensor {
    type Output = BehaviorCoverageOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let coverage_files = self.find_coverage_files(&context.project_path).await?;
        
        if coverage_files.is_empty() {
            return Err(SensorError::Generic(
                "No coverage files found. Please run tests with coverage enabled.".to_string()
            ));
        }
        
        // Parse the first coverage file found
        let coverage_data = coverage::lcov::LcovParser::parse_file(&coverage_files[0])?;
        
        // Identify critical behaviors
        let (total_critical, covered_critical) = self.identify_critical_behaviors(&coverage_data);
        let coverage_ratio = if total_critical > 0 {
            covered_critical as f64 / total_critical as f64
        } else {
            0.0
        };
        
        Ok(BehaviorCoverageOutput {
            total_critical_behaviors: total_critical,
            covered_critical_behaviors: covered_critical,
            coverage_ratio,
            line_coverage: coverage_data.line_coverage / 100.0, // Convert to 0-1 range
            branch_coverage: coverage_data.branch_coverage / 100.0,
            coverage_files: coverage_files.iter()
                .map(|p| p.display().to_string())
                .collect(),
        })
    }
    
    fn name(&self) -> &'static str {
        "behavior_coverage"
    }
}
//! LCOV format parser

use super::{CoverageData, FileCoverage, BranchCoverage};
use crate::{Result, SensorError};
use std::collections::HashMap;
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
            critical_paths: Vec::new(), // TODO: Identify critical paths
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lcov_parsing() {
        let lcov_content = r#"
SF:src/main.rs
DA:1,1
DA:2,1
DA:3,0
DA:4,1
BRDA:2,0,0,1
BRDA:2,0,1,0
LF:4
end_of_record
"#;
        
        let coverage = LcovParser::parse_content(lcov_content).unwrap();
        assert_eq!(coverage.files.len(), 1);
        assert!(coverage.files.contains_key("src/main.rs"));
        
        let file_cov = &coverage.files["src/main.rs"];
        assert_eq!(file_cov.covered_lines.len(), 3);
        assert_eq!(file_cov.uncovered_lines.len(), 1);
        assert_eq!(file_cov.branches.len(), 2);
        
        assert_eq!(coverage.line_coverage, 75.0);
        assert_eq!(coverage.branch_coverage, 50.0);
    }
}
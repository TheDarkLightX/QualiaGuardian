//! Code Health Score (CHS) sensor for comprehensive code quality analysis

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use tokio::fs;
use syn::{visit::Visit, Item, ItemFn};
use crate::{Sensor, SensorContext, SensorError, Result};

/// CHS sensor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CHSOutput {
    /// Raw sub-metrics
    pub sub_metrics: HashMap<String, f64>,
    /// Overall code health score
    pub code_health_score: f64,
    /// Detailed file metrics
    pub file_metrics: Vec<FileMetrics>,
    /// Project-level aggregated metrics
    pub project_metrics: ProjectMetrics,
}

/// Metrics for individual files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetrics {
    /// File path
    pub path: String,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: f64,
    /// Lines of code
    pub lines_of_code: usize,
    /// Number of functions
    pub function_count: usize,
    /// Average function length
    pub avg_function_length: f64,
    /// Documentation coverage
    pub doc_coverage: f64,
}

/// Project-level aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetrics {
    /// Total lines of code
    pub total_lines: usize,
    /// Total number of files
    pub total_files: usize,
    /// Average cyclomatic complexity
    pub avg_complexity: f64,
    /// Maintainability index (0-100)
    pub maintainability_index: f64,
    /// Shannon entropy (information density)
    pub shannon_entropy: f64,
    /// Documentation coverage percentage
    pub doc_coverage_percent: f64,
    /// Test to code ratio
    pub test_ratio: f64,
}

/// Language-specific analyzer trait
trait LanguageAnalyzer: Send + Sync {
    fn analyze_file(&self, content: &str, path: &Path) -> Result<FileMetrics>;
    fn supported_extensions(&self) -> &[&str];
}

/// Rust language analyzer
struct RustAnalyzer;

impl LanguageAnalyzer for RustAnalyzer {
    fn analyze_file(&self, content: &str, path: &Path) -> Result<FileMetrics> {
        let syntax = syn::parse_file(content)
            .map_err(|e| SensorError::Generic(format!("Failed to parse Rust file: {}", e)))?;
        
        let mut visitor = RustComplexityVisitor::new();
        visitor.visit_file(&syntax);
        
        let lines_of_code = content.lines().filter(|line| !line.trim().is_empty()).count();
        let doc_coverage = visitor.calculate_doc_coverage();
        let avg_function_length = if visitor.function_count > 0 {
            visitor.total_function_lines as f64 / visitor.function_count as f64
        } else {
            0.0
        };
        
        Ok(FileMetrics {
            path: path.display().to_string(),
            cyclomatic_complexity: visitor.complexity as f64,
            lines_of_code,
            function_count: visitor.function_count,
            avg_function_length,
            doc_coverage,
        })
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }
}

/// AST visitor for calculating Rust code complexity
struct RustComplexityVisitor {
    complexity: usize,
    function_count: usize,
    documented_functions: usize,
    total_function_lines: usize,
}

impl RustComplexityVisitor {
    fn new() -> Self {
        Self {
            complexity: 1, // Start with 1 for the module
            function_count: 0,
            documented_functions: 0,
            total_function_lines: 0,
        }
    }
    
    fn calculate_doc_coverage(&self) -> f64 {
        if self.function_count == 0 {
            1.0
        } else {
            self.documented_functions as f64 / self.function_count as f64
        }
    }
}

impl<'ast> Visit<'ast> for RustComplexityVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.function_count += 1;
        
        // Check if function has documentation
        if !node.attrs.is_empty() && node.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
            self.documented_functions += 1;
        }
        
        // Estimate function lines (simplified)
        let function_str = node.block.stmts.len();
        self.total_function_lines += function_str.max(1);
        
        // Count complexity from control flow
        syn::visit::visit_item_fn(self, node);
    }
    
    fn visit_expr(&mut self, expr: &'ast syn::Expr) {
        match expr {
            syn::Expr::If(_) | syn::Expr::Match(_) | syn::Expr::While(_) | syn::Expr::ForLoop(_) => {
                self.complexity += 1;
            }
            _ => {}
        }
        syn::visit::visit_expr(self, expr);
    }
}

/// Python language analyzer
struct PythonAnalyzer;

impl LanguageAnalyzer for PythonAnalyzer {
    fn analyze_file(&self, content: &str, path: &Path) -> Result<FileMetrics> {
        let lines_of_code = content.lines().filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        }).count();
        
        // Simple heuristic-based analysis for Python
        let mut complexity = 1;
        let mut function_count = 0;
        let mut in_function = false;
        let mut function_lines = 0;
        let mut documented_functions = 0;
        let mut pending_docstring = false;
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Count functions
            if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                function_count += 1;
                in_function = true;
                pending_docstring = true;
            }
            
            // Check for docstrings
            if pending_docstring && (trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''")) {
                documented_functions += 1;
                pending_docstring = false;
            } else if pending_docstring && !trimmed.is_empty() && !trimmed.ends_with(':') {
                pending_docstring = false;
            }
            
            // Count complexity
            if trimmed.starts_with("if ") || trimmed.starts_with("elif ") || 
               trimmed.starts_with("for ") || trimmed.starts_with("while ") ||
               trimmed.contains(" if ") { // List comprehensions
                complexity += 1;
            }
            
            if in_function && !trimmed.is_empty() {
                function_lines += 1;
            }
            
            // Simple function end detection
            if in_function && trimmed.is_empty() {
                in_function = false;
            }
        }
        
        let avg_function_length = if function_count > 0 {
            function_lines as f64 / function_count as f64
        } else {
            0.0
        };
        
        let doc_coverage = if function_count > 0 {
            documented_functions as f64 / function_count as f64
        } else {
            1.0
        };
        
        Ok(FileMetrics {
            path: path.display().to_string(),
            cyclomatic_complexity: complexity as f64,
            lines_of_code,
            function_count,
            avg_function_length,
            doc_coverage,
        })
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["py"]
    }
}

/// Code Health Score sensor
pub struct CHSSensor {
    /// Language analyzers
    analyzers: Vec<Box<dyn LanguageAnalyzer>>,
}

impl std::fmt::Debug for CHSSensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CHSSensor")
            .field("analyzers", &format!("{} analyzers", self.analyzers.len()))
            .finish()
    }
}

impl CHSSensor {
    pub fn new() -> Self {
        Self {
            analyzers: vec![
                Box::new(RustAnalyzer),
                Box::new(PythonAnalyzer),
            ],
        }
    }
    
    /// Analyze a single file
    async fn analyze_file(&self, path: &Path) -> Result<Option<FileMetrics>> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        // Find appropriate analyzer
        let analyzer = self.analyzers.iter()
            .find(|a| a.supported_extensions().contains(&extension));
        
        if let Some(analyzer) = analyzer {
            let content = fs::read_to_string(path).await
                .map_err(|e| SensorError::Io(e))?;
            
            let metrics = analyzer.analyze_file(&content, path)?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate Shannon entropy for code diversity
    fn calculate_shannon_entropy(content: &str) -> f64 {
        let mut char_counts = HashMap::new();
        let total_chars = content.len() as f64;
        
        for ch in content.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
        
        let mut entropy = 0.0;
        for count in char_counts.values() {
            let probability = *count as f64 / total_chars;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    /// Calculate maintainability index
    fn calculate_maintainability_index(metrics: &ProjectMetrics) -> f64 {
        // Simplified MI calculation
        // MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
        // Using simplified version: MI = 171 - 0.23 * CC - 16.2 * ln(LOC)
        
        let loc_factor = if metrics.total_lines > 0 {
            16.2 * (metrics.total_lines as f64).ln()
        } else {
            0.0
        };
        
        let cc_factor = 0.23 * metrics.avg_complexity;
        
        let mi = 171.0 - loc_factor - cc_factor;
        mi.clamp(0.0, 100.0)
    }
    
    /// Calculate overall code health score
    fn calculate_code_health_score(metrics: &ProjectMetrics) -> f64 {
        // Weighted combination of factors
        let complexity_score = (-metrics.avg_complexity / 10.0).exp(); // Lower is better
        let maintainability_score = metrics.maintainability_index / 100.0;
        let doc_score = metrics.doc_coverage_percent / 100.0;
        let test_score = metrics.test_ratio.min(1.0); // Cap at 1.0
        let entropy_score = (metrics.shannon_entropy / 7.0).min(1.0); // Normalize to ~7 bits
        
        // Weighted average
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1];
        let scores = [complexity_score, maintainability_score, doc_score, test_score, entropy_score];
        
        weights.iter().zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum()
    }
}

impl Default for CHSSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for CHSSensor {
    type Output = CHSOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let mut file_metrics = Vec::new();
        let mut total_lines = 0;
        let mut total_complexity = 0.0;
        let mut total_functions = 0;
        let mut documented_functions = 0;
        let mut test_files = 0;
        let mut source_files = 0;
        let mut all_content = String::new();
        
        // Walk through project files
        for entry in WalkDir::new(&context.project_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            
            // Skip common non-source directories
            if path.components().any(|c| {
                matches!(c.as_os_str().to_str(), Some("target") | Some("node_modules") | 
                        Some(".git") | Some("__pycache__") | Some(".pytest_cache"))
            }) {
                continue;
            }
            
            if let Some(metrics) = self.analyze_file(path).await? {
                // Track test vs source files
                if path.to_str().map(|s| s.contains("test")).unwrap_or(false) {
                    test_files += 1;
                } else {
                    source_files += 1;
                }
                
                // Aggregate metrics
                total_lines += metrics.lines_of_code;
                total_complexity += metrics.cyclomatic_complexity;
                total_functions += metrics.function_count;
                documented_functions += (metrics.doc_coverage * metrics.function_count as f64) as usize;
                
                // Read content for entropy calculation
                if let Ok(content) = fs::read_to_string(path).await {
                    all_content.push_str(&content);
                }
                
                file_metrics.push(metrics);
            }
        }
        
        let total_files = file_metrics.len();
        let avg_complexity = if total_files > 0 {
            total_complexity / total_files as f64
        } else {
            1.0
        };
        
        let doc_coverage_percent = if total_functions > 0 {
            (documented_functions as f64 / total_functions as f64) * 100.0
        } else {
            100.0
        };
        
        let test_ratio = if source_files > 0 {
            test_files as f64 / source_files as f64
        } else {
            0.0
        };
        
        let shannon_entropy = Self::calculate_shannon_entropy(&all_content);
        
        let project_metrics = ProjectMetrics {
            total_lines,
            total_files,
            avg_complexity,
            maintainability_index: 0.0, // Will calculate below
            shannon_entropy,
            doc_coverage_percent,
            test_ratio,
        };
        
        // Calculate maintainability index
        let mut project_metrics = project_metrics;
        project_metrics.maintainability_index = Self::calculate_maintainability_index(&project_metrics);
        
        // Calculate overall health score
        let code_health_score = Self::calculate_code_health_score(&project_metrics);
        
        // Prepare sub-metrics for compatibility
        let mut sub_metrics = HashMap::new();
        sub_metrics.insert("cyclomatic_complexity".to_string(), avg_complexity);
        sub_metrics.insert("maintainability_index".to_string(), project_metrics.maintainability_index);
        sub_metrics.insert("shannon_entropy".to_string(), shannon_entropy);
        sub_metrics.insert("doc_coverage_percent".to_string(), doc_coverage_percent);
        sub_metrics.insert("test_ratio".to_string(), test_ratio * 100.0);
        sub_metrics.insert("total_lines".to_string(), total_lines as f64);
        
        Ok(CHSOutput {
            sub_metrics,
            code_health_score,
            file_metrics,
            project_metrics,
        })
    }
    
    fn name(&self) -> &'static str {
        "chs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shannon_entropy_calculation() {
        let content = "hello world";
        let entropy = CHSSensor::calculate_shannon_entropy(content);
        assert!(entropy > 0.0);
        assert!(entropy < 4.0); // Should be around 3.18 bits
        
        // Test with more diverse content
        let diverse = "abcdefghijklmnopqrstuvwxyz0123456789";
        let diverse_entropy = CHSSensor::calculate_shannon_entropy(diverse);
        assert!(diverse_entropy > entropy); // More diverse = higher entropy
    }
    
    #[test]
    fn test_maintainability_index() {
        let metrics = ProjectMetrics {
            total_lines: 1000,
            total_files: 10,
            avg_complexity: 5.0,
            maintainability_index: 0.0,
            shannon_entropy: 5.0,
            doc_coverage_percent: 80.0,
            test_ratio: 0.5,
        };
        
        let mi = CHSSensor::calculate_maintainability_index(&metrics);
        assert!(mi > 0.0);
        assert!(mi <= 100.0);
    }
    
    #[test]
    fn test_code_health_score() {
        let metrics = ProjectMetrics {
            total_lines: 500,
            total_files: 5,
            avg_complexity: 3.0,
            maintainability_index: 85.0,
            shannon_entropy: 5.5,
            doc_coverage_percent: 90.0,
            test_ratio: 0.8,
        };
        
        let score = CHSSensor::calculate_code_health_score(&metrics);
        assert!(score > 0.0);
        assert!(score <= 1.0);
        assert!(score > 0.7); // Should be high for these good metrics
    }
}
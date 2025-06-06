//! Assertion IQ sensor for measuring test assertion quality

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use syn::{visit::Visit, Expr, ExprCall, ExprMethodCall};
use walkdir::WalkDir;
use std::fs;
use std::path::Path;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Assertion IQ output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionIQOutput {
    /// Mean IQ score across all tests (1-5)
    pub mean_iq: f64,
    /// Total number of test files analyzed
    pub total_test_files: usize,
    /// Total number of test functions
    pub total_test_functions: usize,
    /// Total number of assertions
    pub total_assertions: usize,
    /// Distribution of IQ scores
    pub iq_distribution: IQDistribution,
    /// Per-file results
    pub file_results: Vec<FileAssertionResult>,
}

/// IQ score distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQDistribution {
    /// Count of level 1 assertions (basic existence)
    pub level_1: usize,
    /// Count of level 2 assertions (type/structure)
    pub level_2: usize,
    /// Count of level 3 assertions (value/state)
    pub level_3: usize,
    /// Count of level 4 assertions (behavior/relationships)
    pub level_4: usize,
    /// Count of level 5 assertions (complex invariants)
    pub level_5: usize,
}

/// Assertion results for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAssertionResult {
    pub file_path: String,
    pub test_count: usize,
    pub assertion_count: usize,
    pub mean_iq: f64,
}

/// Assertion IQ sensor
#[derive(Debug)]
pub struct AssertionIQSensor {
    /// Patterns for different assertion levels
    patterns: AssertionPatterns,
}

#[derive(Debug)]
struct AssertionPatterns {
    level_1: Vec<String>, // Basic existence
    level_2: Vec<String>, // Type/structure
    level_3: Vec<String>, // Value/state
    level_4: Vec<String>, // Behavior/relationships
    level_5: Vec<String>, // Complex invariants
}

impl Default for AssertionPatterns {
    fn default() -> Self {
        Self {
            level_1: vec![
                "is_none".to_string(),
                "is_some".to_string(),
                "is_ok".to_string(),
                "is_err".to_string(),
                "is_empty".to_string(),
            ],
            level_2: vec![
                "assert_eq".to_string(),
                "assert_ne".to_string(),
                "instanceof".to_string(),
                "has_attr".to_string(),
            ],
            level_3: vec![
                "assert".to_string(),
                "assert_approx_eq".to_string(),
                "assert_in_delta".to_string(),
                "assert_matches".to_string(),
            ],
            level_4: vec![
                "assert_called_with".to_string(),
                "assert_raises".to_string(),
                "assert_panics".to_string(),
                "assert_changes".to_string(),
            ],
            level_5: vec![
                "prop_assert".to_string(),
                "quickcheck".to_string(),
                "assert_invariant".to_string(),
                "assert_transition".to_string(),
            ],
        }
    }
}

/// AST visitor for analyzing assertions
struct AssertionVisitor {
    assertions: Vec<AssertionInfo>,
    current_function: Option<String>,
}

#[derive(Debug)]
struct AssertionInfo {
    function_name: String,
    assertion_type: String,
    iq_level: u8,
}

impl AssertionVisitor {
    fn new() -> Self {
        Self {
            assertions: Vec::new(),
            current_function: None,
        }
    }
    
    fn analyze_call(&mut self, call: &str) {
        if let Some(func_name) = &self.current_function {
            let iq_level = self.determine_iq_level(call);
            self.assertions.push(AssertionInfo {
                function_name: func_name.clone(),
                assertion_type: call.to_string(),
                iq_level,
            });
        }
    }
    
    fn determine_iq_level(&self, call: &str) -> u8 {
        // Simplified IQ level determination
        if call.contains("is_none") || call.contains("is_some") || call.contains("is_empty") {
            1
        } else if call.contains("assert_eq") || call.contains("assert_ne") {
            2
        } else if call.contains("assert") && !call.contains("assert_") {
            3
        } else if call.contains("assert_called") || call.contains("assert_raises") {
            4
        } else if call.contains("prop_") || call.contains("invariant") {
            5
        } else {
            2 // Default to level 2
        }
    }
}

impl<'ast> Visit<'ast> for AssertionVisitor {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let fn_name = node.sig.ident.to_string();
        
        // Check if this is a test function
        let is_test = node.attrs.iter().any(|attr| {
            attr.path().is_ident("test") || 
            attr.path().is_ident("tokio::test") ||
            attr.path().is_ident("async_std::test")
        });
        
        if is_test || fn_name.starts_with("test_") {
            self.current_function = Some(fn_name);
            syn::visit::visit_item_fn(self, node);
            self.current_function = None;
        }
    }
    
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = &*node.func {
            if let Some(ident) = path.path.get_ident() {
                self.analyze_call(&ident.to_string());
            }
        }
        syn::visit::visit_expr_call(self, node);
    }
    
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.analyze_call(&node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }
}

impl AssertionIQSensor {
    /// Create a new assertion IQ sensor
    pub fn new() -> Self {
        Self {
            patterns: AssertionPatterns::default(),
        }
    }
    
    /// Analyze a single test file
    async fn analyze_file(&self, path: &Path) -> Result<FileAssertionResult> {
        let content = fs::read_to_string(path)
            .map_err(|e| SensorError::Io(e))?;
        
        // Parse Rust code
        let syntax_tree = syn::parse_file(&content)
            .map_err(|e| SensorError::Parse(format!("Failed to parse {}: {}", path.display(), e)))?;
        
        // Visit AST and collect assertions
        let mut visitor = AssertionVisitor::new();
        visitor.visit_file(&syntax_tree);
        
        // Calculate metrics
        let test_functions: std::collections::HashSet<_> = visitor.assertions
            .iter()
            .map(|a| &a.function_name)
            .collect();
        
        let test_count = test_functions.len();
        let assertion_count = visitor.assertions.len();
        
        let mean_iq = if assertion_count > 0 {
            visitor.assertions.iter()
                .map(|a| a.iq_level as f64)
                .sum::<f64>() / assertion_count as f64
        } else {
            1.0
        };
        
        Ok(FileAssertionResult {
            file_path: path.display().to_string(),
            test_count,
            assertion_count,
            mean_iq,
        })
    }
    
    /// Find all test files in the project
    fn find_test_files(&self, project_path: &str) -> Vec<std::path::PathBuf> {
        let mut test_files = Vec::new();
        
        // Look for test files
        for entry in WalkDir::new(project_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                
                // Check if it's a test file
                if (file_name.starts_with("test_") || file_name.ends_with("_test.rs") || 
                    path.components().any(|c| c.as_os_str() == "tests")) &&
                    file_name.ends_with(".rs") {
                    test_files.push(path.to_path_buf());
                }
            }
        }
        
        test_files
    }
}

impl Default for AssertionIQSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for AssertionIQSensor {
    type Output = AssertionIQOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let test_files = self.find_test_files(&context.project_path);
        let total_test_files = test_files.len();
        
        // Analyze all test files
        let mut file_results = Vec::new();
        let mut iq_distribution = IQDistribution {
            level_1: 0,
            level_2: 0,
            level_3: 0,
            level_4: 0,
            level_5: 0,
        };
        
        for file_path in test_files {
            match self.analyze_file(&file_path).await {
                Ok(result) => {
                    file_results.push(result);
                }
                Err(e) => {
                    eprintln!("Error analyzing {}: {}", file_path.display(), e);
                }
            }
        }
        
        // Calculate aggregate metrics
        let total_test_functions: usize = file_results.iter()
            .map(|r| r.test_count)
            .sum();
        
        let total_assertions: usize = file_results.iter()
            .map(|r| r.assertion_count)
            .sum();
        
        let mean_iq = if !file_results.is_empty() {
            file_results.iter()
                .map(|r| r.mean_iq * r.assertion_count as f64)
                .sum::<f64>() / total_assertions.max(1) as f64
        } else {
            1.0
        };
        
        Ok(AssertionIQOutput {
            mean_iq,
            total_test_files,
            total_test_functions,
            total_assertions,
            iq_distribution,
            file_results,
        })
    }
    
    fn name(&self) -> &'static str {
        "assertion_iq"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iq_level_determination() {
        let sensor = AssertionIQSensor::new();
        let visitor = AssertionVisitor::new();
        
        assert_eq!(visitor.determine_iq_level("is_none"), 1);
        assert_eq!(visitor.determine_iq_level("assert_eq"), 2);
        assert_eq!(visitor.determine_iq_level("assert"), 3);
        assert_eq!(visitor.determine_iq_level("assert_called_with"), 4);
        assert_eq!(visitor.determine_iq_level("prop_assert"), 5);
    }
}
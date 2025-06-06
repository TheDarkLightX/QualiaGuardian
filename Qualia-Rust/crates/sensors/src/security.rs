//! Security vulnerability sensor for detecting potential security issues

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use tokio::fs;
use regex::Regex;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Security sensor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityOutput {
    /// Weighted vulnerability density (vulnerabilities per KLOC)
    pub weighted_vulnerability_density: f64,
    /// Number of critical vulnerabilities
    pub critical_vulns: usize,
    /// Number of high vulnerabilities
    pub high_vulns: usize,
    /// Number of medium vulnerabilities
    pub medium_vulns: usize,
    /// Number of low vulnerabilities
    pub low_vulns: usize,
    /// Total lines of code analyzed
    pub total_lines: usize,
    /// Detailed vulnerability findings
    pub vulnerabilities: Vec<Vulnerability>,
}

/// Individual vulnerability finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability type/category
    pub category: String,
    /// Severity level
    pub severity: Severity,
    /// File path
    pub file: String,
    /// Line number
    pub line: usize,
    /// Description
    pub description: String,
    /// CWE ID if applicable
    pub cwe_id: Option<String>,
}

/// Vulnerability severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

impl Severity {
    /// Get weight for weighted density calculation
    fn weight(&self) -> f64 {
        match self {
            Severity::Critical => 10.0,
            Severity::High => 5.0,
            Severity::Medium => 2.0,
            Severity::Low => 1.0,
        }
    }
}

/// Security pattern for detection
struct SecurityPattern {
    name: String,
    regex: Regex,
    severity: Severity,
    description: String,
    cwe_id: Option<String>,
}

/// Language-specific security analyzer
trait SecurityAnalyzer: Send + Sync {
    fn analyze_file(&self, content: &str, path: &Path) -> Vec<Vulnerability>;
    fn supported_extensions(&self) -> &[&str];
}

/// Generic pattern-based analyzer
struct PatternAnalyzer {
    patterns: Vec<SecurityPattern>,
    extensions: Vec<&'static str>,
}

impl PatternAnalyzer {
    fn new(patterns: Vec<SecurityPattern>, extensions: Vec<&'static str>) -> Self {
        Self { patterns, extensions }
    }
    
    fn create_rust_analyzer() -> Self {
        let patterns = vec![
            SecurityPattern {
                name: "Unsafe Block".to_string(),
                regex: Regex::new(r"unsafe\s*\{").unwrap(),
                severity: Severity::Medium,
                description: "Use of unsafe block - requires careful review".to_string(),
                cwe_id: None,
            },
            SecurityPattern {
                name: "Unwrap on Result/Option".to_string(),
                regex: Regex::new(r"\.unwrap\(\)").unwrap(),
                severity: Severity::Low,
                description: "Use of unwrap() can cause panics".to_string(),
                cwe_id: Some("CWE-248".to_string()), // Uncaught Exception
            },
            SecurityPattern {
                name: "SQL Query Construction".to_string(),
                regex: Regex::new(r#"(?i)format!\s*\(\s*".*(?:SELECT|INSERT|UPDATE|DELETE).*\{.*\}"#).unwrap(),
                severity: Severity::High,
                description: "Potential SQL injection via string formatting".to_string(),
                cwe_id: Some("CWE-89".to_string()),
            },
            SecurityPattern {
                name: "Command Injection Risk".to_string(),
                regex: Regex::new(r"Command::new\s*\(\s*[^\x22\x27]+").unwrap(),
                severity: Severity::High,
                description: "Command constructed from variable - potential injection".to_string(),
                cwe_id: Some("CWE-78".to_string()),
            },
        ];
        
        Self::new(patterns, vec!["rs"])
    }
    
    fn create_python_analyzer() -> Self {
        let patterns = vec![
            SecurityPattern {
                name: "Eval Usage".to_string(),
                regex: Regex::new(r"\beval\s*\(").unwrap(),
                severity: Severity::Critical,
                description: "Use of eval() is extremely dangerous".to_string(),
                cwe_id: Some("CWE-95".to_string()),
            },
            SecurityPattern {
                name: "Exec Usage".to_string(),
                regex: Regex::new(r"\bexec\s*\(").unwrap(),
                severity: Severity::Critical,
                description: "Use of exec() can execute arbitrary code".to_string(),
                cwe_id: Some("CWE-95".to_string()),
            },
            SecurityPattern {
                name: "Pickle Load".to_string(),
                regex: Regex::new(r"pickle\.load\s*\(").unwrap(),
                severity: Severity::High,
                description: "Pickle can execute arbitrary code during deserialization".to_string(),
                cwe_id: Some("CWE-502".to_string()),
            },
            SecurityPattern {
                name: "SQL Query Formatting".to_string(),
                regex: Regex::new(r#"(?:execute|query)\s*\(\s*[\x22\x27].*%[sdfr].*[\x22\x27].*%"#).unwrap(),
                severity: Severity::High,
                description: "SQL query using string formatting - SQL injection risk".to_string(),
                cwe_id: Some("CWE-89".to_string()),
            },
            SecurityPattern {
                name: "Hardcoded Password".to_string(),
                regex: Regex::new(r#"(?i)(?:password|passwd|pwd)\s*=\s*[\x22\x27][^\x22\x27]{3,}[\x22\x27]"#).unwrap(),
                severity: Severity::High,
                description: "Hardcoded password detected".to_string(),
                cwe_id: Some("CWE-798".to_string()),
            },
            SecurityPattern {
                name: "Shell Injection Risk".to_string(),
                regex: Regex::new(r"os\.system\s*\(|subprocess\.call\s*\([^,\]]*shell\s*=\s*True").unwrap(),
                severity: Severity::High,
                description: "Shell command execution with potential injection".to_string(),
                cwe_id: Some("CWE-78".to_string()),
            },
            SecurityPattern {
                name: "Weak Random".to_string(),
                regex: Regex::new(r"\brandom\.(?:random|randint|choice)\s*\(").unwrap(),
                severity: Severity::Medium,
                description: "Use of weak random number generator for security purposes".to_string(),
                cwe_id: Some("CWE-330".to_string()),
            },
            SecurityPattern {
                name: "Assert Statement".to_string(),
                regex: Regex::new(r"^\s*assert\s+").unwrap(),
                severity: Severity::Low,
                description: "Assert statements are removed in optimized code".to_string(),
                cwe_id: Some("CWE-617".to_string()),
            },
        ];
        
        Self::new(patterns, vec!["py"])
    }
    
    fn create_javascript_analyzer() -> Self {
        let patterns = vec![
            SecurityPattern {
                name: "Eval Usage".to_string(),
                regex: Regex::new(r"\beval\s*\(").unwrap(),
                severity: Severity::Critical,
                description: "Use of eval() is a security risk".to_string(),
                cwe_id: Some("CWE-95".to_string()),
            },
            SecurityPattern {
                name: "InnerHTML Usage".to_string(),
                regex: Regex::new(r"\.innerHTML\s*=").unwrap(),
                severity: Severity::High,
                description: "innerHTML can lead to XSS vulnerabilities".to_string(),
                cwe_id: Some("CWE-79".to_string()),
            },
            SecurityPattern {
                name: "Document Write".to_string(),
                regex: Regex::new(r"document\.write\s*\(").unwrap(),
                severity: Severity::Medium,
                description: "document.write can be used for XSS attacks".to_string(),
                cwe_id: Some("CWE-79".to_string()),
            },
            SecurityPattern {
                name: "LocalStorage Sensitive Data".to_string(),
                regex: Regex::new(r"localStorage\.setItem\s*\(\s*[\x22\x27](?:password|token|key)[\x22\x27]").unwrap(),
                severity: Severity::High,
                description: "Storing sensitive data in localStorage".to_string(),
                cwe_id: Some("CWE-922".to_string()),
            },
        ];
        
        Self::new(patterns, vec!["js", "jsx", "ts", "tsx"])
    }
}

impl SecurityAnalyzer for PatternAnalyzer {
    fn analyze_file(&self, content: &str, path: &Path) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();
        
        for (line_num, line) in content.lines().enumerate() {
            for pattern in &self.patterns {
                if pattern.regex.is_match(line) {
                    vulnerabilities.push(Vulnerability {
                        category: pattern.name.clone(),
                        severity: pattern.severity,
                        file: path.display().to_string(),
                        line: line_num + 1,
                        description: pattern.description.clone(),
                        cwe_id: pattern.cwe_id.clone(),
                    });
                }
            }
        }
        
        vulnerabilities
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &self.extensions
    }
}

/// Security sensor for vulnerability detection
pub struct SecuritySensor {
    /// Security analyzers for different languages
    analyzers: Vec<Box<dyn SecurityAnalyzer>>,
    /// Whether to run external SAST tools
    use_external_tools: bool,
}

impl std::fmt::Debug for SecuritySensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecuritySensor")
            .field("analyzers", &format!("{} analyzers", self.analyzers.len()))
            .field("use_external_tools", &self.use_external_tools)
            .finish()
    }
}

impl SecuritySensor {
    pub fn new() -> Self {
        Self {
            analyzers: vec![
                Box::new(PatternAnalyzer::create_rust_analyzer()),
                Box::new(PatternAnalyzer::create_python_analyzer()),
                Box::new(PatternAnalyzer::create_javascript_analyzer()),
            ],
            use_external_tools: false,
        }
    }
    
    /// Enable external SAST tool integration
    pub fn with_external_tools(mut self) -> Self {
        self.use_external_tools = true;
        self
    }
    
    /// Analyze a single file
    async fn analyze_file(&self, path: &Path) -> Result<Vec<Vulnerability>> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        // Find appropriate analyzer
        let analyzer = self.analyzers.iter()
            .find(|a| a.supported_extensions().contains(&extension));
        
        if let Some(analyzer) = analyzer {
            let content = fs::read_to_string(path).await
                .map_err(|e| SensorError::Io(e))?;
            
            Ok(analyzer.analyze_file(&content, path))
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Run external SAST tools if configured
    async fn run_external_tools(&self, project_path: &str) -> Result<Vec<Vulnerability>> {
        // Placeholder for external tool integration
        // In real implementation, would run tools like:
        // - cargo-audit for Rust
        // - bandit for Python
        // - npm audit for JavaScript
        // - semgrep for multiple languages
        Ok(Vec::new())
    }
    
    /// Calculate weighted vulnerability density
    fn calculate_weighted_density(vulnerabilities: &[Vulnerability], total_lines: usize) -> f64 {
        if total_lines == 0 {
            return 0.0;
        }
        
        let weighted_sum: f64 = vulnerabilities.iter()
            .map(|v| v.severity.weight())
            .sum();
        
        // Vulnerabilities per thousand lines of code
        (weighted_sum / total_lines as f64) * 1000.0
    }
}

impl Default for SecuritySensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for SecuritySensor {
    type Output = SecurityOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let mut all_vulnerabilities = Vec::new();
        let mut total_lines = 0;
        
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
                        Some(".git") | Some("__pycache__") | Some("vendor"))
            }) {
                continue;
            }
            
            // Analyze file
            let vulnerabilities = self.analyze_file(path).await?;
            all_vulnerabilities.extend(vulnerabilities);
            
            // Count lines
            if let Ok(content) = fs::read_to_string(path).await {
                total_lines += content.lines().count();
            }
        }
        
        // Run external tools if enabled
        if self.use_external_tools {
            let external_vulns = self.run_external_tools(&context.project_path).await?;
            all_vulnerabilities.extend(external_vulns);
        }
        
        // Count vulnerabilities by severity
        let mut critical_vulns = 0;
        let mut high_vulns = 0;
        let mut medium_vulns = 0;
        let mut low_vulns = 0;
        
        for vuln in &all_vulnerabilities {
            match vuln.severity {
                Severity::Critical => critical_vulns += 1,
                Severity::High => high_vulns += 1,
                Severity::Medium => medium_vulns += 1,
                Severity::Low => low_vulns += 1,
            }
        }
        
        // Calculate weighted density
        let weighted_vulnerability_density = Self::calculate_weighted_density(&all_vulnerabilities, total_lines);
        
        Ok(SecurityOutput {
            weighted_vulnerability_density,
            critical_vulns,
            high_vulns,
            medium_vulns,
            low_vulns,
            total_lines,
            vulnerabilities: all_vulnerabilities,
        })
    }
    
    fn name(&self) -> &'static str {
        "security"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_severity_weights() {
        assert_eq!(Severity::Critical.weight(), 10.0);
        assert_eq!(Severity::High.weight(), 5.0);
        assert_eq!(Severity::Medium.weight(), 2.0);
        assert_eq!(Severity::Low.weight(), 1.0);
    }
    
    #[test]
    fn test_pattern_detection() {
        let analyzer = PatternAnalyzer::create_python_analyzer();
        let content = r#"
password = "hardcoded123"
eval(user_input)
subprocess.call(cmd, shell=True)
"#;
        
        let vulns = analyzer.analyze_file(content, Path::new("test.py"));
        assert!(vulns.len() >= 3);
        
        // Check for specific vulnerabilities
        assert!(vulns.iter().any(|v| v.category.contains("Password")));
        assert!(vulns.iter().any(|v| v.category.contains("Eval")));
        assert!(vulns.iter().any(|v| v.category.contains("Shell")));
    }
    
    #[test]
    fn test_weighted_density_calculation() {
        let vulns = vec![
            Vulnerability {
                category: "Test".to_string(),
                severity: Severity::Critical,
                file: "test.py".to_string(),
                line: 1,
                description: "Test".to_string(),
                cwe_id: None,
            },
            Vulnerability {
                category: "Test".to_string(),
                severity: Severity::Low,
                file: "test.py".to_string(),
                line: 2,
                description: "Test".to_string(),
                cwe_id: None,
            },
        ];
        
        // 1 critical (10) + 1 low (1) = 11 weight
        // 11 / 1000 * 1000 = 11.0
        let density = SecuritySensor::calculate_weighted_density(&vulns, 1000);
        assert_eq!(density, 11.0);
    }
}
//! Architecture quality sensor for analyzing code structure and dependencies

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use tokio::fs;
use regex::Regex;
use petgraph::graph::{UnGraph, NodeIndex};
use petgraph::algo::connected_components;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Architecture sensor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchOutput {
    /// Architectural violation score (0.0 to 1.0, lower is better)
    pub violation_score: f64,
    /// Algebraic connectivity (higher is better)
    pub algebraic_connectivity: f64,
    /// Module coupling score (lower is better)
    pub coupling_score: f64,
    /// Module cohesion score (higher is better)
    pub cohesion_score: f64,
    /// Detailed module metrics
    pub module_metrics: Vec<ModuleMetrics>,
    /// Dependency violations found
    pub violations: Vec<ArchViolation>,
}

/// Metrics for individual modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetrics {
    /// Module name
    pub name: String,
    /// Number of incoming dependencies
    pub afferent_coupling: usize,
    /// Number of outgoing dependencies
    pub efferent_coupling: usize,
    /// Instability metric (Ce / (Ca + Ce))
    pub instability: f64,
    /// Number of internal connections
    pub internal_cohesion: usize,
    /// Number of public interfaces
    pub public_interfaces: usize,
}

/// Architectural violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Source module
    pub source: String,
    /// Target module (if applicable)
    pub target: Option<String>,
    /// Description
    pub description: String,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
}

/// Types of architectural violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Circular dependency
    CircularDependency,
    /// Layer violation (e.g., UI depending on database)
    LayerViolation,
    /// Unstable dependency (stable module depending on unstable)
    UnstableDependency,
    /// High coupling
    HighCoupling,
    /// Low cohesion
    LowCohesion,
}

/// Module dependency information
#[derive(Debug, Clone)]
struct Module {
    name: String,
    path: PathBuf,
    imports: HashSet<String>,
    exports: HashSet<String>,
    internal_refs: usize,
}

/// Language-specific dependency analyzer
trait DependencyAnalyzer: Send + Sync {
    fn analyze_file(&self, content: &str, path: &Path) -> DependencyInfo;
    fn supported_extensions(&self) -> &[&str];
}

/// Dependency information from a file
#[derive(Debug, Default)]
struct DependencyInfo {
    imports: HashSet<String>,
    exports: HashSet<String>,
    internal_refs: usize,
}

/// Rust dependency analyzer
struct RustAnalyzer;

impl DependencyAnalyzer for RustAnalyzer {
    fn analyze_file(&self, content: &str, _path: &Path) -> DependencyInfo {
        let mut info = DependencyInfo::default();
        
        // Parse imports
        let use_regex = Regex::new(r"use\s+((?:crate|super|self)?(?:::\w+)+)").unwrap();
        for cap in use_regex.captures_iter(content) {
            if let Some(import) = cap.get(1) {
                info.imports.insert(import.as_str().to_string());
            }
        }
        
        // Parse exports (public items)
        let pub_regex = Regex::new(r"pub\s+(?:fn|struct|enum|trait|type|const|static)\s+(\w+)").unwrap();
        for cap in pub_regex.captures_iter(content) {
            if let Some(export) = cap.get(1) {
                info.exports.insert(export.as_str().to_string());
            }
        }
        
        // Count internal references (simplified)
        info.internal_refs = content.matches("self::").count() + content.matches("Self::").count();
        
        info
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }
}

/// Python dependency analyzer
struct PythonAnalyzer;

impl DependencyAnalyzer for PythonAnalyzer {
    fn analyze_file(&self, content: &str, _path: &Path) -> DependencyInfo {
        let mut info = DependencyInfo::default();
        
        // Parse imports
        let import_regex = Regex::new(r"(?:from\s+(\S+)\s+)?import\s+([^#\n]+)").unwrap();
        for cap in import_regex.captures_iter(content) {
            if let Some(module) = cap.get(1) {
                info.imports.insert(module.as_str().to_string());
            }
            if let Some(items) = cap.get(2) {
                for item in items.as_str().split(',') {
                    let item = item.trim().split(" as ").next().unwrap_or("").trim();
                    if !item.is_empty() && item != "*" {
                        info.exports.insert(item.to_string());
                    }
                }
            }
        }
        
        // Parse class and function definitions
        let def_regex = Regex::new(r"^(?:class|def)\s+(\w+)").unwrap();
        for cap in def_regex.captures_iter(content) {
            if let Some(name) = cap.get(1) {
                info.exports.insert(name.as_str().to_string());
            }
        }
        
        // Count internal references
        info.internal_refs = content.matches("self.").count();
        
        info
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["py"]
    }
}

/// Architecture sensor for structural analysis
pub struct ArchSensor {
    /// Dependency analyzers
    analyzers: Vec<Box<dyn DependencyAnalyzer>>,
    /// Architectural rules
    layer_rules: HashMap<String, Vec<String>>,
}

impl std::fmt::Debug for ArchSensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArchSensor")
            .field("analyzers", &format!("{} analyzers", self.analyzers.len()))
            .field("layer_rules", &self.layer_rules)
            .finish()
    }
}

impl ArchSensor {
    pub fn new() -> Self {
        let mut layer_rules = HashMap::new();
        
        // Define typical layered architecture rules
        // Higher layers can depend on lower layers, but not vice versa
        layer_rules.insert("ui".to_string(), vec!["service".to_string(), "domain".to_string(), "data".to_string()]);
        layer_rules.insert("service".to_string(), vec!["domain".to_string(), "data".to_string()]);
        layer_rules.insert("domain".to_string(), vec!["data".to_string()]);
        layer_rules.insert("data".to_string(), vec![]);
        
        Self {
            analyzers: vec![
                Box::new(RustAnalyzer),
                Box::new(PythonAnalyzer),
            ],
            layer_rules,
        }
    }
    
    /// Identify module from path
    fn identify_module(&self, path: &Path) -> String {
        let components: Vec<_> = path.components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        
        // Try to identify standard module patterns
        for (i, component) in components.iter().enumerate() {
            if matches!(*component, "src" | "lib" | "test" | "tests") {
                if i + 1 < components.len() {
                    return components[i + 1].to_string();
                }
            }
        }
        
        // Default to first meaningful directory
        components.into_iter()
            .find(|c| !c.starts_with('.') && c != &"target" && c != &"node_modules")
            .unwrap_or("root")
            .to_string()
    }
    
    /// Analyze dependencies in the project
    async fn analyze_dependencies(&self, project_path: &str) -> Result<HashMap<String, Module>> {
        let mut modules: HashMap<String, Module> = HashMap::new();
        
        for entry in WalkDir::new(project_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            
            // Skip non-source files
            let extension = path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("");
            
            let analyzer = self.analyzers.iter()
                .find(|a| a.supported_extensions().contains(&extension));
            
            if let Some(analyzer) = analyzer {
                if let Ok(content) = fs::read_to_string(path).await {
                    let module_name = self.identify_module(path);
                    let dep_info = analyzer.analyze_file(&content, path);
                    
                    let module = modules.entry(module_name.clone()).or_insert_with(|| Module {
                        name: module_name,
                        path: path.to_path_buf(),
                        imports: HashSet::new(),
                        exports: HashSet::new(),
                        internal_refs: 0,
                    });
                    
                    module.imports.extend(dep_info.imports);
                    module.exports.extend(dep_info.exports);
                    module.internal_refs += dep_info.internal_refs;
                }
            }
        }
        
        Ok(modules)
    }
    
    /// Build dependency graph
    fn build_dependency_graph(&self, modules: &HashMap<String, Module>) -> UnGraph<String, ()> {
        let mut graph = UnGraph::new_undirected();
        let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();
        
        // Add nodes
        for module_name in modules.keys() {
            let idx = graph.add_node(module_name.clone());
            node_indices.insert(module_name.clone(), idx);
        }
        
        // Add edges based on imports
        for (module_name, module) in modules {
            if let Some(&from_idx) = node_indices.get(module_name) {
                for import in &module.imports {
                    // Try to match import to a module
                    for (other_name, _) in modules {
                        if import.contains(other_name) {
                            if let Some(&to_idx) = node_indices.get(other_name) {
                                if from_idx != to_idx {
                                    graph.add_edge(from_idx, to_idx, ());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        graph
    }
    
    /// Calculate algebraic connectivity (simplified)
    fn calculate_algebraic_connectivity(graph: &UnGraph<String, ()>) -> f64 {
        if graph.node_count() < 2 {
            return 0.0;
        }
        
        let components = connected_components(graph);
        if components > 1 {
            return 0.0; // Disconnected graph
        }
        
        // Simplified: use inverse of average path length as proxy
        let n = graph.node_count() as f64;
        let e = graph.edge_count() as f64;
        
        if e > 0.0 {
            2.0 * e / (n * (n - 1.0))
        } else {
            0.0
        }
    }
    
    /// Calculate module metrics
    fn calculate_module_metrics(&self, modules: &HashMap<String, Module>) -> Vec<ModuleMetrics> {
        let mut metrics = Vec::new();
        
        for (name, module) in modules {
            let afferent_coupling = modules.values()
                .filter(|m| m.name != module.name && m.imports.iter().any(|i| i.contains(&module.name)))
                .count();
            
            let efferent_coupling = module.imports.len();
            let total_coupling = afferent_coupling + efferent_coupling;
            
            let instability = if total_coupling > 0 {
                efferent_coupling as f64 / total_coupling as f64
            } else {
                0.0
            };
            
            metrics.push(ModuleMetrics {
                name: name.clone(),
                afferent_coupling,
                efferent_coupling,
                instability,
                internal_cohesion: module.internal_refs,
                public_interfaces: module.exports.len(),
            });
        }
        
        metrics
    }
    
    /// Detect architectural violations
    fn detect_violations(&self, modules: &HashMap<String, Module>, module_metrics: &[ModuleMetrics]) -> Vec<ArchViolation> {
        let mut violations = Vec::new();
        
        // Check for high coupling
        for metric in module_metrics {
            if metric.efferent_coupling > 10 {
                violations.push(ArchViolation {
                    violation_type: ViolationType::HighCoupling,
                    source: metric.name.clone(),
                    target: None,
                    description: format!("Module has {} outgoing dependencies", metric.efferent_coupling),
                    severity: (metric.efferent_coupling as f64 / 20.0).min(1.0),
                });
            }
        }
        
        // Check for low cohesion
        for metric in module_metrics {
            if metric.public_interfaces > 20 && metric.internal_cohesion < 5 {
                violations.push(ArchViolation {
                    violation_type: ViolationType::LowCohesion,
                    source: metric.name.clone(),
                    target: None,
                    description: "Module has many public interfaces but low internal cohesion".to_string(),
                    severity: 0.5,
                });
            }
        }
        
        // Check for unstable dependencies
        for metric in module_metrics {
            if metric.instability < 0.3 { // Stable module
                if let Some(module) = modules.get(&metric.name) {
                    for import in &module.imports {
                        if let Some(dep_metric) = module_metrics.iter().find(|m| import.contains(&m.name)) {
                            if dep_metric.instability > 0.7 { // Depends on unstable
                                violations.push(ArchViolation {
                                    violation_type: ViolationType::UnstableDependency,
                                    source: metric.name.clone(),
                                    target: Some(dep_metric.name.clone()),
                                    description: "Stable module depends on unstable module".to_string(),
                                    severity: 0.7,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        violations
    }
    
    /// Calculate overall scores
    fn calculate_scores(&self, module_metrics: &[ModuleMetrics], violations: &[ArchViolation]) -> (f64, f64, f64) {
        // Violation score
        let violation_score = if violations.is_empty() {
            0.0
        } else {
            let total_severity: f64 = violations.iter().map(|v| v.severity).sum();
            (total_severity / violations.len() as f64).min(1.0)
        };
        
        // Coupling score (normalized)
        let avg_coupling = if module_metrics.is_empty() {
            0.0
        } else {
            let total_coupling: usize = module_metrics.iter()
                .map(|m| m.efferent_coupling)
                .sum();
            total_coupling as f64 / module_metrics.len() as f64 / 10.0
        }.min(1.0);
        
        // Cohesion score
        let avg_cohesion = if module_metrics.is_empty() {
            1.0
        } else {
            let total_cohesion: usize = module_metrics.iter()
                .map(|m| m.internal_cohesion)
                .sum();
            let total_interfaces: usize = module_metrics.iter()
                .map(|m| m.public_interfaces.max(1))
                .sum();
            (total_cohesion as f64 / total_interfaces as f64).min(1.0)
        };
        
        (violation_score, avg_coupling, avg_cohesion)
    }
}

impl Default for ArchSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for ArchSensor {
    type Output = ArchOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        // Analyze dependencies
        let modules = self.analyze_dependencies(&context.project_path).await?;
        
        // Build dependency graph
        let graph = self.build_dependency_graph(&modules);
        
        // Calculate algebraic connectivity
        let algebraic_connectivity = Self::calculate_algebraic_connectivity(&graph);
        
        // Calculate module metrics
        let module_metrics = self.calculate_module_metrics(&modules);
        
        // Detect violations
        let violations = self.detect_violations(&modules, &module_metrics);
        
        // Calculate scores
        let (violation_score, coupling_score, cohesion_score) = 
            self.calculate_scores(&module_metrics, &violations);
        
        Ok(ArchOutput {
            violation_score,
            algebraic_connectivity,
            coupling_score,
            cohesion_score,
            module_metrics,
            violations,
        })
    }
    
    fn name(&self) -> &'static str {
        "arch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_identification() {
        let sensor = ArchSensor::new();
        
        assert_eq!(sensor.identify_module(Path::new("src/main.rs")), "main.rs");
        assert_eq!(sensor.identify_module(Path::new("src/lib/mod.rs")), "lib");
        assert_eq!(sensor.identify_module(Path::new("tests/integration.rs")), "integration.rs");
    }
    
    #[test]
    fn test_dependency_analysis() {
        let analyzer = RustAnalyzer;
        let content = r#"
use std::collections::HashMap;
use crate::module::function;
use super::parent;

pub fn public_function() {}
pub struct PublicStruct;
        "#;
        
        let info = analyzer.analyze_file(content, Path::new("test.rs"));
        assert_eq!(info.imports.len(), 3);
        assert_eq!(info.exports.len(), 2);
    }
}
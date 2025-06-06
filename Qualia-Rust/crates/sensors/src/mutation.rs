//! Mutation testing sensor using mutmut integration

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Duration;
use tokio::time::timeout;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Output from mutation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationOutput {
    /// Total number of mutants generated
    pub total_mutants: u32,
    /// Number of mutants killed by tests
    pub killed_mutants: u32,
    /// Number of mutants that survived
    pub survived_mutants: u32,
    /// Number of mutants that timed out
    pub timeout_mutants: u32,
    /// Mutation score (killed / total)
    pub mutation_score: f64,
    /// Execution time in seconds
    pub execution_time_secs: f64,
    /// Per-file mutation results
    pub file_results: Vec<FileMutationResult>,
}

/// Mutation results for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMutationResult {
    pub file_path: String,
    pub total_mutants: u32,
    pub killed_mutants: u32,
    pub mutation_score: f64,
}

/// Mutation testing sensor
#[derive(Debug)]
pub struct MutationSensor {
    /// Path to mutmut executable
    mutmut_path: String,
}

impl MutationSensor {
    /// Create a new mutation sensor
    pub fn new() -> Self {
        Self {
            mutmut_path: "mutmut".to_string(),
        }
    }
    
    /// Create with custom mutmut path
    pub fn with_mutmut_path(path: impl Into<String>) -> Self {
        Self {
            mutmut_path: path.into(),
        }
    }
    
    /// Run mutmut via Python integration
    #[cfg(feature = "python")]
    async fn run_mutmut(&self, project_path: &str, _timeout_secs: u64) -> Result<MutationOutput> {
        let project = project_path.to_string();
        let _mutmut = self.mutmut_path.clone();
        
        // Run in blocking task since PyO3 operations are blocking
        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> PyResult<MutationOutput> {
                // Import mutmut module
                let mutmut_module = py.import("mutmut")?;
                
                // Run mutation testing
                let kwargs = PyDict::new(py);
                kwargs.set_item("paths_to_mutate", vec![&project])?;
                kwargs.set_item("tests_dir", vec!["tests"])?;
                
                let result = mutmut_module.call_method("run_mutation_tests", (), Some(kwargs))?;
                
                // Parse results
                let stats: &PyDict = result.extract()?;
                
                let total_mutants: u32 = stats.get_item("total_mutants")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                
                let killed_mutants: u32 = stats.get_item("killed_mutants")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                
                let survived_mutants: u32 = stats.get_item("survived_mutants")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                
                let timeout_mutants: u32 = stats.get_item("timeout_mutants")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0);
                
                let mutation_score = if total_mutants > 0 {
                    killed_mutants as f64 / total_mutants as f64
                } else {
                    0.0
                };
                
                Ok(MutationOutput {
                    total_mutants,
                    killed_mutants,
                    survived_mutants,
                    timeout_mutants,
                    mutation_score,
                    execution_time_secs: 0.0, // Will be set later
                    file_results: Vec::new(), // TODO: Parse file-level results
                })
            })
        }).await
        .map_err(|e| SensorError::Generic(format!("Task join error: {}", e)))?;
        
        result.map_err(|e| SensorError::Tool(format!("Mutmut error: {}", e)))
    }
    
    /// Run mutmut via CLI (fallback when Python feature is disabled)
    #[cfg(not(feature = "python"))]
    async fn run_mutmut(&self, project_path: &str, timeout_secs: u64) -> Result<MutationOutput> {
        self.run_mutmut_subprocess(project_path, timeout_secs).await
    }
    
    /// Alternative: Run mutmut as subprocess
    async fn run_mutmut_subprocess(&self, project_path: &str, timeout_secs: u64) -> Result<MutationOutput> {
        let start = std::time::Instant::now();
        
        // Run mutmut
        let output = timeout(
            Duration::from_secs(timeout_secs),
            tokio::process::Command::new(&self.mutmut_path)
                .arg("run")
                .arg("--paths-to-mutate")
                .arg(project_path)
                .arg("--simple-output")
                .output()
        ).await
        .map_err(|_| SensorError::Timeout)?
        .map_err(|e| SensorError::Tool(format!("Failed to run mutmut: {}", e)))?;
        
        if !output.status.success() {
            return Err(SensorError::Tool(format!(
                "Mutmut failed with status: {}",
                output.status
            )));
        }
        
        // Parse mutmut output
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_mutmut_output(&stdout, start.elapsed().as_secs_f64())
    }
    
    /// Parse mutmut output
    fn parse_mutmut_output(&self, output: &str, execution_time: f64) -> Result<MutationOutput> {
        // Parse mutmut summary output
        // Format: "Killed N out of M mutants"
        let mut total_mutants: u32 = 0;
        let mut killed_mutants: u32 = 0;
        
        for line in output.lines() {
            if line.contains("out of") && line.contains("mutants") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    killed_mutants = parts[1].parse().unwrap_or(0);
                    total_mutants = parts[4].parse().unwrap_or(0);
                }
            }
        }
        
        let survived_mutants: u32 = total_mutants.saturating_sub(killed_mutants);
        let mutation_score = if total_mutants > 0 {
            killed_mutants as f64 / total_mutants as f64
        } else {
            0.0
        };
        
        Ok(MutationOutput {
            total_mutants,
            killed_mutants,
            survived_mutants,
            timeout_mutants: 0, // Not available in simple output
            mutation_score,
            execution_time_secs: execution_time,
            file_results: Vec::new(), // Would need to parse detailed output
        })
    }
}

impl Default for MutationSensor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Sensor for MutationSensor {
    type Output = MutationOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let timeout_secs = context.timeout_secs.unwrap_or(600); // 10 minutes default
        
        // Try Python integration first, fall back to subprocess
        match self.run_mutmut(&context.project_path, timeout_secs).await {
            Ok(output) => Ok(output),
            Err(_) => {
                // Fallback to subprocess
                self.run_mutmut_subprocess(&context.project_path, timeout_secs).await
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "mutation"
    }
    
    fn is_available(&self) -> bool {
        // Check if mutmut is available
        Command::new(&self.mutmut_path)
            .arg("--version")
            .output()
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_mutmut_output() {
        let sensor = MutationSensor::new();
        let output = "Killed 85 out of 100 mutants";
        
        let result = sensor.parse_mutmut_output(output, 10.5).unwrap();
        assert_eq!(result.total_mutants, 100);
        assert_eq!(result.killed_mutants, 85);
        assert_eq!(result.survived_mutants, 15);
        assert_eq!(result.mutation_score, 0.85);
        assert_eq!(result.execution_time_secs, 10.5);
    }
}
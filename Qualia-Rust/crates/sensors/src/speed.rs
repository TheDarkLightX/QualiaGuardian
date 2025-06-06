//! Speed sensor for test execution performance measurement

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::process::Command;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Speed sensor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedOutput {
    /// Median test execution time in milliseconds
    pub median_test_time_ms: f64,
    /// Mean test execution time
    pub mean_test_time_ms: f64,
    /// 95th percentile time
    pub p95_test_time_ms: f64,
    /// 99th percentile time
    pub p99_test_time_ms: f64,
    /// Total test suite execution time
    pub total_time_ms: f64,
    /// Number of tests measured
    pub test_count: usize,
    /// Tests categorized by speed
    pub speed_categories: SpeedCategories,
    /// Individual test timings
    pub test_timings: Vec<TestTiming>,
}

/// Speed categories for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedCategories {
    /// Tests < 100ms
    pub fast: usize,
    /// Tests 100ms - 1s
    pub moderate: usize,
    /// Tests 1s - 5s
    pub slow: usize,
    /// Tests > 5s
    pub very_slow: usize,
}

/// Individual test timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestTiming {
    /// Test name
    pub name: String,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// File containing the test
    pub file: Option<String>,
}

/// Speed sensor for measuring test execution performance
#[derive(Debug)]
pub struct SpeedSensor {
    /// Test framework to use
    test_framework: TestFramework,
    /// Timeout for test execution
    timeout: Duration,
}

/// Supported test frameworks
#[derive(Debug, Clone)]
pub enum TestFramework {
    /// Python pytest
    Pytest,
    /// Rust cargo test
    CargoTest,
    /// JavaScript Jest
    Jest,
    /// Generic test runner
    Generic(String),
}

impl SpeedSensor {
    /// Create a new speed sensor
    pub fn new(test_framework: TestFramework) -> Self {
        Self {
            test_framework,
            timeout: Duration::from_secs(300), // 5 minute default timeout
        }
    }
    
    /// Create sensor with custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Execute tests and measure timing
    async fn run_tests(&self, project_path: &str) -> Result<Vec<TestTiming>> {
        match &self.test_framework {
            TestFramework::Pytest => self.run_pytest(project_path).await,
            TestFramework::CargoTest => self.run_cargo_test(project_path).await,
            TestFramework::Jest => self.run_jest(project_path).await,
            TestFramework::Generic(cmd) => self.run_generic(project_path, cmd).await,
        }
    }
    
    /// Run pytest and parse timing output
    async fn run_pytest(&self, project_path: &str) -> Result<Vec<TestTiming>> {
        let output = Command::new("python")
            .arg("-m")
            .arg("pytest")
            .arg("--durations=0")
            .arg("--tb=no")
            .arg("-q")
            .current_dir(project_path)
            .output()
            .await
            .map_err(|e| SensorError::Io(e))?;
        
        if !output.status.success() && output.status.code() != Some(1) {
            // Exit code 1 might just mean some tests failed, which is OK for timing
            return Err(SensorError::Generic(
                format!("pytest failed with status: {:?}", output.status)
            ));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_pytest_durations(&stdout)
    }
    
    /// Parse pytest durations output
    fn parse_pytest_durations(&self, output: &str) -> Result<Vec<TestTiming>> {
        let mut timings = Vec::new();
        let mut in_durations = false;
        
        for line in output.lines() {
            if line.contains("slowest durations") {
                in_durations = true;
                continue;
            }
            
            if !in_durations {
                continue;
            }
            
            // Parse lines like "0.50s call     test_example.py::TestClass::test_method"
            if let Some(duration_end) = line.find("s ") {
                if let Ok(duration) = line[..duration_end].trim().parse::<f64>() {
                    let remaining = &line[duration_end + 2..];
                    if let Some(test_start) = remaining.find("::") {
                        let file_part = &remaining[..test_start];
                        let test_name = remaining.trim();
                        
                        timings.push(TestTiming {
                            name: test_name.to_string(),
                            duration_ms: duration * 1000.0,
                            file: Some(file_part.to_string()),
                        });
                    }
                }
            }
        }
        
        Ok(timings)
    }
    
    /// Run cargo test and parse timing output
    async fn run_cargo_test(&self, project_path: &str) -> Result<Vec<TestTiming>> {
        let output = Command::new("cargo")
            .arg("test")
            .arg("--")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env("RUST_TEST_TIME", "1")
            .current_dir(project_path)
            .output()
            .await
            .map_err(|e| SensorError::Io(e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_cargo_test_timings(&stdout)
    }
    
    /// Parse cargo test timing output
    fn parse_cargo_test_timings(&self, output: &str) -> Result<Vec<TestTiming>> {
        let mut timings = Vec::new();
        
        // Parse lines like "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.23s"
        for line in output.lines() {
            if line.starts_with("test ") && line.contains(" ... ") {
                let parts: Vec<&str> = line.split(" ... ").collect();
                if parts.len() == 2 {
                    let test_name = parts[0].trim_start_matches("test ").trim();
                    
                    // Look for timing in format "ok (1.23s)"
                    if let Some(time_start) = parts[1].find('(') {
                        if let Some(time_end) = parts[1].find("s)") {
                            let time_str = &parts[1][time_start + 1..time_end];
                            if let Ok(duration) = time_str.parse::<f64>() {
                                timings.push(TestTiming {
                                    name: test_name.to_string(),
                                    duration_ms: duration * 1000.0,
                                    file: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(timings)
    }
    
    /// Run Jest and parse timing output
    async fn run_jest(&self, project_path: &str) -> Result<Vec<TestTiming>> {
        let output = Command::new("npx")
            .arg("jest")
            .arg("--verbose")
            .arg("--no-coverage")
            .current_dir(project_path)
            .output()
            .await
            .map_err(|e| SensorError::Io(e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_jest_timings(&stdout)
    }
    
    /// Parse Jest timing output
    fn parse_jest_timings(&self, output: &str) -> Result<Vec<TestTiming>> {
        let mut timings = Vec::new();
        
        // Parse lines like "✓ should do something (123 ms)"
        for line in output.lines() {
            if line.contains(" ms)") || line.contains(" s)") {
                if let Some(time_start) = line.rfind('(') {
                    if let Some(time_end) = line.rfind(')') {
                        let time_part = &line[time_start + 1..time_end];
                        let (duration_ms, test_name) = if time_part.ends_with(" ms") {
                            let ms_str = time_part.trim_end_matches(" ms");
                            if let Ok(ms) = ms_str.parse::<f64>() {
                                let name = line[..time_start].trim();
                                (ms, name)
                            } else {
                                continue;
                            }
                        } else if time_part.ends_with(" s") {
                            let s_str = time_part.trim_end_matches(" s");
                            if let Ok(s) = s_str.parse::<f64>() {
                                let name = line[..time_start].trim();
                                (s * 1000.0, name)
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        };
                        
                        timings.push(TestTiming {
                            name: test_name.to_string(),
                            duration_ms,
                            file: None,
                        });
                    }
                }
            }
        }
        
        Ok(timings)
    }
    
    /// Run generic test command
    async fn run_generic(&self, project_path: &str, command: &str) -> Result<Vec<TestTiming>> {
        let start = Instant::now();
        
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(project_path)
            .output()
            .await
            .map_err(|e| SensorError::Io(e))?;
        
        let total_duration = start.elapsed();
        
        // For generic commands, we can only measure total time
        Ok(vec![TestTiming {
            name: "All Tests".to_string(),
            duration_ms: total_duration.as_millis() as f64,
            file: None,
        }])
    }
    
    /// Calculate speed categories
    fn categorize_timings(timings: &[TestTiming]) -> SpeedCategories {
        let mut categories = SpeedCategories {
            fast: 0,
            moderate: 0,
            slow: 0,
            very_slow: 0,
        };
        
        for timing in timings {
            if timing.duration_ms < 100.0 {
                categories.fast += 1;
            } else if timing.duration_ms < 1000.0 {
                categories.moderate += 1;
            } else if timing.duration_ms < 5000.0 {
                categories.slow += 1;
            } else {
                categories.very_slow += 1;
            }
        }
        
        categories
    }
    
    /// Calculate percentile from sorted timings
    fn calculate_percentile(sorted_times: &[f64], percentile: f64) -> f64 {
        if sorted_times.is_empty() {
            return 0.0;
        }
        
        let index = ((percentile / 100.0) * (sorted_times.len() - 1) as f64) as usize;
        sorted_times[index.min(sorted_times.len() - 1)]
    }
}

impl Default for SpeedSensor {
    fn default() -> Self {
        Self::new(TestFramework::Pytest)
    }
}

#[async_trait]
impl Sensor for SpeedSensor {
    type Output = SpeedOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        // Run tests and collect timing data
        let timings = self.run_tests(&context.project_path).await?;
        
        if timings.is_empty() {
            return Err(SensorError::Generic(
                "No test timing data collected".to_string()
            ));
        }
        
        // Extract just the durations and sort them
        let mut durations: Vec<f64> = timings.iter()
            .map(|t| t.duration_ms)
            .collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate statistics
        let median_test_time_ms = Self::calculate_percentile(&durations, 50.0);
        let p95_test_time_ms = Self::calculate_percentile(&durations, 95.0);
        let p99_test_time_ms = Self::calculate_percentile(&durations, 99.0);
        
        let mean_test_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        let total_time_ms = durations.iter().sum();
        
        // Categorize speeds
        let speed_categories = Self::categorize_timings(&timings);
        
        Ok(SpeedOutput {
            median_test_time_ms,
            mean_test_time_ms,
            p95_test_time_ms,
            p99_test_time_ms,
            total_time_ms,
            test_count: timings.len(),
            speed_categories,
            test_timings: timings,
        })
    }
    
    fn name(&self) -> &'static str {
        "speed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pytest_duration_parsing() {
        let sensor = SpeedSensor::new(TestFramework::Pytest);
        let output = r#"
========================= slowest durations =========================
1.23s call     tests/test_example.py::TestClass::test_slow_method
0.45s call     tests/test_other.py::test_quick_function
0.01s call     tests/test_fast.py::test_very_fast
"#;
        
        let timings = sensor.parse_pytest_durations(output).unwrap();
        assert_eq!(timings.len(), 3);
        assert_eq!(timings[0].duration_ms, 1230.0);
        assert_eq!(timings[1].duration_ms, 450.0);
        assert_eq!(timings[2].duration_ms, 10.0);
    }
    
    #[test]
    fn test_speed_categorization() {
        let timings = vec![
            TestTiming { name: "fast".to_string(), duration_ms: 50.0, file: None },
            TestTiming { name: "moderate".to_string(), duration_ms: 500.0, file: None },
            TestTiming { name: "slow".to_string(), duration_ms: 2000.0, file: None },
            TestTiming { name: "very_slow".to_string(), duration_ms: 10000.0, file: None },
        ];
        
        let categories = SpeedSensor::categorize_timings(&timings);
        assert_eq!(categories.fast, 1);
        assert_eq!(categories.moderate, 1);
        assert_eq!(categories.slow, 1);
        assert_eq!(categories.very_slow, 1);
    }
    
    #[test]
    fn test_percentile_calculation() {
        let times = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        
        assert_eq!(SpeedSensor::calculate_percentile(&times, 50.0), 50.0);
        assert_eq!(SpeedSensor::calculate_percentile(&times, 95.0), 90.0);
        assert_eq!(SpeedSensor::calculate_percentile(&times, 99.0), 100.0);
    }
}
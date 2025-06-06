//! Flakiness sensor for detecting test instability in CI/CD

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Sensor, SensorContext, SensorError, Result};

/// Flakiness sensor output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlakinessOutput {
    /// Overall flakiness rate (0.0 to 1.0)
    pub flakiness_rate: f64,
    /// Number of flaky tests detected
    pub flaky_test_count: usize,
    /// Total tests analyzed
    pub total_test_count: usize,
    /// CI runs analyzed
    pub ci_runs_analyzed: usize,
    /// Individual test flakiness data
    pub test_flakiness: Vec<TestFlakiness>,
    /// CI/CD platform used
    pub ci_platform: String,
}

/// Individual test flakiness information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFlakiness {
    /// Test name
    pub test_name: String,
    /// Number of times the test passed
    pub pass_count: usize,
    /// Number of times the test failed
    pub fail_count: usize,
    /// Flakiness score (fail_count / total_runs)
    pub flakiness_score: f64,
    /// Recent failure reasons if available
    pub failure_reasons: Vec<String>,
}

/// Flakiness sensor for detecting unstable tests
#[derive(Debug)]
pub struct FlakinessSensor {
    /// CI/CD client to use
    ci_client: Box<dyn CIClient>,
    /// Number of recent runs to analyze
    run_limit: usize,
}

/// CI/CD client trait for different platforms
#[async_trait]
pub trait CIClient: Send + Sync + std::fmt::Debug {
    /// Get recent test results
    async fn get_test_results(&self, project_path: &str, run_limit: usize) -> Result<Vec<TestRun>>;
    /// Get platform name
    fn platform_name(&self) -> &str;
}

/// Test run information
#[derive(Debug, Clone)]
pub struct TestRun {
    /// Run ID
    pub run_id: String,
    /// Test results
    pub test_results: Vec<TestResult>,
    /// Run timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Test status
    pub status: TestStatus,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Test status
#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
}

/// GitHub Actions CI client
#[derive(Debug)]
pub struct GitHubActionsClient {
    token: Option<String>,
}

impl GitHubActionsClient {
    pub fn new() -> Self {
        let token = std::env::var("GITHUB_TOKEN").ok();
        Self { token }
    }
}

#[async_trait]
impl CIClient for GitHubActionsClient {
    async fn get_test_results(&self, _project_path: &str, run_limit: usize) -> Result<Vec<TestRun>> {
        if self.token.is_none() {
            return Err(SensorError::Generic(
                "GITHUB_TOKEN environment variable not set".to_string()
            ));
        }
        
        // In real implementation, would make API calls to GitHub
        // For now, simulate with placeholder data
        let mut runs = Vec::new();
        
        for i in 0..run_limit.min(5) {
            let test_results = vec![
                TestResult {
                    name: "test_example".to_string(),
                    status: if i == 2 { TestStatus::Failed } else { TestStatus::Passed },
                    error_message: if i == 2 { Some("Network timeout".to_string()) } else { None },
                },
            ];
            
            runs.push(TestRun {
                run_id: format!("gh-run-{}", i),
                test_results,
                timestamp: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            });
        }
        
        Ok(runs)
    }
    
    fn platform_name(&self) -> &str {
        "GitHub Actions"
    }
}

/// Jenkins CI client
#[derive(Debug)]
pub struct JenkinsClient {
    url: String,
    user: Option<String>,
    token: Option<String>,
}

impl JenkinsClient {
    pub fn new() -> Self {
        Self {
            url: std::env::var("JENKINS_URL").unwrap_or_else(|_| "http://localhost:8080".to_string()),
            user: std::env::var("JENKINS_USER").ok(),
            token: std::env::var("JENKINS_TOKEN").ok(),
        }
    }
}

#[async_trait]
impl CIClient for JenkinsClient {
    async fn get_test_results(&self, _project_path: &str, run_limit: usize) -> Result<Vec<TestRun>> {
        // In real implementation, would make API calls to Jenkins
        // For now, simulate with placeholder data
        let mut runs = Vec::new();
        
        for i in 0..run_limit.min(3) {
            let test_results = vec![
                TestResult {
                    name: "test_jenkins".to_string(),
                    status: TestStatus::Passed,
                    error_message: None,
                },
            ];
            
            runs.push(TestRun {
                run_id: format!("jenkins-build-{}", i + 100),
                test_results,
                timestamp: chrono::Utc::now() - chrono::Duration::hours(i as i64 * 2),
            });
        }
        
        Ok(runs)
    }
    
    fn platform_name(&self) -> &str {
        "Jenkins"
    }
}

/// Local/mock CI client for testing
#[derive(Debug)]
pub struct LocalCIClient;

#[async_trait]
impl CIClient for LocalCIClient {
    async fn get_test_results(&self, _project_path: &str, run_limit: usize) -> Result<Vec<TestRun>> {
        // Simulate test results for local testing
        let mut runs = Vec::new();
        
        for i in 0..run_limit.min(5) {
            let test_results = vec![
                TestResult {
                    name: "test_stable".to_string(),
                    status: TestStatus::Passed,
                    error_message: None,
                },
                TestResult {
                    name: "test_flaky".to_string(),
                    status: if i % 2 == 0 { TestStatus::Passed } else { TestStatus::Failed },
                    error_message: if i % 2 == 0 { None } else { Some("Random failure".to_string()) },
                },
                TestResult {
                    name: "test_always_fails".to_string(),
                    status: TestStatus::Failed,
                    error_message: Some("Consistent failure".to_string()),
                },
            ];
            
            runs.push(TestRun {
                run_id: format!("local-run-{}", i),
                test_results,
                timestamp: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            });
        }
        
        Ok(runs)
    }
    
    fn platform_name(&self) -> &str {
        "Local"
    }
}

impl FlakinessSensor {
    /// Create a new flakiness sensor with specified CI client
    pub fn new(ci_client: Box<dyn CIClient>) -> Self {
        Self {
            ci_client,
            run_limit: 10, // Default to analyzing last 10 runs
        }
    }
    
    /// Create sensor with custom run limit
    pub fn with_run_limit(mut self, limit: usize) -> Self {
        self.run_limit = limit;
        self
    }
    
    /// Create sensor with auto-detected CI platform
    pub fn auto_detect() -> Self {
        // Detect CI platform based on environment variables
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            Self::new(Box::new(GitHubActionsClient::new()))
        } else if std::env::var("JENKINS_URL").is_ok() {
            Self::new(Box::new(JenkinsClient::new()))
        } else {
            // Default to local/mock client
            Self::new(Box::new(LocalCIClient))
        }
    }
    
    /// Analyze test runs for flakiness
    fn analyze_flakiness(&self, test_runs: &[TestRun]) -> (f64, Vec<TestFlakiness>) {
        let mut test_stats: HashMap<String, (usize, usize)> = HashMap::new();
        let mut test_failures: HashMap<String, Vec<String>> = HashMap::new();
        
        // Collect statistics for each test
        for run in test_runs {
            for result in &run.test_results {
                let (pass_count, fail_count) = test_stats.entry(result.name.clone())
                    .or_insert((0, 0));
                
                match result.status {
                    TestStatus::Passed => *pass_count += 1,
                    TestStatus::Failed => {
                        *fail_count += 1;
                        if let Some(error) = &result.error_message {
                            test_failures.entry(result.name.clone())
                                .or_insert_with(Vec::new)
                                .push(error.clone());
                        }
                    },
                    TestStatus::Skipped => {}, // Don't count skipped tests
                }
            }
        }
        
        // Calculate flakiness for each test
        let mut test_flakiness = Vec::new();
        let mut total_flaky = 0;
        
        for (test_name, (pass_count, fail_count)) in test_stats {
            let total_runs = pass_count + fail_count;
            if total_runs > 0 {
                let flakiness_score = fail_count as f64 / total_runs as f64;
                
                // A test is considered flaky if it sometimes passes and sometimes fails
                if pass_count > 0 && fail_count > 0 {
                    total_flaky += 1;
                }
                
                let failure_reasons = test_failures.get(&test_name)
                    .cloned()
                    .unwrap_or_default();
                
                test_flakiness.push(TestFlakiness {
                    test_name,
                    pass_count,
                    fail_count,
                    flakiness_score,
                    failure_reasons: failure_reasons.into_iter().take(5).collect(), // Keep last 5 failures
                });
            }
        }
        
        // Sort by flakiness score (most flaky first)
        test_flakiness.sort_by(|a, b| 
            b.flakiness_score.partial_cmp(&a.flakiness_score).unwrap()
        );
        
        // Calculate overall flakiness rate
        let total_tests = test_flakiness.len();
        let flakiness_rate = if total_tests > 0 {
            total_flaky as f64 / total_tests as f64
        } else {
            0.0
        };
        
        (flakiness_rate, test_flakiness)
    }
}

impl Default for FlakinessSensor {
    fn default() -> Self {
        Self::auto_detect()
    }
}

#[async_trait]
impl Sensor for FlakinessSensor {
    type Output = FlakinessOutput;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        // Get recent test runs from CI/CD
        let test_runs = self.ci_client.get_test_results(&context.project_path, self.run_limit).await?;
        
        if test_runs.is_empty() {
            return Err(SensorError::Generic(
                "No test runs found in CI/CD history".to_string()
            ));
        }
        
        // Analyze flakiness
        let (flakiness_rate, test_flakiness) = self.analyze_flakiness(&test_runs);
        
        // Count unique tests and flaky tests
        let total_test_count = test_flakiness.len();
        let flaky_test_count = test_flakiness.iter()
            .filter(|t| t.pass_count > 0 && t.fail_count > 0)
            .count();
        
        Ok(FlakinessOutput {
            flakiness_rate,
            flaky_test_count,
            total_test_count,
            ci_runs_analyzed: test_runs.len(),
            test_flakiness,
            ci_platform: self.ci_client.platform_name().to_string(),
        })
    }
    
    fn name(&self) -> &'static str {
        "flakiness"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_flakiness_analysis() {
        let sensor = FlakinessSensor::new(Box::new(LocalCIClient));
        let context = SensorContext {
            project_path: "/tmp/test".to_string(),
            ..Default::default()
        };
        
        let output = sensor.measure(&context).await.unwrap();
        
        // With LocalCIClient, we should have predictable results
        assert!(output.ci_runs_analyzed > 0);
        assert!(output.flaky_test_count > 0); // test_flaky should be detected
        assert!(output.flakiness_rate > 0.0 && output.flakiness_rate < 1.0);
        
        // Check individual test results
        let flaky_test = output.test_flakiness.iter()
            .find(|t| t.test_name == "test_flaky")
            .expect("test_flaky should be in results");
        
        assert!(flaky_test.pass_count > 0);
        assert!(flaky_test.fail_count > 0);
        assert!(flaky_test.flakiness_score > 0.0 && flaky_test.flakiness_score < 1.0);
    }
    
    #[test]
    fn test_flakiness_calculation() {
        let test_runs = vec![
            TestRun {
                run_id: "1".to_string(),
                test_results: vec![
                    TestResult { name: "test_a".to_string(), status: TestStatus::Passed, error_message: None },
                    TestResult { name: "test_b".to_string(), status: TestStatus::Failed, error_message: Some("Error".to_string()) },
                ],
                timestamp: chrono::Utc::now(),
            },
            TestRun {
                run_id: "2".to_string(),
                test_results: vec![
                    TestResult { name: "test_a".to_string(), status: TestStatus::Failed, error_message: Some("Flaky".to_string()) },
                    TestResult { name: "test_b".to_string(), status: TestStatus::Failed, error_message: Some("Error".to_string()) },
                ],
                timestamp: chrono::Utc::now(),
            },
        ];
        
        let sensor = FlakinessSensor::default();
        let (rate, flakiness) = sensor.analyze_flakiness(&test_runs);
        
        assert_eq!(rate, 0.5); // 1 flaky test out of 2
        assert_eq!(flakiness.len(), 2);
        
        // test_a should be flaky (1 pass, 1 fail)
        let test_a = flakiness.iter().find(|t| t.test_name == "test_a").unwrap();
        assert_eq!(test_a.pass_count, 1);
        assert_eq!(test_a.fail_count, 1);
        assert_eq!(test_a.flakiness_score, 0.5);
        
        // test_b should not be flaky (0 pass, 2 fail)
        let test_b = flakiness.iter().find(|t| t.test_name == "test_b").unwrap();
        assert_eq!(test_b.pass_count, 0);
        assert_eq!(test_b.fail_count, 2);
        assert_eq!(test_b.flakiness_score, 1.0);
    }
}
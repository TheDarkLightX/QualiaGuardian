//! Sensor modules for collecting quality metrics

use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;
use std::collections::HashMap;
use thiserror::Error;

pub mod mutation;
pub mod assertion_iq;
pub mod behaviour_coverage;
pub mod speed;
pub mod flakiness;
pub mod chs;
pub mod security;
pub mod arch;

/// Errors that can occur during sensor operations
#[derive(Debug, Error)]
pub enum SensorError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),
    
    /// External tool error
    #[error("External tool error: {0}")]
    Tool(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Network error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    /// Timeout error
    #[error("Operation timed out")]
    Timeout,
    
    /// Generic sensor error
    #[error("Sensor error: {0}")]
    Generic(String),
}

/// Result type for sensor operations
pub type Result<T> = std::result::Result<T, SensorError>;

/// Context provided to sensors for execution
#[derive(Debug, Clone, Default)]
pub struct SensorContext {
    /// Path to the project being analyzed
    pub project_path: String,
    /// Primary programming language
    pub language: String,
    /// Additional configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Timeout in seconds (optional)
    pub timeout_secs: Option<u64>,
}

impl SensorContext {
    /// Create a new sensor context
    pub fn new(project_path: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            project_path: project_path.into(),
            language: language.into(),
            config: HashMap::new(),
            timeout_secs: None,
        }
    }
    
    /// Set a configuration value
    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

/// Base trait for all sensors
#[async_trait]
pub trait Sensor: Send + Sync + Debug {
    /// Output type for this sensor
    type Output: Serialize + DeserializeOwned + Send + Sync + Debug;
    
    /// Measure quality metrics
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output>;
    
    /// Get sensor name
    fn name(&self) -> &'static str;
    
    /// Get sensor version
    fn version(&self) -> &'static str {
        "1.0"
    }
    
    /// Check if sensor is available in current environment
    fn is_available(&self) -> bool {
        true
    }
}

/// Registry for managing sensors
pub struct SensorRegistry {
    sensors: HashMap<String, Box<dyn Sensor<Output = serde_json::Value>>>,
}

impl SensorRegistry {
    /// Create a new sensor registry
    pub fn new() -> Self {
        Self {
            sensors: HashMap::new(),
        }
    }
    
    /// Register a sensor
    pub fn register<S>(&mut self, sensor: S)
    where
        S: Sensor + 'static,
        S::Output: Serialize + DeserializeOwned,
    {
        let name = sensor.name().to_string();
        let boxed = Box::new(SensorAdapter::new(sensor));
        self.sensors.insert(name, boxed);
    }
    
    /// Get a sensor by name
    pub fn get(&self, name: &str) -> Option<&dyn Sensor<Output = serde_json::Value>> {
        self.sensors.get(name).map(|s| s.as_ref())
    }
    
    /// List all registered sensors
    pub fn list(&self) -> Vec<&str> {
        self.sensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Adapter to convert typed sensors to dynamic output
struct SensorAdapter<S: Sensor> {
    inner: S,
}

impl<S: Sensor> SensorAdapter<S> {
    fn new(sensor: S) -> Self {
        Self { inner: sensor }
    }
}

impl<S: Sensor> Debug for SensorAdapter<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SensorAdapter<{}>", self.inner.name())
    }
}

#[async_trait]
impl<S> Sensor for SensorAdapter<S>
where
    S: Sensor + Send + Sync,
    S::Output: Serialize + DeserializeOwned,
{
    type Output = serde_json::Value;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output> {
        let output = self.inner.measure(context).await?;
        Ok(serde_json::to_value(output).map_err(|e| SensorError::Generic(e.to_string()))?)
    }
    
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    
    fn version(&self) -> &'static str {
        self.inner.version()
    }
    
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

/// Parallel sensor executor
pub struct SensorExecutor {
    registry: SensorRegistry,
}

impl SensorExecutor {
    /// Create a new sensor executor
    pub fn new(registry: SensorRegistry) -> Self {
        Self { registry }
    }
    
    /// Execute multiple sensors in parallel
    pub async fn execute_all(
        &self,
        context: &SensorContext,
        sensor_names: &[&str],
    ) -> HashMap<String, Result<serde_json::Value>> {
        use futures::future::join_all;
        
        let futures: Vec<_> = sensor_names
            .iter()
            .filter_map(|name| {
                self.registry.get(name).map(|sensor| {
                    let ctx = context.clone();
                    let sensor_name = name.to_string();
                    async move {
                        let result = sensor.measure(&ctx).await;
                        (sensor_name, result)
                    }
                })
            })
            .collect();
        
        let results = join_all(futures).await;
        results.into_iter().collect()
    }
}

/// Create default sensor registry with all built-in sensors
pub fn create_default_registry() -> SensorRegistry {
    let mut registry = SensorRegistry::new();
    
    // Register all built-in sensors
    registry.register(mutation::MutationSensor::new());
    registry.register(assertion_iq::AssertionIQSensor::new());
    registry.register(behaviour_coverage::BehaviorCoverageSensor::new());
    registry.register(speed::SpeedSensor::new(speed::TestFramework::Pytest));
    registry.register(flakiness::FlakinessSensor::new(
        Box::new(flakiness::GitHubActionsClient::new())
    ));
    registry.register(chs::CHSSensor::new());
    registry.register(security::SecuritySensor::new());
    registry.register(arch::ArchSensor::new());
    
    registry
}
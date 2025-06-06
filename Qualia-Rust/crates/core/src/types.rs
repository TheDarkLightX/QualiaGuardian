//! Core types for quality metrics

use serde::{Deserialize, Serialize};
use ordered_float::OrderedFloat;
use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use crate::error::{QualityError, Result};

/// A quality score bounded between 0.0 and 1.0
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QualityScore(OrderedFloat<f64>);

impl Serialize for QualityScore {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.into_inner().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QualityScore {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        QualityScore::new(value).map_err(de::Error::custom)
    }
}

impl QualityScore {
    /// Create a new quality score
    /// 
    /// # Errors
    /// Returns an error if the value is outside [0.0, 1.0]
    pub fn new(value: f64) -> Result<Self> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(OrderedFloat(value)))
        } else {
            Err(QualityError::ScoreOutOfBounds(value))
        }
    }
    
    /// Create a quality score from a raw value, clamping to [0.0, 1.0]
    pub fn from_clamped(value: f64) -> Self {
        Self(OrderedFloat(value.clamp(0.0, 1.0)))
    }
    
    /// Get the inner value
    pub fn value(&self) -> f64 {
        self.0.into_inner()
    }
}

impl From<QualityScore> for f64 {
    fn from(score: QualityScore) -> f64 {
        score.value()
    }
}

/// Risk classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    /// Aerospace: requires >0.95 score
    Aerospace,
    /// Medical: requires >0.90 score
    Medical,
    /// Financial: requires >0.80 score
    Financial,
    /// Enterprise: requires >0.70 score
    Enterprise,
    /// Standard: requires >0.60 score
    Standard,
    /// Prototype: requires >0.40 score
    Prototype,
    /// Experimental: requires >=0.0 score
    Experimental,
}

impl RiskClass {
    /// Get the minimum required score for this risk class
    pub fn min_score(&self) -> f64 {
        match self {
            RiskClass::Aerospace => 0.95,
            RiskClass::Medical => 0.90,
            RiskClass::Financial => 0.80,
            RiskClass::Enterprise => 0.70,
            RiskClass::Standard => 0.60,
            RiskClass::Prototype => 0.40,
            RiskClass::Experimental => 0.0,
        }
    }
    
    /// Determine risk class from a score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.95 {
            RiskClass::Aerospace
        } else if score >= 0.90 {
            RiskClass::Medical
        } else if score >= 0.80 {
            RiskClass::Financial
        } else if score >= 0.70 {
            RiskClass::Enterprise
        } else if score >= 0.60 {
            RiskClass::Standard
        } else if score >= 0.40 {
            RiskClass::Prototype
        } else {
            RiskClass::Experimental
        }
    }
}

/// Individual component result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    /// Component name
    pub name: String,
    /// Raw value before normalization
    pub raw_value: f64,
    /// Normalized value (0-1)
    pub normalized_value: QualityScore,
    /// Weight applied to this component
    pub weight: f64,
    /// Optional description
    pub description: Option<String>,
}

/// Letter grade for quality scores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityGrade {
    #[serde(rename = "A+")]
    APlus,
    A,
    B,
    C,
    F,
}

impl QualityGrade {
    /// Get grade from score
    pub fn from_score(score: f64) -> Self {
        if score >= 0.9 {
            QualityGrade::APlus
        } else if score >= 0.8 {
            QualityGrade::A
        } else if score >= 0.7 {
            QualityGrade::B
        } else if score >= 0.6 {
            QualityGrade::C
        } else {
            QualityGrade::F
        }
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            QualityGrade::APlus => "A+",
            QualityGrade::A => "A",
            QualityGrade::B => "B",
            QualityGrade::C => "C",
            QualityGrade::F => "F",
        }
    }
}
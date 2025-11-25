//! Core types for quality metrics
//!
//! This module provides type-safe wrappers for quality scores and related types,
//! ensuring compile-time guarantees about value ranges and invariants.

use serde::{Deserialize, Serialize};
use ordered_float::OrderedFloat;
use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use crate::error::{QualityError, Result};

/// A quality score bounded between 0.0 and 1.0
///
/// This type enforces the invariant that all quality scores are in the range [0.0, 1.0]
/// at compile time, preventing invalid scores from propagating through the system.
///
/// # Examples
///
/// ```rust
/// use qualia_core::QualityScore;
///
/// // Valid score
/// let score = QualityScore::new(0.85)?;
/// assert_eq!(score.value(), 0.85);
///
/// // Invalid score (returns error)
/// assert!(QualityScore::new(1.5).is_err());
///
/// // Clamp to valid range
/// let clamped = QualityScore::from_clamped(1.5);
/// assert_eq!(clamped.value(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::QualityScore;
    /// 
    /// let score = QualityScore::new(0.85)?;
    /// assert_eq!(score.value(), 0.85);
    /// ```
    pub fn new(value: f64) -> Result<Self> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(OrderedFloat(value)))
        } else {
            Err(QualityError::ScoreOutOfBounds(value))
        }
    }
    
    /// Create a quality score from a raw value, clamping to [0.0, 1.0]
    /// 
    /// This is useful when you want to ensure a value is in range without
    /// failing on out-of-bounds values.
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::QualityScore;
    /// 
    /// let clamped = QualityScore::from_clamped(1.5);
    /// assert_eq!(clamped.value(), 1.0);
    /// 
    /// let clamped_neg = QualityScore::from_clamped(-0.5);
    /// assert_eq!(clamped_neg.value(), 0.0);
    /// ```
    pub fn from_clamped(value: f64) -> Self {
        Self(OrderedFloat(value.clamp(0.0, 1.0)))
    }
    
    /// Get the inner value
    pub fn value(&self) -> f64 {
        self.0.into_inner()
    }
    
    /// Check if the score is excellent (>= 0.9)
    pub fn is_excellent(&self) -> bool {
        self.value() >= 0.9
    }
    
    /// Check if the score is good (>= 0.7)
    pub fn is_good(&self) -> bool {
        self.value() >= 0.7
    }
    
    /// Check if the score is acceptable (>= 0.6)
    pub fn is_acceptable(&self) -> bool {
        self.value() >= 0.6
    }
}

impl From<QualityScore> for f64 {
    fn from(score: QualityScore) -> f64 {
        score.value()
    }
}

impl std::ops::Add for QualityScore {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_clamped(self.value() + rhs.value())
    }
}

impl std::ops::Sub for QualityScore {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_clamped(self.value() - rhs.value())
    }
}

impl std::ops::Mul<f64> for QualityScore {
    type Output = Self;
    
    fn mul(self, rhs: f64) -> Self::Output {
        Self::from_clamped(self.value() * rhs)
    }
}

/// Risk classification levels
///
/// Different risk classes have different minimum quality score requirements.
/// This enum ensures type safety when working with risk classifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::RiskClass;
    /// 
    /// assert_eq!(RiskClass::Aerospace.min_score(), 0.95);
    /// assert_eq!(RiskClass::Standard.min_score(), 0.60);
    /// ```
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
    /// 
    /// Returns the highest risk class that the score qualifies for.
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::RiskClass;
    /// 
    /// assert_eq!(RiskClass::from_score(0.96), RiskClass::Aerospace);
    /// assert_eq!(RiskClass::from_score(0.75), RiskClass::Enterprise);
    /// assert_eq!(RiskClass::from_score(0.30), RiskClass::Experimental);
    /// ```
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
    
    /// Check if a score meets the requirements for this risk class
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::RiskClass;
    /// 
    /// assert!(RiskClass::Standard.meets_requirement(0.65));
    /// assert!(!RiskClass::Standard.meets_requirement(0.55));
    /// ```
    pub fn meets_requirement(&self, score: f64) -> bool {
        score >= self.min_score()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// 
    /// # Examples
    /// ```rust
    /// use qualia_core::QualityGrade;
    /// 
    /// assert_eq!(QualityGrade::from_score(0.95), QualityGrade::APlus);
    /// assert_eq!(QualityGrade::from_score(0.75), QualityGrade::B);
    /// assert_eq!(QualityGrade::from_score(0.50), QualityGrade::F);
    /// ```
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
    
    /// Get numeric value for comparison
    pub fn numeric_value(&self) -> u8 {
        match self {
            QualityGrade::APlus => 5,
            QualityGrade::A => 4,
            QualityGrade::B => 3,
            QualityGrade::C => 2,
            QualityGrade::F => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_score_bounds() {
        assert!(QualityScore::new(0.0).is_ok());
        assert!(QualityScore::new(1.0).is_ok());
        assert!(QualityScore::new(0.5).is_ok());
        assert!(QualityScore::new(-0.1).is_err());
        assert!(QualityScore::new(1.1).is_err());
    }
    
    #[test]
    fn test_quality_score_clamp() {
        assert_eq!(QualityScore::from_clamped(1.5).value(), 1.0);
        assert_eq!(QualityScore::from_clamped(-0.5).value(), 0.0);
        assert_eq!(QualityScore::from_clamped(0.5).value(), 0.5);
    }
    
    #[test]
    fn test_quality_score_operations() {
        let a = QualityScore::new(0.6).unwrap();
        let b = QualityScore::new(0.3).unwrap();
        
        let sum = a + b;
        assert_eq!(sum.value(), 0.9);
        
        let diff = a - b;
        assert_eq!(diff.value(), 0.3);
        
        let scaled = a * 0.5;
        assert_eq!(scaled.value(), 0.3);
    }
    
    #[test]
    fn test_risk_class_from_score() {
        assert_eq!(RiskClass::from_score(0.96), RiskClass::Aerospace);
        assert_eq!(RiskClass::from_score(0.92), RiskClass::Medical);
        assert_eq!(RiskClass::from_score(0.85), RiskClass::Financial);
        assert_eq!(RiskClass::from_score(0.75), RiskClass::Enterprise);
        assert_eq!(RiskClass::from_score(0.65), RiskClass::Standard);
        assert_eq!(RiskClass::from_score(0.45), RiskClass::Prototype);
        assert_eq!(RiskClass::from_score(0.20), RiskClass::Experimental);
    }
    
    #[test]
    fn test_risk_class_meets_requirement() {
        assert!(RiskClass::Standard.meets_requirement(0.65));
        assert!(!RiskClass::Standard.meets_requirement(0.55));
        assert!(RiskClass::Aerospace.meets_requirement(0.96));
        assert!(!RiskClass::Aerospace.meets_requirement(0.94));
    }
    
    #[test]
    fn test_quality_grade() {
        assert_eq!(QualityGrade::from_score(0.95), QualityGrade::APlus);
        assert_eq!(QualityGrade::from_score(0.85), QualityGrade::A);
        assert_eq!(QualityGrade::from_score(0.75), QualityGrade::B);
        assert_eq!(QualityGrade::from_score(0.65), QualityGrade::C);
        assert_eq!(QualityGrade::from_score(0.50), QualityGrade::F);
        
        assert_eq!(QualityGrade::APlus.as_str(), "A+");
        assert!(QualityGrade::APlus.numeric_value() > QualityGrade::F.numeric_value());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn prop_quality_score_bounded(value in -10.0f64..=10.0f64) {
            let clamped = QualityScore::from_clamped(value);
            prop_assert!((0.0..=1.0).contains(&clamped.value()));
        }
        
        #[test]
        fn prop_risk_class_consistency(score in 0.0f64..=1.0f64) {
            let risk_class = RiskClass::from_score(score);
            prop_assert!(risk_class.meets_requirement(score));
        }
        
        #[test]
        fn prop_grade_ordering(score1 in 0.0f64..=1.0f64, score2 in 0.0f64..=1.0f64) {
            let grade1 = QualityGrade::from_score(score1);
            let grade2 = QualityGrade::from_score(score2);
            
            if score1 > score2 {
                prop_assert!(grade1.numeric_value() >= grade2.numeric_value());
            }
        }
    }
}

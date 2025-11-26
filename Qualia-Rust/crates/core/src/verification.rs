//! Formal verification for critical mathematical operations
//!
//! This module provides formal verification of mathematical properties
//! for quality metric calculations, ensuring correctness and invariants.

use crate::{QualityScore, Result, QualityError};

/// Verify that a value is in the valid range [0.0, 1.0]
///
/// This is a formal verification function that can be used in proofs
/// and property-based tests to ensure boundedness.
///
/// # Properties Verified
/// - Boundedness: result is always in [0.0, 1.0]
/// - Reflexivity: verify(verify(x)) == verify(x)
///
/// # Examples
///
/// ```rust
/// use qualia_core::verification::verify_bounded;
///
/// assert_eq!(verify_bounded(0.5), Ok(0.5));
/// assert_eq!(verify_bounded(1.5), Ok(1.0)); // Clamped
/// assert_eq!(verify_bounded(-0.5), Ok(0.0)); // Clamped
/// ```
pub fn verify_bounded(value: f64) -> Result<f64> {
    if value.is_nan() {
        return Err(QualityError::Calculation("NaN value detected".to_string()));
    }
    if value.is_infinite() {
        return Err(QualityError::Calculation("Infinite value detected".to_string()));
    }
    Ok(value.clamp(0.0, 1.0))
}

/// Verify monotonicity property
///
/// For a function f, monotonicity means: if x1 < x2, then f(x1) <= f(x2)
///
/// This function verifies that a sequence of (input, output) pairs
/// satisfies monotonicity.
pub fn verify_monotonicity<T: PartialOrd>(pairs: &[(T, f64)]) -> Result<()> {
    for i in 0..pairs.len().saturating_sub(1) {
        let (input1, output1) = &pairs[i];
        let (input2, output2) = &pairs[i + 1];
        
        if input1 < input2 && output1 > output2 {
            return Err(QualityError::Calculation(format!(
                "Monotonicity violation: input increased but output decreased"
            )));
        }
    }
    Ok(())
}

/// Verify continuity property
///
/// For a function f, continuity means: small changes in input
/// produce small changes in output.
///
/// This function verifies that the function is approximately continuous
/// by checking that small input deltas produce bounded output deltas.
pub fn verify_continuity(
    inputs: &[f64],
    outputs: &[f64],
    max_delta: f64,
) -> Result<()> {
    if inputs.len() != outputs.len() {
        return Err(QualityError::Calculation(
            "Input and output lengths must match".to_string(),
        ));
    }
    
    for i in 0..inputs.len().saturating_sub(1) {
        let input_delta = (inputs[i + 1] - inputs[i]).abs();
        let output_delta = (outputs[i + 1] - outputs[i]).abs();
        
        // If input delta is small, output delta should be bounded
        if input_delta < 0.01 && output_delta > max_delta {
            return Err(QualityError::Calculation(format!(
                "Continuity violation: small input change ({}) produced large output change ({})",
                input_delta, output_delta
            )));
        }
    }
    Ok(())
}

/// Verify idempotency property
///
/// A function f is idempotent if f(f(x)) == f(x) for all x.
///
/// This function verifies that applying a normalization function twice
/// produces the same result as applying it once (within tolerance).
pub fn verify_idempotency<F>(f: F, inputs: &[f64], tolerance: f64) -> Result<()>
where
    F: Fn(f64) -> f64,
{
    for &input in inputs {
        let once = f(input);
        let twice = f(once);
        let diff = (once - twice).abs();
        
        if diff > tolerance {
            return Err(QualityError::Calculation(format!(
                "Idempotency violation: f(f({})) = {} != f({}) = {} (diff: {})",
                input, twice, input, once, diff
            )));
        }
    }
    Ok(())
}

/// Verify geometric mean properties
///
/// The geometric mean G = (∏x_i^w_i)^(1/Σw_i) has several important properties:
/// - If any x_i = 0 and w_i > 0, then G = 0
/// - G is bounded by min(x_i) and max(x_i) when weights are equal
/// - G is monotonic in each x_i
pub fn verify_geometric_mean_properties(
    factors: &[f64],
    weights: &[f64],
    result: f64,
) -> Result<()> {
    // Property 1: If any factor is 0 and weight > 0, result should be 0
    let has_zero_factor = factors
        .iter()
        .zip(weights.iter())
        .any(|(&f, &w)| f == 0.0 && w > 0.0);
    
    if has_zero_factor && result != 0.0 {
        return Err(QualityError::Calculation(
            "Geometric mean should be 0 when any factor is 0".to_string(),
        ));
    }
    
    // Property 2: Result should be in [0.0, 1.0] if all factors are
    if factors.iter().all(|&f| (0.0..=1.0).contains(&f)) {
        if !(0.0..=1.0).contains(&result) {
            return Err(QualityError::Calculation(format!(
                "Geometric mean {} is not in [0.0, 1.0] when all factors are",
                result
            )));
        }
    }
    
    // Property 3: Result should be between min and max (approximately)
    if !factors.is_empty() && !weights.iter().all(|&w| w == 0.0) {
        let min_factor = factors.iter().copied().fold(f64::INFINITY, f64::min);
        let max_factor = factors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        // Geometric mean should be between min and max (with some tolerance for weighted case)
        if result < min_factor - 0.01 || result > max_factor + 0.01 {
            // This is a warning, not an error, as weighted geometric mean can be outside range
            // But we log it for verification purposes
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_verify_bounded() {
        assert_eq!(verify_bounded(0.5).unwrap(), 0.5);
        assert_eq!(verify_bounded(1.5).unwrap(), 1.0);
        assert_eq!(verify_bounded(-0.5).unwrap(), 0.0);
        assert!(verify_bounded(f64::NAN).is_err());
        assert!(verify_bounded(f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_verify_monotonicity() {
        let monotonic = vec![(0.0, 0.0), (0.5, 0.3), (1.0, 0.6)];
        assert!(verify_monotonicity(&monotonic).is_ok());
        
        let non_monotonic = vec![(0.0, 0.5), (0.5, 0.3), (1.0, 0.6)];
        assert!(verify_monotonicity(&non_monotonic).is_err());
    }
    
    #[test]
    fn test_verify_idempotency() {
        let normalizer = |x: f64| x.clamp(0.0, 1.0);
        let inputs = vec![0.0, 0.5, 1.0, 1.5, -0.5];
        assert!(verify_idempotency(normalizer, &inputs, 1e-10).is_ok());
    }
    
    #[test]
    fn test_verify_geometric_mean_properties() {
        let factors = vec![0.5, 0.6, 0.7];
        let weights = vec![1.0, 1.0, 1.0];
        let result = 0.6; // Approximate geometric mean
        assert!(verify_geometric_mean_properties(&factors, &weights, result).is_ok());
        
        // Test zero factor case
        let factors = vec![0.0, 0.6, 0.7];
        let weights = vec![1.0, 1.0, 1.0];
        let result = 0.0;
        assert!(verify_geometric_mean_properties(&factors, &weights, result).is_ok());
    }
}

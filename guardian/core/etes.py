"""
E-TES: Evolutionary Test Effectiveness Score v2.0

Core implementation of the evolutionary test effectiveness scoring system
with adaptive mutation testing and multi-objective optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
# BETESWeights will be defined in this file now.
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETESGrade(Enum):
    """E-TES Grade classifications"""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    F = "F"


@dataclass
class ETESComponents:
    """Container for E-TES score components"""
    mutation_score: float = 0.0
    evolution_gain: float = 1.0
    assertion_iq: float = 0.0
    behavior_coverage: float = 0.0
    speed_factor: float = 0.0
    quality_factor: float = 0.0
    
    # Metadata
    calculation_time: float = 0.0
    components_calculated: Dict[str, bool] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)

@dataclass
class BETESWeights: # Moved from betes.py
    """Weights for the bE-TES components."""
    w_m: float = 1.0  # Weight for Mutation Score
    w_e: float = 1.0  # Weight for EMT Gain
    w_a: float = 1.0  # Weight for Assertion IQ
    w_b: float = 1.0  # Weight for Behaviour Coverage
    w_s: float = 1.0  # Weight for Speed Factor

@dataclass
class BETESSettingsV31:
    """Configuration for bE-TES v3.1 specific features."""
    smooth_m: bool = False  # Enable sigmoid normalization for M'
    smooth_e: bool = False  # Enable sigmoid normalization for E'
    k_m: float = 14.0       # Steepness for M' sigmoid
    k_e: float = 12.0       # Steepness for E' sigmoid

@dataclass
class OSQIWeightsConfig:
    """Configuration for OSQI v1.0 component weights."""
    w_test: float = 2.0
    w_code: float = 1.0
    w_sec: float = 1.5
    w_arch: float = 1.0

@dataclass
class QualityConfig: # Renamed from ETESConfig
    """Configuration for Quality Score calculation."""
    # General settings
    mode: Literal["etes_v2", "betes_v3", "betes_v3.1", "osqi_v1"] = "etes_v2"

    # bE-TES specific settings
    betes_weights: BETESWeights = field(default_factory=BETESWeights)
    betes_v3_1_settings: BETESSettingsV31 = field(default_factory=BETESSettingsV31)
    
    # OSQI v1.0 specific settings
    osqi_weights: OSQIWeightsConfig = field(default_factory=OSQIWeightsConfig)

    risk_class: Optional[str] = None # Used by both bE-TES and OSQI for classification
    
    # Paths for sensor inputs
    test_root_path: Optional[str] = "tests/"
    coverage_file_path: Optional[str] = "coverage.info"
    critical_behaviors_manifest_path: Optional[str] = None
    ci_platform: Optional[str] = None
    
    # E-TES v2.0 specific parameters (formerly ETESConfig fields)
    # Evolution parameters
    max_generations: int = 10
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    early_stop_threshold: float = 0.01
    early_stop_patience: int = 3
    
    # Quality thresholds (used by E-TES v2.0 insights)
    min_mutation_score: float = 0.80
    min_behavior_coverage: float = 0.90
    max_test_runtime_ms: float = 200.0 # E-TES v2.0 specific
    target_assertion_density: float = 3.0 # E-TES v2.0 specific
    
    # Weights for multi-objective optimization (E-TES v2.0 specific)
    # Renamed 'weights' to 'etes_v2_weights' to avoid conflict if 'weights' is used generally
    etes_v2_weights: Dict[str, float] = field(default_factory=lambda: {
        'mutation_score': 0.25,
        'evolution_gain': 0.15,
        'assertion_iq': 0.20,
        'behavior_coverage': 0.20,
        'speed_factor': 0.10,
        'quality_factor': 0.10
    })


class ETESCalculator:
    """
    Main E-TES v2.0 calculator with evolutionary capabilities
    
    Formula: E-TES = MS × EG × AIQ × BC × SF × QF
    Where:
    - MS: Mutation Score (weighted by severity)
    - EG: Evolution Gain (1 + improvement_rate)
    - AIQ: Assertion Intelligence Quotient
    - BC: Behavior Coverage (critical paths)
    - SF: Speed Factor (logarithmic)
    - QF: Quality Factor (stability × determinism)
    """
    
    def __init__(self, config: Optional[QualityConfig] = None): # Updated type hint
        self.config = config or QualityConfig() # Updated instantiation
        self.history = []
        self.baseline_score = None
        
    def calculate_etes(
        self,
        test_suite_data: Dict[str, Any],
        codebase_data: Dict[str, Any],
        previous_score: Optional[float] = None
    ) -> Tuple[float, ETESComponents]:
        """
        Calculate E-TES v2.0 score with all components
        
        Args:
            test_suite_data: Test suite metrics and data
            codebase_data: Codebase analysis data
            previous_score: Previous E-TES score for evolution gain
            
        Returns:
            Tuple of (etes_score, components)
        """
        start_time = time.time()
        components = ETESComponents()
        
        try:
            # Calculate individual components
            components.mutation_score = self._calculate_mutation_score(
                test_suite_data, codebase_data
            )
            components.evolution_gain = self._calculate_evolution_gain(
                previous_score, components.mutation_score
            )
            components.assertion_iq = self._calculate_assertion_iq(
                test_suite_data
            )
            components.behavior_coverage = self._calculate_behavior_coverage(
                test_suite_data, codebase_data
            )
            components.speed_factor = self._calculate_speed_factor(
                test_suite_data
            )
            components.quality_factor = self._calculate_quality_factor(
                test_suite_data
            )
            
            # Calculate final E-TES score
            etes_score = self._compute_final_score(components)
            
            # Generate insights
            components.insights = self._generate_insights(components)
            
            # Record calculation time
            components.calculation_time = time.time() - start_time
            
            # Update history
            self._update_history(etes_score, components)
            
            logger.debug(f"E-TES v2.0 calculated: {etes_score:.3f} in {components.calculation_time:.2f}s")
            
            return etes_score, components
            
        except Exception as e:
            logger.error(f"Error calculating E-TES: {e}")
            components.calculation_time = time.time() - start_time
            return 0.0, components
    
    def _calculate_mutation_score(
        self, 
        test_suite_data: Dict[str, Any], 
        codebase_data: Dict[str, Any]
    ) -> float:
        """Calculate weighted mutation score by severity"""
        try:
            # Get basic mutation score
            base_score = test_suite_data.get('mutation_score', 0.0)
            
            # Weight by mutant severity if available
            mutant_data = test_suite_data.get('mutants', [])
            if mutant_data:
                total_weight = sum(m.get('severity_weight', 1.0) for m in mutant_data)
                killed_weight = sum(
                    m.get('severity_weight', 1.0) 
                    for m in mutant_data 
                    if m.get('killed', False)
                )
                weighted_score = killed_weight / total_weight if total_weight > 0 else 0.0
                return min(weighted_score, 1.0)
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating mutation score: {e}")
            return 0.0
    
    def _calculate_evolution_gain(
        self, 
        previous_score: Optional[float], 
        current_mutation_score: float
    ) -> float:
        """Calculate evolution gain factor"""
        try:
            if previous_score is None or len(self.history) == 0:
                return 1.0  # No previous data
            
            # Calculate improvement rate
            if len(self.history) >= 2:
                recent_scores = [h['etes_score'] for h in self.history[-5:]]
                if len(recent_scores) >= 2:
                    improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                    return max(1.0, 1.0 + improvement_rate)
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating evolution gain: {e}")
            return 1.0
    
    def _calculate_assertion_iq(self, test_suite_data: Dict[str, Any]) -> float:
        """Calculate Assertion Intelligence Quotient"""
        try:
            assertions = test_suite_data.get('assertions', [])
            if not assertions:
                return 0.0
            
            total_score = 0.0
            assertion_weights = {
                'equality': 1.0,
                'inequality': 1.0,
                'type_check': 1.2,
                'exception': 1.5,
                'invariant': 2.0,
                'property': 1.8,
                'boundary': 1.6,
                'performance': 1.4
            }
            
            for assertion in assertions:
                assertion_type = assertion.get('type', 'equality')
                weight = assertion_weights.get(assertion_type, 1.0)
                
                # Bonus for invariant checking
                if assertion.get('checks_invariant', False):
                    weight *= 1.5
                
                # Penalty for redundancy
                if assertion.get('is_redundant', False):
                    weight *= 0.3
                
                # Weight by target criticality
                criticality = assertion.get('target_criticality', 1.0)
                total_score += weight * criticality
            
            # Normalize by number of assertions
            aiq_score = total_score / len(assertions)
            return min(aiq_score / 2.0, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            logger.warning(f"Error calculating assertion IQ: {e}")
            return 0.0
    
    def _calculate_behavior_coverage(
        self, 
        test_suite_data: Dict[str, Any], 
        codebase_data: Dict[str, Any]
    ) -> float:
        """Calculate user-centric behavior coverage"""
        try:
            covered_behaviors = set(test_suite_data.get('covered_behaviors', []))
            all_behaviors = set(codebase_data.get('all_behaviors', []))
            
            if not all_behaviors:
                return 0.0
            
            # Weight by behavior criticality
            behavior_criticality = codebase_data.get('behavior_criticality', {})
            
            weighted_coverage = sum(
                behavior_criticality.get(behavior, 1.0)
                for behavior in covered_behaviors
            )
            
            total_critical = sum(
                behavior_criticality.get(behavior, 1.0)
                for behavior in all_behaviors
            )
            
            return weighted_coverage / total_critical if total_critical > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating behavior coverage: {e}")
            return 0.0
    
    def _calculate_speed_factor(self, test_suite_data: Dict[str, Any]) -> float:
        """Calculate logarithmic speed factor"""
        try:
            avg_runtime_ms = test_suite_data.get('avg_test_execution_time_ms', 1000.0)
            
            # Logarithmic speed factor: 1 / (1 + log(1 + runtime_ms/100))
            speed_factor = 1.0 / (1.0 + np.log(1.0 + avg_runtime_ms / 100.0))
            return min(speed_factor, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating speed factor: {e}")
            return 0.0
    
    def _calculate_quality_factor(self, test_suite_data: Dict[str, Any]) -> float:
        """Calculate test reliability and maintainability factor"""
        try:
            # Get quality metrics
            determinism = test_suite_data.get('determinism_score', 1.0)
            stability = test_suite_data.get('stability_score', 1.0)
            clarity = test_suite_data.get('readability_score', 1.0)
            independence = test_suite_data.get('independence_score', 1.0)
            
            # Geometric mean for balanced quality
            quality_factor = (determinism * stability * clarity * independence) ** 0.25
            return min(quality_factor, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating quality factor: {e}")
            return 0.0
    
    def _compute_final_score(self, components: ETESComponents) -> float:
        """Compute final E-TES score using the core formula"""
        try:
            # Core E-TES formula: MS × EG × AIQ × BC × SF × QF
            etes_score = (
                components.mutation_score *
                components.evolution_gain *
                components.assertion_iq *
                components.behavior_coverage *
                components.speed_factor *
                components.quality_factor
            )
            
            return min(etes_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error computing final E-TES score: {e}")
            return 0.0
    
    def _generate_insights(self, components: ETESComponents) -> List[str]:
        """Generate actionable insights based on component scores"""
        insights = []
        
        if components.mutation_score < self.config.min_mutation_score:
            insights.append(f"Mutation score ({components.mutation_score:.2f}) below target ({self.config.min_mutation_score})")
        
        if components.behavior_coverage < self.config.min_behavior_coverage:
            insights.append(f"Behavior coverage ({components.behavior_coverage:.2f}) below target ({self.config.min_behavior_coverage})")
        
        if components.assertion_iq < 0.7:
            insights.append("Consider adding more intelligent assertions (invariants, properties)")
        
        if components.speed_factor < 0.8:
            insights.append("Test execution speed needs optimization")
        
        if components.quality_factor < 0.8:
            insights.append("Test quality issues detected (determinism, stability, clarity)")
        
        return insights
    
    def _update_history(self, etes_score: float, components: ETESComponents):
        """Update calculation history"""
        self.history.append({
            'timestamp': time.time(),
            'etes_score': etes_score,
            'components': components,
        })
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_grade(self, etes_score: float) -> ETESGrade:
        """Get letter grade for E-TES score"""
        if etes_score >= 0.9:
            return ETESGrade.A_PLUS
        elif etes_score >= 0.8:
            return ETESGrade.A
        elif etes_score >= 0.7:
            return ETESGrade.B
        elif etes_score >= 0.6:
            return ETESGrade.C
        else:
            return ETESGrade.F
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export E-TES metrics for external analysis"""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        return {
            'etes_score': latest['etes_score'],
            'grade': self.get_grade(latest['etes_score']).value,
            'components': {
                'mutation_score': latest['components'].mutation_score,
                'evolution_gain': latest['components'].evolution_gain,
                'assertion_iq': latest['components'].assertion_iq,
                'behavior_coverage': latest['components'].behavior_coverage,
                'speed_factor': latest['components'].speed_factor,
                'quality_factor': latest['components'].quality_factor,
            },
            'insights': latest['components'].insights,
            'calculation_time': latest['components'].calculation_time,
            'timestamp': latest['timestamp']
        }

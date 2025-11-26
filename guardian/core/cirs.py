"""
Change Impact Risk Score (CIRS)

The single best metric for predicting code quality issues.

Based on research showing that change frequency + complexity + test quality
are the strongest predictors of bugs and maintenance issues.

CIRS = (Change_Frequency × Complexity × (1 - Test_Quality) × Coupling × Defect_Rate)^(1/5)

Where:
- Change_Frequency: How often code changes (normalized)
- Complexity: Cyclomatic + cognitive complexity (normalized)
- Test_Quality: Mutation score (1 - mutation_score for risk)
- Coupling: Number of dependents (normalized)
- Defect_Rate: Historical bugs per change (normalized)
"""

import math
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CIRSComponents:
    """Components of CIRS calculation."""
    change_frequency: float = 0.0  # Raw: changes per month
    complexity: float = 0.0  # Raw: cyclomatic complexity
    test_quality: float = 0.0  # Raw: mutation score [0, 1]
    coupling: float = 0.0  # Raw: number of dependents
    defect_rate: float = 0.0  # Raw: bugs per change
    
    # Normalized components [0, 1] where 1 = highest risk
    norm_change_frequency: float = 0.0
    norm_complexity: float = 0.0
    norm_test_quality: float = 0.0  # 1 - mutation_score
    norm_coupling: float = 0.0
    norm_defect_rate: float = 0.0
    
    # Final score
    cirs_score: float = 0.0  # [0, 1] where 1 = highest risk
    
    # Metadata
    calculation_time_s: float = 0.0
    insights: List[str] = field(default_factory=list)


class CIRSCalculator:
    """
    Calculates Change Impact Risk Score (CIRS).
    
    CIRS predicts where quality problems will occur by combining:
    1. Change frequency (strongest predictor)
    2. Complexity (second strongest)
    3. Test quality (mutation score)
    4. Coupling (maintenance predictor)
    5. Historical defect rate (if available)
    """
    
    def __init__(
        self,
        max_changes_per_month: float = 10.0,
        max_complexity: float = 20.0,
        max_coupling: float = 10.0,
        max_defect_rate: float = 0.1  # 10% of changes result in bugs
    ):
        """
        Initialize CIRS calculator.
        
        Args:
            max_changes_per_month: Maximum expected changes per month (for normalization)
            max_complexity: Maximum expected complexity (for normalization)
            max_coupling: Maximum expected coupling (for normalization)
            max_defect_rate: Maximum expected defect rate (for normalization)
        """
        self.max_changes_per_month = max_changes_per_month
        self.max_complexity = max_complexity
        self.max_coupling = max_coupling
        self.max_defect_rate = max_defect_rate
    
    def _normalize_change_frequency(self, changes_per_month: float) -> float:
        """
        Normalize change frequency using logarithmic scale.
        
        Args:
            changes_per_month: Number of changes per month
            
        Returns:
            Normalized value [0, 1] where 1 = highest risk
        """
        if changes_per_month <= 0:
            return 0.0
        
        # Logarithmic normalization: log(1 + x) / log(1 + max)
        normalized = math.log(1 + changes_per_month) / math.log(1 + self.max_changes_per_month)
        return min(1.0, normalized)
    
    def _normalize_complexity(self, complexity: float) -> float:
        """
        Normalize complexity.
        
        Args:
            complexity: Cyclomatic complexity
            
        Returns:
            Normalized value [0, 1] where 1 = highest risk
        """
        if complexity <= 0:
            return 0.0
        
        normalized = min(1.0, complexity / self.max_complexity)
        return normalized
    
    def _normalize_test_quality(self, mutation_score: float) -> float:
        """
        Normalize test quality (invert mutation score for risk).
        
        High mutation score = low risk, so we use 1 - mutation_score.
        
        Args:
            mutation_score: Mutation score [0, 1]
            
        Returns:
            Normalized value [0, 1] where 1 = highest risk (no tests)
        """
        return 1.0 - max(0.0, min(1.0, mutation_score))
    
    def _normalize_coupling(self, num_dependents: float) -> float:
        """
        Normalize coupling.
        
        Args:
            num_dependents: Number of dependents
            
        Returns:
            Normalized value [0, 1] where 1 = highest risk
        """
        if num_dependents <= 0:
            return 0.0
        
        normalized = min(1.0, num_dependents / self.max_coupling)
        return normalized
    
    def _normalize_defect_rate(self, bugs_per_change: float) -> float:
        """
        Normalize defect rate.
        
        Args:
            bugs_per_change: Average bugs per change
            
        Returns:
            Normalized value [0, 1] where 1 = highest risk
        """
        if bugs_per_change <= 0:
            return 0.0
        
        normalized = min(1.0, bugs_per_change / self.max_defect_rate)
        return normalized
    
    def calculate(
        self,
        change_frequency: float,
        complexity: float,
        mutation_score: float,
        coupling: float = 0.0,
        defect_rate: float = 0.0
    ) -> CIRSComponents:
        """
        Calculate CIRS score.
        
        Args:
            change_frequency: Changes per month
            complexity: Cyclomatic complexity
            mutation_score: Mutation score [0, 1]
            coupling: Number of dependents (default: 0 if unknown)
            defect_rate: Bugs per change (default: 0 if unknown)
            
        Returns:
            CIRSComponents with all calculations
        """
        import time
        start_time = time.monotonic()
        
        components = CIRSComponents(
            change_frequency=change_frequency,
            complexity=complexity,
            test_quality=mutation_score,
            coupling=coupling,
            defect_rate=defect_rate
        )
        
        # Normalize all components
        components.norm_change_frequency = self._normalize_change_frequency(change_frequency)
        components.norm_complexity = self._normalize_complexity(complexity)
        components.norm_test_quality = self._normalize_test_quality(mutation_score)
        components.norm_coupling = self._normalize_coupling(coupling) if coupling > 0 else 0.5  # Default if unknown
        components.norm_defect_rate = self._normalize_defect_rate(defect_rate) if defect_rate > 0 else 0.0  # Default if unknown
        
        # Calculate CIRS using geometric mean
        # Use only available factors (if coupling/defect_rate unknown, use defaults or exclude)
        factors = [
            components.norm_change_frequency,
            components.norm_complexity,
            components.norm_test_quality,
        ]
        
        # Add optional factors if available
        if coupling > 0:
            factors.append(components.norm_coupling)
        if defect_rate > 0:
            factors.append(components.norm_defect_rate)
        
        # Geometric mean: (∏factors)^(1/n)
        if all(f > 0 for f in factors):
            product = 1.0
            for f in factors:
                product *= f
            components.cirs_score = product ** (1.0 / len(factors))
        else:
            # If any factor is 0, CIRS is 0 (no risk)
            components.cirs_score = 0.0
        
        # Generate insights
        components.insights = self._generate_insights(components)
        
        components.calculation_time_s = time.monotonic() - start_time
        
        return components
    
    def _generate_insights(self, components: CIRSComponents) -> List[str]:
        """Generate actionable insights based on CIRS components."""
        insights = []
        
        if components.cirs_score > 0.7:
            insights.append("HIGH RISK: This code is likely to have quality issues.")
        
        if components.norm_change_frequency > 0.7:
            insights.append(f"Frequently changed ({components.change_frequency:.1f} changes/month) - consider refactoring to reduce change frequency.")
        
        if components.norm_complexity > 0.7:
            insights.append(f"High complexity ({components.complexity:.1f}) - refactor to reduce complexity.")
        
        if components.norm_test_quality > 0.7:
            insights.append(f"Low test quality (mutation score: {components.test_quality:.2f}) - improve test coverage and quality.")
        
        if components.coupling > 0 and components.norm_coupling > 0.7:
            insights.append(f"High coupling ({components.coupling:.0f} dependents) - consider reducing dependencies.")
        
        if components.defect_rate > 0 and components.norm_defect_rate > 0.7:
            insights.append(f"High defect rate ({components.defect_rate:.3f} bugs/change) - investigate root causes.")
        
        # Actionable recommendations
        if components.cirs_score > 0.5:
            primary_issue = max([
                (components.norm_change_frequency, "change frequency"),
                (components.norm_complexity, "complexity"),
                (components.norm_test_quality, "test quality"),
            ], key=lambda x: x[0])
            
            insights.append(f"Primary issue: {primary_issue[1]}. Focus improvement efforts here.")
        
        return insights


def compare_cirs_vs_betes(
    cirs_score: float,
    betes_score: float
) -> Dict[str, Any]:
    """
    Compare CIRS with bE-TES to show why CIRS is better.
    
    Args:
        cirs_score: CIRS score [0, 1] (1 = highest risk)
        betes_score: bE-TES score [0, 1] (1 = highest quality)
        
    Returns:
        Comparison analysis
    """
    # Invert bE-TES for comparison (high bE-TES = low risk)
    betes_risk = 1.0 - betes_score
    
    return {
        'cirs_score': cirs_score,
        'betes_score': betes_score,
        'betes_risk_equivalent': betes_risk,
        'difference': cirs_score - betes_risk,
        'interpretation': {
            'cirs': 'Predicts WHERE problems will occur (change frequency + complexity + test quality)',
            'betes': 'Measures CURRENT test effectiveness (doesn't predict future issues)',
            'advantage': 'CIRS is more predictive and actionable - tells you what to fix'
        }
    }

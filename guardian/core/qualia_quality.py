"""
Qualia Quality Improvement System

Unified interface for generating the highest quality code possible.

Combines:
- CQS (Code Quality Score) measurement
- Enhanced CQS with semantic analysis
- Evolutionary algorithms for code improvement
- Automated code generation
- Pattern-based learning

This is Qualia's core quality improvement engine.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .cqs import CQSCalculator, CQSComponents
from .cqs_enhanced import EnhancedCQSCalculator, EnhancedCQSComponents
from .cqs_evolutionary import EvolutionaryCQS, CodeVariant
from .cqs_generator import QualityCodeGenerator, GeneratedCode

logger = logging.getLogger(__name__)


@dataclass
class QualityImprovementResult:
    """Result of quality improvement process."""
    original_code: str
    improved_code: str
    original_cqs: float
    improved_cqs: float
    improvement_percentage: float
    
    # Detailed metrics
    original_components: EnhancedCQSComponents
    improved_components: EnhancedCQSComponents
    
    # Improvement details
    improvements_applied: List[str]
    quality_tier_before: str
    quality_tier_after: str
    
    # Generation info
    generation_method: str
    iterations: int


class QualiaQualityEngine:
    """
    Qualia's quality improvement engine.
    
    Uses multiple algorithms to generate the highest quality code:
    1. Measure current quality
    2. Generate improvements (evolutionary, pattern-based, template-based)
    3. Evaluate improvements
    4. Select best variant
    5. Iterate until target quality reached
    """
    
    def __init__(
        self,
        target_quality: float = 0.9,
        use_evolutionary: bool = True,
        use_generation: bool = True,
        max_iterations: int = 5
    ):
        """
        Initialize Qualia quality engine.
        
        Args:
            target_quality: Target CQS score (default: 0.9)
            use_evolutionary: Use evolutionary algorithms
            use_generation: Use code generation
            max_iterations: Maximum improvement iterations
        """
        self.target_quality = target_quality
        self.use_evolutionary = use_evolutionary
        self.use_generation = use_generation
        self.max_iterations = max_iterations
        
        self.cqs_calc = CQSCalculator()
        self.enhanced_calc = EnhancedCQSCalculator()
        self.evolutionary = EvolutionaryCQS() if use_evolutionary else None
        self.generator = QualityCodeGenerator() if use_generation else None
    
    def improve_code(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> QualityImprovementResult:
        """
        Improve code to highest quality possible.
        
        Args:
            code: Original code
            file_path: Optional file path for context
            
        Returns:
            QualityImprovementResult with improved code
        """
        # Measure original quality
        original_enhanced = self.enhanced_calc.calculate_enhanced(code, file_path)
        original_cqs = original_enhanced.enhanced_cqs_score
        
        logger.info(f"Original CQS: {original_cqs:.3f} (tier: {original_enhanced.quality_tier})")
        
        # If already at target, return original
        if original_cqs >= self.target_quality:
            return QualityImprovementResult(
                original_code=code,
                improved_code=code,
                original_cqs=original_cqs,
                improved_cqs=original_cqs,
                improvement_percentage=0.0,
                original_components=original_enhanced,
                improved_components=original_enhanced,
                improvements_applied=[],
                quality_tier_before=original_enhanced.quality_tier,
                quality_tier_after=original_enhanced.quality_tier,
                generation_method="none",
                iterations=0
            )
        
        # Generate improvements
        best_code = code
        best_score = original_cqs
        improvements_applied = []
        iteration = 0
        method_used = "none"
        
        for iteration in range(self.max_iterations):
            candidates = []
            
            # Method 1: Evolutionary improvement
            if self.use_evolutionary and self.evolutionary:
                try:
                    best_variant, history = self.evolutionary.evolve_code(
                        best_code,
                        lambda c: self.enhanced_calc.calculate_enhanced(c, file_path),
                        self.target_quality
                    )
                    if best_variant.cqs_score > best_score:
                        candidates.append((best_variant.code, best_variant.cqs_score, "evolutionary"))
                except Exception as e:
                    logger.warning(f"Evolutionary improvement failed: {e}")
            
            # Method 2: Code generation
            if self.use_generation and self.generator:
                try:
                    generated = self.generator.generate_improved_code(best_code, self.target_quality)
                    for variant in generated[:3]:  # Top 3
                        if variant.quality_score > best_score:
                            candidates.append((variant.code, variant.quality_score, "generation"))
                except Exception as e:
                    logger.warning(f"Code generation failed: {e}")
            
            # Method 3: Direct refactoring based on suggestions
            try:
                current_enhanced = self.enhanced_calc.calculate_enhanced(best_code, file_path)
                refactored = self._apply_suggested_refactorings(best_code, current_enhanced)
                if refactored:
                    refactored_enhanced = self.enhanced_calc.calculate_enhanced(refactored, file_path)
                    if refactored_enhanced.enhanced_cqs_score > best_score:
                        candidates.append((refactored, refactored_enhanced.enhanced_cqs_score, "refactoring"))
            except Exception as e:
                logger.warning(f"Refactoring failed: {e}")
            
            # Select best candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: x[1])
                if best_candidate[1] > best_score:
                    best_code = best_candidate[0]
                    best_score = best_candidate[1]
                    method_used = best_candidate[2]
                    improvements_applied.append(f"Iteration {iteration + 1}: {method_used} (CQS: {best_score:.3f})")
                    logger.info(f"Iteration {iteration + 1}: Improved to CQS {best_score:.3f} using {method_used}")
                else:
                    break  # No improvement, stop
            else:
                break  # No candidates, stop
            
            # Check if target reached
            if best_score >= self.target_quality:
                logger.info(f"Target quality reached: {best_score:.3f}")
                break
        
        # Final measurement
        improved_enhanced = self.enhanced_calc.calculate_enhanced(best_code, file_path)
        improvement_pct = ((best_score - original_cqs) / original_cqs * 100) if original_cqs > 0 else 0.0
        
        return QualityImprovementResult(
            original_code=code,
            improved_code=best_code,
            original_cqs=original_cqs,
            improved_cqs=best_score,
            improvement_percentage=improvement_pct,
            original_components=original_enhanced,
            improved_components=improved_enhanced,
            improvements_applied=improvements_applied,
            quality_tier_before=original_enhanced.quality_tier,
            quality_tier_after=improved_enhanced.quality_tier,
            generation_method=method_used,
            iterations=iteration + 1
        )
    
    def _apply_suggested_refactorings(
        self,
        code: str,
        components: EnhancedCQSComponents
    ) -> Optional[str]:
        """Apply suggested refactorings from enhanced analysis."""
        # Apply top priority improvements
        improvements = components.generated_improvements[:2]  # Top 2
        
        if not improvements:
            return None
        
        # Simplified - would need full refactoring engine
        # For now, return None (let evolutionary/genetic handle it)
        return None


def improve_code_quality(
    code: str,
    target_quality: float = 0.9,
    file_path: Optional[str] = None
) -> QualityImprovementResult:
    """
    Improve code to highest quality possible.
    
    This is the main entry point for Qualia's quality improvement.
    
    Args:
        code: Source code to improve
        target_quality: Target CQS score (default: 0.9)
        file_path: Optional file path for context
        
    Returns:
        QualityImprovementResult with improved code and metrics
    """
    engine = QualiaQualityEngine(target_quality=target_quality)
    return engine.improve_code(code, file_path)

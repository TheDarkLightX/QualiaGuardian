"""
Final Qualia Quality Engine

Combines all improvements for maximum code quality:
- Enhanced evolutionary algorithms
- Improved automation
- Multi-strategy approach
- Performance optimizations
- Quality validation
"""

from typing import Optional
from .qualia_quality_improved import (
    ImprovedQualiaQualityEngine,
    improve_code_quality_improved
)
from .qualia_quality import QualityImprovementResult

# Re-export for convenience
__all__ = [
    'ImproveCodeQuality',
    'QualiaQualityEngine',
    'QualityImprovementResult'
]


def ImproveCodeQuality(
    code: str,
    target_quality: float = 0.9,
    file_path: Optional[str] = None,
    use_evolutionary: bool = True,
    use_refactoring: bool = True,
    use_generation: bool = True,
    max_iterations: int = 10
) -> QualityImprovementResult:
    """
    Improve code to highest quality possible.
    
    This is Qualia's main quality improvement function.
    Uses the most advanced algorithms to generate the highest quality code.
    
    Args:
        code: Source code to improve
        target_quality: Target CQS score (default: 0.9)
        file_path: Optional file path for context
        use_evolutionary: Use evolutionary algorithms (default: True)
        use_refactoring: Use refactoring engine (default: True)
        use_generation: Use code generation (default: True)
        max_iterations: Maximum improvement iterations (default: 10)
        
    Returns:
        QualityImprovementResult with improved code and metrics
        
    Example:
        >>> result = ImproveCodeQuality(bad_code, target_quality=0.9)
        >>> print(f"Quality: {result.original_cqs:.3f} → {result.improved_cqs:.3f}")
        >>> use_improved_code(result.improved_code)
    """
    engine = ImprovedQualiaQualityEngine(
        target_quality=target_quality,
        use_evolutionary=use_evolutionary,
        use_refactoring=use_refactoring,
        use_generation=use_generation,
        max_iterations=max_iterations
    )
    
    return engine.improve_code_improved(code, file_path)


# Alias for convenience
QualiaQualityEngine = ImprovedQualiaQualityEngine

"""
Self-Improvement Module for QualiaGuardian

Enables QualiaGuardian to analyze and improve itself recursively.
"""

from guardian.self_improvement.self_analyzer import (
    SelfAnalyzer,
    SelfAnalysisResult,
    ImprovementSuggestion,
    ImprovementPriority,
)
from guardian.self_improvement.recursive_improver import (
    RecursiveImprover,
    ImprovementAction,
    ImprovementStatus,
)
from guardian.self_improvement.auto_fixer import AutoFixer, FixResult

__all__ = [
    "SelfAnalyzer",
    "SelfAnalysisResult",
    "ImprovementSuggestion",
    "ImprovementPriority",
    "RecursiveImprover",
    "ImprovementAction",
    "ImprovementStatus",
    "AutoFixer",
    "FixResult",
]

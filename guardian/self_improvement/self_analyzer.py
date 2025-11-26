"""
Self-Analysis Module for QualiaGuardian

Enables QualiaGuardian to analyze and improve itself recursively.
This creates a meta-feedback loop where the tool improves its own quality.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from guardian.cli.analyzer import ProjectAnalyzer
from guardian.core.api import evaluate_subset
from guardian.core.tes import calculate_quality_score, get_etes_grade
from guardian.core.etes import QualityConfig
from guardian.history import HistoryManager
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig

logger = logging.getLogger(__name__)


class ImprovementPriority(Enum):
    """Priority levels for improvements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ImprovementSuggestion:
    """A suggestion for improving the codebase"""
    id: str
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    priority: ImprovementPriority
    category: str  # e.g., "complexity", "test_coverage", "documentation"
    estimated_impact: float  # 0.0 to 1.0
    estimated_effort: float  # 0.0 to 1.0
    suggested_fix: Optional[str] = None
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SelfAnalysisResult:
    """Results from self-analysis"""
    timestamp: datetime
    overall_score: float
    components: Dict[str, float]
    issues_found: List[ImprovementSuggestion]
    trends: Dict[str, Any]
    meta_metrics: Dict[str, Any]  # Metrics about the metrics system itself


class SelfAnalyzer:
    """
    Analyzes QualiaGuardian codebase and suggests improvements.
    
    This creates a recursive improvement loop where the tool improves itself.
    """
    
    def __init__(self, guardian_root: Optional[str] = None):
        """
        Initialize self-analyzer.
        
        Args:
            guardian_root: Root directory of QualiaGuardian (defaults to current)
        """
        if guardian_root is None:
            # Find guardian root by looking for guardian/ directory
            current = Path.cwd()
            while current != current.parent:
                if (current / "guardian").exists() and (current / "pyproject.toml").exists():
                    guardian_root = str(current)
                    break
                current = current.parent
            
            if guardian_root is None:
                guardian_root = str(Path.cwd())
        
        self.guardian_root = Path(guardian_root)
        self.history_manager = HistoryManager()
        self.formatter = OutputFormatter(FormattingConfig())
        
    def analyze_self(self, run_quality: bool = True) -> SelfAnalysisResult:
        """
        Perform comprehensive self-analysis.
        
        Args:
            run_quality: Whether to run quality scoring
            
        Returns:
            SelfAnalysisResult with analysis findings
        """
        logger.info("Starting self-analysis of QualiaGuardian...")
        
        # 1. Run standard analysis
        analyzer = ProjectAnalyzer()
        results = analyzer.analyze_project(
            project_path=str(self.guardian_root),
            test_path=str(self.guardian_root / "tests"),
        )
        
        # 2. Run quality scoring if requested
        quality_score = None
        quality_components = {}
        
        if run_quality:
            try:
                quality_score, quality_components = self._calculate_self_quality_score(results)
            except Exception as e:
                logger.warning(f"Could not calculate quality score: {e}")
        
        # 3. Identify improvement opportunities
        issues = self._identify_improvements(results)
        
        # 4. Calculate meta-metrics
        meta_metrics = self._calculate_meta_metrics(results, quality_score)
        
        # 5. Analyze trends
        trends = self._analyze_trends()
        
        result = SelfAnalysisResult(
            timestamp=datetime.now(),
            overall_score=quality_score or 0.0,
            components=quality_components,
            issues_found=issues,
            trends=trends,
            meta_metrics=meta_metrics,
        )
        
        # 6. Store results for trend analysis
        self._store_analysis_result(result)
        
        return result
    
    def _calculate_self_quality_score(
        self, results: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate quality score for QualiaGuardian itself."""
        # Use bE-TES v3.1 for self-analysis
        from guardian.core.etes import BETESWeights, BETESSettingsV31
        
        config = QualityConfig(
            mode="betes_v3.1",
            betes_weights=BETESWeights(),  # Use default weights
            betes_v3_1_settings=BETESSettingsV31(),  # Use default settings
        )
        
        # Collect metrics from self-analysis
        from guardian.sensors import mutation as mutation_sensor
        
        mutation_config = {
            "mutmut_paths_to_mutate": ["guardian"],
            "mutmut_runner_args": "pytest",
        }
        
        try:
            raw_ms, _, _ = mutation_sensor.get_mutation_score_data(
                config=mutation_config,
                project_path=str(self.guardian_root),
            )
        except Exception as e:
            logger.warning(f"Could not get mutation score: {e}")
            raw_ms = 0.0
        
        raw_metrics = {
            "raw_mutation_score": raw_ms,
            "raw_emt_gain": 0.0,  # Would need history
            "raw_assertion_iq": self._estimate_assertion_iq(results),
            "raw_behaviour_coverage": self._estimate_coverage(results),
            "raw_median_test_time_ms": results.get("test_execution_summary", {}).get(
                "pytest_duration_seconds", 0.0
            ) * 1000,
            "raw_flakiness_rate": 0.0,  # Would need CI data
        }
        
        score, components = calculate_quality_score(
            config=config,
            raw_metrics_betes=raw_metrics,
            test_suite_data=None,
            codebase_data=None,
            previous_score=None,
            project_path=str(self.guardian_root),
            project_language="python",
        )
        
        return score, components.__dict__ if hasattr(components, '__dict__') else {}
    
    def _estimate_assertion_iq(self, results: Dict[str, Any]) -> float:
        """Estimate assertion IQ from analysis results."""
        # Simple heuristic: if we have good test coverage, assume good assertions
        metrics = results.get("metrics", {})
        test_summary = results.get("test_execution_summary", {})
        
        if test_summary.get("pytest_ran_successfully"):
            # Base score of 3.0, increase if tests are comprehensive
            return 3.5
        return 2.0
    
    def _estimate_coverage(self, results: Dict[str, Any]) -> float:
        """Estimate behavior coverage."""
        # Check if we have comprehensive tests
        test_summary = results.get("test_execution_summary", {})
        if test_summary.get("pytest_ran_successfully"):
            # Assume decent coverage if tests pass
            return 0.7
        return 0.3
    
    def _identify_improvements(self, results: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """Identify specific improvement opportunities."""
        issues = []
        
        # Check complexity
        metrics = results.get("metrics", {})
        long_functions = results.get("details", {}).get("long_functions_list", [])
        
        for func in long_functions[:10]:  # Top 10
            issues.append(ImprovementSuggestion(
                id=f"complexity_{func.get('name', 'unknown')}",
                title=f"Long function: {func.get('name', 'unknown')}",
                description=f"Function has {func.get('lines', 0)} lines (threshold: 20)",
                file_path=func.get("file"),
                line_number=func.get("line_number"),
                priority=ImprovementPriority.MEDIUM,
                category="complexity",
                estimated_impact=0.3,
                estimated_effort=0.4,
                suggested_fix="Consider breaking this function into smaller, focused functions",
            ))
        
        # Check test coverage
        test_summary = results.get("test_execution_summary", {})
        if not test_summary.get("pytest_ran_successfully"):
            issues.append(ImprovementSuggestion(
                id="test_execution_failure",
                title="Test execution failed",
                description="Tests are not passing, which impacts quality metrics",
                file_path=None,
                line_number=None,
                priority=ImprovementPriority.CRITICAL,
                category="test_coverage",
                estimated_impact=0.9,
                estimated_effort=0.5,
                suggested_fix="Fix failing tests to enable accurate quality assessment",
            ))
        
        # Check circular dependencies
        circular_deps = metrics.get("circular_dependencies_count", 0)
        if circular_deps > 0:
            issues.append(ImprovementSuggestion(
                id="circular_dependencies",
                title=f"{circular_deps} circular dependencies found",
                description="Circular dependencies can make code harder to maintain",
                file_path=None,
                line_number=None,
                priority=ImprovementPriority.HIGH,
                category="architecture",
                estimated_impact=0.5,
                estimated_effort=0.6,
                suggested_fix="Refactor to break circular dependencies",
            ))
        
        # Check documentation
        # This is a heuristic - in reality, we'd parse docstrings
        issues.append(ImprovementSuggestion(
            id="documentation_coverage",
            title="Improve documentation coverage",
            description="Ensure all public APIs have comprehensive docstrings",
            file_path=None,
            line_number=None,
            priority=ImprovementPriority.MEDIUM,
            category="documentation",
            estimated_impact=0.4,
            estimated_effort=0.3,
            suggested_fix="Add docstrings to all public functions and classes",
        ))
        
        return issues
    
    def _calculate_meta_metrics(
        self, results: Dict[str, Any], quality_score: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate metrics about the metrics system itself."""
        metrics = results.get("metrics", {})
        
        return {
            "self_analysis_score": quality_score or 0.0,
            "codebase_size": metrics.get("total_lines_of_code_python", 0),
            "test_files_count": self._count_test_files(),
            "complexity_avg": metrics.get("average_cyclomatic_complexity", 0.0),
            "quality_tool_quality": quality_score or 0.0,  # Meta: quality of quality tool
            "self_improvement_capability": 1.0 if quality_score and quality_score > 0.7 else 0.5,
        }
    
    def _count_test_files(self) -> int:
        """Count test files in the codebase."""
        test_dir = self.guardian_root / "tests"
        if not test_dir.exists():
            return 0
        
        count = 0
        for path in test_dir.rglob("test_*.py"):
            if path.is_file():
                count += 1
        return count
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends over time."""
        # Get historical data
        try:
            # This would query the history database
            # For now, return placeholder
            return {
                "score_trend": "stable",
                "improvement_rate": 0.0,
                "issues_resolved": 0,
            }
        except Exception as e:
            logger.warning(f"Could not analyze trends: {e}")
            return {}
    
    def _store_analysis_result(self, result: SelfAnalysisResult) -> None:
        """Store analysis result for trend tracking."""
        try:
            # Store in a dedicated table or file
            storage_path = self.guardian_root / ".guardian" / "self_analysis_history.json"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing history
            history = []
            if storage_path.exists():
                with open(storage_path, 'r') as f:
                    history = json.load(f)
            
            # Add new result
            history.append({
                "timestamp": result.timestamp.isoformat(),
                "overall_score": result.overall_score,
                "components": result.components,
                "issues_count": len(result.issues_found),
                "meta_metrics": result.meta_metrics,
            })
            
            # Keep only last 100 analyses
            history = history[-100:]
            
            # Save
            with open(storage_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not store analysis result: {e}")
    
    def generate_improvement_plan(self, result: SelfAnalysisResult) -> Dict[str, Any]:
        """
        Generate an actionable improvement plan based on analysis.
        
        Args:
            result: SelfAnalysisResult from analyze_self()
            
        Returns:
            Dictionary with improvement plan
        """
        # Sort issues by priority and impact/effort ratio
        sorted_issues = sorted(
            result.issues_found,
            key=lambda x: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.priority.value],
                -x.estimated_impact / max(x.estimated_effort, 0.01)
            )
        )
        
        plan = {
            "current_score": result.overall_score,
            "target_score": min(0.95, result.overall_score + 0.1),
            "improvements": [
                {
                    "id": issue.id,
                    "title": issue.title,
                    "priority": issue.priority.value,
                    "impact": issue.estimated_impact,
                    "effort": issue.estimated_effort,
                    "roi": issue.estimated_impact / max(issue.estimated_effort, 0.01),
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in sorted_issues[:20]  # Top 20
            ],
            "estimated_time_to_target": self._estimate_time_to_target(
                result.overall_score,
                min(0.95, result.overall_score + 0.1),
                sorted_issues
            ),
        }
        
        return plan
    
    def _estimate_time_to_target(
        self, current: float, target: float, issues: List[ImprovementSuggestion]
    ) -> str:
        """Estimate time to reach target score."""
        if current >= target:
            return "Already at target"
        
        # Simple heuristic: assume each high-impact issue improves score by 0.01-0.05
        high_impact_issues = [i for i in issues if i.estimated_impact > 0.5]
        needed_improvement = target - current
        
        if not high_impact_issues:
            return "Unknown"
        
        # Rough estimate: 1-2 hours per high-impact issue
        estimated_hours = len(high_impact_issues) * 1.5
        return f"~{estimated_hours:.1f} hours"
    
    def display_self_analysis(self, result: SelfAnalysisResult) -> None:
        """Display self-analysis results in a beautiful format."""
        self.formatter.console.print("\n[bold cyan]🔍 QualiaGuardian Self-Analysis[/bold cyan]\n")
        
        # Overall score
        score_style = "green" if result.overall_score >= 0.8 else "yellow" if result.overall_score >= 0.6 else "red"
        self.formatter.console.print(
            f"[bold]Overall Quality Score:[/bold] [{score_style}]{result.overall_score:.3f}[/{score_style}]"
        )
        
        # Components
        if result.components:
            self.formatter.console.print("\n[bold]Components:[/bold]")
            for key, value in result.components.items():
                if isinstance(value, (int, float)):
                    self.formatter.console.print(f"  • {key}: {value:.3f}")
        
        # Issues
        self.formatter.console.print(f"\n[bold]Issues Found:[/bold] {len(result.issues_found)}")
        
        # Top issues by priority
        critical = [i for i in result.issues_found if i.priority == ImprovementPriority.CRITICAL]
        high = [i for i in result.issues_found if i.priority == ImprovementPriority.HIGH]
        
        if critical:
            self.formatter.console.print("\n[bold red]Critical Issues:[/bold red]")
            for issue in critical[:5]:
                self.formatter.console.print(f"  • {issue.title}")
        
        if high:
            self.formatter.console.print("\n[bold yellow]High Priority Issues:[/bold yellow]")
            for issue in high[:5]:
                self.formatter.console.print(f"  • {issue.title}")
        
        # Meta-metrics
        if result.meta_metrics:
            self.formatter.console.print("\n[bold]Meta-Metrics:[/bold]")
            for key, value in result.meta_metrics.items():
                self.formatter.console.print(f"  • {key}: {value}")
        
        self.formatter.console.print()

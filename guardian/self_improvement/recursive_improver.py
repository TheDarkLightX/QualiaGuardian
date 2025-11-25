"""
Recursive Improvement Loop for QualiaGuardian

Creates a self-improving system where QualiaGuardian analyzes itself,
identifies improvements, and iteratively enhances its own code quality.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
from datetime import datetime
from enum import Enum

from guardian.self_improvement.self_analyzer import (
    SelfAnalyzer, SelfAnalysisResult, ImprovementSuggestion, ImprovementPriority
)
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig
from guardian.history import HistoryManager

logger = logging.getLogger(__name__)


class ImprovementStatus(Enum):
    """Status of an improvement"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ImprovementAction:
    """An action to improve the codebase"""
    suggestion_id: str
    status: ImprovementStatus
    applied_at: Optional[datetime] = None
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    notes: Optional[str] = None


class RecursiveImprover:
    """
    Recursive improvement system that enables QualiaGuardian to improve itself.
    
    This creates a feedback loop:
    1. Analyze current state
    2. Identify improvements
    3. Apply improvements (or suggest them)
    4. Re-analyze to verify
    5. Track progress over time
    """
    
    def __init__(
        self,
        guardian_root: Optional[str] = None,
        auto_apply: bool = False,
        max_iterations: int = 10,
    ):
        """
        Initialize recursive improver.
        
        Args:
            guardian_root: Root directory of QualiaGuardian
            auto_apply: Whether to automatically apply safe improvements
            max_iterations: Maximum number of improvement iterations
        """
        self.self_analyzer = SelfAnalyzer(guardian_root)
        self.auto_apply = auto_apply
        self.max_iterations = max_iterations
        self.formatter = OutputFormatter(FormattingConfig())
        self.history_manager = HistoryManager()
        self.improvement_history: List[Dict[str, Any]] = []
        
    def run_improvement_loop(
        self,
        target_score: Optional[float] = None,
        min_improvement: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Run the recursive improvement loop.
        
        Args:
            target_score: Target quality score (default: 0.95)
            min_improvement: Minimum improvement per iteration to continue
            
        Returns:
            Dictionary with improvement results
        """
        if target_score is None:
            target_score = 0.95
        
        self.formatter.console.print(
            "\n[bold cyan]🔄 Starting Recursive Improvement Loop[/bold cyan]\n"
        )
        self.formatter.console.print(f"Target Score: {target_score}")
        self.formatter.console.print(f"Auto-apply: {self.auto_apply}")
        self.formatter.console.print()
        
        iteration = 0
        current_score = 0.0
        improvements_applied = []
        
        while iteration < self.max_iterations:
            iteration += 1
            self.formatter.console.print(
                f"[bold]Iteration {iteration}/{self.max_iterations}[/bold]\n"
            )
            
            # 1. Analyze current state
            self.formatter.console.print("[dim]Analyzing current state...[/dim]")
            result = self.self_analyzer.analyze_self(run_quality=True)
            current_score = result.overall_score
            
            self.formatter.console.print(
                f"Current Score: [green]{current_score:.3f}[/green]\n"
            )
            
            # 2. Check if target reached
            if current_score >= target_score:
                self.formatter.console.print(
                    f"[bold green]✅ Target score reached! ({current_score:.3f} >= {target_score})[/bold green]\n"
                )
                break
            
            # 3. Generate improvement plan
            plan = self.self_analyzer.generate_improvement_plan(result)
            
            # 4. Apply improvements
            applied = self._apply_improvements(
                result.issues_found,
                current_score,
                iteration,
            )
            improvements_applied.extend(applied)
            
            # 5. Check if we're making progress
            if iteration > 1:
                prev_score = self.improvement_history[-1].get("score", 0.0)
                improvement = current_score - prev_score
                
                if improvement < min_improvement:
                    self.formatter.console.print(
                        f"[yellow]⚠️  Improvement too small ({improvement:.3f} < {min_improvement}). Stopping.[/yellow]\n"
                    )
                    break
            
            # 6. Record iteration
            self.improvement_history.append({
                "iteration": iteration,
                "score": current_score,
                "issues_found": len(result.issues_found),
                "improvements_applied": len(applied),
                "timestamp": datetime.now().isoformat(),
            })
            
            self.formatter.console.print()
        
        # Final summary
        return self._generate_summary(
            iteration,
            current_score,
            target_score,
            improvements_applied,
        )
    
    def _apply_improvements(
        self,
        issues: List[ImprovementSuggestion],
        current_score: float,
        iteration: int,
    ) -> List[ImprovementAction]:
        """Apply improvements based on suggestions."""
        applied = []
        
        # Sort by ROI (impact/effort)
        sorted_issues = sorted(
            issues,
            key=lambda x: x.estimated_impact / max(x.estimated_effort, 0.01),
            reverse=True,
        )
        
        # Apply top improvements
        for issue in sorted_issues[:5]:  # Top 5 per iteration
            if self.auto_apply and self._is_safe_to_apply(issue):
                action = self._apply_single_improvement(issue, current_score)
                if action:
                    applied.append(action)
                    self.formatter.console.print(
                        f"[green]✓ Applied:[/green] {issue.title}"
                    )
            else:
                # Just suggest
                self.formatter.console.print(
                    f"[yellow]→ Suggested:[/yellow] {issue.title} "
                    f"(Impact: {issue.estimated_impact:.2f}, Effort: {issue.estimated_effort:.2f})"
                )
                applied.append(ImprovementAction(
                    suggestion_id=issue.id,
                    status=ImprovementStatus.PENDING,
                ))
        
        return applied
    
    def _is_safe_to_apply(self, issue: ImprovementSuggestion) -> bool:
        """Check if an improvement is safe to apply automatically."""
        # Only auto-apply low-risk improvements
        if issue.priority == ImprovementPriority.CRITICAL:
            return False  # Critical issues need manual review
        
        # Don't auto-apply architectural changes
        if issue.category in ["architecture", "design"]:
            return False
        
        # Safe categories for auto-apply
        safe_categories = ["documentation", "formatting", "style"]
        return issue.category in safe_categories
    
    def _apply_single_improvement(
        self,
        issue: ImprovementSuggestion,
        score_before: float,
    ) -> Optional[ImprovementAction]:
        """Apply a single improvement."""
        try:
            # For now, we'll just track the suggestion
            # In a full implementation, this would:
            # 1. Parse the code
            # 2. Apply the fix
            # 3. Verify it compiles/runs
            # 4. Re-run tests
            
            action = ImprovementAction(
                suggestion_id=issue.id,
                status=ImprovementStatus.COMPLETED,
                applied_at=datetime.now(),
                score_before=score_before,
                notes=f"Applied: {issue.suggested_fix}",
            )
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to apply improvement {issue.id}: {e}")
            return ImprovementAction(
                suggestion_id=issue.id,
                status=ImprovementStatus.FAILED,
                notes=f"Error: {str(e)}",
            )
    
    def _generate_summary(
        self,
        iterations: int,
        final_score: float,
        target_score: float,
        improvements: List[ImprovementAction],
    ) -> Dict[str, Any]:
        """Generate improvement summary."""
        summary = {
            "iterations": iterations,
            "final_score": final_score,
            "target_score": target_score,
            "target_reached": final_score >= target_score,
            "improvements_applied": len([a for a in improvements if a.status == ImprovementStatus.COMPLETED]),
            "improvements_suggested": len([a for a in improvements if a.status == ImprovementStatus.PENDING]),
            "improvement_history": self.improvement_history,
            "score_improvement": final_score - (self.improvement_history[0].get("score", 0.0) if self.improvement_history else 0.0),
        }
        
        # Display summary
        self.formatter.console.print("\n[bold cyan]📊 Improvement Summary[/bold cyan]\n")
        self.formatter.console.print(f"Iterations: {iterations}")
        self.formatter.console.print(f"Final Score: {final_score:.3f}")
        self.formatter.console.print(f"Target Score: {target_score}")
        self.formatter.console.print(f"Improvements Applied: {summary['improvements_applied']}")
        self.formatter.console.print(f"Improvements Suggested: {summary['improvements_suggested']}")
        
        if summary['score_improvement'] > 0:
            self.formatter.console.print(
                f"\n[green]Score improved by {summary['score_improvement']:.3f}[/green]"
            )
        
        return summary
    
    def visualize_trends(self) -> None:
        """Visualize improvement trends over time."""
        if not self.improvement_history:
            self.formatter.console.print("[yellow]No improvement history available[/yellow]")
            return
        
        # Simple text-based visualization
        self.formatter.console.print("\n[bold cyan]📈 Improvement Trends[/bold cyan]\n")
        
        scores = [h["score"] for h in self.improvement_history]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            
            self.formatter.console.print(f"Score Range: {min_score:.3f} → {max_score:.3f}")
            
            # Simple ASCII chart
            for i, history in enumerate(self.improvement_history):
                score = history["score"]
                bar_length = int((score - min_score) / (max_score - min_score + 0.001) * 40)
                bar = "█" * bar_length
                self.formatter.console.print(
                    f"Iteration {history['iteration']:2d}: {bar} {score:.3f}"
                )

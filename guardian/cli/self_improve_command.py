"""
CLI command for self-improvement functionality

Enables QualiaGuardian to analyze and improve itself recursively.
"""

import typer
from typing import Optional
from pathlib import Path

from guardian.self_improvement.self_analyzer import SelfAnalyzer
from guardian.self_improvement.recursive_improver import RecursiveImprover
from guardian.self_improvement.innovation_detector import InnovationDetector
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig

app = typer.Typer(
    name="self-improve",
    help="Self-improvement commands - Enable QualiaGuardian to improve itself"
)

formatter = OutputFormatter(FormattingConfig())


@app.command("analyze")
def self_analyze(
    guardian_root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Root directory of QualiaGuardian (auto-detected if not specified)",
    ),
    run_quality: bool = typer.Option(
        True,
        "--quality/--no-quality",
        help="Run quality scoring",
    ),
    show_plan: bool = typer.Option(
        True,
        "--plan/--no-plan",
        help="Show improvement plan",
    ),
):
    """
    Analyze QualiaGuardian codebase and identify improvement opportunities.
    
    This enables QualiaGuardian to analyze itself and suggest improvements.
    """
    analyzer = SelfAnalyzer(str(guardian_root) if guardian_root else None)
    
    formatter.console.print("\n[bold cyan]🔍 Analyzing QualiaGuardian...[/bold cyan]\n")
    
    result = analyzer.analyze_self(run_quality=run_quality)
    
    # Display results
    analyzer.display_self_analysis(result)
    
    # Show improvement plan
    if show_plan:
        plan = analyzer.generate_improvement_plan(result)
        
        formatter.console.print("\n[bold cyan]📋 Improvement Plan[/bold cyan]\n")
        formatter.console.print(f"Current Score: {plan['current_score']:.3f}")
        formatter.console.print(f"Target Score: {plan['target_score']:.3f}")
        formatter.console.print(f"Estimated Time: {plan['estimated_time_to_target']}")
        formatter.console.print("\n[bold]Top Improvements:[/bold]")
        
        for i, improvement in enumerate(plan['improvements'][:10], 1):
            formatter.console.print(f"\n{i}. {improvement['title']}")
            formatter.console.print(f"   Priority: {improvement['priority']}")
            formatter.console.print(f"   ROI: {improvement['roi']:.2f}")
            if improvement['suggested_fix']:
                formatter.console.print(f"   Fix: {improvement['suggested_fix']}")


@app.command("improve")
def self_improve(
    guardian_root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Root directory of QualiaGuardian",
    ),
    target_score: Optional[float] = typer.Option(
        0.95,
        "--target",
        help="Target quality score",
    ),
    auto_apply: bool = typer.Option(
        False,
        "--auto-apply/--no-auto-apply",
        help="Automatically apply safe improvements",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum number of improvement iterations",
    ),
):
    """
    Run recursive improvement loop to enhance QualiaGuardian quality.
    
    This creates a self-improving feedback loop where QualiaGuardian
    analyzes itself, applies improvements, and re-analyzes to verify.
    """
    improver = RecursiveImprover(
        guardian_root=str(guardian_root) if guardian_root else None,
        auto_apply=auto_apply,
        max_iterations=max_iterations,
    )
    
    summary = improver.run_improvement_loop(target_score=target_score)
    
    # Show trends
    improver.visualize_trends()
    
    return summary


@app.command("trends")
def show_trends(
    guardian_root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Root directory of QualiaGuardian",
    ),
):
    """Show improvement trends over time."""
    improver = RecursiveImprover(
        guardian_root=str(guardian_root) if guardian_root else None,
    )
    
    improver.visualize_trends()


@app.command("innovate")
def detect_innovations(
    guardian_root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Root directory of QualiaGuardian",
    ),
):
    """
    Detect innovation opportunities based on self-analysis.
    
    Identifies opportunities for new features and improvements
    based on patterns in the codebase and analysis results.
    """
    analyzer = SelfAnalyzer(str(guardian_root) if guardian_root else None)
    result = analyzer.analyze_self(run_quality=True)
    
    innovation_detector = InnovationDetector()
    innovations = innovation_detector.detect_innovations(result)
    
    formatter.console.print("\n[bold cyan]💡 Innovation Opportunities[/bold cyan]\n")
    
    if innovations:
        formatter.console.print(f"Found {len(innovations)} innovation opportunities:\n")
        
        for i, innovation in enumerate(innovations, 1):
            score = innovation.potential_impact * innovation.feasibility
            score_style = "green" if score > 0.5 else "yellow"
            
            formatter.console.print(f"[bold]{i}. {innovation.title}[/bold]")
            formatter.console.print(f"   Category: {innovation.category}")
            formatter.console.print(
                f"   Score: [{score_style}]{score:.2f}[/{score_style}] "
                f"(Impact: {innovation.potential_impact:.2f}, "
                f"Feasibility: {innovation.feasibility:.2f})"
            )
            formatter.console.print(f"   {innovation.description}")
            if innovation.suggested_implementation:
                formatter.console.print(f"   Implementation: {innovation.suggested_implementation}")
            formatter.console.print()
    else:
        formatter.console.print("No innovation opportunities detected at this time.\n")

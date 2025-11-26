#!/usr/bin/env python3
"""
Demonstration of QualiaGuardian Self-Improvement System

This script demonstrates how QualiaGuardian can analyze and improve itself.
"""

import sys
from pathlib import Path

# Add guardian to path
sys.path.insert(0, str(Path(__file__).parent))

from guardian.self_improvement.self_analyzer import SelfAnalyzer
from guardian.self_improvement.recursive_improver import RecursiveImprover
from guardian.self_improvement.innovation_detector import InnovationDetector
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig

def main():
    """Demonstrate self-improvement capabilities."""
    formatter = OutputFormatter(FormattingConfig())
    
    formatter.console.print(
        "\n[bold cyan]"
        "╔═══════════════════════════════════════════════════════════╗\n"
        "║  QualiaGuardian Self-Improvement Demonstration           ║\n"
        "║  Recursive Quality Enhancement System                     ║\n"
        "╚═══════════════════════════════════════════════════════════╝\n"
        "[/bold cyan]"
    )
    
    # 1. Self-Analysis
    formatter.console.print("\n[bold yellow]Step 1: Self-Analysis[/bold yellow]\n")
    formatter.console.print("Analyzing QualiaGuardian codebase...\n")
    
    analyzer = SelfAnalyzer()
    result = analyzer.analyze_self(run_quality=True)
    
    analyzer.display_self_analysis(result)
    
    # 2. Improvement Plan
    formatter.console.print("\n[bold yellow]Step 2: Improvement Plan[/bold yellow]\n")
    plan = analyzer.generate_improvement_plan(result)
    
    formatter.console.print(f"Current Score: {plan['current_score']:.3f}")
    formatter.console.print(f"Target Score: {plan['target_score']:.3f}")
    formatter.console.print(f"Estimated Time: {plan['estimated_time_to_target']}\n")
    
    formatter.console.print("[bold]Top 5 Improvements:[/bold]")
    for i, improvement in enumerate(plan['improvements'][:5], 1):
        formatter.console.print(f"\n{i}. {improvement['title']}")
        formatter.console.print(f"   Priority: {improvement['priority']}")
        formatter.console.print(f"   ROI: {improvement['roi']:.2f}")
        if improvement['suggested_fix']:
            formatter.console.print(f"   Fix: {improvement['suggested_fix']}")
    
    # 3. Innovation Detection
    formatter.console.print("\n[bold yellow]Step 3: Innovation Detection[/bold yellow]\n")
    innovation_detector = InnovationDetector()
    innovations = innovation_detector.detect_innovations(result)
    
    if innovations:
        formatter.console.print(f"[bold]Found {len(innovations)} innovation opportunities:[/bold]\n")
        for i, innovation in enumerate(innovations[:5], 1):
            formatter.console.print(f"{i}. {innovation.title}")
            formatter.console.print(f"   Category: {innovation.category}")
            formatter.console.print(f"   Impact: {innovation.potential_impact:.2f}")
            formatter.console.print(f"   Feasibility: {innovation.feasibility:.2f}")
            formatter.console.print(f"   {innovation.description}\n")
    else:
        formatter.console.print("No innovation opportunities detected at this time.\n")
    
    # 4. Recursive Improvement (Demo Mode)
    formatter.console.print("\n[bold yellow]Step 4: Recursive Improvement Loop (Demo)[/bold yellow]\n")
    formatter.console.print(
        "[dim]Note: Running in demo mode. Use --auto-apply to actually apply improvements.[/dim]\n"
    )
    
    improver = RecursiveImprover(auto_apply=False, max_iterations=3)
    
    formatter.console.print("Running 3 iterations of improvement loop...\n")
    
    # Simulate improvement loop
    for iteration in range(1, 4):
        formatter.console.print(f"[bold]Iteration {iteration}[/bold]")
        formatter.console.print(f"  Analyzing...")
        formatter.console.print(f"  Current Score: {result.overall_score:.3f}")
        formatter.console.print(f"  Issues Found: {len(result.issues_found)}")
        formatter.console.print(f"  Improvements Suggested: {min(5, len(result.issues_found))}\n")
    
    # 5. Summary
    formatter.console.print("\n[bold green]Summary[/bold green]\n")
    formatter.console.print("✅ Self-analysis completed")
    formatter.console.print("✅ Improvement plan generated")
    formatter.console.print("✅ Innovation opportunities identified")
    formatter.console.print("✅ Recursive improvement loop demonstrated")
    
    formatter.console.print(
        "\n[bold cyan]"
        "QualiaGuardian can now improve itself recursively!\n"
        "Use 'guardian self-improve improve --auto-apply' to start improving.\n"
        "[/bold cyan]"
    )

if __name__ == "__main__":
    main()

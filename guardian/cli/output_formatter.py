"""
Professional Output Formatter for Guardian CLI

Clean, consistent output formatting with proper styling and structure.
"""

import json
import sys # Added for stderr print in fallback
from typing import Dict, Any, List, Union, Optional # Added Optional
from dataclasses import dataclass, is_dataclass, asdict
from enum import Enum

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.style import Style
from rich.progress_bar import ProgressBar as RichProgressBar # For simple progress bars
from rich.theme import Theme

# Define a custom theme for Guardian (optional, but good for consistency)
guardian_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "header": "bold cyan on black", # Example style
    "section_title": "bold blue",
    "table_header": "bold blue",
    "score_good": "green",
    "score_medium": "yellow",
    "score_bad": "red",
    "progress.bar": "cyan",
    "progress.percentage": "bold",
})


# Helper function to recursively convert dataclasses to dicts
def _asdict_recursive(obj):
    if isinstance(obj, list):
        return [_asdict_recursive(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _asdict_recursive(v) for k, v in obj.items()}
    elif is_dataclass(obj):
        # For Rich, we might not need to convert everything to dict if we print Text/Panel directly
        return asdict(obj, dict_factory=_asdict_recursive)
    elif isinstance(obj, Enum):
        return obj.value
    return obj


class OutputLevel(Enum):
    """Output verbosity levels"""
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"

# Color class is no longer needed, rich handles colors and styles.

@dataclass
class FormattingConfig:
    """Configuration for output formatting"""
    use_colors: bool = True
    max_line_length: int = 80
    indent_size: int = 2
    show_timestamps: bool = False


class OutputFormatter:
    """
    Professional output formatter for Guardian analysis results

    Provides clean, consistent formatting for both human-readable and JSON output.
    Eliminates Python code artifacts and provides polished user experience.
    """

    def __init__(self, config: Optional[FormattingConfig] = None, theme: Optional[Theme] = None):
        self.config = config or FormattingConfig()
        self.level = OutputLevel.NORMAL
        self._progress_bar_enabled = True
        # Use a Rich Console instance.
        # If config.use_colors is False, Rich handles it gracefully (no_color=True).
        # However, Rich's auto-detection is usually good.
        # Forcing no_color based on self.config.use_colors might be too restrictive if terminal supports color.
        # Let Rich decide unless explicitly told no_color by a higher-level CLI arg.
        # For now, we'll respect self.config.use_colors for our manual style choices.
        self.console = Console(theme=theme or guardian_theme, force_terminal=self.config.use_colors if not sys.stdout.isatty() else None)


    def set_level(self, level: OutputLevel):
        """Set output verbosity level"""
        self.level = level

    def enable_progress_bars(self, enabled: bool = True):
        """Enable or disable progress bars for long operations"""
        # This might be used to control whether Rich progress bars are displayed
        self._progress_bar_enabled = enabled
    
    def _print_human_readable(self, results: Dict[str, Any]) -> None:
        """Formats and prints results for human reading using Rich."""
        
        renderables: List[Union[Table, Panel, Text, str]] = []

        # Header
        renderables.append(Panel(Text("Guardian Analysis Report", style="header", justify="center"), expand=False))
        renderables.append("") # For spacing
        
        # Basic info
        project_info_text = Text.assemble(
            ("Project Path: ", "info"), (str(results.get('project_path', 'N/A')), "")
        )
        status_text = self._get_rich_status_text(results.get('status', 'unknown'))
        
        project_info_panel_content = Text("\n").join([project_info_text, status_text])
        renderables.append(Panel(project_info_panel_content, title="[section_title]Project Information[/]", expand=False))
        renderables.append("")

        # TES Scores
        renderables.extend(self._format_tes_scores_rich(results))
        
        # E-TES v2.0 (if enabled)
        if results.get('etes_v2_enabled'):
            renderables.extend(self._format_etes_scores_rich(results))
        
        # Metrics
        renderables.extend(self._format_metrics_rich(results.get('metrics', {})))
        
        # Test execution
        renderables.extend(self._format_test_execution_rich(results.get('test_execution_summary', {})))
        
        # Security analysis
        renderables.extend(self._format_security_analysis_rich(results))
        
        # Issues (if any)
        if results.get('has_critical_issues'):
            renderables.extend(self._format_critical_issues_rich(results))
        
        # Recommendations
        renderables.extend(self._format_recommendations_rich(results))
        
        for r in renderables:
            if isinstance(r, str) and r == "": # Handle empty string for spacing
                self.console.line()
            else:
                self.console.print(r)

    def format_analysis_results(self, results: Dict[str, Any], format_type: str = "human") -> Union[str, None]:
        """
        Format analysis results for output.
        If 'human', prints to console using Rich and returns None.
        If 'json', returns JSON string.
        
        Args:
            results: Analysis results dictionary
            format_type: 'human' or 'json'
            
        Returns:
            Formatted JSON string if format_type is 'json', otherwise None.
        """
        if format_type == "json":
            return self._format_json(results)
        else:
            self._print_human_readable(results)
            return None # Rich prints directly to console
    
    def _format_json(self, results: Dict[str, Any]) -> str:
        """Format results as JSON, handling dataclasses recursively."""
        try:
            serializable_results = _asdict_recursive(results)
            return json.dumps(serializable_results, indent=2, ensure_ascii=False)
        except TypeError as e:
            # Log to stderr if possible, or use Rich for error printing
            self.console.print(f"[error]Error during JSON serialization: {e}. Falling back.[/]")
            return json.dumps({"error": "JSON serialization failed within formatter", "details": str(e)}, indent=2)

    # _format_human_readable is replaced by _print_human_readable
    
    def _format_header_rich(self, title: str) -> Panel: # Changed return type
        """Format section header using Rich Panel."""
        return Panel(Text(title, style="header", justify="center"), expand=False)

    def _format_section_title_rich(self, title: str) -> Text: # Changed return type
        """Format section title using Rich Text."""
        return Text(f"📊 {title}", style="section_title")

    def _format_tes_scores_rich(self, results: Dict[str, Any]) -> List[Union[Text, Table, str]]:
        """Format TES score section using Rich."""
        render_list: List[Union[Text, Table, str]] = []
        render_list.append(self._format_section_title_rich("Test Effectiveness Score (TES)"))
        
        tes_score = results.get('tes_score', 0.0)
        tes_grade = results.get('tes_grade', 'F')
        
        score_style = self._get_rich_score_style(tes_score)
        
        score_text = Text.assemble(
            ("  Score: ", ""), (f"{tes_score:.3f}", score_style),
            (" (Grade: ", ""), (tes_grade, score_style), (")", "")
        )
        render_list.append(score_text)
        
        components = results.get('tes_components', {})
        if components:
            table = Table(show_header=False, box=None, padding=(0,1,0,2)) # No box, more padding
            table.add_column()
            table.add_column()
            table.add_row("  Components:", "")
            table.add_row("    • Mutation Score:", f"{components.get('mutation_score', 0):.3f}")
            table.add_row("    • Behavior Coverage:", f"{components.get('behavior_coverage_calculated', 0):.3f}")
            table.add_row("    • Speed Factor:", f"{components.get('speed_factor_calculated', 0):.3f}")
            render_list.append(table)
        
        render_list.append("")
        return render_list

    def _format_etes_scores_rich(self, results: Dict[str, Any]) -> List[Union[Text, Table, str]]:
        """Format E-TES v2.0 score section using Rich."""
        render_list: List[Union[Text, Table, str]] = []
        render_list.append(self._format_section_title_rich("E-TES v2.0 (Evolutionary Test Effectiveness)"))
        
        etes_score = results.get('etes_score', 0.0)
        etes_grade = results.get('etes_grade', 'F')
        score_style = self._get_rich_score_style(etes_score)

        score_text = Text.assemble(
            ("  Score: ", ""), (f"{etes_score:.3f}", score_style),
            (" (Grade: ", ""), (etes_grade, score_style), (")", "")
        )
        render_list.append(score_text)
        
        etes_comp = results.get('etes_components', {})
        if etes_comp:
            table = Table(show_header=False, box=None, padding=(0,1,0,2))
            table.add_column()
            table.add_column()
            table.add_row("  Components:", "")
            for key, val_name in [
                ("Mutation Score", "mutation_score"), ("Evolution Gain", "evolution_gain"),
                ("Assertion IQ", "assertion_iq"), ("Behavior Coverage", "behavior_coverage"),
                ("Speed Factor", "speed_factor"), ("Quality Factor", "quality_factor")
            ]:
                table.add_row(f"    • {key}:", f"{etes_comp.get(val_name, 0):.3f}")
            render_list.append(table)
            
            insights = etes_comp.get('insights', [])
            if insights:
                render_list.append(Text("  Insights:", style="info"))
                for insight in insights[:5]:
                    render_list.append(Text(f"    💡 {insight}"))
        
        comparison = results.get('etes_comparison', {})
        if comparison:
            improvement = comparison.get('improvement', 0)
            comp_style = "score_good" if improvement > 0 else "score_medium" if improvement < 0 else ""
            sign = "+" if improvement > 0 else ""
            render_list.append(Text.assemble(
                ("  Improvement: ", ""), (f"{sign}{improvement:.3f}", comp_style), (" over legacy TES", "")
            ))
        
        render_list.append("")
        return render_list

    def _format_metrics_rich(self, metrics: Dict[str, Any]) -> List[Union[Text, Table, str]]:
        """Format metrics section using Rich Table."""
        render_list: List[Union[Text, Table, str]] = []
        render_list.append(self._format_section_title_rich("Code Quality Metrics"))

        if not metrics:
            render_list.append(Text("  No metrics available", style="info"))
            render_list.append("")
            return render_list

        table = Table(title=None, show_header=True, header_style="table_header", box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        metric_map = {
            'total_lines_of_code_python': ("Lines of Code", self._get_loc_status_rich),
            'python_files_analyzed': ("Python Files", lambda x: Text("📄")),
            'average_cyclomatic_complexity': ("Avg Complexity", self._get_complexity_status_rich, ".2f"),
            'long_functions_count': ("Long Functions", self._get_function_status_rich),
            'large_classes_count': ("Large Classes", self._get_class_status_rich),
            'unused_imports_count': ("Unused Imports", self._get_unused_imports_status_rich),
            'circular_dependencies_count': ("Circular Dependencies", self._get_dependencies_status_rich)
        }

        for key, (label, status_func, *fmt) in metric_map.items():
            if key in metrics:
                value = metrics[key]
                value_str = f"{value:{fmt[0]}}" if fmt else str(value)
                if key == 'total_lines_of_code_python': value_str = f"{value:,}"
                status_renderable = status_func(value)
                table.add_row(label, value_str, status_renderable)
        
        if table.rows:
            render_list.append(table)
        render_list.append("")
        return render_list
    
    def _format_test_execution_rich(self, test_summary: Dict[str, Any]) -> List[Union[Text, str]]:
        render_list: List[Union[Text, str]] = []
        render_list.append(self._format_section_title_rich("Test Execution"))
        
        if not test_summary:
            render_list.append(Text("  No test execution data available", style="info"))
            render_list.append("")
            return render_list
        
        success = test_summary.get('pytest_ran_successfully', False)
        status_style = "success" if success else "error"
        status_icon = "✓" if success else "✗"
        
        render_list.append(Text.assemble(("  Status: ", ""), (f"{status_icon} {('PASSED' if success else 'FAILED')}", status_style)))
        render_list.append(Text(f"  Exit Code: {test_summary.get('pytest_exit_code', 'N/A')}"))
        
        duration = test_summary.get('pytest_duration_seconds', 0)
        if duration:
            render_list.append(Text(f"  Duration: {duration:.2f}s"))
        
        render_list.append("")
        return render_list
    
    def _format_security_analysis_rich(self, results: Dict[str, Any]) -> List[Union[Text, str]]:
        render_list: List[Union[Text, str]] = []
        render_list.append(self._format_section_title_rich("Security Analysis"))
        
        security = results.get('security_analysis', {})
        if not security:
            render_list.append(Text("  No security analysis data available", style="info"))
            render_list.append("")
            return render_list
        
        vuln_count = security.get('dependency_vulnerabilities_count', 0)
        vuln_style = "error" if vuln_count > 0 else "success"
        render_list.append(Text.assemble(("  Vulnerabilities: ", ""), (str(vuln_count), vuln_style)))
        
        render_list.append(Text(f"  Eval Usage: {security.get('eval_usage_count', 0)}"))
        render_list.append(Text(f"  Hardcoded Secrets: {security.get('hardcoded_secrets_count', 0)}"))
        
        render_list.append("")
        return render_list
    
    def _format_critical_issues_rich(self, results: Dict[str, Any]) -> List[Union[Text, str]]:
        render_list: List[Union[Text, str]] = []
        render_list.append(Text("⚠️ CRITICAL ISSUES DETECTED", style="bold error"))
        render_list.append("")
        
        details = results.get('details', {})
        vulns = details.get('vulnerability_details_list', [])
        if vulns:
            render_list.append(Text("  Security Vulnerabilities:", style="warning"))
            for vuln in vulns[:3]:
                render_list.append(Text(f"    • {vuln.get('name', 'Unknown')} ({vuln.get('id', 'No ID')})"))
        
        evals = details.get('eval_usage_details_list', [])
        if evals:
            render_list.append(Text("  Dangerous eval() Usage:", style="warning"))
            for eval_info in evals[:3]:
                render_list.append(Text(f"    • {eval_info.get('file', 'Unknown')}:{eval_info.get('line_number', '?')}"))
        
        render_list.append("")
        return render_list
    
    def _format_recommendations_rich(self, results: Dict[str, Any]) -> List[Union[Text, str]]:
        render_list: List[Union[Text, str]] = []
        render_list.append(self._format_section_title_rich("Recommendations"))
        
        recommendations_text: List[str] = []
        # ... (logic for populating recommendations_text remains similar) ...
        tes_score = results.get('tes_score', 0.0)
        if tes_score < 0.5: recommendations_text.append("🎯 Focus on improving test coverage and quality")
        etes_comparison = results.get('etes_comparison', {})
        if etes_comparison:
            etes_recs = etes_comparison.get('recommendations', [])
            recommendations_text.extend(f"🧬 {rec}" for rec in etes_recs[:3])
        security = results.get('security_analysis', {})
        if security.get('dependency_vulnerabilities_count', 0) > 0: recommendations_text.append("🔒 Update vulnerable dependencies immediately")
        if security.get('eval_usage_count', 0) > 0: recommendations_text.append("⚠️  Remove or secure eval() usage")
        metrics = results.get('metrics', {})
        if metrics.get('long_functions_count', 0) > 5: recommendations_text.append("📏 Refactor long functions for better maintainability")
        if metrics.get('circular_dependencies_count', 0) > 0: recommendations_text.append("🔄 Resolve circular dependencies")
        if not recommendations_text: recommendations_text.append("✅ No major issues detected - keep up the good work!")

        for rec in recommendations_text:
            render_list.append(Text(f"  {rec}"))
        
        render_list.append("")
        return render_list
    
    def _get_rich_status_text(self, status: str) -> Text:
        """Return Rich Text with style for status."""
        if 'complete' in status.lower() or 'passed' in status.lower():
            return Text(status, style="success")
        elif 'partial' in status.lower():
            return Text(status, style="warning")
        elif 'error' in status.lower() or 'fail' in status.lower():
            return Text(status, style="error")
        return Text(status)

    def _get_rich_score_style(self, score: float) -> str: # Return str (style name)
        """Get Rich style name for score based on value."""
        if score >= 0.8: return "score_good"
        elif score >= 0.6: return "score_medium"
        return "score_bad"

    def format_error(self, error_message: str) -> None: # Changed to print
        """Format and print error message using Rich."""
        self.console.print(Text(f"❌ Error: {error_message}", style="error"))
    
    def format_warning(self, warning_message: str) -> None: # Changed to print
        """Format and print warning message using Rich."""
        self.console.print(Text(f"⚠️ Warning: {warning_message}", style="warning"))
    
    def format_success(self, success_message: str) -> None: # Changed to print
        """Format and print success message using Rich."""
        self.console.print(Text(f"✅ {success_message}", style="success"))

    def format_progress_bar(self, current: int, total: int, description: str = "") -> None: # Changed to print
        """Format and print a simple progress bar using Rich."""
        if not self._progress_bar_enabled or self.level == OutputLevel.QUIET:
            return
        # For a simple, one-line progress bar that updates in place,
        # direct use of rich.progress.Progress with console.status or console.live is better.
        # This is a simplified version.
        bar = RichProgressBar(total=total, completed=current, width=40)
        self.console.print(Text(description + " ") + bar + Text(f" {current}/{total} ({current/total*100:.1f}%)"))


    def format_table(self, headers: List[str], rows: List[List[Any]], title: Optional[str] = None) -> Table: # Returns Rich Table
        """Format data as a Rich Table."""
        table = Table(title=title if title else None, show_header=True, header_style="table_header", box=None)
        for header in headers:
            table.add_column(header)
        
        for row in rows:
            # Ensure all row items are strings for Rich Table
            table.add_row(*(str(item) for item in row))
        return table

    def format_clean_error(self, error_message: str, show_details: bool = False) -> None: # Changed to print
        """Format and print clean error message using Rich."""
        if self.level == OutputLevel.DEBUG and show_details:
            self.format_error(error_message) # This will print
            return

        clean_message = error_message.split('\n')[0]
        clean_message = clean_message.replace('Traceback', '').replace('Exception:', '').strip()
        self.console.print(Text(f"❌ {clean_message}", style="error"))

    # Status emoji/text functions for Rich table cells
    def _get_loc_status_rich(self, loc: int) -> Text:
        if loc < 1000: return Text("📗 SMALL", style="score_good")
        elif loc < 10000: return Text("📘 MEDIUM", style="score_medium")
        return Text("📕 LARGE", style="score_bad")

    def _get_complexity_status_rich(self, complexity: float) -> Text:
        if complexity <= 2.0: return Text("🟢 EXCELLENT", style="score_good")
        elif complexity <= 4.0: return Text("🟡 GOOD", style="score_medium")
        elif complexity <= 6.0: return Text("🟠 MODERATE", style="warning")
        return Text("🔴 HIGH", style="score_bad")

    def _get_function_status_rich(self, count: int) -> Text:
        if count == 0: return Text("✅ EXCELLENT", style="success")
        elif count <= 5: return Text("🟡 GOOD", style="score_medium")
        elif count <= 15: return Text("🟠 MODERATE", style="warning")
        return Text("🔴 HIGH", style="score_bad")

    def _get_class_status_rich(self, count: int) -> Text:
        if count == 0: return Text("✅ EXCELLENT", style="success")
        elif count <= 3: return Text("🟡 GOOD", style="score_medium")
        elif count <= 8: return Text("🟠 MODERATE", style="warning")
        return Text("🔴 HIGH", style="score_bad")

    def _get_unused_imports_status_rich(self, count: int) -> Text:
        if count == 0: return Text("✅ CLEAN", style="success")
        elif count <= 5: return Text("🟡 MINOR", style="warning")
        return Text("🟠 CLEANUP NEEDED", style="score_bad")

    def _get_dependencies_status_rich(self, count: int) -> Text:
        if count == 0: return Text("✅ CLEAN", style="success")
        elif count <= 2: return Text("🟡 MINOR", style="warning")
        return Text("🔴 CRITICAL", style="score_bad")

    # Aliases remain, but their behavior changes (print vs return string)
    def format_console(self, results: Dict[str, Any]) -> None:
        """Format results for console output using Rich (prints directly)."""
        self.format_analysis_results(results, 'human')

    # format_json still returns a string
    # def format_json(self, results: Dict[str, Any]) -> str:
    #     """Format results as JSON (alias for format_analysis_results)"""
    #     return self.format_analysis_results(results, 'json') # This will call _format_json

    def format_html(self, results: Dict[str, Any]) -> str:
        """Format results as HTML using Rich's HTML export."""
        # Capture Rich output to console and then save as HTML
        # This requires printing to a console with record=True
        temp_console = Console(record=True, theme=self.console.theme, force_terminal=True, width=self.config.max_line_length or 80)
        
        # Replicate the printing logic of _print_human_readable to the temp_console
        # This is a bit duplicative but ensures the HTML captures the same structure.
        # A better way might be to have _print_human_readable build a list of Rich renderables
        # that can then be printed to any console. For now, direct print:
        
        temp_renderables: List[Union[Table, Panel, Text, str]] = []
        temp_renderables.append(Panel(Text("Guardian Analysis Report", style="header", justify="center"), expand=False))
        temp_renderables.append("")
        project_info_text = Text.assemble(("Project Path: ", "info"), (str(results.get('project_path', 'N/A')), ""))
        status_text = self._get_rich_status_text(results.get('status', 'unknown'))
        project_info_panel_content = Text("\n").join([project_info_text, status_text])
        temp_renderables.append(Panel(project_info_panel_content, title="[section_title]Project Information[/]", expand=False))
        temp_renderables.append("")
        temp_renderables.extend(self._format_tes_scores_rich(results))
        if results.get('etes_v2_enabled'): temp_renderables.extend(self._format_etes_scores_rich(results))
        temp_renderables.extend(self._format_metrics_rich(results.get('metrics', {})))
        temp_renderables.extend(self._format_test_execution_rich(results.get('test_execution_summary', {})))
        temp_renderables.extend(self._format_security_analysis_rich(results))
        if results.get('has_critical_issues'): temp_renderables.extend(self._format_critical_issues_rich(results))
        temp_renderables.extend(self._format_recommendations_rich(results))

        for r_item in temp_renderables:
            if isinstance(r_item, str) and r_item == "": temp_console.line()
            else: temp_console.print(r_item)
            
        return temp_console.export_html(inline_styles=True, theme=self.console.theme)

    def display_gamify_hud(self, gamify_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Displays a gamified Heads-Up Display (HUD) with current stats.
        For M1 of CLI upgrade, uses placeholder data.
        """
        if self.level == OutputLevel.QUIET:
            return

        # Use placeholder data if none provided
        data = gamify_data or {
            "level": 1,
            "xp": 0,
            "xp_to_next_level": 1000,
            "streak_days": 0,
            "active_quest_name": "Complete your first scan!",
            "active_quest_progress": 0,
            "active_quest_target": 1,
            "badges_earned": 0,
        }

        level_str = str(data.get("level", 1))
        xp_str = str(data.get("xp", 0))
        xp_next_str = str(data.get("xp_to_next_level", 1000))
        streak_str = str(data.get("streak_days", 0))
        quest_name = str(data.get("active_quest_name", "N/A"))
        quest_prog = int(data.get("active_quest_progress", 0))
        quest_target = int(data.get("active_quest_target", 1))
        
        xp_bar_length = 10
        xp_filled = int(xp_bar_length * (data.get("xp",0) / data.get("xp_to_next_level",1000))) if data.get("xp_to_next_level",1000) > 0 else 0
        xp_bar = '█' * xp_filled + '─' * (xp_bar_length - xp_filled)

        quest_bar_length = 10
        quest_filled = int(quest_bar_length * (quest_prog / quest_target)) if quest_target > 0 else 0
        quest_bar = '█' * quest_filled + '─' * (quest_bar_length - quest_filled)


        hud_text = Text.assemble(
            ("LVL: ", "bold white"), (level_str, "bold yellow"),
            (" | XP: ", "bold white"), (f"{xp_str}/{xp_next_str}", "bold green"), (f" [{xp_bar}]", "dim white"),
            (" | 🔥 Streak: ", "bold white"), (streak_str, "bold orange3"), (" days", "orange3"),
            ("\nQuest: ", "bold white"), (f"{quest_name}", "italic cyan"),
            (f" [{quest_bar}] {quest_prog}/{quest_target}", "dim white")
        )
        
        panel = Panel(
            hud_text,
            title="[bold magenta]🛡️ Guardian HQ[/bold magenta]",
            border_style="info", # Use the theme key "info" which maps to "dim cyan"
            expand=False
        )
        self.console.print(panel)

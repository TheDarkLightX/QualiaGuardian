"""
Guardian CLI - Professional Code Quality & Security Analysis Tool

Clean, modern command-line interface with proper error handling,
logging, and beautiful output formatting.
"""

import argparse
import logging
import sys
import os
import json
import yaml # For loading config file
from pathlib import Path # Added for path handling
from typing import Optional, List # Added List

import typer # Added for CLI subcommands
from rich.panel import Panel # Added for gamify crown
from rich.text import Text # Added for gamify crown, if directly used
from rich.status import Status # Added for spinner

from guardian.evolution.adaptive_emt import AdaptiveEMT # Added for ec-evolve
from guardian.evolution.types import EvolutionHistory # Added for ec-evolve
from guardian.history import HistoryManager # Added for gamify status
from guardian.analytics.shapley import calculate_shapley_values, TestId # Added for gamify crown
# from guardian.analytics.metric_stubs import metric_evaluator_stub, TEST_CACHE # No longer directly used by crown
from guardian.core.api import evaluate_subset # Added for Layer 1 Shapley
from guardian.evolution.smart_mutator import Mutant # For type hinting the cache
from guardian.analytics.metric_stubs import TEST_CACHE # Still used for test_ids source in S1

# Define logger at module level
logger = logging.getLogger(__name__)

# Typer application instance
app = typer.Typer(
    name="guardian",
    help="Guardian CLI - Professional Code Quality & Security Analysis Tool",
    add_completion=False,
    rich_markup_mode="markdown" # Enable rich markup for help text
)

gamify_app = typer.Typer(name="gamify", help="Gamification features: XP, badges, quests, leaderboards.")
app.add_typer(gamify_app)

# Import and add self-improvement commands
from guardian.cli.self_improve_command import app as self_improve_app
app.add_typer(self_improve_app)

@gamify_app.command("status", help="Display your current Guardian gamification status (XP, level, badges, active quest).")
def gamify_status(
    username: Optional[str] = typer.Option(None, "--user", "-u", help="Specify the username for status. Defaults to current user.")
):
    """
    Shows your current progress in the Guardian gamification system.
    Includes Level, XP, XP to next level, current streak, and active quest.
    """
    formatter_config = FormattingConfig(use_colors=sys.stdout.isatty())
    formatter = OutputFormatter(config=formatter_config)
    
    try:
        history_manager = HistoryManager() # Uses default DB path
        player_id_for_status = history_manager._get_player_id(username) # Ensures player exists

        player_status = history_manager.get_player_status(player_id_for_status) # Use player_id directly
        active_quest = history_manager.get_active_quest(player_id_for_status)
        badges_earned_count = history_manager.get_earned_badges_count(player_id_for_status) # Get count before closing

        if player_status.get("error"):
            formatter.format_error(f"Could not retrieve player status: {player_status['error']}")
            raise typer.Exit(code=1)

        hud_data = {
            "level": player_status.get("level", 1),
            "xp": player_status.get("xp", 0),
            "xp_to_next_level": player_status.get("xp_to_next_level", 1000), # Default from get_xp_for_next_level(1)
            "streak_days": player_status.get("streak_days", 0),
            "active_quest_name": active_quest.get("description", "No active quest") if active_quest else "No active quest",
            "active_quest_progress": active_quest.get("current_progress", 0) if active_quest else 0,
            "active_quest_target": active_quest.get("target_value", 1) if active_quest else 1,
            "badges_earned": badges_earned_count,
        }
        formatter.display_gamify_hud(hud_data)
        
        # Record this 'gamify status' run
        try:
            # Minimal metrics for a status check, primarily to log activity and assign quests
            status_actions = ["checked_status"]
            history_manager.record_run(
                username=username, # username is from the gamify_status --user option
                command_name="gamify status",
                current_metrics={}, # No specific metrics from status check itself
                actions_taken=status_actions,
                xp_bonus=1 # Small XP for checking status
            )
        except Exception as e_hist_status:
            logger.warning(f"Could not record 'gamify status' run to history: {e_hist_status}", exc_info=False)
        
        history_manager.close() # Close connection after all operations

    except Exception as e:
        logger.error(f"Error in 'gamify status' command: {e}", exc_info=True)
        formatter.format_error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)

@gamify_app.command("crown", help="Show top-N tests contributing most to quality (via Shapley values).")
def gamify_crown(
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top tests to display."),
    permutations: int = typer.Option(200, "--permutations", "-p", help="Number of permutations for Shapley sampling."),
    test_dir: Path = typer.Option(
        "tests",
        "--test-dir",
        "-d",
        help="Directory to scan for test files (e.g., 'tests', 'project/tests'). Relative to project_root.",
        file_okay=False, # Must be a directory
        dir_okay=True,
        readable=True,
        resolve_path=False # Keep it relative to project_root for now
    ),
    project_root: Path = typer.Option(
        ".", # Default to current directory
        "--project-root",
        "-pr",
        help="Root directory of the project to analyze. Test discovery is relative to this path.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True # Resolve to an absolute path
    )
):
    """
    Identifies and displays the tests that are the "crown jewels" of your
    test suite, i.e., those contributing the most to overall quality metrics,
    calculated using Shapley values. Uses a stubbed metric evaluator for now.
    """
    formatter_config = FormattingConfig(use_colors=sys.stdout.isatty())
    formatter = OutputFormatter(config=formatter_config)
    logger.info(f"Calculating Shapley values for tests. This might take a moment (permutations={permutations})...")

    # For S1 Layer 0, we use the TEST_CACHE keys as the list of "all tests".
    # A real implementation would scan test_dir or use a test discovery mechanism.
    # Discover actual test files from the test_dir
    # For simplicity, collect .py files starting with "test_".
    # A more robust discovery might use pytest --collect-only and parse output.
    discovered_tests: List[Path] = []
    # Test discovery should be relative to project_root
    # test_dir itself might be relative, so join it with project_root
    absolute_test_dir = (project_root / test_dir).resolve()

    if not absolute_test_dir.exists() or not absolute_test_dir.is_dir():
        formatter.format_error(f"Test directory '{absolute_test_dir}' does not exist or is not a directory.")
        raise typer.Exit(code=1)

    logger.info(f"Scanning for tests in: {absolute_test_dir} (relative to project root: {project_root})")

    for p_item in absolute_test_dir.rglob("test_*.py"):
        if p_item.is_file():
            # Store paths relative to project_root for consistency if needed,
            # or absolute paths. evaluate_subset expects absolute or resolvable paths.
            # For simplicity, let's use resolved absolute paths for discovered_tests.
            discovered_tests.append(p_item.resolve())
    
    all_test_ids: List[TestId] = discovered_tests

    if not all_test_ids:
        formatter.format_warning(f"No test files (test_*.py) found in '{absolute_test_dir}'. Cannot calculate Shapley values.")
        return

    try:
        # The `project_root` from the command is now the context for `evaluate_subset`.
        # `selected_tests_subset` will be a list of absolute Path objects.
        # Mutant cache is no longer passed from here, as evaluate_subset now uses
        # the mutation_sensor which relies on mutmut's internal caching.
        
        def metric_eval_for_shapley(selected_tests_subset: List[TestId]) -> float:
            # Ensure selected_tests_subset contains Path objects.
            # `discovered_tests` are already resolved Paths.
            return evaluate_subset(
                project_path=project_root,
                selected_tests=selected_tests_subset
                # mutant_cache argument removed from evaluate_subset
            )

        # Wrap the calculation in a status spinner
        with Status("[bold green]Calculating Shapley values...[/]", console=formatter.console, spinner="dots") as status:
            shapley_results = calculate_shapley_values(
                test_ids=all_test_ids, # These are resolved Path objects
                metric_evaluator_func=metric_eval_for_shapley,
                num_permutations=permutations
            )
            status.update("[bold green]Shapley values calculated.[/]")

        if not shapley_results:
            formatter.format_warning("Shapley value calculation returned no results.")
            return

        # Sort by Shapley value (descending) and take top N
        sorted_tests = sorted(shapley_results.items(), key=lambda item: item[1], reverse=True)
        top_tests_data = []
        for i, (test_id, value) in enumerate(sorted_tests[:top_n]):
            # Convert Path to str for display if it's a Path object
            display_test_id = str(test_id) if isinstance(test_id, Path) else test_id
            top_tests_data.append([str(i + 1), display_test_id, f"{value:.4f}"])
        
        if not top_tests_data:
            formatter.console.print(Text("No Shapley values to display.", style="info"))
            return

        table_title = f"👑 Top {len(top_tests_data)} Most Valuable Tests (Shapley Values - Mutation Score)"
        
        # Use the formatter's table method which now returns a Rich Table
        table = formatter.format_table(
            headers=["Rank", "Test Identifier", "Shapley Value (Subset Score)"], # Updated metric name
            rows=top_tests_data,
        )
        
        # Print the table wrapped in a Panel for the title
        formatter.console.print(Panel(table, title=f"[section_title]{table_title}[/section_title]", expand=False))

        # Record the run
        try:
            history_manager = HistoryManager()
            # For current_metrics, we don't have a direct equivalent from Shapley results yet.
            # We could pass the top Shapley value or number of valuable tests found.
            # For now, let's pass minimal info.
            current_run_metrics = {"shapley_tests_found": len(top_tests_data)}
            history_manager.record_run(
                username=None,  # Use default username from HistoryManager
                command_name="gamify crown",
                current_metrics=current_run_metrics,
                # delta_metrics would require comparing to a previous state, not available here.
                actions_taken=[f"top_n_{top_n}", f"permutations_{permutations}"]
            )
            history_manager.close()
        except Exception as e_hist:
            logger.warning(f"Could not record 'gamify crown' run to history: {e_hist}", exc_info=True)
            # Do not fail the main command if history recording fails

    except Exception as e:
        logger.error(f"Error in 'gamify crown' command: {e}", exc_info=True)
        formatter.format_error(f"An unexpected error occurred during Shapley value calculation: {e}")
        raise typer.Exit(code=1)


@app.command("ec-evolve", help="Evolve a test suite using Evolutionary Computation (NSGA-II).")
def ec_evolve(
    codebase_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to the source codebase to test.",
        show_default=False,
    ),
    test_suite_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to the initial test suite directory.",
        show_default=False,
    ),
    output_dir: Path = typer.Option(
        "evolved_tests",
        "--output-dir",
        "-o",
        help="Directory to save evolved tests and evolution history.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    population_size: int = typer.Option(
        50, "--pop-size", help="Number of individuals in the population."
    ),
    generations: int = typer.Option(
        100, "--generations", help="Number of generations to run."
    ),
    mutation_rate: float = typer.Option(
        0.1, "--mutation-rate", help="Probability of mutating an individual."
    ),
    crossover_prob: float = typer.Option(
        0.7, "--crossover-prob", help="Probability of performing crossover."
    ),
    early_stopping_patience: int = typer.Option(
        10,
        "--early-stopping",
        help="Number of generations with no improvement to trigger early stopping. 0 to disable.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to Guardian config YAML for sensor settings (if needed by FitnessEvaluator).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        show_default=False,
    ),
    username: Optional[str] = typer.Option(
        None,
        "--user",
        "-u",
        help="Specify the username for history tracking. Defaults to system user or 'default_guardian_user'."
    ),
):
    """
    Evolves a test suite using the AdaptiveEMT engine (NSGA-II).
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Test Evolution with Guardian EC Evolve command:")
    logger.info(f"  Codebase Path: {codebase_path}")
    logger.info(f"  Initial Test Suite Path: {test_suite_path}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Population Size: {population_size}")
    logger.info(f"  Generations: {generations}")
    logger.info(f"  Mutation Rate: {mutation_rate}")
    logger.info(f"  Crossover Probability: {crossover_prob}")
    logger.info(f"  Early Stopping Patience: {early_stopping_patience}")
    logger.info(f"  Guardian Config File: {config_file if config_file else 'Not provided'}")

    try:
        # Instantiate AdaptiveEMT
        emt_engine = AdaptiveEMT(
            codebase_path=str(codebase_path),
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_probability=crossover_prob,
            early_stopping_patience=early_stopping_patience,
            config_path=str(config_file) if config_file else None,
            # initial_test_suite_path might be part of constructor in future
        )

        # Call the main evolution method
        logger.info("Invoking AdaptiveEMT.evolve()...")
        evolved_population, history = emt_engine.evolve(
            initial_test_suite_path=str(test_suite_path)
        )
        logger.info("Evolution process completed.")

        # Save results
        # Save evolved test suite
        final_suite_dir = output_dir / "final_suite"
        final_suite_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving final evolved test suite to: {final_suite_dir}")

        if not evolved_population:
            logger.warning("Evolution resulted in an empty population. No tests to save.")
        else:
            for i, individual in enumerate(evolved_population):
                test_file_path = final_suite_dir / f"evolved_test_{i+1}.py"
                
                code_to_write = ""
                # Check for setup_code, test_code, teardown_code attributes
                # and concatenate them if they exist.
                # This assumes TestIndividual might have these attributes.
                setup_c = getattr(individual, 'setup_code', None)
                test_c = getattr(individual, 'test_code', '# No test_code attribute found in individual')
                teardown_c = getattr(individual, 'teardown_code', None)

                if setup_c:
                    code_to_write += setup_c + "\n\n"
                code_to_write += test_c
                if teardown_c:
                    code_to_write += "\n\n" + teardown_c
                
                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(code_to_write)
                logger.info(f"Saved evolved test {i+1} to {test_file_path}")

        # Save evolution history
        history_path = output_dir / "evolution_summary.json"
        logger.info(f"Saving evolution history to: {history_path}")
        try:
            history_content_to_save = history
            if hasattr(history, "to_dict"): # Check if a to_dict method exists
                history_content_to_save = history.to_dict()
            elif not isinstance(history, (dict, list)): # If not dict/list and no to_dict, try converting its attributes
                 if hasattr(history, '__dict__'):
                     history_content_to_save = history.__dict__

            with history_path.open("w", encoding="utf-8") as f:
                json.dump(history_content_to_save, f, indent=2, default=str) # Added default=str for non-serializable
            logger.info(f"Evolution history successfully saved to {history_path}")
        except TypeError as e:
            logger.error(f"Failed to serialize EvolutionHistory to JSON: {e}. "
                         f"The EvolutionHistory object or its components (like FitnessVector) might need a "
                         f"more specific to_dict() method that converts all custom objects and numpy types "
                         f"to JSON-compatible types. Fallback: writing string representation.")
            with history_path.open("w", encoding="utf-8") as f:
                f.write(str(history)) # Fallback
        except Exception as e_gen:
            logger.error(f"An unexpected error occurred while saving evolution history: {e_gen}")
            with history_path.open("w", encoding="utf-8") as f:
                f.write(f"Error saving history: {e_gen}\n\n{str(history)}")


        logger.info(f"Evolution complete. All results saved in {output_dir}")
        
        # Record the ec-evolve run
        try:
            history_manager = HistoryManager()
            
            num_generations_ran = 0
            if history and hasattr(history, 'generations') and isinstance(history.generations, list):
                num_generations_ran = len(history.generations)
            
            final_pop_size = len(evolved_population) if evolved_population else 0

            current_run_metrics = {
                "generations_ran": num_generations_ran,
                "final_population_size": final_pop_size,
                # TODO: Potentially add best fitness scores if easily accessible and serializable
            }
            actions_taken_for_history = [
                f"pop_size_{population_size}",
                f"generations_config_{generations}",
                f"mutation_rate_{mutation_rate}",
                f"crossover_prob_{crossover_prob}",
                f"early_stopping_{early_stopping_patience}"
            ]

            history_manager.record_run(
                username=username,
                command_name="ec-evolve",
                current_metrics=current_run_metrics,
                actions_taken=actions_taken_for_history,
                xp_bonus=75 # Assign a bonus for running evolution
            )
            history_manager.close()
            logger.info("EC Evolve run recorded in history.")
        except Exception as e_hist_ec_evolve:
            logger.warning(f"Could not record 'ec-evolve' run to history: {e_hist_ec_evolve}", exc_info=True)
            # Do not fail the main command if history recording fails

        typer.echo(f"✅ Evolution finished. Results are in {output_dir.resolve()}")

    except Exception as e:
        logger.error(f"An error occurred during the evolution process: {e}", exc_info=True)
        typer.echo(f"❌ Error during evolution: {e}", err=True)
        raise typer.Exit(code=1)


from guardian.cli.analyzer import ProjectAnalyzer
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig, OutputLevel
from guardian.cli.quality_scoring_handler import QualityScoringHandler

# Import core calculation and config classes
from guardian.core.tes import get_etes_grade

# Import legacy analysis functions for backward compatibility
from guardian.analysis.static import (
    count_lines_of_code,
    calculate_cyclomatic_complexity,
    find_long_elements,
    find_large_classes,
    analyze_imports,
    find_unused_imports,
    build_import_graph,
    find_circular_dependencies
)
from guardian.analysis.security import (
    check_dependencies_vulnerabilities,
    check_for_eval_usage,
    check_for_hardcoded_secrets
)
from guardian.test_execution.pytest_runner import run_pytest
# calculate_tes, calculate_etes_v2, compare_tes_vs_etes might be deprecated or refactored in tes.py
# For now, keeping get_etes_grade as it's generic for 0-1 scores.
# from guardian.core.tes import calculate_tes, get_tes_grade, calculate_etes_v2, get_etes_grade, compare_tes_vs_etes


def setup_logging(verbose: bool = False) -> None:
    """Setup clean logging configuration without Python artifacts"""
    if verbose:
        level = logging.INFO
        format_str = '%(message)s'
    else:
        level = logging.WARNING
        format_str = '%(message)s'

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stderr)  # Send logs to stderr to keep stdout clean
        ]
    )


def create_analyzer(args) -> ProjectAnalyzer:
    """Create and configure project analyzer"""
    config = {
        'max_function_lines': args.max_function_lines,
        'max_class_methods': args.max_class_methods,
        # 'use_etes_v2': args.use_etes_v2 # This will be handled by quality_mode
    }
    # ProjectAnalyzer might need to be aware of QualityConfig if it's orchestrating
    # For now, assume QualityConfig is handled separately or passed to a dedicated quality analysis step
    return ProjectAnalyzer(config)


def create_formatter(args) -> OutputFormatter:
    """Create and configure output formatter"""
    formatting_config = FormattingConfig(
        use_colors=not args.no_color,
        max_line_length=args.max_line_length if hasattr(args, 'max_line_length') else 80
    )
    formatter = OutputFormatter(formatting_config)

    if hasattr(args, 'verbose') and args.verbose:
        formatter.set_level(OutputLevel.VERBOSE)
    elif hasattr(args, 'quiet') and args.quiet:
        formatter.set_level(OutputLevel.QUIET)

    return formatter


# Legacy function for backward compatibility (deprecated)


# Legacy function for backward compatibility (deprecated)
def analyze_project(project_path, user_stories_file_path=None, max_function_lines=20, max_class_methods=10, test_path=None, use_etes_v2=False):
    """
    Analyzes the specified project path.
    Calculates total lines of code, cyclomatic complexity, finds long functions,
    and large classes for Python files.
    Cognitive complexity calculation is temporarily removed.
    Reads user stories if provided.
    """
    # Clean output without Python artifacts
    if user_stories_file_path:
        pass  # Process silently

    actual_test_path = test_path if test_path else project_path

    if not os.path.exists(project_path):
        print(f"Error: Project path '{project_path}' does not exist.")
        return None
    if not os.path.isdir(project_path):
        print(f"Error: Project path '{project_path}' is not a directory.")
        return None

    total_loc = 0
    # total_cognitive_complexity = 0 # Temporarily removed
    total_cyclomatic_complexity_sum_for_avg = 0 # Sum of complexities for averaging
    analyzed_functions_or_blocks_for_cc_avg = 0 # Count of items used for CC average
    long_functions_found_list = []
    eval_usage_findings_list = []
    large_classes_found_list = []
    hardcoded_secrets_findings_list = []
    all_project_imports = []
    unused_imports_findings_list = []

    python_files_count = 0
    status_message = "Initial analysis complete."

    for root, _, files in os.walk(project_path):
        # Skip .venv directory
        if ".venv" in root.split(os.sep) or ".git" in root.split(os.sep): # also skip .git
            continue
        for file in files:
            if file.endswith(".py"):
                python_files_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        loc = count_lines_of_code(content)
                        total_loc += loc

                        # cog_comp = calculate_cognitive_complexity(content) # Temporarily removed
                        # total_cognitive_complexity += cog_comp # Temporarily removed

                        # Radon's cc_visit returns a list of blocks (functions, methods, classes)
                        # We need to sum their complexities and count them for an average
                        # For simplicity, let's assume calculate_cyclomatic_complexity now returns
                        # a tuple (sum_of_complexities, count_of_blocks) or we adapt.
                        # The current static.py returns an average per file.
                        # Let's adjust to get total and count to make a project-wide average.
                        # OR, average the per-file averages.
                        # For now, let's assume static.py's cc_visit is used and we average those averages.
                        # This is less accurate than a true project-wide average of all functions.
                        # Let's refine static.py to return sum and count for CC.
                        # ---
                        # Re-evaluating: static.py's calculate_cyclomatic_complexity already returns an average
                        # for the file. We can average these file-level averages.
                        file_avg_cc = calculate_cyclomatic_complexity(content)
                        if file_avg_cc > 0: # only add to sum if functions were found and analyzed
                           total_cyclomatic_complexity_sum_for_avg += file_avg_cc
                           analyzed_functions_or_blocks_for_cc_avg +=1
                        
                        long_elements_in_file = find_long_elements(content, max_function_lines)
                        for el in long_elements_in_file:
                            # Add relative file path for better reporting
                            el['file'] = os.path.relpath(file_path, project_path)
                            long_functions_found_list.append(el)
                        
                        eval_findings_in_file = check_for_eval_usage(content)
                        for finding in eval_findings_in_file:
                            finding['file'] = os.path.relpath(file_path, project_path)
                            eval_usage_findings_list.append(finding)

                        large_classes_in_file = find_large_classes(content, max_class_methods)
                        for lc_finding in large_classes_in_file:
                            lc_finding['file'] = os.path.relpath(file_path, project_path)
                            large_classes_found_list.append(lc_finding)
                        
                        secret_findings_in_file = check_for_hardcoded_secrets(content)
                        for finding in secret_findings_in_file:
                            finding['file'] = os.path.relpath(file_path, project_path)
                            hardcoded_secrets_findings_list.append(finding)
                        
                        imports_in_file = analyze_imports(content, file_path_str=file_path)
                        for imp_finding in imports_in_file:
                            imp_finding['file'] = os.path.relpath(file_path, project_path)
                            all_project_imports.append(imp_finding)
                        
                        unused_imports_in_file = find_unused_imports(content, file_path_str=file_path)
                        for unused_imp_finding in unused_imports_in_file:
                            unused_imp_finding['file'] = os.path.relpath(file_path, project_path)
                            unused_imports_findings_list.append(unused_imp_finding)

                except Exception as e:
                    print(f"Warning: Could not read or analyze file {file_path}: {e}")
                    status_message = "Analysis completed with some errors (see warnings)."

    # avg_cognitive_complexity = total_cognitive_complexity / python_files_count if python_files_count > 0 else 0 # Temporarily removed
    avg_cyclomatic_complexity = total_cyclomatic_complexity_sum_for_avg / analyzed_functions_or_blocks_for_cc_avg if analyzed_functions_or_blocks_for_cc_avg > 0 else 0

    # Placeholder TES components
    # These will be replaced by actual calculations in later streams
    mutation_score_val = 0.0
    assertion_density_val = 0.0 # Raw assertion count per test, e.g. 3.0
    
    total_user_stories = 0
    covered_user_stories = 0 # Placeholder for now, will be calculated with mapping
    
    if user_stories_file_path:
        try:
            with open(user_stories_file_path, "r", encoding="utf-8") as f:
                stories = [line.strip() for line in f if line.strip()]
                total_user_stories = len(stories)
            if total_user_stories == 0:
                 print(f"Warning: User stories file '{user_stories_file_path}' is empty or contains only whitespace.")
        except FileNotFoundError:
            print(f"Warning: User stories file '{user_stories_file_path}' not found.")
        except Exception as e:
            print(f"Warning: Could not read user stories file '{user_stories_file_path}': {e}")

    behavior_coverage_val = (covered_user_stories / total_user_stories) if total_user_stories > 0 else 0.0
    
    # Test Execution (silent)
    pytest_results = run_pytest(actual_test_path)
    
    # For now, use total duration for speed factor calculation.
    # This needs refinement if we want per-test average.
    # Assume number of tests run is 1 if not parsed, or a placeholder.
    # For simplicity, if tests ran, use duration. If not, avg_ms is high.
    num_tests_run_placeholder = 1 # Placeholder, to be replaced by actual parsing from pytest_results['summary']
    
    # Use actual duration from pytest if available and successful
    if pytest_results["success"] and pytest_results["duration_seconds"] > 0:
        # This is total duration. For avg_ms, we'd need num_tests.
        # For now, let's assume TES speed_factor expects avg_ms of the *suite* if only one "test" (the suite run)
        # or we need to parse individual test times.
        # Let's use total duration as a proxy for now, and assume 1 "test item" for the suite.
        avg_test_execution_time_ms_val = (pytest_results["duration_seconds"] * 1000) / num_tests_run_placeholder
    else:
        avg_test_execution_time_ms_val = 20000 # Default to a high value (20s) if tests didn't run or no duration

    actual_speed_factor = 1 / (1 + (avg_test_execution_time_ms_val / 100)) if avg_test_execution_time_ms_val >=0 else 0
    actual_speed_factor = max(0, min(actual_speed_factor, 1.0)) # Clamp to 0-1

    tes_score_calculated = calculate_tes(
        mutation_score=mutation_score_val,
        assertion_density=assertion_density_val,
        behavior_coverage=behavior_coverage_val,
        speed_factor=actual_speed_factor, # Use the calculated one directly
        # avg_test_execution_time_ms is handled by speed_factor now
    )
    tes_grade_calculated = get_tes_grade(tes_score_calculated)

    # E-TES v2.0 calculation (if enabled)
    etes_score_calculated = None
    etes_grade_calculated = None
    etes_components = None
    etes_comparison = None

    if use_etes_v2:
        # Calculate E-TES v2.0 silently

        # Prepare comprehensive test suite data for E-TES
        test_suite_data = {
            'mutation_score': mutation_score_val,
            'avg_test_execution_time_ms': avg_test_execution_time_ms_val,
            'assertions': [
                # Placeholder assertions - would be extracted from actual tests
                {'type': 'equality', 'code': 'assert x == y', 'target_criticality': 1.0}
            ],
            'covered_behaviors': [],  # Would be mapped from user stories
            'execution_results': [
                {'passed': pytest_results["success"], 'execution_time_ms': avg_test_execution_time_ms_val}
            ],
            'determinism_score': 1.0,  # Placeholder
            'stability_score': 1.0,    # Placeholder
            'readability_score': 1.0,  # Placeholder
            'independence_score': 1.0, # Placeholder
        }

        # Prepare codebase data for E-TES
        codebase_data = {
            'all_behaviors': [],  # Would be extracted from user stories
            'behavior_criticality': {},
            'complexity_metrics': {
                'avg_cyclomatic_complexity': avg_cyclomatic_complexity,
                'total_loc': total_loc
            }
        }

        # Calculate E-TES v2.0
        etes_score_calculated, etes_components = calculate_etes_v2(
            test_suite_data, codebase_data
        )
        etes_grade_calculated = get_etes_grade(etes_score_calculated)

        # Compare TES vs E-TES
        etes_comparison = compare_tes_vs_etes(
            tes_score_calculated, etes_score_calculated, etes_components
        )

    # Security analysis (silent)
    security_scan_results_obj = check_dependencies_vulnerabilities()
    vulnerability_details_list = []
    vulnerability_count = 0
    security_message = "Dependency check completed."

    if security_scan_results_obj.get("error"):
        security_message = f"Dependency check error: {security_scan_results_obj['error']}"
    elif security_scan_results_obj.get("details"):
        vulnerability_details_list = security_scan_results_obj["details"]
        vulnerability_count = len(vulnerability_details_list)
        if vulnerability_count > 0:
            security_message = f"Found {vulnerability_count} vulnerabilities in dependencies."
        else:
            security_message = "No vulnerabilities found in dependencies."

    # Design Debt: Circular Dependencies (silent analysis)
    import_graph = build_import_graph(project_path)
    circular_dependencies_found = find_circular_dependencies(import_graph)


    results = {
        "project_path": project_path,
        "status": "analysis_partial" if python_files_count > 0 else "analysis_no_python_files",
        "message": status_message,
        "tes_score": tes_score_calculated,
        "tes_grade": tes_grade_calculated,
        "etes_v2_enabled": use_etes_v2,
        "etes_score": etes_score_calculated,
        "etes_grade": etes_grade_calculated,
        "metrics": {
            "total_lines_of_code_python": total_loc,
            "python_files_analyzed": python_files_count,
            "average_cyclomatic_complexity": avg_cyclomatic_complexity,
            "long_functions_count": len(long_functions_found_list),
            "large_classes_count": len(large_classes_found_list),
            "total_imports_found": len(all_project_imports),
            "unique_imported_modules_count": len(set(imp['module'] for imp in all_project_imports if imp.get('module'))),
            "unused_imports_count": len(unused_imports_findings_list),
            "circular_dependencies_count": len(circular_dependencies_found), # New metric
        },
        "test_execution_summary": {
            "pytest_ran_successfully": pytest_results["success"],
            "pytest_exit_code": pytest_results["exit_code"],
            "pytest_duration_seconds": pytest_results["duration_seconds"],
        },
        "security_analysis": {
            "dependency_vulnerabilities_count": vulnerability_count,
            "dependency_check_message": security_message,
            "eval_usage_count": len(eval_usage_findings_list),
            "hardcoded_secrets_count": len(hardcoded_secrets_findings_list),
        },
        "details": {
             "long_functions_list": long_functions_found_list[:10],
             "vulnerability_details_list": vulnerability_details_list[:10],
             "eval_usage_details_list": eval_usage_findings_list[:10],
             "large_classes_details_list": large_classes_found_list[:10],
             "hardcoded_secrets_details_list": hardcoded_secrets_findings_list[:10],
             "all_project_imports_list": all_project_imports[:20],
             "unused_imports_details_list": unused_imports_findings_list[:10],
             "circular_dependencies_list": circular_dependencies_found[:10], # New detail
             "pytest_stdout": pytest_results["stdout"][-1000:] if pytest_results["stdout"] else "",
             "pytest_stderr": pytest_results["stderr"]
        },
        "tes_components": {
            "mutation_score": mutation_score_val,
            "assertion_density_raw": assertion_density_val,
            "total_user_stories": total_user_stories,
            "covered_user_stories": covered_user_stories,
            "behavior_coverage_calculated": behavior_coverage_val,
            "avg_test_execution_time_ms": avg_test_execution_time_ms_val,
            "speed_factor_calculated": actual_speed_factor,
        },
        "etes_components": {
            "mutation_score": etes_components.mutation_score if etes_components else None,
            "evolution_gain": etes_components.evolution_gain if etes_components else None,
            "assertion_iq": etes_components.assertion_iq if etes_components else None,
            "behavior_coverage": etes_components.behavior_coverage if etes_components else None,
            "speed_factor": etes_components.speed_factor if etes_components else None,
            "quality_factor": etes_components.quality_factor if etes_components else None,
            "insights": etes_components.insights if etes_components else [],
            "calculation_time": etes_components.calculation_time if etes_components else None,
        } if use_etes_v2 else None,
        "etes_comparison": etes_comparison if use_etes_v2 else None
    }
    
    results["has_critical_issues"] = (
        vulnerability_count > 0 or
        len(eval_usage_findings_list) > 0 or
        len(hardcoded_secrets_findings_list) > 0 or
        len(circular_dependencies_found) > 0 or # Add circular dependencies to critical issues
        (pytest_results["exit_code"] not in [0, 1, 5, -1, -2])
    )
        
    return results

def main():
    """Enhanced main function using professional CLI interface"""
    parser = argparse.ArgumentParser(
        description="Guardian Code Quality & Security Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  guardian ./my_project                                           # Basic analysis
  guardian ./my_project --run-quality --quality-mode etes_v2     # E-TES v2.0 analysis
  guardian ./my_project --run-quality --quality-mode betes_v3.1  # bE-TES v3.1 analysis
  guardian ./my_project --run-quality --quality-mode betes_v3.1 --smooth-sigmoid all # bE-TES v3.1 with sigmoids
  guardian ./my_project --run-quality --quality-mode osqi_v1     # OSQI v1.0 analysis
  guardian ./my_project --output-format json                      # JSON output
  guardian ./my_project --verbose --no-color                      # Verbose without colors
        """
    )

    parser.add_argument("project_path", help="The path to the project directory to analyze.")
    parser.add_argument(
        "--user-stories-file",
        type=str,
        default=None,
        help="Path to a file containing user stories (one per line)."
    )
    parser.add_argument(
        "--max-function-lines",
        type=int,
        default=20,
        help="Maximum allowed lines for a function/method (default: 20)."
    )
    parser.add_argument(
        "--max-class-methods",
        type=int,
        default=10,
        help="Maximum allowed methods in a class (default: 10)."
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to run pytest on (defaults to project_path if not specified)."
    )
    parser.add_argument(
        "--output-format",
        choices=["human", "json"],
        default="human",
        help="Output format for the analysis results (default: human)."
    )
    # --use-etes-v2 is deprecated in favor of --quality-mode
    # parser.add_argument(
    #     "--use-etes-v2",
    #     action="store_true",
    #     help="Enable E-TES v2.0 evolutionary test effectiveness scoring (experimental)."
    # )

    # New arguments for quality scoring
    quality_group = parser.add_argument_group('Quality Scoring (bE-TES v3.0 and E-TES v2.0)')
    quality_group.add_argument(
        "--quality-mode",
        choices=["etes_v2", "betes_v3", "betes_v3.1", "osqi_v1"],
        default="etes_v2",
        help="Select the quality scoring mode (default: etes_v2 if --run-quality is used)."
    )
    quality_group.add_argument(
        "--smooth-sigmoid",
        type=str,
        default=None,
        help="Enable sigmoid normalization for bE-TES M' and/or E' (for betes_v3.1 mode). "
             "Use 'm', 'e', 'm,e', or 'all'. Example: --smooth-sigmoid m,e"
    )
    quality_group.add_argument(
        "--risk-class",
        type=str,
        default=None,
        help="Risk class from risk_classes.yml to evaluate bE-TES score against (e.g., standard_saas)."
    )
    # bE-TES Weights
    betes_weights_group = parser.add_argument_group('bE-TES v3.0 Weights (default: 1.0 each)')
    betes_weights_group.add_argument("--betes-w-m", type=float, help="Weight for Mutation Score (M')")
    betes_weights_group.add_argument("--betes-w-e", type=float, help="Weight for EMT Gain (E')")
    betes_weights_group.add_argument("--betes-w-a", type=float, help="Weight for Assertion IQ (A')")
    betes_weights_group.add_argument("--betes-w-b", type=float, help="Weight for Behaviour Coverage (B')")
    betes_weights_group.add_argument("--betes-w-s", type=float, help="Weight for Speed Factor (S')")

    # Sensor paths (examples, more might be needed)
    sensor_paths_group = parser.add_argument_group('Sensor Data Paths (for bE-TES v3.0)')
    sensor_paths_group.add_argument("--test-root-path", type=str, help="Root path for test files (for AssertionIQ sensor). Default: tests/")
    sensor_paths_group.add_argument("--coverage-file-path", type=str, help="Path to coverage file (e.g., coverage.info, coverage.xml). Default: coverage.info")
    sensor_paths_group.add_argument("--critical-behaviors-manifest-path", type=str, help="Path to critical behaviors manifest YAML.")
    sensor_paths_group.add_argument("--ci-platform", type=str, help="CI platform for flake rate (e.g., gh-actions).")
    sensor_paths_group.add_argument("--pytest-reportlog-path", type=str, help="Path to pytest reportlog JSONL file (for speed sensor).")
    sensor_paths_group.add_argument("--project-ci-identifier", type=str, help="Project identifier for CI API (e.g., org/repo, for flakiness sensor).")
    
    # Argument for OSQI's CHS sensor
    parser.add_argument(
        "--project-language",
        type=str,
        default="python",
        help="Primary programming language of the project (e.g., python, javascript), used for OSQI CHS normalization (default: python)."
    )
    # Add a flag to explicitly run quality analysis if it's not always part of the main analysis
    parser.add_argument(
        "--run-quality",
        action="store_true",
        help="Run quality scoring (E-TES v2.0 or bE-TES v3.0 based on --quality-mode)."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed information."
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output for better compatibility."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if critical issues are found."
    )
    parser.add_argument(
        "--tests",
        nargs="+", # Accepts one or more arguments
        type=str,
        default=None,
        help="Specify a list of test files or node IDs to run (e.g., path/to/test_one.py tests/test_another.py::test_specific)."
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Specify the username for history tracking. Defaults to system user or 'default_guardian_user'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file for Guardian settings (e.g., sensor configurations)."
    )
 
    args = parser.parse_args()

    # Use professional CLI interface
    return run_analysis(args)


def run_analysis(args):
    """Run analysis using clean CLI interface"""
    # Ensure logger is accessible in this function's scope, referencing the module-level one.
    # This is mostly defensive, as module-level 'logger' should already be accessible.
    logger = logging.getLogger(__name__)
    try:
        # Setup clean logging (no Python artifacts in output)
        setup_logging(getattr(args, 'verbose', False))

        # Create analyzer and formatter
        analyzer = create_analyzer(args)
        formatter = create_formatter(args)

        # Run analysis silently for clean output
        results = analyzer.analyze_project(
            project_path=args.project_path,
            test_path=args.test_path,
            user_stories_file=args.user_stories_file,
            specific_test_targets=args.tests # Pass the new argument
        )
        
        if results is None: # Analyzer might return None on critical error
            print("Error: Analysis failed to produce results.", file=sys.stderr)
            return 1

        # --- Quality Scoring Logic (Refactored) ---
        if args.run_quality:
            # Use QualityScoringHandler to reduce complexity
            scoring_handler = QualityScoringHandler(args.project_path)
            
            # Create quality configuration
            quality_cfg = scoring_handler.create_quality_config(args)
            
            # Load mutation configuration
            mutation_config = scoring_handler.load_mutation_config(args.config)
            
            # Collect metrics and calculate score based on mode
            raw_metrics_betes_dict = None
            test_suite_data_for_etes_v2 = None
            codebase_data_for_etes_v2 = None
            previous_score_for_etes_v2 = None
            
            if quality_cfg.mode in ["betes_v3.1", "osqi_v1"]:
                # Collect bE-TES raw metrics
                raw_metrics_betes_dict = scoring_handler.collect_betes_metrics(
                    quality_cfg,
                    mutation_config,
                    test_targets=args.tests
                )
            elif quality_cfg.mode == "etes_v2":
                # Prepare E-TES v2 data
                test_suite_data_for_etes_v2, codebase_data_for_etes_v2 = (
                    scoring_handler.prepare_etes_v2_data(results)
                )
            
            # Calculate quality score
            quality_score, quality_components_obj = scoring_handler.calculate_quality_score(
                quality_cfg,
                raw_metrics_betes_dict,
                test_suite_data_for_etes_v2,
                codebase_data_for_etes_v2,
                previous_score_for_etes_v2,
                project_language=args.project_language if quality_cfg.mode == "osqi_v1" else "python"
            )
            
            # Update results with mutation score if available
            if raw_metrics_betes_dict and "raw_mutation_score" in raw_metrics_betes_dict:
                if "tes_components" not in results:
                    results["tes_components"] = {}
                results["tes_components"]["mutation_score"] = (
                    raw_metrics_betes_dict["raw_mutation_score"]
                )
            
            # Store quality analysis results
            results["quality_analysis"] = {
                "mode": quality_cfg.mode,
                "score": round(quality_score, 3),
                "grade": get_etes_grade(quality_score),
                "components": (
                    quality_components_obj.__dict__
                    if hasattr(quality_components_obj, '__dict__')
                    else str(quality_components_obj)
                )
            }
            
            # Apply risk classification if configured
            scoring_handler.apply_risk_classification(
                quality_cfg,
                quality_score,
                results
            )
            
            # Check if classification verdict requires strict mode failure
            if args.strict:
                classification = results.get("quality_analysis", {}).get("classification", {})
                if classification.get("verdict") == "FAIL":
                    results["has_critical_issues"] = True
            

        # Format and output results cleanly
        if args.output_format == 'json':
            try:
                output = formatter.format_analysis_results(results, 'json')
            except TypeError as e:
                print(f"Error formatting results to JSON: {e}. Ensure all components are serializable.", file=sys.stderr)
                output = json.dumps({"error": "Result serialization failed", "results_preview": str(results)[:1000]}, indent=2)
        else:
            # OutputFormatter._print_human_readable now prints directly to console
            # So, we just call it and it doesn't return a string to be printed here.
            formatter.format_analysis_results(results, 'human')
            # output variable would be None here, so no need to print it.

        # Determine exit code
        if results.get('status', '').startswith('analysis_failed'):
            return 1
        elif results.get('has_critical_issues', False):
            return 2 if getattr(args, 'strict', False) else 0
        else:
            return 0

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Attempt to record the run if quality analysis was performed
        if args.run_quality and 'results' in locals() and results:
            try:
                history_manager = HistoryManager()
                
                current_run_metrics = {}
                if results.get("quality_analysis"): # Check if quality_analysis block exists
                    # Ensure tes_components exists before trying to get mutation_score
                    tes_components = results.get("tes_components", {})
                    current_run_metrics["mutation_score"] = tes_components.get("mutation_score", 0.0)
                    
                    quality_analysis_data = results.get("quality_analysis", {})
                    current_run_metrics["betes_score"] = quality_analysis_data.get("score") if quality_analysis_data.get("mode") == "betes_v3.1" else None
                    current_run_metrics["osqi_score"] = quality_analysis_data.get("score") if quality_analysis_data.get("mode") == "osqi_v1" else None
                
                # For delta_metrics, we'd need to compare with a previous state.
                # This is complex for a single CLI run without more history context here.
                # For now, pass None or empty for delta_metrics.
                
                actions_taken_for_history = ["cmd_quality_analysis", f"mode_{args.quality_mode}"]
                if args.tests:
                    actions_taken_for_history.append("targeted_tests")

                # The username argument for record_run in HistoryManager handles fetching/creating the player.
                # We don't need to pass a username from here if we want the default behavior.
                # The args.user from the CLI was for the (now removed) Typer command, not argparse.
                history_manager.record_run(
                    username=args.user, # Pass the username from CLI args
                    command_name="quality",
                    current_metrics=current_run_metrics,
                    delta_metrics=None,
                    actions_taken=actions_taken_for_history,
                    xp_bonus=10
                )
                history_manager.close()
                logger.info("Quality analysis run recorded in history.")
            except Exception as e_hist_run_analysis:
                logger.warning(f"Could not record 'quality' run to history: {e_hist_run_analysis}", exc_info=False)


# Keep the original analyze_project function for backward compatibility
analyze_project = analyze_project  # Reference to the existing function

if __name__ == "__main__":
    # main() # Old argparse entry point - TEMPORARILY RE-ENABLED FOR TESTING
    app() # New Typer entry point - TEMPORARILY DISABLED
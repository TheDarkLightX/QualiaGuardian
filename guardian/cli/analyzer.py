"""
Guardian Project Analyzer

Clean, modular project analysis with proper error handling and logging.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

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
from guardian.core.tes import calculate_etes_v2, get_etes_grade, compare_tes_vs_etes # Removed calculate_tes, get_tes_grade
from guardian.core.etes import QualityConfig # Renamed ETESConfig to QualityConfig

logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    """
    Professional project analyzer with clean architecture and proper error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_function_lines = self.config.get('max_function_lines', 20)
        self.max_class_methods = self.config.get('max_class_methods', 10)
        self.use_etes_v2 = self.config.get('use_etes_v2', False)
        
        # Analysis state
        self.total_files_analyzed = 0
        self.errors_encountered = []
        self.warnings_encountered = []
    
    def analyze_project(self, project_path: str, test_path: Optional[str] = None,
                       user_stories_file: Optional[str] = None,
                       specific_test_targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a project and return comprehensive results.

        Args:
            project_path: Path to the project directory.
            test_path: Optional path to test directory.
            user_stories_file: Optional path to user stories file.
            specific_test_targets: Optional list of specific test targets to run.
            
        Returns:
            Dictionary containing analysis results.
        """
        logger.debug(f"Starting analysis of project: {project_path}")
        
        # Validate inputs
        validation_result = self._validate_inputs(project_path, test_path, user_stories_file)
        if validation_result['has_errors']:
            return validation_result
        
        try:
            # Initialize results structure
            results = self._initialize_results(project_path)
            
            # Perform code analysis
            code_metrics = self._analyze_codebase(project_path)
            results.update(code_metrics)
            
            # Perform security analysis
            security_results = self._analyze_security(project_path)
            results['security_analysis'] = security_results
            
            # Analyze user stories (if provided)
            user_story_results = self._analyze_user_stories(user_stories_file)
            results.update(user_story_results)
            
            # Run tests
            test_results = self._run_tests(test_path or project_path, specific_test_targets=specific_test_targets)
            results['test_execution_summary'] = test_results
            
            # Calculate TES score
            tes_results = self._calculate_tes_score(results)
            results.update(tes_results)
            
            # Calculate E-TES v2.0 (if enabled)
            if self.use_etes_v2:
                etes_results = self._calculate_etes_score(results)
                results.update(etes_results)
            
            # Determine critical issues
            results['has_critical_issues'] = self._has_critical_issues(results)
            
            # Add analysis metadata
            results['analysis_metadata'] = {
                'files_analyzed': self.total_files_analyzed,
                'errors_count': len(self.errors_encountered),
                'warnings_count': len(self.warnings_encountered),
                'etes_v2_enabled': self.use_etes_v2
            }
            
            logger.debug(f"Analysis completed successfully. Files analyzed: {self.total_files_analyzed}")
            return results
            
        except Exception as e:
            logger.error(f"Critical error during analysis: {e}", exc_info=True)
            return {
                'status': 'analysis_failed',
                'error': str(e),
                'has_critical_issues': True,
                'project_path': project_path
            }
    
    def _validate_inputs(self, project_path: str, test_path: Optional[str], 
                        user_stories_file: Optional[str]) -> Dict[str, Any]:
        """Validate input parameters"""
        errors = []
        
        # Check project path
        if not os.path.exists(project_path):
            errors.append(f"Project path does not exist: {project_path}")
        elif not os.path.isdir(project_path):
            errors.append(f"Project path is not a directory: {project_path}")
        
        # Check test path (if provided)
        if test_path and not os.path.exists(test_path):
            self.warnings_encountered.append(f"Test path does not exist: {test_path}")
        
        # Check user stories file (if provided)
        if user_stories_file and not os.path.exists(user_stories_file):
            self.warnings_encountered.append(f"User stories file does not exist: {user_stories_file}")
        
        if errors:
            return {
                'status': 'validation_failed',
                'errors': errors,
                'has_errors': True,
                'project_path': project_path
            }
        
        return {'has_errors': False}
    
    def _initialize_results(self, project_path: str) -> Dict[str, Any]:
        """Initialize results structure"""
        return {
            'project_path': project_path,
            'status': 'analysis_in_progress',
            'tes_score': 0.0,
            'tes_grade': 'F',
            'metrics': {},
            'details': {},
            'tes_components': {},
            'etes_v2_enabled': self.use_etes_v2
        }
    
    def _analyze_codebase(self, project_path: str) -> Dict[str, Any]:
        """Analyze codebase for metrics and code quality"""
        logger.debug("Analyzing codebase metrics...")
        
        metrics = {
            'total_lines_of_code_python': 0,
            'python_files_analyzed': 0,
            'average_cyclomatic_complexity': 0.0,
            'long_functions_count': 0,
            'large_classes_count': 0,
            'unused_imports_count': 0,
            'circular_dependencies_count': 0
        }
        
        details = {
            'long_functions_list': [],
            'large_classes_details_list': [],
            'unused_imports_details_list': [],
            'circular_dependencies_list': []
        }
        
        complexity_sum = 0.0
        complexity_count = 0
        
        # Walk through Python files
        for root, _, files in os.walk(project_path):
            # Skip virtual environments and git directories
            if any(skip_dir in root for skip_dir in ['.venv', '.git', '__pycache__', '.pytest_cache']):
                continue
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                
                try:
                    self._analyze_single_file(file_path, relative_path, metrics, details, 
                                            complexity_sum, complexity_count)
                    self.total_files_analyzed += 1
                    
                except Exception as e:
                    error_msg = f"Error analyzing {relative_path}: {str(e)}"
                    self.errors_encountered.append(error_msg)
                    logger.warning(error_msg)
        
        # Calculate average complexity
        if complexity_count > 0:
            metrics['average_cyclomatic_complexity'] = complexity_sum / complexity_count
        
        # Analyze import dependencies
        try:
            import_graph = build_import_graph(project_path)
            circular_deps = find_circular_dependencies(import_graph)
            metrics['circular_dependencies_count'] = len(circular_deps)
            details['circular_dependencies_list'] = circular_deps[:10]  # Limit to 10
        except Exception as e:
            error_msg = f"Error analyzing dependencies: {str(e)}"
            self.errors_encountered.append(error_msg)
            logger.warning(error_msg)
        
        metrics['python_files_analyzed'] = self.total_files_analyzed
        
        return {
            'metrics': metrics,
            'details': details,
            'status': 'analysis_complete' if not self.errors_encountered else 'analysis_partial'
        }
    
    def _analyze_single_file(self, file_path: str, relative_path: str, 
                           metrics: Dict[str, Any], details: Dict[str, Any],
                           complexity_sum: float, complexity_count: int):
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count lines of code
        loc = count_lines_of_code(content)
        metrics['total_lines_of_code_python'] += loc
        
        # Calculate complexity
        file_complexity = calculate_cyclomatic_complexity(content)
        if file_complexity > 0:
            complexity_sum += file_complexity
            complexity_count += 1
        
        # Find long functions
        long_functions = find_long_elements(content, self.max_function_lines)
        for func in long_functions:
            func['file'] = relative_path
            details['long_functions_list'].append(func)
        metrics['long_functions_count'] += len(long_functions)
        
        # Find large classes
        large_classes = find_large_classes(content, self.max_class_methods)
        for cls in large_classes:
            cls['file'] = relative_path
            details['large_classes_details_list'].append(cls)
        metrics['large_classes_count'] += len(large_classes)
        
        # Find unused imports
        unused_imports = find_unused_imports(content, file_path)
        for imp in unused_imports:
            imp['file'] = relative_path
            details['unused_imports_details_list'].append(imp)
        metrics['unused_imports_count'] += len(unused_imports)
    
    def _analyze_security(self, project_path: str) -> Dict[str, Any]:
        """Perform security analysis"""
        logger.debug("Performing security analysis...")
        
        security_results = {
            'dependency_vulnerabilities_count': 0,
            'dependency_check_message': 'Not checked',
            'eval_usage_count': 0,
            'hardcoded_secrets_count': 0
        }
        
        # Check dependency vulnerabilities
        try:
            vuln_results = check_dependencies_vulnerabilities()
            if vuln_results.get('error'):
                security_results['dependency_check_message'] = f"Error: {vuln_results['error']}"
            else:
                vulns = vuln_results.get('details', [])
                security_results['dependency_vulnerabilities_count'] = len(vulns)
                security_results['dependency_check_message'] = (
                    f"Found {len(vulns)} vulnerabilities" if vulns else "No vulnerabilities found"
                )
        except Exception as e:
            error_msg = f"Error checking dependencies: {str(e)}"
            self.errors_encountered.append(error_msg)
            logger.warning(error_msg)
        
        # Check for eval usage and hardcoded secrets
        eval_findings = []
        secret_findings = []
        
        for root, _, files in os.walk(project_path):
            if any(skip_dir in root for skip_dir in ['.venv', '.git', '__pycache__']):
                continue
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for eval usage
                    eval_uses = check_for_eval_usage(content)
                    for eval_use in eval_uses:
                        eval_use['file'] = relative_path
                        eval_findings.append(eval_use)
                    
                    # Check for hardcoded secrets
                    secrets = check_for_hardcoded_secrets(content)
                    for secret in secrets:
                        secret['file'] = relative_path
                        secret_findings.append(secret)
                
                except Exception as e:
                    error_msg = f"Error checking security in {relative_path}: {str(e)}"
                    self.errors_encountered.append(error_msg)
                    logger.warning(error_msg)
        
        security_results['eval_usage_count'] = len(eval_findings)
        security_results['hardcoded_secrets_count'] = len(secret_findings)
        
        return security_results
    
    def _analyze_user_stories(self, user_stories_file: Optional[str]) -> Dict[str, Any]:
        """Analyze user stories file"""
        if not user_stories_file:
            return {
                'total_user_stories': 0,
                'covered_user_stories': 0
            }
        
        try:
            with open(user_stories_file, 'r', encoding='utf-8') as f:
                stories = [line.strip() for line in f if line.strip()]
            
            return {
                'total_user_stories': len(stories),
                'covered_user_stories': 0  # Placeholder - would need test mapping
            }
        
        except Exception as e:
            error_msg = f"Error reading user stories: {str(e)}"
            self.errors_encountered.append(error_msg)
            logger.warning(error_msg)
            return {
                'total_user_stories': 0,
                'covered_user_stories': 0
            }
    
    def _run_tests(self, test_path: str, specific_test_targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run tests using pytest, optionally filtering by specific targets."""
        logger.debug(f"Running tests in: {test_path}, specific targets: {specific_test_targets}")
        
        try:
            pytest_results = run_pytest(target_path=test_path, test_targets=specific_test_targets)
            return {
                'pytest_ran_successfully': pytest_results.get('success', False),
                'pytest_exit_code': pytest_results.get('exit_code', -1),
                'pytest_duration_seconds': pytest_results.get('duration_seconds', 0.0)
            }
        except Exception as e:
            error_msg = f"Error running tests: {str(e)}"
            self.errors_encountered.append(error_msg)
            logger.warning(error_msg)
            return {
                'pytest_ran_successfully': False,
                'pytest_exit_code': -1,
                'pytest_duration_seconds': 0.0
            }
    
    def _calculate_tes_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate TES score"""
        logger.debug("Calculating TES score...")
        
        # Extract components
        total_stories = results.get('total_user_stories', 0)
        covered_stories = results.get('covered_user_stories', 0)
        
        behavior_coverage = covered_stories / total_stories if total_stories > 0 else 0.0
        
        # Calculate speed factor
        duration = results.get('test_execution_summary', {}).get('pytest_duration_seconds', 20.0)
        avg_test_time_ms = duration * 1000  # Convert to ms
        speed_factor = 1 / (1 + (avg_test_time_ms / 100))
        speed_factor = max(0, min(speed_factor, 1.0))
        
        # Calculate TES (with placeholder values for mutation score and assertion density)
        mutation_score = 0.0  # Placeholder
        assertion_density = 0.0  # Placeholder
        
        # calculate_tes was a placeholder and has been removed.
        # This legacy TES calculation is no longer actively supported.
        # Returning default/placeholder values.
        # The main quality scoring is now handled by calculate_quality_score in cli.py
        tes_score = 0.0
        tes_grade = "N/A" # Or use get_etes_grade(0.0) if a grade is strictly needed
        
        return {
            'tes_score': tes_score,
            'tes_grade': tes_grade, # get_etes_grade(tes_score) can be used if a letter grade is desired
            'tes_components': {
                'mutation_score': mutation_score,
                'assertion_density_raw': assertion_density,
                'behavior_coverage_calculated': behavior_coverage,
                'speed_factor_calculated': speed_factor,
                'total_user_stories': total_stories,
                'covered_user_stories': covered_stories,
                'avg_test_execution_time_ms': avg_test_time_ms
            }
        }
    
    def _calculate_etes_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate E-TES v2.0 score"""
        logger.debug("Calculating E-TES v2.0 score...")
        
        try:
            # Prepare test suite data
            tes_components = results.get('tes_components', {})
            test_execution = results.get('test_execution_summary', {})
            
            test_suite_data = {
                'mutation_score': tes_components.get('mutation_score', 0.0),
                'avg_test_execution_time_ms': tes_components.get('avg_test_execution_time_ms', 1000.0),
                'assertions': [
                    {'type': 'equality', 'code': 'placeholder', 'target_criticality': 1.0}
                ],
                'covered_behaviors': [],
                'execution_results': [
                    {
                        'passed': test_execution.get('pytest_ran_successfully', False),
                        'execution_time_ms': tes_components.get('avg_test_execution_time_ms', 1000.0)
                    }
                ],
                'determinism_score': 1.0,
                'stability_score': 1.0,
                'readability_score': 1.0,
                'independence_score': 1.0
            }
            
            # Prepare codebase data
            metrics = results.get('metrics', {})
            codebase_data = {
                'all_behaviors': [],
                'behavior_criticality': {},
                'complexity_metrics': {
                    'avg_cyclomatic_complexity': metrics.get('average_cyclomatic_complexity', 1.0),
                    'total_loc': metrics.get('total_lines_of_code_python', 0)
                }
            }
            
            # Calculate E-TES
            etes_score, etes_components = calculate_etes_v2(test_suite_data, codebase_data)
            etes_grade = get_etes_grade(etes_score)
            
            # Compare with TES
            tes_score = results.get('tes_score', 0.0)
            comparison = compare_tes_vs_etes(tes_score, etes_score, etes_components)
            
            return {
                'etes_score': etes_score,
                'etes_grade': etes_grade,
                'etes_components': {
                    'mutation_score': etes_components.mutation_score,
                    'evolution_gain': etes_components.evolution_gain,
                    'assertion_iq': etes_components.assertion_iq,
                    'behavior_coverage': etes_components.behavior_coverage,
                    'speed_factor': etes_components.speed_factor,
                    'quality_factor': etes_components.quality_factor,
                    'insights': etes_components.insights,
                    'calculation_time': etes_components.calculation_time
                },
                'etes_comparison': comparison
            }
        
        except Exception as e:
            error_msg = f"Error calculating E-TES: {str(e)}"
            self.errors_encountered.append(error_msg)
            logger.warning(error_msg)
            return {
                'etes_score': 0.0,
                'etes_grade': 'F',
                'etes_components': None,
                'etes_comparison': None
            }
    
    def _has_critical_issues(self, results: Dict[str, Any]) -> bool:
        """Determine if there are critical issues"""
        security = results.get('security_analysis', {})
        test_execution = results.get('test_execution_summary', {})
        metrics = results.get('metrics', {})

        return (
            security.get('dependency_vulnerabilities_count', 0) > 0 or
            security.get('eval_usage_count', 0) > 0 or
            security.get('hardcoded_secrets_count', 0) > 0 or
            metrics.get('circular_dependencies_count', 0) > 0 or
            not test_execution.get('pytest_ran_successfully', False) or
            len(self.errors_encountered) > 0
        )


def main():
    """Enhanced CLI entry point with professional output formatting"""
    import argparse
    import sys
    from guardian.cli.output_formatter import OutputFormatter, OutputLevel

    parser = argparse.ArgumentParser(
        description="Guardian AI Tool - Advanced Code Quality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  guardian analyze ./my_project                    # Basic analysis
  guardian analyze ./my_project --etes --security # Full E-TES analysis
  guardian analyze ./my_project --format json     # JSON output
  guardian analyze ./my_project --output report.html --format html
        """
    )

    parser.add_argument('path', help='Path to analyze')
    parser.add_argument('--format', choices=['console', 'json', 'html'],
                       default='console', help='Output format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--etes', action='store_true', help='Use E-TES v2.0 analysis')
    parser.add_argument('--security', action='store_true', help='Include security analysis')
    parser.add_argument('--progress', action='store_true', help='Show progress bars')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Configure output formatter
    formatter = OutputFormatter()
    if args.no_color:
        formatter.config.use_colors = False
    if args.progress:
        formatter.enable_progress_bars(True)
    if args.verbose:
        formatter.set_level(OutputLevel.VERBOSE)
    elif args.quiet:
        formatter.set_level(OutputLevel.QUIET)

    try:
        # Show startup message
        if not args.quiet:
            print(formatter.format_success("Guardian AI Tool - Starting Analysis"))
            if args.progress:
                print(formatter.format_progress_bar(0, 100, "Initializing"))

        # Use professional analyzer
        analyzer = ProjectAnalyzer({
            'use_etes_v2': args.etes,
            'include_security_analysis': args.security,
            'output_format': args.format,
            'verbose': args.verbose
        })

        # Show progress
        if args.progress and not args.quiet:
            print(formatter.format_progress_bar(25, 100, "Analyzing code"))

        # Perform analysis
        results = analyzer.analyze_project(args.path)

        # Show progress
        if args.progress and not args.quiet:
            print(formatter.format_progress_bar(75, 100, "Generating report"))

        # Format output based on requested format
        if args.format == 'json':
            output = formatter.format_json(results)
        elif args.format == 'html':
            output = formatter.format_html(results)
        else:
            output = formatter.format_console(results)

        # Show completion
        if args.progress and not args.quiet:
            print(formatter.format_progress_bar(100, 100, "Complete"))
            print()  # New line after progress bar

        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            if not args.quiet:
                print(formatter.format_success(f"Report saved to {args.output}"))
        else:
            print(output)

    except FileNotFoundError:
        error_msg = f"Path not found: {args.path}"
        print(formatter.format_clean_error(error_msg), file=sys.stderr)
        return 1
    except PermissionError:
        error_msg = f"Permission denied accessing: {args.path}"
        print(formatter.format_clean_error(error_msg), file=sys.stderr)
        return 1
    except Exception as e:
        if args.verbose:
            # Show full error in verbose mode
            print(formatter.format_error(str(e)), file=sys.stderr)
        else:
            # Show clean error in normal mode
            print(formatter.format_clean_error(str(e)), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

"""
Sensor for Code Health Score (CHS) sub-metrics.

This sensor is responsible for collecting RAW sub-metrics related to code health.
The actual normalization and calculation of the final C_HS (0-1 geometric mean)
will be handled by the OSQICalculator, using thresholds defined in
'config/chs_thresholds.yml'.
"""
import logging
import os
from typing import Dict, Any, List, Tuple, Generator, Optional # Added Optional

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

try:
    from pylint import lint
    from pylint.reporters import CollectingReporter
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

logger = logging.getLogger(__name__)

def _walk_python_files(project_path: str) -> Generator[str, None, None]:
    """Walks a project path and yields paths to Python files."""
    for root, _, files in os.walk(project_path):
        if any(skip_dir in root for skip_dir in ['.venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'build', 'dist']):
            continue
        for file_name in files:
            if file_name.endswith(".py"):
                yield os.path.join(root, file_name)

def _analyze_file_with_radon(file_path: str) -> Tuple[List[float], Optional[float]]:
    """
    Analyzes a single Python file with Radon.
    Returns a list of cyclomatic complexities for functions/methods, and a file-level Maintainability Index.
    """
    complexities: List[float] = []
    maintainability_index: Optional[float] = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cyclomatic Complexity
        cc_results = cc_visit(content)
        for item in cc_results:
            if hasattr(item, 'complexity'): # Functions, Methods, Classes (average for class)
                complexities.append(item.complexity)
        
        # Maintainability Index
        # mi_visit returns a single score for the entire file content
        maintainability_index = mi_visit(content, multi=True) # multi=True for original MI formula
        
    except Exception as e:
        logger.warning(f"Radon analysis failed for file {file_path}: {e}")
    return complexities, maintainability_index

def _analyze_file_with_pylint(file_path: str) -> int:
    """
    Analyzes a single Python file with Pylint and returns the total count of messages.
    """
    if not PYLINT_AVAILABLE:
        logger.warning("Pylint is not available. Skipping code smell count.")
        return 0
    
    try:
        reporter = CollectingReporter()
        # Run Pylint. Adjust Pylint options as needed.
        # Using a minimal set of arguments for now.
        # Consider allowing Pylint configuration via .pylintrc or CLI args.
        lint_options = [file_path, "--output-format=json"] # JSON output can be parsed if more details are needed later
        
        # Pylint's Run is a bit tricky to use programmatically for just counts easily.
        # A simpler approach for just counts might be to parse text output, but JSON is cleaner for future.
        # For now, let's use the reporter to count messages.
        
        # Temporarily disable Pylint output to stdout/stderr during programmatic run
        # unless verbose logging is on for this module.
        # This is complex to manage perfectly without deeper Pylint API usage.
        # For now, we'll let Pylint print to its usual streams if it does.
        
        linter = lint.Run(lint_options, reporter=reporter, exit=False).linter
        # linter.check(file_path) # This might be redundant if file_path is in lint_options
        
        # The reporter collects messages.
        num_messages = len(reporter.messages)
        logger.debug(f"Pylint found {num_messages} messages in {file_path}")
        return num_messages
    except Exception as e:
        logger.error(f"Pylint analysis failed for file {file_path}: {e}")
        return 0 # Or a higher penalty value if analysis failure should be penalized


def get_raw_chs_sub_metrics(project_path: str, language: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Collects raw Code Health Score sub-metrics for the project.
    Currently supports Python using Radon for Cyclomatic Complexity and Maintainability Index,
    and Pylint for code smell counts.
    """
    logger.info(f"Collecting CHS sub-metrics for project at: {project_path} (Language: {language})")
    raw_metrics: Dict[str, float] = {}

    if language.lower() == "python":
        if not RADON_AVAILABLE:
            logger.warning("Radon library is not installed. Some CHS metrics for Python will be unavailable or use defaults.")
            raw_metrics["cyclomatic_complexity"] = config.get("default_cc_if_radon_unavailable", 15.0)
            raw_metrics["maintainability_index"] = config.get("default_mi_if_radon_unavailable", 30.0)
        
        if not PYLINT_AVAILABLE:
            logger.warning("Pylint library is not installed. Code smell count will be unavailable or use default.")
            raw_metrics["num_code_smells"] = config.get("default_smells_if_pylint_unavailable", 50.0)


        all_complexities: List[float] = []
        all_maintainability_indices: List[float] = []
        total_pylint_messages = 0
        files_analyzed_count = 0

        if not os.path.isdir(project_path):
            logger.warning(f"Project path '{project_path}' is not a directory. Cannot scan for Python files.")
            return raw_metrics

        for file_path in _walk_python_files(project_path):
            files_analyzed_count += 1
            logger.debug(f"Analyzing CHS for: {file_path}")
            
            if RADON_AVAILABLE:
                complexities, mi_score = _analyze_file_with_radon(file_path)
                all_complexities.extend(complexities)
                if mi_score is not None:
                    all_maintainability_indices.append(mi_score)
            
            if PYLINT_AVAILABLE:
                total_pylint_messages += _analyze_file_with_pylint(file_path)
        
        if RADON_AVAILABLE:
            if all_complexities:
                raw_metrics["cyclomatic_complexity"] = sum(all_complexities) / len(all_complexities)
            else:
                raw_metrics["cyclomatic_complexity"] = config.get("default_cc_if_no_functions", 1.0)
            
            if all_maintainability_indices:
                raw_metrics["maintainability_index"] = sum(all_maintainability_indices) / len(all_maintainability_indices)
            else:
                raw_metrics["maintainability_index"] = config.get("default_mi_if_no_files", 50.0)
        
        if PYLINT_AVAILABLE:
            raw_metrics["num_code_smells"] = float(total_pylint_messages)
            # Could also calculate smells per KLOC if LOC is available here
        
        # Placeholder for other metrics like duplication - would require another tool
        raw_metrics["duplication_percentage"] = config.get("default_duplication_percentage", 5.0) # Placeholder

        logger.info(f"Collected Python CHS raw metrics: {raw_metrics}")

    elif language.lower() == "javascript":
        # Placeholder for JavaScript - tools like ESLint with complexity plugins, or specific tools
        logger.warning("CHS metric collection for JavaScript is currently a placeholder.")
        raw_metrics = {
            "cyclomatic_complexity": config.get("default_js_cc", 10.2),
            "duplication_percentage": config.get("default_js_dup", 6.1),
        }
    else:
        logger.warning(f"CHS sub-metric collection not implemented for language: {language}. Returning empty metrics.")
        return {}
        
    return raw_metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Example usage:
    print("Testing CHS Sensor (with Radon for Python if available)")
    
    # Create a dummy Python project for testing
    dummy_project_dir = tempfile.mkdtemp(prefix="guardian_chs_test_")
    try:
        with open(os.path.join(dummy_project_dir, "module1.py"), "w") as f:
            f.write("def func_a(x, y, z):\n")
            f.write("  if x > y:\n")
            f.write("    if y > z:\n")
            f.write("      return x + y + z\n") # CC = 3
            f.write("  return 0\n")
            f.write("\n")
            f.write("def func_b(): return 1\n") # CC = 1

        with open(os.path.join(dummy_project_dir, "module2.py"), "w") as f:
            f.write("class MyClass:\n")
            f.write("  def method_c(self, val):\n") # CC = 1
            f.write("    return val * 2\n")

        print(f"Dummy project created at: {dummy_project_dir}")
        py_metrics = get_raw_chs_sub_metrics(dummy_project_dir, "python", {})
        print(f"Python Project Raw CHS Metrics: {py_metrics}")
        # Expected CC: (3+1+1)/3 = 5/3 = 1.666...
        # Expected MI: Varies, but should be a number.

    finally:
        import shutil
        shutil.rmtree(dummy_project_dir)

    js_metrics = get_raw_chs_sub_metrics("/path/to/dummy_js_project", "javascript", {})
    print(f"JavaScript Project Raw CHS Metrics (Placeholder): {js_metrics}")

    other_metrics = get_raw_chs_sub_metrics("/path/to/other_project", "java", {})
    print(f"Other Language Project Raw CHS Metrics (Empty): {other_metrics}")
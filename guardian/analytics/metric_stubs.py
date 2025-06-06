"""
Real metric evaluator for Shapley value calculation.
This module provides functions to evaluate test contributions to quality metrics.
"""

import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Cache for test metric contributions to avoid re-computation
_metric_cache: Dict[str, float] = {}


class MetricEvaluator:
    """Evaluates test contributions to quality metrics."""
    
    def __init__(self, project_root: Path, use_cache: bool = True):
        """
        Initialize the metric evaluator.
        
        Args:
            project_root: Root directory of the project to analyze
            use_cache: Whether to use cached metric values
        """
        self.project_root = project_root
        self.use_cache = use_cache
        self._mutation_cache: Dict[str, float] = {}
        
    def evaluate_test_subset(self, selected_tests: List[Path]) -> float:
        """
        Evaluate the quality metric for a subset of tests.
        
        This function calculates the actual contribution of each test
        to the overall quality score by running mutation testing.
        
        Args:
            selected_tests: List of test files or test nodes to evaluate
            
        Returns:
            Aggregated quality score for the subset
        """
        if not selected_tests:
            return 0.0
            
        total_score = 0.0
        
        # Convert paths to strings for caching
        test_keys = [str(test) for test in selected_tests]
        
        # Check cache for already computed values
        uncached_tests = []
        for test_key in test_keys:
            if self.use_cache and test_key in _metric_cache:
                total_score += _metric_cache[test_key]
            else:
                uncached_tests.append(test_key)
        
        # Compute metrics for uncached tests
        if uncached_tests:
            new_scores = self._compute_test_metrics(uncached_tests)
            for test_key, score in new_scores.items():
                if self.use_cache:
                    _metric_cache[test_key] = score
                total_score += score
                
        return total_score
    
    def _compute_test_metrics(self, test_paths: List[str]) -> Dict[str, float]:
        """
        Compute actual metrics for given tests.
        
        Args:
            test_paths: List of test paths to evaluate
            
        Returns:
            Dictionary mapping test paths to their metric scores
        """
        scores = {}
        
        # Try to use mutation testing if available
        if self._is_mutmut_available():
            scores = self._compute_mutation_scores(test_paths)
        else:
            # Fall back to coverage-based scoring
            scores = self._compute_coverage_scores(test_paths)
            
        return scores
    
    def _is_mutmut_available(self) -> bool:
        """Check if mutmut is available for mutation testing."""
        try:
            result = subprocess.run(
                ["mutmut", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _compute_mutation_scores(self, test_paths: List[str]) -> Dict[str, float]:
        """
        Compute mutation scores for tests using mutmut.
        
        Args:
            test_paths: List of test paths to evaluate
            
        Returns:
            Dictionary mapping test paths to mutation scores
        """
        scores = {}
        
        # Run tests in parallel for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_test = {
                executor.submit(self._run_single_test_mutation, test_path): test_path
                for test_path in test_paths
            }
            
            for future in as_completed(future_to_test):
                test_path = future_to_test[future]
                try:
                    score = future.result()
                    scores[test_path] = score
                except Exception as e:
                    logger.warning(f"Failed to compute mutation score for {test_path}: {e}")
                    scores[test_path] = 0.0
                    
        return scores
    
    def _run_single_test_mutation(self, test_path: str) -> float:
        """
        Run mutation testing for a single test.
        
        Args:
            test_path: Path to the test (can include ::test_name)
            
        Returns:
            Mutation score (0.0 to 1.0)
        """
        # Extract test file and test name if specified
        if "::" in test_path:
            test_file, test_name = test_path.split("::", 1)
        else:
            test_file = test_path
            test_name = None
            
        try:
            # Create a temporary config for this specific test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
                config_content = f"[mutmut]\\npaths_to_mutate={self.project_root}\\ntests_dir={Path(test_file).parent}\\n"
                if test_name:
                    config_content += f"test_command=python -m pytest {test_file}::{test_name} -x\\n"
                else:
                    config_content += f"test_command=python -m pytest {test_file} -x\\n"
                f.write(config_content)
                config_file = f.name
                
            # Run mutmut with limited scope
            result = subprocess.run(
                ["mutmut", "run", "--paths-to-mutate", str(self.project_root), 
                 "--tests-dir", str(Path(test_file).parent),
                 "--runner", f"python -m pytest {test_file} -x"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout per test
            )
            
            # Parse results
            if "killed" in result.stdout:
                killed_count = result.stdout.count("killed")
                total_count = result.stdout.count("mutant")
                if total_count > 0:
                    return killed_count / total_count
                    
            Path(config_file).unlink(missing_ok=True)
            return 0.0
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Mutation testing timed out for {test_path}")
            return 0.0
        except Exception as e:
            logger.error(f"Error running mutation testing for {test_path}: {e}")
            return 0.0
    
    def _compute_coverage_scores(self, test_paths: List[str]) -> Dict[str, float]:
        """
        Compute coverage-based scores as fallback.
        
        Args:
            test_paths: List of test paths to evaluate
            
        Returns:
            Dictionary mapping test paths to coverage scores
        """
        scores = {}
        
        for test_path in test_paths:
            try:
                # Run coverage for individual test
                result = subprocess.run(
                    ["python", "-m", "coverage", "run", "-m", "pytest", test_path, "-x"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Get coverage percentage
                    report_result = subprocess.run(
                        ["python", "-m", "coverage", "report", "--format=json"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
                    
                    if report_result.returncode == 0:
                        coverage_data = json.loads(report_result.stdout)
                        # Normalize coverage percentage to 0-1 range
                        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                        scores[test_path] = total_coverage / 100.0
                    else:
                        scores[test_path] = 0.0
                else:
                    scores[test_path] = 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to compute coverage for {test_path}: {e}")
                scores[test_path] = 0.0
                
        return scores


# Legacy function for backward compatibility
def metric_evaluator_stub(selected_tests: List[Path]) -> float:
    """
    Evaluate metrics for a subset of tests.
    
    This function maintains backward compatibility with the old interface
    but now uses real metric evaluation.
    
    Args:
        selected_tests: List of test paths to evaluate
        
    Returns:
        Aggregated quality score
    """
    # Use current working directory as project root if not specified
    project_root = Path.cwd()
    
    # Find the actual project root by looking for setup.py or pyproject.toml
    current = Path.cwd()
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            project_root = current
            break
        current = current.parent
    
    evaluator = MetricEvaluator(project_root)
    return evaluator.evaluate_test_subset(selected_tests)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with real test files if they exist
    test_files = [
        Path("tests/test_core_tes.py"),
        Path("tests/sensors/test_assertion_iq.py"),
    ]
    
    # Filter to only existing files
    existing_tests = [t for t in test_files if t.exists()]
    
    if existing_tests:
        logger.info(f"Evaluating {len(existing_tests)} test files...")
        score = metric_evaluator_stub(existing_tests)
        logger.info(f"Total quality score: {score:.4f}")
    else:
        logger.info("No test files found for evaluation")
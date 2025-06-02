"""
Sensor for Architectural Violation Score.

This sensor is responsible for providing a RAW architectural violation score,
normalized to a 0-1 range. A score of 0 indicates no or minimal architectural
violations, while a score of 1 indicates a high degree of violations.

The OSQICalculator will then use this raw score to calculate the
Architecture Score (Arch_S) pillar as (1 - raw_architectural_violation_score).
"""
import logging
import os
from typing import Dict, Any
from guardian.analysis.static import build_import_graph, find_circular_dependencies

logger = logging.getLogger(__name__)

def get_raw_architectural_violation_score(project_path: str, config: Dict[str, Any]) -> float:
    """
    Calculates a raw, normalized (0-1) architectural violation score for the project,
    primarily based on the number of circular import dependency sets found.

    Args:
        project_path: The root path of the project to analyze.
        config: Sensor-specific configuration. Can include:
                  'arch_circ_deps_for_half_score': Number of circular dep sets for a 0.5 score (default 1).
                  'arch_circ_deps_for_full_score': Number of circular dep sets for a 1.0 score (default 2).

    Returns:
        A float between 0.0 and 1.0 representing the raw architectural violation score.
        0.0 means few/no violations, 1.0 means many/severe violations.
    """
    logger.info(f"Analyzing architectural violations (circular dependencies) for project at: {project_path}")

    if not os.path.isdir(project_path):
        logger.warning(f"Project path '{project_path}' is not a directory. Cannot analyze for architectural violations.")
        return config.get("default_arch_violation_score_if_path_invalid", 0.5) # Default to a mid-score if path is bad

    try:
        import_graph = build_import_graph(project_path)
        circular_dependency_sets = find_circular_dependencies(import_graph)
        num_circular_deps_sets = len(circular_dependency_sets)
    except Exception as e:
        logger.error(f"Error during circular dependency analysis for {project_path}: {e}")
        # Return a high violation score if analysis fails, as issues might be masked
        return config.get("default_arch_violation_score_if_analysis_fails", 0.75)

    logger.info(f"Found {num_circular_deps_sets} circular dependency sets.")

    # Normalize the number of circular dependency sets to a 0-1 score
    # 0 sets -> 0.0 (no violation)
    # `circ_deps_for_half_score` sets -> 0.5
    # `circ_deps_for_full_score` or more sets -> 1.0 (max violation)
    
    circ_deps_for_half_score = float(config.get("arch_circ_deps_for_half_score", 1.0))
    circ_deps_for_full_score = float(config.get("arch_circ_deps_for_full_score", 2.0))

    if circ_deps_for_half_score <= 0: circ_deps_for_half_score = 1.0
    if circ_deps_for_full_score <= circ_deps_for_half_score: circ_deps_for_full_score = circ_deps_for_half_score + 1.0


    raw_violation_score = 0.0
    if num_circular_deps_sets == 0:
        raw_violation_score = 0.0
    elif num_circular_deps_sets < circ_deps_for_half_score:
        # Linear interpolation between 0 and 0.5
        raw_violation_score = 0.5 * (num_circular_deps_sets / circ_deps_for_half_score)
    elif num_circular_deps_sets < circ_deps_for_full_score:
        # Linear interpolation between 0.5 and 1.0
        raw_violation_score = 0.5 + 0.5 * \
            ((num_circular_deps_sets - circ_deps_for_half_score) / (circ_deps_for_full_score - circ_deps_for_half_score))
    else: # num_circular_deps_sets >= circ_deps_for_full_score
        raw_violation_score = 1.0
    
    final_score = max(0.0, min(raw_violation_score, 1.0))
    logger.info(f"Calculated raw architectural violation score: {final_score:.3f}")
    return final_score

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # To test this properly, you'd need a dummy project with actual circular dependencies.
    # For now, we can mock the output of find_circular_dependencies or create a temp project.

    # Create a dummy project structure for testing
    dummy_project_path = tempfile.mkdtemp(prefix="guardian_arch_test_")
    
    # Scenario 1: No circular dependencies
    # (Difficult to ensure without specific file content, mocking is better for unit tests)
    # For a simple run, create empty files or files with no problematic imports.
    Path(os.path.join(dummy_project_path, "module_a.py")).write_text("import os")
    Path(os.path.join(dummy_project_path, "module_b.py")).write_text("import sys")

    print(f"Testing Architectural Violation Sensor on dummy project at: {dummy_project_path}")
    score1 = get_raw_architectural_violation_score(dummy_project_path, {})
    print(f"Project 1 (no circ deps expected) Raw Architectural Violation Score: {score1:.3f}")
    assert score1 == 0.0, "Expected 0.0 for no circular dependencies"

    # Scenario 2: Create circular dependencies
    # module_x imports module_y, module_y imports module_x
    Path(os.path.join(dummy_project_path, "module_x.py")).write_text("import module_y\nclass X: pass")
    Path(os.path.join(dummy_project_path, "module_y.py")).write_text("import module_x\nclass Y: pass")
    
    score2_config = {"arch_circ_deps_for_half_score": 1, "arch_circ_deps_for_full_score": 2}
    score2 = get_raw_architectural_violation_score(dummy_project_path, score2_config)
    print(f"Project 2 (1 circ dep set expected) Raw Architectural Violation Score: {score2:.3f}")
    # With 1 set, and half_score_threshold=1, score should be 0.5
    assert score2 == 0.5, f"Expected 0.5 for 1 circular dependency set, got {score2}"

    # Scenario 3: More circular dependencies
    Path(os.path.join(dummy_project_path, "module_c.py")).write_text("import module_d\nclass C: pass")
    Path(os.path.join(dummy_project_path, "module_d.py")).write_text("import module_c\nclass D: pass")
    # Now we have two sets: (module_x, module_y) and (module_c, module_d)
    
    score3 = get_raw_architectural_violation_score(dummy_project_path, score2_config) # Re-use config
    print(f"Project 3 (2 circ dep sets expected) Raw Architectural Violation Score: {score3:.3f}")
    # With 2 sets, and full_score_threshold=2, score should be 1.0
    assert score3 == 1.0, f"Expected 1.0 for 2 circular dependency sets, got {score3}"

    import shutil
    import tempfile
    from pathlib import Path
    shutil.rmtree(dummy_project_path)
    print("Dummy project cleaned up.")
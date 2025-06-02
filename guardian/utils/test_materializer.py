import logging
from pathlib import Path
from typing import List

from guardian_ai_tool.guardian.evolution.types import TestIndividual

logger = logging.getLogger(__name__)

def materialize_tests(
    individuals: List[TestIndividual], 
    output_dir: Path, 
    base_filename_prefix: str = "test_auto_generated_"
) -> List[Path]:
    """
    Writes a list of TestIndividual objects to Python test files.

    Each TestIndividual's code (obtained via its .code() method) is written
    to a new file in the specified output directory.

    Args:
        individuals: A list of TestIndividual objects to materialize.
        output_dir: The directory where the test files will be created.
                    It will be created if it doesn't exist.
        base_filename_prefix: Prefix for the generated test files. A short
                              hash from the individual's ID will be appended.

    Returns:
        A list of Path objects pointing to the newly created test files.
    """
    created_files: List[Path] = []
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        return created_files

    for i, individual in enumerate(individuals):
        try:
            test_code_content = individual.code()
            if not test_code_content.strip():
                logger.warning(f"Skipping individual {individual.id} as its .code() method returned empty content.")
                continue

            # Use a part of the individual's ID for uniqueness, or a simple counter
            # The spec mentioned test_auto_<hash>.py. individual.id is already "test_<hash>"
            # Let's use a simpler unique part of the id or a counter if id is too long/complex.
            # individual.id is "test_{hash_value}"
            unique_suffix = individual.id.split('_')[-1][:8] # Take first 8 chars of hash part
            
            filename = f"{base_filename_prefix}{unique_suffix}_{i}.py"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(test_code_content)
            
            created_files.append(filepath)
            logger.info(f"Materialized test to {filepath}")

        except Exception as e:
            logger.error(f"Error materializing test for individual {individual.id}: {e}")
            # Optionally, decide if one failure should stop all, or continue
            
    return created_files
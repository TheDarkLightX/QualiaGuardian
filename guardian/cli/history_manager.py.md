# guardian_ai_tool/guardian/cli/history_manager.py
import json
import os
import datetime
import hashlib
from typing import Dict, Any, List, Optional

# Ensure ETESComponents is available for type hinting if it's a defined class
# from guardian.core.etes import ETESComponents # Assuming ETESComponents is the dataclass

STATS_FILE_NAME = ".guardian_stats.json"
MAX_HISTORY_ENTRIES = 10 # Max number of E-TES runs to keep in history

def get_project_id(project_path: str) -> str:
    """Generates a unique ID for the project based on its absolute path."""
    return hashlib.md5(os.path.abspath(project_path).encode('utf-8')).hexdigest()

def load_project_stats(project_path: str) -> Dict[str, Any]:
    """
    Loads project statistics from .guardian_stats.json in the project directory.
    If the file doesn't exist, returns a default structure.
    """
    stats_file_path = os.path.join(project_path, STATS_FILE_NAME)
    project_abs_path = os.path.abspath(project_path)
    default_stats = {
        "project_id": get_project_id(project_path),
        "project_name": os.path.basename(project_abs_path),
        "project_path_stored": project_abs_path, # Store original path for reference
        "last_run_timestamp": None,
        "history": [], # List of ETESComponents-like dicts
        "gamification": {
            "current_streak_count": 0,
            "highest_etes_score_achieved": 0.0,
            "last_etes_score": None, 
        }
    }
    if not os.path.exists(stats_file_path):
        return default_stats
    try:
        with open(stats_file_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            
            # Validate project_id; if it's from a different project, reset.
            if stats.get("project_id") != get_project_id(project_path):
                print(f"Warning: {STATS_FILE_NAME} project_id mismatch. Initializing new stats for {project_abs_path}.")
                return default_stats
            
            # Ensure all keys are present, providing defaults for missing ones (simple migration)
            for key, value in default_stats.items():
                if key not in stats:
                    stats[key] = value
                elif isinstance(value, dict): 
                    for sub_key, sub_value in value.items():
                        if sub_key not in stats.get(key, {}): # Check if sub_key exists in stats[key]
                             if isinstance(stats.get(key), dict): # Ensure stats[key] is a dict before assigning
                                stats[key][sub_key] = sub_value
                             else: # If stats[key] is not a dict (corrupted?), reset it
                                stats[key] = value 
            return stats
    except (json.JSONDecodeError, IOError, TypeError) as e: # Added TypeError for robustness
        print(f"Warning: Could not load or parse {STATS_FILE_NAME}: {e}. Starting with fresh stats for {project_abs_path}.")
        return default_stats

def save_project_stats(project_path: str, stats: Dict[str, Any]) -> None:
    """Saves project statistics to .guardian_stats.json."""
    stats_file_path = os.path.join(project_path, STATS_FILE_NAME)
    try:
        with open(stats_file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save {STATS_FILE_NAME} in {project_path}: {e}")

def update_stats_with_new_run(
    current_stats: Dict[str, Any], 
    new_etes_run_components: Any, # Expects an object with attributes like ETESComponents
    new_etes_score: float
) -> Dict[str, Any]:
    """
    Updates the statistics dictionary with results from a new E-TES run.
    Modifies current_stats in place but also returns it.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Ensure new_etes_run_components has the expected attributes
    # This is a basic check; more robust validation might be needed
    # depending on how ETESComponents is structured.
    # For now, assuming it has .mutation_score, .evolution_gain etc.
    if not all(hasattr(new_etes_run_components, attr) for attr in [
        "mutation_score", "evolution_gain", "assertion_iq", 
        "behavior_coverage", "speed_factor", "quality_factor"
    ]):
        print("Warning: new_etes_run_components object is missing expected attributes. History entry might be incomplete.")
        # Create a dictionary with available attributes or defaults
        components_snapshot = {
            "mutation_score": getattr(new_etes_run_components, 'mutation_score', 0.0),
            "evolution_gain": getattr(new_etes_run_components, 'evolution_gain', 0.0),
            "assertion_iq": getattr(new_etes_run_components, 'assertion_iq', 0.0),
            "behavior_coverage": getattr(new_etes_run_components, 'behavior_coverage', 0.0),
            "speed_factor": getattr(new_etes_run_components, 'speed_factor', 0.0),
            "quality_factor": getattr(new_etes_run_components, 'quality_factor', 0.0),
            "insights": getattr(new_etes_run_components, 'insights', [])
        }
    else:
        components_snapshot = {
            "mutation_score": new_etes_run_components.mutation_score,
            "evolution_gain": new_etes_run_components.evolution_gain,
            "assertion_iq": new_etes_run_components.assertion_iq,
            "behavior_coverage": new_etes_run_components.behavior_coverage,
            "speed_factor": new_etes_run_components.speed_factor,
            "quality_factor": new_etes_run_components.quality_factor,
            "insights": new_etes_run_components.insights, # Assuming insights is a list
        }

    new_history_entry = {
        "timestamp": now_iso,
        "etes_score": new_etes_score,
        "components": components_snapshot
    }
    current_stats["history"].append(new_history_entry)
    if len(current_stats["history"]) > MAX_HISTORY_ENTRIES:
        current_stats["history"] = current_stats["history"][-MAX_HISTORY_ENTRIES:]

    gamification = current_stats["gamification"]
    previous_score = gamification.get("last_etes_score")

    # Streak logic: Increment if score improves or stays above a high threshold (e.g., 0.8)
    # This makes streaks more meaningful than just any non-decreasing score.
    HIGH_SCORE_THRESHOLD_FOR_STREAK = 0.8 
    streak_continued_or_started = False

    if previous_score is not None:
        if new_etes_score > previous_score:
            gamification["current_streak_count"] = gamification.get("current_streak_count", 0) + 1
            streak_continued_or_started = True
        elif new_etes_score >= HIGH_SCORE_THRESHOLD_FOR_STREAK and \
             previous_score >= HIGH_SCORE_THRESHOLD_FOR_STREAK and \
             new_etes_score >= previous_score: # Allow streak to continue if score is high and doesn't drop
            gamification["current_streak_count"] = gamification.get("current_streak_count", 0) + 1
            streak_continued_or_started = True
        else: # Score dropped or didn't meet criteria to continue streak
            gamification["current_streak_count"] = 0 
    elif new_etes_score >= HIGH_SCORE_THRESHOLD_FOR_STREAK: # First run, start streak if high score
        gamification["current_streak_count"] = 1
        streak_continued_or_started = True
    else: # First run, low score
        gamification["current_streak_count"] = 0
        
    # If streak was broken but current score is good, start a new streak of 1
    if not streak_continued_or_started and new_etes_score >= HIGH_SCORE_THRESHOLD_FOR_STREAK:
        gamification["current_streak_count"] = 1
    elif not streak_continued_or_started and new_etes_score < HIGH_SCORE_THRESHOLD_FOR_STREAK:
         gamification["current_streak_count"] = 0


    if new_etes_score > gamification.get("highest_etes_score_achieved", 0.0):
        gamification["highest_etes_score_achieved"] = new_etes_score
    
    gamification["last_etes_score"] = new_etes_score
    current_stats["last_run_timestamp"] = now_iso
    
    return current_stats
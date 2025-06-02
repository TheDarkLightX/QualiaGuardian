import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable # Added Callable
import math # For level calculation
import os # Added for os.getenv
from enum import Enum
import json # Added for params_used_json and verification_details_json

logger = logging.getLogger(__name__)

# Default DB path, can be overridden
DEFAULT_DB_PATH = Path.home() / ".guardian" / "history.db"

class BadgeCode(Enum):
    """Enum for badge codes for type safety and discoverability."""
    FIRST_COMMAND = "first_command"
    STREAK_3_DAYS = "streak_3_days"
    STREAK_7_DAYS = "streak_7_days"
    LEVEL_5_REACHED = "level_5_reached"
    LEVEL_10_REACHED = "level_10_reached"
    MUTATION_BRONZE = "mutation_bronze" # MS > 50%
    MUTATION_SILVER = "mutation_silver" # MS > 75%
    MUTATION_GOLD = "mutation_gold"   # MS > 90%
    CROWN_JEWELER = "crown_jeweler"   # Used gamify crown
    EVOLUTIONIST = "evolutionist"     # Ran ec-evolve
    CONSISTENT_QUALITY_SILVER = "consistent_quality_silver" # bE-TES > 0.70 for 3 consecutive quality runs
    CONSISTENT_QUALITY_GOLD = "consistent_quality_gold"     # bE-TES > 0.85 for 5 consecutive quality runs

class QuestCode(Enum):
    """Enum for quest codes for type safety and discoverability."""
    IMPROVE_MUTATION_SCORE_TO_70 = "improve_ms_70"
    RUN_CROWN_JEWELER_3_TIMES = "run_crown_3_times"
    ACHIEVE_LEVEL_3 = "reach_level_3"
    # Future quest ideas:
    # FIX_CRITICAL_ISSUE = "fix_critical_issue"
    # COMPLETE_STREAK_5_DAYS = "complete_streak_5_days" # Different from badge, could be a quest
    # EXPLORE_ALL_COMMANDS = "explore_all_commands"

# Quest definitions: code, description, target type, target value, XP reward
# target_type can be 'metric', 'action_count', 'player_stat'
# target_key is the specific metric (e.g., "mutation_score"), action (e.g., "cmd_gamify_crown"), or player stat (e.g., "current_level")
QUEST_DEFINITIONS: Dict[QuestCode, Dict[str, Any]] = {
    QuestCode.IMPROVE_MUTATION_SCORE_TO_70: {
        "description": "Boost your Mutation Score to 70% or higher.",
        "target_type": "metric",
        "target_key": "mutation_score", # from current_metrics
        "target_value": 0.70,
        "reward_xp": 150
    },
    QuestCode.RUN_CROWN_JEWELER_3_TIMES: {
        "description": "Become a Gem Expert: Use 'gamify crown' 3 times to identify key tests.",
        "target_type": "action_count",
        "target_key": "cmd_gamify_crown", # from actions_taken
        "target_value": 3, # Number of times
        "reward_xp": 100
    },
    QuestCode.ACHIEVE_LEVEL_3: {
        "description": "Level Up! Reach Player Level 3.",
        "target_type": "player_stat",
        "target_key": "current_level", # from player_stats
        "target_value": 3,
        "reward_xp": 75
    },
}
 
# Badge definitions: code, description, XP reward for earning
BADGE_DEFINITIONS: Dict[BadgeCode, Dict[str, Any]] = {
    BadgeCode.FIRST_COMMAND: {"description": "Ran your first Guardian command!", "xp_reward": 25},
    BadgeCode.STREAK_3_DAYS: {"description": "Maintained a 3-day activity streak!", "xp_reward": 50},
    BadgeCode.STREAK_7_DAYS: {"description": "Maintained a 7-day activity streak!", "xp_reward": 150},
    BadgeCode.LEVEL_5_REACHED: {"description": "Reached Level 5!", "xp_reward": 100},
    BadgeCode.LEVEL_10_REACHED: {"description": "Reached Level 10!", "xp_reward": 250},
    BadgeCode.MUTATION_BRONZE: {"description": "Achieved a Mutation Score over 50%!", "xp_reward": 75},
    BadgeCode.MUTATION_SILVER: {"description": "Achieved a Mutation Score over 75%!", "xp_reward": 150},
    BadgeCode.MUTATION_GOLD: {"description": "Achieved a Mutation Score over 90%!", "xp_reward": 300},
    BadgeCode.CROWN_JEWELER: {"description": "Used the 'gamify crown' command to find valuable tests!", "xp_reward": 50},
    BadgeCode.EVOLUTIONIST: {"description": "Evolved a test suite using 'ec-evolve'!", "xp_reward": 100},
    BadgeCode.CONSISTENT_QUALITY_SILVER: {"description": "Silver Standard: Maintained bE-TES > 0.70 for 3 quality runs!", "xp_reward": 200},
    BadgeCode.CONSISTENT_QUALITY_GOLD: {"description": "Gold Standard: Maintained bE-TES > 0.85 for 5 quality runs!", "xp_reward": 500},
}


class HistoryManager:
    """
    Manages the gamification history database (SQLite).
    Handles player XP, levels, streaks, badges, and quests.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row # Access columns by name
            logger.info(f"Connected to history database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to history database {self.db_path}: {e}")
            self._conn = None # Ensure conn is None if connection failed

    def _execute_query(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False, commit: bool = False):
        """Helper method to execute SQL queries."""
        if not self._conn:
            logger.error("Database connection not available.")
            # Attempt to reconnect
            self._connect()
            if not self._conn:
                 return None if fetch_one or fetch_all else False
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            if commit:
                self._conn.commit()
                return True
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            return True # For non-select, non-commit operations if needed (e.g. table creation)
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}\nQuery: {query}\nParams: {params}")
            return None if fetch_one or fetch_all else False
        finally:
            # For long-lived HistoryManager, we might not close connection here
            # but ensure it's closed on __del__ or via a close() method.
            # For now, assume connection stays open.
            pass

    def _create_tables(self):
        """Creates database tables if they don't exist."""
        player_table_sql = """
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            username TEXT UNIQUE, -- For display, could be GitHub handle
            current_xp INTEGER DEFAULT 0,
            current_level INTEGER DEFAULT 1,
            streak_days INTEGER DEFAULT 0,
            last_active_ts INTEGER DEFAULT 0
        );
        """
        runs_table_sql = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            betes_score REAL,
            osqi_score REAL,
            delta_betes REAL,
            delta_osqi REAL,
            xp_gained INTEGER DEFAULT 0,
            actions_taken TEXT, -- JSON list of strings
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        );
        """
        badges_table_sql = """
        CREATE TABLE IF NOT EXISTS badges (
            badge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            badge_code TEXT NOT NULL, -- e.g., "mutant_mayhem"
            timestamp_earned INTEGER NOT NULL,
            FOREIGN KEY (player_id) REFERENCES players (player_id),
            UNIQUE (player_id, badge_code) -- Player can earn a badge only once
        );
        """
        quests_table_sql = """
        CREATE TABLE IF NOT EXISTS quests (
            quest_assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            quest_code TEXT NOT NULL, -- e.g., "improve_coverage_module_x"
            description TEXT,
            target_pillar TEXT,
            target_value REAL,
            current_progress REAL DEFAULT 0,
            reward_xp INTEGER,
            assigned_ts INTEGER NOT NULL,
            completed_ts INTEGER,
            is_active BOOLEAN DEFAULT 1, -- 1 for true, 0 for false
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        );
        """
        action_outcomes_table_sql = """
        CREATE TABLE IF NOT EXISTS action_outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            action_id TEXT NOT NULL,
            params_used_json TEXT,
            delta_q_achieved REAL,
            cost_cpu_minutes_incurred REAL,
            success BOOLEAN NOT NULL,
            verification_details_json TEXT,
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        );
        """
        self._execute_query(player_table_sql)
        self._execute_query(runs_table_sql)
        self._execute_query(badges_table_sql)
        self._execute_query(quests_table_sql)
        self._execute_query(action_outcomes_table_sql)
        logger.info("Database tables checked/created.")

    def _get_player_id(self, username: Optional[str] = None) -> str:
        """
        Gets current player ID. For now, uses a default or provided username.
        In a real system, this might come from git config, env var, or auth.
        If player doesn't exist, creates them.
        """
        # For S1, let's use a fixed username if none provided, or allow one to be passed.
        # This will also serve as player_id for simplicity in S1.
        effective_username = username or os.getenv("GUARDIAN_USER", "default_guardian_user")
        
        player = self._execute_query("SELECT player_id FROM players WHERE username = ?", (effective_username,), fetch_one=True)
        if player:
            return player["player_id"]
        else:
            # Create new player
            player_id = effective_username # Use username as player_id for now
            insert_sql = "INSERT INTO players (player_id, username, last_active_ts) VALUES (?, ?, ?)"
            success = self._execute_query(insert_sql, (player_id, effective_username, int(time.time())), commit=True)
            if success:
                logger.info(f"Created new player: {effective_username} with ID: {player_id}")
                return player_id
            else:
                logger.error(f"Failed to create new player: {effective_username}")
                # Fallback to ensure a player_id is always returned for operations,
                # even if DB write failed. This might lead to inconsistencies if DB is down.
                return "fallback_error_player_id"


    def _calculate_level(self, xp: int) -> int:
        """Calculates player level based on XP. Simple formula for now."""
        if xp < 0: xp = 0
        # Example: Level 1: 0-99 XP, Level 2: 100-399 XP, Level 3: 400-899 XP
        # Formula: level = floor(sqrt(xp / 100)) + 1
        # XP for next level: (level^2) * 100
        level = math.floor(math.sqrt(xp / 100)) + 1 if xp > 0 else 1
        return max(1, level) # Ensure level is at least 1

    def get_xp_for_next_level(self, current_level: int) -> int:
        """Calculates XP needed to reach the next level."""
        # If current_level = L, next level is L+1.
        # XP for level L is ((L-1)^2) * 100
        # XP for level L+1 is (L^2) * 100
        return (current_level ** 2) * 100

    def get_player_status(self, username: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches current XP, level, streak, and basic info for a player.
        """
        player_id = self._get_player_id(username)
        if player_id == "fallback_error_player_id":
             return {"error": "Could not retrieve or create player."}

        query = "SELECT username, current_xp, current_level, streak_days, last_active_ts FROM players WHERE player_id = ?"
        player_data = self._execute_query(query, (player_id,), fetch_one=True)

        if player_data:
            # Update streak (simple version: if last_active_ts is not today, reset streak)
            # A more robust version would check for consecutive days.
            today_start_ts = int(time.mktime(time.localtime()[:3] + (0,0,0,0,0,0))) # Midnight today
            last_active_day_start_ts = int(time.mktime(time.localtime(player_data["last_active_ts"])[:3] + (0,0,0,0,0,0)))
            
            current_streak = player_data["streak_days"]
            if last_active_day_start_ts < today_start_ts - (24 * 60 * 60): # More than 1 day ago (e.g. yesterday was missed)
                 # Streak lost if not active yesterday or today. Grace period from proposal (36h) is more complex.
                 # For now, simple daily check.
                 # If last active was before yesterday, streak is lost.
                 if last_active_day_start_ts < today_start_ts - (24*60*60): # Not active yesterday
                    current_streak = 0
                    self._execute_query("UPDATE players SET streak_days = 0 WHERE player_id = ?", (player_id,), commit=True)
            
            # If active today, and last active was yesterday, streak might increment (handled by record_run)
            # This function just reads current state. Streak increment/reset is part of record_run.

            xp_to_next = self.get_xp_for_next_level(player_data["current_level"])
            return {
                "player_id": player_id,
                "username": player_data["username"],
                "xp": player_data["current_xp"],
                "level": player_data["current_level"],
                "xp_to_next_level": xp_to_next,
                "streak_days": current_streak # Use potentially updated streak
            }
        return {
            "player_id": player_id, "username": username or player_id,
            "xp": 0, "level": 1, "xp_to_next_level": self.get_xp_for_next_level(1), "streak_days": 0,
            "error": "Player not found, defaults returned." # Should be created by _get_player_id
        }

    def get_active_quest(self, username: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches the current active quest for a player."""
        player_id = self._get_player_id(username)
        if player_id == "fallback_error_player_id": return None

        query = """
        SELECT quest_code, description, target_pillar, target_value, current_progress, reward_xp
        FROM quests 
        WHERE player_id = ? AND is_active = 1 AND completed_ts IS NULL
        ORDER BY assigned_ts DESC LIMIT 1
        """
        quest_data = self._execute_query(query, (player_id,), fetch_one=True)
        
        if quest_data:
            return dict(quest_data)
        return None # No active quest

    def record_run(self,
                   username: Optional[str] = None,
                   command_name: Optional[str] = None,
                   current_metrics: Optional[Dict[str, Any]] = None, # e.g., {"mutation_score": 0.75}
                   delta_metrics: Optional[Dict[str, float]] = None,
                   actions_taken: Optional[List[str]] = None,
                   xp_bonus: int = 0) -> bool:
        """
        Records a run, calculates XP, updates player stats (XP, level, streak),
        and checks for badge awards.
        
        Args:
            username: The username of the player.
            command_name: Name of the Guardian command executed (e.g., "gamify crown").
            current_metrics: Snapshot of key metrics after the run.
            delta_metrics: A dictionary of metric changes, e.g., {"delta_betes": 0.05}.
            actions_taken: A list of string identifiers for specific actions performed.
            xp_bonus: Any direct XP bonus to award for this run.

        Returns:
            True if successful, False otherwise.
        """
        player_id = self._get_player_id(username)
        if player_id == "fallback_error_player_id": return False

        current_metrics = current_metrics or {}
        delta_metrics = delta_metrics or {}
        actions_taken = actions_taken or []
        if command_name and command_name not in actions_taken: # Ensure command_name is part of actions if provided
            actions_taken.append(f"cmd_{command_name}")


        xp_gained_this_run = xp_bonus
        # XP calculation based on proposal: xp_per_0_01_delta = 20
        delta_m_prime = delta_metrics.get("delta_betes", 0.0) # Assuming bE-TES M'
        if delta_m_prime > 0: # Only award XP for improvement
            xp_gained_this_run += int((delta_m_prime / 0.01) * 20)
        
        # Basic XP for any command run
        if command_name:
            xp_gained_this_run += 10 # Small XP for any command usage

        current_ts = int(time.time())
        
        # json is now imported at the top level

        # Insert into runs table
        run_sql = """
        INSERT INTO runs (player_id, timestamp, betes_score, osqi_score, delta_betes, delta_osqi, xp_gained, actions_taken)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        run_params = (
            player_id, current_ts,
            current_metrics.get("betes_score"),
            current_metrics.get("osqi_score"),
            delta_metrics.get("delta_betes"),
            delta_metrics.get("delta_osqi"),
            xp_gained_this_run,
            json.dumps(actions_taken) if actions_taken else None
        )
        if not self._execute_query(run_sql, run_params, commit=True):
            return False

        # Fetch player data for updates
        player_data_row = self._execute_query("SELECT player_id, username, current_xp, current_level, streak_days, last_active_ts FROM players WHERE player_id = ?", (player_id,), fetch_one=True)
        if not player_data_row:
            logger.error(f"Could not fetch player data for {player_id} after inserting run.")
            return False
        
        player_stats_before_update = dict(player_data_row) # For badge checking

        current_xp = player_stats_before_update["current_xp"]
        current_level = player_stats_before_update["current_level"]
        
        new_xp = current_xp + xp_gained_this_run
        new_level = self._calculate_level(new_xp)
        
        # Streak logic
        new_streak = player_stats_before_update["streak_days"]
        last_active_ts = player_stats_before_update["last_active_ts"]
        
        today_start_ts = int(time.mktime(time.localtime(current_ts)[:3] + (0,0,0,0,0,0)))
        yesterday_start_ts = today_start_ts - (24 * 60 * 60)
        
        if last_active_ts < yesterday_start_ts: # Last active was before yesterday
            new_streak = 1 # Reset to 1 for today's activity
        elif last_active_ts < today_start_ts : # Last active was yesterday
            new_streak += 1
        else: # Already active today, streak doesn't change from this run alone (or it's the first activity of the day)
            if new_streak == 0 : new_streak = 1 # Start streak if it was 0

        update_player_sql = """
        UPDATE players 
        SET current_xp = ?, current_level = ?, streak_days = ?, last_active_ts = ?
        WHERE player_id = ?
        """
        if not self._execute_query(update_player_sql, (new_xp, new_level, new_streak, current_ts, player_id), commit=True):
            return False
        
        logger.info(f"Player {player_id}: XP +{xp_gained_this_run} -> {new_xp}. Level: {player_stats_before_update['current_level']} -> {new_level}. Streak: {new_streak}.")
        
        # Check for badge awards based on this run / new stats
        # Pass current_metrics and actions_taken which includes command_name
        player_stats_after_update = {
            "player_id": player_id, "username": player_stats_before_update["username"],
            "current_xp": new_xp, "current_level": new_level,
            "streak_days": new_streak, "last_active_ts": current_ts
        }

        # Update active quest progress and get any XP from completed quests
        # This needs player_stats *after* run XP but *before* badge/quest XP
        # Also pass the 'actions_taken' from the current run for action_count quests
        quest_xp_gained = self._update_active_quest_progress(
            player_id,
            command_name,
            current_metrics,
            player_stats_after_update,
            actions_taken # Pass current run's actions
        )
        if quest_xp_gained > 0:
            new_xp += quest_xp_gained
            new_level = self._calculate_level(new_xp)
            # Update player_stats_after_update for badge checking if quest XP changed level/xp
            player_stats_after_update["current_xp"] = new_xp
            player_stats_after_update["current_level"] = new_level
            logger.info(f"Player {player_id}: Quest XP +{quest_xp_gained} -> {new_xp}. New Level: {new_level}.")
            # Persist this intermediate update if quest XP was gained
            if not self._execute_query(update_player_sql, (new_xp, new_level, new_streak, current_ts, player_id), commit=True):
                 logger.warning(f"Failed to update player stats after quest XP for {player_id}")

        awarded_badge_xp = self._check_and_award_badges(player_id, command_name, current_metrics, player_stats_after_update)
        
        if awarded_badge_xp > 0:
            new_xp += awarded_badge_xp
            new_level = self._calculate_level(new_xp) # Recalculate level if badge XP changed it
            # Update player again if badge XP was awarded
            if not self._execute_query(update_player_sql, (new_xp, new_level, new_streak, current_ts, player_id), commit=True):
                logger.warning(f"Failed to update player stats after badge XP for {player_id}")
            logger.info(f"Player {player_id}: Badge XP +{awarded_badge_xp} -> {new_xp}. New Level: {new_level}.")

        # Assign a new quest if needed (e.g. if one was just completed)
        self._assign_quest_if_needed(player_id)

        return True

    def _award_badge(self, player_id: str, badge_code: BadgeCode) -> int:
        """Awards a badge to a player if not already earned. Returns XP reward for the badge."""
        badge_info = BADGE_DEFINITIONS.get(badge_code)
        if not badge_info:
            logger.warning(f"Attempted to award undefined badge: {badge_code.value}")
            return 0

        # Check if already earned
        check_query = "SELECT 1 FROM badges WHERE player_id = ? AND badge_code = ?"
        if self._execute_query(check_query, (player_id, badge_code.value), fetch_one=True):
            logger.debug(f"Player {player_id} already has badge {badge_code.value}.")
            return 0 # No XP if already earned

        # Award badge
        insert_query = "INSERT INTO badges (player_id, badge_code, timestamp_earned) VALUES (?, ?, ?)"
        current_ts = int(time.time())
        if self._execute_query(insert_query, (player_id, badge_code.value, current_ts), commit=True):
            logger.info(f"Awarded badge '{badge_code.value}' ({badge_info['description']}) to player {player_id}.")
            return badge_info.get("xp_reward", 0)
        else:
            logger.error(f"Failed to award badge {badge_code.value} to player {player_id}.")
            return 0

    def _check_and_award_badges(self,
                               player_id: str,
                               command_name: Optional[str],
                               current_metrics: Dict[str, Any],
                               player_stats: Dict[str, Any]) -> int:
        """Checks all badge conditions and awards them if criteria met. Returns total XP from newly awarded badges."""
        total_badge_xp_gained = 0
        
        # 1. First Command
        # Check if this is the first run ever for this player
        run_count_query = "SELECT COUNT(*) as count FROM runs WHERE player_id = ?"
        run_count_result = self._execute_query(run_count_query, (player_id,), fetch_one=True)
        if run_count_result and run_count_result["count"] == 1: # This is the first run being recorded
             total_badge_xp_gained += self._award_badge(player_id, BadgeCode.FIRST_COMMAND)

        # 2. Streaks
        if player_stats["streak_days"] >= 7:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.STREAK_7_DAYS)
        elif player_stats["streak_days"] >= 3: # Check 3-day only if 7-day not awarded (or make them independent)
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.STREAK_3_DAYS)

        # 3. Level Milestones
        if player_stats["current_level"] >= 10:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.LEVEL_10_REACHED)
        elif player_stats["current_level"] >= 5:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.LEVEL_5_REACHED)
            
        # 4. Mutation Score Badges
        mutation_score = current_metrics.get("mutation_score", 0.0)
        if mutation_score > 0.90:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.MUTATION_GOLD)
        elif mutation_score > 0.75:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.MUTATION_SILVER)
        elif mutation_score > 0.50:
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.MUTATION_BRONZE)

        # 5. Command-specific badges
        if command_name == "gamify crown":
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.CROWN_JEWELER)
        if command_name == "ec-evolve":
            total_badge_xp_gained += self._award_badge(player_id, BadgeCode.EVOLUTIONIST)

        # 6. Consistency Badges (e.g., for bE-TES score over multiple runs)
        # Check only if the current run was a 'quality' analysis type.
        # The command_name stored in actions_taken is "cmd_quality_analysis"
        # or the current_metrics might contain "betes_score".
        
        # We need to check if this run produced a betes_score.
        # The `command_name` passed here is the direct command like "quality".
        if command_name == "quality" and "betes_score" in current_metrics and current_metrics["betes_score"] is not None:
            # Check for CONSISTENT_QUALITY_SILVER (bE-TES > 0.70 for 3 runs)
            # Check if silver already awarded to avoid re-querying if gold is the target
            has_silver_already = self._execute_query("SELECT 1 FROM badges WHERE player_id = ? AND badge_code = ?", (player_id, BadgeCode.CONSISTENT_QUALITY_SILVER.value), fetch_one=True)
            
            if not has_silver_already:
                recent_quality_runs_silver = self._execute_query(
                    "SELECT betes_score FROM runs WHERE player_id = ? AND betes_score IS NOT NULL ORDER BY timestamp DESC LIMIT 3",
                    (player_id,),
                    fetch_all=True
                )
                if recent_quality_runs_silver and len(recent_quality_runs_silver) == 3:
                    if all(run["betes_score"] > 0.70 for run in recent_quality_runs_silver):
                        total_badge_xp_gained += self._award_badge(player_id, BadgeCode.CONSISTENT_QUALITY_SILVER)
                        has_silver_already = True # Mark as awarded for gold check logic

            # Check for CONSISTENT_QUALITY_GOLD (bE-TES > 0.85 for 5 runs)
            # Only check for gold if silver has been (or was just) awarded, or if they are independent.
            # For now, let's make them independent checks but avoid redundant queries if possible.
            has_gold_already = self._execute_query("SELECT 1 FROM badges WHERE player_id = ? AND badge_code = ?", (player_id, BadgeCode.CONSISTENT_QUALITY_GOLD.value), fetch_one=True)

            if not has_gold_already:
                recent_quality_runs_gold = self._execute_query(
                    "SELECT betes_score FROM runs WHERE player_id = ? AND betes_score IS NOT NULL ORDER BY timestamp DESC LIMIT 5",
                    (player_id,),
                    fetch_all=True
                )
                if recent_quality_runs_gold and len(recent_quality_runs_gold) == 5:
                    if all(run["betes_score"] > 0.85 for run in recent_quality_runs_gold):
                        total_badge_xp_gained += self._award_badge(player_id, BadgeCode.CONSISTENT_QUALITY_GOLD)

        return total_badge_xp_gained

    def _update_active_quest_progress(self,
                                     player_id: str,
                                     command_name: Optional[str], # Still useful for some simple checks
                                     current_metrics: Dict[str, Any],
                                     player_stats: Dict[str, Any],
                                     run_actions_taken: List[str]) -> int: # Actions from the current run
        """
        Updates progress for the player's active quest based on the current run.
        Awards XP if a quest is completed.
        Returns XP gained from completing a quest.
        """
        active_quest_row = self._execute_query(
            "SELECT quest_assignment_id, quest_code, current_progress FROM quests WHERE player_id = ? AND is_active = 1 AND completed_ts IS NULL LIMIT 1",
            (player_id,),
            fetch_one=True
        )

        if not active_quest_row:
            return 0

        quest_assignment_id = active_quest_row["quest_assignment_id"]
        quest_code_str = active_quest_row["quest_code"]
        current_db_progress = float(active_quest_row["current_progress"])

        try:
            quest_code_enum = QuestCode(quest_code_str)
        except ValueError:
            logger.error(f"Invalid quest_code '{quest_code_str}' in database for player {player_id}.")
            return 0
            
        quest_def = QUEST_DEFINITIONS.get(quest_code_enum)
        if not quest_def:
            logger.error(f"Quest definition not found for quest_code '{quest_code_enum.value}'.")
            return 0

        target_type = quest_def["target_type"]
        target_key = quest_def["target_key"]
        target_value = float(quest_def["target_value"])
        reward_xp = int(quest_def["reward_xp"])
        
        new_progress = current_db_progress
        quest_completed_this_run = False

        if target_type == "metric":
            metric_val = current_metrics.get(target_key)
            if metric_val is not None:
                # For metrics, progress is usually the metric value itself, not cumulative
                new_progress = float(metric_val)
                if new_progress >= target_value:
                    quest_completed_this_run = True
        
        elif target_type == "action_count":
            # Count occurrences of target_key (e.g., "cmd_gamify_crown") in the current run's actions_taken
            actions_for_this_quest_key = [action for action in run_actions_taken if action == target_key]
            if actions_for_this_quest_key:
                new_progress = current_db_progress + len(actions_for_this_quest_key)
            
            if new_progress >= target_value:
                quest_completed_this_run = True
        
        elif target_type == "player_stat":
            stat_val = player_stats.get(target_key)
            if stat_val is not None:
                # For player stats, progress is the stat value itself
                new_progress = float(stat_val)
                if new_progress >= target_value:
                    quest_completed_this_run = True
        
        xp_from_this_quest = 0
        if quest_completed_this_run:
            # Ensure progress doesn't exceed target for display, but actual value might be higher
            final_progress_display = min(new_progress, target_value) if target_type == "action_count" else new_progress

            update_q_sql = "UPDATE quests SET current_progress = ?, completed_ts = ?, is_active = 0 WHERE quest_assignment_id = ?"
            if self._execute_query(update_q_sql, (final_progress_display, int(time.time()), quest_assignment_id), commit=True):
                logger.info(f"Player {player_id} completed quest '{quest_code_enum.value}'. Awarding {reward_xp} XP.")
                xp_from_this_quest = reward_xp
            else:
                logger.error(f"Failed to mark quest '{quest_code_enum.value}' complete for player {player_id}.")
        elif new_progress != current_db_progress and new_progress > current_db_progress : # Only update if progress increased
            update_q_sql = "UPDATE quests SET current_progress = ? WHERE quest_assignment_id = ?"
            # For action_count, new_progress is cumulative. For others, it's the current value.
            progress_to_store = new_progress
            if self._execute_query(update_q_sql, (progress_to_store, quest_assignment_id), commit=True):
                logger.info(f"Player {player_id} quest '{quest_code_enum.value}' progress: {current_db_progress} -> {progress_to_store}/{target_value}.")
            else:
                logger.error(f"Failed to update progress for quest '{quest_code_enum.value}' for player {player_id}.")
        
        return xp_from_this_quest

    def _assign_quest_if_needed(self, player_id: str):
        """Assigns a new quest to the player if they don't have an active one."""
        active_quest_query = "SELECT 1 FROM quests WHERE player_id = ? AND is_active = 1 AND completed_ts IS NULL LIMIT 1"
        if self._execute_query(active_quest_query, (player_id,), fetch_one=True):
            logger.debug(f"Player {player_id} already has an active quest.")
            return

        # Get all completed quest codes for this player
        completed_quests_query = "SELECT quest_code FROM quests WHERE player_id = ? AND completed_ts IS NOT NULL"
        completed_rows = self._execute_query(completed_quests_query, (player_id,), fetch_all=True)
        completed_quest_codes = {row["quest_code"] for row in completed_rows} if completed_rows else set()

        # Find the first available quest definition that hasn't been completed
        assigned_quest_code = None
        for quest_code_enum, quest_def in QUEST_DEFINITIONS.items():
            if quest_code_enum.value not in completed_quest_codes:
                assigned_quest_code = quest_code_enum
                break
        
        if assigned_quest_code:
            quest_def_to_assign = QUEST_DEFINITIONS[assigned_quest_code]
            insert_quest_sql = """
            INSERT INTO quests (player_id, quest_code, description, target_pillar, target_value, current_progress, reward_xp, assigned_ts, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """
            # Note: target_pillar is used in the DB schema but not in current QUEST_DEFINITIONS.
            # For now, it will be NULL. It was intended for bE-TES pillars.
            # We'll use target_type and target_key from QUEST_DEFINITIONS for logic.
            
            # For simplicity, let's store target_type and target_key in description or a new field if schema changes.
            # Or, rely on QUEST_DEFINITIONS at runtime. For now, description is enough.
            # The DB schema has target_pillar, target_value.
            # We need to map our QUEST_DEFINITIONS to this.
            # Let's assume target_pillar can store target_key for now if target_type is 'metric' or 'player_stat'.
            # For 'action_count', target_pillar could store the action_key.

            db_target_pillar = quest_def_to_assign.get("target_key") # Store the key (e.g. "mutation_score")
            db_target_value = quest_def_to_assign.get("target_value")

            params = (
                player_id,
                assigned_quest_code.value,
                quest_def_to_assign["description"],
                db_target_pillar,
                db_target_value,
                0, # Initial progress
                quest_def_to_assign["reward_xp"],
                int(time.time())
            )
            if self._execute_query(insert_quest_sql, params, commit=True):
                logger.info(f"Assigned new quest '{assigned_quest_code.value}' to player {player_id}.")
            else:
                logger.error(f"Failed to assign quest '{assigned_quest_code.value}' to player {player_id}.")
        else:
            logger.info(f"No new quests available to assign to player {player_id} or all defined quests completed.")

    def get_earned_badges_count(self, username: Optional[str] = None) -> int:
        """Counts the number of unique badges earned by a player."""
        player_id = self._get_player_id(username)
        if player_id == "fallback_error_player_id": return 0
        
        query = "SELECT COUNT(DISTINCT badge_code) as count FROM badges WHERE player_id = ?"
        result = self._execute_query(query, (player_id,), fetch_one=True)
        return result["count"] if result else 0

    def get_action_posteriors(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Provides posterior distributions (mean, variance) for delta_Q and Cost
        for each known action, based on historical performance.
        Also includes gamma_max for A* planning.

        Args:
            player_id: Optional player_id to potentially fetch player-specific posteriors.
                       Currently unused, global posteriors are returned.

        Returns:
            A dictionary where keys are action_ids.
            Example:
            {
                "gamma_max": 0.004, // Best historical E[delta_Q]/Cost
                "actions": {
                    "auto_test": {
                        "delta_q": {"mean": 0.02, "variance": 0.005},
                        "cost_cpu_minutes": {"mean": 5.0, "variance": 1.0}
                    },
                    "flake_heal": {
                        "delta_q": {"mean": 0.005, "variance": 0.001},
                        "cost_cpu_minutes": {"mean": 1.0, "variance": 0.2}
                    }
                }
            }
        """
        # Known actions for which to calculate posteriors.
        # In a more dynamic system, this list could come from a registry.
        known_action_ids = ["auto_test", "flake_heal"] # Add other action IDs as they become available
        
        action_data_summary = {}
        all_successful_action_efficiencies = []

        for action_id_filter in known_action_ids:
            query = """
            SELECT delta_q_achieved, cost_cpu_minutes_incurred
            FROM action_outcomes
            WHERE action_id = ? AND success = 1
            """
            # If player_id is provided, filter by player. Otherwise, global posteriors.
            # For now, let's stick to global posteriors as per the original placeholder.
            # If player_id specific posteriors are needed later, the query can be adapted.
            # query_params = (action_id_filter, player_id) if player_id else (action_id_filter,)
            
            outcomes = self._execute_query(query, (action_id_filter,), fetch_all=True)

            delta_qs = []
            costs = []
            if outcomes:
                for row in outcomes:
                    if row["delta_q_achieved"] is not None:
                        delta_qs.append(float(row["delta_q_achieved"]))
                    if row["cost_cpu_minutes_incurred"] is not None:
                        costs.append(float(row["cost_cpu_minutes_incurred"]))
            
            # Calculate mean and variance for delta_q
            mean_delta_q = sum(delta_qs) / len(delta_qs) if delta_qs else 0.0
            var_delta_q = sum([(x - mean_delta_q) ** 2 for x in delta_qs]) / len(delta_qs) if len(delta_qs) >= 1 else 0.0 # Use N for population variance
            if len(delta_qs) < 2: var_delta_q = 0.0 # Variance is 0 if less than 2 data points

            # Calculate mean and variance for cost
            mean_cost_cpu = sum(costs) / len(costs) if costs else 1.0 # Default cost 1.0 to avoid div by zero for EQRA
            var_cost_cpu = sum([(x - mean_cost_cpu) ** 2 for x in costs]) / len(costs) if len(costs) >= 1 else 0.0
            if len(costs) < 2: var_cost_cpu = 0.0

            action_data_summary[action_id_filter] = {
                "delta_q": {"mean": mean_delta_q, "variance": var_delta_q, "count": len(delta_qs)},
                "cost_cpu_minutes": {"mean": mean_cost_cpu, "variance": var_cost_cpu, "count": len(costs)}
            }
            
            if mean_cost_cpu > 0 and mean_delta_q > 0: # Only consider positive improvements for gamma_max
                all_successful_action_efficiencies.append(mean_delta_q / mean_cost_cpu)

        gamma_max_val = max(all_successful_action_efficiencies) if all_successful_action_efficiencies else 0.0001 # Default small positive gamma_max

        # Fallback to placeholder if no data for an action, or provide defaults
        final_action_posteriors = {}
        default_delta_q = {"mean": 0.001, "variance": 0.0001**2, "count": 0} # Small positive default
        default_cost = {"mean": 1.0, "variance": 0.1**2, "count": 0}

        for aid in known_action_ids:
            if aid in action_data_summary and action_data_summary[aid]["delta_q"]["count"] > 0:
                final_action_posteriors[aid] = action_data_summary[aid]
            else: # No historical data, use defaults
                final_action_posteriors[aid] = {
                    "delta_q": default_delta_q,
                    "cost_cpu_minutes": default_cost
                }
                if final_action_posteriors[aid]["cost_cpu_minutes"]["mean"] > 0 and final_action_posteriors[aid]["delta_q"]["mean"] > 0:
                     all_successful_action_efficiencies.append(
                         final_action_posteriors[aid]["delta_q"]["mean"] / final_action_posteriors[aid]["cost_cpu_minutes"]["mean"]
                     )


        # Recalculate gamma_max if defaults were added and are positive
        if not all_successful_action_efficiencies and final_action_posteriors: # if still empty, but we have defaults
             positive_default_efficiencies = [
                 entry["delta_q"]["mean"] / entry["cost_cpu_minutes"]["mean"]
                 for entry in final_action_posteriors.values()
                 if entry["cost_cpu_minutes"]["mean"] > 0 and entry["delta_q"]["mean"] > 0
             ]
             if positive_default_efficiencies:
                 gamma_max_val = max(positive_default_efficiencies)


        posteriors_result = {
            "gamma_max": gamma_max_val,
            "actions": final_action_posteriors
        }
        
        return posteriors_result

    def record_action_outcome(self,
                               player_id: str, # Changed from username to player_id for consistency
                               action_id: str,
                               params_used: Dict[str, Any],
                               delta_q_achieved: float,
                               cost_cpu_minutes_incurred: float,
                               success: bool,
                               verification_details_json: Optional[str] = None) -> bool: # Accept JSON string
        """
        Records the outcome of an autonomous agent's action.

        Args:
            player_id: The ID of the player/agent context.
            action_id: Identifier of the action taken (e.g., "auto_test").
            params_used: Dictionary of parameters used for the action.
            delta_q_achieved: The change in quality score resulting from the action.
            cost_cpu_minutes_incurred: The CPU cost incurred by the action and its verification.
            success: Boolean indicating if the action's patch was successfully verified and applied.
            verification_details_json: Optional JSON string with logs or messages from verification.

        Returns:
            True if recording was successful, False otherwise.
        """
        current_ts = int(time.time())
        params_json = json.dumps(params_used) if params_used else None
        # verification_details_json is already a string or None

        sql = """
        INSERT INTO action_outcomes (
            player_id, timestamp, action_id, params_used_json,
            delta_q_achieved, cost_cpu_minutes_incurred, success, verification_details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params_tuple = (
            player_id, current_ts, action_id, params_json,
            delta_q_achieved, cost_cpu_minutes_incurred, success, verification_details_json
        )

        if self._execute_query(sql, params_tuple, commit=True):
            logger.info(f"Recorded action outcome for player {player_id}, action {action_id}, success: {success}")
            return True
        else:
            logger.error(f"Failed to record action outcome for player {player_id}, action {action_id}")
            return False

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("History database connection closed.")
            self._conn = None

    def __del__(self):
        self.close()

# Example usage (for testing purposes if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Use a temporary DB for direct script testing
    temp_db_path = Path("./temp_guardian_history.db")
    if temp_db_path.exists():
        temp_db_path.unlink()

    history_manager = HistoryManager(db_path=temp_db_path)
    
    # Test player creation and status
    player_username = "test_user_s1"
    status = history_manager.get_player_status(player_username)
    logger.info(f"Initial status for {player_username}: {status}")
    
    # Record a run
    run_actions = ["auto_test", "flake_heal"]
    run_deltas = {"delta_betes": 0.03, "betes_score": 0.78} # 3 * 20 = 60 XP + 50 for auto_test = 110 XP
    history_manager.record_run(username=player_username, delta_metrics=run_deltas, actions_taken=run_actions)
    
    status_after_run = history_manager.get_player_status(player_username)
    logger.info(f"Status for {player_username} after run: {status_after_run}")

    # Test active quest (none initially)
    active_quest = history_manager.get_active_quest(player_username)
    logger.info(f"Active quest for {player_username}: {active_quest}")

    # TODO: Add tests for assigning and updating quests, awarding badges once those methods are fleshed out.

    history_manager.close()
    if temp_db_path.exists(): # Clean up
        temp_db_path.unlink()
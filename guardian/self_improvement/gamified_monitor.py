"""
Gamified Monitor Interface

Engaging, game-like interface to encourage human participation in code quality improvement.
Makes monitoring E-TES evolution fun and rewarding!
"""

import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os

from .console_interface import BeautifulConsole


class AchievementType(Enum):
    """Types of achievements users can unlock"""
    SCORE_MILESTONE = "score_milestone"
    IMPROVEMENT_STREAK = "improvement_streak"
    SPEED_DEMON = "speed_demon"
    QUALITY_GURU = "quality_guru"
    EVOLUTION_MASTER = "evolution_master"
    BUG_HUNTER = "bug_hunter"
    PERFECTIONIST = "perfectionist"
    CONSISTENCY_KING = "consistency_king"


@dataclass
class Achievement:
    """Individual achievement definition"""
    id: str
    name: str
    description: str
    icon: str
    type: AchievementType
    threshold: float
    points: int
    unlocked: bool = False
    unlock_time: Optional[float] = None
    progress: float = 0.0


@dataclass
class PlayerStats:
    """Player statistics and progress"""
    total_points: int = 0
    level: int = 1
    experience: int = 0
    achievements_unlocked: int = 0
    total_achievements: int = 0
    
    # E-TES specific stats
    best_etes_score: float = 0.0
    total_improvements: int = 0
    improvement_streak: int = 0
    total_mutations_killed: int = 0
    fastest_optimization_time: float = float('inf')
    
    # Session stats
    session_start: float = field(default_factory=time.time)
    session_improvements: int = 0
    session_points_earned: int = 0


class AchievementSystem:
    """
    Gamification system for E-TES monitoring
    
    Encourages human engagement through achievements, levels, and rewards.
    """
    
    def __init__(self, save_file: str = "guardian_achievements.json"):
        self.save_file = save_file
        self.console = BeautifulConsole()
        
        # Initialize achievements
        self.achievements = self._initialize_achievements()
        self.player_stats = PlayerStats()
        self.player_stats.total_achievements = len(self.achievements)
        
        # Load saved progress
        self._load_progress()
        
        # Level thresholds (experience needed for each level)
        self.level_thresholds = [0, 100, 250, 500, 1000, 1750, 2750, 4000, 5500, 7500, 10000]
    
    def _initialize_achievements(self) -> List[Achievement]:
        """Initialize all available achievements"""
        return [
            # Score Milestones
            Achievement("first_steps", "First Steps", "Achieve E-TES score of 0.1", "🚀", 
                       AchievementType.SCORE_MILESTONE, 0.1, 50),
            Achievement("getting_better", "Getting Better", "Achieve E-TES score of 0.3", "📈", 
                       AchievementType.SCORE_MILESTONE, 0.3, 100),
            Achievement("halfway_there", "Halfway There", "Achieve E-TES score of 0.5", "🎯", 
                       AchievementType.SCORE_MILESTONE, 0.5, 200),
            Achievement("quality_seeker", "Quality Seeker", "Achieve E-TES score of 0.7", "⭐", 
                       AchievementType.SCORE_MILESTONE, 0.7, 300),
            Achievement("excellence", "Excellence", "Achieve E-TES score of 0.85", "🏆", 
                       AchievementType.SCORE_MILESTONE, 0.85, 500),
            Achievement("perfection", "Perfection", "Achieve E-TES score of 0.95", "💎", 
                       AchievementType.SCORE_MILESTONE, 0.95, 1000),
            
            # Improvement Streaks
            Achievement("on_a_roll", "On a Roll", "3 consecutive improvements", "🔥", 
                       AchievementType.IMPROVEMENT_STREAK, 3, 150),
            Achievement("unstoppable", "Unstoppable", "5 consecutive improvements", "⚡", 
                       AchievementType.IMPROVEMENT_STREAK, 5, 300),
            Achievement("legendary", "Legendary", "10 consecutive improvements", "🌟", 
                       AchievementType.IMPROVEMENT_STREAK, 10, 750),
            
            # Speed Achievements
            Achievement("speed_demon", "Speed Demon", "Optimization in under 30 seconds", "💨", 
                       AchievementType.SPEED_DEMON, 30.0, 200),
            Achievement("lightning_fast", "Lightning Fast", "Optimization in under 10 seconds", "⚡", 
                       AchievementType.SPEED_DEMON, 10.0, 400),
            
            # Quality Achievements
            Achievement("quality_guru", "Quality Guru", "Quality Factor above 0.9", "🧘", 
                       AchievementType.QUALITY_GURU, 0.9, 250),
            Achievement("zen_master", "Zen Master", "Quality Factor above 0.95", "☯️", 
                       AchievementType.QUALITY_GURU, 0.95, 500),
            
            # Evolution Achievements
            Achievement("evolution_master", "Evolution Master", "Complete 20 evolution cycles", "🧬", 
                       AchievementType.EVOLUTION_MASTER, 20, 400),
            Achievement("darwin_award", "Darwin Award", "Complete 50 evolution cycles", "🦕", 
                       AchievementType.EVOLUTION_MASTER, 50, 800),
            
            # Bug Hunting
            Achievement("bug_hunter", "Bug Hunter", "Kill 100 mutants", "🐛", 
                       AchievementType.BUG_HUNTER, 100, 300),
            Achievement("exterminator", "Exterminator", "Kill 500 mutants", "🔫", 
                       AchievementType.BUG_HUNTER, 500, 600),
            
            # Perfectionist
            Achievement("perfectionist", "Perfectionist", "All E-TES components above 0.8", "✨", 
                       AchievementType.PERFECTIONIST, 0.8, 400),
            Achievement("flawless", "Flawless", "All E-TES components above 0.9", "💯", 
                       AchievementType.PERFECTIONIST, 0.9, 800),
            
            # Consistency
            Achievement("consistent", "Consistent", "Maintain score for 5 iterations", "🎯", 
                       AchievementType.CONSISTENCY_KING, 5, 250),
            Achievement("rock_solid", "Rock Solid", "Maintain score for 10 iterations", "🗿", 
                       AchievementType.CONSISTENCY_KING, 10, 500),
        ]
    
    def update_progress(self, etes_score: float, components: Any, 
                       optimization_time: float = 0.0, mutations_killed: int = 0):
        """Update player progress and check for achievements"""
        # Update stats
        old_level = self.player_stats.level
        
        if etes_score > self.player_stats.best_etes_score:
            self.player_stats.best_etes_score = etes_score
            self.player_stats.improvement_streak += 1
            self.player_stats.total_improvements += 1
            self.player_stats.session_improvements += 1
        else:
            self.player_stats.improvement_streak = 0
        
        if optimization_time > 0 and optimization_time < self.player_stats.fastest_optimization_time:
            self.player_stats.fastest_optimization_time = optimization_time
        
        self.player_stats.total_mutations_killed += mutations_killed
        
        # Check achievements
        newly_unlocked = self._check_achievements(etes_score, components, optimization_time)
        
        # Award experience and level up
        experience_gained = self._calculate_experience(etes_score, newly_unlocked)
        self.player_stats.experience += experience_gained
        self.player_stats.session_points_earned += experience_gained
        
        # Level up check
        new_level = self._calculate_level()
        if new_level > old_level:
            self._level_up(new_level)
        
        # Display updates
        if newly_unlocked:
            self._display_achievements(newly_unlocked)
        
        self._save_progress()
        
        return newly_unlocked
    
    def _check_achievements(self, etes_score: float, components: Any, 
                          optimization_time: float) -> List[Achievement]:
        """Check and unlock achievements"""
        newly_unlocked = []
        
        for achievement in self.achievements:
            if achievement.unlocked:
                continue
            
            unlocked = False
            
            if achievement.type == AchievementType.SCORE_MILESTONE:
                unlocked = etes_score >= achievement.threshold
                achievement.progress = min(etes_score / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.IMPROVEMENT_STREAK:
                unlocked = self.player_stats.improvement_streak >= achievement.threshold
                achievement.progress = min(self.player_stats.improvement_streak / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.SPEED_DEMON:
                if optimization_time > 0:
                    unlocked = optimization_time <= achievement.threshold
                    achievement.progress = max(0, 1.0 - (optimization_time / achievement.threshold))
            
            elif achievement.type == AchievementType.QUALITY_GURU:
                if hasattr(components, 'quality_factor'):
                    unlocked = components.quality_factor >= achievement.threshold
                    achievement.progress = min(components.quality_factor / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.EVOLUTION_MASTER:
                unlocked = self.player_stats.total_improvements >= achievement.threshold
                achievement.progress = min(self.player_stats.total_improvements / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.BUG_HUNTER:
                unlocked = self.player_stats.total_mutations_killed >= achievement.threshold
                achievement.progress = min(self.player_stats.total_mutations_killed / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.PERFECTIONIST:
                if hasattr(components, 'mutation_score'):
                    all_above_threshold = all([
                        components.mutation_score >= achievement.threshold,
                        components.assertion_iq >= achievement.threshold,
                        components.behavior_coverage >= achievement.threshold,
                        components.speed_factor >= achievement.threshold,
                        components.quality_factor >= achievement.threshold
                    ])
                    unlocked = all_above_threshold
                    min_component = min([
                        components.mutation_score, components.assertion_iq,
                        components.behavior_coverage, components.speed_factor,
                        components.quality_factor
                    ])
                    achievement.progress = min(min_component / achievement.threshold, 1.0)
            
            elif achievement.type == AchievementType.CONSISTENCY_KING:
                # This would need more sophisticated tracking
                achievement.progress = min(self.player_stats.improvement_streak / achievement.threshold, 1.0)
            
            if unlocked:
                achievement.unlocked = True
                achievement.unlock_time = time.time()
                self.player_stats.achievements_unlocked += 1
                self.player_stats.total_points += achievement.points
                newly_unlocked.append(achievement)
        
        return newly_unlocked
    
    def _calculate_experience(self, etes_score: float, achievements: List[Achievement]) -> int:
        """Calculate experience points earned"""
        base_exp = int(etes_score * 100)  # Base experience from score
        achievement_exp = sum(a.points for a in achievements)  # Bonus from achievements
        
        return base_exp + achievement_exp
    
    def _calculate_level(self) -> int:
        """Calculate current level based on experience"""
        for level, threshold in enumerate(self.level_thresholds):
            if self.player_stats.experience < threshold:
                return level
        return len(self.level_thresholds)  # Max level
    
    def _level_up(self, new_level: int):
        """Handle level up"""
        old_level = self.player_stats.level
        self.player_stats.level = new_level
        
        self.console.print_celebration(
            f"🎉 LEVEL UP! 🎉",
            f"Level {old_level} → Level {new_level}",
            "You're becoming a true E-TES master!"
        )
    
    def _display_achievements(self, achievements: List[Achievement]):
        """Display newly unlocked achievements"""
        for achievement in achievements:
            self.console.print_achievement(
                achievement.icon,
                achievement.name,
                achievement.description,
                achievement.points
            )
    
    def display_dashboard(self, etes_score: float, components: Any):
        """Display gamified dashboard"""
        self.console.clear_screen()
        
        # Header
        self.console.print_header("🧬 E-TES EVOLUTION DASHBOARD 🧬")
        
        # Player stats
        level_progress = self._get_level_progress()
        self.console.print_player_stats(
            level=self.player_stats.level,
            experience=self.player_stats.experience,
            level_progress=level_progress,
            total_points=self.player_stats.total_points,
            achievements=f"{self.player_stats.achievements_unlocked}/{self.player_stats.total_achievements}"
        )
        
        # Current E-TES score with visual flair
        self.console.print_etes_score(etes_score, components)
        
        # Recent achievements
        recent_achievements = [a for a in self.achievements if a.unlocked and 
                             a.unlock_time and time.time() - a.unlock_time < 300]  # Last 5 minutes
        if recent_achievements:
            self.console.print_recent_achievements(recent_achievements)
        
        # Progress towards next achievements
        next_achievements = self._get_next_achievements()
        if next_achievements:
            self.console.print_achievement_progress(next_achievements)
        
        # Session stats
        session_time = time.time() - self.player_stats.session_start
        self.console.print_session_stats(
            session_time=session_time,
            improvements=self.player_stats.session_improvements,
            points_earned=self.player_stats.session_points_earned,
            streak=self.player_stats.improvement_streak
        )
    
    def _get_level_progress(self) -> float:
        """Get progress towards next level"""
        current_level = self.player_stats.level
        if current_level >= len(self.level_thresholds) - 1:
            return 1.0  # Max level
        
        current_threshold = self.level_thresholds[current_level - 1] if current_level > 0 else 0
        next_threshold = self.level_thresholds[current_level]
        
        progress = (self.player_stats.experience - current_threshold) / (next_threshold - current_threshold)
        return min(progress, 1.0)
    
    def _get_next_achievements(self) -> List[Achievement]:
        """Get next achievements to work towards"""
        unlocked_achievements = [a for a in self.achievements if not a.unlocked and a.progress > 0]
        unlocked_achievements.sort(key=lambda a: a.progress, reverse=True)
        return unlocked_achievements[:3]  # Top 3 closest
    
    def _save_progress(self):
        """Save progress to file"""
        try:
            data = {
                'player_stats': {
                    'total_points': self.player_stats.total_points,
                    'level': self.player_stats.level,
                    'experience': self.player_stats.experience,
                    'achievements_unlocked': self.player_stats.achievements_unlocked,
                    'best_etes_score': self.player_stats.best_etes_score,
                    'total_improvements': self.player_stats.total_improvements,
                    'improvement_streak': self.player_stats.improvement_streak,
                    'total_mutations_killed': self.player_stats.total_mutations_killed,
                    'fastest_optimization_time': self.player_stats.fastest_optimization_time,
                },
                'achievements': [
                    {
                        'id': a.id,
                        'unlocked': a.unlocked,
                        'unlock_time': a.unlock_time,
                        'progress': a.progress
                    }
                    for a in self.achievements
                ]
            }
            
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def _load_progress(self):
        """Load progress from file"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                
                # Load player stats
                stats = data.get('player_stats', {})
                self.player_stats.total_points = stats.get('total_points', 0)
                self.player_stats.level = stats.get('level', 1)
                self.player_stats.experience = stats.get('experience', 0)
                self.player_stats.achievements_unlocked = stats.get('achievements_unlocked', 0)
                self.player_stats.best_etes_score = stats.get('best_etes_score', 0.0)
                self.player_stats.total_improvements = stats.get('total_improvements', 0)
                self.player_stats.improvement_streak = stats.get('improvement_streak', 0)
                self.player_stats.total_mutations_killed = stats.get('total_mutations_killed', 0)
                self.player_stats.fastest_optimization_time = stats.get('fastest_optimization_time', float('inf'))
                
                # Load achievement progress
                achievement_data = {a['id']: a for a in data.get('achievements', [])}
                for achievement in self.achievements:
                    if achievement.id in achievement_data:
                        saved = achievement_data[achievement.id]
                        achievement.unlocked = saved.get('unlocked', False)
                        achievement.unlock_time = saved.get('unlock_time')
                        achievement.progress = saved.get('progress', 0.0)
                        
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")


class GamifiedMonitor:
    """
    Main gamified monitoring interface
    
    Combines achievement system with beautiful console interface
    to create an engaging E-TES monitoring experience.
    """
    
    def __init__(self):
        self.achievement_system = AchievementSystem()
        self.console = BeautifulConsole()
        self.monitoring = False
    
    def start_monitoring(self, update_interval: float = 2.0):
        """Start gamified monitoring session"""
        self.monitoring = True
        self.console.print_welcome()
        
        try:
            while self.monitoring:
                # In a real implementation, this would get live E-TES data
                # For demo, we'll simulate evolving scores
                etes_score = min(0.95, self.achievement_system.player_stats.best_etes_score + random.uniform(-0.05, 0.1))
                
                # Simulate components
                components = type('Components', (), {
                    'mutation_score': etes_score * 0.9 + random.uniform(-0.1, 0.1),
                    'evolution_gain': 1.0 + random.uniform(0, 0.3),
                    'assertion_iq': etes_score * 0.8 + random.uniform(-0.1, 0.1),
                    'behavior_coverage': etes_score * 0.85 + random.uniform(-0.1, 0.1),
                    'speed_factor': 0.7 + random.uniform(0, 0.3),
                    'quality_factor': etes_score * 0.9 + random.uniform(-0.1, 0.1),
                })()
                
                # Update achievements
                optimization_time = random.uniform(5, 60)
                mutations_killed = random.randint(0, 10)
                
                self.achievement_system.update_progress(
                    etes_score, components, optimization_time, mutations_killed
                )
                
                # Display dashboard
                self.achievement_system.display_dashboard(etes_score, components)
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring session"""
        self.monitoring = False
        self.console.print_goodbye()
    
    def display_leaderboard(self):
        """Display achievement leaderboard"""
        self.console.print_leaderboard(self.achievement_system.achievements)


if __name__ == "__main__":
    # Demo the gamified monitor
    monitor = GamifiedMonitor()
    
    print("🎮 Starting Gamified E-TES Monitor Demo")
    print("Press Ctrl+C to stop")
    
    try:
        monitor.start_monitoring(update_interval=3.0)
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Thanks for playing!")

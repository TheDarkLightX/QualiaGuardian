"""
Beautiful Console Interface

Stunning visual interface for E-TES monitoring that makes humans
want to keep watching and engaging with the system.

Refactored to use Rich library for consistent, cross-platform terminal rendering.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.theme import Theme

# Define a consistent theme matching output_formatter.py
CONSOLE_THEME = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "header": "bold cyan on black",
    "section_title": "bold blue",
    "score_good": "green",
    "score_medium": "yellow",
    "score_bad": "red",
    "progress.bar": "cyan",
    "progress.percentage": "bold",
})


class ProgressBar:
    """
    Beautiful progress bar using Rich library.
    
    Provides consistent styling and cross-platform compatibility.
    """
    
    def __init__(self, width: int = 40, show_percentage: bool = True):
        """
        Initialize progress bar.
        
        Args:
            width: Width of the progress bar
            show_percentage: Whether to show percentage
        """
        self.width = width
        self.show_percentage = show_percentage
    
    def render(self, progress: float, description: str = "") -> Text:
        """
        Render progress bar as Rich Text.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            description: Optional description text
            
        Returns:
            Rich Text object with progress bar
        """
        progress = max(0.0, min(1.0, progress))
        filled_width = int(progress * self.width)
        empty_width = self.width - filled_width
        
        bar_text = Text()
        bar_text.append("█" * filled_width, style="progress.bar")
        bar_text.append("░" * empty_width, style="dim")
        
        if self.show_percentage:
            percentage = f"{progress * 100:5.1f}%"
            bar_text.append(f" {percentage}", style="progress.percentage")
        
        if description:
            bar_text.append(f" {description}", style="dim")
        
        return bar_text


class Box:
    """
    Beautiful box drawing using Rich Panels.
    
    Provides consistent styling and better cross-platform support.
    """
    
    @staticmethod
    def create(
        content: str,
        width: Optional[int] = None,
        title: str = "",
        border_style: str = "cyan"
    ) -> Panel:
        """
        Create a beautiful box around content using Rich Panel.
        
        Args:
            content: Content to display in the box
            width: Optional width (None for auto)
            title: Optional title for the box
            border_style: Rich style for the border
            
        Returns:
            Rich Panel object
        """
        return Panel(
            content,
            title=title if title else None,
            border_style=border_style,
            expand=False,
            width=width
        )


class BeautifulConsole:
    """
    Beautiful console interface for E-TES monitoring.
    
    Creates visually stunning output using Rich library for consistent,
    cross-platform terminal rendering.
    """
    
    def __init__(self, theme: Optional[Theme] = None):
        """
        Initialize beautiful console.
        
        Args:
            theme: Optional Rich theme (uses default if not provided)
        """
        self.console = Console(theme=theme or CONSOLE_THEME)
        self.progress_bar = ProgressBar()
        self.width = self._get_terminal_width()
    
    def _get_terminal_width(self) -> int:
        """
        Get terminal width.
        
        Returns:
            Terminal width in characters, or 80 if unable to determine
        """
        try:
            return os.get_terminal_size().columns
        except (OSError, AttributeError):
            return 80  # Default width
    
    def clear_screen(self) -> None:
        """Clear the screen using Rich console."""
        self.console.clear()
    
    def print_header(self, title: str) -> None:
        """
        Print beautiful header using Rich Panel.
        
        Args:
            title: Header title text
        """
        header_text = Text(title, style="bold white", justify="center")
        header_panel = Panel(
            header_text,
            border_style="bright_cyan",
            expand=False,
            width=min(self.width - 4, 80)
        )
        self.console.print()
        self.console.print(header_panel)
        self.console.print()
    
    def print_player_stats(
        self,
        level: int,
        experience: int,
        level_progress: float,
        total_points: int,
        achievements: str
    ) -> None:
        """
        Print player statistics with beautiful formatting.
        
        Args:
            level: Current player level
            experience: Current experience points
            level_progress: Progress to next level (0.0-1.0)
            total_points: Total points earned
            achievements: Achievement count string (e.g., "12/25")
        """
        stats_text = Text()
        stats_text.append("⭐ LEVEL ", style="bright_yellow")
        stats_text.append(str(level), style="bold bright_yellow")
        stats_text.append(" " * 20, style="")
        stats_text.append("🏆 ", style="bright_magenta")
        stats_text.append(f"{achievements} Achievements", style="bright_magenta")
        stats_text.append("\n", style="")
        
        stats_text.append("Experience: ", style="cyan")
        stats_text.append(f"{experience:,}", style="cyan")
        stats_text.append(" " * 15, style="")
        stats_text.append("💎 ", style="bright_green")
        stats_text.append(f"{total_points:,} Points", style="bright_green")
        stats_text.append("\n\n", style="")
        
        stats_text.append("Level Progress:\n", style="bright_blue")
        progress_bar = self.progress_bar.render(level_progress)
        stats_text.append(progress_bar)
        
        stats_panel = Box.create(
            stats_text,
            width=70,
            title="PLAYER STATS",
            border_style="bright_blue"
        )
        self.console.print(stats_panel)
        self.console.print()
    
    def print_etes_score(self, etes_score: float, components: Any) -> None:
        """
        Print E-TES score with visual components.
        
        Args:
            etes_score: E-TES score (0.0-1.0)
            components: Components object with score breakdown
        """
        # Determine grade and style
        if etes_score >= 0.9:
            grade = "A+"
            score_style = "score_good"
        elif etes_score >= 0.8:
            grade = "A"
            score_style = "score_good"
        elif etes_score >= 0.7:
            grade = "B"
            score_style = "score_medium"
        elif etes_score >= 0.6:
            grade = "C"
            score_style = "score_medium"
        else:
            grade = "F"
            score_style = "score_bad"
        
        # Build score display
        score_text = Text()
        score_text.append("E-TES SCORE: ", style="bold")
        score_text.append(f"{etes_score:.3f}", style=score_style)
        score_text.append(" " * 10, style="")
        score_text.append("GRADE: ", style="bold")
        score_text.append(grade, style=score_style)
        score_text.append("\n\n", style="")
        score_text.append("Component Breakdown:\n", style="section_title")
        
        # Component bars
        components_data = [
            ("Mutation Score", getattr(components, 'mutation_score', 0), "🧬"),
            ("Evolution Gain", getattr(components, 'evolution_gain', 1) - 1, "📈"),
            ("Assertion IQ", getattr(components, 'assertion_iq', 0), "🧠"),
            ("Behavior Coverage", getattr(components, 'behavior_coverage', 0), "🎯"),
            ("Speed Factor", getattr(components, 'speed_factor', 0), "⚡"),
            ("Quality Factor", getattr(components, 'quality_factor', 0), "✨"),
        ]
        
        for name, value, icon in components_data:
            # Normalize evolution gain for display
            display_value = (
                min(value, 1.0) if name != "Evolution Gain"
                else min(value, 0.5) * 2
            )
            
            # Determine component style
            if display_value >= 0.8:
                comp_style = "score_good"
            elif display_value >= 0.6:
                comp_style = "score_medium"
            else:
                comp_style = "score_bad"
            
            score_text.append(f"{icon} {name:16} ", style="")
            progress_bar = self.progress_bar.render(display_value)
            score_text.append(progress_bar)
            score_text.append(f" {value:.3f}\n", style=comp_style)
        
        score_panel = Box.create(
            score_text,
            width=80,
            title="E-TES ANALYSIS",
            border_style=score_style
        )
        self.console.print(score_panel)
        self.console.print()
    
    def print_achievement(
        self, icon: str, name: str, description: str, points: int
    ) -> None:
        """
        Print achievement unlock with celebration.
        
        Args:
            icon: Achievement icon emoji
            name: Achievement name
            description: Achievement description
            points: Points earned
        """
        achievement_text = Text()
        achievement_text.append("🎉 ACHIEVEMENT UNLOCKED! 🎉\n\n", style="bold bright_yellow")
        achievement_text.append(f"{icon} ", style="")
        achievement_text.append(name, style="bold white")
        achievement_text.append(f"\n{description}\n\n", style="cyan")
        achievement_text.append(f"+{points} Points Earned!", style="bright_green")
        
        achievement_panel = Box.create(
            achievement_text,
            width=60,
            border_style="bright_yellow"
        )
        self.console.print(achievement_panel)
        self.console.print()
        
        # Brief celebration animation
        celebration = Text()
        for _ in range(3):
            celebration.append("✨", style="bright_yellow")
        self.console.print(celebration)
        time.sleep(0.6)
    
    def print_celebration(
        self, title: str, subtitle: str, message: str
    ) -> None:
        """
        Print celebration message.
        
        Args:
            title: Celebration title
            subtitle: Celebration subtitle
            message: Celebration message
        """
        celebration_text = Text()
        celebration_text.append(title, style="bold white")
        celebration_text.append(f"\n{subtitle}\n\n", style="bright_cyan")
        celebration_text.append(message, style="green")
        
        celebration_panel = Box.create(
            celebration_text,
            width=60,
            border_style="bright_magenta"
        )
        self.console.print(celebration_panel)
        self.console.print()
    
    def print_recent_achievements(self, achievements: List[Any]) -> None:
        """
        Print recently unlocked achievements.
        
        Args:
            achievements: List of achievement objects
        """
        if not achievements:
            return
        
        recent_text = Text()
        recent_text.append("🏆 RECENT ACHIEVEMENTS:\n\n", style="bright_yellow")
        
        for achievement in achievements[:3]:  # Show last 3
            time_ago = time.time() - achievement.unlock_time
            if time_ago < 60:
                time_str = f"{int(time_ago)}s ago"
            else:
                time_str = f"{int(time_ago/60)}m ago"
            
            recent_text.append(f"{achievement.icon} ", style="")
            recent_text.append(achievement.name, style="bold")
            recent_text.append(f" ({time_str})\n", style="dim")
        
        recent_panel = Box.create(
            recent_text,
            width=50,
            border_style="bright_yellow"
        )
        self.console.print(recent_panel)
        self.console.print()
    
    def print_achievement_progress(self, achievements: List[Any]) -> None:
        """
        Print progress towards next achievements.
        
        Args:
            achievements: List of achievement objects with progress
        """
        if not achievements:
            return
        
        progress_text = Text()
        progress_text.append("🎯 NEXT ACHIEVEMENTS:\n\n", style="bright_blue")
        
        for achievement in achievements:
            progress_text.append(f"{achievement.icon} ", style="")
            progress_text.append(f"{achievement.name}\n", style="")
            progress_bar = self.progress_bar.render(achievement.progress)
            progress_text.append(progress_bar)
            progress_text.append("\n\n", style="")
        
        progress_panel = Box.create(
            progress_text,
            width=60,
            border_style="bright_blue"
        )
        self.console.print(progress_panel)
        self.console.print()
    
    def print_session_stats(
        self,
        session_time: float,
        improvements: int,
        points_earned: int,
        streak: int
    ) -> None:
        """
        Print session statistics.
        
        Args:
            session_time: Session duration in seconds
            improvements: Number of improvements made
            points_earned: Points earned this session
            streak: Current improvement streak
        """
        hours = int(session_time // 3600)
        minutes = int((session_time % 3600) // 60)
        seconds = int(session_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        session_text = Text()
        session_text.append("⏱️  Session Time: ", style="")
        session_text.append(time_str, style="bright_white")
        session_text.append(f"\n📈 Improvements: ", style="")
        session_text.append(str(improvements), style="bright_green")
        session_text.append(f"\n💎 Points Earned: ", style="")
        session_text.append(str(points_earned), style="bright_magenta")
        session_text.append(f"\n🔥 Current Streak: ", style="")
        session_text.append(str(streak), style="bright_yellow")
        
        session_panel = Box.create(
            session_text,
            width=40,
            title="SESSION",
            border_style="bright_white"
        )
        self.console.print(session_panel)
        self.console.print()
    
    def print_welcome(self) -> None:
        """Print welcome message using Rich Panel."""
        self.clear_screen()
        
        welcome_text = Text()
        welcome_text.append("🧬 Welcome to E-TES Evolution Monitor! 🧬\n\n", style="bold white")
        welcome_text.append("Watch your code evolve in real-time!\n", style="bright_yellow")
        welcome_text.append("Earn achievements and level up!\n", style="bright_green")
        welcome_text.append("Become the ultimate E-TES master!", style="bright_magenta")
        
        welcome_panel = Panel(
            welcome_text,
            border_style="bright_cyan",
            expand=False,
            width=60
        )
        self.console.print()
        self.console.print(welcome_panel)
        self.console.print()
        time.sleep(2)
    
    def print_goodbye(self) -> None:
        """Print goodbye message."""
        goodbye_text = Text()
        goodbye_text.append("Thanks for using E-TES Evolution Monitor!\n\n", style="bright_yellow")
        goodbye_text.append("Your progress has been saved.\n", style="bright_green")
        goodbye_text.append("Keep evolving your code! 🧬", style="bright_cyan")
        
        goodbye_panel = Box.create(
            goodbye_text,
            width=50,
            title="GOODBYE",
            border_style="bright_yellow"
        )
        self.console.print(goodbye_panel)
    
    def print_leaderboard(self, achievements: List[Any]) -> None:
        """
        Print achievement leaderboard.
        
        Args:
            achievements: List of achievement objects
        """
        unlocked = [a for a in achievements if a.unlocked]
        unlocked.sort(key=lambda a: a.unlock_time or 0, reverse=True)
        
        leaderboard_text = Text()
        leaderboard_text.append("🏆 ACHIEVEMENT LEADERBOARD 🏆\n\n", style="bold bright_yellow")
        
        for i, achievement in enumerate(unlocked[:10], 1):
            medal = (
                "🥇" if i == 1
                else "🥈" if i == 2
                else "🥉" if i == 3
                else f"{i:2d}."
            )
            leaderboard_text.append(f"{medal} ", style="")
            leaderboard_text.append(f"{achievement.icon} ", style="")
            leaderboard_text.append(achievement.name, style="bold")
            leaderboard_text.append(f" ({achievement.points} pts)\n", style="dim")
        
        leaderboard_panel = Box.create(
            leaderboard_text,
            width=60,
            border_style="bright_yellow"
        )
        self.console.print(leaderboard_panel)


class ProgressTracker:
    """
    Track and display progress for long-running operations using Rich.
    
    Provides better progress visualization with ETA and time remaining.
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.console = BeautifulConsole()
        self._progress = None
    
    def __enter__(self):
        """Context manager entry."""
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console.console
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self.total_steps
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._progress:
            self._progress.stop()
    
    def update(self, step: Optional[int] = None, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Optional step number (increments if not provided)
            message: Optional status message
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if self._progress:
            self._progress.update(
                self._task_id,
                completed=self.current_step,
                description=f"{self.description} | {message}" if message else self.description
            )
        
        if self.current_step >= self.total_steps:
            self.console.console.print("[bright_green]✅ Complete![/]")


if __name__ == "__main__":
    # Demo the beautiful console
    console = BeautifulConsole()
    
    console.print_welcome()
    
    # Demo various elements
    console.print_header("E-TES EVOLUTION DASHBOARD")
    
    console.print_player_stats(
        level=5,
        experience=1250,
        level_progress=0.7,
        total_points=3450,
        achievements="12/25"
    )
    
    # Mock components
    components = type('Components', (), {
        'mutation_score': 0.85,
        'evolution_gain': 1.15,
        'assertion_iq': 0.78,
        'behavior_coverage': 0.92,
        'speed_factor': 0.88,
        'quality_factor': 0.91,
    })()
    
    console.print_etes_score(0.847, components)
    
    console.console.print("[bold bright_green]🎮 Beautiful Console Demo Complete![/]")
    console.console.print("[bright_cyan]This interface makes monitoring E-TES evolution a joy! ✨[/]")

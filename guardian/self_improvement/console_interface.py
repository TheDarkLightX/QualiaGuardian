"""
Beautiful Console Interface

Stunning visual interface for E-TES monitoring that makes humans
want to keep watching and engaging with the system.
"""

import os
import sys
import time
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Color:
    """ANSI color codes for beautiful console output"""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    RESET = '\033[0m'
    
    @classmethod
    def rgb(cls, r: int, g: int, b: int) -> str:
        """Create RGB color code"""
        return f'\033[38;2;{r};{g};{b}m'
    
    @classmethod
    def bg_rgb(cls, r: int, g: int, b: int) -> str:
        """Create RGB background color code"""
        return f'\033[48;2;{r};{g};{b}m'


class ProgressBar:
    """Beautiful progress bar with customizable styling"""
    
    def __init__(self, width: int = 40, filled_char: str = '█', 
                 empty_char: str = '░', show_percentage: bool = True):
        self.width = width
        self.filled_char = filled_char
        self.empty_char = empty_char
        self.show_percentage = show_percentage
    
    def render(self, progress: float, color: str = Color.GREEN) -> str:
        """Render progress bar"""
        progress = max(0.0, min(1.0, progress))
        filled_width = int(progress * self.width)
        empty_width = self.width - filled_width
        
        bar = color + self.filled_char * filled_width + Color.DIM + self.empty_char * empty_width + Color.RESET
        
        if self.show_percentage:
            percentage = f"{progress * 100:5.1f}%"
            return f"{bar} {percentage}"
        
        return bar


class Box:
    """Beautiful box drawing for console output"""
    
    # Box drawing characters
    HORIZONTAL = '─'
    VERTICAL = '│'
    TOP_LEFT = '┌'
    TOP_RIGHT = '┐'
    BOTTOM_LEFT = '└'
    BOTTOM_RIGHT = '┘'
    CROSS = '┼'
    T_DOWN = '┬'
    T_UP = '┴'
    T_RIGHT = '├'
    T_LEFT = '┤'
    
    # Double line variants
    DOUBLE_HORIZONTAL = '═'
    DOUBLE_VERTICAL = '║'
    DOUBLE_TOP_LEFT = '╔'
    DOUBLE_TOP_RIGHT = '╗'
    DOUBLE_BOTTOM_LEFT = '╚'
    DOUBLE_BOTTOM_RIGHT = '╝'
    
    @classmethod
    def create(cls, content: str, width: int = 60, title: str = "", 
               double_line: bool = False, color: str = Color.CYAN) -> str:
        """Create a beautiful box around content"""
        lines = content.split('\n')
        max_content_width = max(len(line) for line in lines) if lines else 0
        box_width = max(width, max_content_width + 4, len(title) + 4)
        
        if double_line:
            h_char = cls.DOUBLE_HORIZONTAL
            v_char = cls.DOUBLE_VERTICAL
            tl_char = cls.DOUBLE_TOP_LEFT
            tr_char = cls.DOUBLE_TOP_RIGHT
            bl_char = cls.DOUBLE_BOTTOM_LEFT
            br_char = cls.DOUBLE_BOTTOM_RIGHT
        else:
            h_char = cls.HORIZONTAL
            v_char = cls.VERTICAL
            tl_char = cls.TOP_LEFT
            tr_char = cls.TOP_RIGHT
            bl_char = cls.BOTTOM_LEFT
            br_char = cls.BOTTOM_RIGHT
        
        # Top border
        if title:
            title_padding = (box_width - len(title) - 2) // 2
            top_line = (color + tl_char + h_char * title_padding + 
                       f" {Color.BOLD}{title}{Color.RESET}{color} " + 
                       h_char * (box_width - title_padding - len(title) - 3) + tr_char + Color.RESET)
        else:
            top_line = color + tl_char + h_char * (box_width - 2) + tr_char + Color.RESET
        
        # Content lines
        content_lines = []
        for line in lines:
            padding = box_width - len(line) - 4
            content_lines.append(f"{color}{v_char}{Color.RESET} {line}{' ' * padding} {color}{v_char}{Color.RESET}")
        
        # Bottom border
        bottom_line = color + bl_char + h_char * (box_width - 2) + br_char + Color.RESET
        
        return '\n'.join([top_line] + content_lines + [bottom_line])


class BeautifulConsole:
    """
    Beautiful console interface for E-TES monitoring
    
    Creates visually stunning output that humans love to watch!
    """
    
    def __init__(self):
        self.width = self._get_terminal_width()
        self.progress_bar = ProgressBar()
        
        # Enable ANSI colors on Windows
        if sys.platform == "win32":
            os.system('color')
    
    def _get_terminal_width(self) -> int:
        """Get terminal width"""
        try:
            return os.get_terminal_size().columns
        except:
            return 80  # Default width
    
    def clear_screen(self):
        """Clear the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Print beautiful header"""
        print()
        header_width = min(self.width - 4, 80)
        
        # Gradient effect
        gradient_colors = [
            Color.rgb(138, 43, 226),   # Blue Violet
            Color.rgb(75, 0, 130),     # Indigo
            Color.rgb(0, 0, 255),      # Blue
            Color.rgb(0, 255, 255),    # Cyan
            Color.rgb(0, 255, 0),      # Green
        ]
        
        # Top border with gradient
        top_border = "╔" + "═" * (header_width - 2) + "╗"
        print(f"{Color.BRIGHT_CYAN}{top_border}{Color.RESET}")
        
        # Title with centered alignment and gradient
        title_padding = (header_width - len(title) - 2) // 2
        title_line = f"║{' ' * title_padding}{Color.BOLD}{Color.BRIGHT_WHITE}{title}{Color.RESET}{Color.BRIGHT_CYAN}{' ' * (header_width - len(title) - title_padding - 2)}║"
        print(f"{Color.BRIGHT_CYAN}{title_line}{Color.RESET}")
        
        # Bottom border
        bottom_border = "╚" + "═" * (header_width - 2) + "╝"
        print(f"{Color.BRIGHT_CYAN}{bottom_border}{Color.RESET}")
        print()
    
    def print_player_stats(self, level: int, experience: int, level_progress: float,
                          total_points: int, achievements: str):
        """Print player statistics with beautiful formatting"""
        stats_content = f"""
{Color.BRIGHT_YELLOW}⭐ LEVEL {level}{Color.RESET}                    {Color.BRIGHT_MAGENTA}🏆 {achievements} Achievements{Color.RESET}
{Color.CYAN}Experience: {experience:,}{Color.RESET}              {Color.BRIGHT_GREEN}💎 {total_points:,} Points{Color.RESET}

{Color.BRIGHT_BLUE}Level Progress:{Color.RESET}
{self.progress_bar.render(level_progress, Color.BRIGHT_BLUE)}
"""
        
        print(Box.create(stats_content.strip(), width=70, title="PLAYER STATS", 
                        color=Color.BRIGHT_BLUE))
        print()
    
    def print_etes_score(self, etes_score: float, components: Any):
        """Print E-TES score with visual components"""
        # Determine grade and color
        if etes_score >= 0.9:
            grade = "A+"
            grade_color = Color.BRIGHT_GREEN
            score_color = Color.BRIGHT_GREEN
        elif etes_score >= 0.8:
            grade = "A"
            grade_color = Color.GREEN
            score_color = Color.GREEN
        elif etes_score >= 0.7:
            grade = "B"
            grade_color = Color.BRIGHT_YELLOW
            score_color = Color.BRIGHT_YELLOW
        elif etes_score >= 0.6:
            grade = "C"
            grade_color = Color.YELLOW
            score_color = Color.YELLOW
        else:
            grade = "F"
            grade_color = Color.BRIGHT_RED
            score_color = Color.BRIGHT_RED
        
        # Main score display
        score_display = f"""
{Color.BOLD}{score_color}E-TES SCORE: {etes_score:.3f}{Color.RESET}     {Color.BOLD}{grade_color}GRADE: {grade}{Color.RESET}

{Color.BRIGHT_CYAN}Component Breakdown:{Color.RESET}
"""
        
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
            display_value = min(value, 1.0) if name != "Evolution Gain" else min(value, 0.5) * 2
            
            color = Color.BRIGHT_GREEN if display_value >= 0.8 else Color.BRIGHT_YELLOW if display_value >= 0.6 else Color.BRIGHT_RED
            bar = self.progress_bar.render(display_value, color)
            score_display += f"{icon} {name:16} {bar} {value:.3f}\n"
        
        print(Box.create(score_display.strip(), width=80, title="E-TES ANALYSIS", 
                        double_line=True, color=score_color))
        print()
    
    def print_achievement(self, icon: str, name: str, description: str, points: int):
        """Print achievement unlock with celebration"""
        achievement_text = f"""
{Color.BRIGHT_YELLOW}🎉 ACHIEVEMENT UNLOCKED! 🎉{Color.RESET}

{icon} {Color.BOLD}{Color.BRIGHT_WHITE}{name}{Color.RESET}
{Color.CYAN}{description}{Color.RESET}

{Color.BRIGHT_GREEN}+{points} Points Earned!{Color.RESET}
"""
        
        print(Box.create(achievement_text.strip(), width=60, 
                        color=Color.BRIGHT_YELLOW))
        print()
        
        # Brief celebration animation
        for _ in range(3):
            print(f"{Color.BRIGHT_YELLOW}✨{Color.RESET}", end="", flush=True)
            time.sleep(0.2)
        print()
    
    def print_celebration(self, title: str, subtitle: str, message: str):
        """Print celebration message"""
        celebration_text = f"""
{Color.BOLD}{Color.BRIGHT_WHITE}{title}{Color.RESET}
{Color.BRIGHT_CYAN}{subtitle}{Color.RESET}

{Color.GREEN}{message}{Color.RESET}
"""
        
        print(Box.create(celebration_text.strip(), width=60, 
                        color=Color.BRIGHT_MAGENTA))
        print()
    
    def print_recent_achievements(self, achievements: List[Any]):
        """Print recently unlocked achievements"""
        if not achievements:
            return
        
        recent_text = f"{Color.BRIGHT_YELLOW}🏆 RECENT ACHIEVEMENTS:{Color.RESET}\n\n"
        
        for achievement in achievements[:3]:  # Show last 3
            time_ago = time.time() - achievement.unlock_time
            if time_ago < 60:
                time_str = f"{int(time_ago)}s ago"
            else:
                time_str = f"{int(time_ago/60)}m ago"
            
            recent_text += f"{achievement.icon} {Color.BOLD}{achievement.name}{Color.RESET} ({time_str})\n"
        
        print(Box.create(recent_text.strip(), width=50, color=Color.BRIGHT_YELLOW))
        print()
    
    def print_achievement_progress(self, achievements: List[Any]):
        """Print progress towards next achievements"""
        if not achievements:
            return
        
        progress_text = f"{Color.BRIGHT_BLUE}🎯 NEXT ACHIEVEMENTS:{Color.RESET}\n\n"
        
        for achievement in achievements:
            bar = self.progress_bar.render(achievement.progress, Color.BRIGHT_BLUE)
            progress_text += f"{achievement.icon} {achievement.name}\n{bar}\n\n"
        
        print(Box.create(progress_text.strip(), width=60, color=Color.BRIGHT_BLUE))
        print()
    
    def print_session_stats(self, session_time: float, improvements: int, 
                           points_earned: int, streak: int):
        """Print session statistics"""
        hours = int(session_time // 3600)
        minutes = int((session_time % 3600) // 60)
        seconds = int(session_time % 60)
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        session_text = f"""
{Color.BRIGHT_WHITE}⏱️  Session Time: {time_str}{Color.RESET}
{Color.BRIGHT_GREEN}📈 Improvements: {improvements}{Color.RESET}
{Color.BRIGHT_MAGENTA}💎 Points Earned: {points_earned}{Color.RESET}
{Color.BRIGHT_YELLOW}🔥 Current Streak: {streak}{Color.RESET}
"""
        
        print(Box.create(session_text.strip(), width=40, title="SESSION", 
                        color=Color.BRIGHT_WHITE))
        print()
    
    def print_welcome(self):
        """Print welcome message"""
        self.clear_screen()
        
        welcome_art = f"""
{Color.BRIGHT_CYAN}
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  {Color.BRIGHT_WHITE}🧬 Welcome to E-TES Evolution Monitor! 🧬{Color.BRIGHT_CYAN}           ║
    ║                                                           ║
    ║  {Color.BRIGHT_YELLOW}Watch your code evolve in real-time!{Color.BRIGHT_CYAN}                ║
    ║  {Color.BRIGHT_GREEN}Earn achievements and level up!{Color.BRIGHT_CYAN}                     ║
    ║  {Color.BRIGHT_MAGENTA}Become the ultimate E-TES master!{Color.BRIGHT_CYAN}                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
{Color.RESET}
"""
        print(welcome_art)
        time.sleep(2)
    
    def print_goodbye(self):
        """Print goodbye message"""
        goodbye_text = f"""
{Color.BRIGHT_YELLOW}Thanks for using E-TES Evolution Monitor!{Color.RESET}

{Color.BRIGHT_GREEN}Your progress has been saved.{Color.RESET}
{Color.BRIGHT_CYAN}Keep evolving your code! 🧬{Color.RESET}
"""
        
        print(Box.create(goodbye_text.strip(), width=50, title="GOODBYE", 
                        color=Color.BRIGHT_YELLOW))
    
    def print_leaderboard(self, achievements: List[Any]):
        """Print achievement leaderboard"""
        unlocked = [a for a in achievements if a.unlocked]
        unlocked.sort(key=lambda a: a.unlock_time or 0, reverse=True)
        
        leaderboard_text = f"{Color.BRIGHT_GOLD}🏆 ACHIEVEMENT LEADERBOARD 🏆{Color.RESET}\n\n"
        
        for i, achievement in enumerate(unlocked[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            leaderboard_text += f"{medal} {achievement.icon} {achievement.name} ({achievement.points} pts)\n"
        
        print(Box.create(leaderboard_text.strip(), width=60, 
                        color=Color.BRIGHT_YELLOW))


class ProgressTracker:
    """Track and display progress for long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.console = BeautifulConsole()
    
    def update(self, step: int = None, message: str = ""):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = self.current_step / self.total_steps
        elapsed = time.time() - self.start_time
        
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # Create progress display
        bar = ProgressBar(width=40).render(progress, Color.BRIGHT_GREEN)
        
        status_line = f"\r{Color.BRIGHT_CYAN}{self.description}:{Color.RESET} {bar} {eta_str}"
        if message:
            status_line += f" | {message}"
        
        print(status_line, end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print(f"\n{Color.BRIGHT_GREEN}✅ Complete!{Color.RESET}")


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
    
    print("🎮 Beautiful Console Demo Complete!")
    print("This interface makes monitoring E-TES evolution a joy! ✨")

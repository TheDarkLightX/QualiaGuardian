"""
Guardian CLI Package

Provides the analyzer utilities plus a compatibility bridge so
`guardian.cli.app` exposes the Typer application defined in the
monolithic `guardian/cli.py`.
"""

import importlib.util
import sys
from types import ModuleType
from pathlib import Path

from .analyzer import ProjectAnalyzer
from .output_formatter import OutputFormatter, FormattingConfig, OutputLevel

__all__ = [
    "ProjectAnalyzer",
    "OutputFormatter",
    "FormattingConfig",
    "OutputLevel",
    "app",
]


def _load_typer_app() -> ModuleType | None:
    """Load the legacy Typer CLI module without shadowing this package."""
    cli_entry_path = Path(__file__).resolve().parent.parent / "cli.py"
    if not cli_entry_path.exists():
        return None

    module_name = "guardian._typer_cli_entry"
    existing = sys.modules.get(module_name)
    if existing:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, cli_entry_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_typer_cli_module = _load_typer_app()
app = getattr(_typer_cli_module, "app", None)


def __getattr__(name: str):
    """
    Delegate missing attributes to the Typer CLI module so tests that patch
    members (e.g., evaluate_subset) continue to work.
    """
    if _typer_cli_module and hasattr(_typer_cli_module, name):
        return getattr(_typer_cli_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

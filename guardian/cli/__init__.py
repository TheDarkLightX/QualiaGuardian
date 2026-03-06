"""
Guardian CLI Package

Professional command-line interface for Guardian code analysis.
"""

from importlib import util as _importlib_util
from pathlib import Path as _Path

from .analyzer import ProjectAnalyzer
from .output_formatter import OutputFormatter, FormattingConfig, OutputLevel


def _load_root_cli_module():
    root_cli_path = _Path(__file__).resolve().parents[1] / "cli.py"
    spec = _importlib_util.spec_from_file_location("guardian._root_cli_module", root_cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load root CLI module from {root_cli_path}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root_cli_module = _load_root_cli_module()
main = _root_cli_module.main
app = _root_cli_module.app


def __getattr__(name):
    return getattr(_root_cli_module, name)

__all__ = [
    'app',
    'main',
    'ProjectAnalyzer',
    'OutputFormatter', 
    'FormattingConfig',
    'OutputLevel'
]

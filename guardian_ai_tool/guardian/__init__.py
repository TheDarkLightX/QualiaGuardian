"""
Thin namespace shim so legacy imports like
`guardian_ai_tool.guardian.analytics` resolve to the actively
maintained `guardian.analytics` modules.
"""

from __future__ import annotations

import importlib
import sys
from typing import Iterable

_REAL_MODULE_NAME = "guardian"


def _redirect_module():
    real_guardian = importlib.import_module(_REAL_MODULE_NAME)
    sys.modules[__name__] = real_guardian
    return real_guardian


def _alias_submodules(submodules: Iterable[str]) -> None:
    """Register legacy alias modules so they share identity with `guardian`."""
    for submodule in submodules:
        target = f"{_REAL_MODULE_NAME}.{submodule}"
        alias = f"{__name__}.{submodule}"
        try:
            module = importlib.import_module(target)
        except ImportError:
            continue
        sys.modules[alias] = module


_redirect_module()
_alias_submodules(
    [
        "analytics",
        "analytics.shapley",
        "analytics.metric_stubs",
        "core",
        "core.api",
        "cli",
        "cli.analyzer",
        "cli.output_formatter",
        "test_execution",
        "test_execution.pytest_runner",
    ]
)

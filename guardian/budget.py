from dataclasses import dataclass
from typing import Optional

@dataclass
class Budget:
    """
    Represents the budget constraints for an operation, typically for AdaptiveEMT.

    Attributes:
        cpu_core_min (int): The budget in CPU core-minutes.
        wall_min (Optional[int]): The budget in wall-clock minutes. Defaults to None.
        target_delta_m (Optional[float]): The target improvement in the primary
                                           metric (e.g., mutation score M').
                                           If specified, the process may stop
                                           early if this delta is achieved.
                                           Defaults to None.
    """
    cpu_core_min: int
    wall_min: Optional[int] = None
    target_delta_m: Optional[float] = None
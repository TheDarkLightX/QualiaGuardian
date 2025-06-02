import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PIDController:
    """
    A Proportional-Integral (PI) controller.
    
    Used to adjust a control variable (e.g., population size for AdaptiveEMT)
    based on a setpoint and a process variable. The derivative (D) term
    is omitted for simplicity as it can add noise in short-running processes.
    """
    def __init__(self, kp: float, ki: float, sp: float, 
                 min_output: int = 5, max_output: int = 100):
        """
        Initializes the PIDController.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            sp (float): Setpoint for the process variable.
            min_output (int): Minimum value for the control output. Defaults to 5.
            max_output (int): Maximum value for the control output. Defaults to 100.
        """
        self.kp = kp
        self.ki = ki
        self.sp = sp
        self.min_output = min_output
        self.max_output = max_output
        
        self.integral: float = 0.0
        self.last_pv: Optional[float] = None # Process variable, currently not used for D term

    def next(self, pv: float) -> int:
        """
        Calculates the next control variable output (e.g., population size).

        Args:
            pv (float): The current value of the process variable 
                        (e.g., CPU minutes used, or metric delta achieved).

        Returns:
            int: The calculated control output, clamped between min_output
                 and max_output.
        """
        error = self.sp - pv
        self.integral += error
        
        # Anti-windup for integral term (optional, but good practice)
        # If output is already at max/min and integral term would push it further, clamp integral.
        # This simple version doesn't explicitly do anti-windup on self.integral directly,
        # but the output clamping has a similar effect.
        
        cv_float = (self.kp * error) + (self.ki * self.integral)
        
        # Clamp output
        cv_int = int(round(cv_float))
        clamped_cv = max(self.min_output, min(cv_int, self.max_output))
        
        logger.debug(
            f"PID: sp={self.sp:.2f}, pv={pv:.2f}, error={error:.2f}, "
            f"integral={self.integral:.2f}, cv_float={cv_float:.2f}, clamped_cv={clamped_cv}"
        )
        
        self.last_pv = pv # Store for potential future D term or logging
        return clamped_cv

    def reset_integral(self) -> None:
        """Resets the integral term to zero."""
        self.integral = 0.0
        logger.debug("PID integral term reset.")

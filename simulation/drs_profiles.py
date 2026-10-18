import numpy as np


class DynamicSupportSurface:
    """
    Simulates Dynamic Rigid Surface (DRS) kinematics (position, velocity, acceleration)
    including periodic sway (Cases A-D in Gao et al., 2025), amplitude/phase uncertainties,
    and non-periodic motions.
    """
    def __init__(self, case: str = "A", delta_A: float = 0.0, delta_t: float = 0.0):
        self.case = case.upper()
        self.delta_A = delta_A  # Amplitude uncertainty offset (m)
        self.delta_t = delta_t  # Phase/time uncertainty offset (s)

    def position(self, t: float) -> tuple[float, float]:
        """Returns (x_WS(t), y_WS(t)) in meters in the world frame."""
        t_eff = t + self.delta_t
        if self.case == "A":
            return (0.04 + self.delta_A) * np.cos(2 * np.pi / 0.4 * t_eff), 0.0
        elif self.case == "B":
            return (0.14 + self.delta_A) * np.cos(2 * np.pi / 6.0 * t_eff), 0.0
        elif self.case == "C":
            return 0.0, (0.06 + self.delta_A) * np.cos(2 * np.pi / 0.72 * t_eff)
        elif self.case == "D":
            x = (0.04 + self.delta_A) * np.cos(2 * np.pi / 0.4 * t_eff)
            y = (0.10 + self.delta_A) * np.cos(2 * np.pi / 6.0 * t_eff)
            return x, y
        elif self.case == "NONPERIODIC":
            x = (0.04 / (1.0 + t)) * np.sin(t)
            y = 0.10 * np.cos(2 * np.pi / 6.0 * t)
            return x, y
        return 0.0, 0.0

    def velocity(self, t: float) -> tuple[float, float]:
        """Returns (\dot{x}_WS(t), \dot{y}_WS(t)) in m/s in the world frame."""
        t_eff = t + self.delta_t
        if self.case == "A":
            w = 2 * np.pi / 0.4
            return -(0.04 + self.delta_A) * w * np.sin(w * t_eff), 0.0
        elif self.case == "B":
            w = 2 * np.pi / 6.0
            return -(0.14 + self.delta_A) * w * np.sin(w * t_eff), 0.0
        elif self.case == "C":
            w = 2 * np.pi / 0.72
            return 0.0, -(0.06 + self.delta_A) * w * np.sin(w * t_eff)
        elif self.case == "D":
            wx = 2 * np.pi / 0.4
            wy = 2 * np.pi / 6.0
            dx = -(0.04 + self.delta_A) * wx * np.sin(wx * t_eff)
            dy = -(0.10 + self.delta_A) * wy * np.sin(wy * t_eff)
            return dx, dy
        elif self.case == "NONPERIODIC":
            dx = 0.04 * ((1.0 + t) * np.cos(t) - np.sin(t)) / ((1.0 + t) ** 2)
            dy = -0.10 * (2 * np.pi / 6.0) * np.sin(2 * np.pi / 6.0 * t)
            return dx, dy
        return 0.0, 0.0
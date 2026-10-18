import numpy as np
from math import comb


class BezierGaitGenerator:
    """
    Middle-Layer Walking Pattern Planner (Gao et al., 2025, Section IV-B).
    Generates desired continuous trajectories h_d(s) using M-th order Bezier polynomials
    parameterized by normalized gait phase s in [0, 1).
    """
    def __init__(self, order: int = 5, step_clearance: float = 0.04):
        self.M = order
        self.step_clearance = step_clearance

    def evaluate_bezier(self, coeffs: np.ndarray, s: float) -> float:
        """Evaluates phi_i(s) = sum_{j=0}^M alpha_{i,j} * C(M,j) * s^j * (1-s)^(M-j)."""
        s = float(np.clip(s, 0.0, 1.0))
        val = 0.0
        for j in range(self.M + 1):
            val += coeffs[j] * comb(self.M, j) * (s ** j) * ((1.0 - s) ** (self.M - j))
        return val

    def swing_foot_trajectory(
        self,
        s: float,
        p_sw_init: np.ndarray,
        p_sw_target: np.ndarray,
        clearance_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Computes 3D swing-foot position [x_Sw(s), y_Sw(s), z_Sw(s)] at phase s in [0, 1].
        Continuously updates Bezier control points so the swing foot starts at p_sw_init
        with zero velocity and lands smoothly at p_sw_target.
        """
        x0, y0, z0 = p_sw_init
        xf, yf, zf = p_sw_target
        z_peak = self.step_clearance + clearance_offset

        # 5th-order Bezier coefficients with zero start/end velocity in x and y
        alpha_x = np.array([x0, x0, 0.5 * (x0 + xf), xf, xf, xf])
        alpha_y = np.array([y0, y0, 0.5 * (y0 + yf), yf, yf, yf])
        # Parabolic-like bell profile for vertical swing clearance z_Sw
        alpha_z = np.array([z0, z0 + 1.2 * z_peak, z0 + 2.0 * z_peak, zf + 1.2 * z_peak, zf, zf])

        x_sw = self.evaluate_bezier(alpha_x, s)
        y_sw = self.evaluate_bezier(alpha_y, s)
        z_sw = self.evaluate_bezier(alpha_z, s)
        return np.array([x_sw, y_sw, z_sw])
import numpy as np
from scipy.integrate import quad_vec


class ALIPDRSPlanner:
    """
    Implements the Hybrid ALIP-DRS Model and Discrete-Time Foot Placement Controller
    from Gao et al. (IEEE T-RO 2025) with adaptive parameter hooks for RL Meta-Control.
    """
    def __init__(
        self,
        mass: float = 46.1,
        nom_height: float = 0.9,
        nom_t_step: float = 0.4,
        step_width: float = 0.2,
        g: float = 9.81,
    ):
        self.m = mass
        self.H = nom_height
        self.T_step = nom_t_step
        self.W = step_width
        self.g = g

    @property
    def l(self) -> float:
        return np.sqrt(self.g / self.H)

    def exp_Ax(self, dt: float, H: float = None) -> np.ndarray:
        """Closed-form state transition matrix exp(A_x * dt) from Eq. (14)."""
        H = self.H if H is None else H
        l = np.sqrt(self.g / H)
        ch, sh = np.cosh(l * dt), np.sinh(l * dt)
        return np.array([
            [ch, sh / (self.m * H * l)],
            [self.m * H * l * sh, ch]
        ])

    def exp_Ay(self, dt: float, H: float = None) -> np.ndarray:
        """Closed-form state transition matrix exp(A_y * dt) from Eq. (27)."""
        H = self.H if H is None else H
        l = np.sqrt(self.g / H)
        ch, sh = np.cosh(l * dt), np.sinh(l * dt)
        return np.array([
            [ch, -sh / (self.m * H * l)],
            [-self.m * H * l * sh, ch]
        ])

    def compute_forcing_integrals(
        self, t1: float, t2: float, vel_func, H: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes V_x(t1, t2) and V_y(t1, t2) by integrating the surface velocity
        forcing terms f_x(tau) = [-dx_WS(tau), 0]^T and f_y(tau) = [-dy_WS(tau), 0]^T.
        """
        if t2 <= t1:
            return np.zeros(2), np.zeros(2)
        H = self.H if H is None else H

        def integrand(tau):
            dx_ws, dy_ws = vel_func(tau)
            fx = np.array([-dx_ws, 0.0])
            fy = np.array([-dy_ws, 0.0])
            vx = self.exp_Ax(t2 - tau, H) @ fx
            vy = self.exp_Ay(t2 - tau, H) @ fy
            return np.concatenate([vx, vy])

        res, _ = quad_vec(integrand, t1, t2, epsabs=1e-5, epsrel=1e-5)
        return res[:2], res[2:]

    def desired_lateral_momentum(
        self, next_support_foot: str, H: float = None, T_step: float = None
    ) -> float:
        """
        Desired frontal-plane contact angular momentum \bar{L}_{x,S}(T_{k+1}^-)
        for periodic lateral stepping with step width W.
        """
        H = self.H if H is None else H
        T_step = self.T_step if T_step is None else T_step
        l = np.sqrt(self.g / H)
        mag = 0.5 * self.m * H * self.W * (l * np.sinh(l * T_step)) / (1.0 + np.cosh(l * T_step))
        return mag if next_support_foot == "right" else -mag

    def compute_foot_placement(
        self,
        t_k_minus: float,
        x_minus: np.ndarray,
        y_minus: np.ndarray,
        vx_des: float,
        next_support_foot: str,
        vel_func,
        use_drs_compensation: bool = True,
        meta_mods: dict = None,
    ) -> tuple[float, float, float, float]:
        """
        Computes discrete step lengths (u_x, u_y) and post-impact relative CoM positions
        (x_SwC, y_SwC) at landing instant T_k^- using Eqs. (19)-(20) and (28)-(29).
        """
        meta_mods = meta_mods or {}
        H = np.clip(self.H + meta_mods.get("delta_H", 0.0), 0.65, 1.05)
        T_step = np.clip(self.T_step + meta_mods.get("delta_T_step", 0.0), 0.25, 0.60)
        l = np.sqrt(self.g / H)

        t_k1_minus = t_k_minus + T_step

        if use_drs_compensation:
            Vx_next, Vy_next = self.compute_forcing_integrals(t_k_minus, t_k1_minus, vel_func, H)
        else:
            Vx_next, Vy_next = np.zeros(2), np.zeros(2)

        Ly_des = self.m * H * vx_des + meta_mods.get("delta_Ly_des", 0.0)
        Lx_des = self.desired_lateral_momentum(next_support_foot, H, T_step) + meta_mods.get("delta_Lx_des", 0.0)

        denom = self.m * H * l * np.sinh(l * T_step)

        # Swing-foot landing position relative to CoM (Eqs. 19 & 28) + RL Capture-Point override
        x_SwC = (Ly_des - Vx_next[1] - np.cosh(l * T_step) * x_minus[1]) / denom + meta_mods.get("delta_ux", 0.0)
        y_SwC = (-Lx_des + Vy_next[1] + np.cosh(l * T_step) * y_minus[1]) / denom + meta_mods.get("delta_uy", 0.0)

        # Discrete step lengths u_x and u_y (Eqs. 20 & 29)
        u_x = x_minus[0] - x_SwC
        u_y = y_minus[0] - y_SwC

        return u_x, u_y, x_SwC, y_SwC
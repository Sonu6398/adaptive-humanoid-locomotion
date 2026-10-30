import numpy as np
from scipy.optimize import minimize
from controllers.bezier_gait import BezierGaitGenerator


class LowLevelWholeBodyController:
    """
    Lower-Layer Feedback Controller (Gao et al., 2025, Section IV-C).
    Converts middle-layer Bezier task-space trajectories h_d(s) and RL Meta-Controller
    gain/posture modulations into bounded joint torques tau_u (18 active DoFs).
    """
    def __init__(self, num_actuators: int = 18, torque_limit: float = 120.0, friction_coeff: float = 0.6):
        self.na = num_actuators
        self.torque_limit = torque_limit
        self.mu = friction_coeff
        self.bezier = BezierGaitGenerator(order=5, step_clearance=0.04)

        # Nominal PD gains from Section VI-A2 of Gao et al. (Kp = 729*I, Kd = 27*I)
        self.kp_nom = 729.0 * np.ones(self.na)
        self.kd_nom = 27.0 * np.ones(self.na)

        # Effective task-space inertia diagonal approximation for Digit's 18 active channels
        self.m_eff = np.array([
            46.1, 3.5, 3.5, 2.0,      # CoM z, trunk roll/pitch/yaw
            3.2, 3.2, 3.2,            # Swing leg foot x, y, z
            0.5, 0.5, 0.5,            # Swing foot orientation
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # Upper body joints
        ])
        self.w_tau = 1e-4
        self.w_delta = 1.0

    def build_desired_task_vector(
        self,
        s: float,
        H_target: float,
        theta_torso_target: np.ndarray,
        p_sw_init: np.ndarray,
        p_sw_target: np.ndarray,
        clearance_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Constructs the 18-D desired task-space trajectory vector h_d(s) (Eq. 30 in Gao et al.):
          h_d = [z_SC, gamma_Tr(3), x_Sw, y_Sw, z_Sw, gamma_Sw(3), q_upper(8)]^T
        """
        p_sw = self.bezier.swing_foot_trajectory(s, p_sw_init, p_sw_target, clearance_offset)
        h_d = np.zeros(self.na)
        h_d[0] = H_target                        # Constant/regulated CoM height z_SC
        h_d[1] = theta_torso_target[0]           # Trunk roll reference
        h_d[2] = theta_torso_target[1]           # Trunk pitch reference
        h_d[3] = 0.0                             # Trunk yaw reference
        h_d[4:7] = p_sw                          # Swing-foot position [x_Sw, y_Sw, z_Sw]
        h_d[7:10] = np.zeros(3)                  # Flat swing-foot orientation [roll, pitch, yaw]
        h_d[10:18] = np.zeros(8)                 # Fixed upper-body joint angles q_upper
        return h_d

    def compute_io_qp_torques(
        self,
        h_actual: np.ndarray,
        dh_actual: np.ndarray,
        h_desired: np.ndarray,
        dh_desired: np.ndarray,
        ddh_desired: np.ndarray,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        normal_force_fz: float = 450.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Input-Output Linearizing Control with Quadratic Programming (IO-QP)
        from Eqs. (31)-(33) of Gao et al. (2025).
        Solves:
            min_{tau_u, delta} 0.5 * w_tau * ||tau_u||^2 + 0.5 * w_delta * ||delta||^2
            subject to:
                tau_u = M_eff * (v + delta)
                -tau_limit <= tau_u <= tau_limit
                sqrt(tau_u[4]^2 + tau_u[5]^2) <= mu * f_z
        """
        kp = self.kp_nom * np.clip(kp_scale, 0.5, 2.0)
        kd = self.kd_nom * np.clip(kd_scale, 0.5, 2.0)

        h_err = h_actual - h_desired
        dh_err = dh_actual - dh_desired

        # Ideal stabilizing auxiliary input v = -K_p h - K_d \dot{h} + \ddot{h}_d
        v = ddh_desired - kp * h_err - kd * dh_err

        inv_m = 1.0 / self.m_eff
        max_horiz_force = self.mu * max(float(normal_force_fz), 100.0)

        # Unconstrained / box-constrained solution for independent channels:
        # tau_i = (w_delta * inv_m[i] * v[i]) / (w_tau + w_delta * inv_m[i]**2)
        denom = self.w_tau + self.w_delta * (inv_m ** 2)
        tau_opt = (self.w_delta * inv_m * v) / denom
        tau_opt = np.clip(tau_opt, -self.torque_limit, self.torque_limit)

        # Enforce coupled circular friction cone constraint on horizontal contact channels (indices 4, 5)
        f_horiz_sq = tau_opt[4] ** 2 + tau_opt[5] ** 2
        if f_horiz_sq > max_horiz_force ** 2:
            idx = [4, 5]
            inv_m_sub = inv_m[idx]
            v_sub = v[idx]

            def sub_cost(tau_sub):
                delta_sub = inv_m_sub * tau_sub - v_sub
                return 0.5 * self.w_tau * np.dot(tau_sub, tau_sub) + 0.5 * self.w_delta * np.dot(delta_sub, delta_sub)

            def sub_jac(tau_sub):
                delta_sub = inv_m_sub * tau_sub - v_sub
                return self.w_tau * tau_sub + self.w_delta * (inv_m_sub * delta_sub)

            bounds_sub = [(-self.torque_limit, self.torque_limit), (-self.torque_limit, self.torque_limit)]
            cons_sub = {"type": "ineq", "fun": lambda tau_sub: max_horiz_force ** 2 - (tau_sub[0] ** 2 + tau_sub[1] ** 2)}

            scale = max_horiz_force / np.sqrt(f_horiz_sq)
            tau0_sub = tau_opt[idx] * scale

            res = minimize(
                sub_cost,
                tau0_sub,
                jac=sub_jac,
                bounds=bounds_sub,
                constraints=cons_sub,
                method="SLSQP",
                options={"maxiter": 30, "ftol": 1e-6},
            )
            if res.success:
                tau_opt[idx] = res.x
            else:
                tau_opt[idx] = tau0_sub

        tau_u = tau_opt
        delta_qp = inv_m * tau_u - v

        return tau_u, delta_qp

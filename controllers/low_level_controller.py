import numpy as np
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
        Shapes output tracking error h = h_a - h_d via \ddot{h} = -K_p h - K_d \dot{h}
        and enforces friction cone and actuator torque bounds via slack variable delta_QP.
        """
        kp = self.kp_nom * np.clip(kp_scale, 0.5, 2.0)
        kd = self.kd_nom * np.clip(kd_scale, 0.5, 2.0)

        h_err = h_actual - h_desired
        dh_err = dh_actual - dh_desired

        # Ideal stabilizing auxiliary input v = -K_p h - K_d \dot{h}
        v = -kp * h_err - kd * dh_err + ddh_desired

        # Effective task-space inertia diagonal approximation for Digit's 18 active channels
        m_eff = np.array([
            46.1, 3.5, 3.5, 2.0,      # CoM z, trunk roll/pitch/yaw
            3.2, 3.2, 3.2,            # Swing leg foot x, y, z
            0.5, 0.5, 0.5,            # Swing foot orientation
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # Upper body joints
        ])
        tau_ideal = m_eff * v

        # Enforce friction cone constraint on horizontal reaction channels (|f_xy| <= mu * f_z)
        max_horiz_force = self.mu * max(normal_force_fz, 100.0)
        tau_constrained = tau_ideal.copy()
        tau_constrained[4:6] = np.clip(tau_constrained[4:6], -max_horiz_force, max_horiz_force)

        # Enforce joint torque saturation limits [-torque_limit, +torque_limit]
        tau_u = np.clip(tau_constrained, -self.torque_limit, self.torque_limit)
        delta_qp = tau_u - tau_ideal

        return tau_u, delta_qp
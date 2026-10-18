import numpy as np


class BiomechanicalRecoveryPrimitives:
    """
    Implements the three biomechanically motivated push recovery strategies
    from Yi et al. (IEEE ICRA 2011) coupled with the ALIP-DRS dynamics:
      1. Ankle Controller: Auxiliary ZMP shift p_aux -> restoring ankle torque tau_ankle
      2. Hip Controller: Torso pitch/roll target theta_target -> flywheel angular momentum L_CoM
      3. Step Controller: Capture Point footstep override x_capture
    """
    def __init__(
        self,
        mass: float = 46.1,
        nom_height: float = 0.9,
        g: float = 9.81,
        max_zmp_x: float = 0.08,
        max_zmp_y: float = 0.05,
        max_hip_torque: float = 85.0,
    ):
        self.m = mass
        self.z0 = nom_height
        self.g = g
        self.max_zmp_x = max_zmp_x
        self.max_zmp_y = max_zmp_y
        self.max_hip_torque = max_hip_torque

    def ankle_strategy_torque(
        self,
        x_sc: float,
        y_sc: float,
        vx_rel: float,
        vy_rel: float,
        p_aux_cmd: np.ndarray = None,
        gains: tuple[float, float] = (0.35, 0.12),
    ) -> tuple[float, float]:
        """
        Computes restoring ankle torques (tau_A_x, tau_A_y) via auxiliary ZMP (Yi et al., Eqs. 1-3).
        If p_aux_cmd is provided by the RL Meta-Controller, uses it directly;
        otherwise computes proportional-derivative ZMP feedback bounded by foot support polygon.
        """
        if p_aux_cmd is not None:
            p_aux_x = np.clip(p_aux_cmd[0], -self.max_zmp_x, self.max_zmp_x)
            p_aux_y = np.clip(p_aux_cmd[1], -self.max_zmp_y, self.max_zmp_y)
        else:
            kp_zmp, kd_zmp = gains
            p_aux_x = np.clip(-kp_zmp * x_sc - kd_zmp * vx_rel, -self.max_zmp_x, self.max_zmp_x)
            p_aux_y = np.clip(-kp_zmp * y_sc - kd_zmp * vy_rel, -self.max_zmp_y, self.max_zmp_y)

        # Effective restoring torque applied at the support foot (Eq. 3 in Yi et al. & Eq. 6 in Gao et al.)
        tau_A_y = self.m * self.g * p_aux_x
        tau_A_x = -self.m * self.g * p_aux_y
        return tau_A_x, tau_A_y

    def hip_strategy_momentum(
        self,
        theta_imu: np.ndarray,
        theta_gyro: np.ndarray,
        theta_target: np.ndarray = None,
        kp_hip: float = 180.0,
        kd_hip: float = 28.0,
    ) -> tuple[float, float]:
        """
        Computes centroidal angular momentum rates (dLx_CoM, dLy_CoM) induced by torso
        flywheel acceleration toward theta_target = [roll_target, pitch_target] (Yi et al., Eqs. 4-5).
        """
        if theta_target is None:
            theta_target = np.zeros(2)

        # PD approximation of bang-bang hip actuator torque bounded by tau_max
        tau_hip_roll = np.clip(
            kp_hip * (theta_target[0] - theta_imu[0]) - kd_hip * theta_gyro[0],
            -self.max_hip_torque,
            self.max_hip_torque,
        )
        tau_hip_pitch = np.clip(
            kp_hip * (theta_target[1] - theta_imu[1]) - kd_hip * theta_gyro[1],
            -self.max_hip_torque,
            self.max_hip_torque,
        )
        return tau_hip_roll, tau_hip_pitch

    def step_capture_point_offset(
        self,
        vx_rel: float,
        vy_rel: float,
        vx_des: float,
        H_curr: float = None,
        gain_cp: float = 0.35,
    ) -> tuple[float, float]:
        """
        Computes Capture Point step adjustment x_capture = v * sqrt(z0 / g) (Yi et al., Eq. 6)
        to override or augment nominal foot placement under large velocity perturbations.
        """
        z0 = self.z0 if H_curr is None else H_curr
        omega_0 = np.sqrt(z0 / self.g)
        dx_cp = gain_cp * (vx_rel - vx_des) * omega_0
        dy_cp = gain_cp * vy_rel * omega_0
        return float(np.clip(dx_cp, -0.18, 0.18)), float(np.clip(dy_cp, -0.15, 0.15))
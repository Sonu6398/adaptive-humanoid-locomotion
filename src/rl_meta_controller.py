import numpy as np
from controllers.alip_drs_planner import ALIPDRSPlanner
from controllers.recovery_primitives import BiomechanicalRecoveryPrimitives
from simulation.drs_profiles import DynamicSupportSurface


class RLMetaControllerPolicy:
    """
    Parameterized High-Level RL Meta-Controller Policy (Yi et al., 2011 Eqs. 11-15)
    extended to modulate ALIP-DRS CoM height, step duration, controller gains,
    and Ankle/Hip/Step recovery regimes.

    Policy parameter vector theta (10 dimensions):
      theta[0]: kp_zmp       - Ankle strategy position gain
      theta[1]: kd_zmp       - Ankle strategy velocity gain
      theta[2]: k_hip_roll   - Hip strategy frontal angular gain
      theta[3]: k_hip_pitch  - Hip strategy sagittal angular gain
      theta[4]: k_cp_x       - Step strategy sagittal capture-point gain
      theta[5]: k_cp_y       - Step strategy lateral capture-point gain
      theta[6]: k_int_Ly     - Integral momentum bias adaptation gain (sagittal)
      theta[7]: k_int_Lx     - Integral momentum bias adaptation gain (frontal)
      theta[8]: crouch_gain  - CoM height lowering gain under surface pitch/disturbance
      theta[9]: freq_gain    - Step duration shortening gain under high disturbance
    """
    def __init__(self, theta: np.ndarray = None):
        if theta is None:
            self.theta = np.array([0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.05, 0.05, 0.01, 0.01], dtype=np.float64)
        else:
            self.theta = np.array(theta, dtype=np.float64)

        self.bounds_low = np.zeros(10)
        self.bounds_high = np.array([0.80, 0.35, 0.45, 0.45, 0.35, 0.35, 0.65, 0.65, 0.12, 0.12])

    def clip_params(self):
        self.theta = np.clip(self.theta, self.bounds_low, self.bounds_high)

    def select_continuous_reflex(self, x_sc, y_sc, vx_rel, vy_rel, vx_des, max_zmp_x=0.08, max_zmp_y=0.05):
        """Computes continuous-time Ankle ZMP shift p_aux and Hip target angle theta_target."""
        kp_zmp, kd_zmp = self.theta[0], self.theta[1]
        k_hip_r, k_hip_p = self.theta[2], self.theta[3]

        p_aux_x = np.clip(-kp_zmp * x_sc - kd_zmp * (vx_rel - vx_des), -max_zmp_x, max_zmp_x)
        p_aux_y = np.clip(-kp_zmp * y_sc - kd_zmp * vy_rel, -max_zmp_y, max_zmp_y)

        theta_target = np.array([
            np.clip(-k_hip_r * vy_rel, -0.25, 0.25),
            np.clip(-k_hip_p * (vx_rel - vx_des), -0.25, 0.25),
        ])
        return np.array([p_aux_x, p_aux_y]), theta_target

    def select_step_meta_parameters(
        self, x_minus, y_minus, vx_des, next_foot, planner, int_Ly_err, int_Lx_err, pitch_amp_est, push_detected
    ):
        """Computes discrete step-level meta-modulations for the ALIP-DRS planner."""
        k_cp_x, k_cp_y = self.theta[4], self.theta[5]
        k_int_Ly, k_int_Lx = self.theta[6], self.theta[7]
        crouch_g, freq_g = self.theta[8], self.theta[9]

        disturbance_level = min(1.0, abs(pitch_amp_est) / 0.05 + (1.0 if push_detected else 0.0))
        delta_H = -crouch_g * disturbance_level
        delta_T_step = -freq_g * disturbance_level

        H_curr = np.clip(planner.H + delta_H, 0.72, 1.0)
        T_curr = np.clip(planner.T_step + delta_T_step, 0.28, 0.50)

        Ly_target = planner.m * H_curr * vx_des
        Lx_target = planner.desired_lateral_momentum(next_foot, H_curr, T_curr)

        int_Ly_err = np.clip(int_Ly_err + k_int_Ly * (Ly_target - x_minus[1]), -30.0, 30.0)
        int_Lx_err = np.clip(int_Lx_err + k_int_Lx * (Lx_target - y_minus[1]), -25.0, 25.0)

        vx_est = x_minus[1] / (planner.m * H_curr)
        vy_est = -y_minus[1] / (planner.m * H_curr)
        omega_0 = np.sqrt(planner.g / H_curr)

        dx_cp = np.clip(k_cp_x * (vx_est - vx_des) * omega_0, -0.18, 0.18)
        dy_cp = np.clip(k_cp_y * vy_est * omega_0, -0.15, 0.15)

        meta_mods = {
            "delta_H": delta_H,
            "delta_T_step": delta_T_step,
            "delta_Ly_des": int_Ly_err,
            "delta_Lx_des": int_Lx_err,
            "delta_ux": -dx_cp,
            "delta_uy": -dy_cp,
        }
        return meta_mods, int_Ly_err, int_Lx_err, H_curr, T_curr


def evaluate_policy_episode(
    policy: RLMetaControllerPolicy,
    case: str = "D",
    vx_des: float = 0.1,
    push_force_xy: tuple[float, float] = (350.0, 180.0),
    unknown_payload_kg: float = 5.0,
    surface_pitch_deg: float = 2.5,
    num_steps: int = 15,
    dt: float = 0.005,
) -> dict:
    """
    Simulates one episode on a swaying, pitching support surface with impulsive push and payload.
    Returns total RL reward, survival status, and full state trajectories.
    """
    planner = ALIPDRSPlanner(mass=46.1, nom_height=0.9, nom_t_step=0.4, step_width=0.2)
    primitives = BiomechanicalRecoveryPrimitives(mass=46.1, nom_height=0.9)
    surface = DynamicSupportSurface(case=case)

    m_true = planner.m + unknown_payload_kg
    g = planner.g
    pitch_rad = np.deg2rad(surface_pitch_deg)
    pitch_w = 2.0 * np.pi / 4.5

    x_state = np.array([0.0, 0.0])
    y_state = np.array([0.1, 0.0])
    theta_imu = np.zeros(2)
    theta_gyro = np.zeros(2)
    L_com = np.zeros(2)

    int_Ly_err, int_Lx_err = 0.0, 0.0
    H_curr, T_curr = planner.H, planner.T_step
    support_foot = "left"

    total_cost = 0.0
    fell = False
    t_curr = 0.0
    t_hist, x_hist, y_hist, imu_hist = [], [], [], []

    for k in range(num_steps):
        t_start = t_curr
        n_sub = int(round(T_curr / dt))

        for i in range(n_sub):
            t = t_start + i * dt
            dx_ws, dy_ws = surface.velocity(t)
            fx_push = push_force_xy[0] if (1.6 <= t <= 1.7) else 0.0
            fy_push = push_force_xy[1] if (1.6 <= t <= 1.7) else 0.0
            pitch_s = pitch_rad * np.sin(pitch_w * t)

            vx_rel = (x_state[1] - L_com[1]) / (m_true * H_curr) - dx_ws
            vy_rel = (-y_state[1] + L_com[0]) / (m_true * H_curr) - dy_ws

            p_aux, theta_target = policy.select_continuous_reflex(
                x_state[0], y_state[0], vx_rel, vy_rel, vx_des
            )
            tau_Ax, tau_Ay = primitives.ankle_strategy_torque(
                x_state[0], y_state[0], vx_rel, vy_rel, p_aux_cmd=p_aux
            )
            dLx_com, dLy_com = primitives.hip_strategy_momentum(
                theta_imu, theta_gyro, theta_target=theta_target
            )

            payload_moment_y = unknown_payload_kg * g * 0.18
            pitch_dist_y = m_true * g * H_curr * np.sin(pitch_s)
            pitch_dist_x = -m_true * g * H_curr * 0.5 * np.sin(pitch_s)

            dLy_s = m_true * g * x_state[0] + tau_Ay + H_curr * fx_push + payload_moment_y + pitch_dist_y
            dLx_s = -m_true * g * y_state[0] + tau_Ax - H_curr * fy_push + pitch_dist_x

            x_state = x_state + dt * np.array([vx_rel, dLy_s])
            y_state = y_state + dt * np.array([vy_rel, dLx_s])
            L_com = np.clip(L_com + dt * np.array([dLx_com, dLy_com]), -18.0, 18.0)

            I_torso = 3.5
            dd_roll = (dLx_com - 0.2 * fy_push - 40.0 * (theta_imu[0] - 0.5 * pitch_s) - 8.0 * theta_gyro[0]) / I_torso
            dd_pitch = (dLy_com + 0.2 * fx_push - 40.0 * (theta_imu[1] - pitch_s) - 8.0 * theta_gyro[1]) / I_torso
            theta_gyro = theta_gyro + dt * np.array([dd_roll, dd_pitch])
            theta_imu = theta_imu + dt * theta_gyro

            # Stopping Energy Cost (Yi et al., 2011 Eq. 10) + Velocity/Position Tracking Cost
            stopping_energy = np.sum(theta_gyro ** 2) + (g / H_curr) * np.sum(theta_imu ** 2)
            vel_cost = (vx_rel - vx_des) ** 2 + 0.5 * (x_state[0] ** 2 + (abs(y_state[0]) - 0.1) ** 2)
            total_cost += dt * (2.5 * stopping_energy + 4.0 * vel_cost)

            t_hist.append(t)
            x_hist.append(x_state.copy())
            y_hist.append(y_state.copy())
            imu_hist.append(theta_imu.copy())

            # Fall termination conditions
            if abs(x_state[0]) > 0.42 or abs(y_state[0]) > 0.42 or np.max(np.abs(theta_imu)) > 0.70:
                fell = True
                total_cost += 250.0
                break

        if fell:
            break

        t_curr = t_start + T_curr
        x_minus = x_state.copy()
        y_minus = y_state.copy()
        next_foot = "right" if support_foot == "left" else "left"
        push_active = np.linalg.norm(theta_gyro) > 0.35

        meta_mods, int_Ly_err, int_Lx_err, H_curr, T_curr = policy.select_step_meta_parameters(
            x_minus, y_minus, vx_des, next_foot, planner, int_Ly_err, int_Lx_err, pitch_rad, push_active
        )

        u_x, u_y, _, _ = planner.compute_foot_placement(
            t_k_minus=t_curr,
            x_minus=x_minus,
            y_minus=y_minus,
            vx_des=vx_des,
            next_support_foot=next_foot,
            vel_func=surface.velocity,
            use_drs_compensation=True,
            meta_mods=meta_mods,
        )

        x_state = x_minus + np.array([-u_x, 0.0])
        y_state = y_minus + np.array([-u_y, 0.0])
        support_foot = next_foot

    return {
        "reward": -total_cost,
        "cost": total_cost,
        "fell": fell,
        "t": np.array(t_hist),
        "x": np.array(x_hist),
        "y": np.array(y_hist),
        "imu": np.array(imu_hist),
    }


def train_stochastic_policy_gradient(num_episodes: int = 45, pop_size: int = 10, sigma: float = 0.04, lr: float = 0.08):
    """
    Trains the high-level RL Meta-Controller using Stochastic Policy Gradient
    (Section IV-B of Yi et al., 2011) across multi-directional sway, pitch, and pushes.
    """
    rng = np.random.default_rng(42)
    best_policy = RLMetaControllerPolicy()
    reward_history = []

    for ep in range(num_episodes):
        base_eval = evaluate_policy_episode(best_policy)
        reward_history.append(base_eval["reward"])

        epsilons = rng.normal(0.0, sigma, size=(pop_size, len(best_policy.theta)))
        rewards = np.zeros(pop_size)

        for i in range(pop_size):
            cand_policy = RLMetaControllerPolicy(best_policy.theta + epsilons[i])
            cand_policy.clip_params()
            res = evaluate_policy_episode(cand_policy)
            rewards[i] = res["reward"]

        std_r = np.std(rewards) + 1e-6
        adv = (rewards - np.mean(rewards)) / std_r
        grad = (1.0 / (pop_size * sigma)) * (epsilons.T @ adv)

        best_policy.theta = best_policy.theta + lr * grad
        best_policy.clip_params()

    return best_policy, np.array(reward_history)
import numpy as np
from controllers.alip_drs_planner import ALIPDRSPlanner
from controllers.recovery_primitives import BiomechanicalRecoveryPrimitives
from simulation.drs_profiles import DynamicSupportSurface


class PerturbedHumanoidDRSSimulator:
    """
    Simulates full coupled CoM, Contact Angular Momentum (L_S), Centroidal Momentum (L_CoM),
    and Torso Pitch/Roll dynamics on a moving, pitching, and perturbed support surface.
    """
    def __init__(
        self,
        case: str = "A",
        unknown_payload_kg: float = 0.0,
        push_force_xy: tuple[float, float] = (0.0, 0.0),
        push_time_window: tuple[float, float] = (2.0, 2.1),
        surface_pitch_deg: float = 0.0,
        surface_pitch_period: float = 4.0,
        delta_A: float = 0.0,
        delta_t: float = 0.0,
    ):
        self.planner = ALIPDRSPlanner(mass=46.1, nom_height=0.9, nom_t_step=0.4, step_width=0.2)
        self.nominal_surface = DynamicSupportSurface(case=case, delta_A=0.0, delta_t=0.0)
        self.actual_surface = DynamicSupportSurface(case=case, delta_A=delta_A, delta_t=delta_t)
        self.primitives = BiomechanicalRecoveryPrimitives(mass=46.1, nom_height=0.9)

        self.payload = unknown_payload_kg
        self.push_fx, self.push_fy = push_force_xy
        self.push_t0, self.push_t1 = push_time_window
        self.pitch_amp_rad = np.deg2rad(surface_pitch_deg)
        self.pitch_w = 2.0 * np.pi / surface_pitch_period

    def external_disturbance(self, t: float) -> tuple[float, float, float]:
        """Returns (F_push_x, F_push_y, surface_pitch_angle_rad) at time t."""
        fx = self.push_fx if (self.push_t0 <= t <= self.push_t1) else 0.0
        fy = self.push_fy if (self.push_t0 <= t <= self.push_t1) else 0.0
        pitch_s = self.pitch_amp_rad * np.sin(self.pitch_w * t)
        return fx, fy, pitch_s

    def run_episode(
        self,
        vx_des: float = 0.1,
        mode: str = "adaptive_reflexive",
        num_steps: int = 20,
        dt: float = 0.005,
    ) -> dict:
        """
        Modes:
          - 'static_baseline': Assumes stationary ground, no ankle/hip reflexes
          - 'fixed_alip_drs': Gao et al. (2025) fixed-parameter ALIP-DRS
          - 'adaptive_reflexive': ALIP-DRS + Ankle ZMP + Hip Flywheel + Momentum Bias Adaptation
        """
        m_true = self.planner.m + self.payload
        g = self.planner.g

        # State vectors: x = [x_SC, L_{y,S}], y = [y_SC, L_{x,S}]
        x_state = np.array([0.0, 0.0])
        y_state = np.array([0.1, 0.0])
        theta_imu = np.zeros(2)     # [roll, pitch]
        theta_gyro = np.zeros(2)    # [d_roll, d_pitch]
        L_com = np.zeros(2)         # [L_{x,CoM}, L_{y,CoM}]

        # Integral momentum-error estimator for adaptive payload/drift compensation
        int_Ly_err = 0.0
        int_Lx_err = 0.0

        support_foot = "left"
        t_hist, x_hist, y_hist, imu_hist = [], [], [], []

        for k in range(num_steps):
            H_curr = self.planner.H
            if mode == "adaptive_reflexive" and self.pitch_amp_rad > 0.0:
                # Lower CoM height by 6 cm when surface pitching is detected to improve stability
                H_curr = self.planner.H - 0.06

            t_start = k * self.planner.T_step
            t_end = (k + 1) * self.planner.T_step
            n_sub = int(round(self.planner.T_step / dt))

            for i in range(n_sub):
                t = t_start + i * dt
                dx_ws, dy_ws = self.actual_surface.velocity(t)
                fx_push, fy_push, pitch_s = self.external_disturbance(t)

                # Relative CoM velocities (Eq. 8 in Gao et al.)
                vx_rel = (x_state[1] - L_com[1]) / (m_true * H_curr) - dx_ws
                vy_rel = (-y_state[1] + L_com[0]) / (m_true * H_curr) - dy_ws

                # Reflexive Ankle & Hip actions
                if mode == "adaptive_reflexive":
                    tau_Ax, tau_Ay = self.primitives.ankle_strategy_torque(
                        x_state[0], y_state[0], vx_rel, vy_rel
                    )
                    theta_target = np.array([
                        np.clip(-0.15 * vy_rel, -0.2, 0.2),
                        np.clip(-0.15 * (vx_rel - vx_des), -0.2, 0.2),
                    ])
                    dLx_com, dLy_com = self.primitives.hip_strategy_momentum(
                        theta_imu, theta_gyro, theta_target
                    )
                else:
                    tau_Ax, tau_Ay = 0.0, 0.0
                    dLx_com, dLy_com = -25.0 * L_com[0], -25.0 * L_com[1]

                # Unknown payload induces a forward gravitational moment offset (~0.18 m lever arm)
                payload_moment_y = self.payload * g * 0.18
                # Surface pitching induces unmodeled gravitational & vertical CoM coupling
                pitch_dist_y = m_true * g * H_curr * np.sin(pitch_s)
                pitch_dist_x = -m_true * g * H_curr * 0.5 * np.sin(pitch_s)

                # Continuous-phase dynamics (Eqs. 6-10 in Gao et al. + disturbances)
                dx_sc = vx_rel
                dLy_s = (
                    m_true * g * x_state[0]
                    + tau_Ay
                    + H_curr * fx_push
                    + payload_moment_y
                    + pitch_dist_y
                )

                dy_sc = vy_rel
                dLx_s = (
                    -m_true * g * y_state[0]
                    + tau_Ax
                    - H_curr * fy_push
                    + pitch_dist_x
                )

                # Euler integration
                x_state = x_state + dt * np.array([dx_sc, dLy_s])
                y_state = y_state + dt * np.array([dy_sc, dLx_s])
                L_com = np.clip(L_com + dt * np.array([dLx_com, dLy_com]), -15.0, 15.0)

                # Torso IMU kinematics driven by surface pitch, push, and hip recovery
                I_torso = 3.5
                dd_roll = (dLx_com - 0.2 * fy_push - 40.0 * (theta_imu[0] - 0.5 * pitch_s) - 8.0 * theta_gyro[0]) / I_torso
                dd_pitch = (dLy_com + 0.2 * fx_push - 40.0 * (theta_imu[1] - pitch_s) - 8.0 * theta_gyro[1]) / I_torso
                theta_gyro = theta_gyro + dt * np.array([dd_roll, dd_pitch])
                theta_imu = theta_imu + dt * theta_gyro

                t_hist.append(t)
                x_hist.append(x_state.copy())
                y_hist.append(y_state.copy())
                imu_hist.append(theta_imu.copy())

            # Pre-impact state at t = T_k^-
            x_minus = x_state.copy()
            y_minus = y_state.copy()
            next_foot = "right" if support_foot == "left" else "left"

            # Meta-parameter modulations for adaptive_reflexive mode
            meta_mods = {}
            if mode == "adaptive_reflexive":
                Ly_target = self.planner.m * H_curr * vx_des
                Lx_target = self.planner.desired_lateral_momentum(next_foot, H_curr, self.planner.T_step)
                int_Ly_err = np.clip(int_Ly_err + 0.35 * (Ly_target - x_minus[1]), -25.0, 25.0)
                int_Lx_err = np.clip(int_Lx_err + 0.25 * (Lx_target - y_minus[1]), -20.0, 20.0)

                vx_est = (x_minus[1] / (self.planner.m * H_curr))
                vy_est = (-y_minus[1] / (self.planner.m * H_curr))
                dx_cp, dy_cp = self.primitives.step_capture_point_offset(vx_est, vy_est, vx_des, H_curr, gain_cp=0.12)

                meta_mods = {
                    "delta_H": H_curr - self.planner.H,
                    "delta_Ly_des": int_Ly_err,
                    "delta_Lx_des": int_Lx_err,
                    "delta_ux": -dx_cp,
                    "delta_uy": -dy_cp,
                }

            u_x, u_y, _, _ = self.planner.compute_foot_placement(
                t_k_minus=t_end,
                x_minus=x_minus,
                y_minus=y_minus,
                vx_des=vx_des,
                next_support_foot=next_foot,
                vel_func=self.nominal_surface.velocity,
                use_drs_compensation=(mode != "static_baseline"),
                meta_mods=meta_mods,
            )

            # Discrete impact switching (Eq. 12)
            x_state = x_minus + np.array([-u_x, 0.0])
            y_state = y_minus + np.array([-u_y, 0.0])
            support_foot = next_foot

        return {
            "t": np.array(t_hist),
            "x": np.array(x_hist),
            "y": np.array(y_hist),
            "imu": np.array(imu_hist),
        }
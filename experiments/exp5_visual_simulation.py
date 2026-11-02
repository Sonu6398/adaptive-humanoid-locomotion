import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.alip_drs_planner import ALIPDRSPlanner
from controllers.bezier_gait import BezierGaitGenerator
from controllers.recovery_primitives import BiomechanicalRecoveryPrimitives
from simulation.drs_profiles import DynamicSupportSurface
from src.rl_meta_controller import RLMetaControllerPolicy


def solve_leg_ik_3d(hip_pos: np.ndarray, foot_pos: np.ndarray, l_thigh: float = 0.48, l_shank: float = 0.48) -> np.ndarray:
    """Computes 3D knee joint position from hip and foot coordinates via 2-link analytical IK."""
    vec = foot_pos - hip_pos
    dist = np.linalg.norm(vec)
    max_reach = l_thigh + l_shank - 1e-4
    if dist >= max_reach:
        return hip_pos + 0.5 * vec

    mid = hip_pos + 0.5 * vec
    h_offset = np.sqrt(max(0.0, l_thigh ** 2 - (0.5 * dist) ** 2))
    unit_vec = vec / (dist + 1e-8)
    bend_dir = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), unit_vec) * unit_vec
    norm_b = np.linalg.norm(bend_dir)
    bend_dir = (bend_dir / norm_b) if norm_b > 1e-6 else np.array([0.0, 0.0, 1.0])
    return mid + h_offset * bend_dir


def run_closed_loop_visual_rollout(
    policy: RLMetaControllerPolicy,
    use_drs_planner: bool,
    case: str = "D",
    vx_des: float = 0.12,
    push_force_xy: tuple[float, float] = (420.0, 220.0),
    unknown_payload_kg: float = 5.0,
    surface_pitch_deg: float = 2.5,
    num_steps: int = 11,
    dt: float = 0.005,
) -> list[dict]:
    """
    Executes the actual closed-loop physics + ALIP-DRS planner + Bezier swing generator
    + RL Meta-Controller and logs 3D world-frame coordinates for visualization.
    """
    planner = ALIPDRSPlanner(mass=46.1, nom_height=0.9, nom_t_step=0.4, step_width=0.2)
    primitives = BiomechanicalRecoveryPrimitives(mass=46.1, nom_height=0.9)
    bezier = BezierGaitGenerator(order=5, step_clearance=0.06)
    surface = DynamicSupportSurface(case=case)

    m_true = planner.m + unknown_payload_kg
    g = planner.g
    pitch_rad = np.deg2rad(surface_pitch_deg)
    pitch_w = 2.0 * np.pi / 4.5

    # Initial states: x = [x_SC, L_{y,S}], y = [y_SC, L_{x,S}]
    x_state = np.array([0.0, 0.0])
    y_state = np.array([0.1, 0.0])
    theta_imu = np.zeros(2)
    theta_gyro = np.zeros(2)
    L_com = np.zeros(2)

    int_Ly_err, int_Lx_err = 0.0, 0.0
    H_curr, T_curr = planner.H, planner.T_step
    support_foot = "left"

    # Track stance and swing foot positions in the DRS platform frame
    p_st_drs = np.array([0.0, 0.10, 0.0])
    p_sw_init_drs = np.array([-0.04, -0.10, 0.0])

    # Pre-compute initial target footstep using ALIP-DRS planner
    u_x0, u_y0, _, _ = planner.compute_foot_placement(
        t_k_minus=T_curr,
        x_minus=x_state,
        y_minus=y_state,
        vx_des=vx_des,
        next_support_foot="right",
        vel_func=surface.velocity,
        use_drs_compensation=use_drs_planner,
    )
    p_sw_target_drs = np.array([p_st_drs[0] + max(u_x0, 0.04), p_st_drs[1] - 0.20, 0.0])

    fell = False
    fall_time = None
    fall_pose = None
    t_curr = 0.0
    frames = []

    for k in range(num_steps):
        t_start = t_curr
        n_sub = int(round(T_curr / dt))

        for i in range(n_sub):
            t = t_start + i * dt
            s = i / float(n_sub)
            x_ws, y_ws = surface.position(t)
            dx_ws, dy_ws = surface.velocity(t)
            pitch_s = pitch_rad * np.sin(pitch_w * t)
            push_active = (1.6 <= t <= 1.75)

            if not fell:
                fx_push = push_force_xy[0] if push_active else 0.0
                fy_push = push_force_xy[1] if push_active else 0.0

                vx_rel = (x_state[1] - L_com[1]) / (m_true * H_curr) - dx_ws
                vy_rel = (-y_state[1] + L_com[0]) / (m_true * H_curr) - dy_ws

                # 1. Query RL Meta-Controller for continuous Ankle ZMP & Hip Flywheel targets
                if use_drs_planner:
                    p_aux, theta_target = policy.select_continuous_reflex(
                        x_state[0], y_state[0], vx_rel, vy_rel, vx_des
                    )
                    tau_Ax, tau_Ay = primitives.ankle_strategy_torque(
                        x_state[0], y_state[0], vx_rel, vy_rel, p_aux_cmd=p_aux
                    )
                    dLx_com, dLy_com = primitives.hip_strategy_momentum(
                        theta_imu, theta_gyro, theta_target=theta_target
                    )
                else:
                    p_aux = np.zeros(2)
                    theta_target = np.zeros(2)
                    tau_Ax, tau_Ay = 0.0, 0.0
                    dLx_com, dLy_com = -25.0 * L_com[0], -25.0 * L_com[1]

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

                # 2. Evaluate Bezier swing-foot trajectory in DRS frame
                p_sw_drs = bezier.swing_foot_trajectory(s, p_sw_init_drs, p_sw_target_drs)

                # 3. Compute CoM position in DRS frame from relative state (x_SC, y_SC, H_curr)
                com_drs = np.array([p_st_drs[0] + x_state[0], p_st_drs[1] + y_state[0], H_curr])

                # Check physical fall condition
                if abs(x_state[0]) > 0.42 or abs(y_state[0]) > 0.42 or np.max(np.abs(theta_imu)) > 0.70:
                    fell = True
                    fall_time = t
                    fall_pose = {
                        "com_drs": com_drs.copy(),
                        "p_st_drs": p_st_drs.copy(),
                        "p_sw_drs": p_sw_drs.copy(),
                        "theta_imu": theta_imu.copy(),
                    }
            else:
                # Animate physical collapse after fall trigger
                dt_f = min(t - fall_time, 1.0)
                com_drs = fall_pose["com_drs"].copy()
                com_drs[0] += 0.28 * dt_f
                com_drs[1] -= 0.25 * dt_f
                com_drs[2] = max(0.32, fall_pose["com_drs"][2] - 0.48 * (dt_f ** 2))
                p_st_drs = fall_pose["p_st_drs"].copy()
                p_sw_drs = fall_pose["p_sw_drs"].copy()
                theta_imu = fall_pose["theta_imu"] + np.array([-0.65 * dt_f, 0.85 * dt_f])
                p_aux = np.zeros(2)

            # Convert DRS-frame points to 3D World Frame (accounting for platform sway & pitch)
            def to_world(pt_drs):
                xw = x_ws + pt_drs[0]
                yw = y_ws + pt_drs[1]
                zw = pt_drs[2] + pt_drs[0] * np.sin(pitch_s)
                return np.array([xw, yw, zw])

            p_left_w = to_world(p_st_drs if support_foot == "left" else p_sw_drs)
            p_right_w = to_world(p_sw_drs if support_foot == "left" else p_st_drs)
            p_target_w = to_world(p_sw_target_drs)
            p_st_w = to_world(p_st_drs)
            com_w = to_world(com_drs)

            # Record every 4th integration step (50 Hz) for smooth animation
            if i % 4 == 0:
                frames.append({
                    "t": t,
                    "x_ws": x_ws,
                    "y_ws": y_ws,
                    "pitch_s": pitch_s,
                    "com_w": com_w,
                    "com_drs": com_drs.copy(),
                    "p_left_w": p_left_w,
                    "p_right_w": p_right_w,
                    "p_st_w": p_st_w,
                    "p_target_w": p_target_w,
                    "theta_imu": theta_imu.copy(),
                    "x_state": x_state.copy(),
                    "y_state": y_state.copy(),
                    "H_curr": H_curr,
                    "p_aux": p_aux.copy(),
                    "push_active": push_active,
                    "fell": fell,
                })

        t_curr = t_start + T_curr
        if not fell:
            # Foot touchdown impact switching
            p_st_old = p_st_drs.copy()
            p_st_drs = p_sw_target_drs.copy()
            p_sw_init_drs = p_st_old.copy()

            x_minus = x_state.copy()
            y_minus = y_state.copy()
            next_foot = "right" if support_foot == "left" else "left"
            push_detected = np.linalg.norm(theta_gyro) > 0.35

            if use_drs_planner:
                meta_mods, int_Ly_err, int_Lx_err, H_curr, T_curr = policy.select_step_meta_parameters(
                    x_minus, y_minus, vx_des, next_foot, planner, int_Ly_err, int_Lx_err, pitch_rad, push_detected
                )
            else:
                meta_mods = {}
                H_curr, T_curr = planner.H, planner.T_step

            u_x, u_y, _, _ = planner.compute_foot_placement(
                t_k_minus=t_curr,
                x_minus=x_minus,
                y_minus=y_minus,
                vx_des=vx_des,
                next_support_foot=next_foot,
                vel_func=surface.velocity,
                use_drs_compensation=use_drs_planner,
                meta_mods=meta_mods,
            )

            # Apply discrete jump map Eq. (12)
            x_state = x_minus + np.array([-u_x, 0.0])
            y_state = y_minus + np.array([-u_y, 0.0])
            support_foot = next_foot

            # Next step landing target on the platform
            u_x_vis = np.clip(u_x, -0.15, 0.22)
            u_y_vis = np.clip(u_y, -0.32, 0.32)
            p_sw_target_drs = np.array([
                p_st_drs[0] + max(0.05, vx_des * T_curr + 0.35 * u_x_vis),
                np.clip(p_st_drs[1] + u_y_vis, -0.45, 0.45),
                0.0,
            ])

    return frames


def draw_3d_humanoid_panel(ax, frame: dict, title: str):
    ax.cla()
    x_ws, y_ws, pitch_s = frame["x_ws"], frame["y_ws"], frame["pitch_s"]

    # 1. Draw Swaying + Pitching Support Platform
    px = np.array([-0.35, 1.25, 1.25, -0.35]) + x_ws
    py = np.array([-0.55, -0.55, 0.55, 0.55]) + y_ws
    pz = (px - x_ws) * np.sin(pitch_s)
    ax.add_collection3d(Poly3DCollection([list(zip(px, py, pz))], facecolors="lightgray", edgecolors="dimgray", alpha=0.40))

    # 2. Draw Narrow Safe Walkway (+/- 0.25 m)
    wx = np.array([-0.30, 1.20, 1.20, -0.30]) + x_ws
    wy = np.array([-0.25, -0.25, 0.25, 0.25]) + y_ws
    wz = (wx - x_ws) * np.sin(pitch_s) + 0.002
    ax.add_collection3d(Poly3DCollection([list(zip(wx, wy, wz))], facecolors="deepskyblue", alpha=0.28))
    ax.plot([x_ws - 0.30, x_ws + 1.20], [y_ws, y_ws], [-0.30 * np.sin(pitch_s), 1.20 * np.sin(pitch_s)], "r--", linewidth=1.2)

    # 3. Draw Planned ALIP-DRS Next Footstep Target Marker (Cyan X)
    pt = frame["p_target_w"]
    ax.scatter([pt[0]], [pt[1]], [pt[2] + 0.01], color="cyan", edgecolors="black", marker="X", s=75, label="ALIP-DRS Step Target")

    # 4. Draw Auxiliary ZMP Shift Vector p_aux at Stance Foot (Green Vector)
    p_st = frame["p_st_w"]
    p_aux = frame["p_aux"]
    if np.linalg.norm(p_aux) > 1e-3 and not frame["fell"]:
        ax.quiver(p_st[0], p_st[1], p_st[2] + 0.02, 3.0 * p_aux[0], 3.0 * p_aux[1], 0.0, color="limegreen", linewidth=2.8)

    # 5. Draw Articulated 3D Humanoid
    com = frame["com_w"]
    roll, pitch = frame["theta_imu"][0], frame["theta_imu"][1]
    hip_l = com + np.array([0.0, 0.09, -0.05])
    hip_r = com + np.array([0.0, -0.09, -0.05])
    knee_l = solve_leg_ik_3d(hip_l, frame["p_left_w"])
    knee_r = solve_leg_ik_3d(hip_r, frame["p_right_w"])

    torso_dir = np.array([np.sin(pitch), -np.sin(roll), np.cos(pitch) * np.cos(roll)])
    neck = com + 0.36 * torso_dir
    head = com + 0.47 * torso_dir
    sh_l = neck + np.array([0.0, 0.15, -0.04])
    sh_r = neck + np.array([0.0, -0.15, -0.04])
    hand_l = sh_l + np.array([-0.18 * np.sin(pitch), 0.04, -0.28])
    hand_r = sh_r + np.array([-0.18 * np.sin(pitch), -0.04, -0.28])

    body_col = "crimson" if frame["fell"] else "midnightblue"
    ax.plot([hip_l[0], knee_l[0], frame["p_left_w"][0]], [hip_l[1], knee_l[1], frame["p_left_w"][1]], [hip_l[2], knee_l[2], frame["p_left_w"][2]], "-o", color="teal", linewidth=3.2, markersize=4)
    ax.plot([hip_r[0], knee_r[0], frame["p_right_w"][0]], [hip_r[1], knee_r[1], frame["p_right_w"][1]], [hip_r[2], knee_r[2], frame["p_right_w"][2]], "-o", color="darkorange", linewidth=3.2, markersize=4)
    ax.plot([hip_l[0], hip_r[0]], [hip_l[1], hip_r[1]], [hip_l[2], hip_r[2]], "-", color=body_col, linewidth=3.8)
    ax.plot([com[0], neck[0]], [com[1], neck[1]], [com[2], neck[2]], "-", color=body_col, linewidth=4.8)
    ax.scatter([head[0]], [head[1]], [head[2]], color=body_col, s=75)
    ax.plot([sh_l[0], sh_r[0]], [sh_l[1], sh_r[1]], [sh_l[2], sh_r[2]], "-", color=body_col, linewidth=2.8)
    ax.plot([sh_l[0], hand_l[0]], [sh_l[1], hand_l[1]], [sh_l[2], hand_l[2]], "-", color="slategray", linewidth=2.2)
    ax.plot([sh_r[0], hand_r[0]], [sh_r[1], hand_r[1]], [sh_r[2], hand_r[2]], "-", color="slategray", linewidth=2.2)

    # 6. Draw 420 N External Push Arrow at t in [1.6, 1.75] s
    if frame["push_active"]:
        ax.quiver(com[0] - 0.35, com[1] - 0.30, com[2] + 0.10, 0.28, 0.22, 0.0, color="gold", linewidth=3.5)
        ax.text(com[0] - 0.38, com[1] - 0.30, com[2] + 0.22, "420 N PUSH!", color="darkorange", fontweight="bold", fontsize=8.5)

    state_str = "FELL (CoM Diverged!)" if frame["fell"] else f"STABLE (H={frame['H_curr']:.2f}m)"
    col_str = "red" if frame["fell"] else "darkgreen"
    ax.set_title(f"{title}\nt = {frame['t']:.2f} s | {state_str}", fontsize=9.5, fontweight="bold", color=col_str)
    ax.set_xlim(-0.35, 1.25)
    ax.set_ylim(-0.60, 0.60)
    ax.set_zlim(-0.10, 1.45)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.view_init(elev=22, azim=-55)


if __name__ == "__main__":
    print("[1/3] Running Actual Closed-Loop Code for Baseline vs. Trained RL Meta-Controller...")
    untrained_policy = RLMetaControllerPolicy(np.zeros(10))
    # Use the exact trained meta-parameters from Step 3
    trained_theta = np.array([0.171, 0.150, 0.112, 0.208, 0.0, 0.0, 0.001, 0.372, 0.038, 0.117])
    trained_policy = RLMetaControllerPolicy(trained_theta)

    frames_base = run_closed_loop_visual_rollout(untrained_policy, use_drs_planner=False)
    frames_rl = run_closed_loop_visual_rollout(trained_policy, use_drs_planner=True)
    n_frames = min(len(frames_base), len(frames_rl))

    t_all = np.array([f["t"] for f in frames_rl[:n_frames]])
    Ly_base = np.array([f["x_state"][1] if not f["fell"] else np.nan for f in frames_base[:n_frames]])
    Ly_rl = np.array([f["x_state"][1] for f in frames_rl[:n_frames]])
    pitch_base = np.array([np.rad2deg(f["theta_imu"][1]) if not f["fell"] else np.nan for f in frames_base[:n_frames]])
    pitch_rl = np.array([np.rad2deg(f["theta_imu"][1]) for f in frames_rl[:n_frames]])
    H_rl = np.array([f["H_curr"] for f in frames_rl[:n_frames]])
    p_aux_rl = np.array([f["p_aux"][0] * 100.0 for f in frames_rl[:n_frames]])  # in cm

    print("[2/3] Building 4-Panel Closed-Loop Simulation & Live Telemetry Dashboard...")
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0])
    ax_3d_base = fig.add_subplot(gs[0, 0], projection="3d")
    ax_3d_rl = fig.add_subplot(gs[0, 1], projection="3d")
    ax_tele1 = fig.add_subplot(gs[1, 0])
    ax_tele2 = fig.add_subplot(gs[1, 1])

    def update(idx):
        draw_3d_humanoid_panel(ax_3d_base, frames_base[idx], "1. Fixed Static ALIP Baseline (No RL / No DRS)")
        draw_3d_humanoid_panel(ax_3d_rl, frames_rl[idx], "2. Proposed RL Meta-Controller + ALIP-DRS")

        # Bottom-Left Telemetry: Live Contact Angular Momentum L_{y,S} & Torso Pitch
        ax_tele1.cla()
        ax_tele1.axvspan(1.6, 1.75, color="orange", alpha=0.35, label="420 N Push")
        ax_tele1.plot(t_all[:idx + 1], Ly_base[:idx + 1], "k--", linewidth=1.5, label="Baseline L_{y,S} (Diverges)")
        ax_tele1.plot(t_all[:idx + 1], Ly_rl[:idx + 1], "r-", linewidth=1.8, label="RL + ALIP-DRS L_{y,S}")
        ax_tele1.plot(t_all[:idx + 1], pitch_rl[:idx + 1], "b-", linewidth=1.4, label="RL Torso Pitch (deg)")
        ax_tele1.set_xlim(0.0, t_all[-1])
        ax_tele1.set_ylim(-28.0, 48.0)
        ax_tele1.set_title("Live Closed-Loop Momentum L_{y,S} & Torso Pitch Recovery", fontsize=9.5, fontweight="bold")
        ax_tele1.set_xlabel("Time (s)", fontsize=8.5)
        ax_tele1.set_ylabel("L_{y,S} (kg*m^2/s) / Pitch (deg)", fontsize=8.5)
        ax_tele1.legend(loc="upper right", fontsize=7.5)
        ax_tele1.grid(True, alpha=0.3)

        # Bottom-Right Telemetry: Live RL Meta-Controller Outputs (CoM Height Crouch & Ankle ZMP p_aux)
        ax_tele2.cla()
        ax_tele2.axvspan(1.6, 1.75, color="orange", alpha=0.35, label="420 N Push")
        ax_tele2.plot(t_all[:idx + 1], (H_rl[:idx + 1] - 0.90) * 100.0, "purple", linewidth=1.8, label="RL CoM Height Shift dH (cm)")
        ax_tele2.plot(t_all[:idx + 1], p_aux_rl[:idx + 1], "limegreen", linewidth=1.8, label="RL Ankle ZMP Shift p_aux_x (cm)")
        ax_tele2.set_xlim(0.0, t_all[-1])
        ax_tele2.set_ylim(-9.0, 9.0)
        ax_tele2.set_title("Live RL Meta-Controller Adaptations (CoM Crouch & Ankle ZMP)", fontsize=9.5, fontweight="bold")
        ax_tele2.set_xlabel("Time (s)", fontsize=8.5)
        ax_tele2.set_ylabel("Adaptation Offset (cm)", fontsize=8.5)
        ax_tele2.legend(loc="upper right", fontsize=7.5)
        ax_tele2.grid(True, alpha=0.3)
        return ax_3d_base, ax_3d_rl, ax_tele1, ax_tele2

    print("[3/3] Saving Animated GIF to results/exp5_humanoid_drs_simulation.gif ...")
    step_stride = 2
    frame_indices = list(range(0, n_frames, step_stride))
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=60, blit=False)

    os.makedirs("results", exist_ok=True)
    gif_path = os.path.join("results", "exp5_humanoid_drs_simulation.gif")
    ani.save(gif_path, writer=PillowWriter(fps=16))
    print(f"[SUCCESS] Saved closed-loop visual simulation GIF to {gif_path}")

    # Save a high-res snapshot right after the push recovery (around t = 2.1 s)
    snap_idx = min(int(2.1 / 0.02), n_frames - 1)
    update(snap_idx)
    plt.tight_layout()
    snap_path = os.path.join("results", "exp5_simulation_snapshot.png")
    plt.savefig(snap_path, dpi=200)
    print(f"[SUCCESS] Saved dashboard snapshot to {snap_path}")
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.humanoid_drs_sim import PerturbedHumanoidDRSSimulator


def run_benchmark_suite():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # --- Scenario 1: 400 N Sudden Push on Swaying Surface (Case A, Fig. 11 in Gao et al.) ---
    sim_push = PerturbedHumanoidDRSSimulator(
        case="A", push_force_xy=(400.0, 150.0), push_time_window=(2.0, 2.1)
    )
    res_push_fixed = sim_push.run_episode(vx_des=0.1, mode="fixed_alip_drs", num_steps=20)
    res_push_adapt = sim_push.run_episode(vx_des=0.1, mode="adaptive_reflexive", num_steps=20)

    axes[0, 0].axvspan(2.0, 2.1, color="orange", alpha=0.4, label="400 N Push")
    axes[0, 0].plot(res_push_fixed["t"], res_push_fixed["x"][:, 0], "k--", label="Fixed ALIP-DRS")
    axes[0, 0].plot(res_push_adapt["t"], res_push_adapt["x"][:, 0], "r-", linewidth=1.8, label="Adaptive Reflexive")
    axes[0, 0].set_title("Scenario 1: 400 N Push (CoM x_SC)")
    axes[0, 0].set_ylabel("x_SC (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].axvspan(2.0, 2.1, color="orange", alpha=0.4)
    axes[1, 0].plot(res_push_fixed["t"], res_push_fixed["x"][:, 1], "k--", label="Fixed ALIP-DRS")
    axes[1, 0].plot(res_push_adapt["t"], res_push_adapt["x"][:, 1], "r-", linewidth=1.8, label="Adaptive Reflexive")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("L_{y,S} (kg*m^2/s)")
    axes[1, 0].grid(True, alpha=0.3)

    # --- Scenario 2: Unknown 10-kg Payload on Swaying Surface (Case C, Fig. 12 in Gao et al.) ---
    sim_load = PerturbedHumanoidDRSSimulator(case="C", unknown_payload_kg=10.0)
    res_load_fixed = sim_load.run_episode(vx_des=0.0, mode="fixed_alip_drs", num_steps=20)
    res_load_adapt = sim_load.run_episode(vx_des=0.0, mode="adaptive_reflexive", num_steps=20)

    axes[0, 1].plot(res_load_fixed["t"], res_load_fixed["x"][:, 0], "k--", label="Fixed ALIP-DRS (Drifts)")
    axes[0, 1].plot(res_load_adapt["t"], res_load_adapt["x"][:, 0], "r-", linewidth=1.8, label="Adaptive Reflexive")
    axes[0, 1].set_title("Scenario 2: 10-kg Unknown Load (v_des = 0)")
    axes[0, 1].set_ylabel("x_SC (m)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].axhline(0.0, color="blue", linestyle=":", alpha=0.6, label="Desired L_{y,S} = 0")
    axes[1, 1].plot(res_load_fixed["t"], res_load_fixed["x"][:, 1], "k--", label="Fixed ALIP-DRS (~20 offset)")
    axes[1, 1].plot(res_load_adapt["t"], res_load_adapt["x"][:, 1], "r-", linewidth=1.8, label="Adaptive Reflexive")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("L_{y,S} (kg*m^2/s)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # --- Scenario 3: Unknown Surface Pitching 3 deg + Sway (Fig. 19 in Gao et al.) ---
    sim_pitch = PerturbedHumanoidDRSSimulator(
        case="C", surface_pitch_deg=3.0, surface_pitch_period=5.0, delta_A=0.02
    )
    res_pitch_fixed = sim_pitch.run_episode(vx_des=0.0, mode="fixed_alip_drs", num_steps=20)
    res_pitch_adapt = sim_pitch.run_episode(vx_des=0.0, mode="adaptive_reflexive", num_steps=20)

    axes[0, 2].plot(res_pitch_fixed["y"][:, 0], res_pitch_fixed["y"][:, 1], "k--", alpha=0.65, label="Fixed ALIP-DRS")
    axes[0, 2].plot(res_pitch_adapt["y"][:, 0], res_pitch_adapt["y"][:, 1], "r-", linewidth=1.4, label="Adaptive Reflexive")
    axes[0, 2].set_title("Scenario 3: Phase Portrait under 3 deg Pitch + Sway")
    axes[0, 2].set_xlabel("y_SC (m)")
    axes[0, 2].set_ylabel("L_{x,S} (kg*m^2/s)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 2].plot(res_pitch_fixed["t"], np.rad2deg(res_pitch_fixed["imu"][:, 1]), "k--", label="Fixed ALIP-DRS")
    axes[1, 2].plot(res_pitch_adapt["t"], np.rad2deg(res_pitch_adapt["imu"][:, 1]), "r-", linewidth=1.8, label="Adaptive Reflexive")
    axes[1, 2].set_title("Torso Pitch Stability (theta_IMU)")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Torso Pitch (deg)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("results", "exp2_disturbances_and_recovery.png")
    plt.savefig(out_path, dpi=200)
    print(f"[SUCCESS] Saved disturbance & recovery benchmark plot to {out_path}")


if __name__ == "__main__":
    run_benchmark_suite()
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.alip_drs_planner import ALIPDRSPlanner
from simulation.drs_profiles import DynamicSupportSurface


def run_simulation(case: str, vx_des: float, use_drs: bool, num_steps: int = 20):
    planner = ALIPDRSPlanner(mass=46.1, nom_height=0.9, nom_t_step=0.4, step_width=0.2)
    surface = DynamicSupportSurface(case=case)

    x_curr = np.array([0.0, 0.0])   # [x_SC, L_{y,S}]
    y_curr = np.array([0.1, 0.0])   # [y_SC, L_{x,S}]
    support_foot = "left"

    t_hist, x_hist, y_hist = [], [], []

    for k in range(num_steps):
        t_start = k * planner.T_step
        t_end = (k + 1) * planner.T_step
        t_eval = np.linspace(t_start, t_end, 25)

        for t in t_eval:
            Vx, Vy = planner.compute_forcing_integrals(t_start, t, surface.velocity)
            x_t = planner.exp_Ax(t - t_start) @ x_curr + Vx
            y_t = planner.exp_Ay(t - t_start) @ y_curr + Vy
            t_hist.append(t)
            x_hist.append(x_t)
            y_hist.append(y_t)

        x_minus = x_hist[-1].copy()
        y_minus = y_hist[-1].copy()
        next_foot = "right" if support_foot == "left" else "left"

        u_x, u_y, _, _ = planner.compute_foot_placement(
            t_k_minus=t_end,
            x_minus=x_minus,
            y_minus=y_minus,
            vx_des=vx_des,
            next_support_foot=next_foot,
            vel_func=surface.velocity,
            use_drs_compensation=use_drs,
        )

        # Discrete impact switching Eq. (12)
        x_curr = x_minus + np.array([-u_x, 0.0])
        y_curr = y_minus + np.array([-u_y, 0.0])
        support_foot = next_foot

    return np.array(t_hist), np.array(x_hist), np.array(y_hist)


if __name__ == "__main__":
    cases = [("A", 0.1), ("B", 0.3), ("C", 0.0), ("D", 0.15)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)

    for idx, (case_name, vx_des) in enumerate(cases):
        t_drs, x_drs, y_drs = run_simulation(case_name, vx_des, use_drs=True)
        t_base, x_base, y_base = run_simulation(case_name, vx_des, use_drs=False)

        # Top row: Sagittal Contact Angular Momentum L_{y,S}
        ax_top = axes[0, idx]
        ax_top.plot(t_base, x_base[:, 1], "k--", alpha=0.75, label="Static ALIP Baseline")
        ax_top.plot(t_drs, x_drs[:, 1], "r-", linewidth=1.8, label="ALIP-DRS Planner")
        ax_top.set_title(f"Case {case_name} (v_des = {vx_des} m/s)")
        if idx == 0:
            ax_top.set_ylabel("L_{y,S} (kg*m^2/s)")
            ax_top.legend(loc="upper right", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom row: Lateral Contact Angular Momentum L_{x,S}
        ax_bot = axes[1, idx]
        ax_bot.plot(t_base, y_base[:, 1], "k--", alpha=0.75, label="Static ALIP Baseline")
        ax_bot.plot(t_drs, y_drs[:, 1], "r-", linewidth=1.8, label="ALIP-DRS Planner")
        ax_bot.set_xlabel("Time (s)")
        if idx == 0:
            ax_bot.set_ylabel("L_{x,S} (kg*m^2/s)")
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join("results", "exp1_alip_drs_vs_baseline.png")
    plt.savefig(out_file, dpi=200)
    print(f"[SUCCESS] Plot saved to {out_file}")
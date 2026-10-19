import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rl_meta_controller import (
    RLMetaControllerPolicy,
    evaluate_policy_episode,
    train_stochastic_policy_gradient,
)


def compute_stability_region(policy: RLMetaControllerPolicy, grid_size: int = 13):
    """
    Computes the 2D frontal vs. lateral impulse push stability region on swaying DRS
    (analogous to Fig. 3 in Yi et al., 2011).
    """
    fx_vals = np.linspace(-550.0, 550.0, grid_size)
    fy_vals = np.linspace(-450.0, 450.0, grid_size)
    stable_pts = []

    for fx in fx_vals:
        for fy in fy_vals:
            res = evaluate_policy_episode(
                policy,
                case="D",
                vx_des=0.1,
                push_force_xy=(fx, fy),
                unknown_payload_kg=5.0,
                surface_pitch_deg=2.0,
                num_steps=10,
                dt=0.01,
            )
            if not res["fell"]:
                stable_pts.append((fy, fx))
    return np.array(stable_pts)


if __name__ == "__main__":
    print("[1/3] Training RL Meta-Controller via Stochastic Policy Gradient...")
    untrained_policy = RLMetaControllerPolicy(np.zeros(10))
    trained_policy, reward_curve = train_stochastic_policy_gradient(num_episodes=40, pop_size=10)
    print(f"      Initial Reward: {reward_curve[0]:.2f} -> Final Trained Reward: {reward_curve[-1]:.2f}")
    print(f"      Learned Meta-Parameters: {np.round(trained_policy.theta, 3)}")

    print("[2/3] Evaluating Untrained (Fixed ALIP-DRS) vs. Trained RL Meta-Controller...")
    eval_fixed = evaluate_policy_episode(untrained_policy, push_force_xy=(420.0, 220.0))
    eval_rl = evaluate_policy_episode(trained_policy, push_force_xy=(420.0, 220.0))

    print("[3/3] Computing 2D Push Recovery Stability Region (Fig. 3 of Yi et al.)...")
    region_fixed = compute_stability_region(untrained_policy, grid_size=13)
    region_rl = compute_stability_region(trained_policy, grid_size=13)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: RL Training Curve
    axes[0].plot(np.arange(1, len(reward_curve) + 1), reward_curve, "b-o", markersize=4, linewidth=1.8)
    axes[0].set_title("RL Meta-Controller Learning Curve")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Episode Return (-Stopping Energy & Tracking Cost)")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Trajectory Response under Combined Sway + Pitch + 420N Push + 5kg Load
    axes[1].axvspan(1.6, 1.7, color="orange", alpha=0.4, label="Impulsive Push")
    axes[1].plot(eval_fixed["t"], np.rad2deg(eval_fixed["imu"][:, 1]), "k--", label="Fixed ALIP-DRS (No RL)")
    axes[1].plot(eval_rl["t"], np.rad2deg(eval_rl["imu"][:, 1]), "r-", linewidth=1.8, label="Trained RL Meta-Controller")
    axes[1].set_title("Torso Pitch Recovery on Moving/Pitching Surface")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Torso Pitch Angle (deg)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: 2D Push Stability Region (Yi et al. Fig. 3 comparison)
    if len(region_rl) > 0:
        axes[2].scatter(region_rl[:, 0], region_rl[:, 1], c="red", s=45, alpha=0.7, label="With RL Meta-Controller")
    if len(region_fixed) > 0:
        axes[2].scatter(region_fixed[:, 0], region_fixed[:, 1], c="black", marker="x", s=35, label="Without RL (Fixed ALIP-DRS)")
    axes[2].set_title("Recoverable Push Stability Region on Swaying DRS")
    axes[2].set_xlabel("Lateral Push Force F_y (N)")
    axes[2].set_ylabel("Sagittal Push Force F_x (N)")
    axes[2].set_xlim(-500, 500)
    axes[2].set_ylim(-600, 600)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("results", "exp3_rl_meta_controller_results.png")
    plt.savefig(out_path, dpi=200)
    print(f"[SUCCESS] Saved RL Meta-Controller evaluation plot to {out_path}")
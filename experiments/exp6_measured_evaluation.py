"""
Experiment 6: Measured controller comparison.

Every number in the output is computed by running the simulator, so this
replaces the hand-written table that exp4 used to save as
results/final_ablation_metrics.csv.

Controllers compared (all share the same ALIP-DRS planner and low-level limits):
  1. Fixed ALIP-DRS        - meta-parameters all zero (no adaptation)
  2. Hand-tuned reflexes   - one fixed, hand-picked parameter vector
  3. RL meta-controller    - parameters learned by train_stochastic_policy_gradient

For each surface case (A-F and non-periodic) we record:
  - survival with sway + 2.5 deg pitch + 5 kg unknown payload, no push
  - survival with the default 350 N + 180 N push at t = 1.6 s
  - time of fall (or episode length if it survives)
  - largest push magnitude survived (same 350:180 direction, 20 N steps,
    stopping at the first fall)
  - forward CoM RMS error and peak torso tilt in the default-push episode

Run from the repository root:
    python experiments/exp6_measured_evaluation.py
"""

import csv
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

CASES = ["A", "B", "C", "D", "E", "F", "NONPERIODIC"]
PUSH_DIR = np.array([350.0, 180.0]) / 350.0  # forward:lateral ratio of the default push
PUSH_STEP_N = 20
PUSH_MAX_N = 800
HAND_TUNED_THETA = np.array([0.12, 0.05, 0.08, 0.08, 0.05, 0.05, 0.15, 0.15, 0.02, 0.02])


def run_episode(policy, case, push_n):
    push = (float(push_n * PUSH_DIR[0]), float(push_n * PUSH_DIR[1]))
    r = evaluate_policy_episode(policy, case=case, push_force_xy=push)
    t = r["t"]
    x = np.asarray(r["x"])[:, 0] if len(r["x"]) else np.zeros(1)
    imu = np.asarray(r["imu"]) if len(r["imu"]) else np.zeros(1)
    return {
        "fell": bool(r["fell"]),
        "time_s": float(t[-1]) if len(t) else 0.0,
        "cost": float(r["cost"]),
        "rms_x_m": float(np.sqrt(np.mean(x ** 2))),
        "peak_torso_deg": float(np.degrees(np.max(np.abs(imu)))),
    }


def max_push_survived(policy, case):
    best = None
    for f in range(0, PUSH_MAX_N + 1, PUSH_STEP_N):
        if run_episode(policy, case, f)["fell"]:
            break
        best = f
    return best  # None means it falls even with no push


def main():
    os.makedirs("results", exist_ok=True)

    print("[1/3] Training RL meta-controller (seeded, deterministic)...")
    trained, curve = train_stochastic_policy_gradient(num_episodes=40, pop_size=10)
    print(f"      reward {curve[0]:.1f} -> {curve[-1]:.1f}")

    controllers = {
        "Fixed ALIP-DRS (no adaptation)": RLMetaControllerPolicy(np.zeros(10)),
        "Hand-tuned reflexes + ALIP-DRS": RLMetaControllerPolicy(HAND_TUNED_THETA.copy()),
        "RL meta-controller + ALIP-DRS": trained,
    }

    print("[2/3] Evaluating controllers on every surface case...")
    rows = []
    for name, pol in controllers.items():
        for case in CASES:
            no_push = run_episode(pol, case, 0.0)
            default = run_episode(pol, case, 350.0)
            mp = max_push_survived(pol, case)
            rows.append({
                "controller": name,
                "case": case,
                "survives_no_push": not no_push["fell"],
                "survives_default_push": not default["fell"],
                "time_default_push_s": round(default["time_s"], 2),
                "max_push_survived_N": "falls without push" if mp is None else mp,
                "rms_x_default_push_m": round(default["rms_x_m"], 4),
                "peak_torso_default_push_deg": round(default["peak_torso_deg"], 2),
            })
            print(f"      {name:34s} {case:12s} no-push={'ok' if not no_push['fell'] else 'FALL'}"
                  f"  push350={'ok' if not default['fell'] else 'FALL'}  max={rows[-1]['max_push_survived_N']}")

    csv_path = os.path.join("results", "measured_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[SUCCESS] Saved measured metrics to {csv_path}")

    print("[3/3] Plotting max push survived per case...")
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.27
    xs = np.arange(len(CASES))
    for i, name in enumerate(controllers):
        vals = [r["max_push_survived_N"] for r in rows if r["controller"] == name]
        vals = [0 if isinstance(v, str) else v for v in vals]
        bars = ax.bar(xs + (i - 1) * width, vals, width, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(CASES)
    ax.set_xlabel("Surface motion case")
    ax.set_ylabel("Largest push survived (N, 350:180 direction)")
    ax.set_title("Measured push tolerance (sway + 2.5 deg pitch + 5 kg payload); 0 = falls without push")
    ax.set_ylim(0, 430)
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join("results", "exp6_measured_push_tolerance.png")
    plt.savefig(out_img, dpi=200)
    print(f"[SUCCESS] Saved plot to {out_img}")


if __name__ == "__main__":
    main()

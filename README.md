# Adaptive Reflexive Control of Humanoid Locomotion on Non-Inertial Support Surfaces

## Foundations of Robotics — Course Project

This project develops adaptive and reflexive control strategies for humanoid locomotion on **non-inertial and dynamically moving support surfaces**, such as ships, trains, and moving platforms. 

Our hierarchical framework combines the analytical **Time-Varying ALIP-DRS footstep planner** (Gao et al., *IEEE T-RO* 2025) with a high-level **Reinforcement Learning (RL) Meta-Controller** and **biomechanical push-recovery reflexes** (Yi et al., *IEEE ICRA* 2011).

## Team

| Name | Roll Number | GitHub |
|---|---|---|
| Sonu | IIT2023103 | [Sonu6398](https://github.com/Sonu6398) |
| Siddhant | IIT2023104 | [siddhantsachan](https://github.com/siddhantsachan) |
| Prasannajeet Pradhan | MRM2025012 | [prasannajeet2002](https://github.com/prasannajeet2002) |

---

## Overall Hierarchical Architecture

```text
        Dynamic Support Surface (Sway, Heave, Pitch, Push, Payload)
                 │
                 ▼
       State / Motion Estimation
                 │
                 ▼
       ┌──────────────────────────────────────────┐
       │            RL Meta-Controller            │
       │  • CoM Height & Step Frequency (ΔH, ΔT)  │
       │  • Momentum Drift Compensation (ΔL_S)    │
       │  • Ankle ZMP / Hip Flywheel / Step Gains │
       └────────────────────┬─────────────────────┘
                            │
                            ▼
       ┌──────────────────────────────────────────┐
       │   Structured Mid- & Low-Level Control    │
       │  • ALIP-DRS Footstep Planner (u_x, u_y)  │
       │  • 5th-Order Bézier Gait Generator h_d(s)│
       │  • IO-QP / IK-PD Joint Torque Controller │
       └────────────────────┬─────────────────────┘
                            │
                            ▼
               Humanoid Robot Actuators
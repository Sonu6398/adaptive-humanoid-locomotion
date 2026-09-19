# Adaptive Reflexive Control of Humanoid Locomotion on Non-Inertial Support Surfaces

## Foundations of Robotics — Course Project

This project focuses on developing adaptive and reflexive control strategies for humanoid locomotion on **non-inertial and dynamically moving support surfaces**, such as ships, trains, and moving platforms.

The goal is to enable a humanoid robot to continuously adapt its balance, center-of-mass motion, gait, and recovery behaviour in response to changes in the motion of the supporting surface while maintaining stable locomotion.

## Team

| Name | Roll Number | GitHub |
|---|---|---|
| Sonu | IIT2023103 | [Sonu6398](https://github.com/Sonu6398) |
| Siddhant | IIT2023104 | [siddhantsachan](https://github.com/siddhantsachan) |
| Prasannajeet Pradhan | MRM2025012 | [prasannajeet2002](https://github.com/prasannajeet2002) |

## Problem Statement

Humanoid robots are generally designed to walk on relatively stationary and predictable surfaces. However, environments such as ships, trains, and dynamically moving platforms introduce additional disturbances due to surface translation, acceleration, rotation, and oscillation.

A controller with fixed parameters may not be able to adequately respond to these changing support dynamics. This project investigates adaptive control mechanisms that allow the robot to modify its locomotion and balance strategies according to the current support-surface conditions.

## Objectives

- Develop a control framework for humanoid locomotion on dynamically moving support surfaces.
- Enable continuous adaptation to changes in support-surface motion.
- Maintain stable center-of-mass and balance behaviour during locomotion.
- Develop reflexive recovery strategies for disturbances and near-fall conditions.
- Investigate reinforcement learning for high-level adaptive decision-making.
- Compare adaptive control with conventional fixed-parameter control approaches.
- Evaluate the proposed approach using simulation-based experiments.

## Proposed Approach

The project will follow a **hierarchical control architecture**.

### High-Level: RL Meta-Controller

A reinforcement learning based meta-controller will select or modify appropriate control parameters and regimes based on the robot and environment state.

Possible outputs include:

- Center-of-mass targets
- Controller gains
- Gait parameters
- Step parameters
- Walking speed
- Recovery mode
- Control regime

### Low-Level: Structured Controller

A conventional low-level controller will convert the high-level decisions into feasible and stable robot actions while respecting physical and joint constraints.

The low-level controller may incorporate techniques such as:

- Whole-body control
- Model Predictive Control (MPC)
- Center-of-mass control
- ZMP/CoP based stability control
- Reflexive balance and recovery mechanisms

### Overall Architecture

```text
        Dynamic Support Surface
                 │
                 ▼
       State / Motion Estimation
                 │
                 ▼
       ┌─────────────────────┐
       │   RL Meta-Controller │
       │                     │
       │ Control regime      │
       │ COM targets         │
       │ Gait parameters     │
       │ Controller gains    │
       │ Recovery mode       │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │  Low-Level Controller│
       │                     │
       │ Balance / COM       │
       │ Gait Control        │
       │ Joint Control       │
       │ Recovery            │
       └──────────┬──────────┘
                  │
                  ▼
          Humanoid Robot
                  │
                  ▼
          Environment State
                  │
                  └─────────── Feedback
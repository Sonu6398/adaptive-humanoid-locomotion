# Literature Review: Adaptive & Reflexive Humanoid Locomotion on Dynamic Surfaces

## 1. Humanoid Locomotion on Dynamic and Non-Inertial Surfaces (DRS)
Classical bipedal walking controllers rely on the Zero Moment Point (ZMP) condition[cite: 2], Capture Point (CP)[cite: 2], or Divergent Component of Motion (DCM)[cite: 2] derived from the Linear Inverted Pendulum (LIP) model[cite: 2]. While effective on static ground, these models assume an inertial support frame[cite: 2]. When the support surface accelerates in an inertial frame—such as on ships, trains, buses, or aircraft—unmodeled surface motion acts as a temporally persistent disturbance that leaves the robot insufficient recovery time and can exceed actuator torque limits[cite: 1].

### Time-Varying ALIP-DRS Framework (Gao et al., IEEE T-RO 2025)
To address horizontal surface oscillation, Gao et al. analytically extended the Angular-Momentum-based Linear Inverted Pendulum (ALIP) model[cite: 2, 3] to swaying rigid surfaces, formulating the **ALIP-DRS** model[cite: 3]:
- **State Formulation**: Uses horizontal CoM position relative to the contact point $(x_{SC}, y_{SC})$ and contact angular momentum $(L_{y,S}, L_{x,S})$[cite: 4]. Contact angular momentum is invariant across foot-landing impacts when vertical CoM velocity is zero ($\Delta L_{y,S} = 0, \Delta L_{x,S} = 0$)[cite: 5].
- **Nonhomogeneous Time-Varying Dynamics**: Horizontal surface velocities $(\dot{x}_{WS}(t), \dot{y}_{WS}(t))$ enter the continuous-phase state equations as time-varying forcing terms $f_x(t) = [-\dot{x}_{WS}(t), 0]^T$ and $f_y(t) = [-\dot{y}_{WS}(t), 0]^T$[cite: 4].
- **Three-Layer Hierarchical Control**:
  1. *Higher-Layer Footstep Planner*: Predicts pre-impact contact angular momentum using the integral of surface motion $V_x(t_1, t_2), V_y(t_1, t_2)$ and computes discrete swing-foot landing positions $(x_{SwC}(T_k^-), y_{SwC}(T_k^-))$[cite: 6, 8].
  2. *Middle-Layer Walking Pattern Planner*: Generates full-body reference trajectories $h_d(s)$ via Bézier polynomials parameterized by gait phase $s \in [0, 1)$, regulating CoM height $z_{SC}$ to a constant $H$ and keeping the trunk upright[cite: 8, 9, 10].
  3. *Lower-Layer Feedback Controller*: Tracks $h_d(s)$ using Input-Output Linearizing Control with Quadratic Programming (IO-QP) or Inverse Kinematics with PD control[cite: 10].

### Vertical Surface Motion & Limitations of Fixed-Parameter Models
For vertically oscillating platforms, Iqbal et al. (2023) derived an analytical time-varying LIP model where vertical ground acceleration modulates effective gravity[cite: 2, 18, 20]. However, in Gao et al. (2025), experiments under **unknown pitching surface motion ($1^\circ\text{--}3^\circ$)**, **uneven/compliant surfaces**, and **unknown external loads (10 kg)** revealed noticeable phase-portrait deviations and steady-state momentum drift because the fixed-parameter ALIP-DRS model assumes constant CoM height ($\dot{z}_{SC} = 0$) and zero centroidal angular momentum ($L_{CoM} = 0$)[cite: 4, 15, 18].

---

## 2. Adaptive, Reflexive, and Biomechanical Balance Control
Biomechanical studies demonstrate that humans counter external perturbations using three distinct, composable reflexes[cite: 23, 24]:
1. **Ankle Strategy**: Modulates ankle torque to shift the Center of Pressure (CoP) / ZMP within the Base of Support (BoS)[cite: 23]. For position- or ZMP-controlled robots, adding an auxiliary ZMP offset ($ZMP_{target} = p^{ref} + p^{aux}$) produces an effective restoring ankle torque $\tau_{ankle} = Mgp^{aux}$[cite: 23].
2. **Hip Strategy**: Uses angular acceleration of the torso and limbs (modeled via a flywheel) to generate horizontal ground reaction forces that pull the CoM back toward the BoS ($\tau_{hip}$ bang-bang profile approximated via target hip angle $\theta_{target}$)[cite: 24].
3. **Step Strategy**: Expands or shifts the BoS by stepping to the Capture Point ($x_{capture} = \dot{x}\sqrt{z_0/g}$) while adjusting swing-foot landing orientation and lowering leg stiffness/compliance to absorb landing shock[cite: 24].

---

## 3. Reinforcement Learning for Balance, Push Recovery, and Meta-Control
### Hierarchical RL for Push Recovery (Yi et al., IEEE ICRA 2011)
Direct analytical design of a unified controller that blends ankle, hip, and step strategies across all gait phases without full state knowledge is intractable[cite: 22, 24, 25]. Yi et al. formulated a hierarchical controller where:
- **Sensory State Space**: Combines filtered IMU torso angle $\theta_{IMU}$, raw gyro rate $\theta_{gyro}$, forward-kinematics support-foot tip angle $\theta_{foot}$, discrete walk state $WS \in \{STOP, DS_1, SS, DS_2\}$, and binary reflex states $\{PS_{hip}, PS_{step}\}$[cite: 24, 25].
- **Action Space**: Outputs parameterized control inputs $A = \{p^{aux}, \theta_{target}, x_{capture}\}$ to modulate the three low-level reflexive controllers[cite: 25].
- **Stopping-Energy Reward**: Optimizes policy parameters using reinforcement learning with a cost function $R = |\theta_{gyro}|^2 + \frac{g}{z_0}|\theta_{IMU}|^2$ inspired by residual stopping energy[cite: 25].

### Cascade & Parameterized RL in Modern Bipedal Locomotion
Recent works (Li et al., 2021; Castillo et al., 2022) further demonstrate that hierarchical/cascade policies—where RL modulates reduced-order planner targets, gait parameters, and impedance gains rather than raw joint torques—achieve superior sample efficiency and sim-to-real robustness[cite: 2, 20].

---

## 4. Synthesis: How Our Project Advances the State of the Art
| Feature | Fixed ALIP-DRS (Gao et al., 2025)[cite: 1] | Reflexive RL (Yi et al., 2011)[cite: 22] | Our Proposed Adaptive Hierarchy[cite: 29] |
| :--- | :--- | :--- | :--- |
| **Support Surface** | Horizontally swaying DRS[cite: 1, 3] | Static ground with impulse pushes[cite: 22, 26] | Swaying, pitching, heaving & non-inertial DRS[cite: 29] |
| **Core Planner** | Time-varying ALIP-DRS[cite: 3, 4] | Static LIPM ZMP walk controller[cite: 23, 24] | Adaptive ALIP-DRS with variable $H$, $T_{step}$, and $\bar{L}_S$ |
| **Recovery Modes** | Footstep adjustment only (passive ankles, $L_{CoM}=0$)[cite: 4, 9] | Ankle ($p^{aux}$), Hip ($\theta_{target}$), Step ($x_{capture}$)[cite: 23, 24] | Full Ankle + Hip + ALIP-DRS Step coordination |
| **Adaptation Mechanism** | Fixed analytical gains & parameters[cite: 12, 16] | Policy gradient over reflex gains[cite: 25, 26] | RL Meta-Controller modulating regimes, CoM, gains & reflexes[cite: 29] |

## References
1. Y. Gao, V. Paredes, Y. Gong, Z. He, A. Hereid, and Y. Gu, "Time-Varying Foot Placement Control for Humanoid Walking on Swaying Rigid Surface," *IEEE Transactions on Robotics*, vol. 41, pp. 5740–5760, 2025[cite: 1, 21].
2. S.-J. Yi, B.-T. Zhang, D. Hong, and D. D. Lee, "Learning Full Body Push Recovery Control for Small Humanoid Robots," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2011, pp. 2047–2052[cite: 22, 27].
3. Y. Gong and J. W. Grizzle, "Zero dynamics, pendulum models, and angular momentum in feedback control of bipedal locomotion," *ASME J. Dyn. Syst., Meas., Control*, vol. 144, no. 12, 2022[cite: 20].
4. J. Pratt, J. Carff, S. Drakunov, and A. Goswami, "Capture point: A step toward humanoid push recovery," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots*, 2006, pp. 200–207[cite: 20, 27].
5. A. Iqbal, S. Veer, and Y. Gu, "Analytical solution to a time-varying LIP model for quadrupedal walking on a vertically oscillating surface," *Mechatronics*, vol. 96, 2023[cite: 20].
6. G. A. Castillo, B. Weng, W. Zhang, and A. Hereid, "Reinforcement learning-based cascade motion policy design for robust 3D bipedal locomotion," *IEEE Access*, vol. 10, pp. 20135–20148, 2022[cite: 20].
# Mathematical Formulation & Hierarchical Control Architecture

## 1. Reduced-Order ALIP-DRS Dynamics (Higher Layer)
On a horizontally swaying rigid surface with world-frame velocity $(\dot{x}_{WS}(t), \dot{y}_{WS}(t))$, the continuous-phase dynamics of the robot's relative Center of Mass $(x_{SC}, y_{SC})$ and contact angular momentum $(L_{y,S}, L_{x,S})$ with ankle actuation torque $(\tau_{A,x}, \tau_{A,y})$ and centroidal angular momentum $(L_{x,CoM}, L_{y,CoM})$ are governed by:

$$\dot{x}_{SC} = \frac{1}{mH}L_{y,S} - \dot{x}_{WS}(t) - \frac{1}{mH}L_{y,CoM}, \qquad \dot{L}_{y,S} = mgx_{SC} + \tau_{A,y}$$

$$\dot{y}_{SC} = -\frac{1}{mH}L_{x,S} - \dot{y}_{WS}(t) + \frac{1}{mH}L_{x,CoM}, \qquad \dot{L}_{x,S} = -mgy_{SC} + \tau_{A,x}$$

Across the $k$-th foot-landing event at $t = T_k^-$, contact angular momentum is invariant ($\Delta L_{y,S} = 0, \Delta L_{x,S} = 0$) while the relative CoM position jumps by the step length: $\Delta x_{SC} = -u_x, \Delta y_{SC} = -u_y$.

## 2. RL Meta-Controller & Biomechanical Reflexes
The high-level Reinforcement Learning Meta-Controller optimizes a policy parameter vector $\boldsymbol{\theta} \in \mathbb{R}^{10}$ using Stochastic Policy Gradient to minimize the cumulative Stopping-Energy and Velocity-Tracking cost:

$$J(\boldsymbol{\theta}) = \mathbb{E}\left[ \int_0^T \left( w_1 \left(\|\boldsymbol{\theta}_{gyro}\|^2 + \frac{g}{H}\|\boldsymbol{\theta}_{IMU}\|^2\right) + w_2 \|\mathbf{v}_{CoM} - \mathbf{v}^{des}\|^2 \right) dt \right]$$

At runtime, the meta-controller outputs:
1. **Ankle Strategy ($p_{x,y}^{aux}$)**: Shifts the Center of Pressure within the support polygon to generate restoring ankle torque $\tau_{A,y} = mgp_x^{aux}$ and $\tau_{A,x} = -mgp_y^{aux}$.
2. **Hip Strategy ($\boldsymbol{\theta}_{target}$)**: Commands a torso pitch/roll target to accelerate the upper-body flywheel ($L_{CoM}$), generating horizontal shear forces that pull the CoM back inside the Base of Support.
3. **Adaptive ALIP-DRS Step & Posture Modulation**: Lowers the CoM height $H \leftarrow H + \Delta H(\boldsymbol{\theta})$ during surface pitching, shortens step duration $T_{step} \leftarrow T_{step} + \Delta T_{step}(\boldsymbol{\theta})$ during large pushes, and applies integral angular momentum bias correction $(\Delta \bar{L}_{x,S}, \Delta \bar{L}_{y,S})$ to eliminate drift from unknown payloads.

## 3. Middle-Layer Bézier Planner & Low-Level IO-QP Controller
The middle layer parameterizes the 18-dimensional task-space reference $h_d(s)$ using 5th-order Bézier polynomials over normalized phase $s = \frac{t - T_{k-1}}{T_{step}} \in [0, 1)$. The low-level Input-Output Linearizing Controller shapes the tracking error $h = h_a - h_d$ via $\ddot{h} = -K_p h - K_d \dot{h}$ while enforcing friction cone constraints ($\sqrt{f_x^2 + f_y^2} \le \mu f_z$) and joint torque saturation ($|\tau_u| \le \tau_{max}$).
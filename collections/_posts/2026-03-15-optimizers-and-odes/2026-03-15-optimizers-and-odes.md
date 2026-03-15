---
layout: post
title: Optimizers and ODEs
date: 2026-03-15 22:59 +0000
description: "A continuous-time view of gradient-based optimization: starting from the observation that integrator choice matters in physics simulation, and transferring that insight to understand modern optimizers."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Optimization
  - Differential Equations
  - Gradient Descent
  - Momentum
  - Lion
  - Numerical Methods
  - Game Theory
math: true
---

# Gradient-Based Optimization and Differential Equations

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
Overview
</div>
Many phenomena in physics are modeled using differential equations. For computer simulations in game dev physics engines, we use numerical integration methods to solve them. In this setting, **the integrator is not a detail.** You can write the prettiest equations for gravity and springs, but if you step them forward with the wrong numerical method, your simulation explodes, drifts, or leaks energy — especially over long horizons.

This post makes the case that **the same insight is the right lens for understanding gradient-based optimizers.** Every optimizer is a numerical integrator applied to a dynamical system. Gradient descent is forward Euler on gradient flow. Momentum is a multistep method. Lion is an Euler step on a geometry-modified ODE. Each choice has stability properties, and those properties determine what step sizes, what loss landscapes, and what training regimes will actually work.

But there is a critical difference: in physics simulation, you often have access to cheap force evaluations and can afford implicit solves or multiple stages per step. In deep learning, **a single gradient costs one full forward and backward pass**, so the design space for optimizers is heavily constrained. This post explores both the analogy and where it breaks down.
</blockquote>

---

## 1. The Simulation Perspective: You Write an ODE, Then You Pick an Integrator

If you have built a particle simulation — a game engine, a cloth sim, a molecular dynamics code — the workflow is familiar:

1. **Write down the physics** as a differential equation.
2. **Pick an integrator** to step it forward in discrete time.
3. **Discover that step 2 matters enormously**: explicit Euler blows up on stiff springs, leaks energy in planetary orbits, and drifts in constrained systems. Better integrators (Verlet, symplectic Euler, implicit methods) fix these problems without changing the physics.

The core lesson is: **the continuous equations can be perfectly well-behaved while the discrete simulation is a disaster, and the fix is choosing a better integrator.** This is exactly the same situation in optimization.

### Euler vs. Verlet on a Kepler orbit

Here is the most classic demonstration: a planet orbiting a star under gravity. The physics is the same in both cases — the only difference is how we step it forward in time.

![Orbit simulation: Forward Euler spirals outward (energy leak) while Velocity Verlet stays on the correct orbit.](orbit_euler_vs_verlet.gif)

Forward Euler injects energy on every step — the orbit spirals outward. Velocity Verlet is symplectic: it conserves a modified Hamiltonian, so the orbit stays closed even after many revolutions. **Same physics, different integrator, completely different long-term behavior.**

![Static comparison for reference.](orbit_static.png)

### The quick math

Newton's law in a potential <span class="math-inline" markdown="0">\(U(x)\)</span>:


<div class="math-block" markdown="0">
\[
m\ddot{x}(t) = -\nabla U(x(t)).
\]
</div>


Introduce velocity <span class="math-inline" markdown="0">\(v = \dot{x}\)</span> to get a first-order system:


<div class="math-block" markdown="0">
\[
\dot{x} = v, \qquad \dot{v} = -\frac{1}{m}\nabla U(x).
\]
</div>


**Forward Euler** (one evaluation, first order):


<div class="math-block" markdown="0">
\[
v_{k+1} = v_k + h\!\left(-\frac{1}{m}\nabla U(x_k)\right), \qquad
x_{k+1} = x_k + h\,v_k.
\]
</div>


**Velocity Verlet** (two evaluations, second order, symplectic):


<div class="math-block" markdown="0">
\[
v_{k+\frac{1}{2}} = v_k + \frac{h}{2}\,a(x_k), \quad
x_{k+1} = x_k + h\,v_{k+\frac{1}{2}}, \quad
v_{k+1} = v_{k+\frac{1}{2}} + \frac{h}{2}\,a(x_{k+1}).
\]
</div>


<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
The transferable lesson
</div>
The lesson for optimization is not "use Verlet." It is: **the integrator is not a detail.** It can dominate stability, and stability controls how large a step you can take — which in a training run directly determines convergence speed.
</blockquote>

---

## 2. Translation: Gradient Flow and Euler

In optimization you have a loss <span class="math-inline" markdown="0">\(f(x)\)</span> you want to minimize. The simplest continuous-time model of "go downhill" is **gradient flow**:


<div class="math-block" markdown="0">
\[
\dot{x}(t) = -\nabla f(x(t)).
\]
</div>


This is the optimization analogue of Newton's law: a continuous equation that governs the dynamics. It dissipates energy:


<div class="math-block" markdown="0">
\[
\frac{d}{dt} f(x(t))
= \langle \nabla f(x),\, \dot{x} \rangle
= -\lVert \nabla f(x) \rVert_2^2
\le 0.
\]
</div>


Apply forward Euler <span class="math-inline" markdown="0">\dot{x}=\frac{dx}{dt}\approx \frac{x_{k+1}-x_k}{h}</span> and you recover gradient descent:


<div class="math-block" markdown="0">
\[
x_{k+1} = x_k - h\,\nabla f(x_k).
\]
</div>


So "GD is Euler on gradient flow" is literally true. And just like forward Euler on a stiff spring, it can explode if <span class="math-inline" markdown="0">\(h\)</span> is too large relative to the curvature of the loss.

---

## 3. Toy 1 — A Stiff Quadratic

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Example: Why explicit steps explode and implicit steps do not
</div>
Take the 2-D quadratic


<div class="math-block" markdown="0">
\[
f(x_1, x_2) = \tfrac{1}{2}\!\left(100\,x_1^2 + x_2^2\right),
\qquad
\nabla f(x) = \begin{bmatrix} 100\,x_1 \\ x_2 \end{bmatrix}.
\]
</div>


Think of <span class="math-inline" markdown="0">\(x_1\)</span> as a stiff spring and <span class="math-inline" markdown="0">\(x_2\)</span> as a soft spring — exactly the same setup that causes problems in game physics.
</blockquote>

### 3.1 Gradient flow is always stable


<div class="math-block" markdown="0">
\[
\dot{x}_1 = -100\,x_1, \qquad \dot{x}_2 = -x_2
\]
</div>


Both directions decay exponentially. The continuous physics is perfectly fine.

### 3.2 Gradient descent (forward Euler) has a hard stability limit


<div class="math-block" markdown="0">
\[
x_{1,k+1} = (1 - 100h)\,x_{1,k}, \qquad
x_{2,k+1} = (1 - h)\,x_{2,k}.
\]
</div>


For stability: <span class="math-inline" markdown="0">\(\vert 1 - 100h\vert  < 1\)</span>, i.e. <span class="math-inline" markdown="0">\(h < 0.02\)</span>. Pick <span class="math-inline" markdown="0">\(h = 0.03\)</span> and start at <span class="math-inline" markdown="0">\((1, 1)\)</span>:

- <span class="math-inline" markdown="0">\(x_{1,1} = (1 - 3) \cdot 1 = -2\)</span>
- <span class="math-inline" markdown="0">\(x_{1,2} = (1 - 3) \cdot (-2) = 4\)</span>

The stiff direction explodes immediately — **same story as a stiff spring in Euler.**

### 3.3 Backward Euler (implicit) fixes stiffness


<div class="math-block" markdown="0">
\[
x_{k+1} = (I + hH)^{-1} x_k
\]
</div>


Stable for **all** <span class="math-inline" markdown="0">\(h > 0\)</span>. Same step size that exploded under forward Euler now contracts. The optimization version of "implicit integration" is the **proximal point method**.[^prox]

![Toy 1: Forward Euler (GD) bounces wildly in the stiff direction while backward Euler converges smoothly.](toy1_stiff_quadratic.png)

<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
Edge of Stability
</div>
This mismatch — stable ODE, unstable discretization — is exactly the phenomenon studied in the "edge of stability" line of work (Arora, Li, and Panigrahi, ICML 2022[^eos]): the sharpness (largest Hessian eigenvalue) hovers around <span class="math-inline" markdown="0">\(2/\eta\)</span> and the loss decreases non-monotonically, behavior not captured by the infinitesimal-step ODE picture.
</blockquote>

---

## 4. Why Simulation Methods Don't Transfer Directly to ML

At this point you might ask: if Verlet, implicit methods, and RK4 work so well in physics simulation, why don't we just use them for training neural networks?

The answer comes down to three fundamental differences between simulation and deep learning:

<blockquote class="box-important" markdown="1">
<div class="title" markdown="1">
The cost bottleneck
</div>
**In simulation**, evaluating the force <span class="math-inline" markdown="0">\(\nabla U(x)\)</span> is typically cheap relative to the time step. A molecular dynamics code might compute pairwise interactions in <span class="math-inline" markdown="0">\(O(N \log N)\)</span>, and then you can afford to call that multiple times per step (RK4 uses 4 evaluations, implicit methods iterate to convergence).

**In deep learning**, a single gradient evaluation <span class="math-inline" markdown="0">\(\nabla f(x)\)</span> requires a complete forward and backward pass through the entire network over a batch of data. This is the dominant cost of training. An optimizer that needs 4 gradients per step (like RK4) is 4× more expensive per step — and in practice it rarely converges 4× faster in terms of steps, so you lose.
</blockquote>

Beyond cost, there are deeper issues:

1. **No cheap Hessian.** Implicit methods like backward Euler require solving <span class="math-inline" markdown="0">\((I + hH)^{-1}\)</span>, which requires access to the Hessian <span class="math-inline" markdown="0">\(H = \nabla^2 f\)</span>. For a model with <span class="math-inline" markdown="0">\(d\)</span> parameters, the Hessian is <span class="math-inline" markdown="0">\(d \times d\)</span>. For a 1B-parameter model, that is a <span class="math-inline" markdown="0">\(10^9 \times 10^9\)</span> matrix — it does not fit in memory and you cannot even form it, let alone invert it. In simulation, the "stiffness matrix" is typically sparse and structured (e.g., tridiagonal for 1-D springs).

2. **Stochastic gradients.** In simulation, forces are deterministic: <span class="math-inline" markdown="0">\(\nabla U(x)\)</span> is the exact gradient. In ML, you compute <span class="math-inline" markdown="0">\(\nabla f(x)\)</span> on a random minibatch, so you get a noisy estimate. Higher-order methods that assume exact gradients can amplify this noise (RK4 averages four noisy evaluations, but the noise doesn't cancel like it would for deterministic errors). This pushes you toward simpler methods that are robust to gradient noise.

3. **Non-convex landscape.** Simulation potentials are often well-structured (harmonic, Lennard-Jones, etc.). Neural network loss landscapes are highly non-convex with saddle points, flat regions, and sharp features at many scales. The smoothness assumptions that justify higher-order accuracy guarantees rarely hold.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
What does transfer
</div>
The simulation perspective is still valuable — but the ideas that transfer are **structural**, not method-specific:
- **Stability analysis** tells you why learning rates blow up on sharp features (same as stiff springs).
- **Symplectic structure** explains why momentum methods conserve something (same as Verlet conserving energy).
- **Implicit treatment** motivates proximal methods and adaptive preconditioning (diagonal approximations to <span class="math-inline" markdown="0">\((I + hH)^{-1}\)</span>, which is essentially what Adam does).
- **Geometry changes** (choosing a non-Euclidean norm) give you sign descent, Lion, Muon — cheap alternatives to full implicit solves.

The winning strategy in ML is: take the **structural insight** from simulation, but implement it with methods that only need **one gradient per step** and **no second-order information**.
</blockquote>

---

## 5. Momentum: A Multistep Method in Disguise

In simulation terms, momentum is like switching from a single-step integrator to a **multistep** one that uses history. Two interpretations:

1. **Inertial dynamics.** Optimization as a damped second-order system:


   <div class="math-block" markdown="0">
\[
\ddot{x} + \gamma\,\dot{x} + \nabla f(x) = 0
\]
   </div>


   — a damped harmonic oscillator with the loss as potential.

2. **Multistep integration.** The heavy-ball update


   <div class="math-block" markdown="0">
\[
x_{k+1} = x_k - \eta\,\nabla f(x_k) + \beta\,(x_k - x_{k-1})
\]
   </div>


   is a 2-step method on gradient flow. Scieur et al. show that many accelerated methods are multistep integration schemes, with acceleration arising from larger stable step sizes.[^scieur]

The continuous-time limit of Nesterov acceleration is a damped ODE with time-varying friction <span class="math-inline" markdown="0">\(3/t\)</span> (Su, Boyd, Candès[^su]). The Bregman Lagrangian framework[^wibisono] generates a large class of accelerated methods in continuous time. And Shi et al.[^shi] show that acceleration depends on using a **symplectic** discretization — exactly the kind of integrator choice that game engine engineers already know matters.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
Practical takeaway
</div>
Momentum gives you "more than Euler" while still using **one gradient per step** — it reuses history, just like a multistep integrator.
</blockquote>

---

## 6. Toy 2 — A Minmax Game and Why "Optimism" Helps

In game dev, coupled oscillators that should orbit each other spiral apart under forward Euler — energy leaks into the system. The same thing happens when the vector field is rotational.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Rotation in a bilinear saddle
</div>

<div class="math-block" markdown="0">
\[
\min_x \max_y \; f(x, y) = xy.
\]
</div>


Gradients: <span class="math-inline" markdown="0">\(\nabla_x f = y\)</span>, <span class="math-inline" markdown="0">\(\nabla_y f = x\)</span>. The continuous dynamics is pure rotation.
</blockquote>

**GDA** spirals outward — eigenvalue magnitude <span class="math-inline" markdown="0">\(\sqrt{1 + \eta^2} > 1\)</span>. It is injecting energy, just like Euler on planetary orbits.

**Extragradient** (predictor–corrector) gives eigenvalue magnitude <span class="math-inline" markdown="0">\(\sqrt{1 - \eta^2 + \eta^4} < 1\)</span>. It contracts.

**Optimistic methods** (OGDA/OMD) approximate extragradient using past gradients — momentum **in the vector field**, not just in the state.

![Toy 2: GDA spirals outward (left) while Extragradient contracts toward the saddle (right).](toy2_minmax.png)

<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
The simulation analogy
</div>
GDA on a rotational field is Euler on a Hamiltonian system — it leaks energy. Extragradient is a predictor-corrector that fixes the leak. This is the same pattern as Euler vs. Verlet in the orbit demo.
</blockquote>

---

## 7. Geometry: What "Steepest Descent" Means Depends on the Norm

Given a norm <span class="math-inline" markdown="0">\(\lVert \cdot \rVert\)</span>, steepest descent solves:


<div class="math-block" markdown="0">
\[
d(x) \in \arg\min_{\lVert d \rVert \le 1} \langle \nabla f(x),\, d \rangle.
\]
</div>


<blockquote class="box-fact" markdown="1">
<div class="title" markdown="1">
This recovers familiar algorithms
</div>
- **Normalized GD**: steepest descent in <span class="math-inline" markdown="0">\(\ell_2\)</span>. Direction: <span class="math-inline" markdown="0">\(-\nabla f / \lVert \nabla f \rVert_2\)</span>.
- **Sign descent**: steepest descent in <span class="math-inline" markdown="0">\(\ell_\infty\)</span>. Direction: <span class="math-inline" markdown="0">\(-\mathrm{sign}(\nabla f)\)</span>.

This sets up **Lion** and **Muon**: choosing a geometry and applying a geometry-induced map — a cheap way to change the ODE without needing second-order information.
</blockquote>

---

## 8. Lion-<span class="math-inline" markdown="0">\(\mathcal{K}\)</span>: Concrete Equations

The Lion-<span class="math-inline" markdown="0">\(\mathcal{K}\)</span> paper (Chen et al., ICLR 2024 Spotlight[^lion]) generalizes Lion by replacing <span class="math-inline" markdown="0">\(\mathrm{sign}(\cdot)\)</span> with <span class="math-inline" markdown="0">\(\nabla K(\cdot)\)</span> for a convex function <span class="math-inline" markdown="0">\(K\)</span>.

### 8.1 Discrete-time Lion-<span class="math-inline" markdown="0">\(\mathcal{K}\)</span>

<blockquote class="box-algorithm" markdown="1">
<div class="title" markdown="1">
Discrete-time Lion-<span class="math-inline" markdown="0">\(\mathcal{K}\)</span>
</div>

<div class="math-block" markdown="0">
\[
m_{t+1} = \beta_2\,m_t - (1 - \beta_2)\,\nabla f(x_t),
\]
</div>



<div class="math-block" markdown="0">
\[
x_{t+1} = x_t + \epsilon\!\left(\nabla K\!\left(\beta_1\,m_t - (1 - \beta_1)\,\nabla f(x_t)\right) - \lambda\,x_t\right).
\]
</div>


Lion: <span class="math-inline" markdown="0">\(K(z) = \lVert z \rVert_1\)</span>, so <span class="math-inline" markdown="0">\(\nabla K(z) = \mathrm{sign}(z)\)</span>.
</blockquote>

### 8.2 Continuous-time ODE


<div class="math-block" markdown="0">
\[
\dot{m}(t) = -\alpha\,\nabla f(x(t)) - \gamma\,m(t),
\]
</div>



<div class="math-block" markdown="0">
\[
\dot{x}(t) = \nabla K\!\left(m(t) - \varepsilon\!\left(\alpha\,\nabla f(x(t)) + \gamma\,m(t)\right)\right) - \lambda\,x(t).
\]
</div>


The discrete update is an **Euler discretization** of this ODE — the same Euler-to-ODE link we have been building intuition for since Section 1.

### 8.3 Weight decay as a constraint

<blockquote class="box-fact" markdown="1">
<div class="title" markdown="1">
What problem does the ODE solve?
</div>
For Lion (<span class="math-inline" markdown="0">\(K = \lVert \cdot \rVert_1\)</span>), the optimizer targets:


<div class="math-block" markdown="0">
\[
\min_x\; f(x) \quad \text{s.t.} \quad \lVert x \rVert_\infty \le \frac{1}{\lambda}.
\]
</div>


Decoupled weight decay is not "a bit of regularization" — **it sets the constraint radius directly.** This is a geometry change that only costs one <span class="math-inline" markdown="0">\(\mathrm{sign}(\cdot)\)</span> per step — the kind of structural insight from simulation that is practical in ML.
</blockquote>

---

## 9. Toy 3 — Lion on a 1-D Example

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
How the box constraint shows up
</div>
<span class="math-inline" markdown="0">\(f(x) = (x - 2)^2\)</span>, <span class="math-inline" markdown="0">\(\nabla f(x) = 2(x - 2)\)</span>. Unconstrained minimizer: <span class="math-inline" markdown="0">\(x = 2\)</span>.

With <span class="math-inline" markdown="0">\(\lambda = 1\)</span>, the implied Lion constraint is <span class="math-inline" markdown="0">\(\vert x\vert  \le 1\)</span>. Constrained minimizer: <span class="math-inline" markdown="0">\(x^{\ast} = 1\)</span>.
</blockquote>

If the sign argument stays positive, the update reduces to:


<div class="math-block" markdown="0">
\[
x_{t+1} = (1 - \epsilon\lambda)\,x_t + \epsilon,
\]
</div>


which **contracts to <span class="math-inline" markdown="0">\(x = 1/\lambda\)</span>** regardless of gradient magnitude. The box constraint shows up as a stable fixed point.

---

## 10. Extended Toy — Four Methods Side by Side

This is where the simulation perspective pays off most clearly. One stiff quadratic, four methods, each corresponding to a different integrator / vector-field design choice.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Setup
</div>

<div class="math-block" markdown="0">
\[
f(x) = \tfrac{1}{2}(x - x^{\star})^\top H\,(x - x^{\star}), \quad
x^{\star} = (2, 2), \quad
H = \mathrm{diag}(1000, 1).
\]
</div>


Start at <span class="math-inline" markdown="0">\(x_0 = (0, 0)\)</span>. One stiff direction, one soft direction.
</blockquote>

| Method      | Integrator analogy                                                      | Why it works / fails                      |
| ----------- | ----------------------------------------------------------------------- | ----------------------------------------- |
| GD (stable) | Forward Euler, tiny <span class="math-inline" markdown="0">\(h\)</span> | Stiffness caps the global step            |
| Implicit GD | Backward Euler                                                          | Unconditionally stable, big steps         |
| Lion (sign) | Euler on geometry-modified ODE                                          | Ignores gradient magnitude, hits box wall |
| Lion (tanh) | Smoothed geometry variant                                               | Interpolates sign and magnitude           |

![Per-coordinate convergence of all four methods over 30 steps.](extended_toy_comparison.png)

![2-D trajectory overlay: implicit GD reaches the optimum, Lion variants converge to the box boundary.](extended_toy_2d.png)

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
What to notice
</div>
1. **GD oscillates wildly in the stiff direction** (red). The step size is capped by stiffness — same as in simulation.
2. **Implicit GD takes big steps safely** (blue). This is why implicit methods dominate in stiff simulation — but in ML, it requires a Hessian solve we cannot afford.
3. **Lion ignores gradient magnitude** (green). It converges toward <span class="math-inline" markdown="0">\(1/\lambda = 1\)</span>, not toward <span class="math-inline" markdown="0">\(x^{\star} = 2\)</span> — that is the box constraint from weight decay.
4. **Smoothing (<span class="math-inline" markdown="0">\(\tanh\)</span>)** (purple) interpolates: saturates for large arguments (like sign), but produces smaller steps near zero.
5. **The practical lesson**: Lion achieves stability on the stiff direction without a Hessian — it pays for it by changing *what* it converges to (the constrained optimum), which is often a feature in ML (better generalization).
</blockquote>

---

## 11. A Practical Checklist

When you see a new optimizer, ask:

1. **What is the underlying ODE/SDE?** (What dynamics is it discretizing?)
2. **What is the geometry?** (Euclidean? Preconditioned? Sign? Mirror map?)
3. **What is the discretization?** (Euler? Multistep? Splitting?)
4. **Where is stability gained?** (Larger step sizes? Stiffness robustness? Noise robustness?)
5. **What is the cost per step?** (One gradient? Multiple? Hessian-vector products?)

The last question is what separates practical ML optimizers from theoretical ones. Lion-<span class="math-inline" markdown="0">\(\mathcal{K}\)</span> is a good example because it gives you useful structural properties (box constraint, stiffness robustness) while keeping the cost at **one gradient + one elementwise sign per step**.

---

## References

[^eos]: Arora, S., Li, Z., & Panigrahi, A. (2022). [Understanding Gradient Descent on the Edge of Stability in Deep Learning](https://proceedings.mlr.press/v162/arora22a/arora22a.pdf). *ICML 2022*.

[^prox]: Parikh, N., & Boyd, S. (2014). [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf). *Foundations and Trends in Optimization*.

[^scieur]: Scieur, D., Roulet, V., Bach, F., & d'Aspremont, A. (2017). [Integration Methods and Optimization Algorithms](https://papers.neurips.cc/paper/6711-integration-methods-and-optimization-algorithms.pdf). *NeurIPS 2017*.

[^su]: Su, W., Boyd, S., & Candès, E. J. (2016). [A Differential Equation for Modeling Nesterov's Accelerated Gradient Method](https://stanford.edu/~boyd/papers/pdf/ode_nest_grad.pdf). *JMLR*.

[^wibisono]: Wibisono, A., Wilson, A. C., & Jordan, M. I. (2016). [A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245). *PNAS*.

[^shi]: Shi, B., Du, S. S., Jordan, M. I., & Su, W. J. (2019). [Acceleration via Symplectic Discretization of High-Resolution Differential Equations](https://arxiv.org/abs/1902.03694). *NeurIPS 2019*.

[^lion]: Chen, L., Liu, B., Liang, K., & Liu, Q. (2024). [Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts](https://proceedings.iclr.cc/paper_files/paper/2024/file/986e0caad271b59417287737416d8594-Paper-Conference.pdf). *ICLR 2024 (Spotlight)*.
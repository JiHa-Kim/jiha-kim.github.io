---
layout: post
title: "Dynamics of Higher-Order Residual Networks"
date: 2026-03-31 19:00 +0000
description: "Residual networks are first-order depth dynamics that can only produce exponential modes. Upgrading to second-order by adding a velocity state unlocks bounded oscillatory modes, separates content from transport, and connects depth to the same mathematical framework as LSTMs and state-space models."
categories:
  - Machine Learning
  - Mathematical Foundations
tags:
  - Residual Networks
  - Depth Dynamics
  - Second-Order Systems
  - Neural ODE
  - LSTM
  - Finite Differences
  - State-Space Models
math: true
scholar:
  bibliography: posts/2026-03-31-second-order-residual-networks/second-order-resnets.bib
---

> [!info] Overview
> A residual network computes $x_{l+1} = x_l + f_l(x_l)$, which is a first-order finite difference equation over depth. In this post, we work through the consequences of making this **second-order** by adding a velocity state. The core observation is simple:
>
> - First-order depth dynamics can only produce **exponential** modes ($e^\alpha$).
> - Second-order depth dynamics can also produce **oscillatory** modes ($\cos\beta$).
>
> This is a qualitative expansion of what depth can express. We work through the scalar analysis, trace through numerical examples, and show how the resulting architecture connects to LSTMs and state-space models.

---

## 1. First-Order Depth Dynamics

A residual network{% cite heDeepResidualLearning2015 %} computes

$$
x_{l+1} = x_l + f_l(x_l),
$$

where $l$ indexes layers. This is a first-order finite difference equation:

$$
\Delta x_l := x_{l+1} - x_l = f_l(x_l).
$$

> [!remark] Connection to ODEs
> This is a forward Euler step on the ODE $\dot{x}(t) = f(x(t))$, with the layer index as "time." This interpretation is the basis of the Neural ODE framework{% cite chenNeuralOrdinaryDifferential2019 %} and is discussed in more detail in [Optimizers and ODEs](/posts/optimizers-and-odes/).

The residual form is a good idea because the identity map is built in, so each layer only needs to learn a small correction. But what kinds of depth trajectories can this first-order law produce?

### What first-order dynamics can do

Consider the simplest scalar case: a depth-$L$ network where each layer applies the same linear function $f_l(x) = \frac{\alpha}{L} x$.

$$
x_{l+1} = x_l + \frac{\alpha}{L} \, x_l = \left(1 + \frac{\alpha}{L}\right) x_l
$$

This is a geometric sequence with ratio $r = 1 + \frac{\alpha}{L}$, so after $L$ layers:

$$
x_L = \left(1 + \frac{\alpha}{L}\right)^L x_0 \to e^\alpha \, x_0 \quad \text{as } L \to \infty.
$$

> [!example] Numerical example
> Take $L = 4$ layers and $x_0 = 1$.
>
> **Growing mode** ($\alpha = 1$): $r = 1.25$
>
> $$
> x_0 = 1 \to x_1 = 1.25 \to x_2 = 1.56 \to x_3 = 1.95 \to x_4 = 2.44
> $$
>
> This is heading toward $e^1 \approx 2.72$.
>
> **Decaying mode** ($\alpha = -1$): $r = 0.75$
>
> $$
> x_0 = 1 \to x_1 = 0.75 \to x_2 = 0.56 \to x_3 = 0.42 \to x_4 = 0.32
> $$
>
> This is heading toward $e^{-1} \approx 0.37$.
>
> In both cases the trajectory is **monotone**. There is no way to get bounded oscillation out of a first-order recurrence with a positive ratio. The only exactly neutral point is $\alpha = 0$, which does nothing.

So first-order residual dynamics are restricted to exponential-type behavior in depth: growth, decay, or stasis. There is no nontrivial bounded oscillatory regime.

This is a real limitation. What if we want richer depth dynamics?

---

## 2. The Second-Order Extension

The idea is borrowed from classical mechanics. A particle moving under gravity has both a **position** and a **velocity**. The position changes because of velocity, and the velocity changes because of forces. These two coupled variables give rise to oscillation, which a single first-order equation cannot produce.

We apply the same idea to depth. Instead of a single state $x_l$, maintain two states: a **content** $x_l$ and a **velocity** $v_l$.

> [!definition] Second-Order Residual Block
> $$
> v_{l+1} = m_l \odot v_l + \eta_l \odot f_l(\operatorname{LN}(x_l))
> $$
>
> $$
> x_{l+1} = x_l + v_{l+1}
> $$
>
> where $m_l \in [0, 1)^d$ is a per-channel **carry coefficient** (how much of the old velocity to keep), $\eta_l \geq 0$ is a per-channel **forcing scale**, and $\odot$ is elementwise multiplication.

The content $x_l$ stores the representation. The velocity $v_l$ stores the "direction of travel" through depth. The block output $f_l$ acts as a force that adjusts the velocity, and the velocity in turn moves the content.

> [!fact] Reduction to standard residuals
> Setting $m_l = 0$ and $\eta_l = 1$ gives $v_{l+1} = f_l(\operatorname{LN}(x_l))$ and $x_{l+1} = x_l + f_l(\operatorname{LN}(x_l))$, which is exactly the standard pre-norm residual block. The second-order model is a strict generalization.

### The second-order difference equation

We can eliminate $v_l$ to see what equation $x_l$ satisfies on its own. Since $v_l = x_l - x_{l-1}$ (from the second equation), substituting into the first gives:

$$
x_{l+1} - x_l = m_l (x_l - x_{l-1}) + \eta_l f_l(\operatorname{LN}(x_l)).
$$

Rearranging:

$$
x_{l+1} - (1 + m_l) x_l + m_l x_{l-1} = \eta_l f_l(\operatorname{LN}(x_l)).
$$

In terms of finite differences, this is

$$
\Delta^2 x_l + \gamma_l \, \Delta x_l = \eta_l \, f_l(\operatorname{LN}(x_l)),
$$

where $\gamma_l := 1 - m_l$ is the **damping coefficient**. This is the discrete version of a damped, driven oscillator. The second-order difference $\Delta^2 x_l = x_{l+2} - 2x_{l+1} + x_l$ measures the "curvature" of the depth trajectory, just as the second derivative measures curvature in continuous time.

> [!remark] Prior work
> Sander et al.{% cite sanderMomentumResidualNeural2021 %} proposed Momentum ResNets, adding a momentum term to ResNet blocks motivated by invertibility and memory efficiency. Their architecture is equivalent to this second-order form. The emphasis here is on the finite-difference perspective: what modes of propagation does the second-order structure enable that first-order cannot?

---

## 3. What Can Second-Order Dynamics Do?

This is the main result. Let us work through the scalar second-order toy in the same way we analyzed the first-order case.

### The scalar second-order toy

Consider $L$ layers of:

$$
x_{l+1} = x_l + v_l, \qquad v_{l+1} = v_l - \frac{\beta^2}{L^2} \, x_l,
$$

with initial conditions $x_0 = c$ and $v_0 = 0$. The term $-\frac{\beta^2}{L^2} x_l$ acts as a restoring force: when $x$ is positive, it pulls the velocity negative, and vice versa. This is the discrete analogue of a spring.

### Deriving the characteristic equation

To find the behavior, we eliminate $v_l$. Since $v_l = x_{l+1} - x_l$, substituting into the velocity update:

$$
x_{l+2} - x_{l+1} = (x_{l+1} - x_l) - \frac{\beta^2}{L^2} x_l.
$$

Collecting terms:

$$
x_{l+2} - 2x_{l+1} + \left(1 + \frac{\beta^2}{L^2}\right) x_l = 0.
$$

> [!remark]
> Reindexing by replacing $l \to l-1$, this is $x_{l+1} - 2x_l + (1 + \frac{\beta^2}{L^2}) x_{l-1} = 0$, which is a **linear recurrence with constant coefficients**. We can solve it by trying $x_l = r^l$.

Substituting $x_l = r^l$:

$$
r^2 - 2r + 1 + \frac{\beta^2}{L^2} = 0,
$$

which simplifies to

$$
r^2 - \left(2 - \frac{\beta^2}{L^2}\right) r + 1 = 0.
$$

### Analyzing the roots

The discriminant is

$$
D = \left(2 - \frac{\beta^2}{L^2}\right)^2 - 4 = \frac{\beta^2}{L^2}\left(\frac{\beta^2}{L^2} - 4\right).
$$

When $0 < \beta < 2L$, the discriminant is **negative**, so the roots are complex conjugates. By Vieta's formulas, the product of the roots equals $1$ (the constant term of the polynomial), which means:

$$
r_\pm = e^{\pm i\theta}, \qquad |r_\pm| = 1,
$$

where $\cos\theta = 1 - \frac{\beta^2}{2L^2}$.

> [!proposition] Bounded oscillatory propagation
> For any $0 < \beta < 2L$, the second-order depth dynamics have **bounded, oscillatory** propagation: $|r_\pm| = 1$, so signals neither grow nor decay.
>
> The general solution is $x_l = A \cos(l\theta) + B \sin(l\theta)$. This is a whole **interval** of parameters giving nontrivial bounded dynamics.
>
> Compare this to the first-order case, where only a single point ($\alpha = 0$) is exactly neutral.

### The punchline

Applying the initial conditions $x_0 = c$ and $v_0 = x_1 - x_0 = 0$, we need $A = c$ and $B\sin\theta = 0$, so $B = 0$ and

$$
x_l = c \cos(l\theta).
$$

For large $L$, $\theta \approx \beta/L$, so

$$
x_L \approx c \cos\beta.
$$

This is the key comparison:

| Depth model | Scalar output | Family |
| :--- | :--- | :--- |
| First-order: $\Delta x_l = \frac{\alpha}{L} x_l$ | $x_L \approx e^\alpha \, c$ | Exponentials |
| Second-order: $\Delta^2 x_l = -\frac{\beta^2}{L^2} x_l$ | $x_L \approx \cos(\beta) \, c$ | Oscillations |

First-order residuals naturally express exponential depth effects. Second-order depth laws naturally express oscillatory depth effects. This is a qualitatively richer family of behaviors.

---

## 4. A Worked Example: The Velocity Buffer

The velocity state does more than enable oscillation. It also acts as a buffer that absorbs fast layer-to-layer disagreements, keeping the content smoother. Let us trace through a concrete example.

Suppose we have 4 layers whose block outputs alternate in sign (the layers "disagree"):

$$
u_1 = +1, \quad u_2 = -0.8, \quad u_3 = +0.6, \quad u_4 = -0.4.
$$

### First-order residual

Starting at $x_0 = 0$, each update writes directly into $x$:

$$
\begin{align}
x_1 &= 0 + 1 = 1 \\
x_2 &= 1 - 0.8 = 0.2 \\
x_3 &= 0.2 + 0.6 = 0.8 \\
x_4 &= 0.8 - 0.4 = 0.4
\end{align}
$$

The content bounces: $0 \to 1 \to 0.2 \to 0.8 \to 0.4$. Every layer directly overwrites the representation. The swings are large.

### Second-order residual ($m = 0.5$, $\eta = 1$)

Starting at $x_0 = 0$, $v_0 = 0$:

$$
\begin{align}
v_1 &= 0.5 \cdot 0 + 1 \cdot 1 = 1, & x_1 &= 0 + 1 = 1 \\
v_2 &= 0.5 \cdot 1 + 1 \cdot (-0.8) = -0.3, & x_2 &= 1 + (-0.3) = 0.7 \\
v_3 &= 0.5 \cdot (-0.3) + 1 \cdot 0.6 = 0.45, & x_3 &= 0.7 + 0.45 = 1.15 \\
v_4 &= 0.5 \cdot 0.45 + 1 \cdot (-0.4) = -0.175, & x_4 &= 1.15 + (-0.175) = 0.975
\end{align}
$$

The content trajectory is $0 \to 1 \to 0.7 \to 1.15 \to 0.975$. The oscillation amplitude is much smaller: the velocity buffer absorbs the fast back-and-forth, and the content settles more smoothly.

> [!note] What happened?
> In the first-order model, each opposing update hits $x$ directly, causing large swings ($0.8$ amplitude). In the second-order model, the opposing updates first interact in the velocity $v$: the carry term $m \cdot v_l$ blends the current force with the previous velocity, so a reversal in $u$ only partially reverses $v$. The content $x$ then sees the smoothed velocity, not the raw updates.
>
> This is the concrete mechanism behind "less interference in the residual stream."

---
## 5. Connection to Recurrent Architectures

The second-order residual block is a form of gated recurrence over depth.{% cite hochreiterLongShortTermMemory1997 %} To make this precise, let us compare the update equations side by side.

### LSTM (recurrence over time)

The LSTM cell state update is:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,
$$

where $f_t$ is the forget gate, $i_t$ is the input gate, and $\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t])$ is the candidate update. The key structural feature is that $c_t$ combines a **gated carry** of the old state with a **gated injection** of new information.

### Second-order residual (recurrence over depth)

The velocity update is:

$$
v_{l+1} = m_l \odot v_l + \eta_l \odot f_l(\operatorname{LN}(x_l)).
$$

This has exactly the same structure: $m_l$ plays the role of the forget gate (how much old velocity to keep), and $\eta_l$ plays the role of the input gate (how much new information to inject).

### What maps to what

| LSTM (over time $t$) | Second-order residual (over depth $l$) |
| :--- | :--- |
| Cell state $c_t$ | Velocity $v_l$ |
| Hidden state $h_t$ | Content $x_l$ |
| Forget gate $f_t$ | Carry coefficient $m_l$ |
| Input gate $i_t$ | Forcing scale $\eta_l$ |
| Candidate $\tilde{c}_t$ | Block output $f_l(\operatorname{LN}(x_l))$ |

The content $x_l$ is analogous to the readout of the LSTM: it is the "visible" state that downstream layers operate on. The velocity $v_l$ is the internal memory that accumulates and filters the update stream.

> [!remark] Key differences
> The second-order residual is much simpler than an LSTM. It has only two learnable scalars per layer ($m_l, \eta_l$) instead of full gate matrices. It has no output gate and no $\tanh$ on the cell state. And the recurrence is over depth (layers in a single forward pass), not over time (sequence positions).
>
> The point is not to replicate the full generality of an LSTM. It is to import the most useful structural feature, namely separating a "memory channel" from a "transport/accumulation channel," into the depth axis where standard residual connections offer no such separation.

### The state-space view

Both the LSTM and the second-order residual are special cases of a first-order recurrence on an augmented state. For the second-order residual, stacking $x_l$ and $v_l$ gives:

$$
\begin{bmatrix} x_{l+1} \\ v_{l+1} \end{bmatrix} = \begin{bmatrix} I & I \\ \eta_l J_l & m_l I \end{bmatrix} \begin{bmatrix} x_l \\ v_l \end{bmatrix},
$$

where $J_l$ is the local Jacobian of $f_l$. This is a linear state-space model. Mathematically, any higher-order recurrence can be rewritten as a first-order recurrence on a larger state, so the second-order residual is indeed "an RNN" in the formal sense.

But a generic RNN says nothing about what the recurrence should look like. The second-order structure says something specific: one state (content) integrates another (velocity), and the velocity is a gated combination of its history and fresh input. This specific structure is what gives rise to the two-mode behavior we analyzed in Section 3.

---

## 6. Linearized Stability Analysis

Now let us analyze the full (non-toy) model. Suppose locally that $f_l(x_l) \approx J x_l$ for some Jacobian $J$. In one scalar eigenmode with eigenvalue $j$, the second-order recurrence becomes:

$$
v_{l+1} = m \, v_l + \eta \, j \, x_l, \qquad x_{l+1} = x_l + v_{l+1}.
$$

Substituting $v_l = x_l - x_{l-1}$ and simplifying:

$$
x_{l+1} = (1 + m + \eta j) \, x_l - m \, x_{l-1}.
$$

The characteristic equation is:

$$
r^2 - (1 + m + \eta j) \, r + m = 0.
$$

> [!proposition] Two-mode structure
> By Vieta's formulas, the product of the roots is $m$ and their sum is $1 + m + \eta j$. This tells us:
>
> - If $0 \leq m < 1$, both roots have magnitude less than $1$ when $j$ is small. The recurrence is **stable with built-in damping**.
> - When $j \approx 0$, one root is near $1$ (the **slow content mode**) and the other is near $m$ (the **fast velocity mode**).
>
> This two-mode structure is exactly the content/transport separation we designed for. The slow mode carries the representation, and the fast mode absorbs transient fluctuations.

Compare this to the first-order linearization, which gives

$$
x_{l+1} = (1 + \eta j) \, x_l.
$$

This has a single mode. There is no structural separation between content and transport.

> [!proof]- Detailed root analysis
> Solving the quadratic $r^2 - (1+m+\eta j)r + m = 0$:
>
> $$
> r_\pm = \frac{(1+m+\eta j) \pm \sqrt{(1+m+\eta j)^2 - 4m}}{2}.
> $$
>
> When $\eta j = 0$: $r_+ = 1$, $r_- = m$. As $\eta j$ grows from zero, both roots move continuously. The product $r_+ r_- = m$ stays constant.
>
> For the roots to become complex (oscillatory), we need the discriminant to go negative:
>
> $$
> (1+m+\eta j)^2 < 4m,
> $$
>
> $$
> |1+m+\eta j| < 2\sqrt{m}.
> $$
>
> When this happens, $|r_\pm| = \sqrt{m} < 1$, so the oscillation is **damped**. The damping rate is controlled by $m$: closer to $1$ means slower damping, closer to $0$ means faster damping.

---

## 7. The General Framework

The ideas above extend naturally to higher orders. A $p$-th order depth model satisfies

$$
\sum_{j=0}^p a_j \Delta^j x_l = f_l(x_l),
$$

which can be written as a first-order recurrence on the augmented state

$$
z_l = \begin{bmatrix} x_l \\ \Delta x_l \\ \vdots \\ \Delta^{p-1} x_l \end{bmatrix}, \qquad z_{l+1} = \Phi_l(z_l).
$$

This is the mathematical language shared by many architectures:

| Architecture | Recurrence type |
| :--- | :--- |
| Residual networks{% cite heDeepResidualLearning2015 %} | 1st-order, identity-centered |
| Second-order residuals | 2nd-order, damped |
| RNNs | General 1st-order on augmented state |
| LSTMs{% cite hochreiterLongShortTermMemory1997 %} | Structured 1st-order with gating |
| Linear attention{% cite katharopoulosTransformersAreRNNs2020 %} | Iterative state update |
| State-space models | Linear recurrence $z_{l+1} = A z_l + B u_l$ |

All the standard tools apply to this framework: linearization, characteristic polynomials, root placement, and spectral stability analysis. The second-order case is the simplest non-trivial instance, and as we saw, it already produces qualitatively richer behavior than the first-order case.

---

## 8. Implementation

> [!algorithm] Minimal Second-Order Residual Block
> Given a standard transformer/ResNet block function $f_l$:
>
> $$
> u_l = f_l(\operatorname{LN}(x_l))
> $$
>
> $$
> m_l = \sigma(a_l), \qquad \eta_l = \operatorname{softplus}(b_l)
> $$
>
> $$
> v_{l+1} = m_l \odot v_l + \eta_l \odot u_l
> $$
>
> $$
> x_{l+1} = x_l + v_{l+1}
> $$
>
> Here $a_l, b_l \in \mathbb{R}^d$ are two learnable vectors per layer.

> [!important] Initialization
> Initialize so the model starts as an ordinary residual net:
>
> - $v_0 = 0$
> - $a_l \ll 0$ so that $m_l = \sigma(a_l) \approx 0$
> - $b_l = \log(e - 1)$ so that $\eta_l = \operatorname{softplus}(b_l) \approx 1$
>
> With this initialization, $v_{l+1} \approx 0 \cdot v_l + 1 \cdot u_l = u_l$ and $x_{l+1} \approx x_l + u_l$. The model starts as a standard residual network and only develops second-order behavior if training finds it useful.

> [!tip] What to measure
> Three diagnostics that would test the core hypothesis:
>
> 1. **Norm growth across depth.** The depth profile of $\|x_l\|$ should be smoother than in standard residuals.
> 2. **Cosine similarity between successive updates.** $\mathrm{cossim}(v_l, v_{l+1})$ should show less sign-flipping than $\mathrm{cossim}(u_l, u_{l+1})$ in a first-order model.
> 3. **Gradient magnitude across layers.** More uniform gradient transport is predicted.
---

## References

{% bibliography %}

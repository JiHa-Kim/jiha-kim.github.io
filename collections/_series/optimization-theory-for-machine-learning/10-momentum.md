---
title: "Momentum"
date: 2025-05-18 02:57 -0400
series_index: 10
description: "Exploring momentum in optimization: how it accelerates gradient descent, dampens oscillations, and helps navigate complex loss landscapes in machine learning."
image:
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Iterative Methods
- Gradient Descent
- Optimization Algorithms
- Momentum
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):
    text

    $$
    block
    $$

    text... or:

    $$block$$

    text...
  Use LaTeX commands for symbols as much as possible (e.g. $$\vert$$ for
  absolute value, $$\ast$$ for asterisk). Avoid using the literal vertical bar
  symbol; use \vert and \Vert instead.

  The syntax for lists is:

  1. $$inline$$ item
  2. item $$inline$$
  3. item

      $$
      block
      $$

      (continued) item
  4. item

  Here are examples of syntaxes that do **not** work:

  1. text
    $$
    block
    $$
    text

  2. text
    $$
    text
    $$

    text

  And the correct way to include multiple block equations in a list item:

  1. text

    $$
    block 1
    $$

    $$
    block 2
    $$

    (continued) text

  Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
  `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.

  Here are some blockquote templates you can use:

  <blockquote class="box-definition" markdown="1">
  <div class="title" markdown="1">
  **Definition.** The natural numbers $$\mathbb{N}$$
  </div>
  The natural numbers are defined as $$inline$$.

  $$
  block
  $$

  </blockquote>

  And a details block template:

  <details class="details-block" markdown="1">
  <summary markdown="1">
  **Tip.** A concise title goes here.
  </summary>
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

  Similarly, for boxed environments you can define:
    - box-definition          # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - box-lemma               # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - box-proposition         # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - box-theorem             # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - box-example             # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple) (for example blocks with lightbulb icon)
    - box-info                # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)` (theme-defined)
    - box-tip                 # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)` (theme-defined)
    - box-warning             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)` (theme-defined)
    - box-danger              # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)` (theme-defined)

  For details blocks, use:
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

In our journey through optimization for machine learning, we've seen the power of gradient descent. However, its "vanilla" form, while foundational, often struggles in the complex, high-dimensional landscapes presented by modern ML models. It can be painstakingly slow in certain terrains or oscillate wildly in others. What if we could give our optimizer a sense of "memory" or "inertia"? This is precisely the idea behind **momentum**.

## The Plight of Vanilla Gradient Descent

Recall that standard Gradient Descent (GD), or its stochastic variant (SGD), updates parameters $$\theta$$ by taking steps in the direction opposite to the gradient $$\nabla J(\theta)$$:

$$
\theta_t = \theta_{t-1} - \eta \nabla J(\theta_{t-1})
$$

where $$\eta$$ is the learning rate.

While this approach guarantees descent (for a small enough $$\eta$$) in many cases, it has its drawbacks:
1.  **Slow progress in ravines:** Imagine a loss landscape shaped like a long, narrow valley (a common scenario in ill-conditioned problems). The gradient along the steep walls will be much larger than along the gentle slope towards the valley's minimum. GD tends to oscillate back and forth across the narrow valley, making slow progress along its length.
2.  **Inconsistent gradients:** In stochastic settings (SGD), gradients computed on mini-batches can be noisy and vary significantly from one batch to the next. This can lead to a zig-zagging path towards the minimum, slowing convergence.
3.  **High curvature issues:** If the curvature of the loss landscape changes drastically, a fixed learning rate might be too small for flat regions and too large for highly curved regions, leading to slow convergence or overshooting.

Consider navigating such a terrain. If you were only to look at the slope directly beneath your feet at each step, you might find yourself taking many inefficient, short, zig-zagging steps. So, how can we make our descent smarter and more efficient?

## The Idea of Momentum: Learning from Physics

The concept of momentum in optimization is directly inspired by physics. Imagine a heavy ball rolling down a hill.
*   It **gains velocity** as it rolls downwards in a consistent direction.
*   This velocity helps it **smooth out its path**, not being too perturbed by small bumps or changes in slope.
*   Its **inertia** allows it to power through flat regions or even slightly uphill segments, rather than getting stuck immediately.

Translating this to optimization:
*   We can introduce a "velocity" term, let's call it $$v_t$$, that accumulates a running average of past gradients.
*   This velocity term will guide the parameter updates. If gradients consistently point in the same direction, the velocity builds up, leading to larger steps.
*   If gradients oscillate, their contributions to the velocity will tend to cancel out, dampening the oscillations and smoothing the trajectory.

This "memory" of past gradients is what gives momentum its power. It helps the optimizer to commit to a beneficial direction and avoid being sidetracked by noisy or misleading local gradient information.

## Formalizing Momentum

Let's formalize this. We'll denote our parameters at iteration $$t$$ as $$\theta_t$$, the objective function as $$J(\theta)$$, the gradient as $$\nabla J(\theta_t)$$, the learning rate as $$\eta$$, and a new hyperparameter, the momentum coefficient, as $$\gamma$$.

The core idea is to update a **velocity vector** $$v_t$$ at each step, which is then used to update the parameters $$\theta_t$$.

The velocity $$v_t$$ is updated as an exponentially decaying moving average of past gradients, plus the current gradient:

$$
v_t = \gamma v_{t-1} + \eta \nabla J(\theta_{t-1})
$$

We typically initialize the velocity $$v_0 = 0$$.

Then, the parameters are updated by "moving" with this velocity:

$$
\theta_t = \theta_{t-1} - v_t
$$

Let's break down the components:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Momentum Update Equations**
</div>
Given parameters $$\theta_{t-1}$$ and velocity $$v_{t-1}$$ from the previous step:
1. Compute the gradient of the loss function: $$g_t = \nabla J(\theta_{t-1})$$.
2. Update the velocity:

   $$
   v_t = \gamma v_{t-1} + \eta g_t
   $$

3. Update the parameters:

   $$
   \theta_t = \theta_{t-1} - v_t
   $$

Here:
- $$\theta_t$$: Parameters at iteration $$t$$.
- $$v_t$$: Velocity (or update step) at iteration $$t$$.
- $$\gamma$$: Momentum coefficient (e.g., 0.9, 0.95). This is the factor by which the previous velocity is decayed.
- $$\eta$$: Learning rate (e.g., 0.01). This scales the current gradient's contribution.
- $$g_t$$: Gradient of the loss function $$J$$ with respect to $$\theta_{t-1}$$. (Note: Some formulations use $$g_t = \nabla J(\theta_t)$$, leading to slightly different update forms but similar intuition.)
</blockquote>

**The Momentum Coefficient ($$\gamma$$):**
*   **What it is:** The parameter $$\gamma$$ (gamma) controls how much of the previous velocity $$v_{t-1}$$ is retained in the current velocity $$v_t$$. It typically takes values between 0 and 1 (e.g., common choices are 0.9, 0.95, or even 0.99).
*   **Why it's named/its purpose:** It acts like a "friction" term or determines the "persistence" of movement.
    *   If $$\gamma = 0$$, we recover standard gradient descent: $$v_t = \eta g_t$$, and so $$\theta_t = \theta_{t-1} - \eta g_t$$. There's no memory of past steps.
    *   If $$\gamma$$ is close to 1, past gradients have a strong and lasting influence on the current direction. The "ball" is "heavier" and has more inertia.
*   **What purpose it serves:** It's crucial for building up speed in directions of consistent improvement and for averaging out oscillations.

## Unpacking the Dynamics of Momentum

How exactly does this accumulation of velocity help?

**1. Acceleration in Consistent Directions:**
If the gradients $$g_k = \nabla J(\theta_k)$$ consistently point in a similar direction over several iterations, the velocity term $$v_t$$ will grow larger in that direction. To see this, we can unroll the recurrence for $$v_t$$ (assuming $$v_0 = 0$$ and using $$g_k = \nabla J(\theta_k)$$ for simpler notation in the sum):

$$
\begin{align*}
v_t &= \gamma v_{t-1} + \eta g_{t-1} \\
    &= \gamma (\gamma v_{t-2} + \eta g_{t-2}) + \eta g_{t-1} \\
    & \vdots \\
    &= \eta \sum_{i=0}^{t-1} \gamma^i g_{t-1-i}
\end{align*}
$$

This sum for $$v_t$$ is an **Exponentially Weighted Moving Average (EWMA)** of the scaled past gradients ($$\eta g_k$$), with more recent gradients receiving higher weights (due to smaller powers of $$\gamma$$).
If the gradient $$g$$ were constant (i.e., $$g_k = g$$ for all $$k$$), then $$v_t$$ would become $$\eta g \sum_{i=0}^{t-1} \gamma^i$$. As $$t \to \infty$$, this geometric series sum approaches $$\frac{1}{1-\gamma}$$, so $$v_t \to \frac{\eta g}{1-\gamma}$$.
For example, if $$\gamma = 0.9$$, the effective step size in a consistent direction can become up to $$1/(1-0.9) = 10$$ times larger than the step dictated by the current gradient alone ($$\eta g$$). This allows momentum to "power through" flat regions where gradients are small but consistent.

**2. Dampening Oscillations:**
Consider the scenario of a narrow ravine where the gradient sharply points towards the center of the ravine. In vanilla GD, this causes oscillations across the ravine.
With momentum, if the gradient components alternate in sign (e.g., positive then negative in the direction across the ravine), their contributions to the velocity $$v_t$$ will tend to cancel each other out over time. For example, if the scaled gradient $$\eta g_k$$ in one dimension is $$+\delta$$ and $$\eta g_{k+1}$$ is $$-\delta$$, their contributions to $$v_t$$ will be, after some unrolling, terms like $$\dots + \gamma \eta \delta - \eta \delta + \dots$$. This averaging effect, inherent in the EWMA nature of $$v_t$$, smooths out the trajectory and reduces the magnitude of oscillations, allowing for more steadfast progress along the valley floor.

The result is often a much smoother and faster path to the minimum, especially in challenging landscapes.

## Further Insights and Connections

The intuitive effects of acceleration and dampening stem from deeper mathematical properties of the momentum update.

### Momentum and Variance Reduction in SGD

In Stochastic Gradient Descent (SGD), each gradient $$g_t = \nabla J(\theta_{t-1}; \text{batch}_t)$$ is a noisy estimate of the true gradient over the entire dataset. The velocity $$v_t$$, being an EWMA of these noisy scaled gradients, effectively averages them. This has a crucial consequence: **variance reduction**.

*   **Averaging Smooths Noise:** Just as the average of multiple noisy measurements is generally more reliable than a single measurement, the EWMA $$v_t$$ provides a more stable update direction than relying solely on the noisy current gradient $$\eta g_t$$.
*   **Effective Window:** The momentum parameter $$\gamma$$ controls the "memory" of this average. A higher $$\gamma$$ means more past gradients contribute, leading to greater smoothing. The effective number of past gradients being averaged can be thought of as roughly $$1/(1-\gamma)$$. For $$\gamma=0.9$$, this is about 10 steps; for $$\gamma=0.99$$, it's about 100 steps.
*   **Impact:** By reducing the variance of the update steps, momentum helps SGD converge more smoothly and often faster, especially when mini-batch sizes are small or the gradient noise is high. It prevents the optimizer from being thrown off course by errant individual mini-batch gradients.

### Connection to Polyak's Heavy Ball Method

The momentum update rule we've discussed is closely related to, and can be shown to be equivalent to, a method introduced by Boris Polyak in 1964, known as the "Heavy Ball" method.
Let's rewrite our parameter update $$\theta_t = \theta_{t-1} - v_t$$ and the velocity update $$v_t = \gamma v_{t-1} + \eta \nabla J(\theta_{t-1})$$.
We can express $$v_{t-1}$$ in terms of parameter differences: since $$\theta_{t-1} = \theta_{t-2} - v_{t-1}$$, it follows that $$v_{t-1} = \theta_{t-2} - \theta_{t-1}$$.
Substituting this into the parameter update equation:

$$
\begin{align*}
\theta_t &= \theta_{t-1} - (\gamma v_{t-1} + \eta \nabla J(\theta_{t-1})) \\
         &= \theta_{t-1} - \gamma (\theta_{t-2} - \theta_{t-1}) - \eta \nabla J(\theta_{t-1}) \\
         &= \theta_{t-1} + \gamma (\theta_{t-1} - \theta_{t-2}) - \eta \nabla J(\theta_{t-1})
\end{align*}
$$

This form:

$$
\theta_t = \theta_{t-1} + \underbrace{\gamma (\theta_{t-1} - \theta_{t-2})}_\text{momentum term} - \underbrace{\eta \nabla J(\theta_{t-1})}_\text{gradient term}
$$

is precisely Polyak's Heavy Ball method. Here, $$\theta_{t-1} - \theta_{t-2}$$ is the previous step taken. The update is the current position plus a fraction of the previous step (the "momentum") minus the scaled current gradient.

*   **Significance:** Polyak derived this method by discretizing a second-order differential equation describing the motion of a heavy ball with mass and friction rolling on a surface. This provides a strong theoretical underpinning for the physical analogy.
*   **Perspective:** This two-step recurrence for $$\theta_t$$ (depending on $$\theta_{t-1}$$ and $$\theta_{t-2}$$) highlights that momentum is a type of **linear multi-step method** in numerical ODE simulation, incorporating more history than just the last position and gradient. Thus, the existing research on the stability of such methods can offer insights into the behavior of momentum in optimization.

### Revisiting Ill-Conditioned Landscapes

The properties discussed—EWMA for acceleration and smoothing, variance reduction, and the heavy ball interpretation—collectively explain why momentum is so effective in ill-conditioned landscapes (like narrow ravines):
*   **Consistent gradient components** (along the valley floor) are amplified by the EWMA, leading to acceleration in that direction.
*   **Oscillatory gradient components** (across the ravine) are averaged out by the EWMA, dampening wasteful zig-zagging.
*   **Reduced variance** from SGD noise means the optimizer is less likely to be jolted away from the consistent path along the ravine by a single noisy gradient.

Essentially, momentum allows the optimizer to build up "speed" in directions of steady descent while "gliding over" noisy or rapidly changing transverse gradients.

<details class="details-block" markdown="1">
<summary markdown="1">
**Analogy: The Heavy Ball**
</summary>
Think of the parameter vector $$\theta$$ as the position of a heavy ball, and the loss function $$J(\theta)$$ as the surface it rolls on.
- The gradient $$-\nabla J(\theta)$$ is like a force pulling the ball downhill.
- The velocity $$v_t$$ (or the step $$\theta_t - \theta_{t-1}$$ in Polyak's form) is the actual velocity of the ball.
- The momentum term $$\gamma v_{t-1}$$ (or $$\gamma(\theta_{t-1} - \theta_{t-2})$$) represents the inertia of the ball; it wants to keep moving in its current direction.
- The learning rate $$\eta$$ influences how strongly the current force (gradient) affects the ball's acceleration.

A high $$\gamma$$ means a heavier ball (more inertia), which takes longer to change direction but can bulldoze through small obstacles or shallow regions. A low $$\gamma$$ means a lighter ball, more responsive to immediate changes in the terrain. The Polyak formulation makes this physical analogy very direct.
</details>

## A Glimpse Beyond: Nesterov Accelerated Gradient (NAG)

Standard momentum calculates the gradient at the current position $$\theta_{t-1}$$ and then adds the scaled current gradient $$\eta \nabla J(\theta_{t-1})$$ to the decayed previous velocity $$\gamma v_{t-1}$$. This means the "momentum part" of the step ($$\gamma v_{t-1}$$) is taken without knowing what the gradient will be *after* that part of the step. This can sometimes lead to overshooting, especially if the accumulated velocity is large.

Yurii Nesterov proposed a clever modification, now known as Nesterov Accelerated Gradient (NAG) or Nesterov Momentum. The core idea is to "look ahead" before computing the gradient.

1.  **Approximate future position:** First, make a partial step based *only* on the previous velocity:

    $$
    \tilde{\theta}_{t-1} = \theta_{t-1} - \gamma v_{t-1}
    $$

    (continued) This $$\tilde{\theta}_{t-1}$$ is an approximation of where the parameters would be if we only considered the momentum from the previous step. (Note: if using the Polyak form, this look-ahead involves $$\theta_{t-1} + \gamma(\theta_{t-1} - \theta_{t-2})$$).
2.  **Compute gradient at look-ahead point:** Calculate the gradient not at the current position $$\theta_{t-1}$$, but at this "look-ahead" point $$\tilde{\theta}_{t-1}$$:

    $$
    g_t = \nabla J(\tilde{\theta}_{t-1})
    $$

3.  **Update velocity and parameters (as before):**

    $$
    v_t = \gamma v_{t-1} + \eta g_t
    $$

    $$
    \theta_t = \theta_{t-1} - v_t
    $$

**Intuition for NAG:**
By computing the gradient at the point where momentum is about to carry us ($$\tilde{\theta}_{t-1}$$), NAG gets a better sense of what will happen *after* the momentum update. If the momentum step is leading into a region where the surface curves upwards (i.e., the gradient starts pointing back), NAG will "see" this earlier and can correct its course more effectively. It acts as a smarter correction factor, often leading to faster convergence and preventing oscillations more robustly than standard momentum, particularly in convex optimization settings. It's like a ball that can peek ahead slightly before committing to its full momentum-driven roll.

## Benefits and Considerations of Using Momentum

Incorporating momentum into gradient descent offers several compelling advantages:

**Benefits:**
*   **Faster Convergence:** Often significantly speeds up convergence, especially in landscapes with ravines, plateaus, or noisy gradients.
*   **Smoother Optimization Trajectory:** Reduces oscillations and variance in updates, leading to a more direct path to the minimum.
*   **Navigating Obstacles:** The accumulated velocity can help the optimizer "roll over" small local minima or saddle points in some cases (though it's not a foolproof solution for all non-convex challenges).

**Considerations:**
*   **New Hyperparameter ($$\gamma$$):** Momentum introduces the coefficient $$\gamma$$, which needs to be tuned alongside the learning rate $$\eta$$. Poor choices can lead to suboptimal performance, overshooting, or instability. Common values for $$\gamma$$ like 0.9 often work well as a starting point.
*   **Overshooting:** If $$\gamma$$ is too high or $$\eta$$ is too large, the accumulated velocity can cause the optimizer to overshoot the minimum and oscillate wildly.
*   **Physical Analogy Limits:** While the "heavy ball" analogy is intuitive and has theoretical backing (Polyak's method), optimization dynamics are not perfectly analogous to physical systems. Its primary strength lies in effective gradient averaging and consistent direction amplification.

## Reflection: The Power of Memory in Optimization

The introduction of momentum marked a significant evolution from "memoryless" first-order methods like basic SGD. It demonstrated a fundamental principle: **incorporating history into the update rule can substantially enhance optimization performance.** By maintaining a velocity that aggregates past gradients (an EWMA), momentum-based methods achieve a more nuanced understanding of the loss landscape's geometry, allowing for more intelligent steps. This historical perspective manifests as accelerated convergence in consistent directions, damped oscillations in noisy or ravine-like terrains, and reduced variance in stochastic settings.

The connection to Polyak's Heavy Ball method grounds momentum in the physics of damped oscillators, providing a theoretical justification for its effectiveness. This idea of accumulating past information doesn't stop with momentum. It's a cornerstone of many advanced optimization algorithms. For example:
*   **Adaptive learning rate methods** (like AdaGrad, RMSprop, Adam) also maintain moving averages – not just of the gradients themselves (like momentum's first moment), but also of their squared values (to estimate a per-parameter second moment, akin to variance).
*   Momentum, particularly its Nesterov variant, often serves as a component within these more sophisticated optimizers.

The success of momentum prompts further questions:
*   What other types of information from past iterations could be beneficial to accumulate?
*   How can we make the accumulation process itself (like the decay rate $$\gamma$$) adaptive to the problem or the stage of optimization?
*   Can we combine the benefits of momentum with adaptive learning rates in a principled way?

These questions have driven much of the research in optimization for machine learning, leading to the powerful algorithms we use today. Understanding momentum is not just about grasping one technique; it's about appreciating a key shift in perspective towards more "history-aware" optimization.

---

This exploration of momentum should provide a solid foundation for understanding why it works and how it improves upon simpler gradient descent methods. In subsequent posts, we'll see how this concept of accumulating information is further developed in more advanced optimizers.

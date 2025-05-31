---
title: "Momentum: A Tale of Two ODEs and a Multi-Step Method"
date: 2025-05-18 02:57 -0400
series_index: 10
mermaid: true
description: "Delving into the momentum optimization method, understanding its origins from both second-order 'heavy ball' dynamics and as a linear multi-step method for first-order gradient flow, and extending to adaptive methods like RMSProp and Adam."
image: # Optional: path to an image
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Momentum
- Optimization Algorithms
- ODE Discretization
- Gradient Descent
- Polyak Heavy Ball
- Nesterov Accelerated Gradient
- RMSProp
- Adam
- Adaptive Learning Rates
- Linear Multi-step Methods
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
  symbol; use \vert and \Vert.

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

## Introduction: The "Rolling Ball" Intuition

When we optimize a function, especially in high-dimensional spaces typical of machine learning, vanilla Gradient Descent (GD) can be painstakingly slow. It often zig-zags down narrow valleys and gets stuck in flat regions or saddle points. The **momentum method** is a popular technique to overcome these issues by drawing an analogy to a physical system: a ball rolling down a hill. This ball not only moves in the direction of steepest descent (the gradient) but also accumulates "momentum" from its past movements, helping it to power through flat regions and dampen oscillations in ravines.

In this post, we'll explore two fundamental ways to derive and understand momentum, particularly Polyak's Heavy Ball (PHB) method:
1.  As a discretization of a **system of first-order Ordinary Differential Equations (ODEs)** derived from a second-order ODE representing a physical system (the "heavy ball" with friction).
2.  As a **Linear Multi-step Method (LMM)** applied to the first-order gradient flow ODE.

We will also extend this ODE perspective to understand adaptive optimization algorithms like RMSProp and Adam. These perspectives reveal that momentum and related techniques are not just ad-hoc tricks but principled approaches rooted in the mathematics of dynamical systems and numerical analysis.

## Recap: Gradient Descent and the Gradient Flow ODE

Before diving into momentum, let's briefly recall Gradient Descent. To minimize a differentiable function $$f(x)$$, GD iteratively updates the parameters $$x$$ in the direction opposite to the gradient $$\nabla f(x)$$:

$$
x_{k+1} = x_k - \eta \nabla f(x_k)
$$

where $$\eta > 0$$ is the learning rate.

This update can be seen as an **explicit** or **forward Euler discretization** of the **gradient flow ODE**:

$$
\dot{x}(t) = -\nabla f(x(t))
$$

where $$\dot{x}(t)$$ denotes the derivative of $$x$$ with respect to time $$t$$. The solutions to this ODE, called gradient flow trajectories, continuously follow the path of steepest descent.

## Perspective 1: Momentum from a Second-Order ODE (The Heavy Ball)

The physical intuition of a ball rolling down a hill can be formalized using Newton's second law of motion. Consider a particle of mass $$m$$ moving in a potential field defined by $$f(x)$$. The force exerted on the particle by the potential is $$-\nabla f(x)$$. If there's also a friction or viscous drag force proportional to its velocity $$\dot{x}(t)$$ with a damping coefficient $$\gamma \ge 0$$, the equation of motion is:

$$
m \ddot{x}(t) + \gamma \dot{x}(t) + \nabla f(x(t)) = 0
$$

Here, $$\ddot{x}(t)$$ is the acceleration. This is a second-order ODE.

To derive an optimization algorithm, a standard approach is to first convert this second-order ODE into an equivalent system of two first-order ODEs. Let $$v_{phys}(t) = \dot{x}(t)$$ be the physical velocity. Then $$\dot{v}_{phys}(t) = \ddot{x}(t)$$, and substituting this into the equation of motion (after dividing by $$m$$) yields:
$$ \dot{v}_{phys}(t) = -\frac{\gamma}{m} v_{phys}(t) - \frac{1}{m} \nabla f(x(t)) $$.
Thus, the second-order ODE is equivalent to the following system of first-order ODEs:

$$
\begin{align*}
\dot{x}(t) &= v_{phys}(t) \\
\dot{v}_{phys}(t) &= -\frac{\gamma}{m} v_{phys}(t) - \frac{1}{m} \nabla f(x(t))
\end{align*}
$$

We now discretize this system to obtain an iterative optimization algorithm. The specific way we approximate the derivatives and evaluate terms will determine the resulting algorithm.

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of Polyak's Heavy Ball from the System of First-Order ODEs**
</summary>
Let $$h > 0$$ be the time discretization step size. We denote $$x_k \approx x(kh)$$ and $$v_{phys,k} \approx v_{phys}(kh)$$.

We discretize the system of ODEs as follows:
1.  For the position update $$\dot{x}(t) = v_{phys}(t)$$, we use the velocity at the *end* of the interval, $$v_{phys,k+1}$$, to determine the change in position:

    $$
    \frac{x_{k+1} - x_k}{h} = v_{phys,k+1} \quad \implies \quad x_{k+1} = x_k + h v_{phys,k+1}
    $$

2.  For the velocity update $$\dot{v}_{phys}(t) = -\frac{\gamma}{m} v_{phys}(t) - \frac{1}{m} \nabla f(x(t))$$, we use a forward Euler discretization, evaluating terms at time $$t_k = kh$$:

    $$
    \frac{v_{phys,k+1} - v_{phys,k}}{h} = -\frac{\gamma}{m} v_{phys,k} - \frac{1}{m} \nabla f(x_k)
    $$

    $$
    \implies v_{phys,k+1} = v_{phys,k} - \frac{h\gamma}{m} v_{phys,k} - \frac{h}{m} \nabla f(x_k) = \left(1 - \frac{h\gamma}{m}\right)v_{phys,k} - \frac{h}{m} \nabla f(x_k)
    $$

This specific choice of discretization (updating velocity first using $$x_k$$, then updating position using the new velocity) is a form of symplectic Euler method, often used for Hamiltonian systems, though here applied to a dissipative system.

Now, let's define the "momentum" term $$v_k$$ as it appears in the standard algorithm. This term is typically a scaled version of the physical velocity, representing the accumulated change. Let $$v_k := h v_{phys,k}$$. Then $$v_{k+1} = h v_{phys,k+1}$$.
Substituting these into our discretized equations:

-   The velocity update becomes:

    $$
    \frac{v_{k+1}}{h} = \left(1 - \frac{h\gamma}{m}\right)\frac{v_k}{h} - \frac{h}{m} \nabla f(x_k)
    $$

    Multiplying by $$h$$ gives:

    $$
    v_{k+1} = \left(1 - \frac{h\gamma}{m}\right)v_k - \frac{h^2}{m} \nabla f(x_k)
    $$

-   The position update becomes:

    $$
    x_{k+1} = x_k + v_{k+1}
    $$

If we define the learning rate $$\eta = \frac{h^2}{m}$$ and the momentum parameter $$\beta = 1 - \frac{h\gamma}{m}$$, the update rules are:

$$
\begin{align*}
v_{k+1} &= \beta v_k - \eta \nabla f(x_k) \\
x_{k+1} &= x_k + v_{k+1}
\end{align*}
$$

This is precisely the two-variable form of the Polyak's Heavy Ball (PHB) method.

**Equivalence to the single-variable form:**
This two-variable system can be rewritten as a single second-order difference equation for $$x_k$$. From the position update, we have $$v_{k+1} = x_{k+1} - x_k$$. Consequently, $$v_k = x_k - x_{k-1}$$ (for $$k \ge 1$$, assuming consistent initialization).
Substituting these into the velocity update equation:

$$
(x_{k+1} - x_k) = \beta (x_k - x_{k-1}) - \eta \nabla f(x_k)
$$

Rearranging for $$x_{k+1}$$, we get the single-variable form of PHB:

$$
x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})
$$

**Connection back to the original second-order ODE:**
This second-order difference equation can be shown to be a direct finite difference approximation of the original second-order ODE.
Rearranging the single-variable form:

$$
x_{k+1} - (1+\beta)x_k + \beta x_{k-1} = -\eta \nabla f(x_k)
$$

Substitute back the definitions of $$\eta = \frac{h^2}{m}$$ and $$\beta = 1 - \frac{h\gamma}{m}$$:

$$
x_{k+1} - \left(1 + \left(1 - \frac{h\gamma}{m}\right)\right)x_k + \left(1 - \frac{h\gamma}{m}\right)x_{k-1} = -\frac{h^2}{m} \nabla f(x_k)
$$

$$
x_{k+1} - \left(2 - \frac{h\gamma}{m}\right)x_k + \left(1 - \frac{h\gamma}{m}\right)x_{k-1} = -\frac{h^2}{m} \nabla f(x_k)
$$

Multiply by $$m/h^2$$:

$$
\frac{m}{h^2} \left( x_{k+1} - \left(2 - \frac{h\gamma}{m}\right)x_k + \left(1 - \frac{h\gamma}{m}\right)x_{k-1} \right) + \nabla f(x_k) = 0
$$

Group terms to match standard finite difference formulas:

$$
\frac{m}{h^2} \left( (x_{k+1} - 2x_k + x_{k-1}) + \frac{h\gamma}{m}(x_k - x_{k-1}) \right) + \nabla f(x_k) = 0
$$

$$
m \left( \frac{x_{k+1} - 2x_k + x_{k-1}}{h^2} \right) + \gamma \left( \frac{x_k - x_{k-1}}{h} \right) + \nabla f(x_k) = 0
$$

This equation corresponds to approximating the original ODE $$m \ddot{x}(t) + \gamma \dot{x}(t) + \nabla f(x(t)) = 0$$ at time $$t_k = kh$$ using:
-   Central difference for acceleration: $$\ddot{x}(t_k) \approx \frac{x_{k+1} - 2x_k + x_{k-1}}{h^2}$$
-   Backward difference for velocity (in the damping term): $$\dot{x}(t_k) \approx \frac{x_k - x_{k-1}}{h}$$
This confirms that the derived PHB algorithm is a consistent discretization of the heavy ball ODE. The system-based derivation makes the choice of these specific finite differences more systematic.
</details>

The derivation above shows that discretizing the system of first-order ODEs (equivalent to the second-order "heavy ball" ODE) leads directly to the Polyak's Heavy Ball method. The algorithm's parameters are related to the physical system's parameters and the discretization step size as follows:
-   Learning rate: $$\eta = \frac{h^2}{m}$$
-   Momentum parameter: $$\beta = 1 - \frac{h\gamma}{m}$$

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Polyak's Heavy Ball (PHB) Method**
</div>
The update rule for Polyak's Heavy Ball method is often expressed in two equivalent forms:

1.  **Two-variable (velocity) form:**

    $$
    v_{k+1} = \beta v_k - \eta \nabla f(x_k)
    $$

    $$
    x_{k+1} = x_k + v_{k+1}
    $$

    Here, $$v_k$$ is the momentum term. Initializing $$v_0 = 0$$ is common.

2.  **Single-variable (three-term recurrence) form:**

    $$
    x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})
    $$

    This form requires $$x_0$$ and $$x_{-1}$$ (or $$x_1$$ to be computed differently, e.g., $$x_1 = x_0 - \eta \nabla f(x_0)$$ which corresponds to $$v_0=0$$ giving $$v_1 = -\eta \nabla f(x_0)$$, so $$x_1 = x_0 - \eta \nabla f(x_0)$$; and thus for $$k=0$$, $$x_0-x_{-1}$$ would be zero).

The parameters are the learning rate $$\eta > 0$$ and the momentum coefficient $$\beta \in [0, 1)$$.
The two forms are equivalent if $$v_k = x_k - x_{k-1}$$ (for $$k \ge 1$$), or more generally, if $$v_0$$ is initialized, then $$v_k$$ represents the accumulated momentum, and $$x_{k+1}-x_k = v_{k+1}$$.
</blockquote>

This derivation shows that the momentum term arises naturally from the inertia ($$m$$) and damping ($$\gamma$$) of the physical system, as captured by the discretization scheme:
-   A larger mass $$m$$ (relative to $$h^2$$) implies a smaller effective learning rate $$\eta$$.
-   A larger friction coefficient $$\gamma$$ (relative to $$m/h$$) implies a smaller momentum parameter $$\beta$$.
-   If $$\gamma h / m = 1$$, then $$\beta=0$$. In this case, $$v_{k+1} = -\eta \nabla f(x_k)$$ and $$x_{k+1} = x_k - \eta \nabla f(x_k)$$, effectively recovering standard Gradient Descent (assuming $$v_k$$ becomes zero if $$\beta=0$$ or it's the first step after initialization with $$v_0=0$$). More generally, if $$\beta=0$$, the single-variable form becomes $$x_{k+1} = x_k - \eta \nabla f(x_k)$$, which is GD. This scenario corresponds to a system where momentum effects from past steps (beyond the current gradient) are not explicitly carried forward.
-   If $$\gamma = 0$$ (no friction), then $$\beta=1$$. This can lead to oscillations or instability if not paired with a small enough step size $$\eta$$.

## Perspective 2: Momentum as a Linear Multi-step Method

Another powerful way to understand momentum is by viewing it as a **Linear Multi-step Method (LMM)** for solving an ODE. LMMs are a class of numerical methods that use information from previous time steps to approximate the solution at the current step.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Linear Multi-step Method (LMM)**
</div>
For an initial value problem $$\dot{y}(t) = F(t, y(t))$$ with $$y(t_0) = y_0$$, an $$s$$-step LMM is defined by:

$$
\sum_{j=0}^s \alpha_j y_{n+j} = h_{LMM} \sum_{j=0}^s \beta_j F(t_{n+j}, y_{n+j})
$$

where $$h_{LMM}$$ is the step size, $$\alpha_j$$ and $$\beta_j$$ are method-specific coefficients, and by convention $$\alpha_s=1$$.
- If $$\beta_s = 0$$, the method is **explicit**.
- If $$\beta_s \neq 0$$, the method is **implicit**.
</blockquote>

Let's consider the first-order gradient flow ODE again:

$$
\dot{x}(t) = -\nabla f(x(t))
$$

Here, $$y(t) \equiv x(t)$$ and $$F(t, x(t)) \equiv -\nabla f(x(t))$$.
Now, let's look at the Polyak's Heavy Ball update rule in its single-variable form:

$$
x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})
$$

We can rearrange this to:

$$
x_{k+1} - (1+\beta)x_k + \beta x_{k-1} = -\eta \nabla f(x_k)
$$

This equation perfectly matches the LMM form. It's a 2-step method ($$s=2$$). Let $$n+j$$ map to our indices such that $$n+s = k+1$$. So, if we set $$n=k-1$$, then:
-   $$y_{n+2} = x_{k+1}$$
-   $$y_{n+1} = x_k$$
-   $$y_{n} = x_{k-1}$$

The coefficients $$\alpha_j$$ are:
-   $$\alpha_2 = 1$$ (since $$j=s=2$$)
-   $$\alpha_1 = -(1+\beta)$$
-   $$\alpha_0 = \beta$$

For the right-hand side, the term $$-\eta \nabla f(x_k)$$ involves $$F(t_{n+1}, y_{n+1}) = -\nabla f(x_k)$$.
Comparing $$h_{LMM} \sum_{j=0}^s \beta_j F(t_{n+j}, y_{n+j})$$ with $$-\eta \nabla f(x_k)$$:
-   The method is explicit, so $$\beta_s = \beta_2 = 0$$.
-   The gradient is evaluated only at $$x_k \equiv y_{n+1}$$.
-   So, we must have $$\beta_0 = 0$$.
-   This leaves $$h_{LMM} \beta_1 F(t_{n+1}, y_{n+1}) = -\eta \nabla f(x_k)$$.
-   Since $$F(t_{n+1}, y_{n+1}) = -\nabla f(x_k)$$, we get $$h_{LMM} \beta_1 (-\nabla f(x_k)) = -\eta \nabla f(x_k)$$.
-   This implies $$h_{LMM} \beta_1 = \eta$$. We can choose $$h_{LMM}=1$$ (absorbing it into $$\eta$$ or considering it as a reference step size for the LMM definition), then $$\beta_1 = \eta$$.
    Alternatively, we can set the LMM step size $$h_{LMM} = \eta$$ and then $$\beta_1=1$$. Or, more generally, the coefficients are $$\beta_2=0, \beta_0=0$$ and $$h_{LMM}\beta_1 = \eta$$.

Thus, Polyak's Heavy Ball method *is* a specific explicit 2-step linear multi-step method applied to the gradient flow ODE $$\dot{x} = -\nabla f(x)$$. The "momentum" term $$(x_k - x_{k-1})$$ is how this LMM incorporates past information ($$x_{k-1}$$) to determine the next step $$x_{k+1}$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Characteristic Polynomials and Consistency of the LMM**
</summary>
For an LMM, we define two characteristic polynomials:
- First characteristic polynomial: $$\rho(z) = \sum_{j=0}^s \alpha_j z^j$$
- Second characteristic polynomial: $$\sigma(z) = \sum_{j=0}^s \beta_j z^j$$

For Polyak's method, using our derived coefficients (with $$h_{LMM}=1$$ for simplicity in defining $$\beta_j$$):
- $$\rho(z) = \alpha_2 z^2 + \alpha_1 z + \alpha_0 = z^2 - (1+\beta)z + \beta$$
- $$\sigma(z) = \beta_2 z^2 + \beta_1 z + \beta_0 = \eta z$$ (since only $$\beta_1=\eta$$ is non-zero among $$\beta_0, \beta_1, \beta_2$$)

An LMM is consistent with the ODE $$\dot{y} = F(t,y)$$ if it has order $$p \ge 1$$. This requires two conditions:
1.  $$\rho(1) = 0$$
2.  $$\rho'(1) = \sigma(1)$$

Let's check these for PHB as an LMM for $$\dot{x} = -\nabla f(x)$$:
-   $$\rho(1) = 1^2 - (1+\beta)(1) + \beta = 1 - 1 - \beta + \beta = 0$$. This condition is satisfied for any $$\beta$$.
-   For the second condition:
    $$\rho'(z) = 2z - (1+\beta)$$, so $$\rho'(1) = 2 - (1+\beta) = 1-\beta$$.
    $$\sigma(1) = \eta \cdot 1 = \eta$$.
    So, consistency requires $$1-\beta = \eta$$.

This particular relationship, $$1-\beta = \eta$$, is known as the "critically damped" setting for PHB when analyzing its convergence on quadratic functions. It implies a specific tuning between the momentum parameter and the learning rate for the method to be a first-order accurate approximation (in the LMM sense) of the gradient flow ODE $$\dot{x} = -\nabla f(x)$$.
However, PHB is an effective optimization algorithm even when $$1-\beta \neq \eta$$. The LMM formulation describes its algebraic structure as a numerical integrator. Its effectiveness as an optimizer is analyzed through other means (e.g., Lyapunov stability for the discrete updates, convergence rates on specific function classes). The fact that it *is* an LMM by its form is significant, regardless of whether this specific consistency condition (which relates to the local truncation error of the LMM) is met for arbitrary $$\eta, \beta$$.
</details>

## Connecting the Two Perspectives

The beauty is that these two perspectives are deeply connected.
1.  We started with a physical system (heavy ball with friction) described by a **second-order ODE**.
2.  Converting this to a system of first-order ODEs and then discretizing it led directly to the Polyak's Heavy Ball update rule (in its two-variable and equivalent single-variable forms).
3.  This update rule, particularly in the single-variable form $$x_{k+1} - (1+\beta)x_k + \beta x_{k-1} = -\eta \nabla f(x_k)$$, has the algebraic structure of a **2-step Linear Multi-step Method**.
4.  This LMM can be seen as a way to solve the simpler **first-order gradient flow ODE** $$\dot{x} = -\nabla f(x)$$, but using information from multiple past steps ($$x_k, x_{k-1}$$) instead of just one ($$x_k$$) as in Euler's method (standard GD).

So, the "inertia" from the physical model (which persists velocity) translates into the "memory" of the LMM (which uses previous iterates). The momentum term, crucial for navigating complex optimization landscapes, is essentially how the LMM leverages past states to make a more informed step, aiming to better approximate the trajectory of the underlying gradient flow dynamics or, from the other perspective, to emulate the behavior of a physical system with inertia.

## A Glimpse at Nesterov's Accelerated Gradient (NAG)

Nesterov's Accelerated Gradient (NAG) is another highly successful momentum-based method, often outperforming PHB, especially in convex optimization. Its update rule is subtly different:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Nesterov's Accelerated Gradient (NAG)**
</div>
Using an explicit velocity term $$v_k$$:

$$
v_{k+1} = \beta v_k - \eta \nabla f(x_k + \beta v_k)
$$

$$
x_{k+1} = x_k + v_{k+1}
$$

The key difference from PHB is that the gradient is evaluated at a "look-ahead" point $$x_k + \beta v_k$$, rather than at $$x_k$$. (Note: there are multiple formulations of NAG; this is a common one. Sometimes the lookahead is on $$x_k - \beta v_k$$ if $$v_k$$ is defined as $$x_k-x_{k-1}$$, or the update $$x_{k+1} = (x_k + \beta v_k) + v_{k+1} - \beta v_k$$, i.e. $$x_{k+1} = x_k + v_{k+1} + \beta(v_{k+1}-v_k)$$.)
</blockquote>

Interestingly, NAG also has a continuous-time ODE interpretation. Su, Boyd, and Candès (2014) showed that a version of NAG can be seen as a discretization of the following second-order ODE:

$$
\ddot{x}(t) + \frac{k_d}{t} \dot{x}(t) + \nabla f(x(t)) = 0
$$

For a typical choice for the constant $$k_d \ge 3$$. Notice the time-dependent damping term $$\frac{k_d}{t}$$. As time $$t$$ increases, the damping decreases. This "adaptive" damping is thought to be one reason for NAG's superior performance in some settings. Deriving NAG from this ODE involves a more complex discretization scheme than the one used for PHB.

While NAG can also be written in a multi-step form, its "look-ahead" gradient evaluation makes its LMM interpretation for the simple gradient flow ODE $$\dot{x} = -\nabla f(x)$$ less direct or natural than for PHB. It's more directly a discretization of its own unique second-order ODE.

## Adaptive Methods: RMSProp and Adam from an ODE Perspective

Adaptive optimization methods like RMSProp and Adam adjust the learning rate for each parameter dynamically, often based on moving averages of past gradients and squared gradients. Their continuous-time interpretations typically involve systems of coupled ODEs, reflecting this dynamic interplay. For the derivations below, we simplify by omitting the small $$\epsilon$$ typically added to denominators for numerical stability and the bias correction terms in Adam, focusing on the core dynamics. This helps in revealing the underlying ODE structure. The condition "don't update when the gradient is null" is naturally handled if the update step is proportional to the gradient (like in RMSProp). For methods with momentum (like Adam), a null current gradient means no new "push", but past momentum can still cause an update.

### RMSProp: Adaptive Scaling of Gradients

RMSProp (Root Mean Square Propagation) scales the learning rate for each parameter by an estimate of the root mean square of recent gradients for that parameter.

The discrete update rules for RMSProp (simplified, without $$\epsilon$$ and assuming component-wise operations for vectors like $$(\nabla f(x_k))^2$$ and division/sqrt) are:
1.  Update the squared gradient accumulator $$s_k$$:

    $$
    s_{k+1} = \beta_2 s_k + (1-\beta_2) (\nabla f(x_k))^2
    $$

2.  Update parameters $$x_k$$:

    $$
    x_{k+1} = x_k - \frac{\eta}{\sqrt{s_{k+1}}} \nabla f(x_k)
    $$

    Here, $$\beta_2$$ is the decay rate for the moving average of squared gradients (typically close to 1, e.g., 0.999), and $$\eta$$ is the learning rate. If any component of $$\sqrt{s_{k+1}}$$ is zero, the update for that component of $$x_k$$ would be undefined or handled by a safeguard (like adding $$\epsilon$$ or skipping the update for that component). If $$\nabla f(x_k)=0$$, then $$x_{k+1}=x_k$$, and $$s_{k+1} = \beta_2 s_k$$ (the accumulator decays).

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of the ODE System for RMSProp**
</summary>
We derive the continuous-time limit by considering a small time step $$h_{step} > 0$$ for the algorithm's iteration, which we'll map to the passage of time in the ODE.
Let $$s_k \approx s(t_k)$$, $$x_k \approx x(t_k)$$, where $$t_{k+1} - t_k = h_{step}$$.
1.  For the accumulator $$s_k$$:

    $$
    s_{k+1} - s_k = (\beta_2 - 1)s_k + (1-\beta_2)(\nabla f(x_k))^2 = (1-\beta_2)((\nabla f(x_k))^2 - s_k)
    $$

    Dividing by $$h_{step}$$ and taking the limit $$h_{step} \to 0$$:

    $$
    \dot{s}(t) = \lim_{h_{step} \to 0} \frac{s(t+h_{step}) - s(t)}{h_{step}} = \frac{1-\beta_2}{h_{eff,s}}((\nabla f(x(t)))^2 - s(t))
    $$

    If we consider each step $$k$$ of the algorithm to correspond to a unit time step in the ODE discretization (i.e., setting the effective time scale $$h_{eff,s}=1$$), then let $$\alpha_s = 1-\beta_2$$. The ODE becomes:

    $$
    \dot{s}(t) = \alpha_s ((\nabla f(x(t)))^2 - s(t))
    $$

    This means $$s(t)$$ is an exponential moving average of past squared gradients $$(\nabla f(x(t)))^2$$, with a time constant $$T_s = 1/\alpha_s = 1/(1-\beta_2)$$.

2.  For the parameter update $$x_k$$:

    $$
    x_{k+1} - x_k = -\frac{\eta}{\sqrt{s_{k+1}}} \nabla f(x_k)
    $$

    Dividing by $$h_{step}$$ and taking the limit (assuming $$s_{k+1} \approx s(t)$$ or $$s(t+h_{step}) \approx s(t)$$ for the denominator for small $$h_{step}$$):

    $$
    \dot{x}(t) = \lim_{h_{step} \to 0} \frac{x(t+h_{step}) - x(t)}{h_{step}} = -\frac{\eta}{h_{eff,x}\sqrt{s(t)}} \nabla f(x(t))
    $$

    Again, if we set the effective time scale for the x-update $$h_{eff,x}=1$$, the ODE is:

    $$
    \dot{x}(t) = -\frac{\eta_0}{\sqrt{s(t)}} \nabla f(x(t))
    $$

    where $$\eta_0$$ is the learning rate parameter from the algorithm.
</details>

The continuous-time behavior of RMSProp can thus be described by the following system of ODEs:

$$
\begin{align*}
\dot{x}(t) &= -\frac{\eta_0}{\sqrt{s(t)}} \nabla f(x(t)) \\
\dot{s}(t) &= (1-\beta_2) ((\nabla f(x(t)))^2 - s(t))
\end{align*}
$$

(Operations involving $$s(t)$$ and $$(\nabla f(x(t)))^2$$ are element-wise.)

**Interpretation of the RMSProp ODE System:**
-   The equation for $$\dot{s}(t)$$ shows that $$s(t)$$ tracks an exponential moving average of the squared gradients. Each component $$s_i(t)$$ estimates the recent typical magnitude (squared) of the gradient component $$\nabla_i f(x(t))$$.
-   The equation for $$\dot{x}(t)$$ describes a preconditioned gradient flow. The term $$\text{diag}(1/\sqrt{s_i(t)})$$ acts as a diagonal preconditioner. It adaptively scales the learning rate for each parameter: parameters with historically large gradient components (large $$s_i(t)$$) receive smaller effective learning rates, while those with small gradient components receive larger ones.
-   This system describes a particle moving in the potential $$f(x)$$, where its "mobility" or "inverse friction" in each coordinate direction is dynamically adjusted based on the history of forces experienced in that direction.
-   **Connection to SignSGD:** If $$s(t)$$ could track $$(\nabla f(x(t)))^2$$ perfectly and instantaneously (i.e., $$\beta_2 \to 0$$ or equivalently, $$1-\beta_2 \to 1$$, and assuming steady state where $$\dot{s}(t)=0$$), then $$s(t) = (\nabla f(x(t)))^2$$. The update would become $$\dot{x}(t) = -\eta_0 \frac{\nabla f(x(t))}{\vert \nabla f(x(t)) \vert}$$ (element-wise absolute value for the denominator if vector). Since $$s(t)$$ is a smoothed average, RMSProp provides a smoothed approximation to this normalization.

**Can RMSProp be written as a single second-order ODE for $$x(t)$$?**
Unlike the heavy ball method, reducing this system to a single, clean second-order ODE for $$x(t)$$ is not straightforward. If we formally differentiate $$\dot{x}(t)$$ with respect to time:

$$
\ddot{x}(t) = -\eta_0 \frac{d}{dt}\left( \frac{\nabla f(x(t))}{\sqrt{s(t)}} \right) = -\eta_0 \left( \frac{d}{dt}(s(t)^{-1/2}) \nabla f(x(t)) + s(t)^{-1/2} \nabla^2 f(x(t)) \dot{x}(t) \right)
$$

The term $$\frac{d}{dt}(s_i(t)^{-1/2}) = -\frac{1}{2}s_i(t)^{-3/2}\dot{s}_i(t) = -\frac{1}{2}s_i(t)^{-3/2}(1-\beta_2)((\nabla_i f(x(t)))^2 - s_i(t))$$.
Substituting this into the expression for $$\ddot{x}(t)$$ results in a very complex equation where the "coefficients" for $$\dot{x}(t)$$ and terms involving $$\nabla f(x(t))$$ are themselves complicated functions of $$s(t)$$ and $$\nabla f(x(t))$$. It doesn't simplify to the canonical form $$M\ddot{x} + C\dot{x} + \nabla f(x)=0$$ with constant or simple state-dependent $$M, C$$. The system of first-order ODEs is a more natural representation.

### Adam: Adaptive Moments

Adam (Adaptive Moment Estimation) combines the ideas of momentum (first moment of the gradient) and adaptive scaling like RMSProp (second moment of the gradient).

The simplified discrete update rules (omitting bias correction and $$\epsilon$$):
1.  Update biased first moment estimate $$m_k$$ (momentum):

    $$
    m_{k+1} = \beta_1 m_k + (1-\beta_1) \nabla f(x_k)
    $$

2.  Update biased second moment estimate $$s_k$$ (scaling):

    $$
    s_{k+1} = \beta_2 s_k + (1-\beta_2) (\nabla f(x_k))^2
    $$

3.  Update parameters $$x_k$$:

    $$
    x_{k+1} = x_k - \eta \frac{m_{k+1}}{\sqrt{s_{k+1}}}
    $$

Here, $$\beta_1$$ is the decay rate for the momentum term (e.g., 0.9), and $$\beta_2$$ for the squared gradient accumulator (e.g., 0.999). If $$\nabla f(x_k)=0$$, then $$m_{k+1} = \beta_1 m_k$$ and $$s_{k+1} = \beta_2 s_k$$. The parameter $$x_k$$ will still update due to $$m_{k+1}$$ unless $$m_k$$ was zero. This is typical momentum behavior.

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of the ODE System for Adam**
</summary>
Following a similar procedure as for RMSProp, assuming each step $$k$$ corresponds to a unit time step in the ODE discretization:
1.  For the first moment $$m(t)$$:

    $$
    \dot{m}(t) = (1-\beta_1)(\nabla f(x(t)) - m(t))
    $$

    This means $$m(t)$$ is an EMA of past gradients, with time constant $$T_m = 1/(1-\beta_1)$$. It acts like a velocity term that tries to follow the current gradient.

2.  For the second moment $$s(t)$$:

    $$
    \dot{s}(t) = (1-\beta_2)((\nabla f(x(t)))^2 - s(t))
    $$

    This is identical to the RMSProp accumulator ODE, with time constant $$T_s = 1/(1-\beta_2)$$.

3.  For the parameter update $$x(t)$$:

    $$
    \dot{x}(t) = -\eta_0 \frac{m(t)}{\sqrt{s(t)}}
    $$

    The parameters move in the direction of the momentum $$m(t)$$, scaled by $$1/\sqrt{s(t)}$$.
</details>

The continuous-time behavior of Adam can be described by the following system of three coupled ODEs:

$$
\begin{align*}
\dot{x}(t) &= -\eta_0 \frac{m(t)}{\sqrt{s(t)}} \\
\dot{m}(t) &= (1-\beta_1) (\nabla f(x(t)) - m(t)) \\
\dot{s}(t) &= (1-\beta_2) ((\nabla f(x(t)))^2 - s(t))
\end{align*}
$$

(Element-wise operations for $$m(t), s(t)$$, and terms in their updates, and in $$\dot{x}(t)$$ involving division and square root.)

**Interpretation of the Adam ODE System:**
-   Adam combines a momentum-like update (via $$m(t)$$) with adaptive coordinate-wise scaling (via $$s(t)$$).
-   $$\dot{m}(t)$$ shows $$m(t)$$ as a smoothed version of the gradient, providing inertia.
-   $$\dot{s}(t)$$ provides the adaptive scaling based on recent gradient magnitudes.
-   $$\dot{x}(t)$$ uses the smoothed gradient $$m(t)$$ for direction and scales it adaptively.
-   The system describes a particle whose motion has inertia ($$m(t)$$) and whose "mobility" in each direction is adaptively tuned ($$s(t)$$). This can be thought of as a "heavy ball" moving on a dynamically changing, anisotropic landscape.
-   The bias correction terms ($$1/(1-\beta_1^k)$$ and $$1/(1-\beta_2^k)$$) in practical Adam are important for early iterations to correct the initialization bias of $$m_k$$ and $$s_k$$ (which start at 0). In an ODE context, this could be modeled by time-dependent coefficients $$(1-\beta_1(t))$$ and $$(1-\beta_2(t))$$ or by specific initial conditions or scalings of $$m(t)$$ and $$s(t)$$ for small $$t$$. For large $$t$$, these corrections become negligible.

**Can Adam be written as a single second-order ODE for $$x(t)$$?**
This is even more complex than for RMSProp due to the additional ODE for $$m(t)$$.
We can substitute $$m(t) = -\frac{\sqrt{s(t)}}{\eta_0}\dot{x}(t)$$ (from the first ODE) into the ODE for $$\dot{m}(t)$$:

$$
\frac{d}{dt}\left(-\frac{\sqrt{s(t)}}{\eta_0}\dot{x}(t)\right) = (1-\beta_1)\left(\nabla f(x(t)) - \left(-\frac{\sqrt{s(t)}}{\eta_0}\dot{x}(t)\right)\right)
$$

Expanding the left side:

$$
-\left(\frac{d}{dt}\left(\frac{\sqrt{s(t)}}{\eta_0}\right)\right)\dot{x}(t) - \frac{\sqrt{s(t)}}{\eta_0}\ddot{x}(t) = (1-\beta_1)\nabla f(x(t)) + (1-\beta_1)\frac{\sqrt{s(t)}}{\eta_0}\dot{x}(t)
$$

Rearranging to look like a second-order ODE for $$x(t)$$:

$$
\frac{\sqrt{s(t)}}{\eta_0}\ddot{x}(t) + \left[ \frac{d}{dt}\left(\frac{\sqrt{s(t)}}{\eta_0}\right) + (1-\beta_1)\frac{\sqrt{s(t)}}{\eta_0} \right]\dot{x}(t) + (1-\beta_1)\nabla f(x(t)) = 0
$$

This indeed looks like $$M(t,s)\ddot{x}(t) + C(t,s,\dot{s})\dot{x}(t) + K \nabla f(x(t)) = 0$$.
However, the "mass" term $$M(t,s) = \text{diag}(\sqrt{s_i(t)}/\eta_0)$$ and the "damping" term $$C(t,s,\dot{s})$$ depend on $$s(t)$$, which is governed by its own ODE $$\dot{s}(t) = (1-\beta_2)((\nabla f(x(t)))^2 - s(t)) $$. The derivative $$\frac{d}{dt}(\sqrt{s_i(t)}/\eta_0)$$ within $$C(t,s,\dot{s})$$ involves $$\dot{s}_i(t)$$, which in turn depends on $$(\nabla_i f(x(t)))^2$$ and $$s_i(t)$$.
So, the coefficients of this "single" second-order ODE for $$x(t)$$ are not simple constants nor simple functions of just $$x$$ and $$\dot{x}$$. They depend on an auxiliary variable $$s(t)$$ that has its own dynamics driven by the history of gradients encountered along the path $$x(t)$$.
While formally a second-order ODE, its coefficients are so intricately coupled with the state and history via $$s(t)$$ that the system form is generally more transparent for analysis and interpretation.

The ODE perspective for adaptive methods like RMSProp and Adam reveals them as sophisticated dynamical systems that implement preconditioned, momentum-driven descent, where the preconditioning and momentum effects are themselves evolving based on the optimization trajectory. This provides intuition for their ability to navigate complex loss landscapes effectively.

## Why is the ODE Perspective So Valuable?

Understanding optimization algorithms through the lens of ODEs offers several benefits:
1.  **Deeper Intuition:** It moves beyond algorithmic recipes to physical or mathematical analogies, explaining *why* methods like momentum work.
2.  **Principled Design:** New algorithms can be designed by proposing different ODEs (e.g., with different damping or inertial terms, or adaptive preconditioning) and then discretizing them. The conversion to a system of first-order ODEs and then applying various numerical integration schemes provides a rich framework for algorithm discovery.
3.  **Analysis Tools:** The rich theory of dynamical systems and numerical ODEs can be applied to analyze stability, convergence rates, and behavior of optimization algorithms. For instance, Lyapunov stability theory for ODEs can be adapted to prove convergence for optimization algorithms.
4.  **Hyperparameter Understanding:** The relationship between ODE parameters (like mass $$m$$, friction $$\gamma$$, decay rates for accumulators, discretization step $$h$$) and algorithm hyperparameters ($$\eta, \beta, \beta_1, \beta_2$$) can guide tuning. For example, the condition for critical damping in the ODE can inform choices for $$\beta$$ relative to $$\eta$$.

## Conclusion

Momentum, a cornerstone of modern optimization, is more than just adding a fraction of the previous update. By viewing it through the lens of Ordinary Differential Equations, we've seen Polyak's Heavy Ball method emerge from two distinct but related paths:
-   As a discretization of a **second-order ODE** (via a system of first-order ODEs) describing a physical "heavy ball" system with inertia and friction.
-   As a **Linear Multi-step Method** applied to the fundamental first-order gradient flow ODE.

Furthermore, this ODE perspective extends to adaptive methods like RMSProp and Adam, revealing them as systems of coupled first-order ODEs that describe preconditioned gradient flows, often with inertial components. These viewpoints not only provide a solid theoretical grounding for these advanced optimization techniques but also open avenues for analyzing their behavior and designing new, more effective optimization algorithms. The continuous-time viewpoint reminds us that many discrete algorithms are, at their heart, approximations of underlying continuous dynamical processes.

## Summary of Key Methods and ODEs

| Method                        | Update Rule (one common form, simplified)                                                                                                                                                             | Underlying ODE (Conceptual or Direct, simplified)                                                                                                              | Key Idea                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Gradient Descent (GD)**     | $$x_{k+1} = x_k - \eta \nabla f(x_k)$$                                                                                                                                                                | $$\dot{x}(t) = -\nabla f(x(t))$$ (Gradient Flow)                                                                                                               | Step along negative gradient                |
| **Polyak's Heavy Ball (PHB)** | $$x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})$$                                                                                                                                        | $$m \ddot{x} + \gamma \dot{x} + \nabla f(x) = 0$$                                                                                                              | Inertia + friction, LMM for Gradient Flow   |
| **Nesterov's Accel. (NAG)**   | $$v_{k+1} = \beta v_k - \eta \nabla f(x_k + \beta v_k)$$ <br> $$x_{k+1} = x_k + v_{k+1}$$                                                                                                             | $$\ddot{x}(t) + \frac{k_d}{t} \dot{x}(t) + \nabla f(x(t)) = 0$$ (Su, Boyd, Candès)                                                                             | "Look-ahead" gradient, time-varying damping |
| **RMSProp**                   | $$s_{k+1} = \beta_2 s_k + (1-\beta_2) (\nabla f_k)^2$$ <br> $$x_{k+1} = x_k - \frac{\eta}{\sqrt{s_{k+1}}} \nabla f_k$$                                                                                | $$\dot{x} = -\frac{\eta_0}{\sqrt{s}} \nabla f(x)$$ <br> $$\dot{s} = (1-\beta_2) ((\nabla f(x))^2 - s)$$                                                        | Adaptive per-parameter learning rates       |
| **Adam**                      | $$m_{k+1}\!=\!\beta_1 m_k \!+\! (1\!-\!\beta_1) \nabla f_k$$ <br> $$s_{k+1}\!=\!\beta_2 s_k \!+\! (1\!-\!\beta_2) (\nabla f_k)^2$$ <br> $$x_{k+1}\!=\!x_k \!-\! \eta \frac{m_{k+1}}{\sqrt{s_{k+1}}}$$ | $$\dot{x}\!=\!-\eta_0 \frac{m}{\sqrt{s}}$$ <br> $$\dot{m}\!=\!(1\!-\!\beta_1)(\nabla f(x)\!-\!m)$$ <br> $$\dot{s}\!=\!(1\!-\!\beta_2)((\nabla f(x))^2\!-\!s)$$ | Momentum + adaptive learning rates          |

_Note: $$\nabla f_k = \nabla f(x_k)$$. Vector operations like square, square root, and division are typically element-wise in the adaptive methods._

## Reflection

This exploration of momentum and adaptive optimization methods through ODEs highlights a recurring theme in mathematical optimization for machine learning: many successful discrete algorithms are shadows of underlying continuous processes. The heavy ball analogy gives an intuitive grasp for classical momentum, while the LMM perspective places it firmly within the established field of numerical ODE solvers. Extending this view to adaptive methods like RMSProp and Adam shows them as more complex dynamical systems, yet still amenable to interpretation as preconditioned flows. All these viewpoints enrich our understanding beyond mere algorithmic steps, offering insights into why these methods work and how they might be improved or generalized. This connection between discrete iteration and continuous flow is a powerful paradigm for both analysis and invention in optimization.

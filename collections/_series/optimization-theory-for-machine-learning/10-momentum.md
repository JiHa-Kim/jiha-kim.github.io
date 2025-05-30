---
title: "Momentum: A Tale of Two ODEs and a Multi-Step Method" # Or something similar
date: 2025-05-18 02:57 -0400
series_index: 10 # Based on your series outline
mermaid: true
description: "Delving into the momentum optimization method, understanding its origins from both second-order 'heavy ball' dynamics and as a linear multi-step method for first-order gradient flow."
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
1.  As a direct discretization of a **second-order Ordinary Differential Equation (ODE)** representing a physical system (the "heavy ball" with friction).
2.  As a **Linear Multi-step Method (LMM)** applied to the first-order gradient flow ODE.

These perspectives reveal that momentum is not just an ad-hoc trick but a principled approach rooted in the mathematics of dynamical systems and numerical analysis.

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

To derive an optimization algorithm, we discretize this ODE in time with a step size $$h > 0$$. Let $$x_k \approx x(kh)$$. We can approximate the derivatives:
-   Acceleration: $$\ddot{x}(t_k) \approx \frac{x_{k+1} - 2x_k + x_{k-1}}{h^2}$$ (central difference)
-   Velocity: $$\dot{x}(t_k) \approx \frac{x_k - x_{k-1}}{h}$$ (backward difference, for the damping term at time $$t_k$$)

Substituting these into the ODE at time $$t_k$$:

$$
m \frac{x_{k+1} - 2x_k + x_{k-1}}{h^2} + \gamma \frac{x_k - x_{k-1}}{h} + \nabla f(x_k) = 0
$$

Now, we rearrange to solve for $$x_{k+1}$$:

$$
m(x_{k+1} - 2x_k + x_{k-1}) + \gamma h (x_k - x_{k-1}) + h^2 \nabla f(x_k) = 0
$$

$$
x_{k+1} - 2x_k + x_{k-1} + \frac{\gamma h}{m} (x_k - x_{k-1}) + \frac{h^2}{m} \nabla f(x_k) = 0
$$

$$
x_{k+1} = (2x_k - x_{k-1}) - \frac{\gamma h}{m} (x_k - x_{k-1}) - \frac{h^2}{m} \nabla f(x_k)
$$

$$
x_{k+1} = x_k + (x_k - x_{k-1}) - \frac{\gamma h}{m} (x_k - x_{k-1}) - \frac{h^2}{m} \nabla f(x_k)
$$

$$
x_{k+1} = x_k + \left(1 - \frac{\gamma h}{m}\right)(x_k - x_{k-1}) - \frac{h^2}{m} \nabla f(x_k)
$$

This is precisely the update rule for **Polyak's Heavy Ball (PHB) method**:

$$
x_{k+1} = x_k - \eta_{PHB} \nabla f(x_k) + \beta_{PHB} (x_k - x_{k-1})
$$

where the learning rate $$\eta_{PHB} = \frac{h^2}{m}$$ and the momentum parameter $$\beta_{PHB} = 1 - \frac{\gamma h}{m}$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Polyak's Heavy Ball (PHB) Method**
</div>
The update rule for Polyak's Heavy Ball method is:

$$
x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})
$$

Alternatively, using an explicit velocity term $$v_k$$:

$$
v_{k+1} = \beta v_k - \eta \nabla f(x_k)
$$

$$
x_{k+1} = x_k + v_{k+1}
$$

These are equivalent if we identify $$v_k$$ with the change $$(x_k - x_{k-1})$$ from the previous step when substituting into the first form (or more carefully, if $$v_0$$ is initialized, then $$v_k$$ is the accumulated momentum). For $$k \ge 1$$, if we define $$v_k = (x_k - x_{k-1})$$ in the first expression of the two-variable form for the recurrence, this leads to $$x_{k+1} = x_k + \beta(x_k - x_{k-1}) - \eta \nabla f(x_k)$$.
The parameters are the learning rate $$\eta > 0$$ and the momentum coefficient $$\beta \in [0, 1)$$.
</blockquote>

This derivation shows that the momentum term $$\beta (x_k - x_{k-1})$$ arises naturally from the inertia ($$m$$) and damping ($$\gamma$$) of the physical system.
-   A larger mass $$m$$ (relative to $$h^2$$) implies a smaller learning rate $$\eta_{PHB}$$ but also affects $$\beta_{PHB}$$.
-   A larger friction $$\gamma$$ (relative to $$m/h$$) implies a smaller momentum parameter $$\beta_{PHB}$$. If $$\gamma h / m = 1$$, then $$\beta_{PHB}=0$$, and we recover standard gradient descent (a critically damped system where momentum dies quickly). If $$\gamma = 0$$ (no friction), then $$\beta_{PHB}=1$$.

## Perspective 2: Momentum as a Linear Multi-step Method

Another powerful way to understand momentum is by viewing it as a **Linear Multi-step Method (LMM)** for solving an ODE. LMMs are a class of numerical methods that use information from previous time steps to approximate the solution at the current step.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Linear Multi-step Method (LMM)**
</div>
For an initial value problem $$\dot{y}(t) = F(t, y(t))$$ with $$y(t_0) = y_0$$, an $$s$$-step LMM is defined by:

$$
\sum_{j=0}^s \alpha_j y_{n+j} = h \sum_{j=0}^s \beta_j F(t_{n+j}, y_{n+j})
$$

where $$h$$ is the step size, $$\alpha_j$$ and $$\beta_j$$ are method-specific coefficients, and by convention $$\alpha_s=1$$.
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

This equation perfectly matches the LMM form. It's a 2-step method ($$s=2$$). Let $$n+j$$ map to our indices such that $$n+s = k+1$$. So, $$n = k-1$$.
-   $$y_{n+2} = x_{k+1}$$
-   $$y_{n+1} = x_k$$
-   $$y_{n} = x_{k-1}$$

The coefficients are:
-   $$\alpha_2 = 1$$
-   $$\alpha_1 = -(1+\beta)$$
-   $$\alpha_0 = \beta$$

For the right-hand side, since $$\nabla f(x_k)$$ is evaluated at $$x_k \equiv y_{n+1}$$, this is an explicit method. Assuming the LMM step size $$h_{LMM}$$ is absorbed into the learning rate $$\eta$$ (or set to 1 for simplicity in comparing forms), we can define the $$\beta_j$$ coefficients:
-   $$h_{LMM} \sum_{j=0}^s \beta_j F(t_{n+j}, y_{n+j}) = h_{LMM} \beta_1 F(t_{n+1}, y_{n+1})$$ (since only $$F(x_k)$$ appears)
-   So, $$h_{LMM} \beta_1 (-\nabla f(x_k)) = -\eta \nabla f(x_k)$$.
-   This means we can set (for example, if $$h_{LMM}=1$$):
    -   $$\beta_2 = 0$$
    -   $$\beta_1 = \eta$$
    -   $$\beta_0 = 0$$

Thus, Polyak's Heavy Ball method *is* a specific explicit 2-step linear multi-step method applied to the gradient flow ODE $$\dot{x} = -\nabla f(x)$$. The "momentum" term $$(x_k - x_{k-1})$$ is how this LMM incorporates past information ($$x_{k-1}$$) to determine the next step $$x_{k+1}$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Characteristic Polynomials and Consistency**
</summary>
For an LMM, we define two characteristic polynomials:
- $$\rho(z) = \sum_{j=0}^s \alpha_j z^j$$
- $$\sigma(z) = \sum_{j=0}^s \beta_j z^j$$

For Polyak's method, with $$h_{LMM}=1$$ and parameters as above:
- $$\rho(z) = z^2 - (1+\beta)z + \beta$$
- $$\sigma(z) = \eta z$$

An LMM is consistent if $$\rho(1)=0$$ and $$\rho'(1)=\sigma(1)$$.
- $$\rho(1) = 1 - (1+\beta) + \beta = 0$$. (Satisfied)
- $$\rho'(z) = 2z - (1+\beta)$$, so $$\rho'(1) = 2 - (1+\beta) = 1-\beta$$.
- $$\sigma(1) = \eta$$.
For consistency, we would need $$1-\beta = \eta$$. This is a specific relationship between the momentum parameter and the learning rate, often related to critical damping or specific convergence rate conditions. However, PHB is used with more general choices of $$\eta$$ and $$\beta$$. The LMM formulation describes its algebraic structure regardless of whether this specific consistency condition for approximating $$\dot{x}=-\nabla f(x)$$ with first-order accuracy is met. The method itself *is* an LMM by its form.
</details>

## Connecting the Two Perspectives

The beauty is that these two perspectives are deeply connected.
1.  We started with a physical system (heavy ball with friction) described by a **second-order ODE**.
2.  Discretizing this ODE led directly to the Polyak's Heavy Ball update rule.
3.  This update rule, which involves $$x_{k+1}, x_k, x_{k-1}$$, has the algebraic structure of a **2-step Linear Multi-step Method**.
4.  This LMM can be seen as a way to solve the simpler **first-order gradient flow ODE** $$\dot{x} = -\nabla f(x)$$, but using information from multiple past steps ($$x_k, x_{k-1}$$) instead of just one ($$x_k$$) as in Euler's method (standard GD).

So, the "inertia" from the physical model translates into the "memory" of the LMM. The momentum term, which helps the optimization process navigate complex landscapes, is essentially how the LMM leverages past iterates to make a more informed step for the underlying gradient flow dynamics.

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

The key difference from PHB is that the gradient is evaluated at a "look-ahead" point $$x_k + \beta v_k$$, rather than at $$x_k$$.
</blockquote>

Interestingly, NAG also has a continuous-time ODE interpretation. Su, Boyd, and Candès (2014) showed that a version of NAG can be seen as a discretization of the following second-order ODE:

$$
\ddot{x}(t) + \frac{k}{t} \dot{x}(t) + \nabla f(x(t)) = 0
$$

For a typical choice, $$k=3$$. Notice the time-dependent damping term $$\frac{k}{t}$$. As time $$t$$ increases, the damping decreases. This "adaptive" damping is thought to be one reason for NAG's superior performance in some settings. Deriving NAG from this ODE involves a more complex discretization scheme than the one used for PHB.

While NAG can also be written in a multi-step form, its "look-ahead" gradient makes its LMM interpretation for the simple gradient flow ODE less direct than for PHB.

## Why is the ODE Perspective So Valuable?

Understanding optimization algorithms through the lens of ODEs offers several benefits:
1.  **Deeper Intuition:** It moves beyond algorithmic recipes to physical or mathematical analogies, explaining *why* methods like momentum work.
2.  **Principled Design:** New algorithms can be designed by proposing different ODEs (e.g., with different damping or inertial terms) and then discretizing them.
3.  **Analysis Tools:** The rich theory of dynamical systems and numerical ODEs can be applied to analyze stability, convergence rates, and behavior of optimization algorithms. For instance, Lyapunov stability theory for ODEs can be adapted to prove convergence for optimization algorithms.
4.  **Hyperparameter Understanding:** The relationship between ODE parameters (like mass $$m$$, friction $$\gamma$$, discretization step $$h$$) and algorithm hyperparameters ($$\eta, \beta$$) can guide tuning. For example, the condition for critical damping in the ODE can inform choices for $$\beta$$ relative to $$\eta$$.

## Conclusion

Momentum, a cornerstone of modern optimization, is more than just adding a fraction of the previous update. By viewing it through the lens of Ordinary Differential Equations, we've seen Polyak's Heavy Ball method emerge from two distinct but related paths:
-   As a discretization of a **second-order ODE** describing a physical "heavy ball" system with inertia and friction.
-   As a **Linear Multi-step Method** applied to the fundamental first-order gradient flow ODE.

These perspectives not only provide a solid theoretical grounding for momentum but also open avenues for analyzing its behavior and designing new, more effective optimization algorithms. The continuous-time viewpoint reminds us that many discrete algorithms are, at their heart, approximations of underlying continuous dynamical processes.

## Summary of Key Methods and ODEs

| Method                        | Update Rule (one common form)                                                             | Underlying ODE (Conceptual or Direct)                                            | Key Idea                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| **Gradient Descent (GD)**     | $$x_{k+1} = x_k - \eta \nabla f(x_k)$$                                                    | $$\dot{x}(t) = -\nabla f(x(t))$$ (Gradient Flow)                                 | Step along negative gradient                |
| **Polyak's Heavy Ball (PHB)** | $$x_{k+1} = x_k - \eta \nabla f(x_k) + \beta (x_k - x_{k-1})$$                            | $$m \ddot{x} + \gamma \dot{x} + \nabla f(x) = 0$$                                | Inertia + friction, LMM for Gradient Flow   |
| **Nesterov's Accel. (NAG)**   | $$v_{k+1} = \beta v_k - \eta \nabla f(x_k + \beta v_k)$$ <br> $$x_{k+1} = x_k + v_{k+1}$$ | $$\ddot{x}(t) + \frac{k}{t} \dot{x}(t) + \nabla f(x(t)) = 0$$ (Su, Boyd, Candès) | "Look-ahead" gradient, time-varying damping |

## Reflection

This exploration of momentum through ODEs highlights a recurring theme in mathematical optimization for machine learning: many successful discrete algorithms are shadows of underlying continuous processes. The heavy ball analogy gives an intuitive grasp, while the LMM perspective places momentum firmly within the established field of numerical ODE solvers. Both viewpoints enrich our understanding beyond mere algorithmic steps, offering insights into why momentum works and how it might be improved or generalized. This connection between discrete iteration and continuous flow is a powerful paradigm for both analysis and invention in optimization.

---
title: "Crash Course: Numerical Methods for ODEs in Optimization"
date: 2025-09-30
sort_index: 1
mermaid: true
description: "A primer on numerical methods for solving Ordinary Differential Equations, tailored for understanding optimization algorithms like Gradient Descent, Momentum, and their continuous-time limits."
image:
categories:
- Numerical Analysis
- Mathematical Optimization
tags:
- Ordinary Differential Equations
- Discretization
- Euler Method
- Runge–Kutta
- Linear Multistep
- Stability
- Stiffness
- Symplectic
- SDEs
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line: $$inline$$

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

## Part I — Foundations: Accuracy, Stability, and Core Integrators

This post provides a primer on the foundational concepts of numerical integration for ordinary differential equations (ODEs). We'll explore the core ideas of accuracy and stability and introduce the workhorse integrator families: θ-methods, Runge-Kutta methods, and linear multistep methods. The goal is to build intuition for how these methods connect to and underpin common optimization algorithms.

### 1. Motivation & ODE Models of Optimization

Many optimization algorithms can be viewed as discretizations of continuous-time dynamical systems described by ODEs. This perspective is powerful; it allows us to analyze algorithm behavior using the well-developed theory of numerical integration.

Two fundamental ODE models in optimization are:

*   **Gradient Flow:** The simplest model describes the path of steepest descent on a loss surface $$L(\theta)$$. The trajectory follows the negative gradient, formally written as:

    $$
    \dot\theta(t) = -\nabla L(\theta(t))
    $$

    Here, $$\dot\theta$$ represents the time derivative of the parameters $$\theta$$. The solutions to this ODE trace paths that continuously minimize the loss.

*   **Heavy-Ball Momentum:** This model introduces a second-order momentum term, analogous to a physical object with mass $$m$$ moving through a viscous medium with friction coefficient $$\gamma$$:

    $$
    m\ddot\theta(t) + \gamma\dot\theta(t) + \nabla L(\theta(t)) = 0
    $$

    This system often converges faster than pure gradient flow by allowing the trajectory to build momentum and overshoot local minima. Stochastic extensions of these models, where the gradient is noisy, form the basis for algorithms like Stochastic Gradient Descent (SGD) and Momentum SGD.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**A Note on Existence and Uniqueness**
</div>
For these ODEs to be useful models, we need assurance that a solution exists and is unique for a given starting point. The **Picard–Lindelöf theorem** guarantees this, provided the function defining the dynamics (e.g., $$-\nabla L(\theta)$$) is continuous and satisfies a Lipschitz condition. This means the function's rate of change is bounded, preventing the solution from "blowing up" in finite time.
</blockquote>

### 2. Consistency, Stability, and Convergence

The quality of a numerical integrator is determined by the "triangle" of consistency, stability, and convergence. A method is **convergent** if its solution approaches the true ODE solution as the step size shrinks. The Lax equivalence theorem states that for a consistent method, stability is equivalent to convergence.

*   **Consistency** measures how well the numerical scheme approximates the ODE at a single step. The **local truncation error (LTE)** is the error introduced in one step, assuming the previous point was on the exact solution curve. A method has **order $$p$$** if its LTE is $$O(h^{p+1})$$, where $$h$$ is the step size. The **global error** is the cumulative error after many steps, which for a stable method of order $$p$$ is typically $$O(h^p)$$.

*   **Stability** concerns the propagation and accumulation of errors. For linear multistep methods (LMMs), the key concept is **zero-stability**. An LMM is zero-stable if its solutions remain bounded when applied to the simple ODE $$\dot{y} = 0$$. This property is governed by the roots of the method's characteristic polynomial.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Root Condition for Zero-Stability**
</div>
A linear multistep method is zero-stable if and only if all roots of its first characteristic polynomial $$\rho(\zeta)$$ lie within or on the unit circle in the complex plane, and any roots on the unit circle are simple.
</blockquote>

### 3. Absolute Stability & Stiffness

While zero-stability addresses error accumulation for $$\dot{y}=0$$, we often need to understand stability for more general dynamics. This leads to the concept of **absolute stability**.

We analyze stability using the Dahlquist test equation:

$$
\dot{y} = \lambda y, \quad \text{Re}(\lambda) < 0
$$

Applying a numerical method to this equation with step size $$h$$ yields a recurrence of the form $$y_{n+1} = R(z)y_n$$, where $$z = h\lambda$$. The function $$R(z)$$ is the method's **stability function**. The **region of absolute stability** is the set of all $$z \in \mathbb{C}$$ for which $$\vert R(z) \vert \le 1$$.

*   **A-stability:** A method is A-stable if its stability region contains the entire left half-plane ($$\text{Re}(z) \le 0$$). This is a highly desirable property for problems with dynamics that decay over time.
*   **L-stability:** A method is L-stable if it is A-stable and additionally $$\lim_{\text{Re}(z) \to -\infty} \vert R(z) \vert = 0$$. This ensures that highly stable components of the system are damped out quickly in the numerical solution.
*   **Stiffness:** A problem is considered stiff when the step size required for stability is much smaller than the step size needed for accuracy. A-stable and L-stable methods are essential for solving stiff ODEs efficiently.

For a Runge-Kutta method with Butcher tableau given by coefficients $$(A, b, c)$$, the stability function is:

$$
R(z) = 1 + z b^\top(I - zA)^{-1}e
$$

where $$e$$ is the vector of ones. A crucial result is that **no explicit Runge-Kutta method can be A-stable**, because for an explicit method, $$R(z)$$ is a polynomial, which is unbounded as $$\text{Re}(z) \to -\infty$$.

### 4. The θ-methods: Unifying Euler and Trapezoid

The θ-methods are a simple family of one-step integrators that provide a bridge between explicit and implicit schemes. The update rule is given by:

$$
y_{n+1} = y_n + h \left[ (1-\theta)f(y_n) + \theta f(y_{n+1}) \right]
$$

This family includes three famous methods:

1.  **$$\theta=0$$ (Explicit Euler):** The update is $$y_{n+1} = y_n + h f(y_n)$$. This is the simplest explicit method, with order 1. Its stability function is $$R(z) = 1+z$$.
2.  **$$\theta=1$$ (Implicit Euler):** The update is $$y_{n+1} = y_n + h f(y_{n+1})$$. This method is implicit, requiring a solve for $$y_{n+1}$$ at each step. It is of order 1, but it is both A-stable and L-stable, with stability function $$R(z) = (1-z)^{-1}$$.
3.  **$$\theta=1/2$$ (Trapezoidal Rule):** This is an implicit method of order 2. It is A-stable but not L-stable, as $$\vert R(z) \vert \to 1$$ for $$\text{Re}(z) \to -\infty$$. Its stability function is $$R(z) = \frac{1+z/2}{1-z/2}$$.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Connection to Optimization**
</div>
The explicit Euler method is directly equivalent to the Gradient Descent algorithm. Applying explicit Euler to the gradient flow ODE $$\dot\theta = -\nabla L(\theta)$$ with a step size (learning rate) $$\alpha$$ gives the familiar update:

$$
\theta_{n+1} = \theta_n - \alpha \nabla L(\theta_n)
$$

The stability analysis of explicit Euler on a quadratic model $$L(\theta) = \frac{1}{2}L\theta^2$$ gives the stability bound $$0 < \alpha < 2/L$$, a classic result in optimization.
</blockquote>

### 5. Runge-Kutta (RK) Methods

Runge-Kutta methods are a large and versatile family of one-step methods that achieve higher orders of accuracy by evaluating the function $$f$$ at intermediate points within a step.

An $$s$$-stage RK method is defined by its **Butcher tableau**:

$$
\begin{array}{c|c}
c & A \\
\hline
& b^\top
\end{array}
$$

The order conditions are equations in terms of $$A, b, c$$ that must be satisfied. For example:
*   Order 1: $$\sum b_i = 1$$
*   Order 2: $$\sum b_i c_i = 1/2$$

Well-known schemes include Heun's method (an order 2 method) and the classic RK4 method.

For practical use, **embedded pairs** are invaluable for adaptive step-sizing. These pairs use a single set of function evaluations to compute two solutions of different orders (e.g., order 5 and order 4). The difference between the two solutions provides an error estimate. Popular pairs include:
*   **Dormand-Prince 5(4):** This is the method behind MATLAB's famous `ode45` solver. It is designed to minimize the error of the higher-order (5th order) solution.
*   **Tsitouras 5(4):** A more modern and highly efficient 5(4) pair.

Many efficient pairs feature the **First Same As Last (FSAL)** property. This means the last stage evaluation of one step can be reused as the first stage of the next, saving one function evaluation per successful step.

Step-size controllers, often based on **PI or PID control theory**, use the stream of local error estimates to adjust the step size $$h$$ smoothly, aiming to keep the error below a specified relative tolerance (`rtol`) and absolute tolerance (`atol`).

### 6. Linear Multistep Methods (LMMs)

Unlike one-step methods, a $$k$$-step LMM uses information from the previous $$k$$ steps to compute the next point:

$$
\sum_{j=0}^{k} \alpha_j y_{n+j} = h \sum_{j=0}^{k} \beta_j f_{n+j}
$$

The method is explicit if $$\beta_k=0$$ and implicit otherwise. Two major families of LMMs are:

*   **Adams-Bashforth (explicit) and Adams-Moulton (implicit) methods:** These are widely used for non-stiff problems. They are often combined in predictor-corrector pairs (e.g., an AB4 predictor followed by an AM3 corrector), which provides an error estimate and improves stability. MATLAB's `ode113` is a variable-step, variable-order (VSVO) solver based on the Adams-Bashforth-Moulton family.
*   **Backward Differentiation Formulas (BDFs):** These are the workhorses for stiff problems. They are implicit and have excellent stability properties for higher orders. However, the **second Dahlquist barrier** proves a fundamental limitation: an A-stable LMM cannot have an order greater than 2. BDF1 (Implicit Euler) and BDF2 are A-stable, while higher-order BDFs are not but still have large stability regions suitable for many stiff problems.

### 7. Worked Connections to Optimization

Let's solidify the link between numerical integration and optimization.

*   **GD and Euler Stability:** As mentioned, Gradient Descent is explicit Euler on gradient flow. The stability condition for explicit Euler applied to $$\dot{y} = \lambda y$$ is $$\vert 1 + h\lambda \vert \le 1$$. For a quadratic loss with Hessian $$H$$, the eigenvalues of $$H$$ correspond to $$-\lambda$$. This gives stability limits on the learning rate $$\alpha$$ based on the largest eigenvalue (Lipschitz constant) of the Hessian.
*   **Heavy-Ball Momentum:** The heavy-ball ODE can be written as a first-order system by defining $$v = \dot\theta$$:

    $$
    \begin{cases}
    \dot\theta = v \\
    \dot{v} = -\frac{\gamma}{m} v - \frac{1}{m} \nabla L(\theta)
    \end{cases}
    $$

    Discretizing this system using a semi-implicit Euler method (explicit for $$\theta$$, implicit for $$v$$) leads directly to the Polyak momentum update equations for optimization.
*   **Momentum as an LMM:** The standard momentum update can also be viewed as a 2-step explicit LMM. The zero-stability condition (the root condition) on this LMM imposes constraints on the momentum coefficient $$\mu$$ to ensure the optimization process doesn't diverge.

---
<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**What You Should Remember**
</div>
*   **Order vs. Function Evaluations:** Higher-order methods are more accurate for small step sizes but typically require more function evaluations per step.
*   **Stability Regions:** The size and shape of the stability region determine a method's suitability for different types of problems, especially stiff ones. A-stability is a key property for stiff solvers.
*   **Adaptivity:** For most problems, adaptive step-sizing using embedded pairs (like Dormand-Prince) is far more efficient than using a fixed step size.
*   **Euler ↔ GD:** The simplest numerical method, explicit Euler, is equivalent to the fundamental optimization algorithm, Gradient Descent. This connection provides a bridge for analyzing optimization methods using ODE theory.
</blockquote>

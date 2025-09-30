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
- Runge-Kutta
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

This post provides a primer on the foundational concepts of numerical integration for ordinary differential equations (ODEs). We'll explore the core ideas of accuracy and stability and introduce the workhorse integrator families: θ-methods, Runge-Kutta methods, and linear multistep methods. The goal is to build intuition for how these methods connect to and underpin common optimization algorithms. For simplicity, we often write $$f(y)$$ for autonomous ODEs where the dynamics do not explicitly depend on time $$t$$.

### 1. Motivation & ODE Models of Optimization

Many optimization algorithms can be viewed as discretizations of continuous-time dynamical systems described by ODEs. This perspective is powerful; it allows us to analyze algorithm behavior using the well-developed theory of numerical integration.

Two fundamental ODE models in optimization are:

*   **Gradient Flow:** The simplest model describes the path of steepest descent on a loss surface $$L(\theta)$$. The trajectory follows the negative gradient, formally written as:

    $$
    \dot\theta(t) = -\nabla L(\theta(t))
    $$

    Here, $$\dot\theta$$ represents the time derivative of the parameters $$\theta$$. For gradient flow, $$ \tfrac{d}{dt}L(\theta(t))=\nabla L(\theta(t))^\top \dot\theta(t)=-\Vert\nabla L(\theta(t))\Vert^2\le 0$$, so the loss is non-increasing along trajectories.

*   **Heavy-Ball Momentum:** This model introduces a second-order momentum term, analogous to a physical object with mass $$m$$ moving through a viscous medium with friction coefficient $$\gamma$$:

    $$
    m\ddot\theta(t) + \gamma\dot\theta(t) + \nabla L(\theta(t)) = 0
    $$

    This system often converges faster than pure gradient flow by incorporating inertia, which accelerates convergence, particularly on strongly convex problems, when appropriately damped.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**A Note on Existence and Uniqueness**
</div>
For these ODEs to be useful models, we need assurance that a solution exists and is unique for a given starting point. The **Picard–Lindelöf theorem** guarantees this, provided the function defining the dynamics (e.g., $$-\nabla L(\theta)$$) is continuous and satisfies a Lipschitz condition. This means the function's rate of change is bounded, preventing the solution from "blowing up" in finite time.
</blockquote>

### 2. Consistency, Stability, and Convergence

A method is **consistent** if its one-step local truncation error (LTE) tends to zero as $$h\to 0$$. If the LTE is $$O(h^{p+1})$$, the method has **order $$p$$** and, under stability, its **global error** is $$O(h^{p})$$. For **linear multistep methods (LMMs)**, the **Dahlquist equivalence theorem** states: *consistency + zero-stability ⇒ convergence*. Here **zero-stability** means small perturbations in starting values do not blow up as $$h\to 0$$, and it is characterized by the root condition on the characteristic polynomial (see the box below). For one-step methods (e.g., Runge-Kutta), zero-stability is automatic; stability is analyzed via the stability function on the test equation.

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

Applying a numerical method to this equation with step size $$h$$ yields a recurrence of the form $$y_{n+1} = R(z)y_n$$, where $$z = h\lambda$$. The function $$R(z)$$ is the method's **stability function**. The **region of absolute stability** is the set of all $$z \in \mathbb{C}$$ for which $$\vert R(z) \vert \le 1$$. For asymptotic decay of the numerical solution of $$\dot y=\lambda y$$, one needs $$\vert R(z)\vert<1$$. On the boundary $$\vert R(z)\vert=1$$ the method may be neutrally stable.

*   **A-stability:** A method is A-stable if its stability region contains the entire left half-plane ($$\text{Re}(z) \le 0$$).
*   **L-stability:** A method is L-stable if it is A-stable and additionally $$\lim_{\text{Re}(z) \to -\infty} \vert R(z) \vert = 0$$.
*   **Stiffness:** A problem is considered stiff when the step size required for stability is much smaller than the step size needed for accuracy.

For a Runge-Kutta method with Butcher tableau given by coefficients $$(A, b, c)$$, the stability function is:

$$
R(z) = 1 + z b^\top(I - zA)^{-1}e
$$

where $$e$$ is the vector of ones. A crucial result is that **no explicit Runge-Kutta method can be A-stable**.

### 4. The θ-methods: Unifying Euler and Trapezoid

The θ-methods are a simple family of one-step integrators. The update rule is given by:

$$
y_{n+1}=y_n+h\big[(1-\theta)f(t_n,y_n)+\theta f(t_{n+1},y_{n+1})\big].
$$

Its stability function on $$\dot y=\lambda y$$ is:

$$
R(z)=\frac{1+(1-\theta)z}{1-\theta z},\qquad z=h\lambda.
$$

This family includes three famous methods:

1.  **$$\theta=0$$ (Explicit Euler):** Order 1. Not A-stable.
2.  **$$\theta=1$$ (Implicit Euler):** Order 1. A-stable and L-stable.
3.  **$$\theta=1/2$$ (Trapezoidal Rule):** Order 2. A-stable but not L-stable.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Connection to Optimization**
</div>
The explicit Euler method is directly equivalent to the Gradient Descent algorithm. Applying explicit Euler to the gradient flow ODE $$\dot\theta = -\nabla L(\theta)$$ with a step size (learning rate) $$\alpha$$ gives the familiar update:

$$
\theta_{n+1} = \theta_n - \alpha \nabla L(\theta_n)
$$

For a quadratic loss $$L(\theta)=\tfrac12\theta^\top H\theta$$ with eigenvalues $$0<\lambda_i\le \lambda_{\max}(H)=L$$, explicit Euler (= GD with step $$\alpha$$) evolves mode-wise as $$\theta^{(i)}_{n+1}=(1-\alpha\lambda_i)\theta^{(i)}_n$$. Stability requires $$\vert 1-\alpha\lambda_i\vert<1\ \forall i$$, i.e.

$$
0<\alpha<\frac{2}{L}.
$$

</blockquote>

### 5. Runge-Kutta (RK) Methods

Runge-Kutta methods are one-step methods that achieve higher orders of accuracy by evaluating $$f$$ at intermediate points within a step. An $$s$$-stage RK method is defined by its **Butcher tableau**:

$$
\begin{array}{c|c}
c & A \\
\hline
& b^\top
\end{array}
$$

(Reminder: $$c_i=\sum_j a_{ij}$$ for explicit Runge-Kutta.)

The order conditions are equations in terms of $$A, b, c$$ that must be satisfied.
*   Order 1: $$\sum_i b_i=1$$
*   Order 2: $$\sum_i b_i c_i=\tfrac12$$
*   Order 3: $$\sum_i b_i c_i^2=\tfrac13,\quad \sum_{i,j} b_i a_{ij} c_j=\tfrac16$$

**Embedded pairs & FSAL.** A 5(4) pair computes $$y^{(5)}$$ and $$y^{(4)}$$ with shared stages; their difference estimates error for adaptivity. **Dormand–Prince 5(4)** (the classic `ode45` choice) uses 7 stages but has **FSAL**, so successful steps reuse the last stage as the next step’s first—**about 6 function evaluations per accepted step**. **Tsitouras 5(4)** is a modern pair with similar accuracy and excellent efficiency in practice.

Step-size controllers, often based on **PI or PID control theory**, use the stream of local error estimates to adjust the step size $$h$$ smoothly, aiming to keep the error below specified tolerances (`rtol`/`atol`).

### 6. Linear Multistep Methods (LMMs)

Unlike one-step methods, a $$k$$-step LMM uses information from the previous $$k$$ steps to compute the next point:

$$
\sum_{j=0}^{k} \alpha_j y_{n+j} = h \sum_{j=0}^{k} \beta_j f_{n+j}
$$

*   **Adams-Bashforth (explicit) and Adams-Moulton (implicit) methods:** These are often combined in predictor-corrector pairs. MATLAB's `ode113` is a variable-step, variable-order (VSVO) solver based on this family.

    > **Why “1 fev/step after startup”?** The AB predictor uses only stored $$f$$-values; a single fresh $$f(t_{n+1},\tilde y_{n+1})$$ powers the AM corrector and the step’s error estimate.

*   **BDF (backward differentiation formulas)** are implicit, stiff-robust multistep methods. They are **A-stable only for orders $$q\le 2$$** (BDF1 = implicit Euler, BDF2), and **zero-stable only up to $$q\le 6$$**. Orders (3)–(6) are not A-stable but have large left-half-plane sectors (A($$\alpha$$)-stability with large wedges) that are effective on many stiff problems.

### 7. Worked Connections to Optimization

*   **GD and Euler Stability:** As mentioned, Gradient Descent is explicit Euler on gradient flow. Stability limits on the learning rate $$\alpha$$ correspond directly to the stability region of the explicit Euler method.
*   **Heavy-Ball Momentum:** The heavy-ball ODE can be written as a first-order system. Discretizing this system using a semi-implicit Euler method leads directly to the Polyak momentum update.
*   **Momentum as an LMM:** The standard momentum update can also be viewed as a 2-step explicit LMM. The momentum recurrence $$(\theta_{n+1}=(1+\mu)\theta_n-\mu\theta_{n-1}-\alpha\nabla L(\theta_n))$$ has characteristic polynomial $$\zeta^2-(1+\mu)\zeta+\mu$$ with roots $$1$$ and $$\mu$$. Zero-stability requires all roots in the unit disk and simple on the unit circle ⇒ **$$\mu\in[0,1)$$** in standard uses.

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

## References

* E. Hairer, S. P. Nørsett, G. Wanner. *Solving Ordinary Differential Equations I: Nonstiff Problems* (Springer). Authoritative for RK theory, order conditions, and stability; see Chs. II–IV. ([SpringerLink][1])
* E. Hairer, G. Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (Springer). Core reference for stiffness, BDF, Rosenbrock/linearly-implicit methods. ([SpringerLink][2])
* J. C. Butcher. *Numerical Methods for Ordinary Differential Equations* (Wiley). Deep dive on Runge–Kutta order theory and stability. ([Wiley Online Library][3])
* J. R. Dormand, P. J. Prince. “A family of embedded Runge–Kutta formulae.” *J. Comput. Appl. Math.*, 1980. Classic source of the 5(4) pair behind `ode45`; FSAL and stage coefficients. ([ScienceDirect][4])
* C. Tsitouras. “Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption.” *Appl. Numer. Math.*, 2011. Modern efficient 5(4) pairs (Tsit5). ([ScienceDirect][5])
* G. Dahlquist. “A special stability problem for linear multistep methods.” *BIT*, 1963. The classic paper underpinning LMM stability barriers and the equivalence theorem. ([math.unipd.it][6])
* P. E. Kloeden, E. Platen. *Numerical Solution of Stochastic Differential Equations* (Springer). Standard reference for Euler–Maruyama, Milstein, strong/weak orders (you’ll cite this in Part II). ([SpringerLink][7])
* MATLAB `ode45` documentation. Notes that `ode45` uses the Dormand–Prince explicit RK 5(4) pair with adaptive steps. Useful practical cross-reference. ([mathworks.com][8])

[1]: https://link.springer.com/book/10.1007/978-3-540-78862-1 "Solving Ordinary Differential Equations I: Nonstiff Problems"
[2]: https://link.springer.com/book/10.1007/978-3-642-05221-7 "Solving Ordinary Differential Equations II"
[3]: https://onlinelibrary.wiley.com/doi/book/10.1002/9781119121534 "Numerical Methods for Ordinary Differential Equations"
[4]: https://www.sciencedirect.com/science/article/pii/0771050X80900133/pdf?md5=7337cc66c9ff13c6a764d87cd2327ae1&pid=1-s2.0-0771050X80900133-main.pdf&utm_source=chatgpt.com "A family of embedded Runge-Kutta formulae"
[5]: https://www.sciencedirect.com/science/article/pii/S0898122111004706 "Runge–Kutta pairs of order 5(4) satisfying only the first ..."
[6]: https://www.math.unipd.it/~alvise/AN_2017/LETTURE/DAHLQUIST_STAB.pdf "A special stability problem for linear multistep methods"
[7]: https://link.springer.com/book/10.1007/978-3-662-12616-5 "Numerical Solution of Stochastic Differential Equations"
[8]: https://www.mathworks.com/help/matlab/ref/ode45.html "ode45 - Solve nonstiff differential equations"


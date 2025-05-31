---
title: "Crash Course: Numerical Methods for ODEs in Optimization"
date: 2025-05-17 # Or current date
course_index: 1
mermaid: true
description: "A primer on numerical methods for solving Ordinary Differential Equations, tailored for understanding optimization algorithms like Gradient Descent and Momentum."
image:
categories:
- Numerical Analysis
- Mathematical Optimization # Or just Numerical Analysis if preferred for crash courses
tags:
- Ordinary Differential Equations
- Discretization
- Euler Method
- Gradient Flow
- Stability
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

Welcome to this crash course on numerical methods for Ordinary Differential Equations (ODEs)! This material is designed to provide the essential background for understanding how many optimization algorithms, such as Gradient Descent and Momentum, can be interpreted as ways to numerically solve certain ODEs. This perspective offers deep insights into their behavior and properties.

## 1. Introduction to Ordinary Differential Equations (ODEs)

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Ordinary Differential Equation (ODE)
</div>
An **Ordinary Differential Equation (ODE)** is an equation involving an unknown function of a single independent variable (often time, $$t$$) and its derivatives. A first-order ODE typically takes the form:

$$
\frac{dy}{dt} = f(t, y(t))
$$

Here, $$y(t)$$ is the unknown function we want to find, and $$f(t, y(t))$$ is a given function that defines the rate of change of $$y$$.
If $$y$$ is a vector $$\mathbf{y} \in \mathbb{R}^d$$, then we have a system of ODEs.
</blockquote>

To find a unique solution to an ODE, we usually need an **Initial Condition (IC)**, specifying the value of the function at an initial time $$t_0$$. This forms an **Initial Value Problem (IVP)**:

$$
\frac{dy}{dt} = f(t, y(t)), \quad y(t_0) = y_0
$$

**Why are ODEs relevant to optimization?**

Many optimization processes can be described by continuous-time dynamical systems. For example:
1.  **Gradient Flow:** The path of steepest descent on a loss surface $$L(\theta)$$ can be described by the ODE:

    $$
    \frac{d\theta(t)}{dt} = -\nabla L(\theta(t))
    $$

    The solution $$\theta(t)$$ represents a continuous trajectory moving towards a minimum of $$L$$.
2.  **Heavy Ball ODE:** Optimization with momentum can be related to a second-order ODE from physics (a ball rolling with friction):

    $$
    m\ddot{\theta}(t) + \gamma \dot{\theta}(t) + \nabla L(\theta(t)) = 0
    $$

    where $$\ddot{\theta}$$ is the second derivative (acceleration) and $$\dot{\theta}$$ is the first derivative (velocity).

Since analytical solutions to these ODEs are often intractable for complex functions $$L(\theta)$$, we turn to **numerical methods** to approximate their solutions.

## 2. Fundamentals of Discretization

Numerical methods for ODEs work by discretizing time. Instead of finding a continuous solution $$y(t)$$, we approximate the solution at discrete time points: $$t_0, t_1, t_2, \dots, t_N$$, where $$t_{n+1} = t_n + h_n$$. The interval $$h_n$$ (often constant, $$h$$) is called the **time step** or step size. We denote the numerical approximation to $$y(t_n)$$ as $$y_n$$.

The core idea is to approximate derivatives using **finite differences**. From calculus, the definition of a derivative is:

$$
\frac{dy}{dt} = \lim_{h \to 0} \frac{y(t+h) - y(t)}{h}
$$

For a small, finite $$h$$:
*   **Forward Difference:** Approximates the derivative at $$t_n$$ using the value at $$t_{n+1}$$:

    $$
    y'(t_n) \approx \frac{y(t_{n+1}) - y(t_n)}{h} = \frac{y_{n+1} - y_n}{h}
    $$

*   **Backward Difference:** Approximates the derivative at $$t_n$$ using the value at $$t_{n-1}$$:

    $$
    y'(t_n) \approx \frac{y(t_n) - y(t_{n-1})}{h} = \frac{y_n - y_{n-1}}{h}
    $$

*   **Central Difference:** Often more accurate, approximates derivative at $$t_n$$ using $$t_{n+1}$$ and $$t_{n-1}$$:

    $$
    y'(t_n) \approx \frac{y(t_{n+1}) - y(t_{n-1})}{2h} = \frac{y_{n+1} - y_{n-1}}{2h}
    $$

**Errors in Numerical Methods:**
*   **Local Truncation Error (LTE):** The error introduced in a single step of the method, assuming all previous values were exact. It arises from approximating the continuous derivative with a finite difference.
*   **Global Error:** The accumulated error at a certain time $$t_N$$, resulting from the propagation of local errors over many steps.
*   **Order of Accuracy:** A method is said to be of order $$p$$ if its global error is proportional to $$h^p$$ (i.e., Global Error $$\propto h^p$$). A higher order $$p$$ means the error decreases more rapidly as $$h$$ gets smaller.

## 3. One-Step Methods for First-Order ODEs

One-step methods compute $$y_{n+1}$$ using only information from the previous step $$y_n$$.

### 3.1. Euler's Method (Explicit Euler)

The simplest numerical method. To solve $$\frac{dy}{dt} = f(t,y)$$, we replace $$\frac{dy}{dt}$$ at $$t_n$$ with its forward difference approximation:

$$
\frac{y_{n+1} - y_n}{h} \approx f(t_n, y_n)
$$

Rearranging for $$y_{n+1}$$ gives the **Explicit Euler** update rule:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Algorithm.** Explicit Euler Method
</div>
Given an IVP $$\frac{dy}{dt} = f(t,y)$$, $$y(t_0) = y_0$$, and a step size $$h$$:
For $$n = 0, 1, 2, \dots$$:

$$
y_{n+1} = y_n + h f(t_n, y_n)
$$

$$
t_{n+1} = t_n + h
$$

</blockquote>

Geometrically, Explicit Euler takes a step in the direction of the tangent to the solution curve at $$(t_n, y_n)$$. It has a local truncation error of $$O(h^2)$$ and a global error of $$O(h)$$, so it's a first-order method.

<blockquote class="prompt-info" markdown="1">
<div class="title" markdown="1">
**Connection to Gradient Descent.**
</div>
Consider the gradient flow ODE: $$\frac{d\theta(t)}{dt} = -\nabla L(\theta(t))$$.
Applying Explicit Euler with step size $$h$$ (which we can call the learning rate $$\alpha$$):

$$
\theta_{k+1} = \theta_k + h (-\nabla L(\theta_k))
$$

If we let $$h = \alpha$$, this is exactly the **Gradient Descent** update rule:

$$
\theta_{k+1} = \theta_k - \alpha \nabla L(\theta_k)
$$

Thus, Gradient Descent can be viewed as an Explicit Euler discretization of the gradient flow ODE.
</blockquote>

### 3.2. Implicit Euler Method

Instead of evaluating $$f$$ at $$(t_n, y_n)$$, the Implicit Euler method evaluates $$f$$ at the *next* point $$(t_{n+1}, y_{n+1})$$:

$$
\frac{y_{n+1} - y_n}{h} \approx f(t_{n+1}, y_{n+1})
$$

This gives the update rule:

$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
$$

Notice that $$y_{n+1}$$ appears on both sides. If $$f$$ is non-linear, this requires solving an algebraic equation (often iteratively) for $$y_{n+1}$$ at each step, making it computationally more expensive than explicit methods. However, implicit methods often have better **stability properties**, especially for "stiff" ODEs (where solutions change on very different time scales).

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Higher-Order One-Step Methods
</summary>
While Euler's method is simple, its first-order accuracy is often insufficient. **Runge-Kutta methods** are a family of higher-order one-step methods. For example, the classic fourth-order Runge-Kutta method (RK4) uses a weighted average of four evaluations of $$f$$ within each step to achieve a global error of $$O(h^4)$$. These are more complex but can provide much more accurate solutions for a given step size $$h$$.
</details>

## 4. Stability of Numerical Methods

Stability is a crucial concept. Intuitively, a numerical method is stable if small perturbations (like round-off errors or local truncation errors) do not cause the numerical solution to diverge wildly from the true solution.
*   **Explicit methods** (like Explicit Euler) often have **conditional stability**: they are stable only if the step size $$h$$ is sufficiently small (below a certain threshold that depends on the ODE).
*   **Implicit methods** (like Implicit Euler) often have better stability properties, sometimes being **unconditionally stable** (stable for any $$h > 0$$) for certain classes of problems.

<blockquote class="prompt-info" markdown="1">
<div class="title" markdown="1">
**Connection to Learning Rates.**
</div>
The stability condition for Explicit Euler applied to gradient flow is related to the maximum learning rate $$\alpha$$ you can use in Gradient Descent. If $$\alpha$$ (our $$h$$) is too large, GD can diverge (oscillate with increasing amplitude or "explode"), just like an unstable Euler method. For a quadratic loss $$L(\theta) = \frac{1}{2}\theta^T A \theta$$, GD requires $$\alpha < 2/\lambda_{\text{max}}(A)$$ for stability, where $$\lambda_{\text{max}}(A)$$ is the largest eigenvalue of $$A$$.
</blockquote>

## 5. Solving Systems of First-Order ODEs and Higher-Order ODEs

Many real-world problems involve systems of coupled first-order ODEs or single higher-order ODEs.

A $$k$$-th order ODE of the form $$y^{(k)}(t) = g(t, y, \dot{y}, \dots, y^{(k-1)})$$ can be converted into a system of $$k$$ first-order ODEs.
Let:

$$
z_1(t) = y(t)
$$

$$
z_2(t) = \dot{y}(t)
$$

$$
\vdots
$$

$$
z_k(t) = y^{(k-1)}(t)
$$

Then the system becomes:

$$
\dot{z}_1 = z_2
$$

$$
\dot{z}_2 = z_3
$$

$$
\vdots
$$

$$
\dot{z}_{k-1} = z_k
$$

$$
\dot{z}_k = g(t, z_1, z_2, \dots, z_k)
$$

This system, $$\dot{\mathbf{z}}(t) = \mathbf{F}(t, \mathbf{z}(t))$$, can then be solved using methods like Euler's, applied component-wise to the vector $$\mathbf{z}$$.

<blockquote class="prompt-example" markdown="1">
<div class="title" markdown="1">
**Example.** Heavy Ball ODE Discretization
</div>
Consider the Heavy Ball ODE for optimization parameter $$\theta(t)$$:

$$
m\ddot{\theta}(t) + \gamma \dot{\theta}(t) + \nabla L(\theta(t)) = 0
$$

Rearrange for the highest derivative: $$\ddot{\theta}(t) = -\frac{1}{m} (\gamma \dot{\theta}(t) + \nabla L(\theta(t)))$$.
This is a second-order ODE. Let $$x_1(t) = \theta(t)$$ (position) and $$x_2(t) = \dot{\theta}(t)$$ (velocity).
The system of first-order ODEs is:
1.  $$\dot{x}_1(t) = x_2(t)$$
2.  $$\dot{x}_2(t) = -\frac{1}{m} (\gamma x_2(t) + \nabla L(x_1(t)))$$

Now, apply Explicit Euler with step size $$h$$ to this system:

$$
x_{1, k+1} = x_{1, k} + h x_{2, k}
$$

$$
x_{2, k+1} = x_{2, k} + h \left( -\frac{1}{m} (\gamma x_{2, k} + \nabla L(x_{1, k})) \right)
$$

Let $$\theta_k = x_{1,k}$$ be the parameter at iteration $$k$$, and let $$v_k = x_{2,k}$$ be the "velocity" term.
The updates become:

$$
\theta_{k+1} = \theta_k + h v_k \quad (\ast)
$$

$$
v_{k+1} = v_k - \frac{h\gamma}{m} v_k - \frac{h}{m} \nabla L(\theta_k) = (1 - \frac{h\gamma}{m}) v_k - \frac{h}{m} \nabla L(\theta_k) \quad (\ast\ast)
$$

If we identify $$v_{k+1}$$ from $$(**)$$ with the step in $$(*)$$, i.e., define a slightly different velocity term for Polyak's momentum:
Let $$v_{k+1}^{\text{Polyak}} = \theta_{k+1} - \theta_k$$. Then Polyak's momentum update is often written as:

$$
v_{k+1}^{\text{Polyak}} = \mu v_k^{\text{Polyak}} - \alpha \nabla L(\theta_k)
$$

$$
\theta_{k+1} = \theta_k + v_{k+1}^{\text{Polyak}}
$$

Comparing this with the Euler discretization above ($$\theta_{k+1} - \theta_k = h v_k$$ for the first equation, and the update for $$v_{k+1}$$), we see a strong resemblance. If we set the "velocity" of Polyak's momentum $$v_k^{\text{Polyak}}$$ to be related to $$h v_k$$ from the Euler scheme, and align coefficients:
- $$\mu \approx (1 - \frac{h\gamma}{m})$$
- $$\alpha \approx \frac{h^2}{m}$$ (if we define Polyak's velocity as $$h x_{2,k}$$ for consistency with the position update)
This demonstrates that Polyak's momentum method can be seen as a specific discretization (related to Explicit Euler or semi-implicit Euler) of the Heavy Ball ODE. The precise mapping depends on how terms are grouped and how the discrete velocity is defined.
</blockquote>

## 6. Linear Multistep Methods (LMMs)

While one-step methods use only $$y_n$$ to find $$y_{n+1}$$, **Linear Multistep Methods (LMMs)** use information from several previous steps ($$y_n, y_{n-1}, \dots, y_{n-k+1}$$) to compute $$y_{n+1}$$. A general $$k$$-step LMM has the form:

$$
\sum_{j=0}^{k} a_j y_{n+j} = h \sum_{j=0}^{k} b_j f(t_{n+j}, y_{n+j})
$$

where $$a_j$$ and $$b_j$$ are constants defining the method, and $$a_k \neq 0$$. If $$b_k = 0$$, the method is explicit; otherwise, it's implicit.

<blockquote class="prompt-info" markdown="1">
<div class="title" markdown="1">
**Connection to Polyak's Momentum.**
</div>
Polyak's momentum, often written as $$\theta_{k+1} = \theta_k + \mu (\theta_k - \theta_{k-1}) - \eta \nabla L(\theta_k)$$ (where $$\eta$$ is a learning rate distinct from the momentum factor $$\mu$$), can be rearranged:

$$
\theta_{k+1} - (1+\mu)\theta_k + \mu\theta_{k-1} = - \eta \nabla L(\theta_k)
$$

This is a 2-step ($$k=2$$) explicit LMM where:
- $$a_2 = 1, a_1 = -(1+\mu), a_0 = \mu$$
- $$b_2 = 0, b_1 = -\eta/\text{h} \text{ (if we scale by h)}, b_0 = 0$$
(Or, more directly, the RHS is $$h \cdot (\text{scaled gradient term})$$).
Here, $$f(\theta_k) = -\nabla L(\theta_k)$$. This shows that Polyak's momentum fits the structure of an LMM.
</blockquote>

## 7. Summary: Connecting Numerical ODE Solvers to Optimization

This crash course has introduced fundamental concepts in the numerical solution of ODEs. The key takeaways for understanding optimization algorithms are:

*   Many optimization algorithms can be viewed as **discretizations of continuous-time dynamical systems** (ODEs) that describe ideal optimization paths.
    *   **Gradient Descent** is analogous to **Explicit Euler** applied to the **gradient flow ODE**.
    *   **Momentum methods** (like Polyak's) are analogous to discretizations (like Euler applied to a system or specific LMMs) of **second-order ODEs** like the Heavy Ball equation.
*   The **learning rate** ($$\alpha$$) in optimization algorithms plays a role similar to the **step size** ($$h$$) in numerical ODE solvers.
*   **Stability and convergence properties** of optimization algorithms can often be analyzed by studying the stability and accuracy of the corresponding numerical ODE method. An overly large learning rate leading to divergence in optimization is analogous to an unstable numerical scheme.
*   This ODE perspective provides a powerful framework for:
    *   **Understanding** the behavior of existing optimization methods.
    *   **Designing** new optimization algorithms by discretizing different ODEs or using more sophisticated numerical schemes.

Understanding these connections helps bridge the gap between the continuous mathematical theory of optimization and the discrete iterative algorithms used in practice.

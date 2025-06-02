---
title: "Variational Calculus Part 3: Lagrangian, Hamiltonian, and the Legendre Transform"
date: 2025-05-18 10:00 -0400
course_index: 3 # Third post in the Variational Calculus crash course
mermaid: true
description: "Exploring Lagrangian and Hamiltonian mechanics as applications of variational principles, and introducing the Legendre transform as a bridge to duality and convex analysis."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Euler-Lagrange Equation
- Lagrangian Mechanics
- Hamiltonian Mechanics
- Legendre Transform
- Duality
- Convexity
- Optimization
- Physics
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

In the previous part, we derived the **Euler-Lagrange equation**:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0
$$

This equation is a necessary condition for a function $$y(x)$$ to extremize a functional of the form $$J[y] = \int_a^b F(x, y(x), y'(x)) \, dx$$.

One of the most celebrated and historically significant applications of variational principles is in **classical mechanics**. This not only showcases the power of variational calculus but also introduces concepts and mathematical structures, like the Legendre transform, that are fundamental to understanding duality in optimization, including convex conjugation.

In this post, we will:
1.  Introduce **Lagrangian Mechanics** and the Principle of Stationary Action.
2.  Show how the Euler-Lagrange equation arises naturally in this context.
3.  Discuss the **Hamiltonian** formulation, derived from the Lagrangian via the **Legendre Transform**.
4.  Highlight the connection between the Legendre transform and **convexity**, setting the stage for convex analysis.

## 1. The Principle of Stationary Action and Lagrangian Mechanics

Classical mechanics, as formulated by Newton, describes the motion of objects using forces and accelerations (vectors). However, an alternative and often more powerful perspective was developed by Lagrange and Hamilton, based on the idea that physical systems evolve in a way that optimizes a certain quantity.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Principle.** Principle of Stationary Action
</div>
A physical system evolves from one configuration to another along a path that makes a quantity called the **action** stationary (usually a minimum).
</blockquote>

Let $$q(t)$$ represent the **generalized coordinates** of a system at time $$t$$. These could be positions, angles, or any set of variables that describe the system's configuration. The time derivative $$\dot{q}(t) = dq/dt$$ represents the **generalized velocities**.

The **action**, denoted $$S$$, is defined as the time integral of a function $$L$$, called the **Lagrangian**:

$$
S[q] = \int_{t_1}^{t_2} L(t, q(t), \dot{q}(t)) \, dt
$$

Here, the functional is $$S[q]$$, the independent variable is time $$t$$ (instead of $$x$$), the function being sought is the path $$q(t)$$ (instead of $$y(x)$$), and the integrand function is the Lagrangian $$L(t, q, \dot{q})$$.

**Why does $$L$$ depend on $$t, q, \dot{q}$$?**
This choice is fundamental for describing a vast range of physical systems:
-   $$q(t)$$: The potential energy of a system often depends on its configuration (e.g., $$V(q)$$).
-   $$\dot{q}(t)$$: The kinetic energy typically depends on velocities (e.g., $$T \propto \dot{q}^2$$).
-   $$t$$: Allows for systems where external influences change over time.

The problem is to find the path $$q(t)$$ that makes $$S[q]$$ stationary. This is precisely the type of problem the Euler-Lagrange equation solves! Substituting $$L$$ for $$F$$, $$q$$ for $$y$$, and $$t$$ for $$x$$, the Euler-Lagrange equation for each generalized coordinate $$q_i$$ (if there are multiple) becomes:

$$
\frac{\partial L}{\partial q_i} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) = 0
$$

This is the equation of motion in Lagrangian mechanics.

### What is the Lagrangian $$L$$?

For many classical mechanical systems, the Lagrangian is found to be the difference between the kinetic energy $$T$$ and the potential energy $$V$$ of the system:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Lagrangian for Classical Systems
</div>
For a system with kinetic energy $$T(q, \dot{q})$$ and potential energy $$V(q, t)$$, the Lagrangian is:

$$
L(t, q, \dot{q}) = T(q, \dot{q}) - V(q, t)
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Deriving $$L=T-V$$ from Newton's Laws (1D Example)**
</summary>
Consider a particle of mass $$m$$ in one dimension $$q$$, subject to a conservative force $$F(q) = -\frac{dV}{dq}$$. Newton's second law is $$m\ddot{q} = F(q)$$, or $$m\ddot{q} = -\frac{dV}{dq}$$. We can write this as:

$$
-\frac{dV}{dq} - \frac{d}{dt}(m\dot{q}) = 0
$$

We want to find an $$L(q, \dot{q})$$ such that the Euler-Lagrange equation $$\frac{\partial L}{\partial q} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) = 0$$ matches this.
Comparing terms:
1.  $$\frac{\partial L}{\partial q} = -\frac{dV}{dq}$$
2.  $$\frac{\partial L}{\partial \dot{q}} = m\dot{q}$$

Integrating condition (2) with respect to $$\dot{q}$$ (treating $$q$$ as constant) gives:

$$
L = \int m\dot{q} \, d\dot{q} = \frac{1}{2}m\dot{q}^2 + f(q)
$$

where $$f(q)$$ is an integration "constant" with respect to $$\dot{q}$$. We recognize $$\frac{1}{2}m\dot{q}^2$$ as the kinetic energy $$T(\dot{q})$$. So, $$L = T(\dot{q}) + f(q)$$.
Now use condition (1):

$$
\frac{\partial L}{\partial q} = \frac{\partial}{\partial q}(T(\dot{q}) + f(q)) = \frac{df}{dq}
$$

So, $$\frac{df}{dq} = -\frac{dV}{dq}$$. Integrating with respect to $$q$$ gives $$f(q) = -V(q) + C$$.
Thus, $$L = T(\dot{q}) - V(q) + C$$. The constant $$C$$ doesn't affect the equations of motion, so we can set $$C=0$$, yielding $$L = T - V$$.
</details>

Lagrangian mechanics offers several advantages:
-   **Scalar Formulation:** It works with scalar quantities (energy) rather than vectors (forces).
-   **Coordinate Independence:** The form of the Euler-Lagrange equations is preserved under changes of generalized coordinates.
-   **Symmetries and Conservation Laws:** As we'll touch upon, symmetries in the Lagrangian directly lead to conserved quantities (Noether's Theorem).

## 2. Hamiltonian Mechanics and the Legendre Transform

The Lagrangian formulation uses coordinates $$q$$ and velocities $$\dot{q}$$. An alternative, often more powerful formulation called **Hamiltonian mechanics**, uses coordinates $$q$$ and **generalized momenta** $$p$$. The transition from the ($$q, \dot{q}$$) description to the ($$q, p$$) description is achieved via a **Legendre Transform**.

### Generalized Momenta

In the Lagrangian framework, the generalized momentum $$p_i$$ conjugate to the coordinate $$q_i$$ is defined as:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i}
$$

For a simple particle with $$L = \frac{1}{2}m\dot{x}^2 - V(x)$$, the momentum conjugate to $$x$$ is $$p_x = \frac{\partial L}{\partial \dot{x}} = m\dot{x}$$, which is the familiar linear momentum.

### The Legendre Transform

The Legendre transform is a general mathematical procedure for changing the independent variable of a function. If we have a function $$f(x)$$, and we define a new variable $$p = \frac{df}{dx}$$, the Legendre transform $$f^\ast (p)$$ of $$f(x)$$ is defined as:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Legendre Transform
</div>
Given a function $$f(x)$$, its Legendre transform $$f^\ast (p)$$ with respect to $$x$$ is:

$$
f^\ast(p) = px - f(x)
$$

where $$x$$ on the right-hand side is implicitly expressed as a function of $$p$$ by inverting the relationship $$p = \frac{df}{dx}(x)$$. For this inversion to be unique (at least locally), we require $$\frac{d^2f}{dx^2} \neq 0$$. Typically, $$f(x)$$ is assumed to be **strictly convex**, meaning $$\frac{d^2f}{dx^2} > 0$$, which ensures $$p = f'(x)$$ is strictly increasing and thus invertible.
</blockquote>

An important property of the Legendre transform is its involutivity (it's its own inverse, up to a sign if conventions differ) and the symmetric derivative relationship:
If $$p = \frac{df}{dx}$$, then $$x = \frac{df^\ast }{dp}$$.

### The Hamiltonian

The **Hamiltonian** $$H(q, p, t)$$ is defined as the Legendre transform of the Lagrangian $$L(q, \dot{q}, t)$$ with respect to the generalized velocities $$\dot{q}_i$$. The new variables are the generalized momenta $$p_i$$.

$$
H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)
$$

Here, it's crucial that after performing the transform, the velocities $$\dot{q}_i$$ on the right-hand side are expressed as functions of $$q, p, t$$ by inverting the set of equations $$p_i = \frac{\partial L}{\partial \dot{q}_i}(q, \dot{q}, t)$$. This inversion requires the Hessian matrix $$W_{ij} = \frac{\partial^2 L}{\partial \dot{q}_i \partial \dot{q}_j}$$ to be invertible. If $$L$$ is strictly convex as a function of the velocities $$\dot{q}$$ (i.e., $$W$$ is positive definite), this inversion is well-defined.

**Hamilton's Equations of Motion:**
Using the properties of the Legendre transform and the Euler-Lagrange equations, one can derive Hamilton's equations of motion:

$$
\dot{q}_i = \frac{\partial H}{\partial p_i}
$$

$$
\dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

These are a set of first-order differential equations, contrasted with the second-order Euler-Lagrange equations.

**What is the Hamiltonian $$H$$?**
For many common systems where the kinetic energy $$T$$ is a quadratic function of velocities and the potential energy $$V$$ does not depend on velocities, the Hamiltonian simplifies to the total energy of the system:

$$
H = T + V
$$

This happens because if $$T = \sum_{j,k} \frac{1}{2} M_{jk}(q) \dot{q}_j \dot{q}_k$$ (a quadratic form in $$\dot{q}$$), then Euler's homogeneous function theorem implies $$\sum_i p_i \dot{q}_i = \sum_i \frac{\partial L}{\partial \dot{q}_i} \dot{q}_i = \sum_i \frac{\partial T}{\partial \dot{q}_i} \dot{q}_i = 2T$$.
Then $$H = 2T - L = 2T - (T-V) = T+V$$.

### The Legendre Transform and Convexity

The requirement that $$p = f'(x)$$ (or $$p_i = \partial L / \partial \dot{q}_i$$) be invertible is deeply connected to the **convexity** of the function being transformed.
-   If $$f(x)$$ is strictly convex, then $$f'(x)$$ is strictly increasing, guaranteeing a unique inverse $$x(p)$$.
-   The Legendre transform of a convex function is also convex. Specifically, if $$W = (\partial^2 L / \partial \dot{q} \partial \dot{q})$$ is the Hessian matrix of $$L$$ with respect to $$\dot{q}$$, then the Hessian matrix of $$H$$ with respect to $$p$$ is $$W^{-1}$$. If $$W$$ is positive definite (making $$L$$ convex in $$\dot{q}$$), then $$W^{-1}$$ is also positive definite (making $$H$$ convex in $$p$$).

This connection is not accidental. The Legendre transform is a fundamental example of a **duality transformation**. In optimization, the Legendre-Fenchel transform (or convex conjugate) generalizes this concept and plays a central role in Lagrangian duality, which is used to solve constrained optimization problems.

<blockquote class="prompt-tip" markdown="1">
**Preview: Convex Conjugation**
The Legendre transform $$f^\ast (p) = \sup_x (px - f(x))$$ (using supremum for generality) is known as the convex conjugate or Legendre-Fenchel transform. If $$f$$ is convex and closed, then $$f^{\ast \ast }=f$$. This concept is pivotal in convex analysis and optimization theory, allowing us to switch between "primal" and "dual" representations of problems. We will explore this in much more detail in the crash course on Convex Analysis.
</blockquote>

## 3. Symmetries and Conservation Laws (Noether's Theorem)

A profound consequence of the Lagrangian and Hamiltonian formalisms is Noether's Theorem, which connects continuous symmetries of the Lagrangian to conserved quantities.
-   **Time Translation Invariance:** If $$L$$ does not explicitly depend on time ($$\partial L / \partial t = 0$$), then the Hamiltonian $$H$$ (often total energy) is conserved ($$dH/dt = 0$$).
-   **Spatial Translation Invariance:** If $$L$$ does not depend on a coordinate $$q_k$$ (i.e., $$q_k$$ is "cyclic" or "ignorable", so $$\partial L / \partial q_k = 0$$), then the corresponding generalized momentum $$p_k = \partial L / \partial \dot{q}_k$$ is conserved ($$dp_k/dt = 0$$).
-   **Rotational Invariance:** If $$L$$ is invariant under rotations about an axis, the corresponding angular momentum is conserved.

<blockquote class="prompt-info" markdown="1">
The fact that physical laws often possess these symmetries (e.g., laws of physics are the same today as yesterday, or here as in a distant galaxy) underpins the fundamental conservation laws of energy, momentum, and angular momentum. These ideas of symmetry and invariants are also powerful in analyzing complex systems, including those in machine learning.
</blockquote>

## 4. What's Next?

In this post, we've seen how variational principles lead to Lagrangian and Hamiltonian mechanics, providing an elegant and powerful framework for describing physical systems. We also introduced the Legendre transform, which not only facilitates the switch from Lagrangian to Hamiltonian views but also serves as a crucial link to the concept of duality and convexity.

This detour through classical mechanics was intended to show:
1.  A major historical application and success of variational calculus.
2.  The origin of mathematical tools (like the Legendre transform) that are essential in modern optimization.

In the next parts of this crash course on Variational Calculus, we will:
-   Apply the Euler-Lagrange equation to solve some classic example problems more directly.
-   Discuss generalizations like functionals with higher-order derivatives or multiple independent/dependent variables.
-   Briefly touch upon constrained variational problems.

This will further solidify our understanding of variational methods before we eventually move on to more advanced optimization topics like Convex Analysis, where the seeds sown by the Legendre transform will fully blossom.

---
layout: post
title: Variational Calculus and Duality
date: 2025-04-10 03:32 -0400
description: Exploring variational principles, Lagrangian and Hamiltonian mechanics, and the Legendre transform connection to convex duality.
image:
categories:
- Machine Learning
- Mathematical Optimization
tags: variational-calculus lagrangian hamiltonian legendre-transform duality convex-optimization physics machine-learning
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.
  Please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by two newlines above and below, and newlines between the delimiters and the equation (even in lists):

  $$
  block
  $$

  Use LaTeX commands for symbols as much as possible such as $$\vert$$ or $$\ast$$. For instance, please avoid using the vertical bar symbol, only use \vert for absolute value, and \Vert for norm.

  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$

  Inside HTML environments (like blockquotes), please use the following syntax:

  \( inline \)

  \[
  block
  \]

  like so. Also, HTML markers must be used rather than markdown, e.g. <b>bold</b> rather than **bold**.
---

In this post, we will explore the principles of variational calculus, Lagrangian mechanics, and the Legendre dual transformation which will better expose us to the historical upbringing of convex duality.

## Variational Calculus: Optimizing Paths

Many problems in science, engineering, and even machine learning involve not just finding the optimal *value* or *parameters*, but finding the optimal *function* or *path*. Think about the shortest path between two points on a curved surface, the shape of a hanging chain, or the trajectory an object takes under gravity. These aren't single points, but continuous curves that minimize (or maximize) some overall quantity. Variational calculus is the mathematical framework for tackling such problems.

### Motivation: Why Look at Paths?

Consider a simple physical system, like a ball rolling down a hill. Newtonian mechanics describes its motion using forces and acceleration (vectors). This works incredibly well, but sometimes we want a different perspective.

1.  **Energy Minimization:** Physical systems often tend towards states of lower energy. The path a system *actually* takes between two states in a given time often minimizes (or makes stationary) a quantity related to energy, integrated over time. This is known as the **Principle of Stationary Action**.
2.  **Scalar Quantities:** Instead of juggling vector forces, this principle often involves integrating a *scalar* quantity (called the Lagrangian density) along possible paths. The path integral yields a single number (the "action") for each path. Finding the path that minimizes this scalar action is often simpler conceptually and calculationally than solving vector differential equations directly. This is called the **path integral formulation**.
3.  **Coordinate Invariance:** The action integral, being based on scalar quantities like energy, is often independent of the coordinate system you choose to describe the system. This is a powerful feature, making the approach highly flexible and applicable to complex systems where Cartesian coordinates might be awkward (like using angles for a pendulum).

<blockquote class="prompt-info">
<a href="https://en.wikipedia.org/wiki/Path_integral_formulation#:~:text=The%20basic%20idea,a%20starting%20point.">Historical note</a>: the path integral was originally inspired by Norbert Wiener's work in stochastic calculus, where the Wiener integral was used to study the cumulative behavior of Brownian motion. Paul Dirac extended this idea to quantum mechanics in 1933. The term "path integral" was coined by Feynman in 1948 to describe the integral over paths in quantum mechanics.
</blockquote>

### Defining the Action and the Lagrangian

Let's formalize this. We want to find a path or function, let's call it $$q(t)$$, that describes the state of our system over time $$t$$, from a starting time $$t_1$$ to an ending time $$t_2$$. The "state" $$q(t)$$ could be a position, an angle, or even a set of variables $$(q_1(t), q_2(t), ..., q_n(t))$$ describing a more complex system.

We hypothesize that nature optimizes a quantity called the **action**, typically denoted by $$S$$. The action is calculated by integrating a function, $$L$$, along a path $$q(t)$$:

$$
S[q] = \int_{t_1}^{t_2} L(t, q(t), \dot{q}(t)) \, dt
$$

Here:
*   $$S[q]$$ is the **action functional**. It's called a *functional* because its input is a function ($$q(t)$$) and its output is a scalar number.
*   $$L$$ is the **Lagrangian density** (often just called the **Lagrangian**). It's the function being integrated.
*   $$t$$ is the independent variable (usually time).
*   $$q(t)$$ represents the **generalized coordinates** of the system at time $$t$$. These are the variables needed to specify the system's configuration. For a single particle in 1D, $$q(t) = x(t)$$. For a pendulum, it might be the angle $$\theta(t)$$. For $$n$$ particles in 3D, $$q(t)$$ could be a vector of $$3n$$ coordinates.
*   $$\dot{q}(t)$$ represents the time derivative of the generalized coordinates, $$ \dot{q}(t) = \frac{dq}{dt} $$. These are the **generalized velocities**.

**Why does $$L$$ depend on $$t, q, \dot{q}$$?**

This choice is fundamental for describing a vast range of physical systems, particularly in classical mechanics:
*   **$$q(t)$$: Configuration:** The state or potential energy of a system often depends explicitly on its configuration (e.g., potential energy $$V(q)$$ depends on position).
*   **$$\dot{q}(t)$$: Rate of Change:** The kinetic energy typically depends on velocities ($$T \propto \dot{q}^2$$). More generally, how the system changes depends on its current rate of change. Including $$\dot{q}$$ allows the Lagrangian to encode the system's dynamics.
*   **$$t$$: Explicit Time Dependence:** Allows for systems where the governing rules or external influences change over time (e.g., a time-varying potential field).

This framework models systems whose future state is determined by their current position and velocity, which covers almost all of classical mechanics (particles, rigid bodies, oscillators, fields under certain approximations). It doesn't typically handle systems where the fundamental laws depend directly on acceleration ($$\ddot{q}$$) or higher derivatives, although extensions exist.

In differential geometry terms, the Lagrangian is a function from the tangent space of the manifold to the real numbers. The action is the integral of the Lagrangian over the manifold.

### The Calculus of Variations: Finding the Optimal Path

Our goal is to find the path $$q(t)$$ that makes the action $$S[q] = \int_{t_1}^{t_2} L(t, q, \dot{q}) dt$$ stationary. This means that for any small, arbitrary variation $$\eta(t)$$ away from the true path $$q(t)$$, the *first-order change* in the action, denoted $$\delta S$$, must be zero. We require the variation to vanish at the endpoints: $$\eta(t_1) = \eta(t_2) = 0$$, since the start and end points of the path are fixed. I'll gloss over some of the details, but the key steps are:

1.  **Vary the Action:** The variation of the action is the integral of the variation of the Lagrangian:

    $$
    \delta S = \delta \int_{t_1}^{t_2} L(t, q, \dot{q}) \, dt = \int_{t_1}^{t_2} \delta L \, dt
    $$

    Using the chain rule, the variation $$\delta L$$ depends on the variations $$\delta q$$ (which is just $$\eta(t)$$) and $$\delta \dot{q}$$ (which is $$\frac{d}{dt}\delta q = \dot{\eta}(t)$$) :
    
    $$
    \delta L = \frac{\partial L}{\partial q} \delta q + \frac{\partial L}{\partial \dot{q}} \delta \dot{q} = \frac{\partial L}{\partial q} \eta(t) + \frac{\partial L}{\partial \dot{q}} \dot{\eta}(t)
    $$

    Substituting this back into the action variation:
    
    $$
    \delta S = \int_{t_1}^{t_2} \left( \frac{\partial L}{\partial q} \eta(t) + \frac{\partial L}{\partial \dot{q}} \dot{\eta}(t) \right) dt
    $$

2.  **Integrate by Parts:** The key step is to handle the $$\dot{\eta}(t)$$ term. We use integration by parts on the second term: $$\int u \, dv = uv - \int v \, du$$, with $$u = \frac{\partial L}{\partial \dot{q}}$$ and $$dv = \dot{\eta}(t) dt$$ (so $$v = \eta(t)$$).
    
    $$
    \int_{t_1}^{t_2} \frac{\partial L}{\partial \dot{q}} \dot{\eta}(t) \, dt = \underbrace{\left[ \frac{\partial L}{\partial \dot{q}} \eta(t) \right]_{t_1}^{t_2}}_{=0} - \int_{t_1}^{t_2} \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) \eta(t) \, dt
    $$

    The boundary term vanishes because $$\eta(t_1) = \eta(t_2) = 0$$.

3.  **Combine and Apply the Fundamental Lemma:** Substituting the result of the integration by parts back into the expression for $$\delta S$$ gives:

    $$
    \delta S = \int_{t_1}^{t_2} \left( \frac{\partial L}{\partial q} \eta(t) - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) \eta(t) \right) dt = \int_{t_1}^{t_2} \left( \frac{\partial L}{\partial q} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) \right) \eta(t) \, dt
    $$

    For the action to be stationary, we require $$\delta S = 0$$ for *any* permissible variation $$\eta(t)$$. The **Fundamental Lemma of the Calculus of Variations** states that if $$\int_a^b f(t) \eta(t) dt = 0$$ for all well-behaved $$\eta(t)$$ vanishing at the boundaries, then the function $$f(t)$$ must be identically zero. Applying this here means the term multiplying $$\eta(t)$$ inside the integral must be zero.

4.  **The Euler-Lagrange Equation:** This leads directly to the condition that the optimal path $$q(t)$$ must satisfy:

    $$
    \boxed{ \frac{\partial L}{\partial q} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) = 0 }
    $$

    This is the **Euler-Lagrange equation**. It's a differential equation whose solution gives the path $$q(t)$$ that makes the action integral stationary. For systems with multiple coordinates $$q_i$$, we get one such equation for each $$i$$:

    $$
    \frac{\partial L}{\partial q_i} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) = 0 \quad \text{for } i = 1, ..., n
    $$

### Deriving the Lagrangian from Newtonian Mechanics

We have derived the Euler-Lagrange equation:

$$
\frac{\partial L}{\partial q} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) = 0
$$

This equation tells us the condition that a path $$q(t)$$ must satisfy to make the action integral $$S[q] = \int_{t_1}^{t_2} L(t, q(t), \dot{q}(t)) \, dt$$ stationary. But what function $$L$$ should we use to describe classical mechanics? We want the path predicted by the principle of stationary action to be the *same* path predicted by Newton's laws. Let's see if we can find an $$L$$ that makes the Euler-Lagrange equation *equivalent* to Newton's second law.

Consider the simplest case: a particle of mass $$m$$ moving in one dimension under a conservative force $$F(q)$$. This means the force depends only on the position $$q$$ and can be derived from a potential energy function $$V(q)$$ such that:

$$
F(q) = -\frac{\partial V}{\partial q}
$$

**Deriving the Lagrangian from Newtonian Mechanics**

Our goal is to find a function $$L(q, \dot{q})$$ such that the Euler-Lagrange equation:

$$
\frac{\partial L}{\partial q} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) = 0 \quad \text{(E-L)}
$$

is equivalent to Newton's second law for a particle of mass $$m$$ in one dimension under a conservative force $$F(q) = -\frac{\partial V}{\partial q}$$.

**Newton's Second Law** is:

$$
m \ddot{q} = F(q) \quad \implies \quad m \ddot{q} = -\frac{\partial V}{\partial q}
$$

We can rewrite the mass-acceleration term using the momentum $$p = m\dot{q}$$:

$$
m \ddot{q} = \frac{d}{dt}(m \dot{q})
$$

Substituting this into Newton's law gives:

$$
\frac{d}{dt}(m \dot{q}) = -\frac{\partial V}{\partial q}
$$

Rearranging this to resemble the structure of the E-L equation:

$$
-\frac{\partial V}{\partial q} - \frac{d}{dt}(m \dot{q}) = 0 \quad \text{(Newton)}
$$

Now, we compare this rewritten form of Newton's law with the E-L equation:

$$
\underbrace{-\frac{\partial V}{\partial q}}_{\text{must equal } \partial L / \partial q} \quad - \quad \frac{d}{dt} \Big( \underbrace{m \dot{q}}_{\text{must equal } \partial L / \partial \dot{q}} \Big) = 0
$$

$$
\underbrace{\frac{\partial L}{\partial q}}_{\text{Term 1}} \quad - \quad \frac{d}{dt} \Big( \underbrace{\frac{\partial L}{\partial \dot{q}}}_{\text{Term 2}} \Big) = 0
$$

For these two equations to be equivalent for any path $$q(t)$$, the corresponding terms must be equal. This gives us two conditions that $$L(q, \dot{q})$$ must satisfy:

1.  $$\begin{equation}
    \frac{\partial L}{\partial q} = -\frac{\partial V}{\partial q}
    \label{eq:condition-1} 
\end{equation}$$

2.  $$\begin{equation}
    \frac{\partial L}{\partial \dot{q}} = m \dot{q}
    \label{eq:condition-2}
\end{equation}$$

Let's find a function $$L$$ that satisfies both. Integrating $$\eqref{eq:condition-2}$$ with respect to $$\dot{q}$$ (treating $$q$$ as constant during this partial integration):

$$
L(q, \dot{q}) = \int (m \dot{q}) \, d\dot{q} = \frac{1}{2} m \dot{q}^2 + f(q)
$$

where $$f(q)$$ is an arbitrary function of $$q$$, acting as the "integration constant" with respect to $$\dot{q}$$. We recognize $$\frac{1}{2} m \dot{q}^2$$ as the kinetic energy, $$T(\dot{q})$$. So, $$L(q, \dot{q}) = T(\dot{q}) + f(q)$$.

Now, substitute this form of $$L$$ into $$\eqref{eq:condition-1}$$ to get:

$$
\frac{\partial L}{\partial q} = \frac{\partial}{\partial q} \Big( T(\dot{q}) + f(q) \Big) = \frac{d f}{d q}
$$

Comparing this with $$\eqref{eq:condition-1}$$ gives:

$$
\frac{d f}{d q} = -\frac{\partial V}{\partial q}
$$

Integrating this with respect to $$q$$ gives:

$$
f(q) = -V(q) + C
$$

where C is an arbitrary constant.

Substituting this $$f(q)$$ back into our expression for $$L$$:

$$
L(q, \dot{q}) = T(\dot{q}) - V(q) + C
$$

Since adding a constant $$C$$ to the Lagrangian does not change the Euler-Lagrange equations (as $$ \frac{\partial C}{\partial q} = 0 $$ and $$ \frac{\partial C}{\partial \dot{q}} = 0 $$), we can choose $$C=0$$ for simplicity.

Thus, we have derived that the Lagrangian for a classical particle in a conservative potential is the **Kinetic Energy minus the Potential Energy**:

$$
L = T - V
$$

With this definition, the principle of stationary action ($$\delta \int L \, dt = 0$$) yields Newton's laws of motion.


This formulation has significant advantages:
*   **Scalar Formulation:** It uses scalar quantities (energy) instead of vector quantities (force, acceleration), simplifying calculations, especially in complex coordinate systems.
*   **Generalization:** It naturally handles constraints and generalizes readily to multi-particle systems, rigid bodies, and even field theories using generalized coordinates.
*   **Symmetry and Conservation:** As we will see later, symmetries in the Lagrangian directly lead to conservation laws (like conservation of energy or momentum) via Noether's theorem.

Thus, Lagrangian mechanics provides a profound and elegant reformulation of classical mechanics, rooted in the fundamental principle of optimizing a path integral – the action. This variational perspective is not just useful in physics but also provides foundational ideas for optimization in other fields, including machine learning.

### Changing Gears: The Legendre Transform from Velocity to Momentum

The Lagrangian $$L(q, \dot{q}, t)$$ describes mechanics using generalized coordinates $$q$$ and velocities $$\dot{q}$$. A key quantity derived from the Lagrangian is the **generalized momentum** $$p$$, conjugate to $$q$$:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i}
$$

This relationship suggests viewing the momentum $$p$$ as a new variable derived from the velocity $$\dot{q}$$. We wish to transform our description from ($$q, \dot{q}$$) to ($$q, p$$) using a new function, the Hamiltonian $$H(q, p, t)$$. This transformation should ideally possess the elegant symmetry properties of the **Legendre Transform**.

#### The Legendre Transform Operator $$\ast$$

Let's consider the Legendre transform as an operation $$\ast$$ that takes a function $$f(x)$$ and produces a new function $$f^\ast (p)$$. We desire this transform to have the following properties:

1.  **Derivative Relationship:** The new variable $$p$$ is the derivative of the original function with respect to the original variable:
    
    $$
    p = \frac{df}{dx} 
    $$

2.  **Symmetric Derivative Relationship (Involutivity):** The original variable $$x$$ should be recoverable as the derivative of the *transformed* function $$f^\ast $$ with respect to the *new* variable $$p$$:

    $$ 
    x = \frac{df^\ast }{dp}
    $$

3.  **Invertibility:** To express $$f^\ast $$ purely as a function of $$p$$, the relationship $$p = f'(x)$$ must be invertible, allowing us to write $$x = x(p)$$. Similarly, to transform back, $$x = (f^\ast )'(p)$$ must be invertible.

#### Deriving the Form of the Legendre Transform

Properties 1 and 2 dictate the functional form of $$f^\ast (p)$$. From the differentials $$df = p \, dx$$ and $$df^\ast  = x \, dp$$, compared with $$d(px) = p \, dx + x \, dp$$, we derive $$df^\ast  = d(px - f)$$, which integrates to:

$$
\begin{equation}
    \boxed{ f^\ast (p) = px(p) - f(x(p)) }
    \label{eq:legendre-transform}
\end{equation}
$$

where the notation emphasizes that $$x$$ must be expressed as a function of $$p$$ using the inverted relationship from Property 1.

#### The Invertibility Requirement and Convexity

Crucially, **Property 3 (Invertibility)** is essential for the Legendre transform to be well-defined and useful. For the relationship $$p = f'(x)$$ to be locally invertible, allowing us to uniquely determine $$x$$ for a given $$p$$ (i.e., write $$x=x(p)$$), the derivative $$f'(x)$$ must be strictly monotonic. This occurs when the second derivative does not change sign, meaning $$f''(x)$$ is either always positive or always negative (excluding points where it's zero).
*   If $$f''(x) > 0$$, the function $$f(x)$$ is **strictly convex**, i.e. its graph curves strictly upwards.
*   If $$f''(x) < 0$$, the function $$f(x)$$ is **strictly concave**, i.e. its graph curves strictly downwards.

The standard Legendre transform typically assumes the function $$f(x)$$ is **strictly convex**. This ensures $$p = f'(x)$$ is a strictly increasing function and thus uniquely invertible.

In the context of the Lagrangian $$L(q, \dot{q}, t)$$ transformed with respect to $$\dot{q}$$, the invertibility condition means that the mapping from velocities $$\dot{q}$$ to momenta $$p$$ defined by $$p_i = \partial L / \partial \dot{q}_i$$ must be invertible. This requires the **Hessian matrix** of $$L$$ with respect to the velocities, $$W_{ij} = \frac{\partial^2 L}{\partial \dot{q}_i \partial \dot{q}_j}$$, to be **invertible** (non-singular). If this matrix is also positive definite (which corresponds to convexity of $$L$$ as a function of $$\dot{q}$$), the inversion is typically well-behaved. For most standard mechanical systems where $$L=T-V$$ and $$T$$ is quadratic in velocities, this condition holds.

We will cover more about convexity in future posts.

#### Applying the Legendre Transform to the Lagrangian

Now, we apply the Legendre transform definition $$\eqref{eq:legendre-transform}$$ to transform $$L(q, \dot{q}, t)$$ into the Hamiltonian $$H(q, p, t)$$, performing the transform with respect to the velocity variables $$\dot{q}_i$$.

*   Original function: $$L$$ (viewed primarily as a function of $$\dot{q}$$ for the transform)
*   Original variable: $$\dot{q} = (\dot{q}_1, ..., \dot{q}_n)$$
*   New variable (conjugate momentum): $$p = (p_1, ..., p_n)$$, where 
$$ 
p_i = \frac{\partial L}{\partial \dot{q}_i} 
$$
*   Transformed function: $$H(q, p, t)$$.

The **Hamiltonian $$H$$ is defined as the Legendre Transform of the Lagrangian $$L$$ with respect to the generalized velocities $$\dot{q}$$**:

$$
\boxed{ H(q, p, t) \equiv \sum_{i=1}^n p_i \dot{q}_i(q, p, t) - L(q, \dot{q}(q, p, t), t) }
$$

Here, the notation $$\dot{q}_i(q, p, t)$$ explicitly shows the result of **inverting** the momentum definition ($$p_i = \partial L / \partial \dot{q}_i$$), which relies on the invertibility condition discussed above (typically, $$L$$ being convex in $$\dot{q}$$).

#### Deriving Hamilton's Equations and Second Derivatives

By construction (as verified in the previous version), this definition guarantees the symmetric first derivative relationship $$\dot{q}_i = \partial H / \partial p_i$$. It also yields $$\partial H / \partial q_i = -\partial L / \partial q_i$$ and $$\partial H / \partial t = -\partial L / \partial t$$. Combining with the Euler-Lagrange equation ($$\dot{p}_i = \partial L / \partial q_i$$) gives **Hamilton's Equations**:

$$
\boxed{ \dot{q}_i = \frac{\partial H}{\partial p_i} \quad \text{and} \quad \dot{p}_i = -\frac{\partial H}{\partial q_i} }
$$

**Second Derivative Relationship:** Let's examine the relationship between the second derivatives. Consider the matrix of second derivatives of $$H$$ with respect to $$p$$:

$$
\frac{\partial^2 H}{\partial p_j \partial p_i} = \frac{\partial}{\partial p_j} \left( \frac{\partial H}{\partial p_i} \right) = \frac{\partial \dot{q}_i}{\partial p_j}
$$

Now consider the Hessian matrix of $$L$$ with respect to $$\dot{q}$$, let's call it $$W$$: $$W_{ki} = \frac{\partial^2 L}{\partial \dot{q}_k \partial \dot{q}_i}$$.
From $$p_j = \partial L / \partial \dot{q}_j$$, we can differentiate with respect to $$p_k$$ using the chain rule:

$$
\frac{\partial p_j}{\partial p_k} = \delta_{jk} = \sum_i \frac{\partial}{\partial \dot{q}_i} \left( \frac{\partial L}{\partial \dot{q}_j} \right) \frac{\partial \dot{q}_i}{\partial p_k} = \sum_i \frac{\partial^2 L}{\partial \dot{q}_i \partial \dot{q}_j} \frac{\partial \dot{q}_i}{\partial p_k} = \sum_i W_{ij} \frac{\partial \dot{q}_i}{\partial p_k}
$$

Let $$H_{pk}$$ be the matrix element $$\frac{\partial^2 H}{\partial p_k \partial p_i} = \frac{\partial \dot{q}_i}{\partial p_k}$$. The equation above reads $$\delta_{jk} = \sum_i W_{ij} H_{pk}$$. In matrix notation, this is $$I = W H_p$$, where $$H_p$$ is the matrix of second partial derivatives of $$H$$ with respect to $$p$$. This means the Hessian matrix of $$H$$ with respect to $$p$$ is the inverse of the Hessian matrix of $$L$$ with respect to $$\dot{q}$$:

$$
\boxed{ \left( \frac{\partial^2 H}{\partial p \partial p} \right) = \left( \frac{\partial^2 L}{\partial \dot{q} \partial \dot{q}} \right)^{-1} }
$$

This explicitly shows that if $$L$$ is strictly convex in $$\dot{q}$$ (its Hessian $$W$$ is positive definite), then $$H$$ will be strictly convex in $$p$$ (its Hessian $$H_p = W^{-1}$$ is also positive definite). This relationship relies fundamentally on the invertibility established by the first derivatives.

In conclusion, the Hamiltonian formalism arises from applying the Legendre transform ($$H = \sum p_i \dot{q}_i - L$$) to the Lagrangian, viewing $$L$$ as a function of velocities $$\dot{q}$$. This transform requires the relationship $$p = \partial L / \partial \dot{q}$$ to be invertible (often ensured by $$L$$ being convex in $$\dot{q}$$) and inherently establishes the symmetric first derivative roles ($$\dot{q} = \partial H / \partial p$$) and an inverse relationship between the second derivative matrices.

For many common physical systems where $$L = T(q, \dot{q}) - V(q)$$ and the kinetic energy $$T$$ is a quadratic function of the velocities $$\dot{q}_i$$, it can be shown that $$ \sum_i p_i \dot{q}_i = 2T $$. In such cases, the Hamiltonian simplifies to:
$$
H = 2T - (T - V) = T + V
$$
Thus, for these standard systems, the Hamiltonian $$H$$ represents the **total energy** (kinetic plus potential) of the system.

### (Extra) Symmetries and Conservation Principles: Noether's Insight

One of the most profound insights arising from the Lagrangian and Hamiltonian formulations is the direct connection between **symmetries** of the system and **conserved quantities**. This relationship is formalized by **Noether's Theorem**, a cornerstone of modern physics. While we won't derive the theorem in its full generality, we can easily see how specific symmetries lead to familiar conservation laws using the equations we already have.

The core idea is: If the Lagrangian (and thus the physics it describes) remains unchanged under a certain transformation (like shifting in time, translating in space, or rotating), then there must be a corresponding quantity that remains constant throughout the system's motion.

This is interesting, since many physical systems of interest are invariant under certain transformations. For instance, images should be invariant under translations, rotations, and possibly reflections. So this has implications for image processing, for instance.

#### 1. Conservation of Energy: Time Translation Symmetry

What happens if the Lagrangian $$L$$ does not explicitly depend on time? That is, $$\frac{\partial L}{\partial t} = 0$$. This means the rules governing the system don't change over time – the physics is invariant under time shifts. Let's see what this implies for the Hamiltonian $$H$$.

Recall the relationship between the total time derivative of $$H$$ and the partial derivative of $$L$$ with respect to time, derived directly from the definition $$H = \sum p_i \dot{q}_i - L$$ and the Euler-Lagrange equations:

$$
\frac{dH}{dt} = - \frac{\partial L}{\partial t}
$$

(See derivation in previous section or standard texts). This fundamental result directly connects the rate of change of the Hamiltonian $$H$$ to the explicit time dependence of the Lagrangian $$L$$.

Therefore, **if the Lagrangian $$L$$ does not explicitly depend on time ($$\frac{\partial L}{\partial t} = 0$$), then $$\frac{dH}{dt} = 0$$**. This means the Hamiltonian $$H$$ is a **conserved quantity**.

Since we established that for typical systems $$H = T + V$$ is the total energy, this demonstrates that **energy is conserved if the system's physics (as encoded in $$L$$) does not explicitly change with time**.

#### 2. Conservation of Momentum: Translational Symmetry

What if the Lagrangian $$L$$ does not depend on a particular generalized coordinate, say $$q_k$$? That is, $$\frac{\partial L}{\partial q_k} = 0$$. Such a coordinate is called **cyclic** or **ignorable**. This corresponds to a symmetry under translation along the $$q_k$$ coordinate – the physics doesn't change if you shift the whole system along that coordinate.

Let's look at the Euler-Lagrange equation for this specific coordinate $$q_k$$:

$$
\frac{\partial L}{\partial q_k} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_k}\right) = 0
$$

Since we assumed $$\frac{\partial L}{\partial q_k} = 0$$, the equation simplifies dramatically:

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_k}\right) = 0
$$

But recall the definition of the generalized momentum conjugate to $$q_k$$:

$$
p_k = \frac{\partial L}{\partial \dot{q}_k}
$$

Substituting this into the simplified Euler-Lagrange equation gives:

$$
\frac{d p_k}{dt} = 0
$$

This means that the generalized momentum $$p_k$$ conjugate to the cyclic coordinate $$q_k$$ is **conserved** – it does not change over time.

For example, if $$L$$ doesn't depend on the x-coordinate ($$\frac{\partial L}{\partial x} = 0$$), then the corresponding linear momentum $$p_x = \frac{\partial L}{\partial \dot{x}}$$ is conserved. If the system consists of multiple particles and $$L$$ is invariant under shifting the entire system in the x-direction, the *total* momentum in the x-direction is conserved.

#### 3. Conservation of Angular Momentum: Rotational Symmetry

Similarly, if the Lagrangian is independent of an angular coordinate $$\phi$$ (meaning the physics doesn't change if you rotate the system around a certain axis), then $$\frac{\partial L}{\partial \phi} = 0$$. The Euler-Lagrange equation for $$\phi$$ immediately tells us that the corresponding conjugate momentum is conserved:

$$
p_\phi = \frac{\partial L}{\partial \dot{\phi}} \quad \text{is conserved} \quad \left( \frac{d p_\phi}{dt} = 0 \right)
$$

This quantity $$p_\phi$$ is precisely the **angular momentum** associated with rotation about that axis.

In summary, the Lagrangian (and Hamiltonian) framework elegantly reveals the deep connection between symmetries and conservation laws:
*   **Time Translation Invariance ($$\frac{\partial L}{\partial t} = 0$$) $$\implies$$ Energy Conservation ($$\frac{dH}{dt} = 0$$)**
*   **Spatial Translation Invariance ($$\frac{\partial L}{\partial q_k} = 0$$) $$\implies$$ Momentum Conservation ($$\frac{dp_k}{dt} = 0$$)**
*   **Rotational Invariance ($$\frac{\partial L}{\partial \phi} = 0$$) $$\implies$$ Angular Momentum Conservation ($$\frac{dp_\phi}{dt} = 0$$)**

This is a powerful predictive tool and highlights the fundamental nature of these conservation laws as consequences of the symmetries of the underlying physical laws. This perspective, where optimization principles (stationary action) and symmetries lead to core physical laws and conserved quantities, resonates strongly with finding structure and invariants in complex systems, a theme also relevant in machine learning.

---
title: "Cheat Sheet: Variational Calculus for Optimization"
date: 2025-05-20 11:00 -0400 # Slightly after the last content post
sort_index: 999 # Designates this as a summary/cheat sheet
mermaid: true
description: "A concise summary of key concepts, equations, and examples from the Variational Calculus crash course, including functionals, the Euler-Lagrange equation, Lagrangian/Hamiltonian mechanics, and the Legendre transform."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Cheat Sheet
- Euler-Lagrange Equation
- Functionals
- Lagrangian Mechanics
- Hamiltonian Mechanics
- Legendre Transform
- Optimization
---

This post serves as a concise cheat sheet for the "Variational Calculus for Optimization" crash course. It summarizes the key definitions, equations, and examples discussed throughout the series.

## 1. Core Concepts

| Concept                                         | Description                                                                                                                     | Notation Example                                                             |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------- |
| **Functional**                                  | A mapping from a set of functions to the real numbers.                                                                          | $$J[y]$$ or $$J[y(x)]$$                                                      |
| **Variation**                                   | A small, arbitrary perturbation to a function, $$\epsilon \eta(x)$$.                                                            | $$y(x) + \epsilon \eta(x)$$                                                  |
| **First Variation**                             | The principal linear part of the change in a functional due to a variation. Necessary condition for extremum: $$\delta J = 0$$. | $$\delta J[y; \eta] = \left. \frac{d}{d\epsilon} J[y + \epsilon \eta] \right \vert  _{\epsilon=0}$$ |
| **Admissible Variation**                        | A variation $$\eta(x)$$ that respects the boundary conditions of the problem (e.g., $$\eta(a)=\eta(b)=0$$ for fixed endpoints). |                                                                              |
| **Fundamental Lemma of Calculus of Variations** | If $$\int_a^b g(x)\eta(x)dx = 0$$ for all admissible $$\eta(x)$$, then $$g(x)=0$$ for all $$x \in [a,b]$$.                      |                                                                              |

## 2. The Euler-Lagrange Equation (and its forms)

The Euler-Lagrange equation provides a necessary condition for a function to extremize a functional. For a general functional whose integrand is $$F$$:

| Type of Functional / Condition                                                     | Integrand $$F$$                                                     | Euler-Lagrange Equation(s)                                                                                                                                                                                                                                                      | Resulting Equation Type |
| :--------------------------------------------------------------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
| **Standard Case** <br/> (1 dep. var $$y(x)$$, 1st order deriv. $$y'$$)             | $$F(x, y, y')$$                                                     | $$\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0$$                                                                                                                                                                              | ODE (2nd order)         |
| **Special Case: $$F$$ independent of $$x$$** <br/> ($$\partial F/\partial x = 0$$) | $$F(y, y')$$                                                        | **Beltrami Identity:** $$F - y' \frac{\partial F}{\partial y'} = C$$ (constant)                                                                                                                                                                                                 | ODE (1st order)         |
| **Special Case: $$F$$ independent of $$y$$** <br/> ($$\partial F/\partial y = 0$$) | $$F(x, y')$$                                                        | $$\frac{\partial F}{\partial y'} = C$$ (constant)                                                                                                                                                                                                                               | ODE (1st order)         |
| **Higher-Order Derivatives** <br/> (up to $$y^{(n)}$$)                             | $$F(x, y, y', \dots, y^{(n)})$$                                     | $$\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) + \frac{d^2}{dx^2}\left(\frac{\partial F}{\partial y''}\right) - \dots + (-1)^n \frac{d^n}{dx^n}\left(\frac{\partial F}{\partial y^{(n)}}\right) = 0$$ <br/> (Euler-Poisson Equation) | ODE (2n-th order)       |
| **Multiple Dependent Variables** <br/> ($$y_k(x)$$, $$k=1..m$$)                    | $$F(x, y_1, \dots, y_m, y_1', \dots, y_m')$$                        | $$\frac{\partial F}{\partial y_k} - \frac{d}{dx} \left( \frac{\partial F}{\partial y_k'} \right) = 0$$, for each $$k$$                                                                                                                                                          | System of ODEs          |
| **Multiple Independent Variables** <br/> (e.g., $$u(x,y)$$)                        | $$F(x,y, u, u_x, u_y)$$ <br/> ($$u_x=\partial u/\partial x$$, etc.) | $$\frac{\partial F}{\partial u} - \frac{\partial}{\partial x} \left( \frac{\partial F}{\partial u_x} \right) - \frac{\partial}{\partial y} \left( \frac{\partial F}{\partial u_y} \right) - \dots = 0$$                                                                         | PDE                     |

## 3. Lagrangian and Hamiltonian Mechanics

A prime application of variational calculus in physics.

| Concept                          | Definition / Key Equation                                                                                       | Variables Used    | Notes                                                                        |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :---------------- | :--------------------------------------------------------------------------- |
| **Action $$S$$**                 | $$S[q] = \int_{t_1}^{t_2} L(t, q(t), \dot{q}(t)) \, dt$$                                                        | $$t, q, \dot{q}$$ | Principle of Stationary Action: $$\delta S = 0$$ yields equations of motion. |
| **Lagrangian $$L$$**             | $$L = T - V$$ <br/> (Kinetic Energy $$T$$ minus Potential Energy $$V$$ for classical systems)                   | $$t, q, \dot{q}$$ | The integrand of the action.                                                 |
| **Generalized Momentum $$p_i$$** | $$p_i = \frac{\partial L}{\partial \dot{q}_i}$$                                                                 | $$p, q, \dot{q}$$ | Momentum conjugate to the generalized coordinate $$q_i$$.                    |
| **Hamiltonian $$H$$**            | $$H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)$$ <br/> (Legendre transform of $$L$$ w.r.t. $$\dot{q}$$) | $$t, q, p$$       | Often represents the total energy ($$H=T+V$$) of the system.                 |
| **Hamilton's Equations**         | $$\dot{q}_i = \frac{\partial H}{\partial p_i}$$,  $$\dot{p}_i = -\frac{\partial H}{\partial q_i}$$              | $$t, q, p$$       | A set of first-order ODEs describing motion.                                 |

## 4. Legendre Transform

A mathematical tool for changing variables, crucial for moving from Lagrangian to Hamiltonian mechanics and foundational for convex duality.

-   **Purpose:** Transforms a function $$f(x)$$ into a new function $$f^\ast (p)$$ where the new variable $$p$$ is related to the derivative of $$f$$.
-   **Definition:** Given $$f(x)$$ and defining $$p = \frac{df}{dx}(x)$$, the Legendre transform is:

    $$
    f^\ast(p) = px - f(x)
    $$

    where $$x$$ on the right-hand side must be expressed as a function of $$p$$ by inverting $$p = f'(x)$$.
-   **Invertibility Requirement:** For the inversion $$x(p)$$ to be well-defined, $$f'(x)$$ must be monotonic, which is guaranteed if $$f(x)$$ is strictly convex ($$f''(x) > 0$$) or strictly concave ($$f''(x) < 0$$). Standardly, strict convexity is assumed.
-   **Symmetric Derivative Property:** If $$p = f'(x)$$, then the original variable is recovered by $$x = (f^\ast )'(p)$$.
-   **Relationship between Hessians:** The Hessian of $$f^\ast (p)$$ with respect to $$p$$ is the inverse of the Hessian of $$f(x)$$ with respect to $$x$$:

    $$
    \left( \frac{\partial^2 f^\ast}{\partial p^2} \right) = \left( \frac{\partial^2 f}{\partial x^2} \right)^{-1}
    $$

    This implies that convexity is preserved under the Legendre transform.
-   **Connection to Convex Conjugate (Legendre-Fenchel Transform):**

    $$
    f^\ast(p) = \sup_x (px - f(x))
    $$

## 5. Constrained Variational Problems (Isoperimetric Problems)

Problems where we extremize a functional $$J[y]$$ subject to an integral constraint $$K[y] = L_0$$ (a constant).

-   **Method of Lagrange Multipliers:**
    1.  Form an auxiliary functional $$J^\ast [y]$$ using a Lagrange multiplier $$\lambda$$:

        $$
        J^\ast[y] = J[y] + \lambda K[y] = \int_a^b (F(x, y, y') + \lambda G(x, y, y')) \, dx = \int_a^b H_\lambda(x, y, y', \lambda) \, dx
        $$

        where $$F$$ is the integrand of $$J$$, $$G$$ is the integrand of $$K$$, and $$H_\lambda = F + \lambda G$$.
    2.  Apply the Euler-Lagrange equation to the new integrand $$H_\lambda$$ (treating $$\lambda$$ as a constant for this step):

        $$
        \frac{\partial H_\lambda}{\partial y} - \frac{d}{dx}\left(\frac{\partial H_\lambda}{\partial y'}\right) = 0
        $$

    3.  Solve the resulting differential equation. The solution $$y(x, \lambda)$$ will depend on $$\lambda$$.
    4.  Substitute this solution back into the original constraint equation $$K[y(x, \lambda)] = L_0$$ to determine the value of the Lagrange multiplier $$\lambda$$.

## 6. Classic Problems and Their Solutions

| Problem                                           | Functional's Integrand $$F(x,y,y')$$ (or similar)                   | Solution Curve / Shape           |
| :------------------------------------------------ | :------------------------------------------------------------------ | :------------------------------- |
| **Shortest Path** (in a plane)                    | $$\sqrt{1+(y')^2}$$                                                 | Straight Line                    |
| **Brachistochrone** (fastest descent)             | $$\frac{\sqrt{1+(y')^2}}{\sqrt{2gy}}$$                              | Cycloid                          |
| **Catenary** (hanging chain, min. PE)             | $$y\sqrt{1+(y')^2}$$ (integrand proportional to this)               | Catenary ($$y=a \cosh(x/a)$$)    |
| **Minimal Surface** (e.g., soap film)             | $$\sqrt{1+u_x^2+u_y^2}$$ (for a surface $$u(x,y)$$)                 | Surface with zero mean curvature |
| **Dido's Problem** (max area for fixed perimeter) | Maximize $$\int y dx$$ subject to $$\int \sqrt{1+(y')^2} dx = L_0$$ | Circular Arc                     |

This cheat sheet provides a quick reference to the fundamental tools and results of variational calculus discussed in this series. For detailed derivations and explanations, please refer to the individual posts.

---
title: "Variational Calculus Part 5: Generalizations and Constraints"
date: 2025-05-20 10:00 -0400
sort_index: 5 # Fifth post in the Variational Calculus crash course
mermaid: true
description: "Exploring generalizations of the Euler-Lagrange equation for higher-order derivatives, multiple functions, multiple independent variables, and an introduction to constrained variational problems using Lagrange multipliers."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Euler-Lagrange Equation
- Euler-Poisson Equation
- Isoperimetric Problems
- Lagrange Multipliers
- Partial Differential Equations
---

So far, we've focused on functionals of the form $$J[y] = \int_a^b F(x, y, y') \, dx$$ and derived the corresponding Euler-Lagrange equation. However, many problems in physics, engineering, and optimization involve more complex functionals. In this part, we'll explore some important generalizations of the Euler-Lagrange equation and briefly introduce how constraints are handled in variational problems.

## 1. Functionals with Higher-Order Derivatives

Sometimes, the quantity we want to optimize depends not only on $$y(x)$$ and its first derivative $$y'(x)$$ but also on higher-order derivatives like $$y''(x)$$, $$y'''(x)$$, and so on, up to $$y^{(n)}(x)$$.
Consider a functional of the form:

$$
J[y] = \int_a^b F(x, y, y', y'', \dots, y^{(n)}) \, dx
$$

To find the necessary condition for an extremum, we again set the first variation $$\delta J[y; \eta]$$ to zero. This involves repeated integration by parts. For each derivative $$y^{(k)}$$ in $$F$$, we will get a term involving $$\eta^{(k)}(x)$$. Each such term must be integrated by parts $$k$$ times to isolate $$\eta(x)$$.

The boundary conditions for $$\eta(x)$$ also become more extensive. If the values of $$y, y', \dots, y^{(n-1)}$$ are fixed at the endpoints $$x=a$$ and $$x=b$$, then the admissible variations $$\eta(x)$$ must satisfy:

$$
\eta(a) = \eta'(a) = \dots = \eta^{(n-1)}(a) = 0
$$

$$
\eta(b) = \eta'(b) = \dots = \eta^{(n-1)}(b) = 0
$$

These conditions ensure that all boundary terms arising from the integrations by parts vanish.

The resulting generalized Euler-Lagrange equation, often called the **Euler-Poisson equation**, is:
<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Euler-Poisson Equation (Higher-Order Derivatives)
</div>
For a functional $$J[y] = \int_a^b F(x, y, y', \dots, y^{(n)}) \, dx$$, the extremizing function $$y(x)$$ must satisfy:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) + \frac{d^2}{dx^2}\left(\frac{\partial F}{\partial y''}\right) - \dots + (-1)^n \frac{d^n}{dx^n}\left(\frac{\partial F}{\partial y^{(n)}}\right) = 0
$$

This is a differential equation of order $$2n$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Elastic Beam Theory
</div>
In the theory of elasticity, the potential energy of a thin elastic beam undergoing small deflections $$y(x)$$ is related to its bending stiffness $$EI$$ (where $$E$$ is Young's modulus and $$I$$ is the area moment of inertia) and its curvature, which is approximately $$y''(x)$$. A simple model for the energy might involve a functional like:

$$
J[y] = \int_0^L \left( \frac{1}{2} EI (y''(x))^2 - w(x)y(x) \right) \, dx
$$

where $$w(x)$$ is an external distributed load. Here, $$F(x, y, y'') = \frac{1}{2} EI (y'')^2 - w y$$. (Note: $$F$$ does not depend on $$y'$$ in this simplified model).
The Euler-Poisson equation for $$n=2$$ is:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) + \frac{d^2}{dx^2}\left(\frac{\partial F}{\partial y''}\right) = 0
$$

Calculating the terms:
-   $$\frac{\partial F}{\partial y} = -w(x)$$
-   $$\frac{\partial F}{\partial y'} = 0$$
-   $$\frac{\partial F}{\partial y''} = EI y''(x)$$
So, the Euler-Poisson equation becomes:

$$
-w(x) - 0 + \frac{d^2}{dx^2}(EI y''(x)) = 0
$$

If $$EI$$ is constant:

$$
EI \frac{d^4y}{dx^4} = w(x)
$$

This is the classic fourth-order differential equation governing the deflection of an elastic beam.
</blockquote>

## 2. Functionals with Multiple Dependent Variables

Often, a system is described by multiple functions $$y_1(x), y_2(x), \dots, y_m(x)$$. The functional will then depend on all these functions and their derivatives:

$$
J[y_1, \dots, y_m] = \int_a^b F(x, y_1, \dots, y_m, y_1', \dots, y_m') \, dx
$$

To find the extremum, we consider variations $$\eta_1(x), \dots, \eta_m(x)$$ for each function $$y_i(x)$$. We perturb one function at a time, say $$y_k \to y_k + \epsilon \eta_k$$, while keeping other $$y_j$$ ($$j \neq k$$) fixed. The first variation $$\delta J$$ must be zero for variations in each $$y_k$$ independently.
This leads to a system of $$m$$ Euler-Lagrange equations, one for each dependent variable $$y_k(x)$$:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Euler-Lagrange Equations for Multiple Dependent Variables
</div>
For a functional $$J[y_1, \dots, y_m] = \int_a^b F(x, y_1, \dots, y_m, y_1', \dots, y_m') \, dx$$, the extremizing functions $$y_k(x)$$ must satisfy the system of equations:

$$
\frac{\partial F}{\partial y_k} - \frac{d}{dx} \left( \frac{\partial F}{\partial y_k'} \right) = 0 \quad \text{for each } k = 1, \dots, m
$$

</blockquote>
This is precisely what happens in Lagrangian mechanics for systems with multiple generalized coordinates $$q_k(t)$$.

## 3. Functionals with Multiple Independent Variables

Variational principles are not limited to functions of a single independent variable. We can also consider functionals of functions of multiple independent variables. For example, finding a surface $$u(x,y)$$ that minimizes some quantity over a domain $$\Omega \subset \mathbb{R}^2$$.
The functional might look like:

$$
J[u] = \iint_\Omega F(x, y, u, u_x, u_y) \, dx dy
$$

where $$u_x = \frac{\partial u}{\partial x}$$ and $$u_y = \frac{\partial u}{\partial y}$$.

The derivation of the Euler-Lagrange equation is analogous. We consider a variation $$\eta(x,y)$$ such that $$u \to u + \epsilon \eta$$. The variation $$\eta(x,y)$$ must vanish on the boundary $$\partial\Omega$$ of the domain $$\Omega$$.
Setting $$\delta J[u; \eta] = 0$$ and using a two-dimensional version of integration by parts (Green's theorem or divergence theorem), we arrive at:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Euler-Lagrange Equation for Multiple Independent Variables
</div>
For a functional $$J[u] = \iint_\Omega F(x, y, u, u_x, u_y) \, dx dy$$, the extremizing function $$u(x,y)$$ must satisfy the partial differential equation (PDE):

$$
\frac{\partial F}{\partial u} - \frac{\partial}{\partial x} \left( \frac{\partial F}{\partial u_x} \right) - \frac{\partial}{\partial y} \left( \frac{\partial F}{\partial u_y} \right) = 0
$$

</blockquote>
This can be generalized to more independent variables ($$x_1, x_2, \dots, x_n$$) and higher-order partial derivatives. Such equations form the basis of many field theories in physics (e.g., electromagnetism, general relativity) and are crucial in areas like image processing (e.g., minimizing energy functionals for image denoising or segmentation).

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Minimal Surface (Plateau's Problem)
</div>
Find the surface $$u(x,y)$$ spanning a given closed curve in 3D that has the minimum possible surface area. The area functional is:

$$
A[u] = \iint_\Omega \sqrt{1 + u_x^2 + u_y^2} \, dx dy
$$

Here, $$F(x,y,u,u_x,u_y) = \sqrt{1 + u_x^2 + u_y^2}$$.
-   $$\frac{\partial F}{\partial u} = 0$$
-   $$\frac{\partial F}{\partial u_x} = \frac{u_x}{\sqrt{1 + u_x^2 + u_y^2}}$$
-   $$\frac{\partial F}{\partial u_y} = \frac{u_y}{\sqrt{1 + u_x^2 + u_y^2}}$$
The Euler-Lagrange PDE is:

$$
0 - \frac{\partial}{\partial x} \left( \frac{u_x}{\sqrt{1 + u_x^2 + u_y^2}} \right) - \frac{\partial}{\partial y} \left( \frac{u_y}{\sqrt{1 + u_x^2 + u_y^2}} \right) = 0
$$

This is the **minimal surface equation**. Solutions (like soap films) have zero mean curvature.
</blockquote>

## 4. Constrained Variational Problems

Often, we want to extremize a functional $$J[y]$$ subject to one or more constraints. A common type is an **isoperimetric constraint**, where another functional $$K[y]$$ must take a fixed value:

$$
K[y] = \int_a^b G(x, y, y') \, dx = L_0 \quad (\text{constant})
$$

An example is Dido's problem: find the curve of a given fixed length $$L_0$$ that encloses the maximum area. Or the catenary problem: find the shape of a hanging chain of fixed length $$L_0$$ that minimizes potential energy.

Analogous to constrained optimization in multivariable calculus, we use the method of **Lagrange multipliers**. We form an auxiliary functional:

$$
J^\ast[y] = J[y] + \lambda K[y] = \int_a^b (F(x, y, y') + \lambda G(x, y, y')) \, dx
$$

$$
J^\ast[y] = \int_a^b H(x, y, y', \lambda) \, dx
$$

where $$H = F + \lambda G$$ and $$\lambda$$ is a Lagrange multiplier (a constant to be determined).
We then apply the Euler-Lagrange equation to this new functional $$J^\ast [y]$$ with respect to $$y(x)$$, treating $$\lambda$$ as a constant during this step:

$$
\frac{\partial H}{\partial y} - \frac{d}{dx} \left( \frac{\partial H}{\partial y'} \right) = 0
$$

This gives a differential equation that involves $$\lambda$$. The solution $$y(x, \lambda)$$ will depend on $$\lambda$$. Finally, $$\lambda$$ is determined by substituting this solution back into the constraint equation $$K[y(x, \lambda)] = L_0$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Dido's Problem
</div>
Maximize the area $$A[y] = \int_0^X y(x) \, dx$$ (assuming the x-axis is one boundary) subject to a fixed arc length $$L_0 = \int_0^X \sqrt{1+(y')^2} \, dx$$.
Here, $$F = y$$ and $$G = \sqrt{1+(y')^2}$$.
The augmented integrand is $$H = y + \lambda \sqrt{1+(y')^2}$$.
The Euler-Lagrange equation for $$H$$:
$$\frac{\partial H}{\partial y} = 1$$
$$\frac{\partial H}{\partial y'} = \frac{\lambda y'}{\sqrt{1+(y')^2}}$$
So, $$1 - \frac{d}{dx}\left(\frac{\lambda y'}{\sqrt{1+(y')^2}}\right) = 0$$.
Integrating: $$\frac{\lambda y'}{\sqrt{1+(y')^2}} = x - C_1$$.
Solving for $$y'$$: $$\lambda^2 (y')^2 = (x-C_1)^2 (1+(y')^2)$$.
$$(y')^2 (\lambda^2 - (x-C_1)^2) = (x-C_1)^2$$.
$$y' = \frac{\pm (x-C_1)}{\sqrt{\lambda^2 - (x-C_1)^2}}$$.
Integrating this gives $$(x-C_1)^2 + (y-C_2)^2 = \lambda^2$$, which is the equation of a circle.
The constants $$C_1, C_2, \lambda$$ are determined by the boundary conditions and the length constraint $$L_0$$. Thus, the curve enclosing maximum area for a fixed perimeter is a circular arc.
</blockquote>
Other types of constraints exist, such as algebraic constraints ($$g(x,y)=0$$ directly relating $$x$$ and $$y$$) or differential constraints ($$g(x,y,y')=0$$). These often require different techniques or can sometimes be incorporated by substitution.

## 5. Conclusion of the Crash Course (Part 1 of 2)

Over these five parts, we've journeyed from the basic definition of a functional to the powerful Euler-Lagrange equation and its many applications and generalizations.
-   We defined **functionals** and the problem of **variational calculus**: optimizing functions.
-   We introduced the **first variation** $$\delta J$$ as the "derivative" for functionals.
-   We derived the **Euler-Lagrange equation** $$\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0$$ as the necessary condition for extremizing $$J[y] = \int_a^b F(x, y, y') \, dx$$.
-   We explored its connection to **Lagrangian and Hamiltonian mechanics**, introducing the **Legendre transform** and its link to convexity.
-   We applied the Euler-Lagrange equation to solve **classic problems** like the shortest path and the brachistochrone, and discussed special cases like the **Beltrami identity**.
-   We looked at **generalizations** for higher-order derivatives, multiple functions, multiple independent variables (leading to PDEs), and briefly, **constrained problems**.

Variational calculus is a vast and beautiful field. Its principles extend far beyond these introductory examples, forming the foundation for understanding physical laws, optimal control theory, aspects of differential geometry, and providing tools and intuition for optimization problems in various domains, including machine learning.

For instance, many regularization techniques in machine learning (like Tikhonov regularization or smoothing splines) can be formulated as minimizing a functional that balances data fidelity and a smoothness penalty (e.g., penalizing large values of $$\int (f''(x))^2 dx$$). Path integrals in probabilistic models and the concept of "energy" in energy-based models also resonate with the ideas from variational calculus.

Understanding the calculus of variations gives us a powerful lens through which to view optimization problems where the "variables" are themselves functions or continuous fields. The next step in our broader journey through optimization will often involve discretizing these continuous problems or applying these foundational ideas to finite-dimensional but high-dimensional settings.

This concludes the core content of our crash course on Variational Calculus!

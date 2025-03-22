---
layout: post
title: "A Story of Optimization in ML: Chapter 6b – Formal Foundations: Manifolds, Tangent Spaces, Metrics, and Duality"
description: "Formal definitions of manifolds, tangent spaces, and metrics, and why gradients live in the dual space. A precise foundation for geometric optimization in machine learning."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["differential geometry", "manifolds", "tangent spaces", "dual space", "coordinate transformations", "natural gradient", "Riemannian metric"]
image:
  path: /assets/2025-03-07-optimization-in-machine-learning/formal_differential_geometry.png
  alt: "Formal structure of manifolds and tangent spaces for optimization"
date: 2025-03-07 04:00 +0000
math: true
---

# Chapter 6b: Formal Foundations – Manifolds, Tangent Spaces, Metrics, and Duality

In the previous post, we developed intuition:
- Gradients live in the dual (cotangent) space.
- In Euclidean space, inner products let us blur vectors and covectors.
- On curved spaces, this identification fails.
- Metrics, tangent spaces, and coordinate transformations matter deeply in machine learning.

Now, it’s time to formalize these ideas precisely.

---

## 1. Manifolds: Curved Spaces That Look Flat Locally

At its core, a **manifold** is a space that may be globally curved but **locally behaves like flat Euclidean space**.

### Why Are Manifolds Relevant in ML?

Many ML problems naturally involve manifolds:
- **Data manifolds:** Real-world high-dimensional data lies near low-dimensional, curved manifolds (the **manifold hypothesis**).
- **Constraint spaces:** Normalized vectors (e.g., softmax outputs), rotation matrices (SO(3)), orthogonal matrices—all form manifolds.
- **Latent spaces:** Often non-Euclidean, with geometric structure.

To optimize effectively in these spaces, we need manifold-aware tools.

---

### Formal Definition: Smooth Manifold

A smooth \( n \)-dimensional **manifold** \( \mathcal{M} \) has:

1. **Local Euclidean Structure:**  
   Around every point \( p \in \mathcal{M} \), there is a neighborhood \( U \) and a **chart**:
   \[
   \varphi: U \to \mathbb{R}^n
   \]
   which is a **homeomorphism** (continuous, invertible, with continuous inverse).

2. **Smoothness:**  
   If two charts overlap, the **transition map**:
   \[
   \varphi \circ \psi^{-1}: \mathbb{R}^n \to \mathbb{R}^n
   \]
   is **smooth** (infinitely differentiable).

---

### What Is a Chart?

A **chart** is like a **local coordinate system**.

**Analogy:**
- Think of a chart as a map of a small region of the Earth.
- The Earth is globally curved (a sphere), but locally flat—maps flatten small regions.
- Each chart "unwraps" part of the manifold into flat coordinates.

No single chart covers the whole sphere (just like no flat map covers the globe without distortion), but **locally, charts make the space look like \( \mathbb{R}^n \).**

---

### Transition Maps: Ensuring Smoothness

Where two charts overlap, we must convert between coordinate systems.

The **transition map**:
\[
\varphi \circ \psi^{-1}
\]
tells you how to move from one chart’s coordinates to another’s.

**Key requirement:**  
These maps must be smooth—no sharp jumps or discontinuities.  
This ensures we can smoothly do calculus across the entire manifold.

---

### Examples of Manifolds:

1. **Circle \( S^1 \):**
   - Defined by \( x^2 + y^2 = 1 \) in \( \mathbb{R}^2 \).
   - Locally behaves like \( \mathbb{R}^1 \).
   - Needs at least two charts to avoid coordinate singularities (e.g., near poles).

2. **Sphere \( S^2 \):**
   - Earth's surface is the classic 2D manifold.
   - Locally flat, globally curved.
   - Latitude/longitude charts cover most, but not all, of the sphere (poles require special treatment).

3. **Rotation Group SO(3):**
   - The set of all 3D rotation matrices.
   - Appears in robotics, 3D vision, and embedding spaces.
   - Locally behaves like \( \mathbb{R}^3 \), but has non-trivial global structure.

4. **Probability Simplex:**
   - The set of vectors \( (p_1, p_2, ..., p_n) \) with \( p_i \geq 0 \) and \( \sum p_i = 1 \).
   - Frequently used in ML (e.g., softmax outputs, categorical distributions).
   - A manifold embedded in \( \mathbb{R}^n \).

---

## 2. Tangent Spaces: Directions You Can Move

At each point on a manifold, we need a way to formalize **directions you can move without leaving the manifold**. That’s the role of the **tangent space**.

---

### Definition: Tangent Space

For each point \( p \in \mathcal{M} \), the **tangent space** \( T_p\mathcal{M} \) is a vector space consisting of all possible directions at \( p \).

There are two equivalent formal definitions:

---

### A) Tangent Vectors as Velocities of Curves

Given a smooth curve \( \gamma(t) \) passing through \( p \):
\[
\gamma(0) = p,
\]
we define the **tangent vector** at \( p \) as:
\[
v = \left. \frac{d\gamma}{dt} \right|_{t=0}.
\]
Two curves are equivalent if their derivatives at \( t = 0 \) are equal.

---

### B) Tangent Vectors as Derivations

A more abstract (but powerful) definition:

> A tangent vector at \( p \) is a linear map:
> \[
> v: C^\infty(\mathcal{M}) \to \mathbb{R}
> \]
> satisfying:
> - **Linearity:**
> \[
> v(af + bg) = a v(f) + b v(g),
> \]
> - **Leibniz rule:**
> \[
> v(fg) = f(p) v(g) + g(p) v(f).
> \]

This definition views vectors as **directional derivatives**—they differentiate functions.

---

### Coordinate Basis:

Given a chart \( \varphi \), we can form basis vectors:
\[
\frac{\partial}{\partial x^i} \quad \text{at point} \quad p.
\]
Any tangent vector can be written:
\[
v = v^i \frac{\partial}{\partial x^i}.
\]

---

## 3. Cotangent Spaces: Linear Functionals (Covectors)

The **cotangent space** \( T_p^*\mathcal{M} \) is the **dual space** to \( T_p\mathcal{M} \).

**Definition:**
\[
T_p^*\mathcal{M} = \{ \omega: T_p\mathcal{M} \to \mathbb{R} \quad \text{linear} \}.
\]

These are **covectors**: they take in vectors and output real numbers.

---

### Basis of Cotangent Space:

The natural basis is:
\[
dx^i,
\]
defined by:
\[
dx^i \left( \frac{\partial}{\partial x^j} \right) = \delta^i_j.
\]

Any covector:
\[
\omega = \omega_i dx^i.
\]

It acts on a vector \( v = v^j \frac{\partial}{\partial x^j} \) via:
\[
\omega(v) = \omega_i v^i.
\]

---

## 4. Differentials of Functions: Gradients as Covectors

Given a smooth function \( f: \mathcal{M} \to \mathbb{R} \), its **differential** at point \( p \) is:
\[
df_p: T_p\mathcal{M} \to \mathbb{R}.
\]

It’s defined by:
\[
df_p(v) = v(f).
\]

In coordinates:
\[
df = \frac{\partial f}{\partial x^i} dx^i.
\]

Thus, **the differential is a covector**—it takes in a vector (direction) and outputs how fast \( f \) changes in that direction.

---

## 5. The Metric: Relating Vectors and Covectors

In Euclidean space, we identify vectors and covectors via the dot product:
\[
\langle v, w \rangle = v^i w^i.
\]
This blurs the distinction.

But on manifolds, there is no global dot product.  
To connect vectors and covectors, we introduce a **Riemannian metric**.

---

### Definition: Riemannian Metric

A **Riemannian metric** assigns, at each point \( p \in \mathcal{M} \), an inner product:
\[
\langle v, w \rangle_p = g_{ij}(p) v^i w^j.
\]

**Properties:**
- **Symmetric:** \( g_{ij} = g_{ji} \).
- **Positive-definite.**
- **Smoothly varying.**

---

### Raising and Lowering Indices:

Using the metric:
- **Lower indices (vector → covector):**
\[
v_i = g_{ij} v^j.
\]
- **Raise indices (covector → vector):**
\[
v^i = g^{ij} v_j,
\]
where \( g^{ij} \) is the inverse of \( g_{ij} \).

---

## 6. Coordinate Transformations: Behavior of Vectors, Covectors, and Metrics

When we change coordinates:

- **Vectors transform:**  
  Via the Jacobian \( \frac{\partial \tilde{x}^i}{\partial x^j} \).

- **Covectors transform:**  
  Via the **transpose inverse** of the Jacobian.

- **Metrics adjust:**  
  To preserve inner products:
  \[
  \tilde{g}_{kl} = \frac{\partial x^i}{\partial \tilde{x}^k} \frac{\partial x^j}{\partial \tilde{x}^l} g_{ij}.
  \]

This is key:  
**Geometry cares about invariant relationships, not specific coordinates.**

---

## 7. Gradients Live in the Cotangent Space

Bringing it all together:

- The gradient of \( f \) is the differential \( df \)—a covector.
- To step in a valid direction, we convert to a vector:
\[
(\nabla f)^i = g^{ij} \frac{\partial f}{\partial x^j}.
\]

This is essential in optimization:
- On curved spaces,
- Under coordinate transformations,
- For respecting manifold structure (e.g., in natural gradients).

---

## 8. Summary: Formal Tools for Geometric Optimization

We’ve formalized:

- **Manifolds:** Locally Euclidean, globally curved spaces.
- **Tangent spaces:** Valid directions to move at each point.
- **Cotangent spaces:** Linear functionals measuring vectors.
- **Differentials:** Gradients as covectors.
- **Riemannian metrics:** Relating vectors and covectors.
- **Coordinate transformations:** How everything behaves under change of coordinates.

These concepts form the mathematical bedrock for **optimization on manifolds**, crucial in many ML contexts.

---

## What’s Next:

Having formalized the foundation, we’re ready to:

- Explore **natural gradients** and parameterization-invariant optimization.
- Derive **geodesics** (generalizing straight lines).
- Apply these ideas to concrete ML problems.

Stay tuned!

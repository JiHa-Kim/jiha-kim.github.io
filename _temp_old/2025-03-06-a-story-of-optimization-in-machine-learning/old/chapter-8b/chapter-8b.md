---
layout: post
title: "A Story of Optimization in ML: Chapter 8b – Formal Foundations: Manifolds, Tangent Spaces, Metrics, and Duality"
description: "Formal definitions of manifolds, tangent spaces, and metrics, and why gradients live in the dual space. A precise foundation for geometric optimization in machine learning."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["differential geometry", "manifolds", "tangent spaces", "dual space", "coordinate transformations", "natural gradient", "Riemannian metric"]
image:
  path: /assets/2025-03-07-optimization-in-machine-learning/formal_differential_geometry.png
  alt: "Formal structure of manifolds and tangent spaces for optimization"
date: 2025-03-07 04:00 +0000
math: true
---

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

At each point on a manifold, we want to formalize the idea of **directions you can move without leaving the manifold**. These directions form a **vector space** called the **tangent space**.

---

### 2.1. Geometric Intuition: Tangent Vectors as Velocity Vectors

Imagine walking on the surface of the Earth (a sphere).  
Locally, you can move north-south, east-west—but you can't leave the surface.

At any point:
- The **tangent plane** is the flat plane touching the surface.
- Possible movement directions lie in this plane.

---

### Example: Circle \( S^1 \)

Consider the unit circle:
\[
S^1 = \{ (x, y) \in \mathbb{R}^2 : x^2 + y^2 = 1 \}.
\]

Define a curve:
\[
\gamma(t) = (\cos t, \sin t).
\]

- At \( t = 0 \), \( \gamma(0) = (1, 0) \).
- The derivative:
\[
\frac{d\gamma}{dt}\Big|_{t=0} = (-\sin(0), \cos(0)) = (0, 1).
\]

Interpretation:
- The tangent vector at \( (1, 0) \) is \( (0, 1) \)—it points **upwards**, tangent to the circle.
- It's the **velocity vector** of the curve at \( t = 0 \).

---

### Formal Definition: Tangent Vectors as Curves

At any point \( p \in \mathcal{M} \), we can define:

> The **tangent vector** at \( p \) is the derivative of a smooth curve \( \gamma(t) \) with \( \gamma(0) = p \).

---

### 2.2. Tangent Vectors as Derivations: The Analytical Perspective

There’s a second, more formal definition, deeply connected to calculus:

### Definition:

A **tangent vector at \( p \)** is a linear operator:
\[
v: C^\infty(\mathcal{M}) \to \mathbb{R},
\]
satisfying:
- **Linearity:**  
\[
v(af + bg) = a v(f) + b v(g).
\]
- **Leibniz Rule:**  
\[
v(fg) = f(p) v(g) + g(p) v(f).
\]

---

### Why This Makes Sense:

Think of how **directional derivatives** behave:

In \( \mathbb{R}^n \):
\[
D_v f = \sum_i v^i \frac{\partial f}{\partial x^i}.
\]

- It’s **linear** in \( f \).
- It satisfies the product rule (Leibniz rule).

Thus:
- **Directional derivatives = derivations.**
- Every tangent vector gives you a way to differentiate smooth functions at \( p \).

---

### Example: Tangent Vectors in \( \mathbb{R}^2 \)

At point \( p = (1, 0) \), consider the vector:
\[
v = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
\]

It acts on functions \( f(x, y) \) by:
\[
v(f) = \frac{\partial f}{\partial y}(1, 0).
\]

This:
- Is **linear in \( f \)**.
- Satisfies the product rule.

Therefore, tangent vectors **are equivalent to derivations** at \( p \).

---

### Connection to Charts & Coordinates:

Manifolds are locally like \( \mathbb{R}^n \).

Given a **chart** \( (U, \varphi) \):
- The coordinates \( x^i = \varphi^i(p) \) act like standard \( \mathbb{R}^n \) coordinates.

This gives natural basis vectors:
\[
\frac{\partial}{\partial x^i},
\]
which act on functions by taking partial derivatives:
\[
\frac{\partial}{\partial x^i}(f) = \frac{\partial f}{\partial x^i}(p).
\]

Thus, any tangent vector at \( p \) can be written:
\[
v = v^i \frac{\partial}{\partial x^i}.
\]

Here, we use the Einstein summation convention (summing over repeated indices). This happens when there are contravariant components like \( v^i \) and covariant components like \( v_i \) or \( \dfrac{\partial}{\partial x^i} \).

---

### 2.3. More Concrete Examples:

#### Example: Sphere \( S^2 \)

Consider \( S^2 \) embedded in \( \mathbb{R}^3 \):
\[
x^2 + y^2 + z^2 = 1.
\]

At point \( (0, 0, 1) \), valid tangent vectors lie in the **plane \( z = 1 \)**:
- They are of the form \( (v^1, v^2, 0) \).
  
**Interpretation:**
- Moving along latitude or longitude at the North Pole.
- Tangent space is a 2D plane sitting in \( \mathbb{R}^3 \).

---

#### Example: Tangent Vectors on the Probability Simplex

The probability simplex in \( \mathbb{R}^n \):
\[
\Delta^{n-1} = \left\{ (p_1, ..., p_n) \mid p_i \geq 0, \sum p_i = 1 \right\}.
\]

At any point:
- Valid tangent vectors must satisfy:
\[
\sum_{i=1}^{n} v^i = 0.
\]
This constraint keeps you on the simplex (since total probability must remain 1).

---

## 3. Cotangent Spaces: Linear Functionals (Covectors)

So far, we've seen that **tangent vectors** at a point \( p \) on a manifold describe valid directions you can move locally.

Now, we introduce the **cotangent space**, which consists of objects that act **on vectors** to produce real numbers.

---

### Definition: Cotangent Space

The **cotangent space** \( T_p^*\mathcal{M} \) at point \( p \) is the **dual space** to the tangent space:
\[
T_p^*\mathcal{M} = \{ \omega: T_p\mathcal{M} \to \mathbb{R} \quad \text{linear} \}.
\]
Its elements are called **covectors** (or one-forms).

They take in a tangent vector and return a real number.

---

### Basis of Cotangent Space:

In local coordinates, we already have basis vectors for the tangent space:
\[
\frac{\partial}{\partial x^i}.
\]

The dual basis for the cotangent space is denoted:
\[
dx^i,
\]
defined by:
\[
dx^i \left( \frac{\partial}{\partial x^j} \right) = \delta^i_j.
\]

Thus, any covector at \( p \) can be written:
\[
\omega = \omega_i dx^i.
\]

It acts on a vector \( v = v^j \frac{\partial}{\partial x^j} \) via:
\[
\omega(v) = \omega_i v^i.
\]

---

### Example:

In \( \mathbb{R}^2 \), if:
\[
v = \begin{pmatrix} 1 \\ 2 \end{pmatrix},
\quad
\omega = (3, 4),
\]
then:
\[
\omega(v) = 3 \times 1 + 4 \times 2 = 11.
\]

Covectors "measure" vectors.

---

## 4. Differentials of Functions: Gradients as Covectors

Let’s connect this to gradients.

Given a smooth function \( f: \mathcal{M} \to \mathbb{R} \), the **differential** \( df \) at \( p \) is:

\[
df_p: T_p\mathcal{M} \to \mathbb{R},
\]
defined by:
\[
df_p(v) = v(f).
\]

That is, given a direction (vector) \( v \), it tells you how fast \( f \) increases along \( v \).

---

### Coordinate Representation:

In local coordinates:
\[
df = \frac{\partial f}{\partial x^i} dx^i.
\]

Notice:
- \( df \) is a **covector** (a one-form).
- It lives in the cotangent space.
- It "measures" vectors by computing directional derivatives.

---

### Example:

In \( \mathbb{R}^2 \), for:
\[
f(x, y) = x^2 + y^2,
\]
we compute:
\[
df = 2x \, dx + 2y \, dy.
\]

At \( (1, 0) \), this gives:
\[
df = 2 \times 1 \, dx = 2 \, dx.
\]
Acting on \( v = (v^1, v^2) \), we get:
\[
df(v) = 2v^1.
\]

---

## 5. The Metric: Relating Vectors and Covectors

In Euclidean space, the dot product identifies vectors and covectors:
\[
\langle v, w \rangle = v^i w^i.
\]
But on a general manifold, no natural dot product exists globally.

To bridge vectors and covectors, we introduce the **Riemannian metric**.

---

### Definition: Riemannian Metric

A **Riemannian metric** assigns, at each point \( p \in \mathcal{M} \), an inner product:
\[
\langle v, w \rangle_p = g_{ij}(p) v^i w^j.
\]

**Properties:**
- Symmetric: \( g_{ij} = g_{ji} \).
- Positive-definite.
- Smoothly varying across \( \mathcal{M} \).

---

### Raising and Lowering Indices:

Using the metric:

- **Lowering (vector → covector):**
\[
v_i = g_{ij} v^j.
\]

- **Raising (covector → vector):**
\[
v^i = g^{ij} v_j,
\]
where \( g^{ij} \) is the inverse of \( g_{ij} \).

---

### Why This Matters:

Given a differential \( df \) (a covector), we use the metric to convert it into a **tangent vector**—the direction to move in optimization.

---

### Example:

Consider a metric in \( \mathbb{R}^2 \):
\[
g_{ij} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}.
\]

Given \( df = (4, 5) \), the gradient vector is:
\[
(\nabla f)^i = g^{ij} \frac{\partial f}{\partial x^j}.
\]

Computing:
\[
g^{ij} = \begin{pmatrix} 0.5 & 0 \\ 0 & \frac{1}{3} \end{pmatrix},
\]
so:
\[
\nabla f = \begin{pmatrix} 0.5 \times 4 \\ \frac{1}{3} \times 5 \end{pmatrix} = \begin{pmatrix} 2 \\ \frac{5}{3} \end{pmatrix}.
\]

---

## 6. Coordinate Transformations: Behavior of Vectors, Covectors, and Metrics

Manifolds are about **coordinate freedom**. What happens when we change coordinates?

### Vectors:
Transform via the Jacobian:
\[
\tilde{v}^i = \frac{\partial \tilde{x}^i}{\partial x^j} v^j.
\]

### Covectors:
Transform via the **transpose inverse**:
\[
\tilde{\omega}_i = \frac{\partial x^j}{\partial \tilde{x}^i} \omega_j.
\]

### Metrics:
Adjust to preserve inner products:
\[
\tilde{g}_{kl} = \frac{\partial x^i}{\partial \tilde{x}^k} \frac{\partial x^j}{\partial \tilde{x}^l} g_{ij}.
\]

**Key Insight:**
- Geometry focuses on **invariant relationships**, not specific coordinates.
- This is why naive gradient descent fails under reparameterization, but geometric methods (natural gradients) succeed.

---

## 7. Gradients Live in the Cotangent Space

To summarize:

- The **differential \( df \)** is naturally a covector.
- To compute valid directions to move (vectors), we apply the metric:
\[
(\nabla f)^i = g^{ij} \frac{\partial f}{\partial x^j}.
\]

---

## 8. Summary: Formal Tools for Geometric Optimization

We’ve formalized:

- **Manifolds:** Locally Euclidean, globally curved spaces.
- **Tangent spaces:** Valid directions to move at each point.
- **Cotangent spaces:** Linear functionals measuring vectors.
- **Differentials:** Gradients as covectors.
- **Riemannian metrics:** Relating vectors and covectors.
- **Coordinate transformations:** Predictable behavior under change of variables.

These concepts form the mathematical bedrock for **optimization on manifolds**, crucial in many ML contexts.

---

## 9. Exercises

Here are some exercises to help consolidate the formal concepts introduced:

### 1. **Tangent Vectors as Derivatives**

Given the circle \( S^1 = \{ (x, y) \in \mathbb{R}^2 : x^2 + y^2 = 1 \} \), define a curve:
\[
\gamma(t) = (\cos t, \sin t).
\]
Compute the tangent vector at \( t = 0 \) and interpret it geometrically.

---

### 2. **Cotangent Space Duality**

Given the standard basis vectors \( \frac{\partial}{\partial x^i} \) in \( \mathbb{R}^2 \), write down the dual basis covectors \( dx^1, dx^2 \).

Verify explicitly that:
\[
dx^i \left( \frac{\partial}{\partial x^j} \right) = \delta^i_j.
\]

---

### 3. **Differential of a Function**

Consider \( f(x, y) = x^2 + y^2 \) on \( \mathbb{R}^2 \).

- Compute the differential \( df \).
- Apply it to a general vector \( v = v^1 \frac{\partial}{\partial x} + v^2 \frac{\partial}{\partial y} \).
- Show explicitly how \( df(v) \) computes the directional derivative.

---

### 4. **Metric Lowering Indices**

Given a non-standard metric:
\[
g_{ij} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix},
\]
and a vector:
\[
v = \begin{pmatrix} 1 \\ 2 \end{pmatrix},
\]
compute the covector \( v_i = g_{ij} v^j \).

---

### 5. **Coordinate Transformation Behavior**

Consider the coordinate transformation:
\[
\tilde{x}^1 = 2x^1, \quad \tilde{x}^2 = x^2.
\]

- Write down how a vector \( v^i \) transforms.
- Write down how a covector \( \omega_i \) transforms.
- Verify that the inner product \( \omega_i v^i \) remains invariant.

---

### 6. **Gradient Descent Step on a Manifold**

Consider minimizing \( f(x, y) = x^2 + 4y^2 \) on \( \mathbb{R}^2 \) equipped with the metric:
\[
g_{ij} = \begin{pmatrix} 1 & 0 \\ 0 & 4 \end{pmatrix}.
\]

Compute:

1. The differential \( df \).
2. The gradient vector (after applying the metric).
3. Explain why the naive Euclidean gradient descent step behaves differently from the step using this metric.

---

## What’s Next:

Having formalized the foundations, we’re now ready to:
- Explore **natural gradients** and parameterization-invariant optimization.
- Dive into **geodesics** (generalizing straight lines).
- Apply these tools to practical ML problems.

Stay tuned!

## Further Reading

- [eigenchris' YouTube playlist on tensors for beginners](https://www.youtube.com/watch?v=8ptMTLzV4-I&list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG)
- [eigenchris' YouTube playlist on tensor calculus](https://www.youtube.com/watch?v=kGXr1SF3WmA&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)
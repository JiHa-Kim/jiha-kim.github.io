---
layout: post
title: "A Story of Optimization in ML: Chapter 8a — Vectors, Covectors, and Why Gradients Aren’t What You Think"
description: "Why gradients live in the dual space, how inner products blur the distinction in Euclidean space, and why coordinate transformations in machine learning demand geometric thinking."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["differential geometry", "manifolds", "tangent spaces", "dual space", "inner product", "coordinate transformations", "natural gradient"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/vector_covector_duality.png
  alt: "Visual metaphor for vectors and covectors and their dual relationship"
date: 2025-03-06 04:00 +0000
math: true
---

In machine learning, optimization often feels routine:  
Compute gradients, step downhill, repeat.

But behind the scenes, something subtle is happening. Gradients don’t live where we think they do. To see why, we need to start from something familiar: **linear algebra**.

---

## 1. Motivation: Why Should We Care About Geometry?

One of the guiding principles in ML is the **manifold hypothesis**:
> High-dimensional data (like images, speech, or embeddings) often lies near a much lower-dimensional manifold inside the ambient space.

That’s not just a philosophical point. It affects:
- How optimization behaves.
- How gradients should be interpreted.
- How coordinate transformations (e.g., normalization layers, reparameterizations) impact learning.

Naive gradient descent assumes we’re optimizing in flat, uniform \( \mathbb{R}^n \). But the real world—and real data—lives on curved, constrained spaces.

---

## 2. Vectors and Covectors: Back to Linear Algebra Basics

Let’s revisit concepts you already know:

### Vectors:

Think of a vector space \( V \) over \( \mathbb{R} \).  
Vectors here are familiar objects:
\[
v = \begin{pmatrix} v^1 \\ v^2 \\ \vdots \\ v^n \end{pmatrix}.
\]
They are arrows—directions in space.

---

### Covectors (Dual Vectors):

The **dual space** \( V^* \) consists of **linear functionals**:
\[
\omega: V \to \mathbb{R}.
\]
Given a vector \( v \), the covector returns a number:
\[
\omega(v) = \omega_i v^i.
\]

In matrix terms:
- Vectors = column vectors.
- Covectors = row vectors (transpose).

---

## 3. Inner Products Give an Isomorphism Between Vectors and Covectors

In flat \( \mathbb{R}^n \), we have the dot product:
\[
\langle v, w \rangle = v^i w^i.
\]

This inner product **lets us turn a vector into a covector**:
\[
v^*(\cdot) = \langle v, \cdot \rangle.
\]
Or in matrix language:
\[
v^* = v^\top.
\]

Thus:
- Vectors and covectors are *identified* via the inner product.
- It feels natural to treat them interchangeably.

---

## 4. The Euclidean Illusion: Why We Confuse Vectors and Covectors

In linear algebra and ML:
- Gradients are computed as vectors.
- Steps in gradient descent are vectors.

But technically:
- The **gradient of a function is a covector**—it maps directions (vectors) to rates of change (scalars).

In Euclidean space, the dot product blurs this distinction:
\[
\text{Gradient (row vector)} = \text{Direction (column vector)}^\top.
\]

However, this is **coordinate-dependent**:
- Change the basis or rescale axes → dot product, and thus identification, changes.

---

## 5. Breaking Point: Curved Spaces & Coordinate Changes

Real-world data, and many ML models, live in **curved, constrained spaces**:
- Spheres (normalization constraints),
- Probability simplices (softmax outputs),
- Rotation matrices (SO(3)),
- Embedding spaces with non-trivial geometry.

In these spaces:
- There’s **no global, fixed inner product**.
- **No global way to transpose vectors and covectors.**

Additionally, when we **reparameterize variables** (common in ML), naive gradient descent struggles because:
- The step direction depends heavily on the coordinate choice.
- The underlying geometry isn't being respected.

---

## 6. Fixing the Problem: Introducing the Metric

To properly connect vectors and covectors in curved spaces, we need extra structure:  
A **Riemannian metric**.

### The Metric:

A metric defines an inner product at **each point**:
\[
\langle v, w \rangle = g_{ij}(x) v^i w^j.
\]

This allows:
- **Lowering indices (vector → covector):**
\[
v_i = g_{ij} v^j.
\]
- **Raising indices (covector → vector):**
\[
v^i = g^{ij} v_j.
\]
where \( g^{ij} \) is the inverse of \( g_{ij} \).

---

## 7. Gradients Live in the Dual Space

Let’s return to gradients.

Given a function \( f \):
\[
df = \frac{\partial f}{\partial x^i} dx^i.
\]
This is a **covector** (differential):
- It takes a vector (direction) and tells you how fast \( f \) increases in that direction.

**To move in that direction:**
- We must convert \( df \) into a vector.
- **The metric provides the bridge.**

---

## 8. Coordinate Transformations: Why They Matter in ML

Now, here’s something ML practitioners encounter daily:

**Coordinate changes.**

Examples:
- Scaling variables.
- Applying batch normalization.
- Reparameterizing latent variables.
- Normalizing embeddings.

Naive gradient descent **is not invariant** under these changes:
- Gradient directions change unpredictably.
- Optimization slows down or behaves poorly.

But in differential geometry:
- **Vectors and covectors transform predictably under coordinate changes.**
- The metric adjusts inner products, ensuring consistency.

This is precisely why **natural gradient methods** (Amari) adapt optimization to the manifold’s geometry—they account for these transformations explicitly.

---

## 9. A Simple Example: Reparameterization Trouble

Consider optimizing:
\[
f(x, y) = x^2 + 100 y^2.
\]

Naive gradient descent struggles:
- Steps in \( y \)-direction are tiny (ill-conditioning).

Now, change variables:
\[
\tilde{x} = x, \quad \tilde{y} = 10 y.
\]
Function becomes:
\[
f(\tilde{x}, \tilde{y}) = \tilde{x}^2 + \tilde{y}^2.
\]
In the new coordinates, optimization improves.

**But the gradient transformation isn’t automatic.**

This exposes how:
- Step directions depend heavily on chosen coordinates.
- Ignoring geometry leads to inefficiencies.

---

## 10. Summary: Geometry Guides Optimization

- **Vectors = directions.**
- **Covectors = rulers (linear functionals).**
- In \( \mathbb{R}^n \), dot products blur the distinction (transpose trick).
- On manifolds (curved spaces), no global inner product exists.
- The **metric bridges vectors and covectors locally**.
- Respecting this structure is essential for:
  - Efficient optimization,
  - Reparameterization-invariant methods (natural gradient, Riemannian optimization),
  - ML tasks involving constrained or non-Euclidean parameter spaces.

---

## 11. What’s Next: Cleaning Up the Formalism

In this post, we stayed in intuitive, linear algebra territory.

But we can be much more precise.

In the next chapter, we’ll:
- Formally define **manifolds** (charts, atlases, smoothness).
- Define **tangent and cotangent spaces** rigorously.
- Introduce **Riemannian metrics** properly.
- Show exactly how **coordinate transformations behave**.
- Lay the mathematical foundation for natural gradient methods and geometric optimization.

Stay tuned for **Chapter 6b: Formal Foundations – Manifolds, Tangent Spaces, and Duality**.

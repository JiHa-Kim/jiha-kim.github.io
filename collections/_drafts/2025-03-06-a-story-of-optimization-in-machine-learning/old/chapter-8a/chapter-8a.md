---
layout: post
title: "A Story of Optimization in ML: Chapter 8a — Pencils, Rulers, and Why Gradients Aren’t What You Think"
description: "Gradients live in the dual space, but why? This post introduces vectors and covectors through the physical metaphor of pencils and rulers, leading to metrics, coordinate transformations, and why geometry matters in machine learning optimization."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["differential geometry", "manifolds", "tangent spaces", "dual space", "coordinate transformations", "natural gradient"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/vector_covector_duality.png
  alt: "Visual metaphor for vectors and covectors and their dual relationship"
date: 2025-03-06 04:00 +0000
math: true
---

# Chapter 8a: Pencils, Rulers, and Why Gradients Aren’t What You Think

In machine learning, optimization feels routine:  
Compute gradients, step downhill, repeat.

But behind the scenes, something subtle is happening:  
**Gradients aren’t vectors in the usual sense.**
They don’t naturally point in a direction until we apply a metric to interpret them.

They live in a different space entirely: the **dual space**.

To understand why, we’ll start with a physical analogy: pencils and rulers.

---

## 1. Vectors Are Pencils, Covectors Are Rulers

Here’s the key intuition:

- **Vectors are pencils.**  
  You can imagine them as arrows pointing in some direction—you move along them.

- **Covectors are rulers.**  
  They **measure** how much of a vector points in a certain direction.
  They have grid lines perpendicular to the direction they measure.

### Visualization:

Picture yourself standing on a flat plane:
- You hold a pencil pointing northeast → **that’s a vector**.
- You lay down a ruler pointing north → the ruler **measures how much of your pencil points north**.

A covector (ruler) takes any vector (pencil) and returns **a number**:
> “How much of this pencil lies along me?”

---

## 2. Formalizing Vectors and Covectors

Let’s make this precise.

- A **vector space** \( V \) (think \( \mathbb{R}^n \)) consists of vectors—pencils.
  
- The **dual space** \( V^* \) consists of **covectors**—rulers.

A covector is a **linear function** that eats a vector and returns a real number:
\[
\omega(v) = \text{scalar}.
\]

In \( \mathbb{R}^n \):
- Vectors are column vectors.
- Covectors are row vectors.

**Important:**  
- Covectors live in a different space than vectors, even though they’re related.

---

## 3. Why They Feel Identical in Euclidean Space: The Inner Product

In standard \( \mathbb{R}^n \), we have a familiar tool: the **dot product**:
\[
\langle v, w \rangle = v^1 w^1 + v^2 w^2 + \dots + v^n w^n.
\]

This dot product does something sneaky:

- Given a vector \( v \), we can **turn it into a covector**:
\[
v^*(\cdot) = \langle v, \cdot \rangle.
\]

In matrix terms:
- Vector = column vector.
- Covector = row vector (transpose).

In this flat space:
- Pencils (vectors) and rulers (covectors) can be identified effortlessly.
- So people casually treat gradients (covectors!) as vectors.

---

## 4. What’s Behind This Magic: The Metric

But what **allows us to equate vectors and covectors** like this?

**Answer:**  
> The dot product is a special case of something deeper: the **metric**.

### **What Is the Metric?**

A **metric** tells us:
- How long a vector is.
- How to measure angles.
- How to align rulers and pencils.

In Euclidean space:
- The metric is just the identity matrix.
- Everything’s flat and simple.

But **the metric is what lets us lower and raise indices:**
- It converts vectors ↔ covectors.

---

## 5. Covariance vs. Contravariance: How Vectors & Covectors Transform

Let’s pause here:

- **Basis vectors, as geometric objects, are said to be "covariant".**  
  They exist independently of coordinates—they're the same "pencil" no matter how you redraw the grid.
- **Basis covectors, on the other hand, are said to be "contravariant".**
  Think of it: a more crushed grid in a ruler makes you think the vector is longer. We say that its behavior varies **contrary to basis vectors**, i.e. contravariantly.
- **But vector components transform contravariantly.**  
  When you double the length of a basis pencil, then your pencil's perceived length is half as short. A vector’s components adjust in the **opposite way** to the basis vectors in order to preserve the meaning of the vector itself.
- **Covectors, by contrast, have components that transform covariantly.**  
  Their components scale or rotate **in the same way** as the basis vector transformations, ensuring they continue to measure vectors correctly.

This distinction is crucial:
- **Pencils and rulers transform differently under coordinate changes.**
- Their pairing—the covector acting on the vector—remains invariant, no matter how the coordinates shift.
- In Euclidean space, the identity metric hides this difference, letting us casually conflate them.

See [eigenchris' playlist](https://www.youtube.com/watch?v=d5da-mcVJ20) for a more detailed explanation.

---

## 6. Where Trouble Starts: Curved Spaces, Reparameterizations

What happens when:
- You’re optimizing over a sphere (Earth)?
- Your data lives on a probability simplex (softmax outputs)?
- You apply normalization or scaling in ML?

Now:
- There’s no global, fixed dot product.
- **No natural way to identify pencils and rulers.**
- Vectors and covectors **transform differently**—and the naive transpose trick breaks down.

---

## 7. The Sphere Example: Concrete Breakdown

### **7.1 The Manifold: Earth as a Sphere**

Use spherical coordinates:
- \( \theta = \) latitude
- \( \phi = \) longitude

Locally:
- Valid movement directions = tangent vectors = pencils (east-west, north-south).

---

### **7.2 A Simple Function: Temperature**

Define:
\[
f(\theta, \phi) = \cos(\theta).
\]

- Hottest at equator, coldest at poles.

---

### **7.3 Compute the Differential**

The **differential** (gradient as covector) is:
\[
df = -\sin(\theta) \, d\theta.
\]

This is a ruler:
- Measures how temperature changes in any direction.

---

### **7.4 The Metric of the Sphere**

The sphere has a non-trivial metric:
\[
g = 
\begin{pmatrix} 
1 & 0 \\ 
0 & \sin^2(\theta) 
\end{pmatrix}.
\]

- Moving east-west depends on latitude.
- Near the poles, circles shrink → geometry changes.

---

### **7.5 Why the Naive Gradient Isn't a Valid Direction**

If you naively treat \( df \) as a vector:
- You ignore that east-west directions shrink near poles.
- Steps become distorted—might "step off the sphere."

---

### **7.6 How the Metric Fixes It**

To get a valid **gradient vector**:
\[
\nabla f = g^{-1} df.
\]

Explicitly:
- Apply the metric inverse to convert the covector into a vector.
- The gradient vector now respects the sphere's curvature.

---

## 8. Broader ML Relevance: Where Else This Breaks

In ML, similar problems arise:
- **Normalized vectors, softmax outputs, rotation matrices** → constrained, curved spaces.
- **Batch normalization, reparameterization** → change coordinates.
  
**Naive gradient descent ignores these transformations.**

---

## 9. Summary: The Metric Was Always There

- In flat space, the metric = identity → hides complexity.
- In curved spaces or after coordinate changes:
  - Vectors (pencils) and covectors (rulers) transform differently.
  - You **need the metric** to bridge them correctly.
  
Failing to account for this leads to inefficient or even invalid optimization.

---

## 10. What’s Next: Cleaning Up the Formalism

In this post, we built physical intuition:
- Vectors and covectors, pencils and rulers.
- Why they feel identical in flat space.
- Why metrics matter when things curve.

Next, we’ll formalize:
- Manifolds, charts, tangent and cotangent spaces.
- Metrics, coordinate transformations.
- And fully justify everything we computed here!

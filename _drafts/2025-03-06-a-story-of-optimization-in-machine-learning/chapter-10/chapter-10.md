---
layout: post
title: "A Story of Optimization in ML: Chapter 10 - Reflections in a Distorted Mirror: Duality and Beyond Euclidean Space"
description: "Chapter 10 extends our journey into optimization by exploring Mirror Descent through the lens of duality. Discover how mapping into a dual space via convex potentials transforms our approach to non-Euclidean optimization."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["mirror descent", "non-Euclidean optimization", "Bregman divergence", "duality"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/mirror_descent_chapter10_duality.gif
  alt: "A dual view of optimization: mapping between primal and dual spaces"
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 10: Reflections in a Distorted Mirror – Duality and Beyond Euclidean Space

In our exploration of optimization, we often assume that the landscape is flat and Euclidean. However, many practical problems reside in non-Euclidean spaces, where classical gradient methods can falter. **Mirror Descent** offers a powerful alternative by leveraging the concept of duality—recasting the optimization problem in a dual space where the geometry better reflects the problem’s inherent structure.

---

### 1. Embracing the Dual Perspective in Mirror Descent

Imagine transforming the problem from its original space (the *primal* space) into a mirror (or *dual*) space. By choosing a **strictly convex and differentiable** potential function \( \phi(x) \), we define the mirror map:
\[
\nabla \phi(x).
\]
This mapping translates our variables into a dual space where distances and directions are measured not by the standard Euclidean norm, but by a **Bregman divergence**:
\[
D_\phi(x \| y) = \phi(x) - \phi(y) - \langle \nabla \phi(y),\, x-y \rangle.
\]
This divergence captures the geometry of the problem and guides our update steps in a way that respects its natural structure.

---

### 2. The Math Behind Mirror Descent and Duality

Let’s break down the steps that reveal how duality shapes Mirror Descent:

#### **Step 1: Choosing a Mirror Map**
Select a potential function \( \phi(x) \) such that:
\[
\phi: \mathbb{R}^n \to \mathbb{R} \quad \text{is strictly convex and differentiable}.
\]
Its gradient \( \nabla \phi(x) \) acts as a bridge to the dual space.

#### **Step 2: Defining the Bregman Divergence**
Using the potential, we measure distance in the dual space via:
\[
D_\phi(x \| y) = \phi(x) - \phi(y) - \langle \nabla \phi(y),\, x-y \rangle.
\]
This divergence generalizes the squared Euclidean distance, allowing us to capture more complex geometries.

#### **Step 3: Gradient Step in the Dual Space**
Compute the gradient \( \nabla f(x_k) \) of the objective function in the primal space, and update in the dual space:
\[
\nabla \phi(x_k) - \eta\, \nabla f(x_k),
\]
where \( \eta \) is the step size. Here, the duality principle ensures that the descent direction is adjusted according to the geometry induced by \( \phi(x) \).

#### **Step 4: Mapping Back with Bregman Projection**
To return to the primal space, we perform a Bregman projection:
\[
x_{k+1} = \arg\min_{x \in C} \left\{ \eta\, \langle \nabla f(x_k),\, x \rangle + D_\phi(x \| x_k) \right\}.
\]
This step finds the best next iterate \( x_{k+1} \) that balances the descent direction with the geometry of the space.

---

### 3. Duality: The Bridge Between Primal and Dual Worlds

Duality in Mirror Descent does more than just provide a mathematical trick—it transforms how we view and solve optimization problems:

- **Natural Transformation:**  
  The mapping \( x \mapsto \nabla \phi(x) \) is analogous to computing the **Fenchel conjugate** of a function, a core concept in convex duality. This process reorients the problem, revealing hidden structure and simplifying analysis.

- **Recovery of Classical Methods:**  
  When we choose the Euclidean potential,
  \[
  \phi(x) = \frac{1}{2}\|x\|_2^2,
  \]
  the mirror map becomes the identity function, and the Bregman divergence reduces to:
  \[
  D_\phi(x \| y) = \frac{1}{2}\|x-y\|_2^2.
  \]
  In this case, Mirror Descent recovers the familiar **Projected Gradient Descent** update:
  \[
  x_{k+1} = \arg\min_{x \in C} \left\{ \eta\, \langle \nabla f(x_k),\, x \rangle + \frac{1}{2}\|x-x_k\|_2^2 \right\}.
  \]

- **Adapting to Non-Euclidean Geometries:**  
  For problems defined on the probability simplex or other curved spaces, choosing potentials like the negative entropy,
  \[
  \phi(x) = \sum_{i=1}^n x_i \log x_i,
  \]
  leads to Bregman divergences that mirror the Kullback-Leibler divergence. This enables **Exponentiated Gradient Descent**, naturally preserving constraints without the need for awkward Euclidean projections.

---

### 4. Exercises and Further Thoughts

To deepen your understanding of duality in Mirror Descent, consider the following exercises:

1. **Derive the Euclidean Case:**  
   Show that for 
   \[
   \phi(x) = \frac{1}{2}\|x\|_2^2,
   \]
   the Bregman divergence is the squared Euclidean distance and that the Mirror Descent update simplifies to the standard Projected Gradient Descent rule.

2. **Explore Negative Entropy:**  
   For the negative entropy potential on the probability simplex, derive the mirror map and corresponding Bregman divergence. Use these to obtain the Exponentiated Gradient Descent update.

3. **Connecting to Fenchel Conjugacy:**  
   Explain how the mirror map \( \nabla \phi(x) \) relates to the Fenchel conjugate of \( \phi \), and discuss the geometric intuition behind this connection.

---

### 5. Conclusion

Duality is not merely a theoretical construct—it is the foundation that allows Mirror Descent to thrive in non-Euclidean spaces. By transforming the optimization problem into a dual space, we obtain a more natural measure of distance and direction. This insight unifies classical gradient methods and paves the way for more sophisticated algorithms that respect the intrinsic geometry of modern machine learning problems.

In the next chapter, we will build on these duality principles to explore natural gradients and information geometry, deepening our understanding of how geometry shapes optimization.

Stay tuned, and happy optimizing!

### Further Reading
- [Harvey (2018) - Machine Learning Theory Lecture 20: Mirror Descent](https://www.cs.ubc.ca/~nickhar/F18-531/Notes20.pdf)
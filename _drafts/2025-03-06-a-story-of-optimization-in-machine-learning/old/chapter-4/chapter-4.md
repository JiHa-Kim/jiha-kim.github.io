---
layout: post
title: "A Story of Optimization in ML: Chapter 4 - The Convex Perspective"
description: "Chapter 4 dives into the concrete foundations of convexity, offering specific examples of convex sets, functions, and duality concepts to ground our understanding of optimization."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["convexity", "non-convexity", "convex hull", "duality"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/convexity_chapter4.gif
  alt: "Visual representation of convexity in optimization"
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 4: The Convex Perspective

The loss landscape is often a **non-convex** shape, which might bring us to ask why we should even bother with convex optimization. In this chapter, we will explore the basics of convexity through concrete definitions and examples, and see how these ideas provide a lens to understand—and later tackle—the challenges of non-convex optimization.

Even though many real-world problems are messy and non-convex, the study of convex sets and convex functions gives us a solid, specific foundation that can simplify analysis and inspire practical algorithms.

---

### 1. What Is Convexity?

**Convexity** is a geometric property that makes optimization problems easier to analyze. Let’s look at concrete examples.

#### Convex Sets

A set \( C \subset \mathbb{R}^n \) is **convex** if, for any two points \( x, y \in C \), every point on the line segment connecting them is also in \( C \). Formally:
\[
\forall\, x,y \in C,\quad \forall\, \lambda \in [0,1]: \quad \lambda x + (1-\lambda)y \in C.
\]

**Concrete Example:**  
Consider the set \( C = \{ (x,y) \in \mathbb{R}^2 \mid x^2 + y^2 \leq 1 \} \), the unit disk. Pick any two points in this disk; the straight line between them will lie completely inside the disk, confirming that \( C \) is convex. On the other hand, any polygon that "dents inward" is not convex, since you can draw a line between two points that leaves the polygon.

#### Convex Functions

A function \( f: \mathbb{R}^n \to \mathbb{R} \) is **convex** if its domain is a convex set and for every \( x, y \) in its domain and every \( \lambda \in [0,1] \):
\[
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y).
\]

Geometrically, any line segment connecting two points on the graph of a convex function lies above the graph itself.

**Concrete Example:**  
Take \( f(x) = x^2 \) on \(\mathbb{R}\). For any two numbers \( x \) and \( y \) and any \( \lambda \in [0,1] \), you can verify that:
\[
(\lambda x + (1-\lambda)y)^2 \leq \lambda x^2 + (1-\lambda)y^2.
\]
This “bowl-shaped” behavior ensures that if you start anywhere, you will move downwards smoothly to the unique global minimum at \( x = 0 \).

---

### 2. Why Convex Optimization Matters

Even if many practical loss functions are non-convex, convex optimization remains crucial for several reasons:

- **Unique Global Minimum:**  
  In convex problems, any local minimum is a global minimum. For example, minimizing \( f(x) = x^2 \) always leads you to \( x = 0 \), regardless of your starting point.

- **Efficient Algorithms:**  
  Algorithms such as gradient descent work exceptionally well on convex functions because the lack of multiple minima guarantees that you won’t get trapped. Convergence proofs are more straightforward, making convexity a key building block in algorithm design.

- **Theoretical Insights:**  
  Concepts like duality, subgradients, and convex envelopes not only simplify solving convex problems but also provide tools and approximations that help us understand and address non-convex scenarios.

---

### 3. The Building Blocks of Convex Analysis

#### Convex Sets and Convex Hulls

- **Convex Sets:**  
  As defined above, a convex set is one where the line segment between any two points is completely contained in the set.

- **Convex Hull:**  
  The convex hull of a set \( S \subset \mathbb{R}^n \) is the smallest convex set that contains \( S \).  
  **Concrete Example:**  
  Given a finite set of points (say, the vertices of a star-shaped polygon), the convex hull is like a rubber band stretched around them—forming the outer polygon that encloses all points.

#### Convex Functions and Their Properties

- **First-Order Condition:**  
  For a differentiable convex function \( f \), the inequality
  \[
  f(y) \geq f(x) + \nabla f(x)^T (y - x)
  \]
  holds for every \( x, y \) in its domain.  
  **Concrete Interpretation:**  
  At any point \( x \), the tangent line (or hyperplane) is a global under-estimator. For example, drawing the tangent at \( x = 2 \) for \( f(x) = x^2 \) yields a line that lies completely below the curve.

- **Second-Order Condition:**  
  If \( f \) is twice differentiable, convexity is equivalent to the Hessian being positive semidefinite:
  \[
  \nabla^2 f(x) \geq 0.
  \]
  **Concrete Example:**  
  For \( f(x) = x^2 \), the second derivative \( f''(x) = 2 \) is always positive.

#### Subgradients

When a function isn’t differentiable, we use **subgradients**. A vector \( g \) is a subgradient of \( f \) at \( x \) if:
\[
f(y) \geq f(x) + g^T (y - x) \quad \forall\, y.
\]

**Concrete Example:**  
For \( f(x) = |x| \), which is not differentiable at \( x = 0 \), any \( g \) in the interval \([-1, 1]\) is a subgradient at \( 0 \).

#### Lipschitz Continuity

A function \( f: \mathbb{R}^n \to \mathbb{R} \) is **Lipschitz continuous** if there exists a constant \( L \) such that:
\[
|f(x) - f(y)| \leq L \|x-y\|
\]
for all \( x, y \). 

**Concrete Example:**  
For \( f(x) = \sin(x) \) on \(\mathbb{R}\), since \( f'(x)=\cos(x) \) is bounded by 1 in absolute value, \( \sin(x) \) is Lipschitz continuous with constant 1. In contrast, \( f(x)=x^2 \) is not Lipschitz continuous over \(\mathbb{R}\) since its gradient \( 2x \) grows unbounded.

#### Lipschitz Smoothness

Lipschitz smoothness states that the gradient of a function is Lipschitz continuous:
\[
\|\nabla f(x) - \nabla f(y)\| \leq L \|x-y\|.
\]
This condition gives a quadratic upper bound:
\[
f(y) \leq f(x) + \langle \nabla f(x), y-x \rangle + \frac{L}{2}\lVert y-x\rVert^2.
\]

---

### 4. Duality: Turning Problems Inside Out

Duality allows us to recast an optimization problem, often making it easier to solve or analyze.

- **Fenchel Duality:** This form of duality reveals a deep connection between a convex function and its conjugate, allowing one to switch perspectives between minimization and maximization.
- **Strong Duality:**  
Under certain conditions (like Slater's condition), the optimal values of the primal and dual problems are equal, offering a powerful tool for analysis.

For a deeper dive into these dual relationships and many bounds concerning Lipschitz smoothness, see:

[Zhou (2018) - On the Fenchel Duality between Strong Convexity and Lipschitz Continuous Gradient](https://arxiv.org/abs/1803.06573)

---

### 5. Convexity as a Springboard for Non-Convex Analysis

Even though many optimization problems in machine learning are non-convex, the concepts from convex analysis remain invaluable:

- **Relaxation Techniques:**  
  A non-convex problem can often be approximated by its convex hull or another convex relaxation, simplifying the problem while providing useful bounds.

- **Convex Envelopes:**  
  The convex envelope of a function is the best convex under-estimator, which is useful in global optimization.

- **Algorithmic Inspiration:**  
  Many algorithms originally designed for convex problems (e.g., subgradient methods) inform techniques for non-convex scenarios.

---

### 6. Looking Ahead

In the upcoming chapters, we will build on these concrete, foundational ideas:
- We’ll see how **proximal methods** extend convex analysis to handle composite objectives.
- We’ll explore **convex relaxation** techniques to approximate non-convex problems.
- We’ll delve into how the **geometry of loss landscapes** influences the performance of optimization algorithms.

By starting with specific, tangible examples in convexity, we aim to equip you with the tools and intuition needed to navigate the more challenging non-convex territories that lie ahead.

---

### Exercises

Try these exercises to solidify your understanding of convexity and its related concepts:

1. **Convex Set Verification:**  
   Given the set 
   \[
   C = \{ (x,y) \in \mathbb{R}^2 \mid y \geq x^2 \},
   \]
   determine whether \( C \) is convex. Sketch the set and provide a clear argument supporting your conclusion. Then, check if its complement is convex. Can you find a convex set whose complement is convex?

2. **Convex Function Check:**  
   Verify that the function \( f(x) = e^x \) is convex on \(\mathbb{R}\) by checking its second derivative. What does this imply about the global minimum of \( f \)?

3. **Subgradient of the Absolute Value:**  
   For the function \( f(x) = |x| \), find the subdifferential \( \partial f(0) \). Then, illustrate with a sketch why any \( g \) in \( [-1,1] \) qualifies as a subgradient at \( x=0 \).

4. **Lipschitz Continuity Experiment:**  
   Consider the function \( f(x) = \sin(x) \) on \(\mathbb{R}\). Use the Mean Value Theorem to show that \( f \) is Lipschitz continuous with constant 1. Then, find a function that is not Lipschitz continuous on \(\mathbb{R}\) and explain why.

5. **Convex Hull Construction:**  
   Given a finite set of points in the plane (e.g., \((0,0)\), \((1,2)\), \((2,1)\), \((1,-1)\)), construct the convex hull of these points. Sketch the original points and the resulting convex hull.

6. **Fenchel Conjugate Calculation:**  
   For \( f(x) = \frac{1}{2}x^2 \), derive its conjugate \( f^*(y) \) using the definition:
   \[
   f^*(y) = \sup_{x} \{ yx - \frac{1}{2}x^2 \}.
   \]
   Verify that \( f^*(y) = \frac{1}{2}y^2 \).

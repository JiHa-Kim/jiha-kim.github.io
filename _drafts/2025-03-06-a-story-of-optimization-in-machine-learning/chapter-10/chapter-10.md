---
layout: post
title: "A Story of Optimization in ML: Chapter 8 - Reflections in a Distorted Mirror: Stepping Beyond Euclidean Space"
description: "Chapter 8 introduces Mirror Descent, extending optimization beyond Euclidean spaces by using Bregman divergences to navigate in geometries that are not flat."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["mirror descent", "non-Euclidean optimization", "Bregman divergence", "Bregman projection", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/mirror_descent_chapter8.gif
  alt: "Visual metaphor for mirror descent - a distorted reflection guiding the path"
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 8: Reflections in a Distorted Mirror - Stepping Beyond Euclidean Space

Imagine a hall of mirrors—but not the playful kind. These are *mathematical* mirrors that distort space itself. In our quest to minimize a function, using standard Euclidean gradients in such a warped setting might lead us astray: the “downhill” direction in the mirror could be entirely different from that in the real world.

In traditional optimization, **Projected Gradient Descent (PGD)** works well when our constraints and geometry are Euclidean. We take a gradient step and then project back onto a feasible set using the familiar Euclidean distance. However, when the underlying geometry is non-Euclidean—say, when optimizing over the probability simplex—Euclidean projections can feel unnatural or inefficient.

This is where **Mirror Descent** comes into play. It generalizes PGD by using **Bregman divergences** (distance-like functions tailored to specific convex potentials) to respect the intrinsic geometry of the problem.

---

### Mirror Descent: A Geometric Perspective

The idea behind Mirror Descent is to **transform** the problem into a mirror (or dual) space where the geometry is better aligned with the problem’s structure, take a gradient-like step there, and then map back to the original space.

Here’s the step-by-step intuition:

1. **Choose a Mirror Map:**  
   Select a strictly convex and differentiable potential function $$\phi(x)$$. Its gradient, $$\nabla \phi(x)$$, serves as the mirror map that transforms our variables into a new space. This map induces a **Bregman divergence**,
   
   $$
   D_\phi(x \| y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y \rangle,
   $$
   
   which measures distance in a geometry that need not be Euclidean.

2. **Map to the Mirror Space:**  
   Instead of working directly with the parameters $$x$$, we consider their images under the mirror map, $$\nabla \phi(x)$$. This transformation reorients the space to better capture the underlying structure.

3. **Perform the Gradient Step in Mirror Space:**  
   Compute the gradient of the loss function $$\nabla f(x_k)$$ in the original space, and then take a step in the mirror space. In essence, we modify the mirror image of the current iterate by subtracting a scaled gradient:
   
   $$
   \nabla \phi(x_k) - \eta \nabla f(x_k).
   $$

4. **Map Back via Bregman Projection:**  
   To translate the update back to the original space, we perform a **Bregman projection**. The next iterate $$x_{k+1}$$ is chosen by solving:
   
   $$
   x_{k+1} = \arg\min_{x \in C} \left\{ \eta \langle \nabla f(x_k), x \rangle + D_\phi(x \| x_k) \right\}.
   $$
   
   This step finds the point that is closest to the mirror update (in terms of Bregman divergence) while balancing the descent direction.

---

### Unifying Euclidean and Non-Euclidean Approaches

The strength of Mirror Descent lies in its flexibility. By a careful choice of the potential function $$\phi(x)$$, Mirror Descent can recover standard methods:

#### 1. Recovering Projected Gradient Descent

Choose the Euclidean potential:
$$
\phi(x) = \frac{1}{2}\|x\|_2^2.
$$

- **Mirror Map:**  
  $$\nabla \phi(x) = x.$$
  
- **Bregman Divergence:**  
  $$
  D_\phi(x \| y) = \frac{1}{2}\|x-y\|_2^2.
  $$
  
Plugging these into the update rule gives:

$$
x_{k+1} = \arg\min_{x \in C} \left\{ \eta \langle \nabla f(x_k), x \rangle + \frac{1}{2}\|x-x_k\|_2^2 \right\},
$$

which is exactly the update in Projected Gradient Descent.

#### 2. Recovering Proximal Gradient Descent (and Beyond)

For composite optimization problems of the form

$$
\min_x \; f(x) + g(x),
$$

we can extend Mirror Descent by incorporating the non-smooth term into the update:

$$
x_{k+1} = \arg\min_{x \in C} \left\{ \eta \langle \nabla f(x_k), x \rangle + \eta g(x) + D_\phi(x \| x_k) \right\}.
$$

Again, if we choose the Euclidean potential, this formulation simplifies to the familiar Proximal Gradient Descent update. More generally, selecting non-Euclidean potentials tailors the algorithm to the geometry of the problem—leading, for example, to **Exponentiated Gradient Descent** when optimizing over the probability simplex.

---

### Why Mirror Descent?

Mirror Descent adapts to the geometry of the problem by:

- **Embracing Non-Euclidean Geometries:**  
  In domains like the probability simplex or spaces of positive definite matrices, using Bregman divergences such as KL divergence or Burg divergence can lead to faster convergence and more natural updates than Euclidean methods.

- **Respecting Intrinsic Constraints:**  
  Bregman projections naturally enforce constraints in a way that aligns with the geometry of the space.

- **Generalizing Classical Methods:**  
  By choosing different potential functions, Mirror Descent unifies and extends classical algorithms like Projected and Proximal Gradient Descent, providing a powerful framework for a wide range of optimization problems.

---

### Applications

Mirror Descent has wide-ranging applications:

- **Optimization over the Probability Simplex:**  
  With the negative entropy potential, Mirror Descent becomes Exponentiated Gradient Descent, which is well-suited for probability distributions.
  
- **Online Learning and Regret Minimization:**  
  It is a cornerstone algorithm in online convex optimization, especially when decisions are made in non-Euclidean domains.
  
- **Matrix and Tensor Optimization:**  
  Custom potentials enable efficient updates in problems involving positive definite matrices or tensors.

---

### Stepping Out of Flatland

Mirror Descent reminds us that optimization isn’t solely about following Euclidean gradients—it’s about understanding and leveraging the geometry of the problem. By stepping beyond the confines of Euclidean space and embracing Bregman divergences, Mirror Descent equips us with a versatile toolkit for tackling modern machine learning challenges.

In the chapters to come, we’ll see how these geometric insights, combined with principles like duality and majorization-minimization, lead to the sophisticated optimizers powering contemporary machine learning. For now, solidify your understanding of Mirror Descent with the following exercises.

---

> **Exercise 1: Mirror Descent with Squared Euclidean Norm: Recovering Gradient Descent**
>
> Consider the potential function $$\phi(x) = \frac{1}{2}\|x\|_2^2$$.
>
> **(a)** Show that the Bregman divergence induced by $$\phi$$ is the squared Euclidean distance: $$D_\phi(x\|y) = \frac{1}{2}\|x-y\|_2^2$$.
>
> **(b)** Derive the Mirror Descent update rule for this potential function and show that it simplifies to the standard Gradient Descent update.
>
> **(c)** If we include projection onto a closed convex set $$C$$, what algorithm do we recover?
>
> *Hint:* Substitute $$\phi(x) = \frac{1}{2}\|x\|_2^2$$ and simplify the update.

> **Exercise 2: Mirror Descent with Negative Entropy: Exponentiated Gradient Descent**
>
> Consider the negative entropy potential on the probability simplex $$\Delta^n = \{x \in \mathbb{R}^n_+ : \sum_{i=1}^n x_i = 1\}$$:
> $$
> \phi(x) = \sum_{i=1}^n x_i \log x_i.
> $$
>
> **(a)** Compute $$\nabla \phi(x)$$ and the corresponding Bregman divergence, recognizing it as the Kullback-Leibler (KL) divergence.
>
> **(b)** Derive the Mirror Descent update rule and show that it leads to Exponentiated Gradient Descent, which naturally preserves the simplex constraint.
>
> **(c)** Explain why this approach is preferable to standard Gradient Descent with Euclidean projection on the simplex.
>
> *Hint:* Solve the Bregman projection minimization explicitly for the negative entropy potential.

> **Exercise 3: Bregman Projection for Mirror Descent with KL Divergence**
>
> Using the negative entropy potential, solve the Bregman projection:
> $$
> x_{k+1} = \arg\min_{x \in \Delta^n} \left\{ \eta \langle \nabla f(x_k), x \rangle + D_{KL}(x \| x_k) \right\},
> $$
> to derive the Exponentiated Gradient Descent update.
>
> *Hint:* Use Lagrange multipliers to incorporate the simplex constraints.

---

**Further Reading:**

- [Nemirovski and Yudin (1983) - Problem Complexity and Method Efficiency in Optimization](https://link.springer.com/book/10.1007/978-3-662-09978-1)
- [Beck and Teboulle (2003) - Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization](https://epubs.siam.org/doi/abs/10.1137/S003614290241847X)
- [Shalev-Shwartz (2012) - Online Learning and Online Convex Optimization](http://www.cs.huji.ac.il/~shais/papers/OLbook-v1.pdf)
- [Hazan (2019) - Lecture Notes: Optimization for Machine Learning](https://arxiv.org/abs/1909.03550)

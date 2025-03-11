---
layout: post
title: "A Story of Optimization in ML: Chapter 7 - Reflections in a Distorted Mirror: Stepping Beyond Euclidean Space"
description: "Chapter 7 introduces Mirror Descent, extending optimization beyond Euclidean spaces by using Bregman divergences to navigate in geometries that are not flat."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["mirror descent", "non-Euclidean optimization", "Bregman divergence", "Bregman projection", "optimizer", "projected gradient descent"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/mirror_descent_chapter7.gif # Placeholder image path
  alt: "Visual metaphor for mirror descent - a distorted reflection guiding the path" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 7: Reflections in a Distorted Mirror - Stepping Beyond Euclidean Space

Imagine you are in a hall of mirrors. But these aren't your usual funhouse mirrors; these are *mathematical* mirrors, distorting space itself.  You want to find the lowest point in this distorted reflection of reality. If you naively apply what you know about downhill directions in the *mirror* world, and then try to translate those steps back to the *real* world, you might end up going completely astray.  What if the 'downhill' direction in the mirror corresponds to 'uphill' in reality, or a completely different direction altogether?

This is the challenge we face when our optimization landscape isn't the flat, familiar Euclidean space we've been used to.  Sometimes, the natural geometry of our problem is curved, distorted, non-Euclidean. And in these cases, our trusty Projected Gradient Descent, with its reliance on straight lines and Euclidean projections, might not be the most effective guide.

Think back to **Projected Gradient Descent (PGD)** from Chapter 5. It’s a wonderfully intuitive method when we're confined to a convex set in Euclidean space. We take a gradient step, and then, if we stray outside our allowed boundaries, we simply project ourselves straight back in – using the familiar Euclidean distance as our guide.

[Visual: Simple diagram of Projected Gradient Descent in Euclidean space - step and project]

But what if "straight back in" isn't the most natural or efficient way to correct our course? What if our feasible region isn't best described by Euclidean geometry?

Consider optimizing over the **probability simplex** – the space of all probability distributions.  If you take a gradient step and then project back using Euclidean projection, you might end up with a distribution that's valid, but it might feel… unnatural.  Euclidean distance treats all directions as equally "straight," but in the world of probabilities, some directions are more meaningful than others.

What we need is a way to generalize the idea of "projection" and "descent" to geometries beyond the Euclidean.  We need a method that respects the intrinsic structure of the space we're working in. This is where **Mirror Descent** comes into play.

### Mirror Descent:  A Reflection of Geometry

Mirror Descent is a beautiful generalization of Projected Gradient Descent.  Instead of relying solely on Euclidean distances and projections, it leverages the concept of **Bregman divergences** we explored in the previous chapter. Remember, Bregman divergences are distance-like measures tailored to specific convex functions, offering a way to quantify "dissimilarity" in a non-Euclidean way.

The core idea of Mirror Descent is to perform optimization in a **transformed space**, a "mirror space," and then map the updates back to our original parameter space. This transformation is guided by a carefully chosen **mirror map**, which is derived from a strictly convex function, often called the **potential function**.

Here’s the conceptual outline of Mirror Descent:

1.  **Choose a Mirror Map:** Select a strictly convex and differentiable function $$\phi(x)$$ that defines our "mirror geometry" through its induced Bregman divergence $$D_\phi(x\|y)$$.  The gradient of this function, $$\nabla \phi(x)$$, acts as our **mirror map**.

2.  **Map to the Mirror Space:**  Instead of working directly with our parameters $$x$$ in the original space, we work with their "reflections" in the mirror space, given by $$\nabla \phi(x)$$. Think of $$\nabla \phi(x)$$ as a coordinate transformation, mapping our parameters into a new, potentially non-Euclidean space.

3.  **Perform Gradient Descent in the Mirror Space:** We compute the gradient of our loss function $$\nabla f(x_k)$$ in the *original* space, but then we take our optimization step in the *mirror* space.  Imagine the gradient as a direction in the real world, but we apply this direction in our distorted mirror reflection.

4.  **Map Back to the Original Space (Bregman Projection):**  After taking a step in the mirror space, we need to translate this update back to our original parameter space.  This "inverse mapping" is not a simple inverse function. Instead, it involves a **Bregman projection**. We find the point in the original space that, when mapped to the mirror space, is closest (in terms of Bregman divergence) to our updated point in the mirror space.

[Visual: Diagram of Mirror Descent - Primal space -> Mirror space (via mirror map), GD step in mirror space, Mirror space -> Primal space (via Bregman projection)]

Let’s formalize the Mirror Descent algorithm.  Given a loss function $$f(x)$$, a strictly convex potential function $$\phi(x)$$, and a learning rate $$\eta$$, the Mirror Descent update is as follows:

1.  **Gradient Step (in Primal Space):** Compute the gradient of the loss function at the current point $$x_k$$,  $$\nabla f(x_k)$$.

2.  **Update in Mirror Space:**  Take a gradient descent step *in the mirror space*.  This is done implicitly through the following Bregman projection step. We want to find the next iterate $$x_{k+1}$$ such that its mirror image, $$\nabla \phi(x_{k+1})$$, is close to the mirror image of the previous point, $$\nabla \phi(x_k)$$, *minus* the gradient in the original space, $$\eta \nabla f(x_k)$$.  More formally, we find $$x_{k+1}$$ by solving the following Bregman projection problem:

    $$
    x_{k+1} = \arg\min_{x} \left\{ \eta \langle \nabla f(x_k), x \rangle + D_\phi(x \| x_k) \right\}
    $$

    This minimization problem effectively combines the gradient descent step with the Bregman projection back to the original space in a single, elegant update.

    In many cases, this Bregman projection step has a closed-form solution, making Mirror Descent computationally efficient.

### Why Does This "Mirror" Trick Work?

Why go through all this trouble of transforming to a mirror space and back?  The power of Mirror Descent lies in its ability to **adapt to the geometry of the problem**. By choosing an appropriate potential function $$\phi(x)$$ and its associated Bregman divergence, we can tailor our optimization algorithm to the specific constraints and structure of our parameter space.

*   **Non-Euclidean Geometry:** When the natural geometry is non-Euclidean (e.g., probability simplex, positive definite matrices), Mirror Descent, using a suitably chosen Bregman divergence (like KL divergence or Burg divergence), can lead to much faster convergence and better performance compared to Euclidean methods like Projected Gradient Descent.

*   **Respecting Constraints:**  The Bregman projection step naturally keeps the iterates within the feasible region (if the Bregman divergence is well-defined on that region). It's a more geometry-aware way of enforcing constraints compared to simple Euclidean projection.

*   **Generalized Projections:**  Euclidean projection is just a special case of Bregman projection when we use the squared Euclidean norm as our potential function.  Mirror Descent generalizes the concept of projection to a much broader class of divergences, opening up new possibilities for optimization.

### Examples and Applications

Mirror Descent finds applications in various areas where non-Euclidean geometries are important:

*   **Optimization on the Probability Simplex:** For problems involving probability distributions, using the KL divergence as the Bregman divergence in Mirror Descent leads to algorithms like **Exponentiated Gradient Descent**, which are particularly well-suited for this constrained space.

*   **Online Learning and Regret Minimization:** Mirror Descent is a fundamental algorithm in online learning and regret minimization, especially in settings where actions or predictions are constrained to lie in non-Euclidean sets.

*   **Matrix and Tensor Optimization:**  For optimization problems involving positive definite matrices or tensors, Bregman divergences based on matrix norms or entropy functions can be used to design efficient Mirror Descent algorithms.

###  Stepping Out of the Flatland

Mirror Descent is a powerful reminder that optimization is not just about blindly following gradients. It's about understanding the geometry of the landscape and choosing algorithms that respect that geometry.  By stepping beyond the confines of Euclidean space and embracing the flexibility of Bregman divergences, Mirror Descent provides us with a richer and more versatile toolkit for navigating the complex optimization landscapes of machine learning.

In the chapters to come, we'll explore how these geometric insights, combined with other optimization principles like duality and majorization-minimization, lead to the sophisticated and efficient optimizers that power modern machine learning. But for now, let's solidify our understanding of Mirror Descent with a few exercises.

---

> **Exercise 1: Mirror Descent with Squared Euclidean Norm: Recovering Gradient Descent**
>
> Consider the potential function $$\phi(x) = \frac{1}{2}\|x\|_2^2$$.
>
> **(a)** Show that the Bregman divergence induced by $$\phi$$ is the squared Euclidean distance: $$D_\phi(x\|y) = \frac{1}{2}\|x-y\|_2^2$$.
>
> **(b)** Derive the Mirror Descent update rule for this potential function. Show that it simplifies to the standard Gradient Descent update.
>
> **(c)** If we also consider projection onto a closed convex set $$C$$ in the Mirror Descent framework with this potential function, what algorithm do we recover?
>
> *Hint:* Substitute $$\phi(x) = \frac{1}{2}\|x\|_2^2$$ and its gradient into the Mirror Descent update formula and simplify.

> **Exercise 2: Mirror Descent with Negative Entropy: Exponentiated Gradient Descent**
>
> Consider the negative entropy potential function on the probability simplex $$\Delta^n = \{x \in \mathbb{R}^n_+ : \sum_{i=1}^n x_i = 1\}$$:
> $$
> \phi(x) = \sum_{i=1}^n x_i \log x_i.
> $$
>
> **(a)** Compute the gradient $$\nabla \phi(x)$$ and the Bregman divergence $$D_\phi(x\|y)$$ for this potential function.  Recognize the Bregman divergence as the Kullback-Leibler (KL) divergence.
>
> **(b)** Derive the Mirror Descent update rule for this potential function. Show that it leads to the Exponentiated Gradient Descent update, where updates are multiplicative and preserve the probability simplex constraint.
>
> **(c)** Explain why Exponentiated Gradient Descent is naturally suited for optimization over probability distributions and contrast it with standard Gradient Descent followed by Euclidean projection onto the simplex.
>
> *Hint:* For part (b), you might need to solve the Bregman projection minimization problem explicitly for the negative entropy potential.

> **Exercise 3: Bregman Projection for Mirror Descent with KL Divergence**
>
> Consider Mirror Descent with the negative entropy potential function (leading to KL divergence).  Given a point $$x_k$$ and a gradient $$\nabla f(x_k)$$, explicitly solve the Bregman projection problem:
> $$
> x_{k+1} = \arg\min_{x \in \Delta^n} \left\{ \eta \langle \nabla f(x_k), x \rangle + D_{KL}(x \| x_k) \right\}
> $$
> to derive the update rule for Exponentiated Gradient Descent.
>
> *Hint:* Use Lagrange multipliers to solve the constrained optimization problem. Consider the constraints of the probability simplex.

---

**Further Reading:**

*   [Nemirovski and Yudin (1983) - Problem Complexity and Method Efficiency in Optimization](https://link.springer.com/book/10.1007/978-3-662-09978-1) (Seminal work on Mirror Descent)
*   [Beck and Teboulle (2003) - Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization](https://epubs.siam.org/doi/abs/10.1137/S003614290241847X) (More accessible introduction)
*   [Shalev-Shwartz (2012) - Online Learning and Online Convex Optimization](http://www.cs.huji.ac.il/~shais/papers/OLbook-v1.pdf) (Chapter 4 - Mirror Descent)
*   [Hazan (2019) - Lecture Notes: Optimization for Machine Learning](https://arxiv.org/abs/1909.03550) (Lecture 7 - Mirror Descent)

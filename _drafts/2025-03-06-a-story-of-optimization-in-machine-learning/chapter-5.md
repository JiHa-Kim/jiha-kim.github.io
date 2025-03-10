---
layout: post
title: "A Story of Optimization in ML: Chapter 5 - Staying Within Bounds: Projected Descent"
description: "Chapter 5 introduces Projected Gradient Descent, a method for handling constraints in Euclidean space by combining gradient steps with projections onto feasible sets."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["projected gradient descent", "constrained optimization", "Euclidean projection", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/projected_gd_chapter5.gif # Placeholder image path
  alt: "Visual representation of Projected Gradient Descent" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 5: Staying Within Bounds - Projected Descent in Euclidean Space

So far, we've been exploring unconstrained optimization – freely navigating the loss landscape to find the lowest point. But in many real-world machine learning problems, we need to work with **constraints**. We might want to ensure our model parameters satisfy certain conditions, or stay within specific, valid ranges.

Imagine you're sculpting a statue. You're chipping away at a block of stone (minimizing loss), but you also need to make sure your sculpture stays within the original boundaries of the stone block (constraints). You can't let your chisel strokes wander completely freely; you must stay within the feasible region.

Constraints in machine learning arise in various forms:

*   **Weight Clipping:** To prevent exploding gradients or ensure stability, we might want to keep the weights of our neural network within a certain range, say, between -1 and 1.
*   **Non-Negativity Constraints:** In some models, parameters might represent probabilities or physical quantities that must be non-negative.
*   **Norm Constraints:**  We might want to limit the overall magnitude of our parameter vector to prevent overfitting or improve generalization.
*   **Constraints to Lie on a Manifold:** Parameters might need to reside on a specific manifold, like the probability simplex (for probability distributions) or the Stiefel manifold (for orthogonal matrices).

If we use standard Gradient Descent for constrained optimization, we run into a problem: **gradient steps can take us outside the feasible region**, violating our constraints. Imagine our ball rolling down a hill, but the hill is fenced in.  A simple downhill roll might easily take the ball crashing into the fence, outside the allowed area.

How do we adapt Gradient Descent to handle constraints?  A simple and intuitive approach in Euclidean space is **Projected Gradient Descent (PGD)**.

**Projected Gradient Descent: Step and Project**

The core idea of Projected Gradient Descent is remarkably straightforward:

1.  **Take a standard Gradient Descent step.**  Calculate the gradient and update your parameters as if there were no constraints.
2.  **Project the updated parameters back onto the feasible set.** If the gradient step has taken you outside the allowed region, "project" your parameters back to the closest point within the feasible region.

It's like taking a step downhill, and then, if you find yourself outside the boundaries of the allowed area, immediately "snapping" back to the nearest valid point.

[Visual representation of Projected Gradient Descent - GD step followed by projection back to feasible set]

Let's formalize this. Suppose we want to minimize a smooth function $$f(x)$$ subject to the constraint that $$x$$ must belong to a closed convex set $$C \subseteq \mathbb{R}^n$$.  The **Projected Gradient Descent (PGD)** algorithm is:

1.  **Gradient Descent Step:**
    $$v_k = x_k - \eta \nabla f(x_k)$$
2.  **Projection Step:**
    $$x_{k+1} = \operatorname{proj}_C(v_k)$$

Where $$\operatorname{proj}_C(v_k)$$ denotes the **Euclidean projection** of the point $$v_k$$ onto the set $$C$$.

**Euclidean Projection: Finding the Closest Valid Point**

The **Euclidean projection** of a point $$v$$ onto a closed convex set $$C$$ is the point in $$C$$ that is closest to $$v$$ in terms of Euclidean distance. Mathematically:

$$
\operatorname{proj}_C(v) = \arg\min_{y \in C} \|y - v\|_2
$$

For some simple convex sets, the Euclidean projection is easy to compute analytically. For more complex sets, it might require solving a smaller optimization problem.

**Examples of Euclidean Projections onto Common Convex Sets:**

*   **Projection onto a Box Constraint:**  Suppose $$C = \{x \in \mathbb{R}^n \mid l_i \leq x_i \leq u_i \text{ for all } i=1, \dots, n \}$$, where $$l_i$$ and $$u_i$$ are lower and upper bounds for each component $$x_i$$.  The projection onto this box is simply **component-wise clipping**:

    $$
    [\operatorname{proj}_C(v)]_i = \begin{cases}
        l_i & \text{if } v_i < l_i \\
        v_i & \text{if } l_i \leq v_i \leq u_i \\
        u_i & \text{if } v_i > u_i
    \end{cases}
    $$

*   **Projection onto the Non-negative Orthant:**  Suppose $$C = \{x \in \mathbb{R}^n \mid x_i \geq 0 \text{ for all } i=1, \dots, n \}$$.  The projection onto the non-negative orthant is also component-wise: **clip negative values to zero**:

    $$
    [\operatorname{proj}_C(v)]_i = \max(0, v_i) = \begin{cases}
        v_i & \text{if } v_i \geq 0 \\
        0 & \text{if } v_i < 0
    \end{cases}
    $$

*   **Projection onto the Probability Simplex (More Complex Example):** Suppose $$C = \{x \in \mathbb{R}^n \mid x_i \geq 0, \sum_{i=1}^n x_i = 1 \}$$.  Projection onto the probability simplex is slightly more involved, but there are efficient algorithms to compute it.  It ensures that the projected vector is non-negative and sums to 1, making it a valid probability distribution.

**Connection to Proximal Gradient Descent (Revisited):**

Remember Proximal Gradient Descent from the previous chapter?  We saw that for a composite loss function $$L(x) = f(x) + g(x)$$, we could use the update:

1.  Gradient Step: $$v = x_k - \eta \nabla f(x_k)$$
2.  Proximal Step: $$x_{k+1} = \operatorname{prox}_{\eta, g}(v)$$

Now, consider setting the non-smooth function $$g(x)$$ to be the **indicator function** of the constraint set $$C$$, i.e., $$g(x) = \delta_C(x) = 0$$ if $$x \in C$$ and $$+\infty$$ if $$x \notin C$$.  Then, the Proximal Gradient Descent update becomes:

1.  Gradient Step: $$v = x_k - \eta \nabla f(x_k)$$
2.  Proximal Step: $$x_{k+1} = \operatorname{prox}_{\eta, \delta_C}(v) = \arg\min_{y} \left\{ \delta_C(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$

But we know from the previous chapter (Exercise 2) that the proximal operator of the indicator function $$\delta_C(x)$$ is precisely the **Euclidean projection onto the set $$C$**:  $$\operatorname{prox}_{\eta, \delta_C}(v) = \operatorname{proj}_C(v)$$.

Therefore, **Projected Gradient Descent is a special case of Proximal Gradient Descent**, where the non-smooth part of the loss function is the indicator function of the constraint set.  This provides a deeper connection between these two seemingly different approaches.

**Advantages and Considerations of Projected Gradient Descent:**

*   **Simplicity and Intuition:** PGD is conceptually simple and easy to implement, especially when the projection onto the constraint set is computationally efficient.
*   **Handles Constraints in Euclidean Space:** It provides a straightforward way to incorporate constraints into gradient-based optimization in Euclidean space.
*   **Convergence Guarantees (for Convex Problems):** For convex objective functions and convex constraint sets, PGD often enjoys convergence guarantees under suitable step size conditions.
*   **Still in Euclidean Space:**  PGD is fundamentally a Euclidean space method.  It relies on Euclidean projection and Euclidean gradients.  For problems where the natural geometry is non-Euclidean, or where constraints are more naturally expressed in non-Euclidean spaces, PGD might not be the most efficient or elegant approach.

In the next chapter, we will take a significant step beyond Euclidean space. We will explore **Mirror Descent**, a powerful generalization of Projected Gradient Descent that allows us to perform constrained optimization in non-Euclidean geometries, using more general "projections" based on **Bregman divergences**.  This will open up a new world of optimization techniques tailored to the intrinsic geometry of the problem at hand.

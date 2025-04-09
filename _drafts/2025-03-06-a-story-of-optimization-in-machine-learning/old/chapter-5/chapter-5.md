---
layout: post
title: "A Story of Optimization in ML: Chapter 5 - Navigating Non-Smooth Terrain"
description: "Chapter 5 explores optimization in non-smooth loss landscapes, introducing proximal methods and the Moreau envelope as tools to handle functions that lack gradients everywhere, and extending these ideas to composite problems via proximal gradient descent."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["proximal methods", "non-smooth optimization", "Moreau envelope", "proximal gradient descent", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/nonsmooth_chapter5.gif
  alt: "Visual representation of non-smooth optimization and proximal methods"
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 5: Navigating Non-Smooth Terrain – Beyond the Gradient

In our optimization journey, we've become comfortable with smooth loss landscapes, where the concept of a **gradient** guides our descent. Gradient Descent and its variations thrive in these environments, efficiently leading us toward a minimum. But what happens when the landscape becomes… jagged?

Imagine stepping onto a terrain that's no longer gentle rolling hills, but a fractured landscape of sharp cliffs, abrupt edges, and sudden corners. This is the world of **non-smooth optimization**. Non-smoothness isn’t just a theoretical curiosity—it naturally arises in many practical machine learning scenarios.

Consider these common elements in modern models:

- **ReLU Activation:** The ReLU (Rectified Linear Unit) activation function,
  
  $$
  \operatorname{ReLU}(x) = \max(0, x),
  $$

  is incredibly popular in neural networks. Yet, at $$x=0$$ it has a sharp kink—making the gradient ill-defined there.

- **L1 Regularization:** To encourage simpler, sparser models, we often include L1 regularization, adding terms like
  
  $$
  |x|
  $$

  to our loss function. The absolute value function has a non-differentiable point at zero.

- **Constraints:** Explicit constraints on model parameters—such as non-negativity or bounds—can also introduce non-smoothness into the optimization problem.

In these non-smooth landscapes, our usual guide—the gradient—becomes unreliable. At the sharp edges and corners, it is not well-defined, and standard gradient-based methods may struggle to determine a consistent descent direction.

So, how do we proceed when gradients fail us? How can we navigate such non-smooth terrain?

We need to move **beyond the gradient** as our sole guide. We require a more robust strategy—one that directs us toward the minimum even in the absence of smooth gradients. This is where **proximal methods** come into play.

---

### Moving Beyond the Gradient: The Proximal Idea

Proximal methods replace the direct gradient step with a more robust update by solving a small optimization problem at each iteration. Instead of blindly following the gradient, each step balances two goals:

1. **Reducing the Loss:** We still aim to descend into valleys of lower loss.
2. **Staying Close to the Current Point:** To avoid erratic jumps, especially in unpredictable, non-smooth regions, we constrain our update to remain near our current location.

Remember our gradient flow ODE:

$$
\frac{dx(t)}{dt} = -\nabla L(x(t)),
$$

We've seen that the forward Euler discretization of this equation leads to the familiar gradient descent update rule. But we can also try the backward Euler discretization of gradient flow, where we instead evaluate the gradient at the endpoint of each interval:

$$
x_{k+1} = x_k - \eta \nabla L(x_{k+1}).
$$

For a smooth function, this can be rewritten as an optimization objective:

$$
x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}.
$$

Even when the loss function $$L(y)$$ is non-smooth, this formulation is meaningful. The quadratic term

$$
\frac{1}{2\eta}\|y - x_k\|_2^2
$$

acts as a **regularizer** that smooths the problem and ensures a stable update.

---

### The Proximal Operator: Your Robust Compass

This regularized minimization introduces the **proximal operator**. For any function $$g(x)$$ (smooth or non-smooth) and a parameter $$\eta > 0$$, the proximal operator is defined as:

$$
\operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
$$

Think of it as a **robust compass**: Given a current point $$v$$, the operator guides you to a new point that makes $$g(y)$$ as small as possible while keeping you close to $$v$$. The parameter $$\eta$$ governs this trade-off—a smaller $$\eta$$ forces the update to stick close to $$v$$, while a larger $$\eta$$ allows a greater move toward minimizing $$g$$.

For well-behaved functions (proper, lower semicontinuous, and convex), the proximal operator is well-defined and unique.

---

### From Proximal Point to Proximal Gradient Descent

#### The Proximal Point Algorithm

For a convex (possibly non-smooth) function $$g(x)$$, the **proximal point algorithm** updates via:

$$
x_{k+1} = \operatorname{prox}_{\eta, g}(x_k).
$$

This method can be interpreted as performing gradient descent on the **Moreau envelope** of $$g(x)$$. The Moreau envelope is defined by:

$$
M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\},
$$

and it serves as a smooth approximation to the original non-smooth function. Its gradient is given by

$$
\nabla M_{\eta, g}(v) = \frac{1}{\eta}\Bigl(v - \operatorname{prox}_{\eta, g}(v)\Bigr).
$$

Thus, the proximal point update is equivalent to taking an implicit gradient descent step on $$M_{\eta, g}(x)$$.

#### Transition to Proximal Gradient Descent

Many real-world problems, however, involve an objective function that is a sum of a smooth and a non-smooth component:

$$
L(x) = f(x) + g(x),
$$

where:

- $$f(x)$$ is smooth (e.g., a loss function with differentiable terms).
- $$g(x)$$ is non-smooth (e.g., regularization terms or constraints).

Here, we can blend the ideas from gradient descent and the proximal point algorithm to form **Proximal Gradient Descent**. The algorithm proceeds in two steps:

1. **Gradient Step (Smooth Descent):**

   First, take a gradient descent step on the smooth part $$f(x)$$:

   $$
   v = x_k - \eta\, \nabla f(x_k).
   $$

2. **Proximal Step (Non-Smooth Correction):**

   Next, apply the proximal operator to incorporate the non-smooth term $$g(x)$$:

   $$
   x_{k+1} = \operatorname{prox}_{\eta, g}(v).
   $$

This two-step update,

$$
x_{k+1} = \operatorname{prox}_{\eta, g}\Bigl(x_k - \eta\, \nabla f(x_k)\Bigr),
$$

efficiently navigates composite loss landscapes by leveraging the smooth structure of $$f(x)$$ while robustly handling the non-smooth component $$g(x)$$. One can even view this method as performing gradient descent on a specially constructed smooth surrogate of the composite objective—often referred to as the **forward–backward envelope**—which generalizes the idea of the Moreau envelope to composite problems.

---

### The Moreau Envelope: Smoothing the Rough Edges

The Moreau envelope of a function $$g(x)$$ is not only central to the proximal point algorithm, but also provides the intuition behind why proximal gradient descent works so effectively. Even if $$g(x)$$ is non-differentiable, its Moreau envelope

$$
M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n}\left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
$$

is always differentiable (assuming $$g$$ is closed, proper, and convex). The gradient of this envelope,

$$
\nabla M_{\eta, g}(v) = \frac{1}{\eta}\Bigl(v - \operatorname{prox}_{\eta, g}(v)\Bigr),
$$

shows that the proximal operator is intrinsically linked to a gradient step on a smoothed version of the original function. In the pure non-smooth case (when $$f\equiv0$$), the proximal point method is essentially equivalent to gradient descent on the Moreau envelope. In the composite setting, the proximal gradient step can be interpreted as a gradient-like descent on an envelope that blends both $$f$$ and $$g$$.

---

### Practical Considerations

The parameter $$\eta$$ plays a dual role in these methods. It not only scales the quadratic regularization in the proximal operator but also determines the level of smoothing in the Moreau envelope. A smaller $$\eta$$ means the Moreau envelope is a closer approximation to $$g(x)$$ but might be less smooth, while a larger $$\eta$$ yields a smoother envelope that may not capture all the fine details of $$g$$.

Finding the right balance and choosing an appropriate learning rate (which relates to $$\eta$$) is essential for both convergence speed and accuracy, especially in problems where both smooth and non-smooth elements are present.

Proximal methods, and in particular Proximal Gradient Descent, have become indispensable in machine learning for handling regularization (like L1 for sparsity), constraints, and non-smooth activations. They provide a robust and principled way to navigate challenging optimization landscapes that extend beyond the idealized smooth scenarios.

---

### Bonus: Proximal Mapping and Subdifferentials

For convex functions, the proximal operator’s optimality condition connects directly with subdifferentials. For instance, if

$$
x^\star = \operatorname{prox}_{\eta, g}(v),
$$

then the first-order optimality condition is

$$
0 \in \partial g(x^\star) + \frac{1}{\eta}(x^\star - v),
$$

which rearranges to

$$
\frac{1}{\eta}(v - x^\star) \in \partial g(x^\star).
$$

This relationship shows how the proximal step relates to generalized gradients for non-smooth functions.

---

### **Exercises**

> **Exercise 1: Proximal Operator of the L1 Norm (Soft Thresholding)**
>
> Let
> $$
> g(x) = \|x\|_1 = \sum_{i=1}^n |x_i|.
> $$
> Show that the proximal operator of $$g$$ is the soft thresholding operator:
> $$
> [\operatorname{prox}_{\eta, \|\cdot\|_1}(v)]_i = S_{\eta}(v_i) =
> \begin{cases}
> v_i - \eta, & \text{if } v_i > \eta, \\
> v_i + \eta, & \text{if } v_i < -\eta, \\
> 0, & \text{if } |v_i| \leq \eta.
> \end{cases}
> $$
> *Hint:* Solve the minimization problem component-wise.

> **Exercise 2: Proximal Operator of the Indicator Function (Projection)**
>
> Let $$C$$ be a closed convex set and let the indicator function be
> $$
> \delta_C(x) =
> \begin{cases}
> 0, & \text{if } x\in C, \\
> +\infty, & \text{otherwise}.
> \end{cases}
> $$
> Show that
> $$
> \operatorname{prox}_{\eta, \delta_C}(v) = \operatorname{proj}_C(v) = \arg\min_{y\in C}\|y-v\|_2.
> $$
> *Hint:* Consider the minimization problem with the constraint $$y \in C$$.

> **Exercise 3: Existence, Uniqueness, and Non-Expansiveness of the Proximal Operator**
>
> **(a)** Prove that for any proper, lower semicontinuous, convex function $$g: \mathbb{R}^n\to \mathbb{R}\cup\{+\infty\}$$ and any $$v\in\mathbb{R}^n$$, the mapping
> $$
> \operatorname{prox}_{\eta, g}(v)
> $$
> is well-defined and unique.
>
> **(b)** Show that for all $$v,w\in\mathbb{R}^n$$,
> $$
> \|\operatorname{prox}_{\eta, g}(v)-\operatorname{prox}_{\eta, g}(w)\|_2 \le \|v-w\|_2.
> $$
> *Hint:* Use the optimality conditions and the monotonicity of the subdifferential.

> **Exercise 4: Differentiability and Lipschitz Continuity of the Moreau Envelope**
>
> **(a)** Prove that the Moreau envelope
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n}\left\{g(y) + \frac{1}{2\eta}\|y-v\|_2^2\right\}
> $$
> is differentiable with respect to $$v$$.
>
> **(b)** Show that
> $$
> \nabla M_{\eta, g}(v) = \frac{1}{\eta}\left(v-\operatorname{prox}_{\eta, g}(v)\right),
> $$
> and that this gradient is Lipschitz continuous with constant $$\frac{1}{\eta}$$.
> *Hint:* Leverage the non-expansiveness of the proximal operator.

> **Exercise 5: Moreau Envelope of the Absolute Value Function (Huber Loss)**
>
> **(a)** Let $$g(x)=|x|.$$ Derive the Moreau envelope
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}}\left\{|y|+\frac{1}{2\eta}(v-y)^2\right\},
> $$
> and show that it yields the Huber loss.
>
> **(b)** Identify the regions in $$v$$ where the envelope behaves quadratically versus linearly.
> *Hint:* Analyze the optimality conditions for different ranges of $$v$$.

> **Exercise 6: Moreau Envelope via Infimal Convolution**
>
> **(a)** Verify that
> $$
> M_{\eta, g}(v) = g \square \left(\frac{1}{2\eta}\|\cdot\|_2^2\right)(v),
> $$
> where
> $$
> (f \square h)(v)=\inf_{y\in\mathbb{R}^n}\{f(y)+h(v-y)\}.
> $$
>
> **(b)** Discuss the significance of this formulation in terms of regularization and duality.
> *Hint:* Reflect on the properties of infimal convolution in convex analysis.

---

### Further Reading

For deeper insights into proximal methods, the Moreau envelope, and their extensions to composite optimization, consider exploring:
- [Parikh and Boyd (2013) – Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf) (Section 3.2)
- [Rockafellar and Wets (2009) – Variational Analysis](https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf)
- [Candes (2015) – Advanced Topics in Convex Optimization](https://candes.su.domains/teaching/math301/Lectures/Moreau-Yosida.pdf)
- [Bauschke and Lucet (2011)](https://cmps-people.ok.ubc.ca/bauschke/Research/68.pdf)

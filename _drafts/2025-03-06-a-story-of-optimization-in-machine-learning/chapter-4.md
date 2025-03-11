---
layout: post
title: "A Story of Optimization in ML: Chapter 4 - Navigating Non-Smooth Terrain"
description: "Chapter 4 explores optimization in non-smooth loss landscapes, introducing proximal methods and the Moreau envelope as tools to handle functions that lack gradients everywhere."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["proximal methods", "non-smooth optimization", "Moreau envelope", "proximal gradient descent", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/nonsmooth_chapter4.gif # Placeholder image path
  alt: "Visual representation of non-smooth optimization and proximal methods" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 4: Navigating Non-Smooth Terrain – Beyond the Gradient

In our optimization journey, we've become comfortable with smooth loss landscapes, where the concept of a **gradient** guides our descent.  Gradient Descent and its variations thrive in these terrains, efficiently leading us towards the minimum.  But what happens when the landscape becomes… jagged?

Imagine stepping onto a terrain that's no longer gently sloping hills, but a fractured landscape of sharp cliffs, abrupt edges, and sudden corners. This is the world of **non-smooth optimization**.  Non-smoothness isn't just a theoretical curiosity; it arises naturally in many practical machine learning scenarios.

Think about these common elements in modern models:

*   **ReLU Activation:**  The ReLU (Rectified Linear Unit) activation function, $$\operatorname{ReLU}(x) = \max(0, x)$$, is incredibly popular in neural networks. But at the point $$x=0$$, it has a sharp kink – it's not differentiable there.
*   **L1 Regularization:**  To encourage simpler, sparser models, we often use L1 regularization, adding terms like 
$$|x|$$ to our loss function.  The absolute value function $$|x|$$ also has a sharp, non-differentiable point at zero.
*   **Constraints:**  Sometimes we impose explicit constraints on our model parameters – perhaps they must be non-negative, or lie within a specific range.  These constraints can also introduce non-smoothness into the optimization problem.

In these non-smooth landscapes, the gradient, our trusty guide, becomes unreliable.  At those sharp edges and corners, the gradient simply isn't well-defined in the usual sense.  Standard gradient-based methods might falter, struggling to find a consistent direction of descent.

So, how do we proceed when gradients fail us?  How do we navigate this non-smooth terrain?

We need to move **beyond the gradient** as our sole guide.  We need a more robust compass, one that can point us towards the minimum even in the absence of smooth gradients.  This is where **proximal methods** come to our rescue.

### Moving Beyond the Gradient: The Proximal Idea

The core idea of proximal methods is to replace the direct gradient step with a more robust update. Instead of blindly following the gradient, we formulate each step as a **mini-optimization problem**.  This mini-problem balances two things:

1.  **Reducing the Loss:** We still want to move towards lower loss, to descend further into the valley.
2.  **Staying Close to the Current Point:** We don't want to make wild, uncontrolled jumps, especially in these unpredictable landscapes. We want to take cautious, well-behaved steps.

Let's revisit the **backward Euler discretization** of gradient flow, even for non-smooth functions.  For a smooth function, we saw it led to:

$$
x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}.
$$

Remarkably, this formulation remains meaningful and useful even when the loss function $$L(y)$$ is non-smooth!  The added quadratic term, $$\frac{1}{2\eta}\|y - x_k\|_2^2$$, acts as a **regularizer**, smoothing things out and ensuring our update is well-defined and stable, even when gradients are not.

### The Proximal Operator: A Robust Compass

This leads us to the concept of the **proximal operator**.  For any function $$g(x)$$ (smooth or non-smooth) and a parameter $$\eta > 0$$, the **proximal operator** is defined as:

$$
\operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
$$

Think of it as a **robust compass**. Given a current point $$v$$, the proximal operator points us to a new point $$y = \operatorname{prox}_{\eta, g}(v)$$ that is a good compromise: it tries to make $$g(y)$$ as small as possible, while also staying reasonably close to our starting point $$v$$.  The parameter $$\eta$$ controls this balance: a smaller $$\eta$$ emphasizes proximity, a larger $$\eta$$ emphasizes minimizing $$g(y)$$.

For this to work well, we typically assume that the function $$g$$ has some nice properties: it should be **proper**, **lower semicontinuous**, and **convex**.  Under these conditions, we are guaranteed that the proximal operator is **well-defined** and **unique** – for any starting point $$v$$, there is always a single, best "proximal" point.

### Proximal Gradient Descent: For Composite Landscapes

Now, consider a loss landscape that is a combination of both smooth and non-smooth parts:

$$
L(x) = f(x) + g(x),
$$

where $$f(x)$$ is a smooth, differentiable function (the "smooth hills" part) and $$g(x)$$ is a non-smooth function (the "jagged edges" part).  We can combine our familiar gradient descent for the smooth part with a proximal step for the non-smooth part to create **Proximal Gradient Descent**.

The Proximal Gradient Descent algorithm works in two steps:

1.  **Gradient Step (Smooth Descent):**  First, we take a standard gradient descent step using the gradient of the smooth part $$f(x)$$:
    $$
    v = x_k - \eta\, \nabla f(x_k)
    $$
    This step moves us downhill in the smooth part of the landscape.

2.  **Proximal Step (Non-Smooth Correction):** Then, we apply the proximal operator of the non-smooth part $$g(x)$$ to the point $$v$$ we just obtained:
    $$
    x_{k+1} = \operatorname{prox}_{\eta, g}(v)
    $$
    This proximal step "corrects" our gradient step, pulling us towards points that minimize the non-smooth part $$g(x)$$ while keeping us reasonably close to where the gradient step took us.

By alternating these two steps, Proximal Gradient Descent allows us to effectively navigate composite loss landscapes, leveraging the smoothness where it exists and robustly handling the non-smooth parts.

### The Moreau Envelope: Smoothing the Rough Edges (Mathematically)

There's a beautiful mathematical way to understand why Proximal Gradient Descent works so well.  It turns out we can view it as standard gradient descent, but on a *smoothed* version of our non-smooth function! This smoothed version is called the **Moreau envelope**.

The **Moreau envelope** of a function $$g(x)$$ is defined as:

$$
M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
$$

Notice that the Moreau envelope is defined using the very same minimization problem as the proximal operator!  The proximal operator gives us the *minimizer* $$y = \operatorname{prox}_{\eta, g}(v)$$, while the Moreau envelope gives us the *minimum value* of this minimization problem, $$M_{\eta, g}(v)$$.

The truly remarkable property of the Moreau envelope is that, even if the original function $$g(x)$$ is non-differentiable, its Moreau envelope $$M_{\eta, g}(v)$$ is **always differentiable** (if $$g$$ is closed proper and convex).  It's a smooth approximation of the potentially rough function $$g(x)$$.

And even more surprisingly, the **gradient of the Moreau envelope** is directly related to the proximal operator itself:

$$
\nabla M_{\eta, g}(v) = \frac{1}{\eta}\left( v - \operatorname{prox}_{\eta, g}(v) \right).
$$

This equation reveals a deep insight: when we perform a proximal gradient step, we are, in essence, taking a standard gradient descent step, but not on the original non-smooth function $$g(x)$$, but on its smooth Moreau envelope $$M_{\eta, g}(x)$$.  Proximal Gradient Descent is effectively **Gradient Descent on the Moreau Envelope**.

[Consider adding a diagram here that visually compares a non-smooth function (e.g., absolute value function or ReLU) with its Moreau envelope, showing how the envelope "smooths out" the sharp corners.]

### Practical Considerations and Convergence

The parameter $$\eta$$ in proximal methods plays a dual role. It not only scales the quadratic proximity term, but also controls the degree of smoothing in the Moreau envelope.  A **smaller $$\eta$$** makes the Moreau envelope a *closer* approximation to the original non-smooth function, but potentially less smooth. A **larger $$\eta$$** yields a *smoother* Moreau envelope, but potentially a less accurate approximation of the original function. This trade-off can influence the convergence speed and accuracy of Proximal Gradient Descent.

Choosing the right learning rate (which is related to $$\eta$$) and other hyperparameters in Proximal Gradient Descent often involves empirical tuning, considering the curvature of the smooth part of the loss landscape and the degree of non-smoothness in the non-smooth component.

Proximal methods, particularly Proximal Gradient Descent, have become indispensable tools in machine learning, especially for problems involving sparsity (via L1 regularization), constraints, and non-smooth activations like ReLU. They provide a robust and principled way to navigate loss landscapes that go beyond the smooth, differentiable ideal.

### Bonus: Proximal Mapping and Subdifferentials

For convex functions, the proximal operator’s optimality condition can be expressed in terms of the subdifferential. For example, for

$$
x^\star = \operatorname{prox}_{\eta, g}(v),
$$

the first-order optimality condition is

$$
0 \in \partial g(x^\star) + \frac{1}{\eta}(x^\star - v),
$$

which can be rearranged to show

$$
\frac{1}{\eta}(v - x^\star) \in \partial g(x^\star).
$$

This establishes a connection between the proximal step and generalized gradients for non-smooth functions.

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

For deeper insights into proximal methods and the Moreau envelope, consider exploring:
- [Parikh and Boyd (2013) – Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf) (Section 3.2)
- [Rockafellar and Wets (2009) – Variational Analysis](https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf)
- [Candes (2015) – Advanced Topics in Convex Optimization](https://candes.su.domains/teaching/math301/Lectures/Moreau-Yosida.pdf)
- [Bauschke and Lucet (2011)](https://cmps-people.ok.ubc.ca/bauschke/Research/68.pdf)

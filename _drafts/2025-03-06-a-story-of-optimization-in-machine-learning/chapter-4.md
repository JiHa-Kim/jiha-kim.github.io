---
layout: post
title: "A Story of Optimization in ML: Chapter 4 - Navigating Non-Smooth Terrain"
description: "Chapter 4 explores optimization in non-smooth loss landscapes, introducing proximal methods and the Moreau envelope as tools to navigate functions without gradients everywhere."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["proximal methods", "non-smooth optimization", "Moreau envelope", "proximal gradient descent", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/nonsmooth_chapter4.gif # Placeholder image path
  alt: "Visual representation of non-smooth optimization and proximal methods" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 4: Navigating Non-Smooth Terrain - Beyond the Gradient

In our journey so far, we've relied heavily on the concept of the **gradient**. Gradient Descent, Momentum, and Adaptive methods all fundamentally depend on the gradient to guide our descent.  We’ve implicitly assumed that our loss landscapes are **smooth**, with well-defined gradients everywhere, like gently rolling hills.

But what if the terrain becomes… jagged?  Imagine a loss landscape that's not smooth, but **non-smooth**.  Think of a landscape with sharp edges, sudden cliffs, and corners, rather than gentle curves.  In mathematical terms, this means our loss function is not differentiable everywhere.

Why does this matter?  Why would we encounter non-smooth loss functions in machine learning?

Consider these common scenarios:

*   **ReLU Activation Function:** The Rectified Linear Unit (ReLU), defined as $$ReLU(x) = \max(0, x)$$, is a cornerstone of modern neural networks.  It introduces a sharp "kink" at $$x=0$$. At this point, the derivative is not uniquely defined.
*   **L1 Regularization:**  To encourage sparsity in our models, we often add an L1 regularization term to the loss function, like $$L_1(w) = \|w\|_1 = \sum_i |w_i|$$. The absolute value function $$|w_i|$$ is also non-differentiable at $$w_i = 0$$.
*   **Constraints:**  Sometimes we want to constrain our parameters to lie within a specific set, for example, enforcing non-negativity or limiting the parameter norm.  Constraints often lead to non-smooth optimization problems.

When we encounter non-smooth functions, the gradient, as we traditionally understand it, ceases to exist at certain points, like those sharp corners or edges.  Standard gradient-based methods, which rely on the gradient to determine the descent direction, can struggle or even fail to converge reliably in these non-smooth landscapes.

So, how do we navigate non-smooth terrain?  How do we optimize functions when gradients are not always available?

We need to move **beyond the gradient** in its strict, differentiable sense. We need tools that can handle these "kinks" and discontinuities.  This is where **proximal methods** come into play.

Let's revisit an idea we touched upon briefly in Chapter 1: the **Backward Euler discretization** of gradient flow. Recall the forward Euler discretization led us to standard Gradient Descent.  But what if we discretize the gradient flow differently, using the *backward* Euler method?

The backward Euler discretization of $$\frac{dx(t)}{dt} = -\nabla L(x(t))$$ is:

$$\frac{x_{k+1} - x_k}{\eta} = -\nabla L(x_{k+1})$$

This leads to the **implicit update rule**:

$$x_{k+1} = x_k - \eta \nabla L(x_{k+1})$$

This equation defines $$x_{k+1}$$ in terms of its *own gradient*, $$\nabla L(x_{k+1})$$.  It seems circular and difficult to solve directly.  However, we can rewrite it in a different, insightful form. Rearranging the terms, we get:

$$x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}$$

This seemingly implicit update is actually an **explicit minimization problem**!  We are choosing the next point $$x_{k+1}$$ by minimizing a combination of two terms:

1.  **The loss function itself, $$L(y)$$**: We still want to move towards lower loss.
2.  **A proximity term, $$\frac{1}{2\eta}\|y - x_k\|_2^2$$**:  We want to stay "close" to our current position $$x_k$$. This term penalizes large jumps and encourages stability.

This minimization formulation is the heart of **proximal methods**.  It introduces the concept of the **proximal operator**.

**The Proximal Operator: Your Compass in Non-Smooth Terrain**

For a (possibly non-smooth) function $$g(x)$$ and a parameter $$\eta > 0$$, the **proximal operator** of $$g$$, denoted as $$\operatorname{prox}_{\eta, g}(v)$$, is defined as:

$$
\operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
$$

Given a point $$v$$, the proximal operator finds a new point $$y = \operatorname{prox}_{\eta, g}(v)$$ that balances two competing goals:

*   **Minimize $$g(y)$$**:  We want to decrease the value of the function $$g$$.
*   **Stay close to $$v$$**: We don't want to move too far from our current point $$v$$. The quadratic term $$\frac{1}{2\eta}\|y - v\|_2^2$$ enforces this proximity. The parameter $$\eta$$ controls the strength of this proximity constraint.

The proximal operator acts like a **compass** in non-smooth terrain.  Even if the gradient of $$g$$ is undefined, the proximal operator still provides a well-defined "step" towards minimizing $$g$$, while ensuring we don't make wild, uncontrolled jumps.

Consider a composite loss function of the form $$L(x) = f(x) + g(x)$$, where $$f(x)$$ is a smooth, differentiable function, and $$g(x)$$ is a non-smooth function (e.g., the L1 regularization term).  We can combine gradient descent for the smooth part $$f(x)$$ with the proximal operator for the non-smooth part $$g(x)$$ to create **Proximal Gradient Descent**.

**Proximal Gradient Descent: Combining Smooth and Non-Smooth Descent**

The Proximal Gradient Descent algorithm for minimizing $$L(x) = f(x) + g(x)$$ proceeds as follows:

1.  **Gradient Step (Smooth Part):** Take a standard gradient descent step using the gradient of the smooth part $$f(x)$$:
    $$v = x_k - \eta \, \nabla f(x_k)$$
2.  **Proximal Step (Non-Smooth Part):** Apply the proximal operator of the non-smooth part $$g(x)$$ to the intermediate point $$v$$:
    $$x_{k+1} = \operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$

This two-step approach effectively handles both smooth and non-smooth components of the loss function.  The gradient step descends along the smooth part, and the proximal step "corrects" this step by accounting for the non-smoothness of $$g(x)$$.

Interestingly, Proximal Gradient Descent can be viewed as **Gradient Descent on the Moreau Envelope** of the non-smooth function $$g(x)$$.

**The Moreau Envelope: Smoothing the Rough Edges**

The **Moreau envelope** of a non-smooth function $$g(x)$$ is defined as:

$$
M_{\eta, g}(v) = \min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
$$

Notice that the Moreau envelope is *exactly* the minimization problem we solve in the proximal operator!  In fact, the proximal operator gives us the *minimizer* of this problem:  $$\operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$, and the Moreau envelope gives us the *minimum value*: $$M_{\eta, g}(v) = \min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\} = g(\operatorname{prox}_{\eta, g}(v)) + \frac{1}{2\eta}\|\operatorname{prox}_{\eta, g}(v) - v\|_2^2 $$.

A remarkable property of the Moreau envelope is that it is **always differentiable**, even if the original function $$g(x)$$ is not!  It provides a **smooth approximation** of the non-smooth function.  And it turns out that the gradient of the Moreau envelope is directly related to the proximal operator:

$$
\nabla M_{\eta, g}(v) = \frac{1}{\eta} \left( v - \operatorname{prox}_{\eta, g}(v) \right)
$$

Therefore, when we perform Proximal Gradient Descent, we are essentially performing standard Gradient Descent on a *smoothed* version of our loss function – the Moreau envelope of the non-smooth part.  The proximal step is implicitly computing a gradient step in this smoothed space.

While we won't delve deeply into the mathematical details of non-smooth analysis here, it's worth noting that the concept of the **subdifferential** generalizes the idea of the gradient to non-smooth functions.  For a convex function (which is often the case in optimization), the subdifferential at a point is a *set* of vectors, rather than a single vector, representing all possible "generalized gradients" at that point.  There's a deep connection between proximal operators and subdifferentials, which provides a more rigorous foundation for proximal methods.

> **Bonus Section (Optional): Proximal Mapping and Subdifferentials**
>
> For a convex function $$g$$, a vector $$s$$ is in the subdifferential of $$g$$ at $$x$$, denoted $$s \in \partial g(x)$$, if for all $$y$$:
>
> $$g(y) \geq g(x) + \langle s, y-x \rangle$$
>
> The first-order optimality condition for the proximal mapping problem
> $$
> x^\star = \operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
> can be expressed in terms of subdifferentials as:
>
> $$
> 0 \in \partial g(x^\star) + \frac{1}{\eta}(x^\star - v)
> $$
>
> This can be rewritten as:
>
> $$
> v - x^\star \in \eta \partial g(x^\star)  \quad \text{or} \quad  \frac{1}{\eta}(v - x^\star) \in \partial g(x^\star)
> $$
>
> This relationship shows that the proximal operator step $$x^\star = \operatorname{prox}_{\eta, g}(v)$$ moves from $$v$$ to a point $$x^\star$$ such that the "step" $$ \frac{1}{\eta}(v - x^\star) $$ is a subgradient of $$g$$ at $$x^\star$$.  This provides a deeper connection between proximal mappings and the generalized gradient concept of subdifferentials.

> **Exercise 1: Proximal Operator of the L1 Norm (Soft Thresholding)**
>
> Let
> $$g(x) := \|x\|_1 = \sum_{i=1}^n |x_i|$$.
> Show that the proximal operator of $$g$$ is given by the **soft thresholding operator**, $$S_{\tau}(v)$$, where $$\tau = \eta$$, and for each component $$i$$:
>
> $$
> [\operatorname{prox}_{\eta, \|\cdot\|_1}(v)]_i = S_{\eta}(v_i) =
> \begin{cases}
> v_i - \eta & \text{if } v_i > \eta \\
> v_i + \eta & \text{if } v_i < -\eta \\
> 0 & \text{if } |v_i| \leq \eta
> \end{cases}
> $$
>
> *Hint:* Solve the minimization problem
> $$\arg\min_{y} \left\{ \|y\|_1 + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$ component-wise. Consider the cases $$v_i > \eta$$, $$v_i < -\eta$$, and $$|v_i| \leq \eta$$ separately, by analyzing the subgradient of $$|y_i|$$.

> **Exercise 2: Proximal Operator of the Indicator Function (Projection)**
>
> Let $$C$$ be a closed convex set, and let $$\delta_C(x)$$ be the indicator function of $$C$$, defined as $$ \delta_C(x) = 0 $$ if $$x \in C$$ and $$ \delta_C(x) = +\infty $$ if $$x \notin C$$. Show that the proximal operator of $$\delta_C$$ is the **Euclidean projection onto the set $$C$**:
>
> $$
> \operatorname{prox}_{\eta, \delta_C}(v) = \operatorname{proj}_C(v) = \arg\min_{y \in C} \|y - v\|_2
> $$
>
> *Hint:* Consider the minimization problem $$\arg\min_{y} \left\{ \delta_C(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$.  If $$y \in C$$, then $$\delta_C(y) = 0$$, and we are minimizing $$\frac{1}{2\eta}\|y - v\|_2^2$$ subject to $$y \in C$$. If $$y \notin C$$, then $$\delta_C(y) = +\infty$$, so such $$y$$ cannot be a minimizer.

In summary, proximal methods provide us with a powerful toolkit for navigating non-smooth loss landscapes. By moving beyond gradients and embracing the concept of proximity, we can optimize a broader class of functions, including those with important non-smooth regularizers and constraints, expanding our ability to build more sophisticated and robust machine learning models. In the next chapter, we'll explore yet another perspective shift: moving beyond Euclidean space itself and considering optimization in non-Euclidean geometries.

> **Definition. Proximal Mapping**  
> Given a proper, lower semicontinuous, convex function $$ g: \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ and a parameter $$ \eta > 0 $$, the proximal mapping of $$ g $$ is defined as  
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$  
> This operator finds a point $$ y $$ that balances minimizing $$ g $$ while remaining close to $$ v $$.

> **Definition. Moreau Envelope**  
> The Moreau envelope of a proper lower semi-continuous convex function $$ g $$ with parameter $$ \eta > 0 $$ is given by  
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$
> It provides a smooth approximation of $$ g $$, and its gradient is closely related to the proximal mapping, making it a powerful tool in optimization.

### **Exercises**

> **Exercise 1: Existence, Uniqueness, and Non-Expansiveness of the Proximal Operator**
>
> **(a)** Let $$ g : \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ be a proper, lower semicontinuous, and convex function. Prove that for any $$ v\in\mathbb{R}^n $$ and any $$ \eta>0 $$, the proximal mapping
>
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> is well-defined and unique.
>
> **(b)** Show that the proximal operator is non-expansive; that is, for all $$ v,w\in\mathbb{R}^n $$, prove that
>
> $$
> \|\operatorname{prox}_{\eta, g}(v)-\operatorname{prox}_{\eta, g}(w)\|_2 \le \|v-w\|_2.
> $$
>
>
> *Hint:*
> Use the first-order optimality conditions for the minimization problem and the monotonicity of the subdifferential of $$ g $$.

---

> **Exercise 2: Differentiability and Lipschitz Continuity of the Moreau Envelope**
>
> **(a)** Prove that the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> of any proper lower semicontinuous convex function $$ g $$ is differentiable with respect to $$ v $$.
>
> **(b)** Show that its gradient is given by
>
> $$
> \nabla M_{\eta, g}(v) = \frac{1}{\eta} \left( v - \operatorname{prox}_{\eta, g}(v) \right),
> $$
>
> and prove that this gradient is Lipschitz continuous with Lipschitz constant $$ L = \frac{1}{\eta} $$.
>
>
> *Hint:*
> Relate the first-order optimality condition for the minimization defining $$ M_{\eta, g}(v) $$ with the proximal mapping, and use the non-expansiveness property established in Exercise 1.

---

> **Exercise 3: Smoothing Effect and Convergence of the Moreau Envelope**
>
> **(a)** For a given convex function $$ g $$, demonstrate that the Moreau envelope $$ M_{\eta, g} $$ provides a smooth approximation of $$ g $$. Discuss in detail how the quadratic term
>
> $$
> \frac{1}{2\eta}\|y - v\|_2^2
> $$
>
> facilitates smoothing even when $$ g $$ is non-differentiable.
>
> **(b)** Show that as $$ \eta \to 0 $$, the Moreau envelope converges pointwise to the original function $$ g $$; that is, prove
>
> $$
> \lim_{\eta\to 0} M_{\eta, g}(v) = g(v) \quad \text{for all } v\in\mathbb{R}^n.
> $$
>
>
> *Hint:*
> Consider the behavior of the minimization problem defining $$ M_{\eta, g}(v) $$ as the weight on the quadratic term becomes increasingly dominant.

---

> **Exercise 4: Moreau Envelope of the Absolute Value Function (Huber Loss)**
>
> The Huber loss function is a loss function used in robust statistics, that is less sensitive to outliers in data than the squared error loss.
>
> **(a)** Let 
> $$ g:\mathbb{R}\to\mathbb{R} $$ be defined as $$ g(x)=|x| $$. Derive the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}} \left\{ |y| + \frac{1}{2\eta}(v-y)^2 \right\},
> $$
>
> and show that it yields the Huber loss function.
>
> **(b)** Identify the regions in $$ v $$ for which the Moreau envelope has quadratic behavior versus linear behavior, and explain the intuition behind this smoothing effect.
>
>
> *Hint:*
> Analyze the optimality condition for $$ y $$ and consider the cases when $$ |v| $$ is small versus when $$ |v| $$ is large.

---

> **Exercise 5: Moreau Envelope of an Indicator Function and the Squared Distance Function**
>
> Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set. The indicator function $$ \delta_C(x) $$ is defined as
>
> $$
> \delta_C(x) =
> \begin{cases}
> 0 & \text{if } x\in C, \\
> +\infty & \text{if } x\notin C.
> \end{cases}
> $$
>
> The Euclidean distance from a point $$ v $$ to a set $$ C $$ is defined as $$ \operatorname{dist}(v,C) = \inf_{x \in C} \|v - x\|_2 $$.
>
> **(a)** Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set, and define the indicator function $$ \delta_C(x) $$ as above.
>
> Show that the Moreau envelope of $$ \delta_C $$ is given by
>
> $$
> M_{\eta, \delta_C}(v) = \frac{1}{2\eta}\operatorname{dist}(v,C)^2,
> $$
>
> where $$ \operatorname{dist}(v,C) $$ is the Euclidean distance from $$ v $$ to the set $$ C $$.
>
> **(b)** Explain why this result is significant in the context of projection methods and feasibility problems in optimization.
>
>
> *Hint:*
> Use the fact that the proximal mapping of $$ \delta_C $$ is the Euclidean projection onto $$ C $$.

---

> **Exercise 6: Moreau Envelope via Infimal Convolution**
>
> The infimal convolution of two functions $$ f $$ and $$ g $$ is defined as
>
> $$
> (f \square g)(x) = \inf_{y\in\mathbb{R}^n} \left\{ f(x-y) + g(y) \right\}.
> $$
>
> **(a)** An infimal convolution of two functions $$ f $$ and $$ g $$ is defined as above.
>
> Verify that the Moreau envelope of $$ g $$ can be expressed as the following infimal convolution:
>
> $$
> M_{\eta, g}(v) = g \square \left(\frac{1}{2\eta}\|\cdot\|_2^2\right)(v),
> $$
>
> **(b)** Discuss the significance of expressing the Moreau envelope as an infimal convolution in terms of regularization and duality.
>
>
> *Hint:*
> Discuss the properties of infimal convolution and its relation to Moreau envelope in the context of convex analysis and optimization.

An interesting identity: we have

$$
\operatorname{prox}_{\eta, g} = (\operatorname{id} + \eta \, \partial f)^{-1}
$$

where $$\operatorname{id}$$ is the identity function and $$\partial f$$ is the subdifferential of $$f$$. See [Parikh and Boyd (2013) - Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf) section 3.2 "Resolvent of subdifferential operator" for more details.

More information concerning proximal methods and the Moreau envelope can be found in [Rockafellar and Wets (2009) - VARIATIONAL ANALYSIS](https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf), [Candes (2015) - MATH 301: Advanced Topics in Convex Optimization Lecture 22](https://candes.su.domains/teaching/math301/Lectures/Moreau-Yosida.pdf) and [Bauschke and Lucet (2011)](https://cmps-people.ok.ubc.ca/bauschke/Research/68.pdf).

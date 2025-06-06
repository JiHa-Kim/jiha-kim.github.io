---
title: "Convex Analysis Part 6: Introduction to Convex Optimization Algorithms"
date: 2025-06-02 10:00 -0400
sort_index: 6
description: "A comprehensive overview of fundamental algorithms for convex optimization, their mathematical properties, convergence behaviors, and interrelationships. Covers Gradient Descent, Subgradient Method, Proximal Algorithms, Mirror Descent, Newton's Method, and more."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Optimization Algorithms
- Gradient Descent
- Subgradient Method
- Proximal Algorithms
- Mirror Descent
- Newton's Method
- Accelerated Methods
- Stochastic Gradient Descent
- Interior-Point Methods
- Crash Course
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  Never introduce any non-existant path, like an image.
  This causes build errors. For example, simply put image: # placeholder

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line: $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):
    text

    $$
    block
    $$

    text... or:

    $$block$$

    text...
  Use LaTeX commands for symbols as much as possible (e.g. $$\vert$$ for
  absolute value, $$\ast$$ for asterisk). Avoid using the literal vertical bar
  symbol; use \vert and \Vert instead.

  The syntax for lists is:

  1. $$inline$$ item
  2. item $$inline$$
  3. item

      $$
      block
      $$

      (continued) item
  4. item

  Here are examples of syntaxes that do **not** work:

  1. text
    $$
    block
    $$
    text

  2. text
    $$
    text
    $$

    text

  And the correct way to include multiple block equations in a list item:

  1. text

    $$
    block 1
    $$

    $$
    block 2
    $$

    (continued) text

  Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
  `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.

  Here are some blockquote templates you can use:

  <blockquote class="box-definition" markdown="1">
  <div class="title" markdown="1">
  **Definition.** The natural numbers $$\mathbb{N}$$
  </div>
  The natural numbers are defined as $$inline$$.

  $$
  block
  $$

  </blockquote>

  And a details block template:

  <details class="details-block" markdown="1">
  <summary markdown="1">
  **Tip.** A concise title goes here.
  </summary>
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  Similarly, for boxed environments you can define:
    - box-definition          # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - box-lemma               # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - box-proposition         # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - box-theorem             # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - box-example             # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple) (for example blocks with lightbulb icon)
    - box-info                # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)` (theme-defined)
    - box-tip                 # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)` (theme-defined)
    - box-warning             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)` (theme-defined)
    - box-danger              # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)` (theme-defined)

  For details blocks, use:
    - details-block           # main wrapper (styled like box-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

**Introduction**

Welcome to the sixth part of our Convex Analysis Crash Course! Having explored the foundational theory of convex sets, functions, standard optimization problem formulations, the powerful concept of duality (including KKT conditions and the Fenchel conjugate), we now turn our attention to a crucial question: How do we actually *solve* these convex optimization problems? This post aims to introduce some of the most fundamental iterative algorithms designed for this purpose.

Our goal here is to understand the intuition and core mechanics ("what") of these algorithms, delve into their key mathematical properties, and grasp the reasons for their effectiveness ("why"). Most convex optimization algorithms are iterative, meaning they generate a sequence of points $$x^{(0)}, x^{(1)}, x^{(2)}, \dots$$ that hopefully converge to an optimal solution $$x^\ast$$. While a deep dive into convergence analysis for each method is extensive and beyond the scope of this introductory crash course, we will highlight the main convergence characteristics.

Understanding these foundational convex algorithms is paramount, not just for solving convex problems directly, but also for building intuition for the more complex optimizers frequently encountered in machine learning, especially in non-convex settings like deep learning. Grasping their mathematical properties helps in analyzing their efficiency, predicting their convergence behavior, and assessing their suitability for various problem scales, including large-scale and distributed computation scenarios.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Key Concepts for Algorithm Analysis**
</div>
The behavior and convergence rates of optimization algorithms often depend on specific properties of the objective function. Here are a few key concepts we'll refer to:

-   **$$L$$-smoothness (Lipschitz continuous gradient):** A differentiable function $$f$$ is $$L$$-smooth if its gradient $$\nabla f$$ is Lipschitz continuous with constant $$L \ge 0$$:

    $$
    \Vert \nabla f(x) - \nabla f(y) \Vert_2 \le L \Vert x - y \Vert_2 \quad \forall x, y \in \mathrm{dom}(f)
    $$

    This implies a quadratic upper bound: $$f(y) \le f(x) + \nabla f(x)^T (y-x) + \frac{L}{2} \Vert y-x \Vert_2^2$$.

-   **$$\mu$$-strong convexity:** A function $$f$$ is $$\mu$$-strongly convex (with $$\mu > 0$$) if for some (any, if differentiable) subgradient $$g_x \in \partial f(x)$$:

    $$
    f(y) \ge f(x) + g_x^T (y-x) + \frac{\mu}{2} \Vert y-x \Vert_2^2 \quad \forall x, y \in \mathrm{dom}(f)
    $$

    Strong convexity implies a unique minimizer and provides a quadratic lower bound on the function's growth.

-   **Condition Number ($$\kappa$$):** For an $$L$$-smooth and $$\mu$$-strongly convex function, the condition number is defined as $$\kappa = L/\mu \ge 1$$. This ratio often dictates the convergence speed of first-order methods. A large $$\kappa$$ (an ill-conditioned problem) typically means slower convergence.

-   **Oracle Model:** Algorithms are often analyzed based on the type of information they can access about the function at each iteration (e.g., function value only (zeroth-order), gradient/subgradient (first-order), Hessian (second-order)).
</blockquote>

**1. The General Descent Method Framework**

Many optimization algorithms, particularly for continuous problems, fall under a general iterative framework. At each iteration $$k$$, we are at a point $$x^{(k)}$$ and aim to find a new point $$x^{(k+1)}$$ that is "better" in some sense, typically meaning it has a lower objective function value.

The update rule often takes the form:

$$
x^{(k+1)} = x^{(k)} + t_k d^{(k)}
$$

where:
-   $$x^{(k)} \in \mathbb{R}^n$$ is the current iterate.
-   $$d^{(k)} \in \mathbb{R}^n$$ is the **search direction**. This vector determines the direction in which we move from $$x^{(k)}$$.
-   $$t_k > 0$$ is the **step size** (often called the learning rate in machine learning). This scalar determines how far we move along the search direction $$d^{(k)}$$.

For a method to be a "descent method" when minimizing a differentiable function $$f(x)$$, the search direction $$d^{(k)}$$ should ideally satisfy the **descent condition**:

$$
\nabla f(x^{(k)})^T d^{(k)} < 0
$$

This condition ensures that the directional derivative of $$f$$ at $$x^{(k)}$$ in the direction $$d^{(k)}$$ is negative. Consequently, for a sufficiently small positive step size $$t_k$$, we are guaranteed to have $$f(x^{(k+1)}) < f(x^{(k)})$$, meaning we make progress in reducing the objective function value.

Throughout this post, we will consider the general convex optimization problem template:

$$
\min_{x \in \mathcal{D}} F(x) \quad \text{or} \quad \min_{x \in \mathcal{D}} (f(x) + h(x))
$$

where $$F(x)$$ (or $$f(x)+h(x)$$) is a convex function, and $$\mathcal{D}$$ is a convex set representing the feasible region. Often, $$f(x)$$ is assumed to be smooth (differentiable), while $$h(x)$$ might be convex but non-smooth (e.g., a regularization term like the L1 norm, or an indicator function for constraints).

**2. Gradient Descent (GD)**

Gradient Descent is arguably the most fundamental first-order optimization algorithm for minimizing a differentiable convex function $$f(x)$$.

*   **Objective:** Minimize a differentiable convex function $$f(x)$$.
*   **Search Direction:** The negative gradient, $$d^{(k)} = -\nabla f(x^{(k)})$$. This is the direction of steepest descent in the Euclidean norm.
*   **Update Rule:**

    $$
    x^{(k+1)} = x^{(k)} - t_k \nabla f(x^{(k)})
    $$

*   **Step Size ($$t_k$$) Selection:**
    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Common Step Size Strategies for GD**
    </summary>
    -   **Fixed step size:** A small constant $$t_k = t$$. For convergence with an $$L$$-smooth convex function, $$t$$ typically needs to be in $$(0, 1/L]$$ for guaranteed monotone decrease ($$f(x^{(k+1)}) \le f(x^{(k)})$$). A range up to $$(0, 2/L)$$ can ensure convergence of $$x^{(k)}$$ to $$x^\ast$$ under $$L$$-smoothness, but monotonicity might be lost. $$t=1/L$$ is a common choice.
    -   **Diminishing step sizes:** A sequence $$t_k$$ such that $$t_k > 0$$, $$t_k \to 0$$, and $$\sum_{k=0}^\infty t_k = \infty$$ (e.g., $$t_k = c/\sqrt{k+1}$$ or $$t_k = c/(k+1)$$).
    -   **Line search:** Choose $$t_k$$ at each iteration to (approximately) minimize $$f(x^{(k)} - t \nabla f(x^{(k)}))$$ with respect to $$t > 0$$. Common methods include backtracking line search (which checks the Armijo condition) or, for simple functions, exact line search.
    </details>
*   **Mathematical Properties & Convergence:**
    *   Assumes $$f$$ is convex and differentiable. If $$f$$ is also $$L$$-smooth:
        *   For convex $$f$$: The function value error $$f(x^{(k)}) - f^\ast$$ converges as $$O(1/k)$$ with an appropriate constant step size (e.g., $$1/L$$) or line search.
        *   For $$\mu$$-strongly convex $$f$$: The convergence is linear (or geometric), i.e., $$f(x^{(k)}) - f^\ast = O(\rho^k)$$ for some $$\rho < 1$$ (e.g., $$\rho \approx (1 - \mu/L)$$ if $$t_k=1/L$$). The distance to optimum $$\Vert x^{(k)} - x^\ast \Vert^2$$ also converges linearly.
    *   The performance is sensitive to the conditioning ($$\kappa = L/\mu$$) of the problem; ill-conditioned problems converge slowly.

    **Constrained Problems: Projected Gradient Descent**
    When minimizing $$f(x)$$ subject to $$x \in C$$, where $$C$$ is a closed convex set, Gradient Descent can be adapted to Projected Gradient Descent:

    $$
    x^{(k+1)} = \Pi_C (x^{(k)} - t_k \nabla f(x^{(k)}))
    $$

    where $$\Pi_C(z) = \arg\min_{x \in C} \Vert x-z \Vert_2^2$$ is the Euclidean projection of $$z$$ onto the set $$C$$. The convergence properties are similar to unconstrained GD if the projection is computationally feasible.

**3. Subgradient Method (SM)**

When the convex function $$f(x)$$ is not necessarily differentiable, we can use the Subgradient Method.

*   **Objective:** Minimize a convex function $$f(x)$$, which may be non-differentiable.
*   **Search Direction Idea:** Instead of a gradient, we use any subgradient $$g^{(k)} \in \partial f(x^{(k)})$$ (the subdifferential of $$f$$ at $$x^{(k)}$$).
*   **Update Rule:**

    $$
    x^{(k+1)} = x^{(k)} - t_k g^{(k)}, \quad \text{where } g^{(k)} \in \partial f(x^{(k)})
    $$

*   **Mathematical Properties & Convergence:**
    *   The subgradient method is **not necessarily a descent method**. That is, it's possible that $$f(x^{(k+1)}) > f(x^{(k)})$$ for some iterations. However, it can be shown that the distance to the optimal set $$\Vert x^{(k)} - x^\ast \Vert_2$$ tends to decrease, or the best function value found so far, $$\min_{0 \le i \le k} f(x^{(i)})$$, converges to $$f^\ast$$.
    *   **Step size selection ($$t_k$$) is crucial** and more delicate than in GD.
        <details class="details-block" markdown="1">
        <summary markdown="1">
        **Common Step Size Strategies for Subgradient Method**
        </summary>
        -   **Constant step size:** $$t_k = t$$. Leads to convergence to a neighborhood of the optimum.
        -   **Diminishing step sizes satisfying non-summable but square-summable condition:** $$t_k > 0$$, $$\sum_{k=0}^\infty t_k = \infty$$ and $$\sum_{k=0}^\infty t_k^2 < \infty$$ (e.g., $$t_k = a/(k+1)$$ or $$t_k = a/\sqrt{k+1}$$ for some $$a > 0$$). These ensure convergence of $$\min_{i \le k} f(x^{(i)})$$ to $$f^\ast$$.
        -   Other common choices exist, like Polyak's step size (if $$f^\ast$$ is known or estimated).
        </details>
    *   Convergence is generally slower than Gradient Descent for smooth functions. For a general convex function (Lipschitz continuous), the error in terms of the best objective value, $$\min_{0 \le i \le k} f(x^{(i)}) - f^\ast$$, typically converges as $$O(1/\sqrt{k})$$. This means achieving an $$\epsilon$$-accurate solution requires $$O(1/\epsilon^2)$$ iterations.
    *   For non-smooth $$L$$-Lipschitz convex functions, Nesterov (2005) proposed an "optimal" variant achieving $$O(1/k)$$ convergence for $$f(x_k) - f^\ast$$ (or more precisely, for the gap of an averaged iterate), often requiring two step-sizes and computation of a prox-like step involving the Bregman divergence used in the mirror map.

    **Constrained Problems: Projected Subgradient Method**
    Similar to Projected GD, for problems $$\min_{x \in C} f(x)$$:

    $$
    x^{(k+1)} = \Pi_C (x^{(k)} - t_k g^{(k)})
    $$

**4. The Concept of Acceleration (Nesterov's Momentum)**

A significant breakthrough in first-order methods was the development of "accelerated" methods, pioneered by Yurii Nesterov. These methods often achieve provably faster convergence rates than standard gradient descent for smooth convex functions, without a significant increase in per-iteration cost.

*   **Core Idea:** Acceleration is typically achieved by introducing a "momentum" term. Instead of just using the gradient at the current point, accelerated methods intelligently combine information from previous iterates and gradients. The iterates $$x^{(k)}$$ are often updated based on an auxiliary sequence $$y^{(k)}$$, which can be thought of as an "extrapolation" or "search point."
*   **General Update Forms (Conceptual):** While specific formulas vary, a common structure for minimizing a smooth $$f(x)$$ looks like:
    1.  Compute an extrapolated point: $$y^{(k)} = x^{(k)} + \beta_k (x^{(k)} - x^{(k-1)})$$ (momentum step, $$\beta_k \ge 0$$).
    2.  Take a gradient step from this extrapolated point: $$x^{(k+1)} = y^{(k)} - t_k \nabla f(y^{(k)})$$.
    The choice of $$\beta_k$$ and $$t_k$$ is critical. For example, Nesterov's original scheme for unconstrained minimization often uses $$\beta_k = \frac{k-1}{k+2}$$.
*   **Impact on Convergence Rates:**
    *   For $$L$$-smooth convex functions: Accelerated methods achieve a convergence rate of $$f(x^{(k)}) - f^\ast = O(1/k^2)$$, which is significantly faster than the $$O(1/k)$$ rate of standard GD.
    *   For $$L$$-smooth and $$\mu$$-strongly convex functions: They achieve a linear convergence rate of $$O(\rho^k)$$ with $$\rho \approx (1 - \sqrt{\mu/L})$$, improving upon GD's $$\rho \approx (1 - \mu/L)$$, especially when $$\kappa = L/\mu$$ is large.
    These rates are known to be optimal (in terms of dependence on $$k$$, $$L$$, and $$\mu$$) for first-order methods that only use gradient and function value information.
*   **Applicability:** The acceleration principle is quite general and has been applied to various settings, including proximal gradient methods (leading to FISTA, discussed later), coordinate descent, and stochastic methods. A notable characteristic is that accelerated methods are often not monotonic descent methods; the objective function value may occasionally increase.

**5. Stochastic Gradient Descent (SGD)**

In many large-scale machine learning problems, the objective function $$f(x)$$ is a sum of many component functions or an expectation:

$$
f(x) = \frac{1}{m} \sum_{i=1}^m f_i(x) \quad \text{or} \quad f(x) = \mathbb{E}_\xi [f(x; \xi)]
$$

Computing the full gradient $$\nabla f(x)$$ can be very expensive if $$m$$ is large. Stochastic Gradient Descent (SGD) addresses this by using a noisy but computationally cheap estimate of the gradient at each iteration.

*   **Core Idea:** At iteration $$k$$, instead of computing the full gradient $$\nabla f(x^{(k)})$$, SGD uses a **stochastic gradient** $$\hat{g}^{(k)}$$. This $$\hat{g}^{(k)}$$ is an unbiased estimator of the true gradient, meaning $$\mathbb{E}[\hat{g}^{(k)} \mid x^{(k)}] = \nabla f(x^{(k)})$$ (the expectation is over the random selection of the sample or mini-batch).
    *   For finite sums: $$\hat{g}^{(k)} = \nabla f_i(x^{(k)})$$ where $$i$$ is chosen uniformly at random from $$\{1, \dots, m\}$$, or $$\hat{g}^{(k)} = \frac{1}{\vert \mathcal{B}_k \vert} \sum_{j \in \mathcal{B}_k} \nabla f_j(x^{(k)})$$ where $$\mathcal{B}_k$$ is a randomly chosen mini-batch of indices.
    *   For expectations: $$\hat{g}^{(k)} = \nabla f(x^{(k)}; \xi_k)$$ where $$\xi_k$$ is a sample drawn from the underlying distribution.
*   **Update Rule:**

    $$
    x^{(k+1)} = x^{(k)} - t_k \hat{g}^{(k)}
    $$

*   **Mathematical Properties & Convergence:**
    *   **Low per-iteration cost:** The main advantage of SGD. Independent of $$m$$ if using single samples.
    *   **Noisy updates:** The path taken by SGD iterates is much noisier than GD.
    *   **Step size ($$t_k$$):** Typically diminishing step sizes are required for convergence to the true optimum (e.g., $$t_k \propto 1/k$$ or $$t_k \propto 1/\sqrt{k}$$). Constant step sizes usually lead to convergence to a neighborhood of the optimum.
    *   **Convergence (in expectation, assuming bounded variance of stochastic gradients $$\mathbb{E}[\Vert \hat{g}^{(k)} - \nabla f(x^{(k)}) \Vert^2] \le \sigma^2$$):**
        *   For $$L$$-smooth convex $$f$$: With $$t_k \propto 1/\sqrt{k}$$, $$\mathbb{E}[f(\bar{x}^{(K)})] - f^\ast = O(1/\sqrt{K})$$, where $$\bar{x}^{(K)}$$ is an average of iterates.
        *   For $$L$$-smooth and $$\mu$$-strongly convex $$f$$: With $$t_k = \Theta(1/(\mu k))$$, $$\mathbb{E}[f(x^{(K)})] - f^\ast = O(1/(\mu K))$$ or $$\mathbb{E}[\Vert x^{(K)} - x^\ast \Vert^2] = O(\sigma^2/(\mu^2 K))$$.
    *   The variance of the stochastic gradients plays a key role in the convergence analysis.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Variance Reduction in SGD**
</div>
For finite sum problems ($$f(x) = \frac{1}{m} \sum f_i(x)$$), standard SGD suffers from variance that doesn't diminish even near the optimum, preventing linear convergence for strongly convex functions with constant or slowly decaying step sizes. **Variance-Reduced SGD** methods like SVRG, SAGA, and SARAH cleverly use occasional full gradient computations or store past gradients to construct stochastic gradient estimators with diminishing variance. This allows them to achieve linear convergence rates, often $$O((m+\kappa)\log(1/\epsilon))$$, rivaling full gradient methods while potentially being much faster in terms of total computation if $$m \gg \kappa$$.
</blockquote>

**6. Other Notable First-Order Methods (Briefly)**

*   **Coordinate Descent (CD):**
    *   Instead of updating all components of $$x$$ simultaneously, CD methods update one coordinate (or a block of coordinates) at a time, while keeping others fixed. The update for coordinate $$j$$ might involve minimizing $$f$$ with respect to $$x_j$$: $$x_j^{(k+1)} = \arg\min_{z} f(x_1^{(k)}, \dots, x_{j-1}^{(k)}, z, x_{j+1}^{(k)}, \dots, x_n^{(k)})$$.
    *   Particularly efficient if these 1D minimizations are cheap, or if the objective function has a structure that simplifies coordinate-wise updates (e.g., quadratic loss + L1 regularization).
    *   **Randomized Coordinate Descent (RCD):** If coordinates are chosen uniformly at random, for an $$L$$-smooth function with coordinate-wise Lipschitz constants $$L_i$$, the expected suboptimality after $$k$$ iterations can be $$O(d \sum L_i R^2 / k)$$ where $$R$$ bounds $$\Vert x^{(0)}-x^\ast \Vert_\infty$$ and $$d$$ is dimension. RCD can be faster than GD on sparse datasets or when some coordinates have much larger impact.

*   **Frank-Wolfe (Conditional Gradient):**
    *   Designed for minimizing a smooth convex function $$f(x)$$ over a **compact** convex set $$C$$.
    *   **Core Idea:** At each iteration, it linearizes the objective $$f(x^{(k)}) + \nabla f(x^{(k)})^T (s - x^{(k)})$$ and minimizes this over $$C$$ to find $$s^{(k)} = \arg\min_{s \in C} \nabla f(x^{(k)})^T s$$. The next iterate is $$x^{(k+1)} = (1-\gamma_k) x^{(k)} + \gamma_k s^{(k)}$$.
    *   **Advantages:** "Projection-free" if linear minimization over $$C$$ is easier than projection (e.g., $$C$$ is L1-ball, simplex). Iterates often inherit sparsity.
    *   **Convergence:** Typically $$f(x^{(k)}) - f^\ast = O(1/k)$$.

**7. Proximal Algorithms**

Proximal algorithms are a powerful class of methods, especially for solving optimization problems where the objective function is a sum of two convex terms, one of which might be smooth and the other non-smooth but "simple" in a specific way. They are built around the concept of the proximal operator.

**The Proximal Operator**

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Proximal Operator
</div>
For a proper, closed, convex function $$h: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ and a parameter $$t > 0$$, the **proximal operator** $$\mathbf{prox}_{th}(v)$$ of $$h$$ at point $$v$$ is defined as:

$$
\mathbf{prox}_{th}(v) = \arg\min_x \left( h(x) + \frac{1}{2t} \Vert x - v \Vert_2^2 \right)
$$

The proximal operator finds a point $$x$$ that makes a trade-off: it tries to minimize $$h(x)$$ while staying close to the input point $$v$$ (in Euclidean distance). The parameter $$t$$ controls this trade-off.
</blockquote>

*   **Examples of Proximal Operators:**
    *   If $$h(x) = 0$$ (zero function), then $$\mathbf{prox}_{th}(v) = v$$.
    *   If $$h(x) = \mathcal{I}_C(x)$$ (the indicator function of a closed convex set $$C$$), then $$\mathbf{prox}_{th}(v) = \Pi_C(v)$$, the Euclidean projection of $$v$$ onto $$C$$.
        ($$\mathcal{I}_C(x) = 0$$ if $$x \in C$$, and $$\infty$$ if $$x \notin C$$).
    *   If $$h(x) = \lambda \Vert x \Vert_1$$ (L1 norm, commonly used for sparsity), then $$\mathbf{prox}_{t (\lambda \Vert \cdot \Vert_1)}(v)$$ is the **soft-thresholding operator**, applied element-wise:

        $$
        (\mathbf{prox}_{t\lambda \Vert \cdot \Vert_1}(v))_i = S_{t\lambda}(v_i) = \mathrm{sgn}(v_i) \max(0, \vert v_i \vert - t\lambda)
        $$

    The efficiency of proximal algorithms often hinges on being able to compute the proximal operator of $$h$$ easily (analytically or with a very fast algorithm).

**Proximal Point Algorithm (PPA)**

This algorithm is conceptually fundamental for minimizing a single (possibly non-smooth) convex function $$h(x)$$.

*   **Objective:** Minimize a convex function $$h(x)$$.
*   **Update Rule:**

    $$
    x^{(k+1)} = \mathbf{prox}_{t_k h}(x^{(k)})
    $$

*   **Intuition:** Implicit method; find new point making progress on $$h$$ without moving too far.
    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Interpretations: Backward Euler and Moreau Envelope**
    </summary>
    The Proximal Point Algorithm has deep connections to continuous-time dynamics and related optimization concepts.

    1.  **Connection to Gradient Flow (Backward Euler Discretization):**
        The PPA update rule can be seen as an implicit discretization of a gradient flow. Recall the definition of the proximal operator:

        $$
        x^{(k+1)} = \mathbf{prox}_{t_k h}(x^{(k)}) = \arg\min_x \left( h(x) + \frac{1}{2t_k} \Vert x - x^{(k)} \Vert_2^2 \right)
        $$

        The first-order optimality condition for this minimization is (using the subdifferential $$\partial h$$ since $$h$$ may be non-smooth):

        $$
        0 \in \partial h(x^{(k+1)}) + \frac{1}{t_k} (x^{(k+1)} - x^{(k)})
        $$

        Rearranging this gives:

        $$
        \frac{x^{(k+1)} - x^{(k)}}{t_k} \in -\partial h(x^{(k+1)})
        $$

        This is precisely the **Backward Euler** (or implicit Euler) discretization of the (sub)gradient flow differential inclusion:

        $$
        \dot{x}(t) \in -\partial h(x(t))
        $$

        In contrast, if $$h$$ were smooth, the standard Gradient Descent update $$x^{(k+1)} = x^{(k)} - t_k \nabla h(x^{(k)})$$ can be rewritten as $$\frac{x^{(k+1)} - x^{(k)}}{t_k} = -\nabla h(x^{(k)})$$, which is a **Forward Euler** (or explicit Euler) discretization of $$\dot{x}(t) = -\nabla h(x(t))$$. Implicit schemes like backward Euler are often more stable, especially for stiff problems, which aligns with the known stability and robustness of the PPA.

    2.  **Connection to Moreau Envelope (Gradient Descent):**
        The **Moreau envelope** (or Moreau-Yosida regularization) of $$h$$ with parameter $$t > 0$$ is defined as:

        $$
        M_{th}(v) = \min_x \left( h(x) + \frac{1}{2t} \Vert x - v \Vert_2^2 \right)
        $$

        Notice that the minimizer in this definition is precisely $$\mathbf{prox}_{th}(v)$$, so $$M_{th}(v) = h(\mathbf{prox}_{th}(v)) + \frac{1}{2t} \Vert \mathbf{prox}_{th}(v) - v \Vert_2^2$$.
        The Moreau envelope $$M_{th}(v)$$ is a smoothed version of $$h(x)$$; it is always convex if $$h$$ is, and it is continuously differentiable (for convex, proper, l.s.c. $$h$$) with gradient:

        $$
        \nabla M_{th}(v) = \frac{1}{t} (v - \mathbf{prox}_{th}(v))
        $$

        Now, consider applying a gradient descent step to minimize $$M_{t_k h}(v)$$ starting from $$v=x^{(k)}$$, using a step size equal to $$t_k$$:

        $$
        v' = x^{(k)} - t_k \nabla M_{t_k h}(x^{(k)})
        $$

        Substituting the gradient of the Moreau envelope:

        $$
        v' = x^{(k)} - t_k \left( \frac{1}{t_k} (x^{(k)} - \mathbf{prox}_{t_k h}(x^{(k)})) \right)
        $$

        $$
        v' = x^{(k)} - (x^{(k)} - \mathbf{prox}_{t_k h}(x^{(k)}))
        $$

        $$
        v' = \mathbf{prox}_{t_k h}(x^{(k)})
        $$

        Thus, the PPA update $$x^{(k+1)} = \mathbf{prox}_{t_k h}(x^{(k)})$$ is equivalent to performing a single gradient descent step on the Moreau envelope $$M_{t_k h}$$ at $$x^{(k)}$$ with step size $$t_k$$. This provides another perspective: PPA implicitly smooths the function via its Moreau envelope and then takes a gradient step on this smoothed version.
    </details>
*   **Convergence:** If $$h$$ is $$\mu$$-strongly convex, PPA with a constant step size $$t_k=t>0$$ converges linearly. For general convex $$h$$, $$h(x^{(k)})-h^\ast = O(1/k)$$ (using $$h$$ instead of $$f$$ as per the objective) with appropriate $$t_k$$.

**Proximal Gradient Method (PGM) / Forward-Backward Splitting (FBS) / ISTA**

This method is highly popular for solving composite optimization problems of the form $$\min_x F(x) = f(x) + h(x)$$, where:
-   $$f(x)$$ is convex and differentiable (smooth part).
-   $$h(x)$$ is convex and "proximable" (non-smooth part, but its proximal operator is easy to compute).

*   **Update Rule:**

    $$
    x^{(k+1)} = \mathbf{prox}_{t_k h} (x^{(k)} - t_k \nabla f(x^{(k)}))
    $$

*   **Intuition (Two-step process):**
    1.  **Forward step (Gradient step on $$f$$):** $$z^{(k)} = x^{(k)} - t_k \nabla f(x^{(k)})$$.
    2.  **Backward step (Proximal step on $$h$$):** $$x^{(k+1)} = \mathbf{prox}_{t_k h}(z^{(k)})$$.
*   **Mathematical Properties & Convergence:**
    *   If $$f$$ is $$L_f$$-smooth, and a step size like $$t_k \in (0, 2/L_f)$$ is used (often $$t_k = 1/L_f$$ or chosen by backtracking line search):
        *   For convex $$F(x)$$: $$F(x^{(k)}) - F^\ast = O(1/k)$$.
        *   If $$F(x)$$ is $$\mu_F$$-strongly convex: Linear convergence, $$F(x^{(k)}) - F^\ast = O(\rho^k)$$.
*   **Relationship:** Generalizes GD (if $$h=0$$), Projected GD (if $$h = \mathcal{I}_C(x)$$), and PPA (if $$f=0$$).

    **Accelerated Proximal Gradient (e.g., FISTA)**
    The acceleration principle (Section 4) can be applied to PGM. A prominent example is **FISTA (Fast ISTA)**.
    *   FISTA uses a Nesterov-like momentum scheme. A common form:

        $$
        y^{(k)} = x^{(k)} + \beta_k (x^{(k)} - x^{(k-1)})
        $$

        $$
        x^{(k+1)} = \mathbf{prox}_{t_k h} (y^{(k)} - t_k \nabla f(y^{(k)}))
        $$

        where $$\beta_k$$ is chosen carefully (e.g., $$\beta_k = \frac{k-1}{k+2}$$ if $$t_k$$ is constant).
    *   **Convergence:** For $$L_f$$-smooth $$f$$ and convex $$h$$:
        *   For convex $$F(x)$$: $$F(x^{(k)}) - F^\ast = O(1/k^2)$$.
        *   If $$F(x)$$ is $$\mu_F$$-strongly convex, FISTA itself may not achieve the accelerated linear rate without modifications. Variants (e.g., restarted FISTA or FISTA adapted for strong convexity) achieve a faster linear convergence rate, $$O(\rho^k)$$ with $$\rho \approx (1 - \sqrt{\mu_F/L_f})$$.

**8. Mirror Descent (MD)**

Mirror Descent is a generalization of gradient and subgradient descent that can be more effective when the optimization variables lie in a non-Euclidean space or when non-Euclidean "distances" are more natural for the problem structure.

*   **Core Idea:** Instead of performing gradient steps directly in the "primal space" using Euclidean geometry, Mirror Descent maps the current iterate to a "dual space" via a **mirror map**, performs a gradient-like update there, and maps back. The geometry is defined by a **Bregman divergence**.
*   **Key Components:**
    1.  **Mirror Map (Potential Function):** $$\psi(x)$$, strongly convex and differentiable on $$\mathcal{D}$$.
    2.  **Bregman Divergence:** $$D_\psi(x, y) = \psi(x) - \psi(y) - \nabla \psi(y)^T (x-y)$$.
*   **Update Rule (Conceptual Form):** For $$g^{(k)} \in \partial f(x^{(k)})$$,

    $$
    x^{(k+1)} = \arg\min_{x \in \mathcal{D}} \{ t_k \langle g^{(k)}, x \rangle + D_\psi(x, x^{(k)}) \}
    $$

*   **Relationship to Gradient Descent:** Standard (Sub)Gradient Descent is a special case if $$\psi(x) = \frac{1}{2} \Vert x \Vert_2^2$$.
*   **Applications:** Optimization over probability simplex (using KL divergence), matrix spaces, online learning.
*   **Convergence:** For general $$L$$-Lipschitz convex $$f$$, Mirror Descent typically achieves $$O(L R/\sqrt{k})$$ for $$f(\bar{x}^{(k)}) - f^\ast$$ (where $$R^2$$ bounds $$D_\psi(x^\ast, x^{(0)})$$). For non-smooth functions, Nesterov's dual averaging or variants can achieve $$O(1/k)$$ with specific conditions, similar to the accelerated subgradient methods.

**9. Newton's Method**

Newton's method is a second-order optimization algorithm that uses curvature information (Hessian matrix) for faster convergence.

*   **Objective:** Minimize a twice differentiable, typically strongly convex function $$f(x)$$.
*   **Core Idea:** Minimize the second-order Taylor expansion of $$f$$ at $$x^{(k)}$$.
*   **Search Direction (Newton Step):** $$d_N^{(k)} = -(\nabla^2 f(x^{(k)}))^{-1} \nabla f(x^{(k)})$$.
*   **Update Rule:**
    *   **Pure Newton Method:** $$x^{(k+1)} = x^{(k)} + d_N^{(k)}$$ (i.e., $$t_k=1$$).
    *   **Damped Newton Method:** $$x^{(k+1)} = x^{(k)} + t_k d_N^{(k)}$$, with $$t_k \in (0,1]$$ from line search.
*   **Convergence:**
    *   **Local Quadratic Convergence:** If $$f$$ is strongly convex and Hessian is Lipschitz near $$x^\ast$$, and $$x^{(k)}$$ is close, $$\Vert x^{(k+1)} - x^\ast \Vert_2 \le C \Vert x^{(k)} - x^\ast \Vert_2^2$$.
    *   Damped Newton for global convergence. Affine invariant.
*   **Drawbacks:** Hessian cost ($$O(n^3)$$ for solve), Hessian must be positive definite.

**10. Quasi-Newton Methods (e.g., BFGS, L-BFGS)**

Quasi-Newton methods aim for Newton-like speed without forming/inverting the exact Hessian.

*   **Core Idea:** Approximate Hessian $$B_k \approx \nabla^2 f(x^{(k)})$$ or inverse $$H_k \approx (\nabla^2 f(x^{(k)}))^{-1}$$ using only gradient info.
*   **Update Rule (using $$H_k$$):**
    1.  $$d^{(k)} = -H_k \nabla f(x^{(k)})$$.
    2.  Line search for $$t_k$$ (e.g., Wolfe conditions).
    3.  $$x^{(k+1)} = x^{(k)} + t_k d^{(k)}$$.
    4.  Update $$H_k \to H_{k+1}$$ using $$s^{(k)} = x^{(k+1)} - x^{(k)}$$ and $$y^{(k)} = \nabla f(x^{(k+1)}) - \nabla f(x^{(k)})$$.
    <details class="details-block" markdown="1">
    <summary markdown="1">
    **The Secant Condition in Quasi-Newton Methods**
    </summary>
    A key idea in deriving Quasi-Newton updates (like BFGS) is the **secant condition** (or quasi-Newton condition). For an approximation $$B_{k+1}$$ to the Hessian $$\nabla^2 f(x^{(k+1)})$$, we require it to satisfy:

    $$
    B_{k+1} s^{(k)} = y^{(k)}
    $$

    Or, for an approximation $$H_{k+1}$$ to the inverse Hessian $$(\nabla^2 f(x^{(k+1)}))^{-1}$$:

    $$
    H_{k+1} y^{(k)} = s^{(k)}
    $$

    This condition is derived from a finite-difference approximation of the derivative: if $$f$$ were quadratic, $$B_{k+1}$$ would be the exact Hessian. For general $$f$$, it ensures that the new Hessian/inverse Hessian approximation is consistent with the most recent change in $$x$$ and $$\nabla f$$. The BFGS update is one specific way to achieve this while maintaining symmetry and positive definiteness.
    </details>
*   **BFGS (Broyden-Fletcher-Goldfarb-Shanno):** Popular, maintains symmetry and positive definiteness of $$H_k$$ if $$(y^{(k)})^T s^{(k)} > 0$$.
*   **L-BFGS (Limited-memory BFGS):** Stores $$m$$ recent $$(s^{(i)}, y^{(i)})$$ pairs to implicitly compute $$H_k \nabla f(x^{(k)})$$. Memory $$O(mn)$$, cost $$O(mn)$$.
*   **Convergence:** Superlinear for BFGS. Global with Wolfe line search. L-BFGS is often very effective for large-scale smooth problems.

**11. Interior-Point Methods (IPMs) (Brief Overview)**

IPMs solve constrained convex problems by traversing the interior of the feasible region.

*   **Core Idea:** Transform constrained problem (e.g., $$\min f_0(x) \text{ s.t. } f_i(x) \le 0, Ax=b$$) into a sequence of unconstrained problems using a **barrier function** (e.g., $$\phi(x) = -\sum_i \log(-f_i(x))$$). Subproblem: $$\min t f_0(x) + \phi(x)$$.
*   **Central Path:** Solutions $$x^\ast(t)$$ to subproblems as $$t \to \infty$$ form a central path to the true optimum.
*   **Method:** Use modified Newton's method to solve subproblems for increasing $$t$$.
*   **Convergence:** Polynomial-time for LPs, QPs, SOCPs, SDPs. High accuracy.
*   **Applications:** Core of most solvers for structured convex optimization.

**12. Other Advanced Methods (Briefly)**

*   **Alternating Direction Method of Multipliers (ADMM):**
    *   For problems $$\min_{x,z} f(x) + g(z)$$ s.t. $$Ax + Bz = c$$. Decomposes into smaller subproblems by alternating minimizations and dual updates. Popular for distributed optimization.

<details class="details-block" markdown="1">
<summary markdown="1">
**Operator Splitting and Monotone Operator Theory**
</summary>

Many advanced algorithms like ADMM, Douglas-Rachford Splitting (DRS), and Primal-Dual Hybrid Gradient (PDHG, also known as Chambolle-Pock) can be elegantly unified under the theory of **monotone operators** and **operator splitting**.
- **Forward-Backward Splitting (FBS)**, which underpins Proximal Gradient, seeks a zero of $$A+B$$ where $$A$$ is cocoercive (e.g., a gradient) and $$B$$ is maximally monotone (e.g., a subdifferential).
- **Douglas-Rachford Splitting (DRS)** finds a zero of $$A+B$$ where both $$A, B$$ are maximally monotone, using reflections of their resolvents (proximal operators).
- **ADMM** can be derived by applying DRS to the dual of the original problem, or by formulating it as finding a saddle point of an augmented Lagrangian.
- **PDHG / Chambolle-Pock** is often viewed as an FBS applied to a primal-dual saddle-point formulation, effectively finding a zero of a structured monotone operator in a product space.
This unifying perspective highlights deep connections between these methods and provides a powerful framework for analysis and development of new algorithms.
</details>

*   **Cutting-Plane/Bundle Methods:**
    *   For general non-smooth convex optimization. Iteratively build a piecewise-linear lower model of $$f(x)$$ using subgradients. Bundle methods add stabilization (e.g., proximal term) for robustness.

**13. Relationships, Summary, and Choosing an Algorithm**

The world of convex optimization algorithms is rich and interconnected.

*   **Conceptual Map & Relationships:**
    *   **Smoothness Handling:** GD/Newton vs. Subgradient/Bundle vs. Proximal/ADMM.
    *   **Order of Information:** First-order vs. Second-order vs. Quasi-Newton.
    *   **Acceleration:** Nesterov's principle (AGD, FISTA).
    *   **Stochasticity:** GD vs. SGD & variance-reduced variants.
    *   **Geometry:** GD (Euclidean) vs. Mirror Descent (Bregman).
    *   **Decomposition & Splitting:** Proximal, ADMM, DRS.
    *   **Special Cases:** PGM generalizes GD and Projected GD. IPMs use Newton.

*   **Hierarchy of Assumptions & Typical Convergence Rates (Iterations for $$\epsilon$$-accuracy):**
    *   **Non-smooth, Convex (Subgradient Method):** $$O(1/\epsilon^2)$$
    *   **Non-smooth, Convex (Accelerated Subgradient/Mirror Descent variants):** $$O(1/\epsilon)$$
    *   **Smooth, Convex (Gradient Descent):** $$O(1/\epsilon)$$
    *   **Smooth, Strongly Convex (Gradient Descent):** $$O(\kappa \log(1/\epsilon))$$
    *   **Smooth, Convex (Accelerated Gradient Descent):** $$O(1/\sqrt{\epsilon})$$
    *   **Smooth, Strongly Convex (Accelerated Gradient Descent):** $$O(\sqrt{\kappa} \log(1/\epsilon))$$
    *   **Smooth, Finite Sum, Strongly Convex (Variance-Reduced SGD, e.g., SVRG):** $$O((m+\kappa)\log(1/\epsilon))$$
    *   **Twice Smooth, Strongly Convex (Newton's Method):** Locally $$O(\log \log(1/\epsilon))$$ iterations.
    *   **Structured Convex (e.g., LPs, SDPs with IPMs):** Often $$O(\sqrt{N} \log(1/\epsilon))$$ "Newton iterations."

*   **Table: Algorithm Selection Guidelines**

    | Problem Characteristics                                         | Suggested Algorithm(s)                                                             | Key Considerations                                                              |
    | :-------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
    | Small to medium scale, smooth, high accuracy needed             | Newton's Method, Interior-Point Methods (via solvers), L-BFGS                      | Hessian cost (Newton/IPM), IPM setup complexity, L-BFGS good general choice     |
    | Large scale, smooth, differentiable                             | L-BFGS, Accelerated Gradient Descent (Nesterov/FISTA if $$h=0$$)                   | Per-iteration cost, memory (L-BFGS good), ease of implementation                |
    | Large scale, composite (smooth $$f$$ + simple non-smooth $$h$$) | Proximal Gradient (ISTA), Accelerated Proximal Gradient (FISTA), ADMM              | Efficiency of proximal operator calculation is key                              |
    | Very large scale / Online / Sums of functions                   | Stochastic Gradient Descent (SGD), Stochastic Mirror Descent, Variance-Reduced SGD | Variance, step size tuning, convergence is in expectation, data access patterns |
    | Non-differentiable (general, prox not easy)                     | Subgradient Method, Bundle Methods                                                 | Slower convergence, robustness of Bundle methods vs. simplicity of Subgradient  |
    | Specific constraints (e.g., simplex, $$L_1$$-ball)              | Mirror Descent, Frank-Wolfe, Projected GD/PGM (if projection cheap)                | Exploiting geometry, cost of linear oracle (Frank-Wolfe) or projection          |
    | Separable objective + linear constraints (e.g., distributed)    | ADMM                                                                               | Decomposability, tuning penalty parameter $$\rho$$                              |
    | Standard LPs, QPs, SOCPs, SDPs                                  | Interior-Point Methods (typically via specialized solvers)                         | Availability of high-quality solvers, problem formulation complexity            |

**Conclusion of the Crash Course**

This post has provided an overview of some of the most fundamental algorithms used to solve convex optimization problems. We've touched upon Gradient Descent and its variants (Subgradient, Stochastic, Accelerated), explored the power of Proximal Algorithms for composite optimization, looked at Mirror Descent for non-Euclidean geometries, and discussed second-order methods like Newton's method, Quasi-Newton methods, and Interior-Point Methods. Each algorithm comes with its own set of assumptions, strengths, and trade-offs regarding computational cost, convergence speed, and applicability to different problem structures.

With this, we conclude our Convex Analysis Crash Course! We embarked on a journey from the basic definitions of convex sets and functions, delved into the elegant theory of convex optimization problems and duality (including the Fenchel conjugate and KKT conditions), and have now surveyed the algorithmic tools to tackle these problems.

Convex optimization is a beautiful and powerful field, forming a cornerstone of modern machine learning, operations research, signal processing, control theory, and many other engineering and scientific disciplines. Its principles allow for the design of efficient and reliable algorithms, often accompanied by strong theoretical guarantees of convergence to a global optimum. While this crash course has only scratched the surface, we hope it has equipped you with a solid intuition, a map of the essential concepts, and the foundational knowledge to navigate further studies in optimization theory and its diverse applications. Remember, true mastery of these concepts often comes from working through examples, deriving proofs, and applying these algorithms to practical problems.

Good luck with your continued exploration of optimization!

**Further Reading**

For those interested in diving deeper, here are a few highly recommended resources:

1.  **Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.** (Available online) - The definitive textbook, covering theory, applications, and algorithms.
2.  **Nesterov, Y. (2004, 2018). *Introductory Lectures on Convex Optimization: A Basic Course*. Kluwer Academic Publishers / Springer.** - A foundational text by one of the pioneers of accelerated methods, focusing on algorithmic complexity.
3.  **Bubeck, S. (2015). *Convex Optimization: Algorithms and Complexity*. Foundations and Trends® in Machine Learning.** (Available online) - A concise and modern survey of algorithms and their complexity bounds.
4.  **Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization Methods for Large-Scale Machine Learning. *SIAM Review*, 60(2), 223-311.** - An excellent review of stochastic and batch methods prevalent in machine learning.

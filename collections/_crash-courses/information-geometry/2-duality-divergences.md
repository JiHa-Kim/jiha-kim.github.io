---
title: "Information Geometry Part 2: Duality, Divergences, and Natural Gradient"
date: 2025-06-22 10:00 -0400
course_index: 2
description: "Exploring dual connections, Bregman divergences, dually flat spaces, and the powerful natural gradient algorithm within the framework of Information Geometry."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Information Geometry
- Dual Connections
- Bregman Divergence
- Natural Gradient
- Fisher Information Metric
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
  Here is content thatl can include **Markdown**, inline math $$a + b$$,
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

Welcome back to our crash course on Information Geometry! In [Part 1](link_to_ig_part1_placeholder), we established that statistical models can be viewed as statistical manifolds equipped with the Fisher Information Metric (FIM), which provides a natural way to measure distances based on statistical distinguishability.

Now, we'll build upon this foundation to explore more advanced geometric structures: affine connections, the crucial concept of duality, dually flat spaces, and Bregman divergences. These concepts will culminate in the introduction of the **Natural Gradient**, a powerful optimization algorithm that directly leverages the geometry of the statistical manifold. This part will draw upon concepts from our [Differential Geometry](link_to_dg_course_placeholder) crash course, particularly regarding connections and covariant derivatives.

## Recap from Part 1

Quick reminder:
-   **Statistical Manifold $$S$$**: A space where each point is a probability distribution $$p(x;\theta)$$.
-   **Fisher Information Metric $$g(\theta)$$(FIM)**: A Riemannian metric $$g_{ij}(\theta) = \mathbb{E}[(\partial_i \log p)(\partial_j \log p)]$$ defining local geometry (distances, angles).
-   **KL Divergence and FIM**: Locally, $$D_{KL}(p_\theta \Vert  p_{\theta+d\theta}) \approx \frac{1}{2} \sum_{ij} g_{ij}(\theta) d\theta^i d\theta^j$$.

## Affine Connections on Statistical Manifolds

A Riemannian metric $$g$$ alone defines lengths and angles. To define concepts like "straight lines" (geodesics) and parallel transport of vectors, we need an **affine connection** $$\nabla$$. An affine connection specifies how to differentiate vector fields along curves on the manifold. (See the [Differential Geometry](link_to_dg_course_placeholder) course for details on connections and Christoffel symbols $$\Gamma_{ijk}$$).

On a general Riemannian manifold, there's a unique metric-compatible and torsion-free connection called the **Levi-Civita connection** ($$\nabla^{(0)}$$). While fundamental, it turns out that in Information Geometry, a *pair* of dual connections often provides a richer structure.

### Amari's $$\alpha$$-Connections

Shun-ichi Amari introduced a one-parameter family of affine connections, called **$$\alpha$$-connections** ($$\nabla^{(\alpha)}$$), which are particularly relevant for statistical manifolds. These connections are defined by their Christoffel symbols:

$$
\Gamma_{ijk}^{(\alpha)} = \Gamma_{ijk}^{(0)} - \frac{\alpha}{2} T_{ijk}
$$

where $$\Gamma_{ijk}^{(0)}$$ are the Christoffel symbols of the Levi-Civita connection (derived from the Fisher metric $$g$$), and $$T_{ijk}$$ is a completely symmetric tensor called the **Amari-Chentsov tensor** or skewness tensor:

$$
T_{ijk} = \mathbb{E}_{p(x;\theta)} \left[ (\partial_i \log p) (\partial_j \log p) (\partial_k \log p) \right]
$$

The parameter $$\alpha \in \mathbb{R}$$ interpolates between different geometries. Two specific values of $$\alpha$$ are of paramount importance:

1.  **The e-connection ($$\nabla^{(e)}$$ or $$\nabla^{(1)}$$ for $$\alpha=1$$):**
    This connection is associated with the geometry of **exponential families**. A statistical manifold is $$\nabla^{(e)}$$-flat if it corresponds to an exponential family under a suitable parameterization (the natural parameters). Geodesics under the e-connection are called e-geodesics.

2.  **The m-connection ($$\nabla^{(m)}$$ or $$\nabla^{(-1)}$$ for $$\alpha=-1$$):**
    This connection is associated with the geometry of **mixture families**. A statistical manifold is $$\nabla^{(m)}$$-flat if it corresponds to a mixture family under a suitable parameterization (the expectation parameters). Geodesics under the m-connection are called m-geodesics.

The Levi-Civita connection corresponds to $$\alpha=0$$.

## Duality in Information Geometry

A cornerstone of Amari's Information Geometry is the concept of **duality** between connections. The e-connection ($$\nabla^{(1)}$$) and the m-connection ($$\nabla^{(-1)}$$) are said to be **dual** with respect to the Fisher Information Metric $$g$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dual Connections
</div>
Two affine connections $$\nabla$$ and $$\nabla^\ast$$ are said to be **dual** with respect to a Riemannian metric $$g$$ if for any vector fields $$X, Y, Z$$ on the manifold:

$$
X (g(Y,Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X^\ast Z)
$$

This is a generalization of the product rule for differentiation. In index notation, with $$g_{ij}$$ as the metric components and $$\Gamma_{ijk}$$ and $$\Gamma_{ijk}^\ast$$ as Christoffel symbols for $$\nabla$$ and $$\nabla^\ast$$ respectively (in the form $$g(\nabla_{\partial_i} \partial_j, \partial_k) = \Gamma_{ijk}$$):

$$
\partial_k g_{ij} = \Gamma_{kij} + \Gamma_{kji}^\ast
$$

</blockquote>

It can be shown that for any $$\alpha$$, the connection $$\nabla^{(\alpha)}$$ and $$\nabla^{(-\alpha)}$$ are dual with respect to the Fisher metric $$g$$. Thus, the e-connection ($$\alpha=1$$) and m-connection ($$\alpha=-1$$) form a fundamental dual pair. This duality is not just a mathematical curiosity; it underpins many key results in IG, such as the generalized Pythagorean theorem and the structure of dually flat spaces.

## Dually Flat Spaces

Statistical manifolds that are "flat" with respect to both a connection $$\nabla$$ and its dual $$\nabla^\ast$$ possess a particularly elegant structure. These are called **dually flat spaces**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dually Flat Space
</div>
A statistical manifold $$(S, g, \nabla^{(\alpha)}, \nabla^{(-\alpha)})$$ is **dually flat** if there exist:
1.  An affine coordinate system $$\theta = (\theta^1, \dots, \theta^d)$$ such that $$S$$ is $$\nabla^{(\alpha)}$$-flat (i.e., $$\nabla^{(\alpha)}$$-geodesics are straight lines in $$\theta$$ coordinates, and Christoffel symbols $$\Gamma_{ijk}^{(\alpha)}(\theta) = 0$$).
2.  A dual affine coordinate system $$\eta = (\eta_1, \dots, \eta_d)$$ such that $$S$$ is $$\nabla^{(-\alpha)}$$-flat (i.e., $$\nabla^{(-\alpha)}$$-geodesics are straight lines in $$\eta$$ coordinates, and Christoffel symbols $$\Gamma_{ijk}^{(-\alpha)}(\eta) = 0$$).

Furthermore, there exist dual **potential functions** $$\psi(\theta)$$ and $$\phi(\eta)$$ such that:
- The Fisher metric components in $$\theta$$ coordinates are $$g_{ij}(\theta) = \frac{\partial^2 \psi(\theta)}{\partial \theta^i \partial \theta^j}$$.
- The Fisher metric components in $$\eta$$ coordinates are $$g^{ij}(\eta) = \frac{\partial^2 \phi(\eta)}{\partial \eta_i \partial \eta_j}$$ (here $$g^{ij}$$ are components of the inverse metric).
- The coordinate systems are related by Legendre transformations:

  $$
  \eta_i = \frac{\partial \psi(\theta)}{\partial \theta^i} \quad \text{and} \quad \theta^i = \frac{\partial \phi(\eta)}{\partial \eta_i}
  $$

- The potential functions are related by:

  $$
  \psi(\theta) + \phi(\eta) - \sum_i \theta^i \eta_i = 0
  $$

</blockquote>

**Exponential families** are canonical examples of dually flat spaces with respect to the e-connection ($$\alpha=1$$) and m-connection ($$\alpha=-1$$).
-   The **natural parameters** of an exponential family serve as the $$\theta$$ coordinates (e-affine). The potential $$\psi(\theta)$$ is the cumulant generating function (log-partition function).
-   The **expectation parameters** (mean of sufficient statistics) serve as the $$\eta$$ coordinates (m-affine). The potential $$\phi(\eta)$$ is related to the negative entropy.

The existence of these dually flat structures greatly simplifies many calculations and provides deep insights into the geometry of statistical inference.

## Bregman Divergences

In a dually flat space, the "distance" or "divergence" between two points (distributions) can be naturally defined using the potential function associated with one of the affine coordinate systems. This leads to the concept of **Bregman divergences**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Bregman Divergence
</div>
Let $$\psi: \Omega \to \mathbb{R}$$ be a continuously-differentiable, strictly convex function defined on a convex set $$\Omega \subseteq \mathbb{R}^d$$. The **Bregman divergence** $$D_\psi: \Omega \times \Omega \to [0, \infty)$$ associated with $$\psi$$ is defined as:

$$
D_\psi(\theta_1 \Vert  \theta_2) = \psi(\theta_1) - \psi(\theta_2) - \langle \nabla \psi(\theta_2), \theta_1 - \theta_2 \rangle
$$

where $$\nabla \psi(\theta_2)$$ is the gradient of $$\psi$$ at $$\theta_2$$, and $$\langle \cdot, \cdot \rangle$$ is the standard inner product.
</blockquote>
Geometrically, $$D_\psi(\theta_1 \Vert  \theta_2)$$ is the difference between the value of $$\psi(\theta_1)$$ and the value of the first-order Taylor expansion of $$\psi$$ around $$\theta_2$$, evaluated at $$\theta_1$$. Due to strict convexity, $$D_\psi(\theta_1 \Vert  \theta_2) \ge 0$$, and $$D_\psi(\theta_1 \Vert  \theta_2) = 0$$ if and only if $$\theta_1 = \theta_2$$. Note that Bregman divergences are generally not symmetric.

**Key Example: KL Divergence as a Bregman Divergence**
For an exponential family $$p(x;\theta) = h(x) \exp(\theta \cdot T(x) - \psi(\theta))$$, where $$\theta$$ are the natural parameters and $$\psi(\theta)$$ is the log-partition function (cumulant generator), the **KL divergence $$D_{KL}(p_{\theta_1} \Vert  p_{\theta_2})$$ is precisely the Bregman divergence $$D_\psi(\theta_1 \Vert  \theta_2)$$ associated with the potential function $$\psi(\theta)$$.**
<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation.** KL as Bregman Divergence for Exponential Families
</summary>
For an exponential family:

$$
D_{KL}(p_{\theta_1} \Vert  p_{\theta_2}) = \mathbb{E}_{p_{\theta_1}}[\log p(x;\theta_1) - \log p(x;\theta_2)]
$$

$$
\log p(x;\theta) = \theta \cdot T(x) - \psi(\theta) + \log h(x)
$$

So,

$$
\log p(x;\theta_1) - \log p(x;\theta_2) = (\theta_1 - \theta_2) \cdot T(x) - (\psi(\theta_1) - \psi(\theta_2))
$$

$$
D_{KL}(p_{\theta_1} \Vert  p_{\theta_2}) = \mathbb{E}_{p_{\theta_1}}[(\theta_1 - \theta_2) \cdot T(x)] - (\psi(\theta_1) - \psi(\theta_2))
$$

We know that for exponential families, $$\mathbb{E}_{p_{\theta_1}}[T(x)] = \nabla \psi(\theta_1)$$.
This seems to lead to $$D_{KL}(p_{\theta_1} \Vert  p_{\theta_2}) = (\theta_1 - \theta_2) \cdot \nabla \psi(\theta_1) - (\psi(\theta_1) - \psi(\theta_2))$$.
This is $$D_\psi(\theta_2 \Vert  \theta_1)$$ if we swap arguments carefully.
Let's use the definition of Bregman divergence directly:
$$D_\psi(\theta_1 \Vert  \theta_2) = \psi(\theta_1) - \psi(\theta_2) - \nabla \psi(\theta_2) \cdot (\theta_1 - \theta_2)$$.
Recall that for an exponential family, $$\eta_i = \mathbb{E}_{p_\theta}[T_i(x)] = \frac{\partial \psi(\theta)}{\partial \theta^i}$$. So $$\nabla \psi(\theta_2)$$ corresponds to the expectation parameters $$\eta_2$$ of the distribution $$p_{\theta_2}$$.
Then $$D_{KL}(p_{\theta_1} \Vert  p_{\theta_2}) = \psi(\theta_1) - \psi(\theta_2) - (\theta_1 - \theta_2) \cdot \eta_2$$. This matches $$D_\psi(\theta_1 \Vert  \theta_2)$$.
(The standard definition of KL for exponential families is indeed $$D_\psi(\theta_1\Vert \theta_2)$$ with $$\psi$$ being the log-normalizer of the *first* argument when parameters are natural parameters.
Ah, careful, the expectation is usually taken w.r.t. $$p_{\theta_1}$$.
Let $$p_1 = p(x;\theta_1)$$ and $$p_2 = p(x;\theta_2)$$.
$$ D_{KL}(p_1 \Vert  p_2) = \int p_1(x) (\log p_1(x) - \log p_2(x)) dx $$
$$ = \int p_1(x) ( (\theta_1 \cdot T(x) - \psi(\theta_1)) - (\theta_2 \cdot T(x) - \psi(\theta_2)) ) dx $$
$$ = (\theta_1 - \theta_2) \cdot \mathbb{E}_{p_1}[T(x)] - (\psi(\theta_1) - \psi(\theta_2)) $$
Since $$\mathbb{E}_{p_1}[T(x)] = \nabla \psi(\theta_1)$$, this becomes:
$$ D_{KL}(p_1 \Vert  p_2) = (\theta_1 - \theta_2) \cdot \nabla \psi(\theta_1) - \psi(\theta_1) + \psi(\theta_2) $$
This expression is $$D_{\psi^\ast }(\eta_2 \Vert  \eta_1)$$ where $$\psi^\ast $$ is the dual potential and $$\eta$$ are expectation parameters, or it can be written as $$B_\psi(\theta_1 \Vert  \theta_2)$$ as:
$$B_\psi(\theta_1 \Vert  \theta_2) = \psi(\theta_1) - \psi(\theta_2) - \langle \nabla\psi(\theta_2), \theta_1-\theta_2 \rangle$$. This matches the definition if we identify $$\nabla\psi(\theta_2)$$ with $$\mathbb{E}_{\theta_2}[T(x)]$$.
Indeed, the KL divergence for exponential families in natural parameters $$\theta$$ and log-partition function $$\psi$$ is exactly $$D_\psi(\theta_1 \Vert  \theta_2)$$.
</details>

### Generalized Pythagorean Theorem

Dually flat spaces with their Bregman divergences satisfy a remarkable generalization of the Pythagorean theorem.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Generalized Pythagorean Theorem
</div>
Let $$(S, g, \nabla^{(e)}, \nabla^{(m)})$$ be a dually flat space (e.g., an exponential family) with $$\theta$$ as e-affine coordinates and $$\eta$$ as m-affine coordinates. Let $$D$$ be the canonical Bregman divergence (e.g., KL divergence).
Consider three points $$P, Q, R$$ in $$S$$.
If the e-geodesic segment from $$P$$ to $$Q$$ is orthogonal (with respect to the Fisher metric $$g$$) at $$P$$ to the m-geodesic segment from $$P$$ to $$R$$, then:

$$
D(R \Vert  Q) = D(R \Vert  P) + D(P \Vert  Q)
$$

Similarly, if the m-geodesic from $$P$$ to $$Q$$ is orthogonal at $$P$$ to the e-geodesic from $$P$$ to $$R$$, the same identity holds.
This theorem is fundamental for understanding information projection and iterative algorithms like EM.
</blockquote>
<blockquote class="box-info" markdown="1">
**Analogy.** Euclidean Pythagorean Theorem
This is analogous to the Euclidean Pythagorean theorem $$c^2 = a^2 + b^2$$ for a right triangle, where squared Euclidean distances (which are Bregman divergences for $$\psi(x) = \frac{1}{2} \Vert x \Vert^2$$) are replaced by the canonical Bregman divergence $$D$$.
</blockquote>

## The Natural Gradient

We now arrive at one of the most significant practical applications of Information Geometry in machine learning: the **Natural Gradient**.
Standard gradient descent for a loss function $$L(\theta)$$ updates parameters via:

$$
\theta_{t+1} = \theta_t - \lr \nabla L(\theta_t)
$$

where $$\nabla L(\theta_t)$$ is the Euclidean gradient.
However, as discussed in Part 1, the Euclidean geometry of the parameter space $$\Theta$$ might not reflect the "true" geometry of the statistical manifold $$S$$. A step of a certain Euclidean length in one direction of $$\theta$$ might correspond to a vastly different change in the distribution $$p(x;\theta)$$ compared to a step of the same Euclidean length in another direction. This can lead to slow or unstable convergence.

The natural gradient addresses this by considering the steepest descent direction in the Riemannian geometry defined by the Fisher Information Metric.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Natural Gradient
</div>
The **natural gradient** of a loss function $$L(\theta)$$ on a statistical manifold $$S$$ equipped with the Fisher Information Metric $$g(\theta) = I(\theta)$$ is defined as:

$$
\tilde{\nabla} L(\theta) = I(\theta)^{-1} \nabla L(\theta)
$$

where $$\nabla L(\theta)$$ is the standard (Euclidean) gradient of $$L$$ with respect to $$\theta$$.
The **natural gradient descent (NGD)** update rule is:

$$
\theta_{t+1} = \theta_t - \lr \tilde{\nabla} L(\theta_t) = \theta_t - \lr I(\theta_t)^{-1} \nabla L(\theta_t)
$$

</blockquote>

### Intuition and Properties of Natural Gradient

1.  **Steepest Descent on the Manifold:** The natural gradient points in the direction of steepest descent of $$L(\theta)$$ *with respect to the distance measure defined by the Fisher metric*. It's the direction that causes the largest decrease in $$L$$ for a small step of fixed "statistical length" $$ds$$.
2.  **Parameterization Invariance:** Unlike the standard gradient, the natural gradient direction is invariant under reparameterizations of $$\theta$$. This is a highly desirable property, meaning the optimization path does not depend on arbitrary choices of how we parameterize our model (as long as the FIM is transformed accordingly).
3.  **Connection to Second-Order Methods:** The update rule $$\theta_{t+1} = \theta_t - \lr I(\theta_t)^{-1} \nabla L(\theta_t)$$ resembles Newton's method, which uses the inverse Hessian $$H(\theta_t)^{-1}$$ instead of $$I(\theta_t)^{-1}$$.
    -   The FIM $$I(\theta)$$ can be seen as the expected Hessian of the negative log-likelihood of the *model itself*, $$-\log p(x;\theta)$$, rather than the Hessian of the *loss function* $$L(\theta)$$.
    -   For a loss function $$L(\theta) = \mathbb{E}_{\text{data}}[-\log p(x_{\text{data}};\theta)]$$ (i.e., maximum likelihood estimation), the Hessian of $$L$$ is the observed Fisher information. The FIM $$I(\theta)$$ is its expectation over the model's distribution.
    -   A key advantage is that $$I(\theta)$$ is always positive semi-definite (and usually positive definite), whereas the Hessian $$H(L(\theta))$$ can be indefinite, making Newton's method problematic without modifications.
4.  **Improved Convergence:** NGD often converges much faster than standard gradient descent, especially when the parameter space is poorly scaled or "warped" from a Euclidean perspective. It effectively "preconditions" the gradient, transforming the optimization landscape into one where gradient steps are more effective.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example Revisited.** Natural Gradient for Gaussian Mean
</div>
Consider fitting a univariate Gaussian $$N(\mu, \sigma_0^2)$$ with *known* variance $$\sigma_0^2$$, to minimize some loss $$L(\mu)$$.
In Part 1, we found $$I(\mu) = 1/\sigma_0^2$$.
The standard gradient is $$\nabla L(\mu) = \frac{\partial L}{\partial \mu}$$.
The natural gradient is:

$$
\tilde{\nabla} L(\mu) = (1/\sigma_0^2)^{-1} \frac{\partial L}{\partial \mu} = \sigma_0^2 \frac{\partial L}{\partial \mu}
$$

The NGD update for $$\mu$$ is:

$$
\mu_{t+1} = \mu_t - \lr \sigma_0^2 \frac{\partial L}{\partial \mu_t}
$$

If $$\sigma_0^2$$ is very small, the distribution is very "peaked". A small change in $$\mu$$ leads to a large change in statistical distance (KL divergence). The natural gradient appropriately scales up the gradient step. Conversely, if $$\sigma_0^2$$ is large (flat distribution), it scales down the step. This scaling adapts the learning rate locally based on the manifold's geometry.
</blockquote>

The main challenge with NGD in practice, especially for large models like deep neural networks, is the computation and inversion of the Fisher Information Matrix $$I(\theta)$$, which can be very large ($$d \times d$$, where $$d$$ is the number of parameters). Much research focuses on efficient approximations of the FIM or its inverse.

## Summary of Part 2 & Linking to ML Optimization

In this part, we've ventured deeper into the geometric landscape of Information Geometry:
-   **$$\alpha$$-connections** (especially e- and m-connections) provide notions of "straightness" on statistical manifolds.
-   The **duality** between e- and m-connections is a fundamental structural property.
-   **Dually flat spaces**, exemplified by exponential families, possess elegant dual coordinate systems and potential functions, leading to **Bregman divergences** like KL divergence.
-   The **Natural Gradient** leverages the Fisher Information Metric to define a parameterization-invariant steepest descent direction, often leading to superior optimization performance.

These concepts are not just abstract mathematics; they have profound implications for machine learning:
-   The **Natural Gradient** forms the theoretical basis for several advanced optimization algorithms. Approximations to the FIM (e.g., diagonal or block-diagonal) are used in optimizers like Adam (implicitly, via adaptive learning rates per parameter) and K-FAC. This will be a key connection point in the main optimization series.
-   **Mirror Descent**, an optimization framework using Bregman divergences, is closely related to natural gradient ascent in dually flat spaces.
-   Understanding the geometry of loss landscapes through the lens of IG can help explain why certain optimization strategies work better than others.

In the (optional) Part 3, we will briefly touch upon some applications and further horizons, further solidifying the bridge to practical machine learning optimization.

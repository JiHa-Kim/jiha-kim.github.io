---
title: "Information Geometry: Cheat Sheet"
date: 2025-07-01 10:00 -0400
course_index: 999
description: "A quick reference guide and cheat sheet for key concepts, definitions, and formulas from the Information Geometry crash course."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Information Geometry
- Cheat Sheet
- Statistical Manifolds
- Fisher Information Metric
- Natural Gradient
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

This cheat sheet provides a quick summary of the key concepts, definitions, and formulas covered in our Information Geometry (IG) crash course. Refer back to [Part 1](link_to_ig_part1_placeholder), [Part 2](link_to_ig_part2_placeholder), and [Part 3](link_to_ig_part3_placeholder) for detailed explanations.

## Core Concepts

| Concept                        | Description                                                                                                                                                                |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Statistical Manifold $$S$$** | A differentiable manifold where each point represents a probability distribution $$p(x;\theta)$$, parameterized by $$\theta$$.                                             |
| **Parameters $$\theta$$**      | Coordinates $$(\theta^1, \dots, \theta^d)$$ on the statistical manifold.                                                                                                   |
| **Score Function**             | Vector with components $$\partial_i \log p(x;\theta) = \frac{\partial \log p(x;\theta)}{\partial \theta^i}$$. Has zero expectation: $$\mathbb{E}[\partial_i \log p] = 0$$. |

## Fisher Information Metric (FIM)

The FIM $$g(\theta)$$ (or $$I(\theta)$$) is a Riemannian metric fundamental to IG.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fisher Information Metric Components $$g_{ij}(\theta)$$
</div>
1.  Using score functions (outer product):

    $$
    g_{ij}(\theta) = \mathbb{E}_{p(x;\theta)} \left[ \left( \frac{\partial \log p(x; \theta)}{\partial \theta^i} \right) \left( \frac{\partial \log p(x; \theta)}{\partial \theta^j} \right) \right]
    $$

2.  Using second derivatives of log-likelihood (Hessian form):

    $$
    g_{ij}(\theta) = - \mathbb{E}_{p(x;\theta)} \left[ \frac{\partial^2 \log p(x; \theta)}{\partial \theta^i \partial \theta^j} \right]
    $$

(Under regularity conditions, these are equivalent.)
</blockquote>

**Properties:**
-   Symmetric and positive semi-definite (positive definite if parameters are identifiable).
-   Appears in Cramér-Rao bound: $$\text{Var}(\hat{\theta}_i) \ge (I(\theta)^{-1})_{ii}$$.
-   Invariant under reparameterization of data $$x$$; transforms covariantly for parameter $$\theta$$ changes.
-   Unique (up to a constant) Riemannian metric satisfying certain statistical invariances (Chentsov's Theorem).

**Geometric Interpretation:**
-   Infinitesimal squared statistical distance: $$ds^2 = \sum_{i,j} g_{ij}(\theta) d\theta^i d\theta^j = d\theta^T I(\theta) d\theta$$.
-   Local relation to KL Divergence: $$D_{KL}(p_\theta \Vert  p_{\theta+d\theta}) \approx \frac{1}{2} ds^2$$.

**Examples of FIM:**
| Distribution Family                      | Parameter(s) $$\theta$$    | FIM $$I(\theta)$$                               |
| :--------------------------------------- | :------------------------- | :---------------------------------------------- |
| Bernoulli                                | $$\pi$$                    | $$\frac{1}{\pi(1-\pi)}$$                        |
| Gaussian (known $$\sigma^2$$)            | $$\mu$$                    | $$\frac{1}{\sigma^2}$$                          |
| Gaussian (known $$\mu$$)                 | $$\sigma^2$$ (or $$\tau$$) | $$\frac{1}{2(\sigma^2)^2} = \frac{1}{2\tau^2}$$ |
| Multivariate Gaussian (known $$\Sigma$$) | $$\mu$$ (vector)           | $$\Sigma^{-1}$$                                 |

## $$\alpha$$-Connections and Duality

-   **$$\alpha$$-Connections ($$\nabla^{(\alpha)}$$):** A family of affine connections parameterized by $$\alpha \in \mathbb{R}$$.
    -   Christoffel symbols: $$\Gamma_{ijk}^{(\alpha)} = \Gamma_{ijk}^{(0)} - \frac{\alpha}{2} T_{ijk}$$, where $$\Gamma_{ijk}^{(0)}$$ are for Levi-Civita and $$T_{ijk} = \mathbb{E}[(\partial_i \log p)(\partial_j \log p)(\partial_k \log p)]$$.
    -   **e-connection ($$\alpha = 1$$):** $$\nabla^{(e)}$$ or $$\nabla^{(1)}$$. Associated with exponential families.
    -   **m-connection ($$\alpha = -1$$):** $$\nabla^{(m)}$$ or $$\nabla^{(-1)}$$. Associated with mixture families.
    -   **Levi-Civita connection ($$\alpha=0$$):** Metric-compatible, torsion-free.
-   **Duality:** Connections $$\nabla^{(\alpha)}$$ and $$\nabla^{(-\alpha)}$$ are dual with respect to the FIM $$g$$.
    -   Rule: $$X(g(Y,Z)) = g(\nabla_X^{(\alpha)} Y, Z) + g(Y, \nabla_X^{(-\alpha)} Z)$$.

## Dually Flat Spaces and Bregman Divergences

-   **Dually Flat Space:** A manifold that is flat under both $$\nabla^{(\alpha)}$$ (in $$\theta$$ coords) and $$\nabla^{(-\alpha)}$$ (in dual $$\eta$$ coords).
    -   Possesses dual potential functions $$\psi(\theta)$$ and $$\phi(\eta)$$.
    -   Metric in $$\theta$$ coords: $$g_{ij}(\theta) = \frac{\partial^2 \psi(\theta)}{\partial \theta^i \partial \theta^j}$$.
    -   Legendre transformation: $$\eta_i = \frac{\partial \psi(\theta)}{\partial \theta^i}$$, $$\theta^i = \frac{\partial \phi(\eta)}{\partial \eta_i}$$, and $$\psi(\theta) + \phi(\eta) - \theta \cdot \eta = 0$$.
    -   Canonical example: Exponential families (e-flat in natural parameters $$\theta$$, m-flat in expectation parameters $$\eta$$).
-   **Bregman Divergence ($$D_\psi$$):** Associated with a strictly convex function $$\psi$$.

    $$
    D_\psi(\theta_1 \Vert  \theta_2) = \psi(\theta_1) - \psi(\theta_2) - \langle \nabla \psi(\theta_2), \theta_1 - \theta_2 \rangle
    $$

    -   KL divergence for exponential families is a Bregman divergence where $$\psi$$ is the log-partition function.
-   **Generalized Pythagorean Theorem:** In a dually flat space, for points $$P,Q,R$$ with appropriate orthogonality between e- and m-geodesics: $$D(R\Vert Q) = D(R\Vert P) + D(P\Vert Q)$$.

## Natural Gradient

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Natural Gradient $$\tilde{\nabla} L(\theta)$$
</div>
For a loss function $$L(\theta)$$:

$$
\tilde{\nabla} L(\theta) = I(\theta)^{-1} \nabla L(\theta)
$$

where $$\nabla L(\theta)$$ is the standard (Euclidean) gradient and $$I(\theta)$$ is the FIM.
</blockquote>

**Natural Gradient Descent (NGD) Update:**

$$
\theta_{t+1} = \theta_t - \lr \tilde{\nabla} L(\theta_t) = \theta_t - \lr I(\theta_t)^{-1} \nabla L(\theta_t)
$$

**Properties of Natural Gradient:**
-   Direction of steepest descent on the statistical manifold (w.r.t. Fisher metric).
-   Parameterization invariant.
-   Often yields faster convergence than standard gradient descent.
-   Resembles Newton's method but uses FIM (always positive semi-definite) instead of Hessian of loss.

**Challenges & Approximations for Deep Learning:**
-   Computing and inverting full FIM ($$I(\theta)$$) is often intractable ($$O(d^2)$$ to $$O(d^3)$$).
-   **Approximations:**
    -   **Diagonal FIM:** Leads to adaptive per-parameter learning rates. Related to Adam.
    -   **Block-Diagonal FIM (e.g., K-FAC):** Kronecker-factored approximations.
    -   Matrix-free methods (iterative solvers for $$I(\theta)s = \nabla L$$).

## Mirror Descent

-   **Generalization of Gradient Descent:**

    $$
    \theta_{t+1} = \arg \min_{\theta \in \mathcal{D}} \left\{ \langle \nabla L(\theta_t), \theta \rangle + \frac{1}{\lr} D_\psi(\theta \Vert  \theta_t) \right\}
    $$

    Uses Bregman divergence $$D_\psi$$ instead of squared Euclidean distance.
-   **Connection to Natural Gradient:** In dually flat spaces, Mirror Descent with canonical Bregman divergence can be equivalent to Natural Gradient.
-   **Exponentiated Gradient:** Result of Mirror Descent with KL-like divergence on probability simplex.

## Key Takeaways

-   Information Geometry provides a principled way to define geometry on spaces of probability distributions using the Fisher Information Metric.
-   This geometric viewpoint offers deeper understanding of statistical inference and can lead to more efficient optimization algorithms like the Natural Gradient.
-   Duality, dually flat spaces, and Bregman divergences are core theoretical constructs with practical implications.
-   While computationally intensive, the principles of IG (especially NGD) inspire practical approximations used in modern machine learning.

This cheat sheet is a starting point. The richness of Information Geometry extends far beyond these summaries, offering a powerful lens for analyzing information processing systems.

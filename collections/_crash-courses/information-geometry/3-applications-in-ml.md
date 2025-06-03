---
title: "Information Geometry Part 3: Applications in ML and Further Horizons"
date: 2025-06-03 10:00 -0400
course_index: 3
description: "Connecting Information Geometry to machine learning applications like natural gradient in deep learning, mirror descent, and other advanced topics, highlighting its practical relevance."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Information Geometry
- Natural Gradient
- Mirror Descent
- Machine Learning Optimization
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

Welcome to the concluding part of our Information Geometry (IG) crash course! In Part 1, we introduced statistical manifolds and the Fisher Information Metric (FIM). In link_to_ig_part2_placeholder, we explored dual connections, dually flat spaces, Bregman divergences, and the crucial concept of the Natural Gradient.

This final part aims to bridge these theoretical concepts more directly to their applications in machine learning and briefly outline some further horizons. The goal is to see how the geometric insights from IG can inform the design and understanding of learning algorithms.

## Recap of Key IG Concepts

Before diving into applications, let's quickly recall the main takeaways:
-   **Statistical Manifolds ($$S$$):** Spaces of probability distributions $$p(x;\theta)$$ where parameters $$\theta$$ are coordinates.
-   **Fisher Information Metric ($$I(\theta)$$ or $$g(\theta)$$)**: A Riemannian metric defining local geometry based on statistical distinguishability.
-   **Dual Connections ($$\nabla^{(e)}, \nabla^{(m)}$$)**: A pair of affine connections offering a rich geometric structure, especially for exponential families.
-   **Dually Flat Spaces**: Manifolds flat under dual connections, characterized by dual potential functions ($$\psi(\theta), \phi(\eta)$$) and Bregman divergences (e.g., KL divergence).
-   **Natural Gradient ($$\tilde{\nabla} L = I(\theta)^{-1} \nabla L$$)**: The direction of steepest descent on the statistical manifold, invariant to parameterization.

## Natural Gradient in Practice for Deep Learning

The Natural Gradient Descent (NGD) algorithm, $$\theta_{t+1} = \theta_t - \lr I(\theta_t)^{-1} \nabla L(\theta_t)$$, is theoretically appealing due to its parameterization invariance and potential for faster convergence. However, for modern deep neural networks, the number of parameters $$d$$ can be in the millions or billions. Computing, storing, and inverting the $$d \times d$$ Fisher Information Matrix $$I(\theta)$$ is computationally prohibitive:
-   Computing $$I(\theta)$$: $$O(N d^2)$$ or $$O(M d^2)$$ where $$N$$ is dataset size for empirical Fisher, $$M$$ is number of samples for expectation.
-   Inverting $$I(\theta)$$: $$O(d^3)$$.
-   Storing $$I(\theta)$$ or $$I(\theta)^{-1}$$: $$O(d^2)$$.

Thus, direct application of NGD is often infeasible. Research has focused on various **approximations to the FIM** or its inverse action $$I(\theta)^{-1}v$$:

1.  **Diagonal Approximation (Empirical Fisher):**
    Assume $$I(\theta)$$ is diagonal. This means ignoring correlations between parameter updates. The inverse $$I(\theta)^{-1}$$ is then trivial to compute (element-wise reciprocal of diagonal).
    The diagonal elements are $$I_{ii}(\theta) = \mathbb{E}[(\partial_i \log p)^2]$$.
    The update becomes:

    $$
    \theta_i^{t+1} = \theta_i^t - \lr \frac{1}{I_{ii}(\theta_t)} \frac{\partial L}{\partial \theta_i^t}
    $$

    This is reminiscent of adaptive learning rate methods. For example, the **Adam optimizer**, which we will cover in detail in the main series, uses squared gradients (related to the diagonal of the empirical FIM) to adapt per-parameter learning rates. While not explicitly derived as NGD, its success can be partly understood from an IG perspective as an efficient, albeit rough, approximation.

2.  **Block-Diagonal Approximation (e.g., K-FAC):**
    For layered neural networks, one can approximate the FIM as block-diagonal, where blocks correspond to layers or even individual weights/biases within a layer. This is more accurate than a purely diagonal approximation but still manageable.
    **Kronecker-Factored Approximate Curvature (K-FAC)** is a well-known method that approximates blocks of the FIM (or its inverse) using Kronecker products of smaller matrices derived from activations and pre-activation gradients. This exploits the structure of neural network layers.

3.  **Low-Rank Approximations:**
    Approximate the FIM or its inverse using low-rank matrices.

4.  **Matrix-Free Methods:**
    Instead of explicitly forming $$I(\theta)^{-1}$$, use iterative methods (like conjugate gradient) to solve the linear system $$I(\theta) s_t = \nabla L(\theta_t)$$ for the natural gradient direction $$s_t$$. This requires efficiently computing matrix-vector products $$I(\theta)v$$, which can sometimes be done without forming $$I(\theta)$$ itself (e.g., using "Pearlmutter's trick" or finite differences).

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Connection to Main Series.** Adam and Information Geometry
</div>
One of the key posts in the main optimization series will explore the Adam optimizer. We will discuss how its adaptive learning rate mechanism, particularly the use of squared past gradients (second moment estimates), can be interpreted as an approximation of a diagonal Fisher Information Metric. This IG view helps explain why Adam often works well by implicitly accounting for the varying sensitivity of the loss function to different parameters.
</blockquote>

## Mirror Descent

Mirror Descent is a powerful generalization of gradient descent that is deeply connected to Information Geometry, especially in the context of dually flat spaces and Bregman divergences.

The standard gradient descent step can be viewed as solving:

$$
\theta_{t+1} = \arg \min_{\theta} \left\{ \langle \nabla L(\theta_t), \theta - \theta_t \rangle + \frac{1}{2\lr} \Vert \theta - \theta_t \Vert^2 \right\}
$$

This penalizes deviation from $$\theta_t$$ using squared Euclidean distance.

Mirror Descent replaces the Euclidean distance with a Bregman divergence $$D_\psi(\cdot \Vert  \cdot)$$ associated with a strictly convex "potential" or "mirror map" function $$\psi$$:

$$
\theta_{t+1} = \arg \min_{\theta \in \mathcal{D}} \left\{ \langle \nabla L(\theta_t), \theta \rangle + \frac{1}{\lr} D_\psi(\theta \Vert  \theta_t) \right\}
$$

where $$\mathcal{D}$$ is the domain of $$\psi$$.

The update often involves a "mapping" to a "dual space" (defined by $$\nabla \psi$$), performing an update there, and then "mapping back".
1.  Map to dual space: $$\eta_t = \nabla \psi(\theta_t)$$
2.  Update in dual space (like gradient step): $$\tilde{\eta}_{t+1} = \eta_t - \lr \nabla L(\theta_t)$$
3.  Map back to primal space: $$\theta_{t+1} = (\nabla \psi)^{-1}(\tilde{\eta}_{t+1})$$ (where $$(\nabla \psi)^{-1} = \nabla \psi^\ast $$ for the Legendre dual $$\psi^\ast $$).

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Exponentiated Gradient from Mirror Descent
</div>
If parameters $$\theta$$ represent a probability distribution over $$K$$ items (i.e., $$\theta_i \ge 0, \sum \theta_i = 1$$, living on the probability simplex), and we choose $$\psi(\theta) = \sum_i \theta_i \log \theta_i$$ (negative entropy), the corresponding Bregman divergence is the unnormalized KL divergence.
Mirror descent with this setup leads to the **Exponentiated Gradient** algorithm, which involves multiplicative updates:

$$
\theta_i^{t+1} \propto \theta_i^t \exp(-\lr (\nabla L(\theta_t))_i)
$$

This is particularly useful for online learning settings like prediction with expert advice.
</blockquote>

**Connection to Natural Gradient:**
In dually flat spaces (like exponential families), if $$\psi$$ is the potential function associated with the e-affine coordinates $$\theta$$ (e.g., log-partition function), then Mirror Descent using $$D_\psi$$ is equivalent to Natural Gradient Ascent (for maximization) in the $$\theta$$ coordinates. The "mirror map" effectively transforms the problem into a space where Euclidean gradient steps are natural.

## Other Connections and Advanced Topics (Briefly)

Information Geometry offers insights into many other areas of statistics and machine learning:

1.  **Expectation-Maximization (EM) Algorithm:**
    The EM algorithm, used for maximum likelihood estimation with latent variables, can be interpreted geometrically. Each E-step and M-step can be seen as projections in a dually flat space (often involving KL divergence). The E-step projects the current estimate onto a manifold of distributions consistent with observed data and expected latent variables, and the M-step projects back to the model manifold. This is often called an "em-algorithm" (not a typo, e for expectation/exponential, m for maximization/mixture).

2.  **Information Criteria (AIC, BIC):**
    Model selection criteria like AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) penalize model complexity. AIC, in particular, has roots in estimating the KL divergence between the true distribution and the model, and relates to the geometry of the model manifold.

3.  **Minimum Description Length (MDL):**
    The MDL principle, related to Kolmogorov complexity, seeks models that provide the shortest description of the data. This has deep connections to information theory and can be studied using IG tools.

4.  **Non-Parametric Information Geometry:**
    Extending IG concepts to infinite-dimensional statistical manifolds (e.g., spaces of all probability density functions) is an active research area, important for kernel methods and Bayesian non-parametrics.

5.  **Information Bottleneck Principle:**
    This principle aims to find compressed representations of an input variable that preserve as much information as possible about a relevant target variable. It can be formulated using KL divergences and studied geometrically.

## Limitations and Open Research Areas

While powerful, IG faces challenges:
-   **Computational Cost:** As mentioned, the FIM is often too costly for direct use in large-scale ML. Developing scalable approximations remains crucial.
-   **Non-Identifiability:** If parameters are not identifiable (multiple $$\theta$$ values give the same distribution $$p(x;\theta)$$), the FIM becomes singular, and the manifold structure is degenerate.
-   **Beyond Exponential Families:** While IG is very elegant for exponential families (which are dually flat), its application to more general models (like complex neural networks that are not necessarily dually flat globally) can be more intricate. Local approximations are often used.
-   **Choice of $$\alpha$$:** While $$\alpha=\pm 1$$ (e- and m-connections) are canonical, the role and optimal choice of other $$\alpha$$-connections for specific problems is an area of study.

## Conclusion of the Information Geometry Crash Course

Across these three parts, we've journeyed from the basic idea of viewing statistical models as manifolds to the sophisticated machinery of dual connections, Bregman divergences, and the natural gradient. We've seen that the Fisher Information Metric provides a statistically meaningful way to define geometry, and this geometric perspective can:
-   Offer deep insights into the structure of statistical models and inference.
-   Lead to the development of more principled and efficient optimization algorithms.
-   Provide a unifying framework for understanding various concepts in information theory and machine learning.

The natural gradient, despite its computational challenges, stands as a prime example of how IG can directly inspire practical algorithms. Many modern adaptive optimizers in deep learning can be seen as trying to capture some of the benefits of NGD in a computationally feasible way.

As you proceed with the main series on mathematical optimization in ML, the concepts from this Information Geometry crash course, particularly the FIM and the natural gradient, will resurface when we analyze advanced optimizers and discuss the geometry of loss landscapes. We hope this introduction has equipped you with a new lens through which to view these topics!

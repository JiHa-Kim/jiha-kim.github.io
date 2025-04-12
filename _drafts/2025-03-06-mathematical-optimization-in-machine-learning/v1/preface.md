---
layout: post
title: Preface
date: 2025-03-06 00:00:00 +0000
description:
image:
categories:
tags:
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.
  Please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by two newlines above and below, and newlines between the delimiters and the equation (even in lists):

  $$
  block
  $$

  Use LaTeX commands for symbols as much as possible such as $$\vert$$ or $$\ast$$. For instance, please avoid using the vertical bar symbol, only use \vert for absolute value, and \Vert for norm.

  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$
  3. item

    $$
    block
    $$

    (continued) item


  Inside HTML environments (like blockquotes), please use the following syntax:

  \( inline \)

  \[
  block
  \]

  like so. Also, HTML markers must be used rather than markdown, e.g. <b>bold</b> rather than **bold**, and <i>italic</i> rather than *italic*.

  Example:

  <blockquote class="prompt-info">
    <b>Definition (Vector Space):</b>
    A vector space \(V\) is a set of vectors equipped with a <b>scalar multiplication</b> operation:

    \[
    \forall v, w \in V, \quad \forall \alpha \in \mathbb{R}, \quad v \cdot (w \cdot \alpha) = v \cdot (\alpha \cdot w)
    \]

    where \(\cdot\) is the <b>dot product</b> of two vectors.
  </blockquote>

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

I will present the structure of this blog post series.

1. Goal
2. Approach
3. Prerequisites

## Goal

Optimization is a cornerstone in modern machine learning. Whether we are fitting a model to data, tuning hyperparameters, or learning weights in a neural network, we are solving optimization problems that are often large, noisy, and non-convex.

Yet, there tends to be a significant disconnect between the theory and practice of optimization in machine learning. This blog post series aims to explore the mathematical landscape of **optimization theory** applied to machine learning.

The series is mostly intended to be a **self-contained introduction**, theoretically grounded and accessible.

Our goal is not just to apply optimization algorithms, but to **understand** them. Throughout this series, we will see how many concepts of optimization theory are inspired and cleverly re-used from physics and mathematics, and how they relate to machine learning. We will explore:
- The **intuition** behind optimization algorithms and how they emerged through connections to other fields.
- Why certain methods succeed (or fail).
- How theoretical guarantees relate to practical performance.
- When and how to choose (or design) the right optimizer.

## Approach

This series takes a **problem-first, theory-second** top-down approach. The presentation is centered on mathematical derivations rather than code or implementation details. The order of presentation is intended to be:

1. Real-world problem
2. Intuition with a concrete example
3. Investigating desired properties
4. Formalizing the theory
5. Applying the theory to practical problems

We draw from diverse perspectives:
- **First-order methods** like SGD and Adam.
- **Second-order insights** from Newton’s method.
- **Bayesian interpretations** of regularization.
- **Convex duality and variational principles**.
- **Continuous-time and physical analogies**.
- **Modern adaptive optimizers and deep learning practices**.

## Prerequisites

This series is designed for readers with:
- A working knowledge of **linear algebra** and **calculus**.
- Comfort with **mathematical notation and reasoning**.
- Basic familiarity with **machine learning terminology** (e.g., regression, classification, neural networks).

We build from foundational principles, but we don't shy away from depth. Expect equations, proofs, and careful arguments — always in service of clarity and insight.

## Series Outline

<b>What to Expect:</b> Here's a preview of the full series.

1. <b>The Optimization Quest Begins</b>: Why gradients win, and why Newton fails to scale.
2. <b>Gradient Descent in Practice</b>: From steepest descent to noisy SGD and tuning learning rates.
3. <b>Momentum Matters</b>: Acceleration and navigation through ravines.
4. <b>Adaptive Methods</b>: Adagrad, RMSProp, and the Adam revolution.
5. <b>Regularization and Priors</b>: The Bayesian story behind weight decay.
6. <b>Non-Convexity and Deep Learning</b>: Landscapes, plateaus, and generalization puzzles.
7. <b>Optimization as Flow</b>: The ODE view of gradient methods.
8. <b>Variational Viewpoints</b>: Physics, paths, and duality.
9. <b>Convex Optimization and Duality</b>: Solid ground for guarantees.
10. <b>Proximal Algorithms</b>: Handling non-smoothness with care.
11. <b>Curvature Tools</b>: Preconditioning, mirror descent, and L-BFGS.
12. <b>Next-Gen Optimizers</b>: Shampoo, Muon, and the structured future.
13. <b>Practical Tradeoffs</b>: Computation, scale, and choosing your optimizer.
14. <b>The Optimization Field Guide</b>: Summary, cheat sheet, and reflections.

Let’s begin.
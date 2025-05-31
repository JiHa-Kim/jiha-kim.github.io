---
title: "Adam Optimizer: Online Learning of Updates and Efficacy with EMA"
date: 2025-05-25 10:00 -0400
series_index: 13
mermaid: true
description: "Exploring novel theoretical understandings of the Adam optimizer through the lens of Follow-The-Regularized-Leader for its updates, and the provable benefits of combining Adam with model exponential moving average in non-convex optimization."
image: 
categories:
  - Mathematical Optimization
  - Machine Learning
tags:
  - Adam Optimizer
  - Online Learning
  - FTRL
  - Exponential Moving Average
  - Non-convex Optimization
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

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
  symbol; use \vert and \Vert.

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

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

## Introduction: Bridging Adam and Online Learning

The Adam optimizer has become ubiquitous in deep learning due to its robust performance across diverse architectures and datasets. Yet its theoretical foundations have remained elusive, with prior analyses failing to explain *why* Adam works so well in practice. Recent breakthroughs by Ahn (2024a) and Ahn et al. (2024b) reveal Adam's profound connection to **online learning theory** while providing optimal convergence guarantees for nonconvex optimization when combined with model exponential moving average (EMA).

In this post, we'll explore:
1. How Adam implements **discounted Follow-the-Regularized-Leader (β-FTRL)**
2. Why momentum corresponds to **loss discounting** in online learning
3. How EMA enables **optimal convergence** in nonconvex settings
4. When coordinate-wise adaptivity provides **provable acceleration**

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Standard Adam Update
</div>
For gradients $$\mathbf{g}_t$$, Adam computes:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2 \\
\Delta_t &= -\alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon},\quad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
\end{aligned}
$$

</blockquote>

## Online Learning of Updates Framework

The key insight from Ahn (2024a) is to reframe optimization as **online learning of parameter increments**. Instead of directly learning parameters $$\mathbf{x}_t$$, we learn updates $$\Delta_t$$:

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta_{t-1}
$$

At each step $$t$$, the learner suffers a linear loss:

$$
\ell_t(\Delta) = \langle \mathbf{v}_t, \Delta \rangle
$$

where $$\mathbf{v}_t$$ is constructed from gradient history. This **Online Learning of Updates (OLU)** framework transforms optimization into an online decision problem.

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation 1: Constructing Discounted Losses**
</summary>
To recover Adam, we define scaled gradients with exponential discounting:

$$
\mathbf{v}_s = \beta_1^{-s} \mathbf{g}_s
$$

The cumulative discounted loss becomes:

$$
L_t(\Delta) = \sum_{s=1}^t \beta_1^{t-s} \langle \mathbf{v}_s, \Delta \rangle = \beta_1^t \left\langle \sum_{s=1}^t \beta_1^{-s}\mathbf{g}_s, \Delta \right\rangle
$$

This linear loss function encodes the entire gradient history with exponential decay controlled by $$\beta_1$$. The discount factor $$\beta_1^{t-s}$$ assigns higher weight to recent gradients.
</details>

## Adam as Discounted Follow-the-Regularized-Leader

Adam emerges naturally as the solution to a **discounted FTRL** problem with adaptive regularization:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Adam is β-FTRL
</div>
The Adam update solves:

$$
\Delta_t = \underset{\Delta \in \mathbb{R}^d}{\text{argmin}} \left( \eta_t \sum_{s=1}^t \beta_1^{t-s} \langle \mathbf{v}_s, \Delta \rangle + \frac{1}{2} \|\Delta\|_2^2 \right)
$$

with learning rate schedule:

$$
\eta_t = \alpha \beta_1^t / \sqrt{\sum_{s=1}^t \mathbf{v}_s^2}
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation 2: Solving the FTRL Objective**
</summary>
1. Take the gradient of the FTRL objective and set to zero:

   $$
   \nabla_{\Delta} \left[ \eta_t \beta_1^t \sum_{s=1}^t \beta_1^{-s} \langle \mathbf{g}_s, \Delta \rangle + \frac{1}{2} \|\Delta\|_2^2 \right] = 0
   $$

2. Substitute $$\mathbf{v}_s = \beta_1^{-s}\mathbf{g}_s$$:

   $$
   \eta_t \beta_1^t \sum_{s=1}^t \mathbf{v}_s + \Delta = 0
   $$

3. Solve for $$\Delta$$:

   $$
   \Delta_t = -\eta_t \beta_1^t \sum_{s=1}^t \mathbf{v}_s
   $$

4. Plug in the learning rate:

   $$
   \Delta_t = -\alpha \beta_1^t \frac{\sum_{s=1}^t \mathbf{v}_s}{\sqrt{\sum_{s=1}^t \mathbf{v}_s^2}} \cdot \frac{1}{\beta_1^t} = -\alpha \frac{\sum_{s=1}^t \mathbf{v}_s}{\sqrt{\sum_{s=1}^t \mathbf{v}_s^2}}
   $$

5. Recover standard Adam terms:

   $$
   \begin{aligned}
   \text{Numerator: } \sum_{s=1}^t \mathbf{v}_s &= \beta_1^{-t} \mathbf{m}_t \\
   \text{Denominator: } \sqrt{\sum_{s=1}^t \mathbf{v}_s^2} &= \beta_1^{-t} \sqrt{\mathbf{v}_t} + \mathcal{O}(\epsilon)
   \end{aligned}
   $$

   Thus:

   $$
   \Delta_t = -\alpha \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}
   $$

</details>

This derivation reveals Adam's components as fundamental to online learning:
- **Momentum $$\beta_1$$**: Discount factor for past losses
- **Adaptive scaling $$\mathbf{v}_t$$**: Scale-free regularization
- **Learning rate $$\alpha$$**: Global step size scaling

## Dynamic Regret Analysis

The OLU framework enables rigorous analysis through **dynamic regret**:

$$
\mathcal{R}_T = \sum_{t=1}^T \ell_t(\Delta_t) - \min_{\{\mathbf{u}_t\}} \sum_{t=1}^T \ell_t(\mathbf{u}_t - \mathbf{u}_{t-1})
$$

where $$\mathbf{u}_t$$ is a comparator sequence. Adam achieves strong regret bounds:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Dynamic Regret of β-FTRL
</div>
For any comparator sequence $$\{\mathbf{u}_t\}$$ with $$\|\mathbf{u}_t - \mathbf{u}_{t-1}\| \leq D$$:

$$
\mathcal{R}_T \leq \frac{\alpha}{\sqrt{1-\beta_1}} \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2} + \frac{D^2}{2\alpha} \sqrt{\sum_{t=1}^T \|\mathbf{u}_t - \mathbf{u}_{t-1}\|^2}
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof Sketch: Regret Decomposition**
</summary>
1. Decompose regret into stability and prediction terms:

   $$
   \mathcal{R}_T = \underbrace{\sum_t \ell_t(\Delta_t) - \sum_t \ell_t(\mathbf{u}_t)}_{\text{Static regret}} + \underbrace{\sum_t \langle \mathbf{v}_t, \mathbf{u}_t - \mathbf{u}_{t-1} \rangle}_{\text{Stability penalty}}
   $$

2. Bound static regret via scale-free FTRL analysis:

   $$
   \sum_{t=1}^T \langle \mathbf{v}_t, \Delta_t - \mathbf{u} \rangle \leq \frac{\alpha}{\sqrt{1-\beta_1}} \sum_{i=1}^d \|\mathbf{g}_{1:T,i}\|_2
   $$

3. Control stability term with path length:

   $$
   \sum_t \langle \mathbf{v}_t, \mathbf{u}_t - \mathbf{u}_{t-1} \rangle \leq \frac{D^2}{2\alpha} \sqrt{\sum_{t=1}^T \|\mathbf{u}_t - \mathbf{u}_{t-1}\|^2}
   $$

4. Combine using Cauchy-Schwarz:

   $$
   \mathcal{R}_T \leq \frac{\alpha}{\sqrt{1-\beta_1}} \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2} + \frac{D^2}{2\alpha} \sqrt{\sum_{t=1}^T \|\mathbf{u}_t - \mathbf{u}_{t-1}\|^2}
   $$

</details>

## EMA for Nonconvex Optimization

While Adam excels in online settings, nonconvex optimization requires additional stabilization. Ahn et al. (2024b) prove that combining clipped Adam with **model EMA** achieves optimal convergence:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Convergence of Adam+EMA
</div>
For $$L$$-smooth nonconvex $$F$$ with stochastic gradients satisfying $$\mathbb{E}[\|\mathbf{g}_t - \nabla F(\mathbf{x}_t)\|^2] \leq \sigma^2$$, clipped Adam with EMA attains:

$$
\mathbb{E}\left[ \|\nabla F(\bar{\mathbf{x}}_T)\|^2 \right] \leq \mathcal{O}\left( \frac{\sigma}{\sqrt{T}} + \frac{\sigma^2}{T} \right)
$$

which matches the lower bound for nonsmooth nonconvex optimization.
</blockquote>

The EMA update:

$$
\bar{\mathbf{x}}_t = (1 - \gamma)\bar{\mathbf{x}}_{t-1} + \gamma\mathbf{x}_t
$$

provides three key benefits:
1. **Smoothing**: Convexifies the optimization landscape
2. **Variance Reduction**: Averages out gradient noise
3. **Implicit Iterate Averaging**: Stabilizes convergence

<details class="details-block" markdown="1">
<summary markdown="1">
**Why EMA Outperforms Uniform Averaging**
</summary>
Compared to uniform averaging $$\bar{\mathbf{x}}_T = \frac{1}{T} \sum_{t=1}^T \mathbf{x}_t$$, EMA:
- Assigns higher weight to recent iterates
- Adapts faster to curvature changes
- Requires only $$\mathcal{O}(d)$$ memory
- Provides better empirical performance in generative models

The optimal weighting scheme balances:

$$
\gamma_t \propto \mathbb{E}[\|\nabla F(\mathbf{x}_t)\|^{-1}]
$$

which EMA approximates through exponential discounting.
</details>

## Coordinate-Wise Adaptivity Advantage

Adam's per-coordinate scaling provides provable acceleration under **gradient scale heterogeneity**:

| **Condition**  | **Isotropic Methods**                              | **Adam (β-FTRL)**                                     |
| -------------- | -------------------------------------------------- | ----------------------------------------------------- |
| Uniform scales | $$\mathcal{O}(\sqrt{dT})$$                         | $$\mathcal{O}(\sqrt{dT})$$                            |
| High variance  | $$\mathcal{O}(\sqrt{d\sum_t \|\mathbf{g}_t\|^2})$$ | $$\mathcal{O}(\sum_{i=1}^d \sqrt{\sum_t g_{t,i}^2})$$ |

The coordinate-adaptive regret bound:

$$
\mathcal{R}_T \leq \mathcal{O}\left( \sum_{i=1}^d \sqrt{\sum_{t=1}^T g_{t,i}^2 } \right)
$$

can be $$\sqrt{d}$$-times smaller than isotropic methods when gradient norms vary significantly across coordinates.

## Summary: Online Learning Perspective

### Adam Component Correspondence
| **Adam Element**         | **Online Learning Interpretation** |
| ------------------------ | ---------------------------------- |
| Momentum $$\beta_1$$     | Loss discount factor               |
| Scaling $$\mathbf{v}_t$$ | Adaptive regularization            |
| Learning rate $$\alpha$$ | Step size multiplier               |
| EMA $$\gamma$$           | Implicit iterate averaging         |

### Key Insights
1. Adam implements **discounted FTRL** for update directions
2. Momentum enables **adaptation to non-stationarity**
3. EMA provides **optimal convergence** in nonconvex settings
4. Per-coordinate scaling **accelerates convergence** under gradient heterogeneity

## References
1. Ahn, K. (2024a). *Understanding Adam Optimizer via Online Learning of Updates: Adam is FTRL in Disguise*. arXiv:2402.01567  
2. Ahn, K., Lee, J. D., Sra, S., & Oh, S. (2024b). *Adam with model exponential moving average is effective for nonconvex optimization*. arXiv:2405.18199  
3. Kingma, D. P., & Ba, J. L. (2015). *Adam: A Method for Stochastic Optimization*. ICLR  
4. McMahan, H. B. (2017). *A Survey of Algorithms and Analysis for Adaptive Online Learning*. JMLR

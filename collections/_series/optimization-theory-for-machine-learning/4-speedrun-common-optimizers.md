---
title: "Speedrun of Common Gradient-Based ML Optimizers"
date: 2025-05-20 00:45 -0400
series_index: 4
mermaid: true
description: A quick tour of popular gradient-based optimization algorithms in machine learning, detailing their mechanics and empirical performance characteristics.
image:
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Optimizers
- Gradient Descent
- SGD
- Momentum
- AdaGrad
- RMSProp
- Adam
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

In our previous exploration of iterative optimization methods, we established the prominence of gradient-based techniques in machine learning. These methods, which leverage the gradient of the loss function to guide the search for optimal parameters, are the backbone of training for many ML models, especially deep neural networks. However, "gradient-based optimization" is a broad term encompassing a diverse family of algorithms. The choice of a specific optimizer can dramatically affect not only the speed of convergence but also the final performance of the trained model.

This post is a "speedrun" through some of the most common gradient-based optimizers. Our goal is to quickly familiarize ourselves with their core mechanics and empirical behavior. We'll focus on the update rules and the practical implications of each algorithm, deferring deeper theoretical analyses for later installments.

Let $$\theta_t$$ denote the model parameters at iteration $$t$$, and $$J(\theta)$$ be the objective (loss) function we aim to minimize. The gradient of the loss function with respect to the parameters at iteration $$t$$ is $$g_t = \nabla_{\theta} J(\theta_t)$$. In practice, especially for large datasets, $$g_t$$ is often computed on a mini-batch of data, making the methods stochastic. Most gradient-based optimizers follow a general update rule:

$$
\theta_{t+1} = \theta_t - \Delta \theta_t
$$

The crucial difference between optimizers lies in how they determine the update step $$\Delta \theta_t$$. Let's dive in.

## Foundational Methods

We begin with the simplest gradient-based methods, which form the basis for more advanced techniques.

### 1. Stochastic Gradient Descent (SGD)

**Intuition:**
SGD approximates the true gradient using a mini-batch of data, nudging parameters in the negative direction of this estimate.

**Algorithm:**
Require: Learning rate $$\eta$$, initial parameters $$\theta_0$$.
Initialize parameters $$\theta = \theta_0$$.
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_t$$ in the training data:
     1. Compute gradient: $$g_t = \nabla_{\theta} J(\theta_t; \mathcal{B}_t)$$.
     2. Update parameters:

        $$
        \theta_{t+1} = \theta_t - \eta g_t
        $$

**Empirical Performance:**
*   **Pros:**
    *   Computationally inexpensive per update, making it scalable to large datasets.
    *   The noise from mini-batch sampling can help escape shallow local minima and saddle points.
*   **Cons:**
    *   High variance in parameter updates due to noisy gradients, leading to a characteristic "zig-zag" path towards the minimum.
    *   Can be slow to converge if the loss landscape has high curvature (e.g., narrow ravines) or flat plateaus.
    *   Highly sensitive to the choice of learning rate $$\eta$$ and requires careful tuning of a learning rate schedule (decaying $$\eta$$ over time) for best performance.

<blockquote class="prompt-tip" markdown="1">
**Tip:** Vanilla SGD is often used as a baseline. Despite its simplicity, well-tuned SGD (with momentum and a good learning rate schedule) can achieve state-of-the-art results on many tasks.
</blockquote>

### 2. SGD with Momentum

**Intuition:**
Momentum adds a fraction of the previous update vector to the current one, accelerating SGD in relevant directions and dampening oscillations, like a ball rolling down a hill.

**Algorithm:**
Require: Learning rate $$\eta$$, momentum parameter $$\gamma$$ (e.g., 0.9), initial parameters $$\theta_0$$.
Initialize parameters $$\theta = \theta_0$$.
Initialize velocity $$v_0 = \mathbf{0}$$.
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_t$$ in the training data:
     1. Compute gradient: $$g_t = \nabla_{\theta} J(\theta_t; \mathcal{B}_t)$$.
     2. Update velocity:

        $$
        v_t = \gamma v_{t-1} + \eta g_t
        $$

     3. Update parameters:

        $$
        \theta_{t+1} = \theta_t - v_t
        $$

**Empirical Performance:**
*   **Pros:**
    *   Significantly accelerates convergence compared to plain SGD, especially in landscapes with high curvature or consistent gradients.
    *   Dampens oscillations common in SGD, leading to a more direct path to the minimum.
*   **Cons:**
    *   Introduces a new hyperparameter, $$\gamma$$, which needs tuning.
    *   The accumulated momentum can cause the optimizer to overshoot minima or oscillate around them if not well-tuned.

### 3. Nesterov Accelerated Gradient (NAG)

**Intuition:**
NAG "looks ahead" by calculating the gradient at a position projected by the previous momentum step, then makes a correction. This anticipates movement and can prevent overshooting.

**Algorithm:**
Require: Learning rate $$\eta$$, momentum parameter $$\gamma$$, initial parameters $$\theta_0$$.
Initialize parameters $$\theta = \theta_0$$.
Initialize velocity $$v_0 = \mathbf{0}$$.
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_t$$ in the training data:
     1. Compute gradient at the "look-ahead" position:

        $$
        g_t = \nabla_{\theta} J(\theta_t - \gamma v_{t-1}; \mathcal{B}_t)
        $$

     2. Update velocity:

        $$
        v_t = \gamma v_{t-1} + \eta g_t
        $$

     3. Update parameters:

        $$
        \theta_{t+1} = \theta_t - v_t
        $$

**Empirical Performance:**
*   **Pros:**
    *   Often converges faster than classical momentum, particularly for convex optimization problems.
    *   The look-ahead mechanism provides a better "correction factor" to the momentum, leading to more stable and efficient updates.
*   **Cons:**
    *   Slightly more complex to implement than classical momentum.
    *   Like momentum, it introduces the hyperparameter $$\gamma$$.

## Adaptive Learning Rate Methods

These methods adjust the learning rate for each parameter individually.

### 4. AdaGrad (Adaptive Gradient Algorithm)

**Intuition:**
AdaGrad adapts learning rates by making them inversely proportional to the square root of the sum of all historical squared values of that gradient. This means parameters with large past gradients get smaller learning rates.

**Algorithm:**
Require: Global learning rate $$\eta$$, initial parameters $$\theta_0$$, small constant $$\epsilon$$ (e.g., $$10^{-8}$$).
Initialize parameters $$\theta = \theta_0$$.
Initialize gradient accumulator $$G_0 = \mathbf{0}$$ (same shape as $$\theta$$).
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_t$$ in the training data:
     1. Compute gradient: $$g_t = \nabla_{\theta} J(\theta_t; \mathcal{B}_t)$$.
     2. Accumulate squared gradients (element-wise):

        $$
        G_t = G_{t-1} + g_t \odot g_t
        $$

     3. Update parameters (element-wise operations for division and square root):

        $$
        \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t
        $$

**Empirical Performance:**
*   **Pros:**
    *   Well-suited for problems with sparse features, as it assigns higher learning rates to infrequent parameters.
    *   Eliminates the need to manually tune the learning rate as aggressively; the global learning rate $$\eta$$ is less sensitive.
*   **Cons:**
    *   The main drawback is that the learning rate monotonically decreases because the sum of squared gradients $$G_t$$ in the denominator keeps growing. This can cause the learning rate to become infinitesimally small, prematurely halting training.

### 5. RMSProp (Root Mean Square Propagation)

**Intuition:**
RMSProp modifies AdaGrad by using an exponentially decaying average of squared gradients, preventing the learning rate from shrinking too quickly.

**Algorithm:**
Require: Global learning rate $$\eta$$, decay rate $$\beta_2$$ (e.g., 0.9), initial parameters $$\theta_0$$, small constant $$\epsilon$$ (e.g., $$10^{-8}$$).
Initialize parameters $$\theta = \theta_0$$.
Initialize moving average of squared gradients $$E[g^2]_0 = \mathbf{0}$$.
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_t$$ in the training data:
     1. Compute gradient: $$g_t = \nabla_{\theta} J(\theta_t; \mathcal{B}_t)$$.
     2. Update decaying average of squared gradients (element-wise):

        $$
        E[g^2]_t = \beta_2 E[g^2]_{t-1} + (1-\beta_2) (g_t \odot g_t)
        $$

     3. Update parameters:

        $$
        \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \odot g_t
        $$

**Empirical Performance:**
*   **Pros:**
    *   Effectively resolves AdaGrad's issue of rapidly vanishing learning rates.
    *   Generally converges faster and performs better than AdaGrad in non-stationary settings.
    *   Still provides per-parameter learning rate adaptation.
*   **Cons:**
    *   Requires tuning of the decay rate $$\beta_2$$ in addition to the global learning rate $$\eta$$.

## Hybrid Methods

These methods combine momentum with adaptive learning rates.

### 6. Adam (Adaptive Moment Estimation)

**Intuition:**
Adam computes adaptive learning rates for each parameter from an estimate of both the first moment (mean, like momentum) and the second moment (uncentered variance) of the gradients. It also includes bias correction for these estimates.

**Algorithm:**
Require: Step size $$\eta$$ (e.g., 0.001), exponential decay rates for moment estimates $$\beta_1$$ (e.g., 0.9) and $$\beta_2$$ (e.g., 0.999), small constant $$\epsilon$$ (e.g., $$10^{-8}$$), initial parameters $$\theta_0$$.
Initialize parameters $$\theta = \theta_0$$.
Initialize 1st moment vector $$m_0 = \mathbf{0}$$.
Initialize 2nd moment vector $$v_0 = \mathbf{0}$$.
Initialize timestep $$t = 0$$.
Loop for each epoch or until convergence:
  1. For each mini-batch $$\mathcal{B}_{\text{iter}}$$ in the training data (let current iteration be `iter`):
     1. Increment timestep: $$t = t + 1$$.
     2. Compute gradient: $$g_t = \nabla_{\theta} J(\theta_{t-1}; \mathcal{B}_{\text{iter}})$$.
     3. Update biased first moment estimate:

        $$
        m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
        $$

     4. Update biased second raw moment estimate (element-wise square):

        $$
        v_t = \beta_2 v_{t-1} + (1-\beta_2) (g_t \odot g_t)
        $$

     5. Compute bias-corrected first moment estimate:

        $$
        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
        $$

     6. Compute bias-corrected second raw moment estimate:

        $$
        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
        $$

     7. Update parameters:

        $$
        \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
        $$

     *(Note: The parameter update here is shown from $$\theta_{t-1}$$ to $$\theta_t$$ to align with the timestep $$t$$ used for bias correction. Some presentations might show $$\theta_{t+1} = \theta_t - \dots$$; ensure consistency in indexing.)*

**Empirical Performance:**
*   **Pros:**
    *   Combines the benefits of momentum (faster convergence through accumulated gradients) and adaptive learning rates (per-parameter scaling).
    *   Often works very well in practice across a wide range of problems and architectures with little hyperparameter tuning (default values for $$\beta_1, \beta_2, \epsilon$$ are often effective).
    *   Computationally efficient and has low memory requirements.
*   **Cons:**
    *   Despite its popularity, Adam is not a panacea. On some problems, it may converge to less optimal solutions compared to well-tuned SGD with momentum.
    *   Concerns about its generalization performance compared to SGD have been raised in some studies, though this is an area of ongoing research and debate. Variants like AdamW (Adam with decoupled weight decay) aim to address some of these issues.
    *   While defaults are good, optimal performance may still require tuning of $$\eta, \beta_1, \beta_2$$.

## Summary / Cheat Sheet

Here's a quick comparison of the optimizers discussed:

| Optimizer    | Key Idea                                                                 | Simplified Update Sketch ($$\Delta \theta_t$$ for step from $$\theta_t$$)                 | Pros                                                                                    | Cons                                                                                  |
| ------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **SGD**      | Use mini-batch gradient, simple update.                                  | $$\eta g_t$$                                                                              | Simple, cheap updates, noise can escape local minima.                                   | High variance, slow in ravines/plateaus, sensitive to $$\eta$$.                       |
| **Momentum** | Accumulate past gradients to accelerate and dampen oscillations.         | $$v_t = \gamma v_{t-1} + \eta g_t$$ <br/> (update is $$v_t$$)                             | Faster convergence, dampens oscillations.                                               | New hyperparameter ($$\gamma$$), can overshoot.                                       |
| **NAG**      | "Look ahead" before computing gradient for momentum step.                | $$g'_t = \nabla J(\theta_t - \gamma v_{t-1})$$ <br/> $$v_t = \gamma v_{t-1} + \eta g'_t$$ | Often better convergence than classical momentum, more responsive.                      | Slightly more complex, hyperparameter $$\gamma$$.                                     |
| **AdaGrad**  | Per-parameter LR, smaller for frequent/large past gradients.             | $$\frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$$                                          | Good for sparse data, less LR tuning.                                                   | LR decays too aggressively, may stop training early.                                  |
| **RMSProp**  | AdaGrad with decaying average of squared gradients.                      | $$\frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \odot g_t$$                                     | Solves AdaGrad's decaying LR, good for non-stationary problems.                         | New hyperparameter ($$\beta_2$$).                                                     |
| **Adam**     | Combines momentum (1st moment) and RMSProp-like adaptivity (2nd moment). | $$\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$                                    | Often works well with defaults, combines benefits of momentum & adaptive LR. Efficient. | Can have generalization issues, may not always be optimal, still has hyperparameters. |

<div class="box-info" markdown="1">
In the table above:
- $$g_t$$ is the gradient at iteration $$t$$ (or computed using $$\theta_t$$ or a lookahead based on $$\theta_t$$).
- For Momentum and NAG, $$v_t$$ represents the velocity vector whose components are used for the update.
- For AdaGrad, $$G_t$$ is the sum of past squared gradients up to iteration $$t$$.
- For RMSProp, $$E[g^2]_t$$ is the decaying average of squared gradients at iteration $$t$$.
- For Adam, $$\hat{m}_t$$ and $$\hat{v}_t$$ are the bias-corrected first and second moment estimates at iteration $$t$$.
The "Simplified Update Sketch" shows the term subtracted from $$\theta_t$$ to get $$\theta_{t+1}$$.
</div>

## Reflection

This speedrun has introduced us to a selection of popular gradient-based optimizers, from the fundamental SGD to the sophisticated Adam. Each algorithm builds upon the insights and addresses the limitations of its predecessors, offering different strategies for navigating complex loss landscapes.

The existence of such a diverse "zoo" of optimizers underscores a crucial point: **there is no single "best" optimizer for all machine learning problems.** The optimal choice often depends on the specific characteristics of the dataset, the model architecture, the nature of the loss surface, and available computational resources. Factors like convergence speed, generalization ability, sensitivity to hyperparameters, and memory footprint all play a role in this decision.

While adaptive methods like Adam are often excellent default choices due to their robustness and ease of use, understanding the mechanics of simpler methods like SGD with Momentum remains vital. Not only do they form the building blocks for more advanced techniques, but they can also, when carefully tuned, outperform more complex optimizers on certain tasks.

This survey provides a practical overview. In subsequent posts, we will delve deeper into the theoretical underpinnings of concepts like momentum and adaptive learning rates, exploring *why* they work, their connections to concepts like preconditioning, and how they attempt to tackle the challenges of high-dimensional, non-convex optimization common in modern machine learning.
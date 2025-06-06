---
title: "Online Learning Crash Course – Part 7: Beyond Convexity (Teaser)"
date: 2025-06-01 09:00 -0400
sort_index: 7
mermaid: true
description: A brief look into online learning scenarios beyond convex optimization, including bandit feedback and non-convex losses.
image: # placeholder
categories:
- Machine Learning
- Online Learning
tags:
- Online Learning
- Non-convex Optimization
- Bandit Algorithms
- Reinforcement Learning
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

The majority of this crash course has centered on Online Convex Optimization (OCO), where convexity of the decision set and loss functions enables powerful regret guarantees. However, the principles of online learning extend to more challenging scenarios, including those with non-convex losses and limited feedback. This module provides a brief teaser into these advanced areas.

## 1. Road-map at a Glance

We are currently at **Module 7: Beyond Convexity (Teaser)**.

```mermaid
flowchart TD
    C0("Setting & Motivation") --> C1["Regret & Benchmarks"]
    C1 --> C2["Gradient-based Algorithms (OGD)"]
    C2 --> C3["Regularization: FTRL"]
    C3 --> C4["Mirror Descent & Geometry"]
    C4 --> C5["Adaptivity (AdaGrad et al.)"]
    C5 --> C6["Online-to-Batch Conversions"]
    C6 --> C7["Beyond Convexity (Teaser)"]
    C7 --> C8["Summary & Practical Guidance"]

    style C7 fill:#E0BBE4,color:#000,stroke:#333,stroke-width:2px % Highlight current module
```

<details class="details-block" markdown="1">
<summary markdown="1">
**Navigational Tip.** This flowchart will appear in each post, with the current module highlighted, to orient you within the crash course.
</summary>
</details>

## 2. Module 7: Beyond Convexity (Teaser)

This module serves as a high-level pointer to topics beyond the scope of this introductory crash course. The aim is to illustrate the breadth of the online learning field, not to provide detailed explanations.

### Online Learning with Bandit Feedback

In our discussions so far (full-information setting), the learner observes the entire loss function $$\ell_t(\cdot)$$ or at least its gradient $$\nabla \ell_t(x_t)$$ after playing $$x_t$$. A more challenging scenario arises when the feedback is much more limited.

**Bandit Feedback (or Partial Information Setting):**
*   After playing $$x_t$$, the learner only observes the loss value $$\ell_t(x_t)$$.
*   The learner does not directly see the entire loss function or its gradient.
*   This setting is named after the "multi-armed bandit" problem, where a gambler must choose which slot machine (arm) to play to maximize reward, only observing the outcome of the chosen arm.

**Key Challenges and Approaches:**
*   **Exploration vs. Exploitation:** The learner must balance exploring new actions to discover their potential (even if currently suboptimal) with exploiting actions known to be good.
*   **Gradient Estimation:** Since gradients are not directly observed, algorithms often need to estimate them, e.g., by perturbing actions or using techniques from zeroth-order optimization.
*   **Types of Bandit Problems:**
    *   **Multi-Armed Bandits (MAB):** Decision set $$\mathcal{X}$$ is finite (e.g., choosing one of $$K$$ arms). Algorithms like UCB (Upper Confidence Bound) and Thompson Sampling are common for stochastic MABs, while Exp3 (Exponential-weights for Exploration and Exploitation) is used for adversarial MABs.
    *   **Contextual Bandits:** The learner also observes a context (features) before choosing an action, and the loss depends on both the action and the context. This blends MAB with supervised learning.
    *   **Linear/Convex Bandits:** The unknown reward/loss function is assumed to be linear or convex in the action $$x_t$$, and $$\mathcal{X}$$ can be continuous.

Regret in bandit settings is typically higher than in the full-information setting due to the cost of exploration. For example, in adversarial linear bandits, regret might be $$O(d\sqrt{T})$$ or $$O(\sqrt{dT \log T})$$, where $$d$$ is the dimension.

### Online Non-Convex Optimization

Modern machine learning, particularly deep learning, often involves optimizing highly non-convex loss functions. Extending online learning guarantees to such settings is an active area of research.

**Challenges with Non-Convexity:**
*   **Multiple Local Minima:** Gradient-based methods may converge to suboptimal local minima or saddle points.
*   **Defining "Optimal":** The notion of a single best fixed action $$x^\ast $$ may be less meaningful if many good local minima exist. The global minimum might be computationally intractable to find.
*   **Regret Benchmarks:** Static regret against the global minimizer can be too strong. Alternative benchmarks include:
    *   Regret against the best local minimum found.
    *   Regret against any point satisfying first-order or second-order stationarity conditions ($$\nabla L(x) \approx 0$$).
    *   "No-regret" guarantees for finding stationary points.

**Approaches:**
*   Algorithms like OGD (or SGD in the stochastic setting) are still widely used. While they may not find the global optimum, they can often find "good enough" solutions in practice for deep learning.
*   Theoretical analysis often focuses on convergence to stationary points or characterizing the landscape of non-convex functions that allow for efficient optimization (e.g., Polyak-Łojasiewicz condition, gradient dominance).
*   Some FTRL and OMD variants can be analyzed for non-convex losses, though typically yielding weaker guarantees (e.g., bounds on the expected squared gradient norm).

### Connections to Reinforcement Learning (RL)

Online learning principles are foundational to many areas of Reinforcement Learning.
*   **Sequential Decisions:** RL agents make sequences of actions in an environment.
*   **Delayed Feedback:** Rewards/losses might be delayed and sparse.
*   **State Dependence:** The environment's response (and future states) depends on the agent's actions. This introduces complexities beyond the standard online learning protocol where loss functions $$\ell_t$$ might not depend on past actions $$x_1, \dots, x_{t-1}$$ in such a structured way (though adversarial $$\ell_t$$ can).
*   Policy gradient methods, Q-learning, and actor-critic algorithms in RL often employ iterative updates and exploration strategies reminiscent of online learning and bandit algorithms.

---

This brief overview is merely a glimpse into the vast landscape beyond Online Convex Optimization with full information. These areas are rich with their own specialized algorithms, theoretical tools, and applications. Understanding the core OCO principles from this crash course provides a solid foundation for venturing into these more advanced topics.

**Next Up:** Module 8: Summary & Practical Guidance

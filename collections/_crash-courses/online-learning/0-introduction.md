---
title: "Online Learning Crash Course – Part 0: Setting & Motivation"
date: 2025-06-01 09:00 -0400
sort_index: 0
mermaid: true
description: An introduction to the online learning paradigm, its core principles, motivations, and the sequential decision-making framework.
image: # placeholder
categories:
- Machine Learning
- Online Learning
tags:
- Online Learning
- Sequential Decision Making
- Optimization
- Prerequisite
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

Welcome to this crash course on Online Learning! This series provides foundational knowledge for understanding sequential decision-making algorithms, crucial for many areas of modern machine learning and optimization.

## 1. Road-map at a Glance

This crash course is structured into several modules. We are currently at **Module 0: Setting & Motivation**.

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

    style C0 fill:#E0BBE4,color:#000,stroke:#333,stroke-width:2px % Highlight current module
```

<details class="details-block" markdown="1">
<summary markdown="1">
**Navigational Tip.** This flowchart will appear in each post, with the current module highlighted, to orient you within the crash course.
</summary>
</details>

## 2. Module 0: Setting & Motivation

### What is Online Learning?

At its core, online learning is about making a sequence of predictions or decisions over time. Data arrives sequentially, instance by instance, and the learning algorithm must make a choice (e.g., a prediction, an action, or setting model parameters) for the current instance before (or as) the next one is revealed. After each choice, feedback is typically received, which is then used to update the algorithm's strategy for future interactions. This contrasts with batch learning, which processes an entire dataset at once.

### Why Online Learning?

The motivations for studying and employing online learning are multifaceted:

1.  **Streaming Data:** Many real-world applications involve data that is continuously generated, such as sensor networks, financial tickers, or user interactions on websites. Online algorithms are naturally suited for these environments.
2.  **Resource Constraints:** For massive datasets, batch processing can be infeasible due to memory limitations or computational cost. Online methods offer a way to learn from such data with fixed, often small, resource footprints.
3.  **Adaptive Systems:** When the environment or data distribution is non-stationary (i.e., changes over time), online algorithms can adapt their models more readily than batch methods, which would require frequent retraining on new, complete datasets.
4.  **Theoretical Underpinning:** Online learning provides a powerful theoretical framework for analyzing iterative algorithms, including those used in stochastic optimization (like Stochastic Gradient Descent, SGD). Concepts like regret minimization offer insights into algorithm performance and generalization.
5.  **Game-Theoretic Perspective:** Online learning can be framed as a repeated game between a **learner** and an **environment** (or adversary). The learner makes sequential decisions, and the environment reveals information (e.g., costs or losses) associated with those decisions. This perspective is particularly useful for designing robust algorithms.

### Illustrative Examples

The online learning framework finds application in diverse areas:

*   **Spam Filtering:** Classifying emails as spam or not-spam as they arrive, updating the filter based on user feedback or new spam characteristics.
*   **Recommendation Systems:** Suggesting items (e.g., products, news articles) to users in real-time, learning from their immediate responses (clicks, purchases).
*   **Real-time Ad Prediction:** Deciding which ad to show a user based on their current context and historical data, optimizing for click-through rates or conversions.
*   **Dynamic Resource Allocation:** Adjusting resource provisioning in cloud computing or network routing based on evolving demand patterns.

### The Online Learning Protocol

The interaction between the learner and the environment is formalized by the online learning protocol.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Online Learning Protocol
</div>
The learning process unfolds over a sequence of rounds $$t = 1, 2, \dots, T$$. In each round $$t$$:

1.  The **learner** chooses an action (or prediction, model parameters, etc.) $$x_t$$ from a pre-defined **decision set** $$\mathcal{X}$$.
2.  Concurrently or subsequently, the **environment** (or nature, or an adversary) reveals a **loss function** $$\ell_t : \mathcal{X} \to \mathbb{R}$$. This function quantifies the penalty associated with any possible action the learner could have taken.
3.  The learner incurs the loss $$\ell_t(x_t)$$ for the chosen action $$x_t$$.
4.  The learner observes information about $$\ell_t$$ (e.g., the full function, its value $$\ell_t(x_t)$$, or its gradient $$\nabla \ell_t(x_t)$$) and uses this to update its strategy for subsequent rounds.

The cumulative loss incurred by the learner up to round $$T$$ is $$\sum_{t=1}^T \ell_t(x_t)$$.
</blockquote>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**A Note on Assumptions: Online Convex Optimization (OCO)**
</div>
A significant portion of online learning theory, and much of this crash course, focuses on **Online Convex Optimization (OCO)**. In the OCO setting:
*   The decision set $$\mathcal{X}$$ is a **convex subset** of $$\mathbb{R}^d$$.
*   Each loss function $$\ell_t(\cdot)$$ is **convex** over $$\mathcal{X}$$.

These assumptions enable powerful analytical tools and guarantees, many of which have direct implications for understanding widely used optimization algorithms in machine learning. We will explore these in detail starting from Module 2.
</blockquote>

The primary goal in online learning is typically to design a sequence of actions $$\{x_t\}_{t=1}^T$$ such that the cumulative loss is minimized, often relative to some benchmark. This leads directly to the concept of **regret**, which we will define and explore in the next module.

---

This concludes our introduction to the setting and motivation of online learning. Next, we will delve into how we measure performance in this paradigm.

**Next Up:** Module 1: Regret & Benchmarks

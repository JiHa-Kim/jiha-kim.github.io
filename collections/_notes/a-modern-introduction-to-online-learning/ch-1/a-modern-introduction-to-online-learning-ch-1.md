---
title: A Modern Introduction to Online Learning - Ch 1
date: 2025-04-29 02:08 +0000 # Or use a dynamic date if preferred: {{ site.time | date_to_xmlschema }}
math: true
categories:
- Machine Learning
- Online Learning
tags: 
- Francesco Orabona
- Online Learning
- Batch Learning
- Incremental Learning
llm-instructions: |-
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

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
  4. item

  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

These are notes for the text A Modern Introduction to Online Learning by Francesco Orabona on [arXiv](https://arxiv.org/abs/1912.13213).

## Chapter 1 - What is Online Learning?

The core idea of Online Learning in this context is presented through a repeated game framework where the goal is to minimize **Regret**.

**The Basic Game (Example):**

1.  For rounds $$t = 1, \dots, T$$:
    *   An adversary chooses a secret number $$y_t \in [0, 1]$$.
    *   You choose a number $$x_t \in [0, 1]$$.
    *   The adversary reveals $$y_t$$, and you pay the squared loss $$(x_t - y_t)^2$$.

**Goal: Minimize Regret**

Initially, if we assume $$y_t$$ are i.i.d from some distribution with variance $$\sigma^2$$, the best strategy is to always predict the mean, incurring an expected loss of $$\sigma^2 T$$. A natural goal is to minimize the *excess loss* compared to this optimal (but unknown) strategy.

Removing the stochastic assumption, we consider an *arbitrary* sequence $$y_t$$. The performance metric becomes the **Regret**: how much worse our total loss is compared to the best *single* fixed action chosen in hindsight.

<blockquote class="prompt-info" markdown="1">
### Definition - Regret (for the example game)

The regret after $$T$$ rounds is defined as:

$$
\text{Regret}_T := \sum_{t=1}^T (x_t - y_t)^2 - \min_{x \in [0, 1]} \sum_{t=1}^T (x - y_t)^2
$$

An algorithm is considered successful ("wins the game") if $$\text{Regret}_T$$ grows sublinearly in $$T$$ (i.e., $$\lim_{T\to\infty} \frac{\text{Regret}_T}{T} = 0$$).
</blockquote>

**General Online Learning Framework:**

*   At each round $$t$$, the algorithm chooses an action $$x_t$$ from a feasible set $$V \subseteq \mathbb{R}^d$$.
*   It incurs a loss $$\ell_t(x_t)$$, where the loss function $$\ell_t$$ can be chosen arbitrarily (adversarially) at each round.
*   The goal is to minimize the regret compared to the best *fixed* competitor $$u \in V$$ in hindsight.
    *   Basically, we want to capture the difficulty of the problem, so we want a comparison, and a fixed competitor is the simplest way to do that. Only our algorithm will be adapting over time to keep analysis simple.

<blockquote class="prompt-info" markdown="1">
### Definition - Regret (General)

For a sequence of loss functions $$\ell_1, \dots, \ell_T$$ and algorithm predictions $$x_1, \dots, x_T$$, the regret with respect to a competitor $$u \in V$$ is:

$$
\text{Regret}_T(u) := \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(u)
$$

The online algorithm does *not* know $$u$$ or the future losses when making its prediction $$x_t$$.
</blockquote>

**Adversarial Nature:** The sequence of losses $$\ell_t$$ can be chosen by an adversary, potentially based on the algorithm's past actions. This makes standard statistical assumptions (like i.i.d data) invalid for the analysis.

**A Strategy: Follow-the-Leader (FTL)**

A natural strategy is to choose the action at time $$t$$ that would have been optimal for the *past* rounds $$1, \dots, t-1$$. This uses all information available (from past rounds).  

*   In the example game: The best action in hindsight after $$T$$ rounds is $$x^*_T = \frac{1}{T} \sum_{t=1}^T y_t$$.
*   FTL Strategy: $$x_t = x^*_{t-1} = \frac{1}{t-1} \sum_{i=1}^{t-1} y_i$$ (for $$t > 1$$).

**Analysis of FTL for the Guessing Game:**

*   **Lemma 1.2 (Hannan's Lemma):** Let $$x^*_t = \arg\min_{x \in V} \sum_{i=1}^t \ell_i(x)$$. Then for any sequence of loss functions $$\ell_t$$:

    $$
    \sum_{t=1}^T \ell_t(x^*_t) \le \sum_{t=1}^T \ell_t(x^*_T)
    $$

    (Playing adaptively based on past data is no worse than playing the single best action in hindsight).
*   **Theorem 1.3:** For the number guessing game ($$\ell_t(x)=(x-y_t)^2, y_t \in [0,1]$$), the FTL strategy ($$x_t = x^*_{t-1}$$) achieves:

    $$
    \text{Regret}_T \le 4 + 4 \ln T
    $$

    This is sublinear in $$T$$, so FTL "wins" this specific game.
*   **Proof Idea:** Use Lemma 1.2 to bound the regret against the *sequence* $$x^*_1, \dots, x^*_T$$. Then bound the difference between consecutive optimal actions $$|x^*_{t-1} - x^*_t|$$. Show this difference decreases quickly enough (like $$O(1/t)$$), and the sum is bounded by $$O(\ln T)$$.

**Key Takeaways from FTL Example:**

*   The FTL strategy for this specific game is **parameter-free** (no learning rates etc. to tune).
*   It is computationally efficient (only requires maintaining a running average).
*   It **does not use gradients**.
*   Online learning is distinct from statistical learning; concepts like overfitting don't directly apply.

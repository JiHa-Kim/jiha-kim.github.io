---
layout: note
title: A Modern Introduction to Online Learning - Ch 2
date: 2025-04-29 02:07 +0000
math: true
categories:
- Notes
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by a newlines between the text above and below, and newlines between the delimiters and the equation (even in lists):

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

These are notes for the text A Modern Introduction to Online Learning by Francesco Orabona on [ArXiV](https://arxiv.org/abs/1912.13213).

### Chapter 2 - Online Subgradient Descent

The main contribution of this chapter is presenting the **online subgradient descent algorithm** (which is not a pure descent method!), which generalizes the gradient descent method to non-differentiable convex losses.

The general online learning problem:


<blockquote class="prompt-info" markdown="1">
### Definition - Online Learning Game
- For $$t=1,\dots,T \in \mathbb{N}$$
  - Output $$x_t \in \mathcal{V} \subseteq \mathbb{R}^d$$
  - Pay $$\ell_t(x_t)$$ for $$\ell_t: \mathcal{V} \to \mathbb{R}$$
  - Receive some feedback on $$\ell_t$$
- End for

Goal: minimize regret.

### Definition - Regret (General)

For a sequence of loss functions $$\ell_1, \dots, \ell_T$$ and algorithm predictions $$x_1, \dots, x_T$$, the regret with respect to a competitor $$u \in V$$ is:

$$
\text{Regret}_T(u) := \sum_{t=1}^T \ell_t(x_t) - \sum_{t=1}^T \ell_t(u)
$$

The online algorithm does *not* know $$u$$ or the future losses when making its prediction $$x_t$$.

*   Why fixed competitor $$u \in \mathcal{V}$$? Basically, we want to capture the difficulty of the problem, so we want a comparison, and a fixed competitor is the simplest way to do that. Only our algorithm will be adapting over time to keep analysis simple.
</blockquote>


Problem: this formulation is too general and we cannot make any progress. We must choose *reasonable* assumptions.

Plausible:
- Convex loss
- Lipschitz loss

Implausible:
- Knowing future

<blockquote class="prompt-info" markdown="1">
### Definition 2.2 - Convex Set

A set $$\mathcal{V} \subseteq \mathbb{R}^d is **convex** iff it contains the line segment between any two points in the set.

Intuitively, the shape doesn't bend inward to create a hollow area.

$$
0 < \lambda < 1 \text{ and } x,y \in \mathcal{V} \implies
\\
y + \lambda (x-y) = \lambda x + (1-\lambda)y \in \mathcal{V}. 
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Extended Real Number Line

$$
[-\infty,+\infty]
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Domain of a Function

$$
\text{dom } f := \{ x \in \mathbb{R}^d : f(x) < +\infty \}
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Indicator Function of a Set

$$
i_\mathcal{V}(x) := \begin{cases}
0 \quad &x \in \mathcal{V}
\\
+\infty &\text{otherwise}
\end{cases}
$$

</blockquote>

<blockquote class="prompt-tip" markdown="1">
### Tip - Constrained to Unconstrained Formulation

The indicator function and extended real number line allow us to convert a constrained optimization problem formulation into an unconstrained one.

$$
\mathop{\arg \min}\limits_{x \in \mathcal{V}} f(x) = \mathop{\arg \min}\limits_{x\in \mathbb{R}^d} f(x) + i_\mathcal{V} (x)
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Epigraph of a Function

The epigraph of $$f: \mathcal{V} \subseteq \mathbb{R}^d \to [-\infty,+\infty]$$ is the set of all points above its graph:

$$
\text{epi }f = \{(x,y)\in \mathcal{V} \times \mathbb{R} : y \ge f(x)}
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Convex Function

$$f: \mathcal{V} \subseteq \mathbb{R}^d \to [-\infty,+\infty]$$ is convex iff its epigraph $$\text{epi }f$$ is convex.

Implicitly, its domain must therefore be convex (we can separate the convexity definition by coordinates, so $$\mathcal{V}$$ must be convex).
</blockquote>

<blockquote class="prompt-tip" markdown="1">
If $$f: \mathcal{V} \subseteq \mathbb{R}^d \to [-\infty,+\infty]$$ is convex, then

$$
f + i_\mathcal{V}: \mathbb{R}^d \to [-\infty,+\infty]
$$

is also convex. Moreover, $$i_\mathcal{V}$$ is convex iff $$\mathcal{V}$$ is convex, so each convex set is associated with a convex function.
</blockquote>



![Figure 2.2: Epigraph of convex (left) and non-convex functions (right)](./image.png)
_Figure 2.2: Epigraph of convex (left) and non-convex functions (right)_

<blockquote class="prompt-info" markdown="1">

</blockquote>

\[test\]

<blockquote class="prompt-definition" markdown="1">
### Definition - Test

</blockquote>


<blockquote class="box-definition" markdown="1">
### Definition - Test

</blockquote>


<blockquote class="box-definition" markdown="1">
<div class="title">Definition - Test</div>
### Test
Test

</blockquote>

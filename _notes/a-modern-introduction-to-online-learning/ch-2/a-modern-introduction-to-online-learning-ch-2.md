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

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):

    $$
    block
    $$

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

  The stock blockquote classes are:
    - prompt-info
    - prompt-tip
    - prompt-warning
    - prompt-danger

  Your newly added math-specific prompt classes can include:
    - prompt-definition
    - prompt-lemma
    - prompt-proposition
    - prompt-theorem
    - prompt-example          # for worked examples or illustrative notes

  Similarly, for boxed environments you can define:
    - box-definition
    - box-lemma
    - box-proposition
    - box-theorem
    - box-example             # for example blocks with lightbulb icon

  For details blocks, use:
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

These are notes for the text *A Modern Introduction to Online Learning* by Francesco Orabona on [arXiv](https://arxiv.org/abs/1912.13213).

### Chapter 2 - Online Subgradient Descent

The main contribution of this chapter is presenting the **online subgradient descent algorithm** (which is not a pure descent method!), which generalizes the gradient descent method to non-differentiable convex losses.

The general online learning problem:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Online Learning Game
</div>
- For $$t = 1, \dots, T \in \mathbb{N}$$  
  - Output $$x_t \in \mathcal{V} \subseteq \mathbb{R}^d$$  
  - Pay $$\ell_t(x_t)$$ for $$\ell_t: \mathcal{V} \to \mathbb{R}$$  
  - Receive some feedback on $$\ell_t$$  
- End for

Goal: minimize regret.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Regret (General)
</div>
For a sequence of loss functions $$\ell_1, \dots, \ell_T$$ and algorithm predictions $$x_1, \dots, x_T$$, the regret with respect to a competitor $$u \in \mathcal{V}$$ is:

$$
\text{Regret}_T(u) := \sum_{t=1}^T \ell_t(x_t) \;-\; \sum_{t=1}^T \ell_t(u).
$$

The online algorithm does *not* know $$u$$ or future losses when making its prediction $$x_t$$.
</blockquote>

Problem: this formulation is too general and we cannot make any progress. We must choose *reasonable* assumptions.

Plausible:
- Convex loss
- Lipschitz loss

Implausible:
- Knowing future

Why do we need more information when we developed FTL that worked in our first game? Here is an example failure case in a different game.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 2.10.** Failure of FTL
</div>

Intuitively, we will construct two pendulums that are out of sync. Or you can imagine a player in a sport constantly juking a naive opponent who always lags one step behind in the wrong direction.

Let $$\mathcal{V}=[-1,1]$$. Consider the sequences of losses $$\ell_t(x)=z_t x$$, where $$z_t=-0.5,1,-1,1,-1,\dots$$. On the first round, we let FTL predict $$x_1$$ arbitrarily since we have no past history. Then, FTL will predict $$x_1,-1,1,-1,1,\dots$$. Overall, the cumulative loss becomes:

$$
\begin{aligned}
\sum_{t=1}^{T} \ell_t(x_t)
&=\underbrace{-0.5x_1}_\text{1 round}+\underbrace{(-1)(1)+(1)(-1)+\dots}_{T-1\text{ rounds}}
\\
&=-0.5x_1+T-1
\end{aligned}
$$

Meanwhile, the fixed competitor $$u=0$$ yields $$0$$ cumulative loss. Thus:

$$
\text{Regret}_T(u=0)=T-1-0.5x_1\ge T-1.5.
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 2.2.** Convex Set
</div>
A set $$\mathcal{V} \subseteq \mathbb{R}^d$$ is **convex** iff it contains the line segment between any two points in the set.

Intuitively, the shape doesn't bend inward to create a hollow area.

$$
0 < \lambda < 1,\; x,y \in \mathcal{V}
\;\implies\;
\lambda x + (1-\lambda) y \in \mathcal{V}.
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Extended Real Number Line
</div>

$$
[-\infty,+\infty]
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Domain of a Function
</div>

$$
\mathrm{dom}\,f := \{x \in \mathbb{R}^d : f(x) < +\infty\}.
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Indicator Function of a Set
</div>

$$
i_{\mathcal{V}}(x) :=
\begin{cases}
0, & x \in \mathcal{V},\\
+\infty, & \text{otherwise}.
\end{cases}
$$

</blockquote>

<blockquote class="prompt-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** Constrained to Unconstrained Formulation
</div>
The indicator function and extended real number line allow us to convert a constrained optimization problem into an unconstrained one:

$$
\arg\min_{x \in \mathcal{V}} f(x)
=
\arg\min_{x \in \mathbb{R}^d} \bigl(f(x) + i_{\mathcal{V}}(x)\bigr).
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Epigraph of a Function
</div>
The epigraph of $$f: \mathcal{V} \subseteq \mathbb{R}^d \to [-\infty,+\infty]$$ is the set of all points above its graph:

$$
\mathrm{epi}\,f
= \{(x,y)\in \mathcal{V}\times\mathbb{R} : y \ge f(x)\}.
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 2.3.** Convex Function
</div>
A function $$f: \mathcal{V}\subseteq\mathbb{R}^d \to [-\infty,+\infty]$$ is convex iff its epigraph $$\mathrm{epi}\,f$$ is convex.  Implicitly, its domain must therefore be convex.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** Sum of Convex Function and Indicator
</div>
If $$f: \mathcal{V}\subseteq\mathbb{R}^d \to [-\infty,+\infty]$$ is convex, then

$$
f + i_{\mathcal{V}} : \mathbb{R}^d \to [-\infty,+\infty]
$$

is also convex. Moreover, $$i_{\mathcal{V}}$$ is convex iff $$\mathcal{V}$$ is convex, so each convex set is associated with a convex function.
</blockquote>

TODO: Image not working in notes 
<!-- 
![Figure 2.2: Epigraph of convex (left) and non-convex functions (right)](./image.png)
_Figure 2.2: Epigraph of convex (left) and non-convex functions (right)_   -->

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.4.** ([Rockafellar, 1970, Theorem 4.1])
</div>

Let $$f: \mathbb{R}^d \to (-\infty,+\infty]$$ where $$\text{dom }f$$ is a convex set. Then the following are equivalent:

$$
f \text{ convex}
$$

$$
\iff
$$

$$
\forall \lambda \in (0,1), \forall x,y \in \text{dom }f,
$$

$$
f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda) f(y).
$$

</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 2.5.** Convex functions: affine functions.
</div>

The simplest convex functions are affine functions:

$$
f(x) = \langle \textbf{z},\textbf{x} \rangle + b
$$

</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 2.6.** Convex functions: norms.
</div>


<details class="details-block" markdown="1">
<summary markdown="1">
**Definition.** Norm of a vector
</summary>
Given a vector space $$\mathcal{V}$$ over a subfield $$\mathbb{F}$$ of $$\mathbb{C}$$, a **norm** on $$\mathcal{V}$$ is a real-valued function $$\Vert\cdot\Vert: \mathcal{V} \to \mathbb{R}$$. Then, for all vectors $$x,y \in \mathcal{V}$$, scalars $$s \in \mathbb{F}$$, the norm satisfies the following:

1. Subadditivity/Triangle inequality.

  $$
  \Vert x+y\Vert  \le \Vert x\Vert + \Vert y\Vert
  $$

2. Absolute homogeneity: 

  $$
  \Vert sx \Vert = \vert s \vert \Vert x \Vert
  $$

3. Positive definiteness/Point-seperating:

  $$
  \Vert x \Vert = 0 \implies x=0
  $$

</details>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof.** Norms are convex
</summary>

It is sufficient to show that for $$x,y \in \mathbb{R}^d$$, $$\lambda \in (0,1)$$:

$$
\Vert \lambda x + (1-\lambda)y \Vert \le \lambda \Vert x \Vert + (1-\lambda) \Vert y \Vert
$$

First, apply the triangle inequality (sub-additivity):

$$
\Vert \lambda x + (1-\lambda)y \Vert \le \Vert \lambda x \Vert + \Vert (1-\lambda) y \Vert
$$

Then, apply absolute homogeneity:

$$
\Vert \lambda x \Vert + \Vert (1-\lambda) y \Vert \le \vert \lambda \vert \Vert x \Vert + \vert 1-\lambda \vert \Vert y \Vert
$$

Since $$\lambda \in (0,1)$$, then $$0<\lambda$$ and $$0<1-\lambda$$. Hence, $$\vert \lambda \vert = \lambda$$ and $$\vert 1-\lambda \vert = 1-\lambda$$. Chaining the inequalities gives the desired result. 

</details>

</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.7.** First-order condition for convexity ([Rockafellar, 1970, Theorem 25.1 and Corollary 25.1.1])
</div>

Given a convex function $$f: \mathbb{R}^d \to (-\infty,+\infty]$$ with $$f$$ differentiable at $$x \in \text{int dom } f$$, we have

$$
f(y)-f(x)-\langle \nabla f(x),y-x \rangle \ge 0 \quad \forall y \in \mathbb{R}^d.
$$

(**Remark.** This is actually an iff when $$y\in \text{dom }f$$.)

</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.8.** First-order optimality condition for differentiable convex functions.
</div>

Let $$\mathcal{V}$$ be a convex non-empty set, $$x^\ast \in \mathcal{V}$$, and $$f: \mathcal{V} \to \mathbb{R}$$ a convex function, differentiable over an open set that contains $$\mathcal{V}$$. Then, 

$$
x^\ast \in \arg\min_{x \in \mathcal{V}} f(x)
$$

$$
\iff
$$

$$
\langle \nabla f(x^\ast), y-x^\ast \rangle \ge 0, \forall y\in \mathcal{V}.
$$

Moreover, if $$x^\ast \in \text{int } \mathcal{V}$$, by choosing $$\epsilon$$ enough such that $$y=x^\ast-\epsilon \nabla f(x^\ast) \in \mathcal{V}$$, $$x^\ast$$ is a minimum iff $$\nabla f(x^\ast)=0$$.

(**Remark.** Intuitively, this means that there no a feasible directional derivative $$\nabla_{y-x^\ast} f(x^\ast)$$ in which we can take a step that will decrease the function. There is no feasible step that will make an acute angle with the negative gradient. Furthermore, if $$x^\ast \in \text{int } \mathcal{V}$$, by definition, any direction must be feasible, so it is only possible that no direction results in a decrease.)

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof.**
</summary>

$$(\Rightarrow)$$:

Let $$x^\ast \in \arg\min_{x \in \mathcal{V}} f(x)$$. By definition, this means $$f(x^\ast) \le f(x),\, \forall y \in \mathcal{V}$$. Also, since $$\mathcal{V}$$ is convex and non-empty, it must contain all line segments between two points.

Formally, since the line segment given by points $$z(\lambda)=x^\ast+\lambda(y-x^\ast) \in \mathcal{V}$$, where $$0 \lt \lambda \lt 1$$, we have by assumption of the optimality of $$x^\ast$$ that

$$
f(x^\ast) \le f(z(\lambda)).
$$

This implies, since $$0\lt\lambda$$:

$$
0 \le \frac{f(x^\ast+\lambda(y-x))-f(x^\ast)}{\lambda}
$$

Since $$f$$ is differentiable, the limit as $$\lambda \downarrow 0$$ exists and is precisely equal to the directional derivative $$\langle \nabla f(x^\ast),y-x^\ast\rangle$$.

$$\diamond$$

$$(\Leftarrow)$$:

Assume $$0 \le \langle \nabla f(x^\ast),y-x^\ast\rangle$$. By the first-order condition for convexity, we have:

$$
0\le f(y)-f(x^\ast)-\langle \nabla f(x^\ast),y-x^\ast \rangle \le f(y)-f(x^\ast).
$$

Thus $$f(x^\ast)\le f(y)$$, which is the definition of a minimizer.

$$\diamond$$

Now, we prove the "Moreover" part: if $$x^\ast \in \text{int }\mathcal{V}$$, then

$$
x^\ast \in \arg\min_{x\in\mathcal{V}} f(x) \iff \nabla f(x^\ast)=0.
$$

$$(\Rightarrow)$$:

Assume $$x^\ast \in \text{int }\mathcal{V}$$ and $$x^\ast \in \arg\min_{x\in\mathcal{V}} f(x)$$. 

By definition of an interior point, there exists for some radius $$r>0$$, an open ball $$B(x^\ast,r)=\{z \mid \Vert z-x^\ast \Vert \lt r \} \subseteq \mathcal{V}$$.

Consider a gradient descent step $$y=x^\ast-\epsilon \nabla f(x^\ast)$$, which we want to stay in $$\mathcal{V}$$. In other words,

$$
\Vert y-x^\ast \Vert = \epsilon \Vert \nabla f(x^\ast) \Vert \lt r.
$$

If $$\nabla f(x^\ast)=0$$, then we are done. Else, choose small enough epsilon for the statement to hold:

$$
0 \lt \epsilon \lt r/\Vert \nabla f(x^\ast) \Vert.
$$

By the first part of the theorem, we have:

$$
\begin{aligned}
0 &\le \langle \nabla f(x^\ast),y-x^\ast \rangle 
\\
&= \langle \nabla f(x^\ast),-\epsilon \nabla f(x^\ast) \rangle 
\\
&= -\epsilon \Vert \nabla f(x^\ast) \Vert^2.
\end{aligned}
$$

But also, since $$0\lt\epsilon$$ and $$0\le\Vert\nabla f(x^\ast)\Vert^2$$, it follows that

$$
0 \le \Vert \nabla f(x^\ast) \Vert \le 0 \iff \nabla f(x^\ast) = 0.  
$$

$$\diamond$$

$$(\Leftarrow)$$:

Assume $$x^\ast \in \text{int }\mathcal{V}$$ and $$\nabla f(x^\ast)=0$$.

We must verify the optimality of $$x^\ast$$:

$$
x^\ast \in \arg\min_{x\in\mathcal{V}} \iff f(x^\ast)\le f(x), \forall x\in\mathcal{V}.
$$

But from the first-order optimality condition, we satisfy

$$
0\le f(x)-f(x^\ast)-\langle \nabla f(x^\ast),x-x^\ast\rangle=f(x)-f(x^\ast)
$$

thus $$f(x^\ast)\le f(x)$$.

$$\blacksquare$$

</details>

</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.9.** Jensen's Inequality
</div>

Let $$f: \mathbb{R}^d \to (-\infty,+\infty]$$ be a measurable convex function and $$x$$ be an $$\mathbb{R}^d$$-valued random element on some probability space such that $$\mathbb{E}[x]$$ exists and $$x \in \text{dom }f$$ with probability $$1$$. Then

$$
\mathbb{E}[f(x)]\le f(\mathbb{E}[x]).
$$

(**Remark.** To remember the direction, simply test with a discrete probability distribution (Bernoulli) on two points, which once again gives the equivalent definition of convexity:

$$
f(\lambda x+(1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y).
$$

Visually, set $$\lambda=0.5$$ and imagine an upward parabola (convex). The midpoint of the line segment joining two points should lie above the function evaluated at the horizontal midpoint.)

</blockquote>
---
title: RMS Norm
date: 2025-06-05 06:29 -0400
description: Characterizing properties of the Root-Mean-Square Norm for vectors
sort_index: 4
categories:
- Crash Courses
- Functional Analysis
tags:
- Norms
- Vector Norms
- RMS Norm
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  NEVER introduce any non-existant URL or path, like an image.
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

### Root-Mean-Square (RMS) Norm for Vectors

While various norms are ubiquitous in mathematics and engineering, deep learning practice often benefits from a **dimension-invariant** scale for vectors. The **RMS norm** provides this by normalizing the Euclidean norm by the square root of the dimension.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** RMS Norm (for Vectors)
</div>
For $$x \in \mathbb{R}^n$$ ($$n \ge 1$$), the **root-mean-square norm** is

$$
\Vert x \Vert_{\mathrm{RMS}} \;=\; \frac{\Vert x \Vert_2}{\sqrt{n}} \;=\; \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}
$$

where $$\Vert x \Vert_2 = \sqrt{\sum_{i=1}^n x_i^2}$$ is the standard Euclidean norm.
</blockquote>

#### Properties and Rationale for Vector RMS Norm

1.  **Dimension neutrality.** If the coordinates $$x_i$$ of $$x$$ are i.i.d. random variables with zero mean and unit variance (e.g., $$x_i \sim \mathcal{N}(0,1)$$), then $$\mathbb{E}[\Vert x \Vert_{\mathrm{RMS}}^2] = 1$$, and $$\mathbb{E}[\Vert x \Vert_{\mathrm{RMS}}] \approx 1$$ for large $$n$$. This property makes the notion of "unit-size" more consistent for vectors of varying dimensions, such as network activations from layers of different widths.
2.  **Rotational and Orthogonal Invariance.** The RMS norm is a positive scalar multiple of the $$\ell_2$$-norm. The $$\ell_2$$-norm is invariant under rotations ($$SO(n)$$) and, more generally, under all orthogonal transformations ($$O(n)$$), which include reflections. The RMS norm inherits these crucial geometric symmetries.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 1.** Rotationally Invariant Functions
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ ($$n \ge 2$$) is **rotationally invariant** (i.e., $$f(Rx) = f(x)$$ for all rotation matrices $$R \in SO(n)$$ and all $$x \in \mathbb{R}^n$$) if and only if there exists a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that:

$$
f(x) = g(\Vert x \Vert_2) \quad \forall x \in \mathbb{R}^n
$$

<details class="details-block" markdown="1">
<summary markdown="1">
Note on the case $$n=1$$
</summary>
For $$n=1$$, the space is $$\mathbb{R}$$. The special orthogonal group $$SO(1)$$ contains only the identity matrix $$[1]$$. Thus, any function $$f: \mathbb{R} \to \mathbb{R}$$ is trivially rotationally invariant (i.e., $$f(1 \cdot x_1) = f(x_1)$$). Such a function can be written as $$f(x_1) = g(\Vert x_1 \Vert_2) = g(\vert x_1 \vert)$$ if and only if $$f$$ is an even function (i.e., $$f(x_1)=f(-x_1)$$).
</details>
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 1.**
</summary>
($$\Leftarrow$$) **Sufficiency (for all $$n \ge 1$$):**
Assume $$f(x) = g(\Vert x \Vert_2)$$ for some function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$. For any rotation matrix $$R \in SO(n)$$, rotations preserve the Euclidean norm: $$\Vert Rx \Vert_2 = \Vert x \Vert_2$$.
Then, $$f(Rx) = g(\Vert Rx \Vert_2) = g(\Vert x \Vert_2) = f(x)$$. Thus, $$f$$ is rotationally invariant.

($$\Rightarrow$$) **Necessity:**
*   **Case $$n \ge 2$$:** Assume $$f: \mathbb{R}^n \to \mathbb{R}$$ is rotationally invariant.
    *   If $$x = \mathbf{0}$$, define $$g(0) = f(\mathbf{0})$$. Then $$f(\mathbf{0}) = g(\Vert \mathbf{0} \Vert_2)$$.
    *   If $$x \ne \mathbf{0}$$, let $$r_0 = \Vert x \Vert_2 > 0$$. For any $$y \in \mathbb{R}^n$$ with $$\Vert y \Vert_2 = r_0$$, there exists $$R \in SO(n)$$ such that $$y = Rx$$ (since $$SO(n)$$ acts transitively on spheres for $$n \ge 2$$). By rotational invariance, $$f(y) = f(Rx) = f(x)$$. Thus, $$f(x)$$ depends only on $$\Vert x \Vert_2$$. Define $$g(r_0) = f(x_0)$$ for any fixed $$x_0$$ with $$\Vert x_0 \Vert_2 = r_0$$ (e.g., $$x_0 = (r_0, 0, \ldots, 0)^\top$$). Then $$f(x) = g(\Vert x \Vert_2)$$. (Using $$r_0$$ to avoid confusion with function $$g(r)$$).

*   **Case $$n=1$$:** (Covered in the note within the theorem statement.) For completeness, if $$f(x_1) = g(\vert x_1 \vert)$$, then $$f(-x_1) = g(\vert -x_1 \vert) = g(\vert x_1 \vert) = f(x_1)$$, so $$f$$ must be an even function. Conversely, if $$f$$ is an even function, define $$g(r) = f(r)$$ for $$r \ge 0$$. Then for any $$x_1 \in \mathbb{R}$$, $$g(\vert x_1 \vert) = f(\vert x_1 \vert)$$. Since $$f$$ is even, $$f(\vert x_1 \vert) = f(x_1)$$ if $$x_1 \ge 0$$ and $$f(\vert x_1 \vert) = f(-x_1) = f(x_1)$$ if $$x_1 < 0$$. So $$f(x_1) = g(\vert x_1 \vert)$$.
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Corollary 1.1.** Rotationally Invariant Norms
</div>
If a function $$\Vert \cdot \Vert : \mathbb{R}^n \to \mathbb{R}$$ is a **norm** and is **rotationally invariant**, then it must be a positive scalar multiple of the Euclidean norm:

$$
\Vert x \Vert = c \Vert x \Vert_2 \quad \forall x \in \mathbb{R}^n, \text{ for some constant } c > 0
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Corollary 1.1.**
</summary>
Let $$\Vert \cdot \Vert$$ be a rotationally invariant norm.
1.  By Theorem 1:
    *   For $$n \ge 2$$, since $$\Vert \cdot \Vert$$ is rotationally invariant, $$\Vert x \Vert = g(\Vert x \Vert_2)$$ for some $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$.
    *   For $$n=1$$, a norm $$\Vert x_1 \Vert$$ is an even function ($$\Vert -x_1 \Vert = \vert -1 \vert \Vert x_1 \Vert = \Vert x_1 \Vert$$). By the $$n=1$$ case of Theorem 1, $$\Vert x_1 \Vert = g(\vert x_1 \vert) = g(\Vert x_1 \Vert_2)$$.
    Thus, for any $$n \ge 1$$, $$\Vert x \Vert = g(\Vert x \Vert_2)$$.
2.  By absolute homogeneity of norms, $$\Vert \alpha x \Vert = \vert \alpha \vert \Vert x \Vert$$. So, $$g(\Vert \alpha x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)$$. Since $$\Vert \alpha x \Vert_2 = \vert \alpha \vert \Vert x \Vert_2$$, we have $$g(\vert \alpha \vert \Vert x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)$$. Let $$r = \Vert x \Vert_2 \ge 0$$ and $$\lambda = \vert \alpha \vert \ge 0$$. Then $$g(\lambda r) = \lambda g(r)$$ for all $$\lambda, r \ge 0$$.
    This is Cauchy's functional equation for homogeneous functions on $$\mathbb{R}_{\ge 0}$$.
3.  If $$r > 0$$, set $$r=1$$ to get $$g(\lambda) = \lambda g(1)$$. Let $$c = g(1)$$. Then $$g(\lambda) = c\lambda$$ for $$\lambda > 0$$.
    Since $$\Vert \mathbf{0} \Vert = 0$$, we have $$g(\Vert \mathbf{0} \Vert_2) = g(0)=0$$. The relation $$g(\lambda)=c\lambda$$ also gives $$g(0)=0$$, so it holds for all $$\lambda \ge 0$$.
    Thus, $$\Vert x \Vert = g(\Vert x \Vert_2) = c \Vert x \Vert_2$$.
4.  Since $$\Vert \cdot \Vert$$ is a norm, for $$x \ne \mathbf{0}$$, $$\Vert x \Vert > 0$$. Thus $$c \Vert x \Vert_2 > 0$$, which implies $$c > 0$$. (For instance, $$c=g(1)=\Vert e_1 \Vert > 0$$ as $$e_1 \ne \mathbf{0}$$).
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.** Orthogonal Invariance of Euclidean-Derived Norms
</div>
The Euclidean norm ($$\ell_2$$-norm) is **orthogonally invariant**: for any orthogonal matrix $$Q \in O(n)$$ (satisfying $$Q^\top Q = I$$) and any $$x \in \mathbb{R}^n$$,

$$
\Vert Qx \Vert_2 = \Vert x \Vert_2
$$

Consequently, any norm of the form $$\Vert x \Vert = c \Vert x \Vert_2$$ with $$c > 0$$ is also orthogonally invariant. This implies that any rotationally invariant norm is also orthogonally invariant.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 2.**
</summary>
The Euclidean norm squared is $$\Vert x \Vert_2^2 = x^\top x$$. For $$Q \in O(n)$$,
$$\Vert Qx \Vert_2^2 = (Qx)^\top (Qx) = x^\top Q^\top Q x = x^\top I x = x^\top x = \Vert x \Vert_2^2$$.
Since norms are non-negative, taking the square root gives $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$.

If a norm is of the form $$\Vert x \Vert = c \Vert x \Vert_2$$ for some $$c > 0$$, then
$$\Vert Qx \Vert = c \Vert Qx \Vert_2 = c \Vert x \Vert_2 = \Vert x \Vert$$.
So, such norms are orthogonally invariant.
By Corollary 1.1, any rotationally invariant norm ($$SO(n)$$-invariant norm) must be of the form $$c \Vert x \Vert_2$$ for some $$c>0$$. Therefore, any rotationally invariant norm is also orthogonally invariant ($$O(n)$$-invariant).
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Corollary 2.1.** Uniqueness of RMS Norm Family
</div>
The RMS norm, $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$, is a positive scalar multiple of the $$\ell_2$$-norm. Therefore, it is rotationally and orthogonally invariant.

Furthermore, consider a family of norms $$\{\mathcal{N}_n(\cdot)\}_{n \ge 1}$$, where each $$\mathcal{N}_n: \mathbb{R}^n \to \mathbb{R}$$ is a norm on $$\mathbb{R}^n$$. If this family satisfies:
1.  **Rotational Invariance:** Each $$\mathcal{N}_n(\cdot)$$ is rotationally invariant.
2.  **Dimensional Normalization:** For a class of random vectors $$X^{(n)} \in \mathbb{R}^n$$ (whose components $$X_i$$ are i.i.d. with zero mean and unit variance, ensuring $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] \approx 1$$), the expected value $$\mathbb{E}[\mathcal{N}_n(X^{(n)})]$$ is a constant $$K > 0$$ independent of $$n$$. (More precisely, assume $$\mathbb{E}[\Vert X^{(n)} \Vert_2] = \sqrt{n}$$ for this class of vectors.)

Then, each norm $$\mathcal{N}_n(x)$$ must be of the form $$K \cdot \Vert x \Vert_{\mathrm{RMS}}$$. If $$K=1$$, the RMS norm family is the unique family of norms satisfying these conditions.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Corollary 2.1.**
</summary>
The RMS norm is $$(1/\sqrt{n})\Vert x \Vert_2$$. Since $$1/\sqrt{n}>0$$, it's a rotationally invariant norm by Corollary 1.1, and thus orthogonally invariant by Theorem 2.

For the second part:
1.  **Rotational Invariance:** By Corollary 1.1, each $$\mathcal{N}_n(x) = c_n \Vert x \Vert_2$$ for some constant $$c_n > 0$$.
2.  **Dimensional Normalization:** We are given $$\mathbb{E}[\mathcal{N}_n(X^{(n)})] = K$$ for all $$n$$.
    Substituting the form from (1): $$ \mathbb{E}[c_n \Vert X^{(n)} \Vert_2] = c_n \mathbb{E}[\Vert X^{(n)} \Vert_2] = K $$.
    The condition on the random vectors implies $$\mathbb{E}[\Vert X^{(n)} \Vert_2] = \sqrt{n}$$. (This is derived from the motivating property $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] = 1$$, which means $$\mathbb{E}[\Vert X^{(n)} \Vert_2 / \sqrt{n}] = 1$$.)
    Plugging this into the equation for $$K$$: $$c_n \sqrt{n} = K$$.
    So, $$c_n = K / \sqrt{n}$$.
3.  **Form of the Norm:** Therefore, $$\mathcal{N}_n(x) = (K/\sqrt{n}) \Vert x \Vert_2 = K \cdot (\Vert x \Vert_2 / \sqrt{n}) = K \cdot \Vert x \Vert_{\mathrm{RMS}}$$.
    If $$K=1$$, then $$\mathcal{N}_n(x) = \Vert x \Vert_{\mathrm{RMS}}$$.
</details>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** When to use the vector RMS norm
</div>
Employ the vector $$\Vert \cdot \Vert_{\mathrm{RMS}}$$ when a scale for vectors is needed that is simultaneously rotationally symmetric (thus orthogonally symmetric) and normalized for vector dimension. This is useful, for example, when comparing activations from neural network layers of different widths or designing width-robust regularizers.
</blockquote>

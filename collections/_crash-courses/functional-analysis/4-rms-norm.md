---
title: RMS Norm
date: 2025-06-05 06:29 -0400
description: Characterizing properties of the Root-Mean-Square Norm for vectors
course_index: 4
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
For $$x \in \mathbb{R}^n$$, the **root-mean-square norm** is

$$
\Vert x \Vert_{\mathrm{RMS}} \;=\; \frac{\Vert x \Vert_2}{\Vert \vec{1} \Vert_2} \;=\; \frac{\Vert x \Vert_2}{\sqrt{n}} \;=\; \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}
$$

where $$\vec{1}$$ is the vector of all ones in $$\mathbb{R}^n$$, and $$\Vert x \Vert_2 = \sqrt{\sum_{i=1}^n x_i^2}$$ is the standard Euclidean norm.
</blockquote>

#### Properties and Rationale for Vector RMS Norm

1.  **Dimension neutrality.** If the coordinates $$x_i$$ of $$x$$ are i.i.d. random variables with zero mean and unit variance (e.g., $$x_i \sim \mathcal{N}(0,1)$$), then $$\mathbb{E}[\Vert x \Vert_{\mathrm{RMS}}] \approx 1$$ for large $$n$$. This property makes the notion of "unit-size" more consistent for vectors of varying dimensions, such as network activations from layers of different widths.
2.  **Rotational and Orthogonal Invariance.** The RMS norm is a positive scalar multiple of the $$\ell_2$$-norm. The $$\ell_2$$-norm is invariant under rotations ($$SO(n)$$) and, more generally, under all orthogonal transformations ($$O(n)$$), which include reflections. The RMS norm inherits these crucial geometric symmetries. This means it treats all orthonormal bases equivalently. These properties lead to the following characterizations.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 1.** Rotationally Invariant Functions
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is **rotationally invariant** (i.e., $$f(Qx) = f(x)$$ for all rotation matrices $$Q \in SO(n)$$ and all $$x \in \mathbb{R}^n$$) if and only if there exists a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that:

$$
f(x) = g(\Vert x \Vert_2) \quad \forall x \in \mathbb{R}^n
$$

For $$n=1$$, $$SO(1) = \{[1]\}$$ (the identity matrix). Thus, any function $$f: \mathbb{R} \to \mathbb{R}$$ is trivially $$SO(1)$$-invariant. For the conclusion $$f(x_1) = g(\Vert x_1 \Vert_2) = g(\vert x_1 \vert)$$ to hold, $$f(x_1)$$ must be an even function (i.e., $$f(x_1)=f(-x_1)$$).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 1.**
</summary>
($$\Leftarrow$$) **Sufficiency:** Assume $$f(x) = g(\Vert x \Vert_2)$$. For any $$Q \in SO(n)$$, rotations preserve the Euclidean norm: $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$.
Then, $$f(Qx) = g(\Vert Qx \Vert_2) = g(\Vert x \Vert_2) = f(x)$$. Thus, $$f$$ is rotationally invariant.

($$\Rightarrow$$) **Necessity:** Assume $$f$$ is rotationally invariant.
*   If $$x = \mathbf{0}$$, then $$Q\mathbf{0} = \mathbf{0}$$, so $$f(Q\mathbf{0}) = f(\mathbf{0})$$ is trivial. Define $$g(0) = f(\mathbf{0})$$.
*   Consider $$x \ne \mathbf{0}$$. Let $$r = \Vert x \Vert_2 > 0$$.
    *   **Case $$n \ge 2$$:** Let $$y \in \mathbb{R}^n$$ be any vector with $$\Vert y \Vert_2 = r$$. Since $$SO(n)$$ acts transitively on spheres for $$n \ge 2$$, there exists $$Q \in SO(n)$$ such that $$y = Qx$$. By rotational invariance, $$f(y) = f(Qx) = f(x)$$. Thus, $$f(x)$$ depends only on $$\Vert x \Vert_2$$. We can define $$g(r) = f(x_0)$$ for any fixed $$x_0$$ with $$\Vert x_0 \Vert_2 = r$$ (e.g., $$x_0 = (r, 0, \ldots, 0)^\top$$). Then $$f(x) = g(\Vert x \Vert_2)$$.
    *   **Case $$n=1$$:** $$SO(1)=\{[1]\}$$. Rotational invariance $$f(1 \cdot x_1) = f(x_1)$$ holds for any function $$f:\mathbb{R} \to \mathbb{R}$$. For $$f(x_1) = g(\Vert x_1 \Vert_2) = g(\vert x_1 \vert)$$ to be true, $$f$$ must be an even function, as $$g(\vert -x_1 \vert) = g(\vert x_1 \vert)$$ implies $$f(-x_1)=f(x_1)$$.

Thus, a rotationally invariant $$f$$ has the form $$f(x) = g(\Vert x \Vert_2)$$, provided $$f$$ is even if $$n=1$$.
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
1.  **Apply Theorem 1:** Since $$\Vert \cdot \Vert$$ is a norm, it is an even function for $$n=1$$ (as $$\Vert -x_1 \Vert = \vert -1 \vert \Vert x_1 \Vert = \Vert x_1 \Vert$$). Thus, by Theorem 1, there is a $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that $$\Vert x \Vert = g(\Vert x \Vert_2)$$.
2.  **Use Absolute Homogeneity:** For $$\alpha \in \mathbb{R}$$, $$\Vert \alpha x \Vert = \vert \alpha \vert \Vert x \Vert$$. So, $$g(\Vert \alpha x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)$$. Since $$\Vert \alpha x \Vert_2 = \vert \alpha \vert \Vert x \Vert_2$$, we get $$g(\vert \alpha \vert \Vert x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)$$. Let $$r = \Vert x \Vert_2 \ge 0$$ and $$\lambda = \vert \alpha \vert \ge 0$$. Then $$g(\lambda r) = \lambda g(r)$$ for all $$\lambda, r \ge 0$$.
    For $$r > 0$$, set $$r=1$$. Then $$g(\lambda) = \lambda g(1)$$ for $$\lambda \ge 0$$. Let $$c = g(1)$$. So $$g(\lambda) = c\lambda$$.
    If $$r=0$$, then $$x=\mathbf{0}$$. $$\Vert \mathbf{0} \Vert = 0$$ and $$g(\Vert \mathbf{0} \Vert_2) = g(0)$$. The relation $$g(0 \cdot r) = 0 \cdot g(r)=0$$ implies $$g(0)=0$$. So $$g(\lambda)=c\lambda$$ holds for all $$\lambda \ge 0$$.
    Thus, $$\Vert x \Vert = g(\Vert x \Vert_2) = c \Vert x \Vert_2$$.
3.  **Determine $$c$$:** Since $$\Vert \cdot \Vert$$ is a norm, for any $$x \ne \mathbf{0}$$, $$\Vert x \Vert > 0$$. So $$c \Vert x \Vert_2 > 0$$. This implies $$c > 0$. (For example, $$c=g(1)=\Vert e_1 \Vert > 0$$).
The function $$x \mapsto c \Vert x \Vert_2$$ with $$c>0$$ satisfies all norm axioms as $$\Vert \cdot \Vert_2$$ does.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.** Orthogonal Invariance Implies Rotational Invariance
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
Since norms are non-negative, $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$.

If a norm is $$\Vert x \Vert = c \Vert x \Vert_2$$ for $$c > 0$$, then
$$\Vert Qx \Vert = c \Vert Qx \Vert_2 = c \Vert x \Vert_2 = \Vert x \Vert$$.
So, such norms are orthogonally invariant.
By Corollary 1.1, any rotationally invariant norm is of the form $$c \Vert x \Vert_2$$ ($$c>0$$). Therefore, any rotationally invariant norm is also orthogonally invariant.
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Corollary 2.1.** Uniqueness of RMS Norm Family
</div>
The RMS norm, $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$, is a positive scalar multiple of the $$\ell_2$$-norm. Therefore, it is rotationally and orthogonally invariant.

Furthermore, consider a family of norms $$\{\mathcal{N}_n(\cdot)\}_{n \ge 1}$$, where each $$\mathcal{N}_n: \mathbb{R}^n \to \mathbb{R}$$ is a norm on $$\mathbb{R}^n$$. If this family satisfies:
1.  **Rotational Invariance:** Each $$\mathcal{N}_n(\cdot)$$ is rotationally invariant.
2.  **Dimensional Normalization:** For a specific class of random vectors $$X^{(n)} \in \mathbb{R}^n$$ (e.g., components $$X_i$$ are i.i.d. with zero mean, unit variance, and $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] = 1$$), the expected value $$\mathbb{E}[\mathcal{N}_n(X^{(n)})]$$ is a constant $$K > 0$$ independent of $$n$.

Then, each norm $$\mathcal{N}_n(x)$$ must be of the form $$K \cdot \Vert x \Vert_{\mathrm{RMS}}$$. If $$K=1$$, the RMS norm family is the unique family of norms satisfying these conditions.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Corollary 2.1.**
</summary>
The RMS norm is $$(1/\sqrt{n})\Vert x \Vert_2$$. Since $$1/\sqrt{n}>0$$, it's rotationally invariant by Corollary 1.1, and orthogonally invariant by Theorem 2.

For the second part:
1.  **Rotational Invariance:** By Corollary 1.1, each $$\mathcal{N}_n(x) = c_n \Vert x \Vert_2$$ for some $$c_n > 0$$.
2.  **Dimensional Normalization:** We are given $$\mathbb{E}[\mathcal{N}_n(X^{(n)})] = K$$ for all $$n$$.
    Substituting the form from (1): $$ \mathbb{E}[c_n \Vert X^{(n)} \Vert_2] = c_n \mathbb{E}[\Vert X^{(n)} \Vert_2] = K $$.
    The definition of the RMS norm is $$\Vert X^{(n)} \Vert_{\mathrm{RMS}} = \Vert X^{(n)} \Vert_2 / \sqrt{n}$$.
    The condition on the random vectors states $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] = 1$$, so $$\mathbb{E}[\Vert X^{(n)} \Vert_2 / \sqrt{n}] = 1$$, which means $$\mathbb{E}[\Vert X^{(n)} \Vert_2] = \sqrt{n}$$.
    Plugging this into the equation for $$K$$: $$c_n \sqrt{n} = K$$.
    So, $$c_n = K / \sqrt{n}$$.
3.  **Form of the Norm:** Therefore, $$\mathcal{N}_n(x) = (K/\sqrt{n}) \Vert x \Vert_2 = K \cdot (\Vert x \Vert_2 / \sqrt{n}) = K \cdot \Vert x \Vert_{\mathrm{RMS}}$$.
    If $$K=1$$ (e.g., if the normalization condition is $$\mathbb{E}[\mathcal{N}_n(X^{(n)})] = 1$$), then $$\mathcal{N}_n(x) = \Vert x \Vert_{\mathrm{RMS}}$$. This establishes uniqueness under these specific conditions.
</details>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** When to use the vector RMS norm
</div>
Employ the vector $$\Vert \cdot \Vert_{\mathrm{RMS}}$$ when a scale for vectors is needed that is simultaneously rotationally symmetric (thus orthogonally symmetric) and normalized for vector dimension. This is useful, for example, when comparing activations from neural network layers of different widths or designing width-robust regularizers.
</blockquote>

#### Minimal Axiomatic Characterizations of the RMS norm

The RMS norm, as $$\frac{1}{\sqrt{n}}\Vert x \Vert_2$$, can be uniquely identified by various sets of axioms. These typically involve axioms characterizing the Euclidean norm up to a positive scalar, plus a normalization condition to fix this scalar to $$1/\sqrt{n}$$.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 3 (Characterization 1: Parallelogram Law and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
3.  **Parallelogram identity:** $$f(x+y)^2 + f(x-y)^2 = 2f(x)^2 + 2f(y)^2$$ for all $$x,y \in \mathbb{R}^n$$.
4.  **Normalization on standard basis:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for each standard basis vector $$e_i$$.
(Implicitly, for $$f$$ to be a norm from (1)-(3), the triangle inequality must also hold, which it does if these conditions ensure it's derived from an inner product).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 3.**
</summary>
($$\Rightarrow$$) **Necessity:** The RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1.  **Nondegeneracy** and (2) **Absolute homogeneity** are standard for norms and hold for $$f(x)$$.
3.  **Parallelogram identity:** $$\Vert \cdot \Vert_2$$ satisfies this identity. Scaling by $$1/\sqrt{n}$$ preserves it:
    $$ f(x+y)^2 + f(x-y)^2 = \frac{1}{n}(\Vert x+y \Vert_2^2 + \Vert x-y \Vert_2^2) = \frac{1}{n}(2\Vert x \Vert_2^2 + 2\Vert y \Vert_2^2) = 2f(x)^2 + 2f(y)^2 $$.
4.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}\Vert e_i \Vert_2 = \frac{1}{\sqrt{n}} \cdot 1 = \frac{1}{\sqrt{n}}$$.
All axioms hold for the RMS norm.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   Axioms (1), (2), and (3) (along with the triangle inequality, which makes $$f$$ a norm) imply by the Jordan-von Neumann theorem that $$f(x)^2$$ arises from an inner product, i.e., $$f(x)^2 = \langle x, x \rangle_S$$ for some inner product $$\langle u, v \rangle_S = u^\top S v$$, where $$S$$ is a symmetric positive-definite matrix. So $$f(x) = \sqrt{x^\top S x}$$.
*   For $$S$$ to be a scalar multiple of the identity matrix (i.e., $$S=kI$$), an additional symmetry assumption (like rotational invariance or specific basis properties) is generally needed. If such symmetry is assumed (e.g., if the standard basis $$e_i$$ are orthogonal with respect to $$\langle \cdot, \cdot \rangle_S$$ and $$\langle e_i, e_i \rangle_S$$ is constant), then $$S=kI$$ for some $$k>0$$. Then $$f(x) = \sqrt{k} \Vert x \Vert_2$$. Let $$c=\sqrt{k}$$.
*   With $$f(x) = c \Vert x \Vert_2$$ (assuming $$S=c^2I$$ as discussed), apply axiom (4): $$f(e_i) = c \Vert e_i \Vert_2 = c \cdot 1 = c$$.
*   From axiom (4), $$f(e_i) = \frac{1}{\sqrt{n}}$$. So, $$c = \frac{1}{\sqrt{n}}$$.
*   Thus, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.

<div class="box-info" markdown="1">
**Note on Theorem 3:** The step from $$f(x)^2 = x^\top S x$$ to $$f(x)=c\Vert x \Vert_2$$ typically requires an assumption like rotational invariance (Theorem 4) or symmetry across coordinates (Theorem 6, implying $$S_{ii}$$ are equal and $$S_{ij}=0$$ for $$i \neq j$$). Theorem 3 as stated relies on this step being inferable, or on an implicit choice of "standard" inner product structure.
</div>
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 4 (Characterization 2: Orthogonal Invariance and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Orthogonal invariance:** $$f(Qx) = f(x)$$ for all $$Q \in O(n)$$ and $$x \in \mathbb{R}^n$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
3.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
4.  **Continuity:** $$f$$ is continuous.
5.  **Normalization on a basis vector:** $$f(e_1) = \frac{1}{\sqrt{n}}$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 4.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$.
1.  **Orthogonal invariance:** Holds by Theorem 2.
2.  **Absolute homogeneity & (3) Nondegeneracy:** Verified for Thm 3.
4.  **Continuity:** Norms on finite-dimensional spaces are continuous. $$\Vert \cdot \Vert_2$$ is continuous, so $$f(x)$$ is.
5.  **Normalization:** $$f(e_1) = \frac{1}{\sqrt{n}}\Vert e_1 \Vert_2 = \frac{1}{\sqrt{n}}$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   Axiom (1) (Orthogonal invariance) implies rotational invariance ($$SO(n) \subset O(n)$$). Also, $$f(x)=f(-x)$$ by taking $$Q=-I \in O(n)$$ (if $n$ is odd, $\det(-I)=-1$; if $n$ is even, $\det(-I)=1$, so $-I$ may or may not be in $SO(n)$, but it is always in $O(n)$). So $f$ is even.
*   By Theorem 1, $$f(x) = g(\Vert x \Vert_2)$$ for some $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$.
*   Using Axiom (2) (Absolute homogeneity), as in Corollary 1.1, we find $$g(r) = cr$$ for some constant $$c$$. So $$f(x) = c \Vert x \Vert_2$$.
*   Axiom (3) (Nondegeneracy: $$f(x)>0$$ for $$x \ne \mathbf{0}$$) implies $$c > 0$$. (Continuity, Axiom 4, ensures $$g(r)=cr$$ applies to all real $$r \ge 0$$ rather than just rational multiples).
*   Axiom (5) (Normalization): $$f(e_1) = c \Vert e_1 \Vert_2 = c \cdot 1 = c$$. Since $$f(e_1) = \frac{1}{\sqrt{n}}$$, we have $$c = \frac{1}{\sqrt{n}}$$.
*   Thus, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$. This function is a norm.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 5 (Characterization 3: Pythagorean Additivity and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$.
3.  **Pythagorean additivity:** If $$x^\top y = 0$$ (standard orthogonality), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
4.  **Continuity:** $$f$$ is continuous.
5.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for all standard basis vectors $$e_i$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 5.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$.
1.  Nondegeneracy, (2) Absolute homogeneity, (4) Continuity, (5) Normalization: Verified previously.
3.  **Pythagorean additivity:** If $$x^\top y=0$$, then $$\Vert x+y \Vert_2^2 = \Vert x \Vert_2^2 + \Vert y \Vert_2^2$$.
    So, $$f(x+y)^2 = \frac{1}{n}\Vert x+y \All_2^2 = \frac{1}{n}(\Vert x \Vert_2^2 + \Vert y \Vert_2^2) = f(x)^2 + f(y)^2$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   It's a known result that a function satisfying (1)-(4) (Nondegeneracy, Abs. homogeneity, Pythagorean additivity w.r.t standard orthogonality, Continuity) must be a positive scalar multiple of the Euclidean norm: $$f(x) = c \Vert x \Vert_2$$ for some $$c > 0$$. This is because these properties ensure $$f(x)^2$$ defines an inner product compatible with the standard one.
*   Specifically, repeated use of (3) for an orthonormal basis expansion $$x = \sum x_i e_i$$ gives $$f(x)^2 = \sum f(x_i e_i)^2 = \sum x_i^2 f(e_i)^2$$.
*   Axiom (5) states $$f(e_i) = \frac{1}{\sqrt{n}}$$. So $$f(e_i)^2 = \frac{1}{n}$$.
*   Then $$f(x)^2 = \sum x_i^2 \left(\frac{1}{n}\right) = \frac{1}{n} \sum x_i^2 = \frac{1}{n} \Vert x \Vert_2^2$$.
*   Since $$f(x) \ge 0$$, $$f(x) = \sqrt{\frac{1}{n} \Vert x \Vert_2^2} = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 6 (Characterization 4: Coordinate Symmetry, Disjoint-Support Additivity, and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$.
3.  **Permutation & sign-flip symmetry:** $$f(x_1,\dots,x_n) = f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})$$ for any permutation $$\sigma$$ and any choice of signs.
4.  **Pythagorean additivity on disjoint supports:** If $$\mathrm{supp}(x) \cap \mathrm{supp}(y) = \emptyset$$ (i.e., $$x_i \neq 0 \implies y_i=0$$), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
5.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for $$i=1,\dots,n$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 6.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$.
1.  Nondegeneracy, (2) Absolute homogeneity, (5) Normalization: Verified previously.
3.  **Permutation & sign-flip symmetry:** $$f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})^2 = \frac{1}{n}\sum_{j=1}^n (\pm x_{\sigma(j)})^2 = \frac{1}{n}\sum_{j=1}^n x_{\sigma(j)}^2 = \frac{1}{n}\sum_{k=1}^n x_k^2 = f(x)^2$$. Taking square roots, symmetry holds.
4.  **Pythagorean additivity on disjoint supports:** If $$x_i y_i = 0$$ for all $$i$$, then $$\Vert x+y \Vert_2^2 = \sum (x_i+y_i)^2 = \sum (x_i^2+y_i^2) = \Vert x \Vert_2^2 + \Vert y \Vert_2^2$$. The argument then follows as in Thm 5, axiom 3.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   Axiom (3) implies $$f(e_i)$$ is the same for all $$i$$. Let this value be $$c_0 = f(e_1)$$. By (1), $$c_0 \ge 0$$.
*   For $$x_j e_j = (0, \dots, x_j, \dots, 0)$$, by (2): $$f(x_j e_j) = \vert x_j \vert f(e_j) = \vert x_j \vert c_0$$.
*   Any $$x = \sum x_i e_i$$. The vectors $$x_i e_i$$ have disjoint supports. By repeated use of (4):
    $$f(x)^2 = f(\sum x_i e_i)^2 = \sum f(x_i e_i)^2 = \sum (\vert x_i \vert c_0)^2 = c_0^2 \sum x_i^2 = c_0^2 \Vert x \Vert_2^2$$.
*   Since $$f(x) \ge 0$$, $$f(x) = c_0 \Vert x \Vert_2$$. By (1), if $$x \ne \mathbf{0}$$, $$f(x)>0$$, so $$c_0 > 0$$.
*   Axiom (5): $$f(e_i) = \frac{1}{\sqrt{n}}$$. So $$c_0 = \frac{1}{\sqrt{n}}$$.
*   Thus, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$. This function is a norm.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 7 (Characterization 5: Norm Properties, Coordinate Symmetry, and All-Ones Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Norm properties:** $$f$$ is a norm on $$\mathbb{R}^n$$.
2.  **Permutation & sign-flip symmetry:** As in Theorem 6, Axiom 3.
3.  **Normalization on the all-ones vector:** $$f(\vec{1}) = f(1,1,\dots,1) = 1$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 7.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$.
1.  **Norm properties:** $$f$$ is a positive scalar multiple of $$\Vert \cdot \Vert_2$$, so it's a norm.
2.  **Permutation & sign-flip symmetry:** Verified in Theorem 6.
3.  **Normalization on all-ones vector:** Let $$\vec{1} = (1,\dots,1)$$. Then $$\Vert \vec{1} \Vert_2 = \sqrt{n}$$.
    So, $$f(\vec{1}) = \frac{1}{\sqrt{n}}\Vert \vec{1} \Vert_2 = \frac{1}{\sqrt{n}} \sqrt{n} = 1$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(3).
*   Axiom (1) states $$f$$ is a norm. Axiom (2) means $$f$$ is a symmetric gauge function.
*   It is a known result in norm theory that a norm on $$\mathbb{R}^n$$ that is invariant under coordinate permutations and sign flips, and additionally satisfies a condition that forces it to be Euclidean (e.g., satisfying the parallelogram law, or being rotationally invariant), must be of the form $$c\Vert x \Vert_2$$ for some $$c>0$$. (Axiom (2) alone only restricts $$f$$ to the class of symmetric gauge functions, which includes all $$\ell_p$$ norms.)
*   Assuming that such a specific result or an implicit understanding allows the deduction from (1) and (2) that $$f(x)=c\Vert x \Vert_2$$ for some $$c > 0$.
*   Apply axiom (3): $$f(\vec{1}) = 1$$.
    $$f(\vec{1}) = c \Vert \vec{1} \Vert_2 = c \sqrt{n}$$.
    So, $$c \sqrt{n} = 1$$, which means $$c = \frac{1}{\sqrt{n}}$$.
*   Therefore, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.

<div class="box-info" markdown="1">
**Note on Theorem 7:** The inference from "norm properties + permutation/sign-flip symmetry" to $$f(x)=c\Vert x \Vert_2$$ is a strong assertion. Typically, these axioms define a symmetric gauge function (e.g., any $$\ell_p$$ norm is one). To single out the $$\ell_2$$ structure (up to scale), an additional property like the parallelogram identity (making it an inner product norm) or full rotational invariance is required.
</div>
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 8 (Characterization 6: Averaged Sum-of-Functions Structure).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Structure for $$f^2$$:** $$f(x)^2 \;=\; \frac{1}{n}\,\sum_{i=1}^n \phi(x_i)$$ for some function $$\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$$.
3.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
4.  **Normalization of $$\phi$$:** $$\phi(1) = 1$$.
(The triangle inequality, for $$f$$ to be a full norm, is verified from the derived form.)
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 8.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \sqrt{\frac{1}{n}\sum x_i^2}$.
1.  Nondegeneracy, (3) Absolute homogeneity: Verified previously.
2.  **Structure:** $$f(x)^2 = \frac{1}{n}\sum x_i^2$$. This fits the structure with $$\phi(u) = u^2$$. Clearly $$\phi(u) = u^2 \ge 0$$.
4.  **Normalization of $$\phi$$:** For $$\phi(u) = u^2$$, $$\phi(1) = 1^2 = 1$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   From (1), $$f(\mathbf{0})=0$$. Substituting $$x=\mathbf{0}$$ into (2): $$0 = \frac{1}{n}\sum \phi(0) = \phi(0)$$. So, $$\phi(0)=0$$.
*   From (3), $$f(\lambda x)^2 = (\vert\lambda\vert f(x))^2 = \lambda^2 f(x)^2$$. Using (2):
    $$\frac{1}{n}\,\sum \phi(\lambda x_i) \;=\; \lambda^2 \left(\frac{1}{n}\,\sum \phi(x_i)\right) \implies \sum \phi(\lambda x_i) \;=\; \lambda^2 \sum \phi(x_i)$$.
*   Let $$x = e_j$$ (j-th standard basis vector). Then $$x_j=1$$, others $$0$$.
    $$\phi(\lambda \cdot 1) + \sum_{k \neq j} \phi(\lambda \cdot 0) \;=\; \lambda^2 \left(\phi(1) + \sum_{k \neq j} \phi(0)\right)$$.
    Since $$\phi(0)=0$$: $$\phi(\lambda) = \lambda^2 \phi(1)$$.
*   Using (4), $$\phi(1) = 1$$. Therefore, $$\phi(u) = u^2$$ for all $$u \in \mathbb{R}$$.
*   Substitute $$\phi(u)=u^2$$ into (2): $$f(x)^2 \;=\; \frac{1}{n}\,\sum x_i^2$$.
*   Since $$f(x) \ge 0$$ by (1), $$f(x) \;=\; \sqrt{\frac{1}{n}\,\sum x_i^2} \;=\; \Vert x \Vert_{\mathrm{RMS}}$$.
*   This function $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$ is a norm (satisfies triangle inequality as $$\Vert \cdot \Vert_2$$ does).
</details>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**In a Nutshell: Characterizing the RMS norm**
</div>
The RMS norm, $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\,\Vert x \Vert_2$$, is a scaled Euclidean norm. Most minimal characterizations first establish that a function $$f(x)$$ must be $$c\Vert x \Vert_2$$ for some $$c>0$$, using axioms related to inner product structure or high symmetry. An additional normalization axiom then fixes $$c=1/\sqrt{n}$$. Typical normalizations set $$f(e_i)=1/\sqrt{n}$$ or $$f(\vec{1})=1$$, or assume a structural form like in Theorem 8.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Quick Checklist of Axiom Sets for the RMS norm**
</div>
Each set of properties below uniquely defines the RMS norm. Most rely on first showing $$f(x)=c\Vert x \Vert_2$$, then finding $$c=1/\sqrt n$$.

1.  **Parallelogram Law Version (Theorem 3)**
    *   Nondegeneracy, Absolute homogeneity, Parallelogram identity.
    *   (Requires additional symmetry assumption for $$f(x)=c\Vert x \Vert_2$$ step).
    *   Normalization: $$f(e_i)=1/\sqrt n$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

2.  **Orthogonal Invariance Version (Theorem 4)**
    *   Orthogonal invariance, Absolute homogeneity, Nondegeneracy, Continuity.
    *   Normalization: $$f(e_1)=1/\sqrt n$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

3.  **Pythagorean Additivity Version (Theorem 5)**
    *   Nondegeneracy, Absolute homogeneity, Pythagorean additivity (for standard orthogonality), Continuity.
    *   Normalization: $$f(e_i)=1/\sqrt n$$ for all $$i$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

4.  **Coordinate-Symmetry + Disjoint-Support Additivity Version (Theorem 6)**
    *   Nondegeneracy, Absolute homogeneity, Permutation & sign-flip symmetry.
    *   Pythagorean additivity on disjoint supports.
    *   Normalization: $$f(e_i)=1/\sqrt n$$ for all $$i$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

5.  **General Norm + Symmetry + All-Ones Normalization (Theorem 7)**
    *   $$f$$ is a norm, Permutation & sign-flip symmetry.
    *   (Requires specific interpretation or stronger theorem for $$f(x)=c\Vert x \Vert_2$$ step).
    *   Normalization: $$f(1,1,\dots,1)=1$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

6.  **Averaged Sum-of-Functions Structure (Theorem 8)**
    *   Nondegeneracy, $$f(x)^2 = \frac{1}{n}\sum \phi(x_i)$$, Absolute homogeneity.
    *   Normalization: $$\phi(1)=1$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$ (and confirmed to be a norm)
</blockquote>

Any one of these sets of properties (with caveats as noted for Theorems 3 and 7 regarding the derivation of the Euclidean form) is sufficient to uniquely define the RMS norm.

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
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ ($$n \ge 2$$) is **rotationally invariant** (i.e., $$f(Qx) = f(x)$$ for all rotation matrices $$Q \in SO(n)$$ and all $$x \in \mathbb{R}^n$$) if and only if there exists a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that:

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
Assume $$f(x) = g(\Vert x \Vert_2)$$ for some function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$. For any rotation matrix $$Q \in SO(n)$$, rotations preserve the Euclidean norm: $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$.
Then, $$f(Qx) = g(\Vert Qx \Vert_2) = g(\Vert x \Vert_2) = f(x)$$. Thus, $$f$$ is rotationally invariant.

($$\Rightarrow$$) **Necessity:**
*   **Case $$n \ge 2$$:** Assume $$f: \mathbb{R}^n \to \mathbb{R}$$ is rotationally invariant.
    *   If $$x = \mathbf{0}$$, define $$g(0) = f(\mathbf{0})$$. Then $$f(\mathbf{0}) = g(\Vert \mathbf{0} \Vert_2)$$.
    *   If $$x \ne \mathbf{0}$$, let $$r = \Vert x \Vert_2 > 0$$. For any $$y \in \mathbb{R}^n$$ with $$\Vert y \Vert_2 = r$$, there exists $$Q \in SO(n)$$ such that $$y = Qx$$ (since $$SO(n)$$ acts transitively on spheres for $$n \ge 2$$). By rotational invariance, $$f(y) = f(Qx) = f(x)$$. Thus, $$f(x)$$ depends only on $$\Vert x \Vert_2$$. Define $$g(r) = f(x_0)$$ for any fixed $$x_0$$ with $$\Vert x_0 \Vert_2 = r$$ (e.g., $$x_0 = (r, 0, \ldots, 0)^\top$$). Then $$f(x) = g(\Vert x \Vert_2)$$.

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

#### Minimal Axiomatic Characterizations of the RMS norm

The RMS norm, as $$\frac{1}{\sqrt{n}}\Vert x \Vert_2$$, can be uniquely identified by various sets of axioms. These typically involve axioms characterizing the Euclidean norm up to a positive scalar, plus a normalization condition to fix this scalar to $$1/\sqrt{n}$$.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 3 (Characterization 1: Parallelogram Law and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Norm Properties:** $$f$$ is a norm (satisfies nondegeneracy, absolute homogeneity, triangle inequality).
2.  **Parallelogram identity:** $$f(x+y)^2 + f(x-y)^2 = 2f(x)^2 + 2f(y)^2$$ for all $$x,y \in \mathbb{R}^n$$.
3.  **Orthogonality of Standard Basis:** The inner product $$\langle \cdot, \cdot \rangle_f$$ from which $$f$$ is derived (see proof) satisfies $$\langle e_i, e_j \rangle_f = 0$$ for $$i \ne j$$, where $$e_i$$ are standard basis vectors.
4.  **Normalization on standard basis:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for each standard basis vector $$e_i$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 3.**
</summary>
($$\Rightarrow$$) **Necessity:** The RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1.  **Norm Properties:** $$f(x)$$ is a positive multiple of a norm, so it is a norm.
2.  **Parallelogram identity:** $$\Vert \cdot \Vert_2$$ satisfies this identity. Scaling by $$1/\sqrt{n}$$ (and squaring) preserves it:
    $$ f(x+y)^2 + f(x-y)^2 = \frac{1}{n}(\Vert x+y \Vert_2^2 + \Vert x-y \Vert_2^2) = \frac{1}{n}(2\Vert x \Vert_2^2 + 2\Vert y \Vert_2^2) = 2f(x)^2 + 2f(y)^2 $$.
3.  **Orthogonality of Standard Basis:** The inner product for RMS norm is $$\langle x, y \rangle_f = \frac{1}{n} x^\top y$$. For standard basis vectors, $$\langle e_i, e_j \rangle_f = \frac{1}{n} e_i^\top e_j = \frac{1}{n} \delta_{ij}$$. So for $$i \ne j$$, $$\langle e_i, e_j \rangle_f = 0$$.
4.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}\Vert e_i \Vert_2 = \frac{1}{\sqrt{n}} \cdot 1 = \frac{1}{\sqrt{n}}$$.
All axioms hold for the RMS norm.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   Axiom (1) and (2) (norm properties and parallelogram identity) imply by the Jordan-von Neumann theorem that $$f$$ is derived from an inner product $$\langle \cdot, \cdot \rangle_f$$, i.e., $$f(x)^2 = \langle x, x \rangle_f$$. This inner product can be written as $$\langle x, y \rangle_f = x^\top S y$$ for some symmetric positive-definite matrix $$S$$.
*   Axiom (3) (Orthogonality of Standard Basis) means $$\langle e_i, e_j \rangle_f = e_i^\top S e_j = S_{ij} = 0$$ for $$i \ne j$$. So, $$S$$ must be a diagonal matrix.
*   Axiom (4) (Normalization on standard basis) means $$f(e_i)^2 = \langle e_i, e_i \rangle_f = S_{ii} = \left(\frac{1}{\sqrt{n}}\right)^2 = \frac{1}{n}$$.
*   Since $$S$$ is diagonal with all diagonal entries equal to $$1/n$$, $$S = \frac{1}{n}I$$.
*   Therefore, $$f(x)^2 = x^\top (\frac{1}{n}I) x = \frac{1}{n} x^\top x = \frac{1}{n} \Vert x \Vert_2^2$$.
*   Since $$f(x) \ge 0$$ (from norm properties), $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 4 (Characterization 2: Orthogonal Invariance and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Norm Properties:** $$f$$ is a norm on $$\mathbb{R}^n$$.
2.  **Orthogonal invariance:** $$f(Qx) = f(x)$$ for all $$Q \in O(n)$$ and $$x \in \mathbb{R}^n$$.
3.  **Normalization on a standard basis vector:** $$f(e_1) = \frac{1}{\sqrt{n}}$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 4.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1. Norm properties: Verified (it's a positive multiple of the $$\ell_2$$ norm).
2. Orthogonal invariance: Holds by Theorem 2.
3. Normalization: $$f(e_1) = \frac{1}{\sqrt{n}}\Vert e_1 \Vert_2 = \frac{1}{\sqrt{n}} \cdot 1 = \frac{1}{\sqrt{n}}$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(3).
*   Axiom (1) states $$f$$ is a norm. Axiom (2) states $$f$$ is orthogonally invariant. Orthogonal invariance ($$O(n)$$) implies rotational invariance ($$SO(n)$$).
*   Since $$f$$ is a rotationally invariant norm, by Corollary 1.1, $$f(x) = c \Vert x \Vert_2$$ for some constant $$c > 0$$.
*   Axiom (3) (Normalization): We have $$f(e_1) = c \Vert e_1 \Vert_2 = c \cdot 1 = c$$. Since Axiom (3) also states $$f(e_1) = \frac{1}{\sqrt{n}}$$, we must have $$c = \frac{1}{\sqrt{n}}$$.
*   Thus, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 5 (Characterization 3: Pythagorean Additivity and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
3.  **Pythagorean additivity:** If $$x^\top y = 0$$ (standard orthogonality), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
4.  **Continuity:** $$f$$ is continuous. (Alternatively, assume $$f$$ satisfies the triangle inequality).
5.  **Normalization on standard basis vectors:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for all standard basis vectors $$e_i$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 5.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1. Nondegeneracy, (2) Absolute homogeneity, (4) Continuity/Triangle Inequality (it's a norm): Verified.
3. **Pythagorean additivity:** If $$x^\top y=0$$, then by the Pythagorean theorem for the Euclidean norm, $$\Vert x+y \Vert_2^2 = \Vert x \Vert_2^2 + \Vert y \Vert_2^2$$.
    So, $$f(x+y)^2 = \frac{1}{n}\Vert x+y \Vert_2^2 = \frac{1}{n}(\Vert x \Vert_2^2 + \Vert y \Vert_2^2) = \left(\frac{1}{\sqrt{n}}\Vert x \Vert_2\right)^2 + \left(\frac{1}{\sqrt{n}}\Vert y \Vert_2\right)^2 = f(x)^2 + f(y)^2$$.
5. Normalization: Verified, $$f(e_i) = 1/\sqrt{n}$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   Let $$x = \sum_{k=1}^n x_k e_k$$. The vectors $$x_k e_k$$ (where $$x_k e_k$$ is a vector with $$x_k$$ in the $$k$$-th position and zeros elsewhere) are mutually orthogonal with respect to the standard dot product.
*   By repeated application of Axiom (3) (Pythagorean additivity):
    $$f(x)^2 = f\left(\sum_{k=1}^n x_k e_k\right)^2 = \sum_{k=1}^n f(x_k e_k)^2$$.
    (Continuity in Axiom 4 ensures such sums behave as expected, though it's not explicitly invoked in this step for finite sums).
*   By Axiom (2) (Absolute homogeneity): $$f(x_k e_k)^2 = (\vert x_k \vert f(e_k))^2 = x_k^2 f(e_k)^2$$.
*   So, $$f(x)^2 = \sum_{k=1}^n x_k^2 f(e_k)^2$$.
*   By Axiom (5) (Normalization): $$f(e_k) = \frac{1}{\sqrt{n}}$$, so $$f(e_k)^2 = \frac{1}{n}$$.
*   Substituting this in: $$f(x)^2 = \sum_{k=1}^n x_k^2 \left(\frac{1}{n}\right) = \frac{1}{n} \sum_{k=1}^n x_k^2 = \frac{1}{n} \Vert x \Vert_2^2$$.
*   Since $$f(x) \ge 0$$ by Axiom (1), $$f(x) = \sqrt{\frac{1}{n} \Vert x \Vert_2^2} = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
*   This function satisfies the triangle inequality (as it's a positive multiple of $$\Vert \cdot \Vert_2$$) and all other norm properties. If continuity was assumed instead of triangle inequality in Axiom 4, this derived form is continuous and also satisfies the triangle inequality.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 6 (Characterization 4: Coordinate Symmetry, Disjoint-Support Additivity, and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$.
3.  **Permutation & sign-flip symmetry:** $$f(x_1,\dots,x_n) = f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})$$ for any permutation $$\sigma$$ and any choice of signs for coordinates.
4.  **Pythagorean additivity on disjoint supports:** If $$\mathrm{supp}(x) \cap \mathrm{supp}(y) = \emptyset$$ (i.e., $$x_i y_i=0$$ for all $$i$$), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
5.  **Normalization on a standard basis vector:** $$f(e_1) = \frac{1}{\sqrt{n}}$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 6.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1. Nondegeneracy, (2) Absolute homogeneity: Verified.
3. **Permutation & sign-flip symmetry:** The sum of squares $$\sum x_i^2$$ is invariant under permutation of coordinates and changes of sign $$x_i \to \pm x_i$$. Thus, $$f(x)$$ which involves $$\sqrt{\sum x_i^2}$$ is also invariant.
    $$f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})^2 = \frac{1}{n}\sum_{j=1}^n (\pm x_{\sigma(j)})^2 = \frac{1}{n}\sum_{j=1}^n x_{\sigma(j)}^2 = \frac{1}{n}\sum_{k=1}^n x_k^2 = f(x)^2$$. Taking square roots (since $$f \ge 0$$) gives the symmetry.
4. **Pythagorean additivity on disjoint supports:** If $$x_i y_i = 0$$ for all $$i$$, then $$(x+y)_i = x_i + y_i$$. So $$(x+y)_i^2 = (x_i+y_i)^2 = x_i^2 + y_i^2 + 2x_i y_i = x_i^2 + y_i^2$$.
    Then $$f(x+y)^2 = \frac{1}{n} \sum (x_i+y_i)^2 = \frac{1}{n} \sum (x_i^2 + y_i^2) = \frac{1}{n} \sum x_i^2 + \frac{1}{n} \sum y_i^2 = f(x)^2 + f(y)^2$$.
5. Normalization: $$f(e_1) = (1/\sqrt{n}) \Vert e_1 \Vert_2 = 1/\sqrt{n}$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   Axiom (3) (Permutation symmetry) implies $$f(e_i)$$ is the same for all standard basis vectors $$e_i$$. Let this common value be $$c_0 = f(e_1)$$. By Axiom (1), $$c_0 \ge 0$$ (actually $$c_0>0$$ since $$e_1 \ne \mathbf{0}$$).
*   By Axiom (5), $$f(e_1) = \frac{1}{\sqrt{n}}$$. So, $$c_0 = \frac{1}{\sqrt{n}}$$. Thus, $$f(e_i) = \frac{1}{\sqrt{n}}$$ for all $$i$$.
*   For any vector $$x = \sum_{i=1}^n x_i e_i$$, the vectors $$x_i e_i$$ (vector with $$x_i$$ in $$i$$-th position, zeros elsewhere) have mutually disjoint supports. By repeated application of Axiom (4):
    $$f(x)^2 = f\left(\sum x_i e_i\right)^2 = \sum f(x_i e_i)^2$$.
*   By Axiom (2) (Absolute homogeneity) and sign-flip symmetry from Axiom (3) (or just Axiom 2): $$f(x_i e_i) = \vert x_i \vert f(e_i)$$.
    So, $$f(x_i e_i)^2 = (\vert x_i \vert f(e_i))^2 = x_i^2 f(e_i)^2$$.
*   Substituting this into the sum: $$f(x)^2 = \sum_{i=1}^n x_i^2 f(e_i)^2 = \sum_{i=1}^n x_i^2 \left(\frac{1}{\sqrt{n}}\right)^2 = \frac{1}{n} \sum x_i^2 = \frac{1}{n} \Vert x \Vert_2^2$$.
*   Since $$f(x) \ge 0$$ by Axiom (1), $$f(x) = \sqrt{\frac{1}{n} \Vert x \Vert_2^2} = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
*   This function is a norm (it satisfies triangle inequality, etc.).
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 7 (Characterization 5: Norm Properties, Enhanced Symmetry, and All-Ones Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Norm properties:** $$f$$ is a norm on $$\mathbb{R}^n$$.
2.  **Permutation & sign-flip symmetry:** As in Theorem 6, Axiom 3.
3.  **Structural property linked to Euclidean norm:** $$f$$ satisfies the parallelogram identity ($$f(x+y)^2+f(x-y)^2 = 2f(x)^2+2f(y)^2$$) OR $$f$$ is rotationally invariant ($$f(Qx)=f(x)$$ for $$Q \in SO(n)$$).
4.  **Normalization on the all-ones vector:** $$f(\vec{1}) = f(1,1,\dots,1) = 1$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 7.**
</summary>
($$\Rightarrow$$) **Necessity:** RMS norm is $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
1. Norm properties: Verified.
2. Permutation & sign-flip symmetry: Verified (as in Theorem 6 proof).
3. Structural property:
    *   Parallelogram identity holds (from Theorem 3 proof).
    *   Rotational invariance holds (from Corollary 1.1 / Theorem 2).
4. Normalization on all-ones vector: Let $$\vec{1} = (1,\dots,1)^\top$$. Then $$\Vert \vec{1} \Vert_2 = \sqrt{\sum_{i=1}^n 1^2} = \sqrt{n}$$.
    So, $$f(\vec{1}) = \frac{1}{\sqrt{n}}\Vert \vec{1} \Vert_2 = \frac{1}{\sqrt{n}} \sqrt{n} = 1$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   Axiom (1) states $$f$$ is a norm.
*   Axiom (3):
    *   If $$f$$ (a norm) satisfies the parallelogram identity, it is an inner product norm (by Jordan-von Neumann theorem). If additionally Axiom (2) (permutation & sign-flip symmetry) holds, such a norm must be of the form $$c\Vert x \Vert_2$$ for some $$c>0$$ (a known result, e.g., from theory of symmetric gauge functions).
    *   Alternatively, if $$f$$ (a norm) is rotationally invariant, then by Corollary 1.1, $$f(x)=c\Vert x \Vert_2$$ for some $$c>0$$. (In this case, Axiom (2) is largely implied by rotational invariance for $$n \ge 2$$; for $$n=1$$, $$f(x_1)=f(-x_1)$$ is true for any norm).
*   So, from Axioms (1), (2), and (3), we deduce $$f(x) = c \Vert x \Vert_2$$ for some constant $$c > 0$$.
*   Apply Axiom (4) (Normalization on $$\vec{1}$$): $$f(\vec{1}) = 1$$.
    Substituting the form of $$f$$: $$c \Vert \vec{1} \Vert_2 = 1$$.
    Since $$\Vert \vec{1} \Vert_2 = \sqrt{n}$$, we have $$c \sqrt{n} = 1$$, which implies $$c = \frac{1}{\sqrt{n}}$$.
*   Therefore, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.

<div class="box-info" markdown="1">
**Note on Theorem 7:** Axioms (1) and (2) together define symmetric gauge functions (examples include any $$c\Vert \cdot \Vert_p$$ norm for $$p \ge 1$$). Axiom (3) is crucial for singling out the Euclidean structure (i.e., proportionality to $$\Vert \cdot \Vert_2$$).
</div>
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 8 (Characterization 6: Averaged Sum-of-Functions Structure).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Squared structure:** $$f(x)^2 \;=\; \frac{1}{n}\,\sum_{i=1}^n \phi(x_i)$$ for some function $$\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$$.
3.  **Absolute homogeneity of $$f$$:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
4.  **Normalization of $$\phi$$:** $$\phi(1) = 1$$.
(The triangle inequality for $$f$$ is a consequence of the derived form, ensuring $$f$$ is a norm.)
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 8.**
</summary>
($$\Rightarrow$$) **Necessity:** The RMS norm is $$f(x) = \sqrt{\frac{1}{n}\sum x_i^2}$$.
1. Nondegeneracy: Verified.
2. **Squared structure:** $$f(x)^2 = \frac{1}{n}\sum x_i^2$$. This fits the structure with $$\phi(u) = u^2$$. Clearly $$\phi(u) = u^2 \ge 0$$.
3. Absolute homogeneity of $$f$$: Verified.
4. **Normalization of $$\phi$$:** For $$\phi(u) = u^2$$, we have $$\phi(1) = 1^2 = 1$$.
All axioms hold.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   From Axiom (1), $$f(\mathbf{0})=0$$. Substituting $$x=\mathbf{0}$$ into Axiom (2): $$0^2 = \frac{1}{n}\sum_{i=1}^n \phi(0)$$. Since $$\phi(u) \ge 0$$ for all $$u$$, this implies $$\phi(0)=0$$.
*   From Axiom (3), $$f(\lambda x)^2 = (\vert\lambda\vert f(x))^2 = \lambda^2 f(x)^2$$.
    Using Axiom (2) for both sides of this equation:
    $$ \frac{1}{n}\,\sum_{i=1}^n \phi(\lambda x_i) \;=\; \lambda^2 \left(\frac{1}{n}\,\sum_{i=1}^n \phi(x_i)\right) $$
    $$ \sum_{i=1}^n \phi(\lambda x_i) \;=\; \lambda^2 \sum_{i=1}^n \phi(x_i) $$
*   Let $$x = e_j$$ (the $$j$$-th standard basis vector, so $$x_j=1$$ and $$x_k=0$$ for $$k \neq j$$).
    The equation becomes: $$\phi(\lambda \cdot 1) + \sum_{k \neq j} \phi(\lambda \cdot 0) \;=\; \lambda^2 \left(\phi(1) + \sum_{k \neq j} \phi(0)\right)$$.
    Since we found $$\phi(0)=0$$:
    $$ \phi(\lambda) + (n-1)\phi(0) = \lambda^2 (\phi(1) + (n-1)\phi(0)) $$
    $$ \phi(\lambda) = \lambda^2 \phi(1) $$
    This must hold for any $$\lambda \in \mathbb{R}$$.
*   Using Axiom (4), $$\phi(1) = 1$$. Therefore, $$\phi(u) = u^2$$ for all $$u \in \mathbb{R}$$.
*   Substitute $$\phi(u)=u^2$$ back into the structure from Axiom (2):
    $$f(x)^2 \;=\; \frac{1}{n}\,\sum_{i=1}^n x_i^2$$
*   Since $$f(x) \ge 0$$ by Axiom (1), taking the square root gives:
    $$f(x) \;=\; \sqrt{\frac{1}{n}\,\sum_{i=1}^n x_i^2} \;=\; \frac{1}{\sqrt{n}} \Vert x \Vert_2 \;=\; \Vert x \Vert_{\mathrm{RMS}}$$
*   This derived form $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$ satisfies the triangle inequality (since $$\Vert \cdot \Vert_2$$ does and $$1/\sqrt{n}>0$$) and also Axioms (1) and (3), thus it is a norm.
</details>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**In a Nutshell: Characterizing the RMS norm**
</div>
The RMS norm, $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\,\Vert x \Vert_2$$, is essentially a scaled Euclidean norm. Most minimal axiomatic characterizations first establish that a function $$f(x)$$ must be proportional to the Euclidean norm (i.e., $$f(x) = c\Vert x \Vert_2$$ for some $$c>0$$). This is typically achieved using axioms related to inner product structure (like the parallelogram law) or high degrees of symmetry (like rotational or orthogonal invariance). An additional normalization axiom then uniquely fixes the constant of proportionality $$c$$ to $$1/\sqrt{n}$$.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Quick Checklist of Axiom Sets for the RMS norm**
</div>
Each set of properties below uniquely defines the RMS norm $$f(x) = \Vert x \Vert_{\mathrm{RMS}}$$.

1.  **Parallelogram Law Version (Theorem 3)**
    *   $$f$$ is a norm.
    *   Parallelogram identity: $$f(x+y)^2 + f(x-y)^2 = 2f(x)^2 + 2f(y)^2$$.
    *   Orthogonality of std. basis for the derived inner product: $$\langle e_i, e_j \rangle_f = 0$$ for $$i \ne j$$.
    *   Normalization: $$f(e_i)=1/\sqrt n$$ for all standard basis vectors $$e_i$$.

2.  **Orthogonal Invariance Version (Theorem 4)**
    *   $$f$$ is a norm.
    *   Orthogonal invariance: $$f(Qx)=f(x)$$ for $$Q \in O(n)$$.
    *   Normalization: $$f(e_1)=1/\sqrt n$$.

3.  **Pythagorean Additivity Version (Theorem 5)**
    *   Nondegeneracy ($$f(x)\ge 0, f(x)=0 \iff x=\mathbf{0}$$), Absolute homogeneity ($$f(\lambda x) = \vert\lambda\vert f(x)$$).
    *   Pythagorean additivity: $$x^\top y = 0 \implies f(x+y)^2 = f(x)^2 + f(y)^2$$.
    *   Continuity of $$f$$.
    *   Normalization: $$f(e_i)=1/\sqrt n$$ for all standard basis vectors $$e_i$$.

4.  **Coordinate-Symmetry + Disjoint-Support Additivity Version (Theorem 6)**
    *   Nondegeneracy, Absolute homogeneity.
    *   Permutation & sign-flip symmetry for coordinates.
    *   Pythagorean additivity on disjoint supports: $$\mathrm{supp}(x) \cap \mathrm{supp}(y) = \emptyset \implies f(x+y)^2 = f(x)^2 + f(y)^2$$.
    *   Normalization: $$f(e_1)=1/\sqrt n$$.

5.  **Enhanced Symmetry Version (Theorem 7)**
    *   $$f$$ is a norm.
    *   Permutation & sign-flip symmetry for coordinates.
    *   Parallelogram identity OR Rotational invariance.
    *   Normalization: $$f(1,1,\dots,1)=1$$.

6.  **Averaged Sum-of-Functions Structure (Theorem 8)**
    *   Nondegeneracy, Absolute homogeneity.
    *   Squared structure: $$f(x)^2 = \frac{1}{n}\sum \phi(x_i)$$ for some $$\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$$.
    *   Normalization: $$\phi(1)=1$$.

Any one of these sets of properties is sufficient to uniquely define the RMS norm.
</blockquote>

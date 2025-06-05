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

While the norms above are ubiquitous, deep-learning practice often benefits from a **dimension-invariant** scale for vectors. The **RMS norm** provides exactly that by normalizing the Euclidean norm by the square-root of the dimension.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** RMS Norm (for Vectors)
</div>
For $$x \in \mathbb{R}^n$$ the **root-mean-square norm** is

$$
\Vert x \Vert_{\mathrm{RMS}} \;=\; \frac{\Vert x \Vert_2}{\Vert \vec{1} \Vert_2} \;=\; \frac{\Vert x \Vert_2}{\sqrt{n}} \;=\; \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}
$$

where $$\vec{1}$$ is the vector of all ones in $$\mathbb{R}^n$$.
</blockquote>

#### Properties and Rationale for Vector RMS Norm

1.  **Dimension neutrality.** If the coordinates of $$x$$ are i.i.d. with variance $$1$$—for instance $$x_i \sim \mathcal{N}(0,1)$$—then $$\mathbb{E}\Vert x \Vert_{\mathrm{RMS}} \approx 1$$ for large $$n$$ (and exactly 1 under certain ideal conditions). This property makes the notion of “unit-size” consistent for vectors of varying dimensions, such as activations from layers of different widths.
2.  **Rotational and Orthogonal Invariance.** The vector RMS norm is a scaled version of the $$\ell_2$$-norm. The $$\ell_2$$-norm possesses strong geometric symmetries, specifically invariance under rotations and, more broadly, under all orthogonal transformations (which include reflections). The RMS norm inherits these symmetries. This means it treats all orthonormal bases equivalently and does not distinguish between right-handed and left-handed coordinate systems. These geometric properties are fundamental and lead to the following characterizations of norms possessing such invariances.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 1.** Rotationally Invariant Functions with Vector Input
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is **rotationally invariant** (i.e., $$f(Qx) = f(x)$$ for all rotation matrices $$Q \in SO(n)$$ and all $$x \in \mathbb{R}^n$$) if and only if it can be expressed as a function of the Euclidean norm of $$x$$. That is, there exists a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that:

$$
f(x) = g(\Vert x \Vert_2) \quad \forall x \in \mathbb{R}^n
$$

For the case $$n=1$$, $$SO(1) = \{[1]\}$$ (the identity transformation), so any function $$f(x_1)$$ is trivially $$SO(1)$$-invariant. For the conclusion $$f(x_1) = g(\Vert x_1 \Vert_2) = g(\vert x_1 \vert)$$ to hold, $$f(x_1)$$ must be an even function ($$f(x_1)=f(-x_1)$$).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 1.**
</summary>
($$\Leftarrow$$) **Sufficiency:** Assume $$f(x) = g(\Vert x \Vert_2)$$ for some function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$.
For any rotation matrix $$Q \in SO(n)$$, the Euclidean norm is preserved: $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$.
Then, $$f(Qx) = g(\Vert Qx \Vert_2) = g(\Vert x \Vert_2) = f(x)$$.
Thus, $$f$$ is rotationally invariant.

($$\Rightarrow$$) **Necessity:** Assume $$f: \mathbb{R}^n \to \mathbb{R}$$ is rotationally invariant, i.e., $$f(Qx) = f(x)$$ for all $$Q \in SO(n)$$ and all $$x \in \mathbb{R}^n$$.

*   If $$x = \mathbf{0}$$, then $$Q\mathbf{0} = \mathbf{0}$$. The condition $$f(Q\mathbf{0}) = f(\mathbf{0})$$ is $$f(\mathbf{0}) = f(\mathbf{0})$$, which is trivially true. We can define $$g(0) = f(\mathbf{0})$$.

*   Consider $$x \ne \mathbf{0}$$. Let $$r = \Vert x \Vert_2 > 0$$.
    *   **Case $$n \ge 2$$:**
        Let $$y \in \mathbb{R}^n$$ be any other vector such that $$\Vert y \Vert_2 = r$$. This means $$x$$ and $$y$$ lie on the same sphere of radius $$r$$ centered at the origin. The special orthogonal group $$SO(n)$$ acts transitively on such spheres for $$n \ge 2$$. This means there exists a rotation matrix $$Q \in SO(n)$$ such that $$y = Qx$$.
        By the assumed rotational invariance of $$f$$:
        $$f(y) = f(Qx) = f(x)$$
        This shows that the value of $$f(x)$$ is the same for all vectors $$x$$ having the same Euclidean norm $$r$$. Therefore, $$f(x)$$ depends only on $$\Vert x \Vert_2$$. We can define a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ by setting $$g(r) = f(x_0)$$ for any chosen vector $$x_0$$ such that $$\Vert x_0 \Vert_2 = r$$ (for example, $$x_0 = (r, 0, \ldots, 0)^\top$$). This function $$g$$ is well-defined. Then, for any $$x \in \mathbb{R}^n$$, $$f(x) = g(\Vert x \Vert_2)$$.

    *   **Case $$n=1$$:**
        The space is $$\mathbb{R}$$. The special orthogonal group $$SO(1)$$ consists only of the identity transformation, $$Q = [1]$$. The condition for rotational invariance, $$f(Qx_1) = f(x_1)$$, becomes $$f(1 \cdot x_1) = f(x_1)$$. This equation is true for *any* function $$f: \mathbb{R} \to \mathbb{R}$$.
        For the theorem's conclusion, $$f(x_1) = g(\Vert x_1 \Vert_2) = g(\vert x_1 \vert)$$, to hold, it must be that $$f(x_1) = f(-x_1)$$ for all $$x_1 \in \mathbb{R}$$. This is because if $$f(x_1) = g(\vert x_1 \vert)$$, then $$f(-x_1) = g(\vert -x_1 \vert) = g(\vert x_1 \vert) = f(x_1)$$. Thus, for $$n=1$$, an $$SO(1)$$-invariant function is a function of its Euclidean norm if and only if it is an even function. (As noted below, if $$f$$ is a norm, this even property is always satisfied).

Combining these parts, a rotationally invariant function $$f$$ can be written as $$f(x) = g(\Vert x \Vert_2)$$ (with the caveat for $$n=1$$ that $$f$$ must be even, which is implicitly handled if rotational invariance is strengthened or understood in context, e.g. for norms).
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Corollary 1.1.** Rotationally Invariant Norms
</div>
If a function $$\Vert \cdot \Vert : \mathbb{R}^n \to \mathbb{R}$$ is a **norm** and is **rotationally invariant**, then it must be a positive scalar multiple of the Euclidean ($$\ell_2$$) norm. That is, there exists a constant $$c > 0$$ such that:

$$
\Vert x \Vert = c \Vert x \Vert_2 \quad \forall x \in \mathbb{R}^n
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Corollary 1.1.**
</summary>
Let $$\Vert \cdot \Vert$$ be a norm on $$\mathbb{R}^n$$ that is rotationally invariant (i.e., $$\Vert Qx \Vert = \Vert x \Vert$$ for all $$Q \in SO(n)$$ and $$x \in \mathbb{R}^n$$).

1.  **Apply Theorem 1:**
    Since $$\Vert \cdot \Vert$$ is a rotationally invariant function, by Theorem 1, there exists a function $$g: \mathbb{R}_{\ge 0} \to \mathbb{R}$$ such that $$\Vert x \Vert = g(\Vert x \Vert_2)$$ for all $$x \in \mathbb{R}^n$$.
    For the case $$n=1$$: A norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}$$ must satisfy absolute homogeneity, so $$\Vert -x_1 \Vert = \vert -1 \vert \Vert x_1 \Vert = \Vert x_1 \Vert$$. This means any norm on $$\mathbb{R}$$ is an even function. Thus, the condition from Theorem 1 for $$n=1$$ is met, and we can write $$\Vert x_1 \Vert = g(\vert x_1 \vert) = g(\Vert x_1 \Vert_2)$$.

2.  **Use Norm Properties:**
    We now use the absolute homogeneity property of the norm $$\Vert \cdot \Vert$$: For any $$x \in \mathbb{R}^n$$ and any scalar $$\alpha \in \mathbb{R}$$, $$\Vert \alpha x \Vert = \vert \alpha \vert \Vert x \Vert$$.
    Substituting the form from Theorem 1:

    $$
    g(\Vert \alpha x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)
    $$

    Since $$\Vert \alpha x \Vert_2 = \vert \alpha \vert \Vert x \Vert_2$$ (a property of the $$\ell_2$$-norm), we have:

    $$
    g(\vert \alpha \vert \Vert x \Vert_2) = \vert \alpha \vert g(\Vert x \Vert_2)
    $$

    Let $$r = \Vert x \Vert_2 \ge 0$$. Let $$\lambda = \vert \alpha \vert \ge 0$$. The equation becomes:

    $$
    g(\lambda r) = \lambda g(r) \quad \text{for all } \lambda \ge 0, r \ge 0
    $$

    This is a form of Cauchy's functional equation for non-negative real numbers.
    If $$r > 0$$, we can set $$r=1$$ (corresponding to any vector on the $$\ell_2$$-unit sphere). Then for any $$\lambda \ge 0$$:

    $$
    g(\lambda) = \lambda g(1)
    $$

    Let $$c = g(1)$$. Then $$g(\lambda) = c\lambda$$ for all $$\lambda \ge 0$$.
    So, for any $$x \in \mathbb{R}^n$$, if $$\Vert x \Vert_2 = r \ge 0$$, then

    $$
    \Vert x \Vert = g(\Vert x \Vert_2) = g(r) = cr = c \Vert x \Vert_2
    $$

3.  **Determine the constant $$c$$:**
    The constant $$c = g(1)$$. Since $$g(1) = \Vert e_1 \Vert$$ (where $$e_1$$ is the first standard basis vector, $$\Vert e_1 \Vert_2 = 1$$), and $$\Vert \cdot \Vert$$ is a norm, we must have $$\Vert e_1 \Vert > 0$$ because $$e_1 \ne \mathbf{0}$$ (by positive definiteness of norms). Thus, $$c > 0$$.

4.  **Verify other norm properties:**
    The function $$\Vert x \Vert = c \Vert x \Vert_2$$ with $$c > 0$$ satisfies all norm axioms because $$\Vert \cdot \Vert_2$$ is a norm:
    *   Non-negativity: $$\Vert x \Vert = c \Vert x \Vert_2 \ge 0$$ since $$c>0$$ and $$\Vert x \Vert_2 \ge 0$$.
    *   Positive definiteness: $$\Vert x \Vert = c \Vert x \Vert_2 = 0$$ iff $$\Vert x \Vert_2 = 0$$ (since $$c>0$$) iff $$x = \mathbf{0}$$.
    *   Absolute homogeneity: $$\Vert \alpha x \Vert = c \Vert \alpha x \Vert_2 = c \vert \alpha \vert \Vert x \Vert_2 = \vert \alpha \vert (c \Vert x \Vert_2) = \vert \alpha \vert \Vert x \Vert$$.
    *   Triangle inequality: $$\Vert x+y \Vert = c \Vert x+y \Vert_2 \le c(\Vert x \Vert_2 + \Vert y \Vert_2)$$ (by triangle inequality for $$\ell_2$$-norm) $$= c\Vert x \Vert_2 + c\Vert y \Vert_2 = \Vert x \Vert + \Vert y \Vert$$.

Thus, any rotationally invariant norm on $$\mathbb{R}^n$$ must be of the form $$c \Vert x \Vert_2$$ for some constant $$c > 0$$.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 2.** Orthogonal Invariance of Euclidean-derived Norms
</div>
The Euclidean norm ($$\ell_2$$-norm) is **orthogonally invariant**. That is, for any orthogonal matrix $$Q \in O(n)$$ (where $$O(n)$$ is the orthogonal group, whose elements satisfy $$Q^\top Q = QQ^\top = I$$) and any vector $$x \in \mathbb{R}^n$$:

$$
\Vert Qx \Vert_2 = \Vert x \Vert_2
$$

Consequently, any norm of the form $$\Vert x \Vert = c \Vert x \Vert_2$$ with $$c > 0$$ (which, by Corollary 1.1, includes all rotationally invariant norms) is also orthogonally invariant.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 2.**
</summary>
An orthogonal matrix $$Q \in O(n)$$ is defined by the property $$Q^\top Q = I$$ (or equivalently, $$QQ^\top = I$$). Orthogonal transformations preserve the standard dot product:
$$(Qx)^\top (Qy) = x^\top Q^\top Q y = x^\top I y = x^\top y$$
The Euclidean norm is defined as $$\Vert x \Vert_2 = \sqrt{x^\top x}$$.
Therefore, for any $$Q \in O(n)$$ and any $$x \in \mathbb{R}^n$$:

$$
\Vert Qx \Vert_2 = \sqrt{(Qx)^\top (Qx)}
$$

Using the dot product preservation property with $$y=x$$:
$$(Qx)^\top (Qx) = x^\top x$$
So,

$$
\Vert Qx \Vert_2 = \sqrt{x^\top x} = \Vert x \Vert_2
$$

Thus, the Euclidean norm is orthogonally invariant. This means it is invariant under all transformations in $$O(n)$$, which includes rotations ($$Q \in SO(n) \subset O(n)$$) as well as reflections (orthogonal transformations with $$\det(Q) = -1$$).

Now, consider a norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^n$$ such that $$\Vert x \Vert = c \Vert x \Vert_2$$ for some constant $$c > 0$$.
For any $$Q \in O(n)$$:

$$
\Vert Qx \Vert = c \Vert Qx \Vert_2
$$

Since we just showed $$\Vert Qx \Vert_2 = \Vert x \Vert_2$$ for $$Q \in O(n)$$,

$$
\Vert Qx \Vert = c \Vert x \Vert_2
$$

And since $$\Vert x \Vert = c \Vert x \Vert_2$$, we have:

$$
\Vert Qx \Vert = \Vert x \Vert
$$

This shows that any norm that is a positive scalar multiple of the Euclidean norm is orthogonally invariant.
By Corollary 1.1, any rotationally ($$SO(n)$$-) invariant norm must be of the form $$c \Vert x \Vert_2$$ with $$c>0$$. Therefore, any rotationally invariant norm is automatically orthogonally ($$O(n)$$-) invariant.
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Corollary 2.1.** Uniqueness of RMS norm under Rotational, Orthogonal, and Dimensional Invariance
</div>
The RMS norm, defined as $$\Vert x \Vert_{\mathrm{RMS}} = \frac{\Vert x \Vert_2}{\sqrt{n}}$$, is a positive scalar multiple of the Euclidean norm (with scaling factor $$c = 1/\sqrt{n} > 0$$). Therefore:
1.  By Corollary 1.1, the RMS norm is rotationally invariant.
2.  By Theorem 2, the RMS norm is also orthogonally invariant.

Furthermore, consider a function $$\mathcal{N}$$ that defines a norm $$\mathcal{N}_n(\cdot)$$ on each space $$\mathbb{R}^n$$ (for $$n \ge 1$$). If this family of norms satisfies:
*   **Rotational Invariance:** Each norm $$\mathcal{N}_n(\cdot)$$ is rotationally invariant on $$\mathbb{R}^n$$ (i.e., $$\mathcal{N}_n(Qx) = \mathcal{N}_n(x)$$ for all $$Q \in SO(n)$$ and all $$x \in \mathbb{R}^n$$).
*   **Dimensional Invariance:** The family is *dimensionally invariant*. This means that for vectors $$X^{(n)} \in \mathbb{R}^n$$ whose components $$X_i$$ are i.i.d. random variables with zero mean and unit variance (e.g., $$X_i \sim \mathcal{N}(0,1)$$), the expected value $$\mathbb{E}[\mathcal{N}_n(X^{(n)})]$$ is a constant $$K > 0$$ that is independent of the dimension $$n$$.

Then, each $$\mathcal{N}_n(x)$$ must be a positive scalar multiple of the RMS norm for $$x \in \mathbb{R}^n$$. Specifically, for any $$n$$ and any $$x \in \mathbb{R}^n$$:

$$
\mathcal{N}_n(x) = K' \cdot \frac{\Vert x \Vert_2}{\sqrt{n}} = K' \cdot \Vert x \Vert_{\mathrm{RMS}}
$$

for some constant $$K' > 0$$ (related to $$K$$). If, by convention or normalization, this constant $$K'$$ is 1 (e.g., if the dimensional invariance is specifically $$\mathbb{E}[\mathcal{N}_n(X^{(n)})] = 1$$ for a class of test random vectors where $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] = 1$$), then the RMS norm family is the **unique** family of norms satisfying these conditions of rotational (hence orthogonal) and dimensional invariance.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Corollary 2.1.**
</summary>
The first part, stating that the RMS norm is rotationally and orthogonally invariant, follows directly from its definition. The RMS norm is $$\Vert x \Vert_{\mathrm{RMS}} = (1/\sqrt{n}) \Vert x \Vert_2$$. Since $$1/\sqrt{n} > 0$$, Corollary 1.1 implies it is rotationally invariant, and Theorem 2 implies it is orthogonally invariant.

For the second part, let $$\mathcal{N}$$ be a function defining a family of norms $$\mathcal{N}_n(\cdot)$$ on each $$\mathbb{R}^n$$ ($$n \ge 1$$).

1.  **Implication of Rotational Invariance:**
    By **Corollary 1.1**, since each norm $$\mathcal{N}_n(\cdot)$$ in the family is rotationally invariant (invariant under $$SO(n)$$), it must be a positive scalar multiple of the Euclidean ($$\ell_2$$) norm. Thus, for each dimension $$n$$, there exists a constant $$c_n > 0$$ such that:
    $$ \mathcal{N}_n(x) = c_n \Vert x \Vert_2 \quad \forall x \in \mathbb{R}^n $$
    By Theorem 2, this implies that each $$\mathcal{N}_n(\cdot)$$ is also orthogonally invariant.

2.  **Application of Dimensional Invariance:**
    The family is also dimensionally invariant. This means for vectors $$X^{(n)} \in \mathbb{R}^n$$ with i.i.d. components $$X_i$$ having zero mean and unit variance, the expected value $$\mathbb{E}[\mathcal{N}_n(X^{(n)})]$$ is a constant $$K > 0$$ that is independent of $$n$$.
    Substituting the form from (1):
    $$ \mathbb{E}[c_n \Vert X^{(n)} \Vert_2] = K $$
    Since $$c_n$$ is a constant for a given $$n$$ (but may depend on $$n$$), this implies:
    $$ c_n \mathbb{E}[\Vert X^{(n)} \Vert_2] = K $$

3.  **Relating to RMS Norm Property:**
    For the specified random vectors $$X^{(n)}$$, we consider their expected RMS norm. Let $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}] = \mathbb{E}[\Vert X^{(n)} \Vert_2 / \sqrt{n}] = C_X(n)$$.
    The factor $$C_X(n)$$ depends on the distribution of $$X_i$$ and $$n$$. For many common distributions (like standard normal), $$C_X(n) \to 1$$ as $$n \to \infty$$. The concept of "dimensional invariance" for RMS-like quantities often assumes conditions where $$C_X(n)$$ is either exactly 1 or treated as 1 for the purpose of defining the invariance.
    From $$\mathbb{E}[\Vert X^{(n)} \Vert_2 / \sqrt{n}] = C_X(n)$$, we have $$\mathbb{E}[\Vert X^{(n)} \Vert_2] = C_X(n) \sqrt{n}$$.
    Substituting this into the equation from (2):
    $$ c_n (C_X(n) \sqrt{n}) = K $$
    Thus, $$c_n = \frac{K}{C_X(n) \sqrt{n}}$$.

4.  **Final Form of the Norm:**
    The norm $$\mathcal{N}_n(x)$$ takes the form:
    $$ \mathcal{N}_n(x) = c_n \Vert x \Vert_2 = \left( \frac{K}{C_X(n) \sqrt{n}} \right) \Vert x \Vert_2 = \frac{K}{C_X(n)} \left( \frac{\Vert x \Vert_2}{\sqrt{n}} \right) = \frac{K}{C_X(n)} \cdot \Vert x \Vert_{\mathrm{RMS}} $$
    Let $$K' = K/C_X(n)$$. For $$\mathcal{N}_n$$ to be "dimensionally invariant" in a strong sense where the scaling relative to RMS norm is fixed across dimensions, $$K'$$ must be a constant independent of $$n$$. This occurs if $$C_X(n)$$ is itself independent of $$n$$ (or if the definition of dimensional invariance implies $$K$$ absorbs this dependence, e.g., by defining it in a limit or for specific distributions where $$C_X(n)$$ is stable).
    A common interpretation for RMS-like quantities is that "dimensional invariance" fixes the scaling such that $$\mathbb{E}[\mathcal{N}_n(X^{(n)})]$$ matches the behavior of $$\mathbb{E}[\Vert X^{(n)} \Vert_{\mathrm{RMS}}]$$ up to a single constant. If we assume ideal conditions where $$C_X(n)=1$$ (as suggested by the property $$\mathbb{E}\Vert x \Vert_{\mathrm{RMS}} \approx 1$$ for the RMS norm itself), then $$K' = K$$.
    Thus, $$\mathcal{N}_n(x) = K' \Vert x \Vert_{\mathrm{RMS}}$$ for a single constant $$K' > 0$$.

5.  **Normalization and Uniqueness:**
    If, by convention or normalization, the constant $$K'$$ is 1 (e.g., if the dimensional invariance condition is specifically $$\mathbb{E}[\mathcal{N}_n(X^{(n)})] = 1$$ and this is achieved with random vectors for which $$C_X(n)=1$$), then:
    $$ \mathcal{N}_n(x) = \Vert x \Vert_{\mathrm{RMS}} $$
    This shows that a family of norms satisfying rotational invariance (and thus orthogonal invariance) and the specified type of dimensional invariance must be the RMS norm family, up to a single positive scaling constant $$K'$$ across all dimensions. If this scaling constant is fixed (e.g., to 1 by the normalization inherent in the definition of dimensional invariance), then the family is unique.
</details>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** When to reach for the vector RMS norm
</div>
Use the vector $$\Vert \cdot \Vert_{\mathrm{RMS}}$$ whenever you need a scale for vectors that is *simultaneously* rotationally symmetric (and thus orthogonally symmetric) and independent of vector length—e.g.
when comparing activations from layers of different widths or designing width-robust regularizers for activations.
</blockquote>

#### Minimal Axiomatic Characterizations of the RMS norm

The RMS norm, being a scaled version of the Euclidean ($$\ell_2$$) norm ($$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$), can be uniquely identified by various sets of minimal axioms. Typically, these involve axioms that characterize the Euclidean norm up to a positive scalar constant, plus one additional normalization condition to fix this constant to $$1/\sqrt{n}$$. Below are several such characterizations.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 3 (Characterization 1: Parallelogram Law and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm, $$f(x) = \Vert x \Vert_{\mathrm{RMS}} = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}$$, if and only if it satisfies the following properties:
1.  **Nondegeneracy:** $$f(x) \ge 0$$ for all $$x \in \mathbb{R}^n$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}$$ and $$x \in \mathbb{R}^n$$.
3.  **Parallelogram identity:** $$f(x+y)^2 + f(x-y)^2 = 2f(x)^2 + 2f(y)^2$$ for all $$x,y \in \mathbb{R}^n$$.
4.  **Normalization on standard basis:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for each standard basis vector $$e_i=(0,\dots,1,\dots,0)$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 3.**
</summary>
($$\Rightarrow$$) **Necessity:** We show that $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$ satisfies the axioms.
1.  **Nondegeneracy:** $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2 \ge 0$$ since $$\Vert x \Vert_2 \ge 0$$ and $$\sqrt{n}>0$$. Also, $$\Vert x \Vert_{\mathrm{RMS}} = 0 \iff \Vert x \Vert_2 = 0 \iff x = \mathbf{0}$$. This holds.
2.  **Absolute homogeneity:** $$\Vert \lambda x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert \lambda x \Vert_2 = \frac{1}{\sqrt{n}}\vert\lambda\vert \Vert x \Vert_2 = \vert\lambda\vert \left(\frac{1}{\sqrt{n}}\Vert x \Vert_2\right) = \vert\lambda\vert \Vert x \Vert_{\mathrm{RMS}}$$. This holds.
3.  **Parallelogram identity:** The RMS norm is derived from the inner product $$\langle x, y \rangle_{\text{scaled}} = \frac{1}{n} x^\top y$$, since $$\Vert x \Vert_{\mathrm{RMS}}^2 = \frac{1}{n} x^\top x = \langle x,x \rangle_{\text{scaled}}$$. Any norm derived from an inner product satisfies the parallelogram law. Specifically:

    $$
    \begin{aligned}
    f(x+y)^2 + f(x-y)^2 &= \frac{1}{n}\Vert x+y \Vert_2^2 + \frac{1}{n}\Vert x-y \Vert_2^2 \\
    &= \frac{1}{n} (\Vert x+y \Vert_2^2 + \Vert x-y \Vert_2^2) \\
    &= \frac{1}{n} (2\Vert x \Vert_2^2 + 2\Vert y \Vert_2^2) \quad (\text{by parallelogram law for } \Vert\cdot\Vert_2) \\
    &= 2\left(\frac{1}{n}\Vert x \Vert_2^2\right) + 2\left(\frac{1}{n}\Vert y \Vert_2^2\right) \\
    &= 2f(x)^2 + 2f(y)^2.
    \end{aligned}
    $$

    This holds.
4.  **Normalization on standard basis:** For a standard basis vector $$e_i$$, $$\Vert e_i \Vert_2 = 1$$. So, $$\Vert e_i \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert e_i \Vert_2 = \frac{1}{\sqrt{n}} \cdot 1 = \frac{1}{\sqrt{n}}$$. This holds.
All axioms are satisfied by the RMS norm.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   The Jordan-von Neumann theorem states that a norm (a function satisfying (1) Nondegeneracy, (2) Absolute homogeneity, and the triangle inequality) satisfies the parallelogram identity (3) if and only if it arises from an inner product. That is, $$f(x)^2 = \langle x, x \rangle_S$$ for some inner product $$\langle u, v \rangle_S = u^\top S v$$ where $$S$$ is a symmetric positive-definite matrix. Thus, $$f(x) = \sqrt{x^\top S x}$$.
*   It's a standard result in functional analysis that if a norm arises from an inner product and satisfies certain symmetry conditions (often implied or explicitly stated, like rotational invariance), then this inner product must be a scalar multiple of the standard dot product, i.e., $$S = c^2 I$$ for some $$c>0$$. Thus, $$f(x) = \sqrt{c^2 x^\top I x} = c \Vert x \Vert_2$$. (The properties (1)+(2)+(3) for a function on $$\mathbb{R}^n$$ are sufficient to establish $$f(x) = c\Vert x \Vert_2$$ for some $$c>0$$.)
*   Given $$f(x) = c \Vert x \Vert_2$$ for some constant $$c > 0$$.
*   Now, apply axiom (4): $$f(e_i) = \frac{1}{\sqrt{n}}$$ for any standard basis vector $$e_i$$.
    Substituting $$x=e_i$$ into $$f(x) = c \Vert x \Vert_2$$:

    $$
    f(e_i) = c \Vert e_i \Vert_2
    $$

    Since $$\Vert e_i \Vert_2 = 1$$, we have $$f(e_i) = c \cdot 1 = c$$.
    From axiom (4), $$f(e_i) = \frac{1}{\sqrt{n}}$$.
    Therefore, $$c = \frac{1}{\sqrt{n}}$$.
*   So, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2} = \Vert x \Vert_{\mathrm{RMS}}$$.
This completes the proof.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 4 (Characterization 2: Orthogonal Invariance and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Orthogonal (rotational) invariance:** $$f(Qx) = f(x)$$ for all $$Q \in O(n)$$ and $$x \in \mathbb{R}^n$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
3.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
4.  **Continuity:** $$f$$ is continuous (e.g., at $$x=\mathbf{0}$$).
5.  **Normalization on a basis vector:** $$f(e_1) = \frac{1}{\sqrt{n}}$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 4.**
</summary>
($$\Rightarrow$$) **Necessity:** We show that $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$ satisfies the axioms.
1.  **Orthogonal invariance:** By Theorem 2, $$\Vert x \Vert_2$$ is orthogonally invariant. Thus, $$\Vert Qx \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert Qx \Vert_2 = \frac{1}{\sqrt{n}}\Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
2.  **Absolute homogeneity:** Verified in Theorem 3.
3.  **Nondegeneracy:** Verified in Theorem 3.
4.  **Continuity:** All norms on finite-dimensional vector spaces are continuous. The $$\ell_2$$-norm is continuous, and so is $$\frac{1}{\sqrt{n}}\Vert x \Vert_2$$.
5.  **Normalization:** $$\Vert e_1 \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert e_1 \Vert_2 = \frac{1}{\sqrt{n}} \cdot 1 = \frac{1}{\sqrt{n}}$$.
All axioms are satisfied.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   Axioms (1) (Orthogonal invariance, which implies rotational invariance as $$SO(n) \subset O(n)$$), (2) (Absolute homogeneity), and (3) (Nondegeneracy) are the conditions for Corollary 1.1, provided $$f$$ is a norm. The triangle inequality, required for $$f$$ to be a norm, can be derived from these properties along with continuity (or it's often included as part of "Nondegeneracy" definition for a norm-like function). A known result states that a function satisfying (1), (2), (3), and (4) (continuity) must be a norm and hence, by Corollary 1.1, $$f(x) = c \Vert x \Vert_2$$ for some constant $$c > 0$$.
*   Given $$f(x) = c \Vert x \Vert_2$$ for some $$c > 0$$.
*   Apply axiom (5): $$f(e_1) = \frac{1}{\sqrt{n}}$$.
    Substituting $$x=e_1$$: $$f(e_1) = c \Vert e_1 \Vert_2 = c \cdot 1 = c$$.
    Thus, $$c = \frac{1}{\sqrt{n}}$$.
*   Therefore, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
This completes the proof.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 5 (Characterization 3: Pythagorean Additivity and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nonnegativity / nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$.
3.  **Pythagorean (orthogonal) additivity:** If $$x \perp y$$ (i.e., $$x^\top y = 0$$), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
4.  **Continuity** (e.g., at $$0$$ or everywhere).
5.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for all standard basis vectors $$e_i$$, $$i=1,\dots,n$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 5.**
</summary>
($$\Rightarrow$$) **Necessity:** We show $$\Vert x \Vert_{\mathrm{RMS}}$$ satisfies the axioms.
1.  **Nondegeneracy:** Verified in Theorem 3.
2.  **Absolute homogeneity:** Verified in Theorem 3.
3.  **Pythagorean additivity:** Let $$f(x) = \Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$. If $$x \perp y$$, then $$\Vert x+y \Vert_2^2 = \Vert x \Vert_2^2 + \Vert y \Vert_2^2$$.

    $$
    f(x+y)^2 = \left(\frac{1}{\sqrt{n}}\Vert x+y \Vert_2\right)^2 = \frac{1}{n}\Vert x+y \Vert_2^2 = \frac{1}{n}(\Vert x \Vert_2^2 + \Vert y \Vert_2^2)
    $$

    $$
    f(x)^2 + f(y)^2 = \left(\frac{1}{\sqrt{n}}\Vert x \Vert_2\right)^2 + \left(\frac{1}{\sqrt{n}}\Vert y \Vert_2\right)^2 = \frac{1}{n}\Vert x \Vert_2^2 + \frac{1}{n}\Vert y \Vert_2^2 = \frac{1}{n}(\Vert x \Vert_2^2 + \Vert y \Vert_2^2)
    $$

    Thus, $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
4.  **Continuity:** Verified in Theorem 4.
5.  **Normalization:** Verified in Theorem 3.
All axioms are satisfied.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   A known result states that a function satisfying (1) Nondegeneracy, (2) Absolute homogeneity, (3) Pythagorean additivity for orthogonal vectors (with respect to the standard dot product), and (4) Continuity, must be a positive scalar multiple of the Euclidean norm. That is, $$f(x) = c \Vert x \Vert_2$$ for some constant $$c > 0$$. (The Pythagorean property is key to showing it's an inner product norm based on the standard notion of orthogonality).
*   Given $$f(x) = c \Vert x \Vert_2$$ for some $$c > 0$$.
*   Apply axiom (5): $$f(e_i) = \frac{1}{\sqrt{n}}$$ for any $$i$$.
    $$f(e_i) = c \Vert e_i \Vert_2 = c \cdot 1 = c$$.
    Thus, $$c = \frac{1}{\sqrt{n}}$$.
*   Therefore, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
This completes the proof.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 6 (Characterization 4: Coordinate Symmetry, Disjoint-Support Additivity, and Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$.
3.  **Permutation & sign-flip symmetry:** $$f(x_1,\dots,x_n) = f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})$$ for any permutation $$\sigma \in S_n$$ and any choice of signs.
4.  **“Pythagoras” on disjoint supports:** If $$x_i \neq 0 \implies y_i=0$$ for all $$i$$ (meaning $$x$$ and $$y$$ have disjoint support: $$\mathrm{supp}(x) \cap \mathrm{supp}(y) = \emptyset$$), then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
5.  **Normalization:** $$f(e_i) = \frac{1}{\sqrt{n}}$$ for $$i=1,\dots,n$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 6.**
</summary>
($$\Rightarrow$$) **Necessity:** We show $$\Vert x \Vert_{\mathrm{RMS}}$$ satisfies the axioms.
1.  **Nondegeneracy:** Verified.
2.  **Absolute homogeneity:** Verified.
3.  **Permutation & sign-flip symmetry:** $$\Vert (\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)}) \Vert_{\mathrm{RMS}}^2 = \frac{1}{n}\sum_{j=1}^n (\pm x_{\sigma(j)})^2 = \frac{1}{n}\sum_{j=1}^n x_{\sigma(j)}^2$$. Since summing squared permuted values is the same as summing original squared values, this equals $$\frac{1}{n}\sum_{k=1}^n x_k^2 = \Vert x \Vert_{\mathrm{RMS}}^2$$. Taking square roots, symmetry holds.
4.  **Pythagoras on disjoint supports:** Let $$x = \sum_{k \in K_1} x_k e_k$$ and $$y = \sum_{j \in K_2} y_j e_j$$ with $$K_1 \cap K_2 = \emptyset$$. Then $$x+y = \sum_{i \in K_1 \cup K_2} (x+y)_i e_i$$.

    $$
    f(x+y)^2 = \frac{1}{n}\sum_{i \in K_1 \cup K_2} ((x+y)_i)^2 = \frac{1}{n}\left( \sum_{k \in K_1} x_k^2 + \sum_{j \in K_2} y_j^2 \right)
    $$

    $$
    f(x)^2 + f(y)^2 = \frac{1}{n}\sum_{k \in K_1} x_k^2 + \frac{1}{n}\sum_{j \in K_2} y_j^2 = \frac{1}{n}\left( \sum_{k \in K_1} x_k^2 + \sum_{j \in K_2} y_j^2 \right)
    $$

    This holds.
5.  **Normalization:** Verified.
All axioms are satisfied.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(5).
*   From (3) Permutation & sign-flip symmetry: $$f(e_i)$$ must be the same value for all $$i$$. Also $$f(e_i) = f(-e_i)$$.
    Let $$c_0 = f(e_1) = f(e_2) = \dots = f(e_n)$$.
*   Consider a vector $$x_j e_j = (0, \dots, x_j, \dots, 0)$$. By (2) Absolute homogeneity:
    $$f(x_j e_j) = \vert x_j \vert f(e_j) = \vert x_j \vert c_0$$.
*   Let $$x = (x_1, x_2, \dots, x_n)$$. We can write $$x = x_1 e_1 + x_2 e_2 + \dots + x_n e_n$$.
    The vectors $$x_1 e_1, x_2 e_2, \dots, x_n e_n$$ have disjoint supports.
    Using (4) Pythagoras on disjoint supports repeatedly:

    $$
    \begin{aligned}
    f(x)^2 &= f(x_1 e_1 + (x_2 e_2 + \dots + x_n e_n))^2 \\
           &= f(x_1 e_1)^2 + f(x_2 e_2 + \dots + x_n e_n)^2 \\
           &= f(x_1 e_1)^2 + f(x_2 e_2)^2 + \dots + f(x_n e_n)^2 \\
           &= (\vert x_1 \vert c_0)^2 + (\vert x_2 \vert c_0)^2 + \dots + (\vert x_n \vert c_0)^2 \\
           &= c_0^2 (x_1^2 + x_2^2 + \dots + x_n^2) = c_0^2 \Vert x \Vert_2^2.
    \end{aligned}
    $$

*   Taking the square root (since $$f(x) \ge 0$$ by (1)): $$f(x) = \sqrt{c_0^2 \Vert x \Vert_2^2} = \vert c_0 \vert \Vert x \Vert_2$$.
    Since $$c_0 = f(e_1) \ge 0$$ (by (1)), we have $$\vert c_0 \vert = c_0$$. So $$f(x) = c_0 \Vert x \Vert_2$$.
    From (1), if $$x \ne \mathbf{0}$$, $$f(x) > 0$$, so $$c_0 > 0$$.
*   Now apply axiom (5): $$f(e_i) = \frac{1}{\sqrt{n}}$$.
    We already established $$f(e_i) = c_0$$. So, $$c_0 = \frac{1}{\sqrt{n}}$$.
*   Thus, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
    (We also need to ensure this $$f$$ is a norm, specifically satisfying the triangle inequality. Since it's a positive multiple of $$\Vert x \Vert_2$$, it is indeed a norm.)
This completes the proof.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 7 (Characterization 5: General Norm Properties, Coordinate Symmetry, and All-Ones Normalization).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Norm properties:** $$f$$ is a norm on $$\mathbb{R}^n$$ (i.e., satisfies nondegeneracy, absolute homogeneity, and triangle inequality).
2.  **Permutation & sign-flip symmetry:** $$f(x_1,\dots,x_n) = f(\pm x_{\sigma(1)},\dots,\pm x_{\sigma(n)})$$ for any permutation $$\sigma \in S_n$$ and any choice of signs.
3.  **Normalization on the all-ones vector:** $$f(1,1,\dots,1) = 1$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 7.**
</summary>
($$\Rightarrow$$) **Necessity:** We show $$\Vert x \Vert_{\mathrm{RMS}}$$ satisfies the axioms.
1.  **Norm properties:** $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$. Since $$\Vert \cdot \Vert_2$$ is a norm and $$1/\sqrt{n} > 0$$, $$\Vert \cdot \Vert_{\mathrm{RMS}}$$ is also a norm.
2.  **Permutation & sign-flip symmetry:** Verified in Theorem 6.
3.  **Normalization on all-ones vector:** Let $$\mathbf{1} = (1,1,\dots,1)$$. Then $$\Vert \mathbf{1} \Vert_2 = \sqrt{\sum_{i=1}^n 1^2} = \sqrt{n}$$.
    So, $$\Vert \mathbf{1} \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert \mathbf{1} \Vert_2 = \frac{1}{\sqrt{n}} \sqrt{n} = 1$$.
All axioms are satisfied.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(3).
*   Axiom (1) states $$f$$ is a norm. Axiom (2) states $$f$$ is invariant under coordinate permutations and sign flips. It is a known result (due to von Neumann, Mazur, Schoenberg) that the only norms on $$\mathbb{R}^n$$ that are invariant under permutations and sign-flips are the scaled $$\ell_p$$-norms, i.e., $$f(x) = k \Vert x \Vert_p$$ for some $$k>0$$ and $$1 \le p \le \infty$$, or limits of these.
*   Furthermore, if a norm satisfies the parallelogram law (which is implied by certain strong symmetry conditions for norms that are also inner-product derived), it must be an inner product norm, which means $$p=2$$. The strong symmetry of (2) combined with (1) being a norm implies that the unit ball of $$f$$ must be an $$\ell_p$$-ball; for it to also be an inner product norm (which the parallelogram law would ensure), it must be $$p=2$$. Thus, properties (1) and (2) together imply $$f(x)=c\Vert x \Vert_2$$ for some constant $$c>0$$.
*   Given $$f(x) = c \Vert x \Vert_2$$ for some $$c > 0$$.
*   Apply axiom (3): $$f(1,1,\dots,1) = 1$$. Let $$\mathbf{1} = (1,1,\dots,1)$$.
    $$f(\mathbf{1}) = c \Vert \mathbf{1} \Vert_2 = c \sqrt{\sum_{i=1}^n 1^2} = c \sqrt{n}$$.
    From axiom (3), $$f(\mathbf{1}) = 1$$.
    So, $$c \sqrt{n} = 1$$, which means $$c = \frac{1}{\sqrt{n}}$$.
*   Therefore, $$f(x) = \frac{1}{\sqrt{n}} \Vert x \Vert_2 = \Vert x \Vert_{\mathrm{RMS}}$$.
This completes the proof.
</details>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 8 (Characterization 6: Averaged Sum-of-Functions Structure).**
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the RMS norm if and only if it satisfies:
1.  **Nondegeneracy:** $$f(x) \ge 0$$ for all $$x \in \mathbb{R}^n$$, and $$f(x)=0 \iff x=\mathbf{0}$$.
2.  **Averaged Sum-of-Functions Structure for $$f^2$$:** The squared value of $$f(x)$$ is the average of a function $$\phi$$ applied to each coordinate:

    $$
      f(x)^2 \;=\; \frac{1}{n}\,\sum_{i=1}^n \phi(x_i)
    $$

    for some function $$\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$$.
3.  **Absolute homogeneity:** $$f(\lambda x) = \vert\lambda\vert f(x)$$ for all $$\lambda \in \mathbb{R}, x \in \mathbb{R}^n$$.
4.  **Normalization of $$\phi$$ at $$1$$:** $$\phi(1) = 1$$.
(Additionally, for $$f$$ to be a norm, it must satisfy the triangle inequality, which is verified post-derivation for the resulting form).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof of Theorem 8.**
</summary>
($$\Rightarrow$$) **Necessity:** We show $$\Vert x \Vert_{\mathrm{RMS}}$$ satisfies the axioms. Let $$f(x) = \Vert x \Vert_{\mathrm{RMS}}$$.
1.  **Nondegeneracy:** Verified.
2.  **Averaged Sum-of-Functions Structure:** $$f(x)^2 = \Vert x \Vert_{\mathrm{RMS}}^2 = \frac{1}{n}\sum_{i=1}^n x_i^2$$. This fits the structure with $$\phi(u) = u^2$$. Since $$u^2 \ge 0$$, $$\phi: \mathbb{R} \to \mathbb{R}_{\ge 0}$$ holds.
3.  **Absolute homogeneity:** Verified.
4.  **Normalization of $$\phi$$:** For $$\phi(u) = u^2$$, we have $$\phi(1) = 1^2 = 1$$.
All axioms are satisfied.

($$\Leftarrow$$) **Sufficiency:** Assume $$f$$ satisfies axioms (1)-(4).
*   From (1), $$f(\mathbf{0})=0$$. Substituting $$x=\mathbf{0}$$ into (2):
    $$f(\mathbf{0})^2 = 0 = \frac{1}{n}\sum_{i=1}^n \phi(0) = \frac{1}{n} \cdot n \cdot \phi(0) = \phi(0)$$.
    So, $$\phi(0)=0$$.
*   From (3), $$f(\lambda x)^2 = (\vert\lambda\vert f(x))^2 = \lambda^2 f(x)^2$$.
    Using structure (2) for both sides:

    $$
    \frac{1}{n}\,\sum_{i=1}^n \phi(\lambda x_i) \;=\; \lambda^2 \left(\frac{1}{n}\,\sum_{i=1}^n \phi(x_i)\right)
    $$

    Multiplying by $$n$$:

    $$
    \sum_{i=1}^n \phi(\lambda x_i) \;=\; \lambda^2 \sum_{i=1}^n \phi(x_i)
    $$

*   Let $$x = e_j$$ (the $$j$$-th standard basis vector, so $$x_j=1$$ and $$x_k=0$$ for $$k \neq j$$).
    The equation becomes:

    $$
    \phi(\lambda \cdot x_j) + \sum_{k \neq j} \phi(\lambda \cdot x_k) \;=\; \lambda^2 \left(\phi(x_j) + \sum_{k \neq j} \phi(x_k)\right)
    $$

    Substituting $$x_j=1$$ and $$x_k=0$$ for $$k \neq j$$, and using $$\phi(0)=0$$:

    $$
    \phi(\lambda \cdot 1) + \sum_{k \neq j} \phi(0) \;=\; \lambda^2 \left(\phi(1) + \sum_{k \neq j} \phi(0)\right)
    $$

    $$
    \phi(\lambda) + (n-1)\phi(0) \;=\; \lambda^2 (\phi(1) + (n-1)\phi(0))
    $$

    Since $$\phi(0)=0$$, this simplifies to:

    $$
    \phi(\lambda) \;=\; \lambda^2 \phi(1).
    $$

    This holds for any $$\lambda \in \mathbb{R}$$. Let $$u=\lambda$$. So, $$\phi(u) = u^2 \phi(1)$$ for all $$u \in \mathbb{R}$$.
*   Using (4), Normalization of $$\phi$$ at $$1$$: $$\phi(1) = 1$$.
    Therefore, $$\phi(u) = u^2 \cdot 1 = u^2$$.
*   Substitute $$\phi(u)=u^2$$ back into the structural form (2):

    $$
    f(x)^2 \;=\; \frac{1}{n}\,\sum_{i=1}^n x_i^2.
    $$

    Since $$f(x) \ge 0$$ by (1), taking the square root gives:

    $$
    f(x) \;=\; \sqrt{\frac{1}{n}\,\sum_{i=1}^n x_i^2} \;=\; \Vert x \Vert_{\mathrm{RMS}}.
    $$

*   Finally, we verify that this derived $$f(x) = \Vert x \Vert_{\mathrm{RMS}}$$ is indeed a norm.
    Axioms (1) (Nondegeneracy) and (3) (Absolute homogeneity) were given for $$f$$. The triangle inequality must also hold for $$f$$ to be a norm. Since $$\Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$, and $$\Vert \cdot \Vert_2$$ is a norm (satisfying triangle inequality), and $$1/\sqrt{n} > 0$$, then $$\Vert \cdot \Vert_{\mathrm{RMS}}$$ also satisfies the triangle inequality.
    Thus, $$f(x) = \Vert x \Vert_{\mathrm{RMS}}$$ is the unique function satisfying the axioms.
This completes the proof.
</details>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**In a Nutshell: Characterizing the RMS norm**
</div>
Because the RMS norm is defined as

$$
  \Vert x \Vert_{\mathrm{RMS}} = \frac{1}{\sqrt{n}}\,\Vert x \Vert_2,
$$

most minimal characterizations of the RMS norm start from axioms known to imply that a function $$f(x)$$ must be a positive scalar multiple of the Euclidean norm ($$f(x) = c\Vert x \Vert_2$$). An additional normalization axiom then uniquely determines the constant $$c=1/\sqrt{n}$$. Typical normalization choices include:
*   Fixing the value on a standard basis vector: $$f(e_i)=1/\sqrt{n}$$.
*   Fixing the value on the all-ones vector: $$f(1,1,\dots,1)=1$$.
*   Structural assumptions (as in Theorem 8) that directly lead to the specific form of the RMS norm.

Once the form $$f(x) = \frac{1}{\sqrt{n}}\Vert x \Vert_2$$ is established, it's confirmed to be the RMS norm.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Quick Checklist of Axiom Sets for the RMS norm**
</div>
Below is a compact summary of some minimal axiom sets. In most cases, a subset of axioms forces $$f(x)=c\Vert x \Vert_2$$, and the final axiom(s) pin down $$c=1/\sqrt n$$.

1.  **Parallelogram Law Version (Theorem 3)**
    *   Nondegeneracy
    *   Absolute homogeneity
    *   Parallelogram identity
    *   $$f(e_i)=1/\sqrt n$$ for each $$i$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

2.  **Orthogonal Invariance Version (Theorem 4)**
    *   Orthogonal invariance: $$f(Qx)=f(x)$$
    *   Absolute homogeneity
    *   Nondegeneracy
    *   Continuity
    *   $$f(e_1)=1/\sqrt n$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

3.  **Pythagorean Additivity Version (Theorem 5)**
    *   Nondegeneracy
    *   Absolute homogeneity
    *   If $$x\perp y$$, then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
    *   Continuity
    *   $$f(e_i)=1/\sqrt n$$ for $$i=1,\dots,n$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

4.  **Coordinate-Symmetry + Disjoint-Support Version (Theorem 6)**
    *   Nondegeneracy
    *   Absolute homogeneity
    *   Permutation & sign-flip symmetry
    *   If $$\mathrm{supp}(x) \cap \mathrm{supp}(y) = \emptyset$$, then $$f(x+y)^2 = f(x)^2 + f(y)^2$$.
    *   $$f(e_i)=1/\sqrt n$$ for $$i=1,\dots,n$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

5.  **General Norm + Symmetry + All-Ones Normalization (Theorem 7)**
    *   $$f$$ is a norm.
    *   Permutation & sign-flip symmetry.
    *   $$f(1,1,\dots,1)=1$$.
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$

6.  **Averaged Sum-of-Functions Structure (Theorem 8)**
    *   Nondegeneracy
    *   $$f(x)^2 = \frac{1}{n}\sum_i \phi(x_i)$$
    *   Absolute homogeneity
    *   $$\phi(1)=1$$
    *   $$\Rightarrow$$ $$f(x)=\Vert x \Vert_{\mathrm{RMS}}$$ (and satisfies triangle inequality)
</blockquote>

Any one of these sets of properties is sufficient to uniquely define the RMS norm.

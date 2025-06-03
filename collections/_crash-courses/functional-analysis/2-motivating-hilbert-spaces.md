---
title: "Motivating Hilbert Spaces: Encoding Geometry"
date: 2025-05-29 09:00 -0400 # Adjusted date
course_index: 2
mermaid: false # Can be set to true if diagrams are added later
description: Why generalizing the dot product to function spaces (hello, inner product!) and demanding completeness leads to the powerful concept of Hilbert spaces, essential for geometry in infinite dimensions.
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Functional Analysis
- Hilbert Spaces
- Inner Product Spaces
- Completeness
- L2 Spaces
- Fourier Analysis
- Orthogonality
- Crash Course
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
  symbol; use \vert and \Vert.

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

Welcome back to our crash course on Functional Analysis! In the first post, we established that functions can be viewed as vectors in infinite-dimensional spaces and introduced the fundamental distinction between "kets" (vectors) and "bras" (linear functionals). Now, we'll take this idea further by asking: how can we imbue these function spaces with geometric structure?

## 1. Introduction: Beyond "Just" Vector Spaces

In our previous discussion, we saw that collections of functions (like polynomials or continuous functions on an interval) satisfy the axioms of a vector space: we can add functions and scale them by constants. This is a powerful realization, suggesting we can transfer our knowledge from familiar finite-dimensional vector spaces like $$\mathbb{R}^n$$ to these more abstract settings.

However, knowing that functions form a vector space is only part of the story. In $$\mathbb{R}^n$$, we have more than just vector addition and scalar multiplication. We have the **dot product**, which gives us crucial geometric notions:
*   **Length (Norm):** How "long" is a vector?
*   **Distance:** How "far apart" are two vectors?
*   **Angles & Orthogonality:** What is the angle between two vectors? When are they "perpendicular"?

If we want to reason about functions in a similar geometric way – to ask "how big is this function?", "how similar are these two functions?", or "is this function 'orthogonal' to that one?" – we need to generalize the concept of a dot product to function spaces. This journey will lead us to **inner products** and ultimately to **Hilbert spaces**, which are central to many areas of mathematics, physics, and machine learning.

## 2. The Inner Product: A "Dot Product" for Functions

### Intuition from $$\mathbb{R}^n$$
For two vectors $$\mathbf{x} = (x_1, \dots, x_n)$$ and $$\mathbf{y} = (y_1, \dots, y_n)$$ in $$\mathbb{R}^n$$, their dot product is:

$$
\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^n x_i y_i
$$

This sum pairs corresponding components of the vectors, multiplies them, and sums the results.

### Generalizing to Functions: The Road to the $$L_2$$ Inner Product
How can we extend this idea to functions, say, real-valued functions $$f(t)$$ and $$g(t)$$ defined on an interval $$[a,b]$$? We can think of a function as a "vector" with infinitely many components, indexed by $$t \in [a,b]$$.

1.  **Discretize and Approximate with Step Functions:**
    To make the sum-of-products idea tractable, let's first approximate $$f(t)$$ and $$g(t)$$ using simpler functions. Divide the interval $$[a,b]$$ into $$N$$ small subintervals. For simplicity, let each subinterval $$I_k = [t_{k-1}, t_k]$$ have width $$\Delta t_k = t_k - t_{k-1}$$. We can pick a sample point $$t_k^\ast  \in I_k$$ (e.g., the midpoint or left endpoint).
    Now, define two **step functions**, $$f_N(t)$$ and $$g_N(t)$$, that approximate $$f(t)$$ and $$g(t)$$:
    *   $$f_N(t) = f(t_k^\ast )$$ for all $$t \in I_k$$
    *   $$g_N(t) = g(t_k^\ast )$$ for all $$t \in I_k$$
    Each step function is constant on each subinterval.

2.  **An "Inner Product" for Step Functions:**
    What would be a natural inner product for these step functions $$f_N$$ and $$g_N$$? If we were to simply take the values $$(f(t_1^\ast ), \dots, f(t_N^\ast ))$$ and $$(g(t_1^\ast ), \dots, g(t_N^\ast ))$$ as vectors in $$\mathbb{R}^N$$, their dot product would be $$\sum_{k=1}^N f(t_k^\ast ) g(t_k^\ast )$$. However, this doesn't account for the fact that these values represent the function's behavior over intervals of potentially varying lengths $$\Delta t_k$$.
    A more appropriate generalization of the sum $$ \sum x_i y_i $$ would be to "sum" the products $$f_N(t)g_N(t)$$ across the entire interval $$[a,b]$$, where each product $$f(t_k^\ast )g(t_k^\ast )$$ is weighted by the length of the subinterval $$\Delta t_k$$ over which it applies. This leads to the sum:

    $$
    \sum_{k=1}^N f(t_k^\ast )g(t_k^\ast ) \Delta t_k
    $$

    This sum can also be seen as the integral of the product of our step function approximations: $$\int_a^b f_N(t)g_N(t)dt = \sum_{k=1}^N f(t_k^\ast )g(t_k^\ast ) \Delta t_k$$. This is precisely a **Riemann sum** for the integral of $$f(t)g(t)$$ (assuming $$f$$ and $$g$$ are Riemann integrable).

3.  **Taking the Limit:**
    As we make our approximation finer by increasing the number of subintervals $$N \to \infty$$ (and ensuring the maximum $$\Delta t_k \to 0$$), our step functions $$f_N(t)$$ and $$g_N(t)$$ should (under suitable conditions on $$f$$ and $$g$$, like continuity or Riemann integrability) converge to $$f(t)$$ and $$g(t)$$, respectively. Correspondingly, the Riemann sum converges to the definite integral:

    $$
    \lim_{N \to \infty} \sum_{k=1}^N f(t_k^\ast )g(t_k^\ast ) \Delta t_k = \int_a^b f(t)g(t)dt
    $$

This integral, $$\int_a^b f(t)g(t)dt$$, emerges as the natural extension of the Euclidean dot product to real-valued functions.
For **complex-valued functions**, to ensure that the "length squared" of a function $$f$$ (i.e., its inner product with itself) is real and non-negative, we introduce a complex conjugate on the first function in the product. This leads to the standard definition of the **$$L_2$$ inner product**:

$$
\langle f \vert g \rangle_{L_2} = \int_a^b \overline{f(t)} g(t) dt
$$

This expression satisfies $$\langle f \vert f \rangle_{L_2} = \int_a^b \overline{f(t)}f(t)dt = \int_a^b \vert f(t) \vert^2 dt \ge 0$$.
This integral motivates the formal definition of an inner product. We'll use the bra-ket notation $$\langle f \vert g \rangle$$ generally to denote the inner product of $$f$$ and $$g$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 2.1: Inner Product**
</div>
An **inner product** on a vector space $$V$$ over a field $$\mathbb{F}$$ (where $$\mathbb{F}$$ is typically $$\mathbb{R}$$ or $$\mathbb{C}$$) is a function $$\langle \cdot \vert \cdot \rangle : V \times V \to \mathbb{F}$$ that associates any two kets $$\vert f \rangle, \vert g \rangle \in V$$ with a scalar $$\langle f \vert g \rangle \in \mathbb{F}$$, satisfying for all kets $$\vert f \rangle, \vert g \rangle, \vert h \rangle \in V$$ and scalars $$\alpha, \beta \in \mathbb{F}$$:
1.  **Conjugate Symmetry (or Symmetry for real spaces):** $$\langle f \vert g \rangle = \overline{\langle g \vert f \rangle}$$. (For real spaces, this is just $$\langle f \vert g \rangle = \langle g \vert f \rangle$$).
2.  **Linearity in the second argument (the "ket"):** $$\langle f \vert (\alpha \vert g \rangle + \beta \vert h \rangle) \rangle = \alpha \langle f \vert g \rangle + \beta \langle f \vert h \rangle$$.
    (This, combined with conjugate symmetry, implies *conjugate-linearity* in the first argument (the "bra"): $$\langle (\alpha \vert f \rangle + \beta \vert g \rangle) \vert h \rangle = \bar{\alpha} \langle f \vert h \rangle + \bar{\beta} \langle g \vert h \rangle$$).
3.  **Positive-definiteness:** $$\langle f \vert f \rangle \ge 0$$, and $$\langle f \vert f \rangle = 0 \iff \vert f \rangle = \vert \mathbf{0} \rangle$$ (the zero ket).

A vector space equipped with an inner product is called an **inner product space**.
</blockquote>

### Example: The $$L_2$$ Inner Product (Formalized)
As derived from our Riemann sum analogy, one of the most common inner products for functions is the **$$L_2$$ inner product**. For (complex-valued, typically square-integrable) functions $$f, g$$ defined on an interval $$[a,b]$$ (or more generally, on a measure space $$(\Omega, \Sigma, \mu)$$), it is formally defined as:

$$
\langle f \vert g \rangle = \int_a^b \overline{f(x)} g(x) dx
$$

(Or $$\int_\Omega \overline{f(x)} g(x) d\mu(x)$$ for a general measure space with measure $$\mu$$).

As noted, the complex conjugate $$\overline{f(x)}$$ on the first function ensures that $$\langle f \vert f \rangle = \int_a^b \overline{f(x)}f(x)dx = \int_a^b \vert f(x) \vert^2 dx \ge 0$$, satisfying the positive-definiteness axiom. For real-valued functions, this simplifies to the already familiar $$\int_a^b f(x)g(x)dx$$. The space of functions for which this integral of the squared magnitude, $$\int_a^b \vert f(x) \vert^2 dx$$, is finite is called the **$$L_2$$ space**, denoted $$L_2([a,b])$$ (or $$L_2(\Omega, \mu)$$). Technically, $$L_2$$ spaces consist of equivalence classes of functions that differ only on sets of measure zero.

## 3. Geometric Toolkit from the Inner Product

Once an inner product is defined on a vector space, a whole suite of geometric concepts becomes available.

### Induced Norm (Length/Magnitude)
The inner product naturally defines a **norm**, which measures the "length" or "magnitude" of a vector (or function):

$$
\Vert f \Vert = \sqrt{\langle f \vert f \rangle}
$$

For the $$L_2$$ inner product, this gives the $$L_2$$-norm:

$$
\Vert f \Vert_2 = \left( \int_a^b \vert f(x) \vert^2 dx \right)^{1/2}
$$

This norm often represents some form of "energy" or "root mean square (RMS) value" of the function/signal.

### Metric (Distance)
The norm, in turn, defines a **metric** or distance function:

$$
d(f,g) = \Vert f - g \Vert = \sqrt{\langle f-g \vert f-g \rangle}
$$

This quantifies how "far apart" or "dissimilar" two functions $$f$$ and $$g$$ are in the sense defined by the inner product.

### Angles and Orthogonality
The inner product allows us to define angles, just like the dot product. The **Cauchy-Schwarz Inequality** is fundamental here:

$$
\vert \langle f \vert g \rangle \vert \le \Vert f \Vert \Vert g \Vert
$$

This inequality guarantees that for real-valued functions, $$-1 \le \frac{\langle f \vert g \rangle}{\Vert f \Vert \Vert g \Vert} \le 1$$, allowing us to define the angle $$\theta$$ between $$f$$ and $$g$$ via $$\cos \theta = \frac{\langle f \vert g \rangle}{\Vert f \Vert \Vert g \Vert}$$.

Most importantly, it gives us the concept of **orthogonality**:
Two kets (functions) $$\vert f \rangle$$ and $$\vert g \rangle$$ are **orthogonal** if their inner product is zero:

$$
\langle f \vert g \rangle = 0
$$

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">**Example: Orthogonal Sines**</div>
Consider real-valued functions on the interval $$[-\pi, \pi]$$ with the inner product $$\langle f \vert g \rangle = \int_{-\pi}^{\pi} f(x)g(x)dx$$.
Let $$\vert f_m \rangle$$ represent the function $$f_m(x) = \sin(mx)$$ and $$\vert f_n \rangle$$ represent $$f_n(x) = \sin(nx)$$, for positive integers $$m, n$$.

Their inner product is:

$$
\langle f_m \vert f_n \rangle = \int_{-\pi}^{\pi} \sin(mx)\sin(nx)dx
$$

Using trigonometric identities, one can show:
*   If $$m \neq n$$, then $$\langle f_m \vert f_n \rangle = 0$$. The functions are orthogonal.
*   If $$m = n$$, then $$\langle f_n \vert f_n \rangle = \Vert f_n \Vert^2 = \int_{-\pi}^{\pi} \sin^2(nx)dx = \pi$$.

This orthogonality is a cornerstone of Fourier series, allowing us to decompose complex periodic functions into simpler sinusoidal components.
</blockquote>

### Projections
Just as in $$\mathbb{R}^n$$, we can project one function onto another (or onto a subspace spanned by other functions). If $$\Vert f \Vert \neq 0$$, the **projection** of a function $$\vert g \rangle$$ onto the "direction" defined by $$\vert f \rangle$$ is:

$$
\text{proj}_{\vert f \rangle} \vert g \rangle = \frac{\langle f \vert g \rangle}{\langle f \vert f \rangle} \vert f \rangle = \frac{\langle f \vert g \rangle}{\Vert f \Vert^2} \vert f \rangle
$$

This gives the "component" of $$\vert g \rangle$$ that lies along $$\vert f \rangle$$. Projections are fundamental for approximation theory and for constructing orthogonal bases. For example, if we have an orthogonal set of functions $$\{\vert \phi_k \rangle\}$$, the best approximation of a function $$\vert f \rangle$$ in the subspace spanned by $$\{\vert \phi_k \rangle\}$$ is given by the sum of its projections onto each $$\vert \phi_k \rangle$$:

$$
\vert f_{\text{approx}} \rangle = \sum_k \frac{\langle \phi_k \vert f \rangle}{\Vert \phi_k \Vert^2} \vert \phi_k \rangle
$$

### Projections and Interpolation Formulas: A Duality Perspective

The idea of projecting a function onto a subspace spanned by basis functions has deep connections to how various interpolation formulas are constructed. The key is often finding a set of basis functions $$\{\vert \phi_i \rangle\}$$ for our approximating functions (e.g., polynomials of degree $$n$$) and a set of linear functionals $$\{\langle \Lambda_j \vert\}$$ (representing evaluation at points, derivatives, etc.) such that they form a **biorthogonal system**.

<details class="details-block" markdown="1">
<summary markdown="1">
**Biorthogonality and Interpolation**
</summary>

Let $$V$$ be a vector space of functions (e.g., polynomials of degree at most $$n$$, $$\mathcal{P}_n$$).
Suppose we have:
1.  A basis for $$V$$: $$\{\vert \phi_0 \rangle, \vert \phi_1 \rangle, \dots, \vert \phi_n \rangle\}$$.
2.  A set of $$n+1$$ linearly independent linear functionals on $$V$$: $$\{\langle \Lambda_0 \vert, \langle \Lambda_1 \vert, \dots, \langle \Lambda_n \vert\}$$. These functionals define our interpolation conditions. For example, $$\langle \Lambda_j \vert f \rangle = f(x_j)$$ (point evaluation) or $$\langle \Lambda_j \vert f \rangle = f^{(j)}(x_0)$$ (derivative evaluation).

We say that the set of basis functions $$\{\vert \phi_i \rangle\}$$ and the set of functionals $$\{\langle \Lambda_j \vert\}$$ are **biorthogonal** if:

$$
\langle \Lambda_j \vert \phi_i \rangle = \delta_{ji}
$$

where $$\delta_{ji}$$ is the Kronecker delta.

If such a biorthogonal system exists, then any function $$\vert P \rangle \in V$$ can be expanded as:

$$
\vert P \rangle = \sum_{i=0}^n \langle \Lambda_i \vert P \rangle \vert \phi_i \rangle
$$

**Proof Sketch:** Let $$\vert Q \rangle = \sum_{i=0}^n \langle \Lambda_i \vert P \rangle \vert \phi_i \rangle$$. Apply any functional $$\langle \Lambda_j \vert$$ to $$\vert Q \rangle$$:

$$
\langle \Lambda_j \vert Q \rangle = \sum_{i=0}^n \langle \Lambda_i \vert P \rangle \langle \Lambda_j \vert \phi_i \rangle = \sum_{i=0}^n \langle \Lambda_i \vert P \rangle \delta_{ji} = \langle \Lambda_j \vert P \rangle
$$

Since $$\langle \Lambda_j \vert (P-Q) \rangle = 0$$ for all $$j$$, and the $$\langle \Lambda_j \vert$$ form a basis for the dual space (or are sufficient to uniquely determine a polynomial of degree $$n$$), it implies $$\vert P \rangle - \vert Q \rangle = \vert \mathbf{0} \rangle$$, so $$\vert P \rangle = \vert Q \rangle$$.

**Application to Interpolation:**
Suppose we want to find an interpolating function $$\vert P \rangle \in V$$ that satisfies $$n+1$$ conditions $$\langle \Lambda_j \vert P \rangle = y_j$$ for given values $$y_j$$. If we use the basis $$\{\vert \phi_i \rangle\}$$ that is biorthogonal to $$\{\langle \Lambda_j \vert\}$$, then the interpolating function is simply:

$$
\vert P \rangle = \sum_{j=0}^n y_j \vert \phi_j \rangle
$$

The coefficients of the expansion are directly the target values $$y_j$$! This "derives" the form of many interpolation formulas once the appropriate biorthogonal basis functions $$\vert \phi_j \rangle$$ are identified or constructed for the given set of functionals $$\langle \Lambda_j \vert$$.

Let's see some examples:

**1. Lagrange Interpolation:**
   *   **Functionals:** Point evaluations at distinct points $$x_0, x_1, \dots, x_n$$. So, $$\langle \Lambda_j \vert P \rangle = P(x_j)$$.
   *   **Target Conditions:** We want $$\langle \Lambda_j \vert P \rangle = y_j$$, i.e., $$P(x_j) = y_j$$.
   *   **Biorthogonal Basis Functions (Lagrange Polynomials):** We need to find polynomials $$\vert L_i \rangle \equiv L_i(x)$$ in $$\mathcal{P}_n$$ such that $$\langle \Lambda_j \vert L_i \rangle = L_i(x_j) = \delta_{ji}$$.
       The Lagrange basis polynomial $$L_i(x)$$ is explicitly constructed as:

       $$
       L_i(x) = \prod_{k=0, k \neq i}^n \frac{x-x_k}{x_i-x_k}
       $$

       You can easily verify that $$L_i(x_i)=1$$ and $$L_i(x_j)=0$$ for $$j \neq i$$.
   *   **Interpolation Formula:** Using the general result with $$y_j$$ as the target values for $$\langle \Lambda_j \vert P \rangle$$, and $$\vert \phi_j \rangle = \vert L_j \rangle$$:

       $$
       P(x) = \sum_{j=0}^n y_j L_j(x)
       $$

       This is the familiar Lagrange interpolation formula, derived by finding the basis dual to point evaluation.

**2. Taylor Series (Polynomial Approximation):**
   *   **Functionals:** Evaluation of derivatives at a single point $$x_0$$. Let $$\langle \Lambda_j \vert P \rangle = P^{(j)}(x_0)$$ (the $$j$$-th derivative at $$x_0$$).
   *   **Target Conditions:** We want $$P^{(j)}(x_0) = f^{(j)}(x_0)$$ for some target function $$f$$. So, $$y_j = f^{(j)}(x_0)$$.
   *   **Biorthogonal Basis Functions:** We need polynomials $$\vert \phi_i \rangle \equiv \phi_i(x)$$ such that $$\langle \Lambda_j \vert \phi_i \rangle = \phi_i^{(j)}(x_0) = \delta_{ji}$$.
       The functions $$\phi_i(x) = \frac{(x-x_0)^i}{i!}$$ satisfy this:
       $$ \phi_i^{(j)}(x_0) = \frac{d^j}{dx^j} \left( \frac{(x-x_0)^i}{i!} \right) \bigg\vert _{x=x_0} = \delta_{ji} $$
   *   **Interpolation Formula (Taylor Polynomial):**

       $$
       P(x) = \sum_{j=0}^n f^{(j)}(x_0) \frac{(x-x_0)^j}{j!}
       $$

       This is the Taylor polynomial of degree $$n$$ for $$f$$ around $$x_0$$.

**3. Newton's Divided Difference Formula:**
   This formula takes the form $$P(x) = \sum_{i=0}^n f[x_0, \dots, x_i] \pi_i(x)$$, where $$\pi_i(x) = \prod_{k=0}^{i-1} (x-x_k)$$ are the Newton basis polynomials (with $$\pi_0(x)=1$$), and $$f[x_0, \dots, x_i]$$ are the divided differences.
   The functionals implicitly at play here are a bit more complex. The matrix formed by $$\Lambda_j(\pi_i) = \pi_i(x_j)$$ is lower triangular, which makes solving for the coefficients (the divided differences) a sequential process (forward substitution). While not a direct $$\delta_{ji}$$ biorthogonality with simple point evaluation functionals for the coefficients *as written*, the structure allows efficient computation. One can define functionals related to combinations of point evaluations that *are* dual to the Newton basis polynomials. For instance, the functional that extracts $$f[x_0, \dots, x_i]$$ is dual to $$\pi_i(x)$$.

**4. Discrete Fourier Transform (Trigonometric Interpolation):**
   Suppose we have $$N$$ data points $$(t_k, y_k)$$ where $$t_k = k \frac{2\pi}{N}$$ for $$k=0, \dots, N-1$$ (equally spaced points on $$[0, 2\pi)$$). We want to interpolate these points with a trigonometric polynomial:

   $$
   P(t) = \sum_{m=0}^{N-1} C_m e^{imt}
   $$

   (Adjusting frequency scaling for simplicity here). The basis functions are $$\vert \phi_m \rangle \equiv \phi_m(t) = e^{imt}$$.
   The interpolation conditions are $$P(t_k) = y_k$$, so:

   $$
   \sum_{m=0}^{N-1} C_m e^{im t_k} = \sum_{m=0}^{N-1} C_m e^{imk \frac{2\pi}{N}} = y_k
   $$

   Let $$W_N = e^{i 2\pi/N}$$. Then $$\sum_{m=0}^{N-1} C_m (W_N^{mk}) = y_k$$. This is an inverse DFT form.
   The vectors $$(\mathbf{v}_m)_k = W_N^{mk}$$ (columns of the DFT matrix) are orthogonal in $$\mathbb{C}^N$$:

   $$
   \sum_{k=0}^{N-1} \overline{(W_N^{m_1 k})} (W_N^{m_2 k}) = \sum_{k=0}^{N-1} W_N^{(m_2-m_1)k} = N \delta_{m_1 m_2}
   $$

   This orthogonality of the *basis functions evaluated at the sample points* allows us to find the coefficients $$C_m$$ easily using the forward DFT:

   $$
   C_m = \frac{1}{N} \sum_{k=0}^{N-1} y_k e^{-im t_k} = \frac{1}{N} \sum_{k=0}^{N-1} y_k W_N^{-mk}
   $$

   Here, the "duality" arises from the orthogonality of the columns of the DFT matrix. The rows of the inverse DFT matrix (which is proportional to the conjugate transpose of the DFT matrix) act as the dual basis vectors in the data space $$\mathbb{C}^N$$.

**Important Note on Convergence:**
This framework of biorthogonality helps establish the *existence and form* of such interpolating functions and provides a way to determine their coefficients. However, whether the sequence of interpolating functions $$P_n(x)$$ (as $$n \to \infty$$ or as the number of points/conditions increases) converges to the true underlying function $$f(x)$$ is a separate and often complex issue in approximation theory (e.g., Runge's phenomenon for polynomial interpolation at equally spaced points, convergence of Fourier series, etc.). The discussion here is primarily about the algebraic structure that yields the interpolation formulas.

This connection shows how abstract concepts like dual bases and biorthogonality have very concrete manifestations in constructing practical numerical formulas.
</details>

## 4. The Crucial Ingredient: Completeness

Having an inner product gives our function space a rich geometric structure. However, for many analytical tasks (like ensuring algorithms converge or that differential equations have solutions), we need one more property: **completeness**.

### Convergence and Cauchy Sequences
In a normed space (like an inner product space), we say a sequence of functions $$(\vert f_n \rangle)$$ **converges** to a limit function $$\vert f \rangle$$ if the distance between them goes to zero:

$$
\lim_{n \to \infty} \Vert \vert f_n \rangle - \vert f \rangle \Vert = 0
$$

A sequence $$(\vert f_n \rangle)$$ is called a **Cauchy sequence** if its terms get arbitrarily close to *each other* as $$n$$ and $$m$$ get large:

$$
\lim_{n,m \to \infty} \Vert \vert f_n \rangle - \vert f_m \rangle \Vert = 0
$$

Every convergent sequence is necessarily a Cauchy sequence. But the converse is not always true: a Cauchy sequence might "try" to converge to something that isn't actually in the original space. The space might have "holes."

### The Problem of "Missing Limits"

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">**Warning: Not All Inner Product Spaces Are Complete**</div>
Consider the space $$C([-1,1])$$ of all continuous real-valued functions on the interval $$[-1,1]$$, equipped with the $$L_2$$ inner product $$\langle f \vert g \rangle = \int_{-1}^1 f(x)g(x)dx$$ and its induced norm $$\Vert f \Vert_2 = \left(\int_{-1}^1 (f(x))^2 dx\right)^{1/2}$$.

Let's define a sequence of functions $$\vert f_n \rangle$$ by $$f_n(x) = \tanh(nx)$$.
*   Each $$f_n(x)$$ is continuous on $$[-1,1]$$.
*   One can show that this sequence $$(\vert f_n \rangle)$$ is a Cauchy sequence with respect to the $$L_2$$-norm. Intuitively, the functions are "settling down."

However, as $$n \to \infty$$, $$f_n(x)$$ converges pointwise to the sign function:

$$
f(x) = \lim_{n\to\infty} \tanh(nx) = \begin{cases} -1 & \text{if } x < 0 \\ 0 & \text{if } x = 0 \\ 1 & \text{if } x > 0 \end{cases}
$$

This limit function $$f(x)$$ is indeed in $$L_2([-1,1])$$ (it's square-integrable). However, $$f(x)$$ is *not continuous* at $$x=0$$. Therefore, the limit of this Cauchy sequence of continuous functions is not itself a continuous function; it doesn't belong to our original space $$C([-1,1])$$.

This means $$C([-1,1])$$ with the $$L_2$$-norm is *not complete*. It has "holes" where Cauchy sequences might converge to.
</blockquote>

### Why Completeness is Vital
Completeness is essential for the robustness of analytical methods. When we develop iterative algorithms (like those in optimization or numerical solutions to PDEs), we often generate a sequence of approximate solutions. We want to be sure that if this sequence is Cauchy (meaning it "should" converge), its limit actually exists *within our working space* and represents a valid solution. Without completeness, we might find our algorithms pointing towards an "object" that our space doesn't contain.

This leads us to the definition of a Hilbert space.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 4.1: Hilbert Space**
</div>
A **Hilbert space** is an inner product space that is **complete** with respect to the norm induced by its inner product.
</blockquote>

Essentially, a Hilbert space is an inner product space that has "filled in all its holes." The space $$L_2([a,b])$$, consisting of all (complex-valued, measurable) functions $$f$$ such that $$\int_a^b \vert f(x) \vert^2 dx < \infty$$, equipped with the inner product $$\langle f \vert g \rangle = \int_a^b \overline{f(x)}g(x)dx$$, *is* a Hilbert space. It is, in fact, the completion of $$C([a,b])$$ under the $$L_2$$-norm.

## 5. Hilbert Spaces: Power and Applications

Hilbert spaces are foundational in many areas because they possess both:
1.  Rich **geometric structure** (lengths, angles, orthogonality) from the inner product.
2.  Desirable **analytical properties** (limits of Cauchy sequences exist) from completeness.

### Fourier Analysis as a Prime Example
The theory of Fourier series and Fourier transforms finds its natural home in Hilbert spaces.
*   The space $$L_2([-\pi, \pi])$$ (or $$L_2([0, 2\pi])$$) of square-integrable complex-valued functions on the interval $$[-\pi, \pi]$$ is a Hilbert space.
*   The set of complex exponential functions $$\left\{ \vert \phi_k \rangle \mid \phi_k(x) = \frac{1}{\sqrt{2\pi}} e^{ikx} \right\}_{k \in \mathbb{Z}}$$ forms an **orthonormal basis** for $$L_2([-\pi, \pi])$$.
    *   **Orthogonal:** $$\langle \phi_k \vert \phi_j \rangle = \int_{-\pi}^{\pi} \overline{\left(\frac{e^{ikx}}{\sqrt{2\pi}}\right)} \left(\frac{e^{ijx}}{\sqrt{2\pi}}\right) dx = \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{i(j-k)x} dx = \delta_{kj}$$ (Kronecker delta).
    *   **Basis:** Any function $$\vert f \rangle \in L_2([-\pi, \pi])$$ can be uniquely represented as a Fourier series (which converges in the $$L_2$$-norm):

        $$
        \vert f \rangle = \sum_{k=-\infty}^{\infty} c_k \vert \phi_k \rangle \quad \text{or} \quad f(x) = \sum_{k=-\infty}^{\infty} c_k \frac{e^{ikx}}{\sqrt{2\pi}}
        $$

        where the Fourier coefficients $$c_k$$ are found by projection:

        $$
        c_k = \langle \phi_k \vert f \rangle = \frac{1}{\sqrt{2\pi}}\int_{-\pi}^{\pi} e^{-ikx} f(x) dx
        $$

*   **Parseval's Identity:** A direct consequence is Parseval's identity, which is an infinite-dimensional version of the Pythagorean theorem:

    $$
    \Vert f \Vert^2 = \sum_{k=-\infty}^{\infty} \vert c_k \vert^2
    $$

    This means the "total energy" of the function is the sum of the energies in its orthogonal frequency components.
*   **Fourier Transform:** Similarly, the **Fourier Transform** acts as a unitary operator on the Hilbert space $$L_2(\mathbb{R})$$ (the space of square-integrable functions on the entire real line). This means it preserves inner products and, therefore, norms (lengths/energies). Functions like Gaussians $$e^{-ax^2}$$ are special because they are eigenfunctions of the Fourier transform (up to scaling and argument change).

<details class="details-block" markdown="1">
<summary markdown="1">
**Is the Normal PDF the Only Eigenfunction of the Fourier Transform?**
</summary>
There's a common assertion that "the normal PDF is the only (up to affine transformation due to linearity of Fourier Transform) eigenfunction of the Fourier Transform." This statement, while pointing to an important property of Gaussian functions, requires some clarification:

*   **Gaussian Eigenfunctions:** Gaussian functions, such as $$e^{-ax^2}$$ (which are proportional to a centered normal PDF), are indeed eigenfunctions of the Fourier Transform. For specific choices of $$a$$ and the FT definition, the eigenvalue can be 1. For example, $$e^{-\pi x^2}$$ is an eigenfunction of $$\mathcal{F}[f](\xi) = \int_{-\infty}^{\infty} f(x)e^{-2\pi i x\xi}dx$$ with eigenvalue 1.

*   **The Hermite Functions:** The complete set of orthogonal eigenfunctions of the Fourier Transform in $$L_2(\mathbb{R})$$ is given by the Hermite functions: $$\psi_n(x) = C_n H_n(x) e^{-x^2/2}$$, where $$H_n(x)$$ are Hermite polynomials and $$n=0, 1, 2, \dots$$. The corresponding eigenvalues are $$(-i)^n$$.
    The $$n=0$$ Hermite function is $$\psi_0(x) = C_0 e^{-x^2/2}$$, which is a Gaussian. Other Hermite functions (for $$n>0$$) involve polynomial factors (e.g., $$x e^{-x^2/2}$$, $$(4x^2-2)e^{-x^2/2}$$) and are thus not simple Gaussians/normal PDFs.

*   **"Only" and "Affine Transformation":** The claim often means that the Gaussian (the $$n=0$$ case) is the only eigenfunction in this Hermite series that is purely Gaussian in shape, without polynomial multipliers.
    An "affine transformation" of a function $$f(x)$$ like $$g(x) = K \cdot f(Ax+B)$$ does not always preserve the eigenfunction property in a simple way for arbitrary $$A, B$$. For instance, the Fourier Transform of $$f(x+B)$$ is $$e^{2\pi i B\xi} \hat{f}(\xi)$$. If $$\hat{f}(\xi) = \lambda f(\xi)$$, then $$\mathcal{F}\{f(x+B)\}(\xi) = \lambda e^{2\pi i B\xi} f(\xi)$$, which is not generally of the form $$\lambda' f(\xi+B)$$ unless $$B=0$$ (i.e., the function is centered).
    The "linearity of the Fourier Transform" part correctly notes that if $$f(x)$$ is an eigenfunction with eigenvalue $$\lambda$$, then $$K f(x)$$ is also an eigenfunction with the same eigenvalue $$\lambda$$.

In summary, the Gaussian (or normal PDF shape) is unique as the foundational ($$n=0$$) eigenfunction among the complete set of Hermite eigenfunctions, being the only one without polynomial factors.
</details>

<details class="details-block" markdown="1">
<summary markdown="1">
**Deep Dive: The Fourier Transform as an "Imaginary Unit" in Function Space**
</summary>
It's a fascinating and profound fact that the Fourier Transform ($$\mathcal{F}$$), in certain settings, behaves remarkably like the imaginary unit $$i = \sqrt{-1}$$. This connection opens up a rich theoretical landscape with many parallels to familiar concepts from linear algebra and complex numbers.

1.  **The Fourier Transform and Parity:**
    Depending on the precise definition and normalization, applying the Fourier Transform twice to a function $$f(x)$$ often results in the parity-flipped version of the original function:

    $$
    \mathcal{F}^2[f](x) = (\mathcal{F} \circ \mathcal{F})[f](x) \propto f(-x)
    $$

    For instance, with a common unitary definition of the Fourier Transform, we get $$\mathcal{F}^2[f](x) = f(-x)$$.
    This means applying the FT *four* times brings us back to the original function:

    $$
    \mathcal{F}^4[f](x) = \mathcal{F}^2[\mathcal{F}^2[f]](x) = \mathcal{F}^2[f(-x)] = f(-(-x)) = f(x)
    $$

    So, $$\mathcal{F}$$ acts as an operator whose fourth power is the identity ($$\mathcal{F}^4 = \mathbb{I}$$), much like $$i^4 = 1$$.

2.  **Eigenvalues and Analogy to $$i$$:**
    The eigenvalues of the Fourier Transform operator (acting on $$L_2(\mathbb{R})$$) are precisely the fourth roots of unity: $$1, -i, -1, i$$.
    *   Functions that are their own Fourier transform (eigenvalue $$1$$).
    *   Functions for which $$\mathcal{F}[f] = -i f$$ (eigenvalue $$-i$$).
    *   Functions for which $$\mathcal{F}[f] = -f$$ (eigenvalue $$-1$$).
    *   Functions for which $$\mathcal{F}[f] = i f$$ (eigenvalue $$i$$).
    The Hermite functions (related to Hermite polynomials multiplied by a Gaussian, e.g., $$H_n(x)e^{-x^2/2}$$) are a famous family of eigenfunctions of the Fourier Transform, with eigenvalues $$(-i)^n$$.

3.  **The Fractional Fourier Transform (FrFT): A Continuous Analog**
    Just as the complex exponential $$e^{i\theta}$$ provides a continuous "rotation" in the complex plane, generalizing discrete powers of $$i$$ (e.g., $$i = e^{i\pi/2}$$, $$i^2 = e^{i\pi} = -1$$), we can define a **Fractional Fourier Transform (FrFT)**, denoted $$\mathcal{F}^\alpha$$.
    This operator generalizes the ordinary Fourier Transform to non-integer "powers" or "orders" $$\alpha$$.
    *   $$\mathcal{F}^0[f] = f$$ (the identity operator).
    *   $$\mathcal{F}^1[f] = \mathcal{F}[f]$$ (the ordinary Fourier Transform).
    *   $$\mathcal{F}^2[f](x) \propto f(-x)$$ (the parity/reversal operator, as discussed).
    *   $$\mathcal{F}^4[f] = f$$.
    The FrFT has the additive property for its order: $$\mathcal{F}^\alpha \mathcal{F}^\beta = \mathcal{F}^{\alpha+\beta}$$.
    The parameter $$\alpha$$ can be interpreted as related to an angle of rotation (specifically, $$\phi = \alpha \pi/2$$) in the time-frequency phase space (e.g., of the Wigner distribution of the signal).

4.  **Rich Theory from Linear Algebra:**
    This perspective allows us to leverage many familiar concepts from linear algebra. The Fourier Transform is a unitary operator. The existence of eigenfunctions (like Hermite functions) means we can decompose functions into components that behave very simply under the FT. The FrFT itself can be defined via spectral decomposition using these eigenfunctions, much like how one can define $$A^\alpha$$ for a diagonalizable matrix $$A$$ using its eigenvalues $$D$$ and eigenvectors $$U$$ ($$A = U D U^{-1}$$ implies $$A^\alpha = U D^\alpha U^{-1}$$).

This deep connection between the Fourier Transform and the structure of complex numbers or rotations is not just a mathematical curiosity. It has practical implications in signal processing (e.g., filter design, signal analysis in the time-frequency domain) and physics (e.g., quantum mechanics, optics). It highlights how fundamental mathematical structures reappear in diverse contexts, enriching our understanding of each.
</details>

### Other Key Areas Benefiting from Hilbert Space Theory
*   **Quantum Mechanics:** The state of a quantum system is described by a vector (ket) in a complex Hilbert space. Observables (like energy or momentum) are represented by self-adjoint operators on this space.
*   **Signal Processing:** Analyzing, filtering, and compressing signals often involves decomposing them using orthogonal bases (like Fourier bases or wavelet bases) within a Hilbert space framework.
*   **Partial Differential Equations (PDEs):** Many techniques for proving existence and uniqueness of solutions to PDEs, as well as numerical methods like the Finite Element Method (FEM), are formulated in specific Hilbert spaces called Sobolev spaces.

### Relevance to Machine Learning & Optimization
*   **Function Approximation & Learning:** Many machine learning problems can be framed as finding an optimal function from some class that best fits data or minimizes a loss. Hilbert spaces provide a setting for such problems.
*   **Reproducing Kernel Hilbert Spaces (RKHS):** These are very special Hilbert spaces of functions that play a central role in kernel methods (e.g., Support Vector Machines, Gaussian Processes, Kernel PCA). The "kernel trick" is implicitly performing operations in an RKHS. We will likely delve deeper into RKHS in later parts of broader ML discussions.
*   **Optimization in Function Spaces:** The Riesz Representation Theorem, which we touched upon in Post 1 for finite dimensions, extends to Hilbert spaces. It states that every continuous linear functional (bra) on a Hilbert space $$H$$ can be uniquely represented by an inner product with some ket in $$H$$. This is incredibly important because the derivative of a scalar-valued functional (like a loss function defined on functions) is a bra. Riesz allows us to convert this derivative bra into a gradient ket, an element of the Hilbert space itself, enabling gradient-based optimization directly in function spaces.

## 6. Conclusion: Geometry and Analysis United

We started by recognizing that treating functions merely as elements of a vector space was insufficient for many practical and theoretical needs. By introducing the **inner product**, we endowed these spaces with rich geometric structure: notions of length, distance, and orthogonality, analogous to those in $$\mathbb{R}^n$$.

However, this geometric structure alone isn't robust enough for the demands of analysis if the space has "holes." The crucial property of **completeness** ensures that Cauchy sequences (sequences that "ought to" converge) do indeed converge to a limit *within the space*.

**Hilbert spaces** are precisely those inner product spaces that are complete. This combination makes them an incredibly powerful and versatile framework. They allow us to rigorously apply geometric intuition and sophisticated analytical techniques to infinite-dimensional problems, forming a bedrock for fields ranging from quantum physics and engineering to the theoretical underpinnings of modern machine learning and optimization.

**Next Up:** What happens if our function space has a well-defined notion of length (a norm) and is complete, but this norm doesn't necessarily arise from an inner product? This means we might lose the rich geometric structure of angles and orthogonality inherent to inner products, but we still retain a strong analytical framework. This will lead us to the broader class of **Banach spaces** in our next post.

## 7. Summary Cheat Sheet

| Concept               | Description                                                                                           | Example/Analogy in $$\mathbb{R}^n$$                             | Key Implication in Function Spaces                                                |
| :-------------------- | :---------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Inner Product**     | Generalizes dot product; defines geometric relations.                                                 | $$\mathbf{x} \cdot \mathbf{y}$$                                 | $$\langle f \vert g \rangle = \int \overline{f(x)}g(x) dx$$ (for $$L_2$$)         |
| **Induced Norm**      | Length/magnitude of a vector/function derived from the inner product.                                 | $$\Vert \mathbf{x} \Vert = \sqrt{\mathbf{x} \cdot \mathbf{x}}$$ | $$\Vert f \Vert = \sqrt{\langle f \vert f \rangle}$$                              |
| **Orthogonality**     | Two vectors/functions are "perpendicular" if their inner product is zero.                             | $$\mathbf{x} \cdot \mathbf{y} = 0$$                             | $$\langle f \vert g \rangle = 0$$                                                 |
| **Cauchy Sequence**   | A sequence whose terms get arbitrarily close to each other ($$\Vert f_n - f_m \Vert \to 0$$).         | "Looks like it should converge."                                | Defines "converging-like" behavior.                                               |
| **Completeness**      | Property that every Cauchy sequence in the space converges to a limit *within* that same space.       | $$\mathbb{R}^n$$ is complete.                                   | No "holes" in the space; limits always exist for Cauchy sequences.                |
| **Hilbert Space**     | An inner product space that is complete with respect to its induced norm.                             | $$\mathbb{R}^n$$ with dot product.                              | $$L_2([a,b])$$; ideal for combining geometry & analysis.                          |
| **$$L_2$$ Space**     | Space of (measurable) functions $$f$$ for which $$\int \vert f(x) \vert^2 dx < \infty$$.              | -                                                               | Prototypical infinite-dimensional Hilbert space.                                  |
| **Orthonormal Basis** | A set of mutually orthogonal unit-norm vectors/functions that can represent any element in the space. | Standard basis $$(\mathbf{e}_i)$$.                              | Fourier basis ($$\frac{1}{\sqrt{2\pi}}e^{ikx}$$); enables function decomposition. |

## 8. Reflection

In this post, we've journeyed from the basic idea of functions as vectors to the sophisticated concept of Hilbert spaces. The key was to recognize the need for more structure than a plain vector space offers. By introducing the **inner product**, we unlocked a wealth of geometric intuition – lengths, distances, angles, and crucially, orthogonality. This allowed us to think about functions in ways directly analogous to vectors in familiar Euclidean spaces, enabling concepts like projection and decomposition.

However, the world of infinite dimensions brings challenges. The notion of **completeness** became paramount, ensuring that our analytical tools, particularly those involving limits and convergence, are well-behaved. A space might have all the nice geometric properties of an inner product but still be "leaky" if Cauchy sequences don't find their limits within it.

Hilbert spaces elegantly resolve this by mandating completeness. They provide a stable and robust environment where geometric intuition and powerful analytical machinery can work hand-in-hand. Understanding Hilbert spaces is not just an academic exercise; it opens the door to comprehending Fourier analysis, quantum mechanics, advanced signal processing, and the theoretical foundations of many optimization algorithms and machine learning models, especially those operating in function spaces or dealing with infinite-dimensional parameter vectors. The journey emphasizes that the right mathematical abstraction can turn complex problems into more manageable, and often more beautiful, ones.

## References

Kowalski, E. (2013). Spectral theory in Hilbert spaces (ETH Z¨urich, FS 09). https://people.math.ethz.ch/~kowalski/spectral-theory.pdf

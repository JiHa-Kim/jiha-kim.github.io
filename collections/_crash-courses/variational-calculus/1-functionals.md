---
title: "Variational Calculus Part 1: Functionals and the Quest for Optimal Functions"
date: 2025-06-16 10:00 -0400
course_index: 1 # First post in the Variational Calculus crash course
mermaid: true
description: "An introduction to variational calculus: exploring functionals, the challenge of optimizing entire functions, and developing the concept of the first variation as a 'derivative' for functionals."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Functionals
- Variations
- Optimization
- Euler-Lagrange Equation
- Calculus of Variations
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
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

Welcome to our crash course on Variational Calculus! In standard calculus, a cornerstone of optimization is finding points where the derivative of a function vanishes.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Principle.** Stationary Points in Single-Variable Calculus
</div>
For a 'nice' (e.g., continuously differentiable) function $$f(x)$$, its local maxima and minima occur at points $$x_0$$ where its derivative is zero, i.e., $$f'(x_0)=0$$. These are called stationary or critical points.
</blockquote>

But what if the quantity we want to optimize isn't determined by a single variable $$x$$, but by an entire function or path? Consider questions like:
-   What is the shortest path between two points in a plane?
-   What path does a ray of light take through a medium with a continuously varying refractive index? (Fermat's Principle of Least Time)
-   What is the shape of a soap film spanning a given wire frame that minimizes surface area?

These problems require us to optimize a value that depends on the *choice of a function*. This is the realm of **variational calculus**. This field provides the tools to find functions that make certain quantities (called functionals) stationary. It's a powerful framework with deep roots in physics and engineering, and increasingly relevant in machine learning for understanding regularization, optimal control, and energy-based models.

In this first post, we will:
1.  Introduce **functionals** through motivating examples.
2.  Frame the core problem: optimizing functionals.
3.  Develop the concept of a **variation** of a function, which allows us to define a "derivative" for functionals, leading to the **first variation**.

<blockquote class="prompt-info" markdown="1">
**Assumption: 'Nice' Cases**

Throughout this crash course, we will generally assume that all functions involved are sufficiently "nice" – meaning they are smooth (continuously differentiable as many times as needed) and well-behaved, allowing us to avoid pathological exceptions and focus on the core concepts. This is a common practice to make the introduction to the subject more accessible.
</blockquote>

## 1. From Functions to Functionals: Motivating Examples

Let's start by seeing how problems naturally lead to the concept of a functional.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 1.** Shortest Path Between Two Points
</div>
Given two points $$(x_1, y_1)$$ and $$(x_2, y_2)$$ in a plane, we want to find the curve $$y(x)$$ connecting them that has the shortest possible length. We know intuitively that the answer is a straight line. But how can we derive this formally, especially if the problem were more complex?

The length $$L$$ of a curve $$y(x)$$ from $$x=x_1$$ to $$x=x_2$$ is given by the arc length formula from calculus:

$$
L = \int_{x_1}^{x_2} \sqrt{1 + (y'(x))^2} \, dx
$$

Notice what's happening here:
-   The **input** is an entire function $$y(x)$$ (representing a path).
-   The **output** $$L$$ is a single real number (the length of that path).

Different functions $$y(x)$$ (different paths) will generally yield different lengths $$L$$. This "function of a function" is what we call a functional. We denote it as $$L[y]$$ or $$L[y(x)]$$ to emphasize that its argument is a function.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 2.** Fermat's Principle of Least Time (Simplified)
</div>
Imagine light traveling from point A to point B through a medium where its speed $$v$$ can vary depending on position. For instance, let speed depend only on the horizontal coordinate $$x$$, so $$v = v(x)$$. Fermat's principle states that light takes the path that minimizes travel time.

If the path is described by a function $$y(x)$$, an infinitesimal segment of the path has length $$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y'(x))^2} \, dx$$. The time taken to traverse this segment is $$dt = ds / v(x)$$.
The total time $$T$$ to travel from $$x=a$$ to $$x=b$$ along the path $$y(x)$$ is:

$$
T[y] = \int_a^b \frac{\sqrt{1 + (y'(x))^2}}{v(x)} \, dx
$$

Again, the input is the path function $$y(x)$$, and the output $$T[y]$$ is the total travel time (a scalar). Unlike the straight-line case, the minimizing path here is generally *not* obvious and depends on the form of $$v(x)$$. This highlights the need for a systematic method to find such optimal functions.
</blockquote>

These examples lead us to a general definition:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Functional
</div>
A **functional** $$J$$ is a mapping from a vector space of functions $$\mathcal{Y}$$ (the space of admissible functions) to the real numbers $$\mathbb{R}$$. If $$y(x)$$ is a function in $$\mathcal{Y}$$, then $$J[y]$$ denotes the scalar value of the functional evaluated at $$y$$.

The square brackets $$J[y]$$ are conventionally used to distinguish functionals from ordinary functions $$f(x)$$.
</blockquote>

<blockquote class="prompt-tip" markdown="1">
**Single-Variable Calculus vs. Calculus of Variations**
-   **Single-Variable Calculus:** Optimizes a function $$f(x)$$ where $$x$$ is a variable (a point).
-   **Calculus of Variations:** Optimizes a functional $$J[y]$$ where $$y$$ is a function (a curve, path, etc.).
</prompt-tip>

## 2. The Core Problem: Optimizing Functionals

The fundamental goal of variational calculus is to find a function $$y_0(x)$$ from a specified class of admissible functions that makes a given functional $$J[y]$$ stationary (i.e., a minimum, maximum, or saddle point).

This is analogous to finding critical points in ordinary calculus, but the "variable" we are optimizing over is now an entire function, which can be thought of as a point in an infinite-dimensional function space.

## 3. Developing the "Derivative": The Concept of Variations

How do we find the "derivative" of a functional $$J[y]$$ with respect to the function $$y(x)$$? This is the key question. We can't just differentiate with respect to $$y$$ as if it were a simple variable, because $$y$$ is a function, and the functional $$J[y]$$ often depends not only on $$y(x)$$ at a point $$x$$ but also on its derivatives like $$y'(x)$$, and values across an entire interval via an integral.

Let's draw inspiration from how derivatives are defined. The derivative of $$f(x)$$ tells us how $$f$$ changes for an infinitesimal change in $$x$$. For functionals, we need to see how $$J[y]$$ changes when we make a small "perturbation" or "variation" to the entire function $$y(x)$$.

Consider a candidate function $$y(x)$$ that we suspect might extremize $$J[y]$$. We create a "nearby" or "varied" function $$\tilde{y}(x)$$ by adding a small, arbitrary perturbation:

$$
\tilde{y}(x; \epsilon) = y(x) + \epsilon \eta(x)
$$

Let's break this down:
-   $$y(x)$$: The function we are testing for extremality.
-   $$\eta(x)$$: An arbitrary, sufficiently smooth function called the **variation function** or **test function**. It represents the "direction" in the space of functions along which we are perturbing $$y(x)$$.
-   $$\epsilon$$: A small real number (a scalar parameter). As $$\epsilon \to 0$$, the perturbed function $$\tilde{y}(x; \epsilon)$$ approaches $$y(x)$$.

**Boundary Conditions for Variations:**
If the problem requires $$y(x)$$ to satisfy fixed boundary conditions, say $$y(a) = y_a$$ and $$y(b) = y_b$$, then any admissible perturbed function $$\tilde{y}(x; \epsilon)$$ must also satisfy these same boundary conditions for all $$\epsilon$$.
Since $$y(x) + \epsilon \eta(x)$$ must equal $$y_a$$ at $$x=a$$, and $$y(a)=y_a$$, this implies $$\epsilon \eta(a) = 0$$. Similarly, $$\epsilon \eta(b) = 0$$. For these to hold for any small non-zero $$\epsilon$$, the variation function $$\eta(x)$$ must itself vanish at the boundaries:

$$
\eta(a) = 0 \quad \text{and} \quad \eta(b) = 0
$$

Such an $$\eta(x)$$ is called an **admissible variation** for problems with fixed endpoints.

<details class="details-block" markdown="1">
<summary markdown="1">
**Analogy.** Directional Derivatives in Multivariable Calculus
</summary>
This approach is very similar to how directional derivatives are defined for a multivariable function $$f(\mathbf{x})$$ where $$\mathbf{x} \in \mathbb{R}^n$$. To find the rate of change of $$f$$ at $$\mathbf{x}_0$$ in the direction of a vector $$\mathbf{v}$$, we consider the function $$g(\epsilon) = f(\mathbf{x}_0 + \epsilon \mathbf{v})$$. The directional derivative is then $$g'(0)$$.

In our case:
-   The function $$y(x)$$ is analogous to the point $$\mathbf{x}_0$$.
-   The variation function $$\eta(x)$$ is analogous to the direction vector $$\mathbf{v}$$.
-   The scalar $$\epsilon$$ is analogous to the step size.
-   The functional $$J[y]$$ is analogous to $$f(\mathbf{x})$$.
</details>

Now, if we substitute $$\tilde{y}(x; \epsilon) = y(x) + \epsilon \eta(x)$$ into the functional $$J[y]$$, the value of the functional becomes a regular function of the single real variable $$\epsilon$$ (assuming $$y$$ and $$\eta$$ are fixed):

$$
\Phi(\epsilon) = J[y + \epsilon \eta]
$$

If $$y(x)$$ is indeed an extremizing function for $$J[y]$$, then for any choice of admissible variation $$\eta(x)$$, the function $$\Phi(\epsilon)$$ must have a stationary point at $$\epsilon = 0$$. From ordinary calculus, this means its derivative with respect to $$\epsilon$$ must be zero at $$\epsilon = 0$$:

$$
\left. \frac{d\Phi(\epsilon)}{d\epsilon} \right|_{\epsilon=0} = \left. \frac{d}{d\epsilon} J[y + \epsilon \eta] \right|_{\epsilon=0} = 0
$$

This derivative is precisely what we call the **first variation** of the functional $$J$$ at $$y$$ in the "direction" $$\eta$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The First Variation (or Gâteaux Derivative)
</div>
The **first variation** of a functional $$J[y]$$ at the function $$y$$ with respect to a variation function $$\eta(x)$$ (often denoted $$\delta y = \epsilon \eta$$ for infinitesimal $$\epsilon$$) is given by:

$$
\delta J[y; \eta] = \left. \frac{d}{d\epsilon} J[y + \epsilon \eta] \right|_{\epsilon=0}
$$

A necessary condition for $$y(x)$$ to be an extremizer of $$J[y]$$ (among functions satisfying the given boundary conditions) is that its first variation must be zero for **all** admissible variation functions $$\eta(x)$$:

$$
\delta J[y; \eta] = 0 \quad \text{for all admissible } \eta(x)
$$

</blockquote>

<blockquote class="prompt-info" markdown="1">
**Remark.** The notation $$\delta J$$ is common. Sometimes it refers to $$\delta J[y; \eta]$$ (the derivative itself), and sometimes it informally refers to the principal linear part of the change $$\Delta J = J[y+\epsilon\eta] - J[y] \approx \epsilon \cdot \delta J[y;\eta]$$. The definition using the derivative with respect to $$\epsilon$$ is the most practical for calculations.
</blockquote>

This condition, $$\delta J[y; \eta] = 0$$ for all admissible $$\eta$$, is the cornerstone of variational calculus. It's the direct analogue of $$f'(x)=0$$ for finding extrema of ordinary functions. The power of this condition comes from the requirement that it must hold for *every* possible (admissible) way of varying the function.

## 4. What's Next?

We've established what functionals are and introduced the first variation as a way to detect stationary "points" (which are actually functions!) in the landscape defined by a functional. The crucial necessary condition for an extremum is $$\delta J[y; \eta] = 0$$ for all admissible $$\eta(x)$$.

But how do we use this condition? It still seems abstract. In the next post, we will:
1.  Consider the common form of functionals: $$J[y] = \int_a^b F(x, y(x), y'(x)) \, dx$$.
2.  Explicitly calculate $$\delta J[y; \eta]$$ for such functionals.
3.  Use the fact that $$\delta J = 0$$ must hold for *all* admissible $$\eta(x)$$, along with a key result called the **Fundamental Lemma of Variational Calculus**, to derive the famous **Euler-Lagrange equation**.

The Euler-Lagrange equation is a differential equation that the extremizing function $$y(x)$$ must satisfy. Solving this differential equation (subject to boundary conditions) will give us the candidate functions that make the functional stationary.

Stay tuned as we turn this abstract condition into a concrete computational tool!

---
title: "Variational Calculus Part 2: The Euler-Lagrange Equation"
date: 2025-05-17 10:00 -0400
course_index: 2 # Second post in the Variational Calculus crash course
mermaid: true
description: "Deriving the Euler-Lagrange equation, the fundamental differential equation that extremizing functions must satisfy in variational problems, using the first variation and the fundamental lemma."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Euler-Lagrange Equation
- Functionals
- Optimization
- Fundamental Lemma
- Integration by Parts
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

In Part 1, we introduced **functionals** $$J[y]$$ – functions of functions – and established that a necessary condition for a function $$y(x)$$ to extremize $$J[y]$$ is that its **first variation** must vanish for all admissible variations $$\eta(x)$$:

$$
\delta J[y; \eta] = \left. \frac{d}{d\epsilon} J[y + \epsilon \eta] \right\vert_{\epsilon=0} = 0
$$

This is the variational calculus equivalent of setting $$f'(x)=0$$ in ordinary calculus. Now, our goal is to transform this abstract condition into a concrete, usable tool. We will focus on a very common type of functional and derive the celebrated **Euler-Lagrange equation**, a differential equation that the extremizing function $$y(x)$$ must satisfy.

## 1. The Standard Functional Form

Many problems in physics, engineering, and even machine learning involve functionals of the following form:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Standard Functional
</div>
A common type of functional depends on a function $$y(x)$$, its first derivative $$y'(x) = dy/dx$$, and the independent variable $$x$$, integrated over an interval $$[a, b]$$:

$$
J[y] = \int_a^b F(x, y(x), y'(x)) \, dx
$$

Here, $$F(x, y, y')$$ is a given function of three variables, often called the **Lagrangian** or the **integrand function**. We assume $$F$$ has continuous partial derivatives with respect to its arguments. We also assume $$y(x)$$ is twice continuously differentiable.
</blockquote>

Examples from Part 1, like the arc length functional ($$F = \sqrt{1+(y')^2}$$) and the Fermat's principle functional ($$F = \sqrt{1+(y')^2}/v(x)$$), fit this form.

## 2. Calculating the First Variation Explicitly

Let's compute $$\delta J[y; \eta]$$ for the standard functional. Recall that $$\tilde{y}(x; \epsilon) = y(x) + \epsilon \eta(x)$$. Then, its derivative is $$\tilde{y}'(x; \epsilon) = y'(x) + \epsilon \eta'(x)$$.
Substituting into the functional:

$$
J[y + \epsilon \eta] = \int_a^b F(x, y(x) + \epsilon \eta(x), y'(x) + \epsilon \eta'(x)) \, dx
$$

To find the first variation, we differentiate this expression with respect to $$\epsilon$$ and then set $$\epsilon = 0$$. Assuming we can differentiate under the integral sign (Leibniz integral rule, valid here due to our smoothness assumptions):

$$
\delta J[y; \eta] = \left. \frac{d}{d\epsilon} \int_a^b F(x, y + \epsilon \eta, y' + \epsilon \eta') \, dx \right\vert_{\epsilon=0}
$$

$$
\delta J[y; \eta] = \int_a^b \left. \frac{d}{d\epsilon} F(x, y + \epsilon \eta, y' + \epsilon \eta') \right\vert_{\epsilon=0} \, dx
$$

Now, we apply the chain rule to the integrand $$F$$. Let $$Y = y + \epsilon \eta$$ and $$Y' = y' + \epsilon \eta'$$. Then $$F = F(x, Y, Y')$$.
So, $$\frac{dF}{d\epsilon} = \frac{\partial F}{\partial Y} \frac{\partial Y}{\partial \epsilon} + \frac{\partial F}{\partial Y'} \frac{\partial Y'}{\partial \epsilon}$$.
We have:
-   $$\frac{\partial Y}{\partial \epsilon} = \frac{\partial}{\partial \epsilon}(y + \epsilon \eta) = \eta$$
-   $$\frac{\partial Y'}{\partial \epsilon} = \frac{\partial}{\partial \epsilon}(y' + \epsilon \eta') = \eta'$$

Therefore,

$$
\frac{d}{d\epsilon} F(x, y + \epsilon \eta, y' + \epsilon \eta') = \frac{\partial F}{\partial (y + \epsilon \eta)} \eta + \frac{\partial F}{\partial (y' + \epsilon \eta')} \eta'
$$

Setting $$\epsilon = 0$$, we get:

$$
\left. \frac{d}{d\epsilon} F(x, y + \epsilon \eta, y' + \epsilon \eta') \right\vert_{\epsilon=0} = \frac{\partial F}{\partial y}(x, y, y') \eta(x) + \frac{\partial F}{\partial y'}(x, y, y') \eta'(x)
$$

For brevity, we'll write $$\frac{\partial F}{\partial y}$$ and $$\frac{\partial F}{\partial y'}$$}, understanding they are evaluated at $$(x, y(x), y'(x))$$.

Plugging this back into the integral for $$\delta J[y; \eta]$$:

$$
\delta J[y; \eta] = \int_a^b \left( \frac{\partial F}{\partial y} \eta(x) + \frac{\partial F}{\partial y'} \eta'(x) \right) \, dx
$$

The necessary condition for an extremum is $$\delta J[y; \eta] = 0$$:

$$
\int_a^b \left( \frac{\partial F}{\partial y} \eta(x) + \frac{\partial F}{\partial y'} \eta'(x) \right) \, dx = 0
$$

This equation must hold for all admissible variation functions $$\eta(x)$$.

## 3. The Key Maneuver: Integration by Parts

The expression above involves both $$\eta(x)$$ and its derivative $$\eta'(x)$$. To make progress, we want to factor out $$\eta(x)$$ from the entire integrand. We can achieve this by applying **integration by parts** to the second term: $$\int u \, dv = uv - \int v \, du$$.

Let $$u = \frac{\partial F}{\partial y'}$$ and $$dv = \eta'(x) \, dx$$.
Then $$du = \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) \, dx$$ and $$v = \eta(x)$$.

So, the second term becomes:

$$
\int_a^b \frac{\partial F}{\partial y'} \eta'(x) \, dx = \left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b - \int_a^b \eta(x) \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) \, dx
$$

The boundary term $$\left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b = \frac{\partial F}{\partial y'}(b) \eta(b) - \frac{\partial F}{\partial y'}(a) \eta(a)$$.
Recall from Part 1 that for problems with fixed endpoints $$y(a)=y_a$$ and $$y(b)=y_b$$, the admissible variations $$\eta(x)$$ must satisfy $$\eta(a) = 0$$ and $$\eta(b) = 0$$.
Therefore, for such problems, the boundary term vanishes:

$$
\left[ \frac{\partial F}{\partial y'} \eta(x) \right]_a^b = 0
$$

<blockquote class="prompt-info" markdown="1">
**Note on Boundary Conditions:** If the endpoints are not fixed (so-called "natural boundary conditions"), then $$\eta(a)$$ and $$\eta(b)$$ are not necessarily zero, and the boundary terms must be handled differently. This leads to additional conditions on $$\frac{\partial F}{\partial y'}$$ at the endpoints. We will focus on fixed endpoints for now.
</blockquote>

Substituting the result of the integration by parts (with the vanishing boundary term) back into the equation for $$\delta J = 0$$:

$$
\int_a^b \frac{\partial F}{\partial y} \eta(x) \, dx - \int_a^b \eta(x) \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) \, dx = 0
$$

Combining the integrals:

$$
\int_a^b \left( \frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) \right) \eta(x) \, dx = 0
$$

This equation is crucial. It states that the integral of the product of the term in the parenthesis and $$\eta(x)$$ is zero for *any* admissible variation function $$\eta(x)$$. This leads us to a powerful lemma.

## 4. The Fundamental Lemma of Variational Calculus

The equation we've reached is of the form $$\int_a^b g(x) \eta(x) \, dx = 0$$, where $$g(x) = \frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right)$$.

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
**Lemma.** Fundamental Lemma of Variational Calculus (du Bois-Reymond)
</div>
If a function $$g(x)$$ is continuous on the interval $$[a, b]$$, and if

$$
\int_a^b g(x) \eta(x) \, dx = 0
$$

for every continuously differentiable function $$\eta(x)$$ such that $$\eta(a) = 0$$ and $$\eta(b) = 0$$, then $$g(x) = 0$$ for all $$x \in [a, b]$$.
</blockquote>

**Intuition behind the Lemma:**
Suppose, for the sake of contradiction, that $$g(x_0) \neq 0$$ for some $$x_0 \in (a, b)$$. Let's say $$g(x_0) > 0$$. Since $$g(x)$$ is continuous, there must be a small subinterval around $$x_0$$, say $$[c, d] \subset (a, b)$$, where $$g(x) > 0$$ throughout this subinterval.

Now, we can construct a specific variation function $$\eta(x)$$ that is positive within $$[c, d]$$ and zero outside this subinterval (and still satisfies $$\eta(a)=\eta(b)=0$$ because $$[c,d]$$ is strictly inside $$(a,b)$$). Such functions, often called "bump functions," can be made smooth.
For example, one could choose $$\eta(x) = (x-c)^2(x-d)^2$$ for $$x \in [c,d]$$ and $$\eta(x)=0$$ otherwise (or a smoother version using exponentials as shown in the reference materials from the prompt).

For such an $$\eta(x)$$:
-   $$g(x) \eta(x) > 0$$ for $$x \in (c, d)$$
-   $$g(x) \eta(x) = 0$$ for $$x \notin [c, d]$$

Then the integral $$\int_a^b g(x) \eta(x) \, dx = \int_c^d g(x) \eta(x) \, dx$$ would be strictly positive. This contradicts our premise that the integral is zero for *all* admissible $$\eta(x)$$.
Therefore, our assumption that $$g(x_0) \neq 0$$ must be false. Thus, $$g(x) = 0$$ for all $$x \in [a, b]$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**More on Bump Functions**
</summary>
A common example of a smooth bump function that is non-zero only on a finite interval, say $$(-1, 1)$$, is:

$$
B(t) = \begin{cases} \exp\left(-\frac{1}{1-t^2}\right) & \text{if } \vert t \vert < 1 \\ 0 & \text{if } \vert t \vert \ge 1 \end{cases}
$$

This function is infinitely differentiable everywhere, including at $$t=\pm 1$$ where all derivatives are zero. By scaling and translating $$t$$, we can create such a bump function $$\eta(x)$$ over any desired subinterval $$[c, d]$$ within $$[a, b]$$. This rigorous construction underpins the Fundamental Lemma.
</details>

## 5. The Euler-Lagrange Equation

Applying the Fundamental Lemma of Variational Calculus to our equation:

$$
\int_a^b \left( \frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) \right) \eta(x) \, dx = 0
$$

The term in the parenthesis plays the role of $$g(x)$$. If this integral is zero for all admissible $$\eta(x)$$, then the term itself must be identically zero:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** The Euler-Lagrange Equation
</div>
A function $$y(x)$$ that extremizes the functional $$J[y] = \int_a^b F(x, y(x), y'(x)) \, dx$$ with fixed boundary conditions $$y(a)=y_a$$ and $$y(b)=y_b$$, must satisfy the following second-order ordinary differential equation:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0
$$

This is known as the **Euler-Lagrange equation**.
</blockquote>

**Understanding the terms:**
-   $$\frac{\partial F}{\partial y}$$: The partial derivative of $$F(x, y, y')$$ with respect to its second argument $$y$$, treating $$x$$ and $$y'$$ as constants.
-   $$\frac{\partial F}{\partial y'}$$: The partial derivative of $$F(x, y, y')$$ with respect to its third argument $$y'$$, treating $$x$$ and $$y$$ as constants.
-   $$\frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right)$$: The total derivative with respect to $$x$$ of the expression $$\frac{\partial F}{\partial y'}$$. Since $$y$$ and $$y'$$ are functions of $$x$$, this derivative will generally involve $$y'(x)$$ and $$y''(x)$$ via the chain rule:

    $$
    \frac{d}{dx} \left( \frac{\partial F}{\partial y'}(x, y(x), y'(x)) \right) = \frac{\partial^2 F}{\partial x \partial y'} + \frac{\partial^2 F}{\partial y \partial y'} y' + \frac{\partial^2 F}{\partial y'^2} y''
    $$

The Euler-Lagrange equation is a differential equation for the unknown function $$y(x)$$. Solving it (subject to the boundary conditions) provides the candidate functions that could extremize the functional.

## 6. Significance and What's Next

The derivation of the Euler-Lagrange equation is a monumental step in variational calculus. It converts the problem of optimizing over an infinite-dimensional space of functions into the more familiar problem of solving a differential equation.
1.  **Global Criterion to Local Rule:** The original problem was to minimize a global quantity (the integral $$J[y]$$). The Euler-Lagrange equation provides a local condition (a differential equation) that must hold at every point $$x$$.
2.  **Universality:** This single method, encapsulated by the Euler-Lagrange equation, can tackle a vast array of problems that seek to find an optimal function, simply by identifying the correct integrand $$F(x, y, y')$$.

In the next part of this crash course, we will:
-   Apply the Euler-Lagrange equation to solve some classic variational problems, such as finding the shortest path between two points (revisiting our initial example) and the brachistochrone problem.
-   Discuss some special cases and first integrals of the Euler-Lagrange equation (Beltrami identity).

This will demonstrate the power and utility of the machinery we've developed.

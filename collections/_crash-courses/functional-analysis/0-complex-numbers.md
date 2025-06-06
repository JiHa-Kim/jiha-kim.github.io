---
title: Basics of Complex Numbers
date: 2025-06-06 01:50 -0400
sort_index: 0
description: The imaginary unit and algebraic and geometric properties of complex numbers.
image: # placeholder
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Complex Numbers
- Imaginary Unit
- Euler's Formula
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

Welcome to our crash course series on the mathematical foundations for machine learning and optimization. While our main focus is on real-valued optimization, many concepts from linear algebra and functional analysis are most naturally understood through the lens of **complex numbers**. For instance, a real matrix can have complex eigenvalues and eigenvectors. To build a solid foundation, we need to be comfortable with the basic properties of complex numbers.

This post will cover just enough to get you started: the definition of complex numbers, their algebraic and geometric properties, and why they are indispensable for matrix analysis.

## The Imaginary Unit

We know that for any real number $$x$$, its square $$x^2$$ is always non-negative. This means a simple equation like $$x^2 = -1$$ has no solution within the real numbers $$\mathbb{R}$$. To solve this, we introduce a new number, called the **imaginary unit**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Imaginary Unit
</div>
The **imaginary unit**, denoted by $$i$$, is defined as a number that satisfies the property:

$$
i^2 = -1
$$

</div>
</blockquote>

By introducing $$i$$, we can now find solutions to equations like $$x^2 = -9$$, which would be $$x = \pm \sqrt{-9} = \pm \sqrt{9 \cdot -1} = \pm 3i$$.

## The Complex Number System

A complex number is a number that can be expressed in a form that combines a real number and an imaginary number.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Complex Number
</div>
A **complex number** is a number of the form $$z = a + bi$$, where $$a$$ and $$b$$ are real numbers and $$i$$ is the imaginary unit. The set of all complex numbers is denoted by $$\mathbb{C}$$.

For a complex number $$z = a + bi$$:
- The **real part** of $$z$$ is $$\text{Re}(z) = a$$.
- The **imaginary part** of $$z$$ is $$\text{Im}(z) = b$$.
</blockquote>

Note that the real numbers are a subset of the complex numbers: any real number $$a$$ can be written as $$a + 0i$$.

### The Complex Plane

One of the most powerful ways to visualize complex numbers is by plotting them on a two-dimensional plane called the **complex plane** or **Argand diagram**. A complex number $$z = a + bi$$ is identified with the point $$(a, b)$$ in the Cartesian plane.
- The horizontal axis is the **real axis**.
- The vertical axis is the **imaginary axis**.

This geometric view allows us to think of complex numbers not just as abstract quantities but as vectors in a 2D space, which gives us intuition for operations like addition and for concepts like magnitude.

## Basic Arithmetic

Arithmetic with complex numbers follows the standard rules of algebra, with the additional rule that $$i^2 = -1$$. Let $$z_1 = a + bi$$ and $$z_2 = c + di$$.

1.  **Addition and Subtraction**: We add or subtract the real and imaginary parts separately, just like vector addition.

    $$
    z_1 \pm z_2 = (a \pm c) + (b \pm d)i
    $$

2.  **Multiplication**: We use the distributive property (or FOIL method), then simplify using $$i^2 = -1$$.

    $$
    \begin{aligned}
    z_1 z_2 &= (a + bi)(c + di) \\
    &= ac + adi + bci + bdi^2 \\
    &= ac + (ad + bc)i + bd(-1) \\
    &= (ac - bd) + (ad + bc)i
    \end{aligned}
    $$

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Multiplication
</div>
Let $$z_1 = 2 + 3i$$ and $$z_2 = 1 - 4i$$.

$$
\begin{aligned}
z_1 z_2 &= (2)(1) - (3)(-4) + ((2)(-4) + (3)(1))i \\
&= 2 + 12 + (-8 + 3)i \\
&= 14 - 5i
\end{aligned}
$$

</blockquote>

## The Complex Conjugate and Modulus

Two of the most important concepts in complex number theory are the conjugate and the modulus.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Complex Conjugate
</div>
The **complex conjugate** of a complex number $$z = a + bi$$ is denoted by $$\bar{z}$$ and is defined as:

$$
\bar{z} = a - bi
$$

Geometrically, the conjugate is the reflection of $$z$$ across the real axis.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Modulus
</div>
The **modulus** (or absolute value) of a complex number $$z = a+bi$$ is denoted by $$\vert z \vert$$ and is defined as its distance from the origin in the complex plane:

$$
\vert z \vert = \sqrt{a^2 + b^2}
$$

The modulus is always a non-negative real number.
</blockquote>

A crucial property connects the modulus and the conjugate:

$$
z \bar{z} = (a+bi)(a-bi) = a^2 - (bi)^2 = a^2 - b^2 i^2 = a^2 + b^2 = \vert z \vert^2
$$

This identity $$z \bar{z} = \vert z \vert^2$$ is extremely useful, especially for division. To divide two complex numbers, we can multiply the numerator and denominator by the conjugate of the denominator, turning the denominator into a real number.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Division
</div>
Let's compute $$\frac{2 + 3i}{1 - 4i}$$. We multiply the top and bottom by the conjugate of the denominator, which is $$\overline{1-4i} = 1+4i$$.

$$
\begin{aligned}
\frac{2 + 3i}{1 - 4i} &= \frac{2 + 3i}{1 - 4i} \cdot \frac{1 + 4i}{1 + 4i} \\
&= \frac{(2+3i)(1+4i)}{(1-4i)(1+4i)} \\
&= \frac{(2 - 12) + (8 + 3)i}{1^2 + (-4)^2} \\
&= \frac{-10 + 11i}{1 + 16} \\
&= \frac{-10 + 11i}{17} = -\frac{10}{17} + \frac{11}{17}i
\end{aligned}
$$

</blockquote>

## Polar Form and Euler's Formula

While the Cartesian form $$a+bi$$ is convenient for addition, the **polar form** is far better for multiplication and exponentiation. A complex number $$z$$ can be represented by its modulus $$r = \vert z \vert$$ and its angle $$\theta$$ (called the **argument**) relative to the positive real axis.

From trigonometry, we have $$a = r\cos\theta$$ and $$b = r\sin\theta$$. This gives the polar form:

$$
z = r(\cos\theta + i\sin\theta)
$$

This expression is dramatically simplified by one of the most beautiful and profound formulas in mathematics.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Euler's Formula
</div>
For any real number $$\theta$$,

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

</blockquote>

Using Euler's formula, the polar form of a complex number becomes a compact exponential:

$$
z = re^{i\theta}
$$

Now, multiplication and division become trivial. If $$z_1 = r_1 e^{i\theta_1}$$ and $$z_2 = r_2 e^{i\theta_2}$$:

1.  **Multiplication**: Multiply the moduli and add the angles.

    $$
    z_1 z_2 = (r_1 e^{i\theta_1})(r_2 e^{i\theta_2}) = r_1 r_2 e^{i(\theta_1 + \theta_2)}
    $$

2.  **Division**: Divide the moduli and subtract the angles.

    $$
    \frac{z_1}{z_2} = \frac{r_1 e^{i\theta_1}}{r_2 e^{i\theta_2}} = \frac{r_1}{r_2} e^{i(\theta_1 - \theta_2)}
    $$

This shows that multiplying by a complex number corresponds to a rotation and scaling in the complex plane.

## Why This Matters for Functional and Matrix Analysis

So, why did we take this detour? Complex numbers are not just a mathematical curiosity; they are essential for two main reasons in our context.

### 1. The Fundamental Theorem of Algebra

This theorem guarantees the existence of solutions to polynomial equations, which is the cornerstone of matrix eigenvalue theory.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** The Fundamental Theorem of Algebra
</div>
Every non-constant single-variable polynomial of degree $$n$$ with complex coefficients has exactly $$n$$ complex roots, counted with multiplicity.
</blockquote>

A direct consequence is that the characteristic polynomial of any $$n \times n$$ matrix $$A$$ has $$n$$ roots in $$\mathbb{C}$$. Therefore, **every $$n \times n$$ matrix has exactly $$n$$ eigenvalues**. These eigenvalues might be complex even if the matrix $$A$$ contains only real numbers. Without complex numbers, the theory of eigenvalues would be incomplete.

### 2. Inner Products in Complex Vector Spaces

In functional analysis, we work with vector spaces equipped with an inner product, which generalizes the dot product and allows us to define concepts like length (norm) and orthogonality.

- For a real vector space, the squared norm of a vector $$x$$ is $$\langle x, x \rangle = \sum_i x_i^2$$, which is always non-negative.
- If we naively extended this to a complex vector $$z$$, we could get $$\langle z, z \rangle = \sum_i z_i^2$$, which might not be a real number. For example, if $$z = [i]$$, its squared "length" would be $$i^2 = -1$$. This is problematic for defining a distance.

The solution is to use the complex conjugate. The standard **complex inner product** on $$\mathbb{C}^n$$ is defined as:

$$
\langle u, v \rangle = \sum_{i=1}^n u_i \bar{v}_i
$$

<details class="details-block" markdown="1">
<summary markdown="1">
**Note on Convention.**
</summary>
This definition is common in physics and engineering, often written in matrix notation as $$v^H u$$, where $$v^H$$ is the conjugate transpose of $$v$$. Some math literature defines it as $$\sum_i \bar{u}_i v_i$$. The choice is a matter of convention, but the key is that one of the vectors must be conjugated.
</details>

With this definition, the squared norm of a complex vector $$z$$ is:

$$
\Vert z \Vert^2 = \langle z, z \rangle = \sum_{i=1}^n z_i \bar{z}_i = \sum_{i=1}^n \vert z_i \vert^2
$$

This sum is guaranteed to be a non-negative real number, providing a solid foundation for defining length and distance in complex vector spaces (like Hilbert spaces).

## Conclusion

This primer introduced the essential concepts of complex numbers needed for our journey into functional and matrix analysis. We've seen how they extend the real numbers, how to perform basic arithmetic, and how their geometric and polar representations provide deep insights. Most importantly, complex numbers provide the algebraic completeness required for eigenvalue analysis and the structural foundation for defining norms in complex vector spaces. With this background, we are now better equipped to tackle more advanced topics.

---
layout: post
title: Testing SCSS templates
date: 2025-10-14 00:12 -0400
description:
image:
categories:
-
-
tags:
-
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


  Note that the following will incorrectly render the block equation as an inline equation due to a lack of newline after the MathJax delimiter:

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

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The natural numbers \\(\mathbb{N}\\)
</div>
The natural numbers are defined as \\(inline\\).
\\[block\\]
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Definition 1.1 (Natural Numbers)
</div>
The set of natural numbers is $$\mathbb{N} = \{1,2,3,\ldots\}.$$
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
Theorem 1.2 (Fundamental Theorem of Arithmetic)
</div>
Every integer $$n>1$$ can be written uniquely as a product of primes.
</blockquote>

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
Lemma 1.3
</div>
If $$p$$ divides $$ab$$ and $$p$$ is prime, then $$p$$ divides $$a$$ or $$b$$.
</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Proposition 1.4
</div>
All squares are congruent to 0 or 1 mod 4.
</blockquote>

<blockquote class="box-corollary" markdown="1">
<div class="title" markdown="1">
Corollary 1.5
</div>
If $$n$$ is odd, then $$n^2 \equiv 1 \pmod{8}.$$
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Example 1
</div>
Compute $$\int_0^1 x^2\,dx = \tfrac13.$$
</blockquote>

<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
Remark
</div>
The integral result above generalizes to $$\int_0^1 x^n\,dx = \frac{1}{n+1}.$$
</blockquote>

<details class="details-block box-proof" markdown="1">
<summary markdown="1">
Proof (sketch)
</summary>
We apply the power rule for integrals:
$$
\int x^n\,dx = \frac{x^{n+1}}{n+1}+C.
$$
Evaluating from 0 to 1 gives the claimed formula.
<span class="qed">$\square$</span>
</details>

<blockquote class="box-principle" markdown="1">
Energy cannot be created or destroyed—only transformed.
</blockquote>

<blockquote class="box-axiom" markdown="1">
Through any two distinct points there exists exactly one line.
</blockquote>

<blockquote class="box-postulate" markdown="1">
The sum of the angles in a triangle is $$180^\circ.$$
</blockquote>

<blockquote class="box-conjecture" markdown="1">
There are infinitely many twin primes.
</blockquote>

<blockquote class="box-claim" markdown="1">
For all real $$x>0$$, we have $$\ln(x) < x-1.$$
</blockquote>

<blockquote class="box-notation" markdown="1">
We denote by $$\lfloor x \rfloor$$ the greatest integer less than or equal to $$x$$.
</blockquote>

<blockquote class="box-algorithm" markdown="1">
<div class="title" markdown="1">
Algorithm 1 (Euclidean Algorithm)
</div>
1. Given integers $$a,b$$ with $$a>b>0$$.  
2. Divide $$a = bq + r$$.  
3. Replace $$(a,b) \leftarrow (b,r)$$ and repeat until $$r=0$$.  
4. Output $$b$$ as $$\gcd(a,b)$$.
</blockquote>

<blockquote class="box-problem" markdown="1">
Find all real solutions to $$x^2 - 4x + 3 = 0.$$
</blockquote>

<blockquote class="box-solution" markdown="1">
$$x = 1 \text{ or } x = 3.$$
</blockquote>

<blockquote class="box-assumption" markdown="1">
Assume all functions are continuously differentiable on $$[0,1].$$
</blockquote>

<blockquote class="box-convention" markdown="1">
Vectors are written in **bold lowercase**, e.g. $$\mathbf{v}$$.
</blockquote>

<blockquote class="box-fact" markdown="1">
Every continuous function on $$[a,b]$$ attains a maximum and minimum.
</blockquote>


<!-- starts CLOSED -->
<details class="details-block box-theorem" markdown="1">
<summary markdown="1">
Theorem 2 (Collapsible Closed)
</summary>
Hidden proof content …  
$$a^2 + b^2 = c^2$$
</details>

<!-- starts OPEN -->
<details class="details-block box-theorem" open markdown="1">
<summary markdown="1">
Theorem 3 (Collapsible Open)
</summary>
Visible by default.  
$$E = mc^2$$
</details>

<blockquote class="box-info" markdown="1">
**Info.** This is general information.
</blockquote>

<blockquote class="box-tip" markdown="1">
**Tip.** You can collapse any proof block by wrapping it in `<details>`.
</blockquote>

<blockquote class="box-warning" markdown="1">
**Warning.** Ensure math delimiters have blank lines before/after in lists.
</blockquote>

<blockquote class="box-danger" markdown="1">
**Danger.** Never reference nonexistent images – it will break the build.
</blockquote>

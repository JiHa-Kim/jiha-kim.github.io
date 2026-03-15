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

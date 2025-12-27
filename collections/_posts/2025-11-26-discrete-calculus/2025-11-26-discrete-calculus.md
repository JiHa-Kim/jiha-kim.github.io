---
layout: post
title: Discrete Calculus
date: 2025-11-26 05:52 +0000
description: An introduction to Discrete Calculus, a theory for sums and differences of sequences as opposed to derivatives and integrals of functions in infinitesimal calculus.
image: ./discrete_calculus_thumbnail.svg
categories:
- Mathematical Foundations
tags:
- Sequence
- Forward Difference
- Finite Difference
- Sum
- Generating Function
- Binomial Coefficient
- Stirling Number
---

### Introduction
The modern term "calculus" pretty much always refers in common contexts to the study of infinitesimal calculus for continuous changes. However, there is a parallel world for the discrete case, with finite changes. This theory is quite overpowered for discrete math, e.g. in competitive mathematics and programming.

These worlds turns out to be very similar to each other, and learning about the discrete version makes some intuition for results much more clear. Typically, we are better at keeping track of a finite number of things in the discrete case rather than infinitely many infinitesimal changes in the continuous case.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Gradient descent discretizes gradient flow
</div>
Numerical algorithms to carry out large computations on computers are almost always done in discrete bits. In ML, the gradient descent algorithm models the gradient flow ODE:

<div class="math-block" markdown="0">
\[
\frac{d}{dt}w_{t}=-\nabla_{w_{t}} f(w_{t})
\]
</div>

for some weights <span class="math-inline" markdown="0">\(w_{t}\)</span> evolving through training time <span class="math-inline" markdown="0">\(t\)</span> under a loss <span class="math-inline" markdown="0">\(f\)</span>.
We approximate by discretizing the derivative to a *finite difference quotient*.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Forward difference operator
</div>
The forward difference operator

<div class="math-block" markdown="0">
\[
\Delta_{h}f(t):=f(t+h)-f(t)
\]
</div>

</blockquote>

Note that by definition of a derivative,


<div class="math-block" markdown="0">
\[
\frac{dw_{t}}{dt}=\lim_{ h \to 0 } \frac{w_{t+h}-w_{t}}{h}=\lim_{ h \to 0 } \frac{\Delta_{h}w_{t}}{\Delta_{h}t}
\]
</div>

So in the discrete case, we use a finite <span class="math-inline" markdown="0">\(h\)</span> instead of the limit as <span class="math-inline" markdown="0">\(h\to 0\)</span>. This gives us the finite difference quotient <span class="math-inline" markdown="0">\(\frac{\Delta_{h}w_{t}}{\Delta_{h}t}=\frac{w_{t+h}-w_{t}}{h}\)</span>.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Gradient descent algorithm
</div>
Discretizing the gradient flow ODE through the forward difference operator is called forward Euler integration, and recovers the gradient descent algorithm.
We replace <span class="math-inline" markdown="0">\(\frac{d}{dt}\)</span> by <span class="math-inline" markdown="0">\(\frac{\Delta_{h}}{\Delta_{h}t}\)</span> in the gradient flow ODE:

<div class="math-block" markdown="0">
\[
\frac{\Delta_{h}}{\Delta_{h}t}w_{t}=-\nabla f(w_{t})=\frac{w_{t+h}-w_{t}}{h}
\]
</div>

Solving for <span class="math-inline" markdown="0">\(w_{t+h}\)</span> gives the explicit form:

<div class="math-block" markdown="0">
\[
w_{t+h}=w_{t}-h\nabla f(w_{t})
\]
</div>

</blockquote>

Indeed, discrete calculus is more far-reaching than you might think. Now, we can index discrete objects using the natural numbers. Hence, we will often think of time as steps <span class="math-inline" markdown="0">\(t=1,2,3,\dots\)</span> and analyze the specific case with normalized increment to <span class="math-inline" markdown="0">\(h=1\)</span> so that <span class="math-inline" markdown="0">\(\Delta f(x)=f(x+1)-f(x)\)</span>. 

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Forward difference operator <span class="math-inline" markdown="0">\(\Delta\)</span>
</div>

<div class="math-block" markdown="0">
\[
\Delta f(x):=f(x+1)-f(x)
\]
</div>


(Note: the notation <span class="math-inline" markdown="0">\(\Delta\)</span> clashes with the symbol commonly used for the Laplacian, so be careful to avoid confusion.)
</blockquote>

This has the convenient effect that the denominator in the finite difference quotient just cancels, so we only need to keep track of the difference itself:

<div class="math-block" markdown="0">
\[
\frac{\Delta f(x)}{\Delta x}=\frac{f(x+1)-f(x)}{(x+1)-x}=\frac{f(x+1)-f(x)}{1}=\Delta f(x)
\]
</div>


### Difference and Sum Operators
From the section above, the discrete analog to the derivative is the difference <span class="math-inline" markdown="0">\(\Delta\)</span>. Then what is the integral? Straightforwardly, it is simply the sum operator <span class="math-inline" markdown="0">\(\sum\)</span>. Then, some results in discrete calculus seem very familiar to their infinitesimal counterparts.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Properties of forward difference and sum
</div>
- Linearity: for constants <span class="math-inline" markdown="0">\(a,b\)</span>, 

<div class="math-block" markdown="0">
\[
\Delta (af+bg)=a\Delta f+b\Delta g
\]
</div>


<div class="math-block" markdown="0">
\[
\sum(af+bg)=a\sum f+b\sum g
\]
</div>

- Constant rule: for a constant <span class="math-inline" markdown="0">\(c\)</span>, <span class="math-inline" markdown="0">\(\Delta c=0\)</span>.
- Fundamental theorem of discrete calculus:
1. Telescoping: 
<div class="math-block" markdown="0">
\[
\sum_{k=a}^{b}\Delta f(k)=f(b+1)-f(a)
\]
</div>

2. This doesn't really have a name, maybe we should call it "microscoping" in spirit of the other part, since we zoom in on a single term. 
<div class="math-block" markdown="0">
\[
\frac{\Delta}{\Delta n} \sum_{k=a}^{n-1}f(k)=f(n)
\]
</div>

- Compared to the infinitesimal case, where we have <span class="math-inline" markdown="0">\(\int_{[a,b)}=\int_{[a,b]}\)</span> so <span class="math-inline" markdown="0">\(\int_{[a,b)} \frac{df}{dx} \,dx=f(b)-f(a)\)</span>, the distinction does matter in the discrete case, where we sum <span class="math-inline" markdown="0">\(\Delta f(k)\)</span> over the integer interval <span class="math-inline" markdown="0">\([a,n)=\{ a, \dots, n-1 \}\)</span> to recover <span class="math-inline" markdown="0">\(f(n)\)</span>.
However, in the case of the product rule, the discrete version differs from its infinitesimal counterpart, because we don't discard the second-order term:
- Discrete product rule: <span class="math-inline" markdown="0">\(\Delta (fg)=f\Delta g+\Delta f g+\Delta f \Delta g\)</span>
- Compare to continuous product rule: <span class="math-inline" markdown="0">\(d(fg)=(f+df)(g+dg)-fg=f\,dg+g\,df+\cancel{ df\,dg }\)</span>
</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Summation by parts
</div>
- The product rule can be simplified to 

<div class="math-block" markdown="0">
\[
\Delta(fg)(x)=f(x+1)\Delta g(x)+\Delta f(x)g(x)=\Delta f(x)g(x+1)+f(x)\Delta g(x)
\]
</div>

Re-arranging and summing both sides gives summation by parts:

<div class="math-block" markdown="0">
\[
\sum_{x=a}^{b-1} f(x) \Delta g(x)=f(b)g(b)-f(a)g(a)-\sum_{x=a}^{b-1} \Delta f(x)g(x+1)
\]
</div>

</blockquote>

When I first learned about the fundamental theorems of calculus, I just thought of them as mechanical rules to apply to crunch algebra. But the discrete versions seem so obvious, and they explain what's really going on: because we are accumulating changes, everything along your path will eventually cancel out, leaving only the endpoints relevant. 

A similar logic generalizes in several dimensions if you think about walking along a square, or a cube. In the continuous case, Green's theorem and Stokes' generalized integration theorem basically say that a change in one direction will be cancelled by a change in another direction unless you lie along the boundary.

This image from [Wikipedia](https://en.wikipedia.org/wiki/Discrete_calculus#Discrete_differential_forms:_cochains) sums it well:
![Paths in the interior cancel each other out](https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Stokes_patch.svg/2560px-Stokes_patch.svg.png)

### Polynomials: Falling Powers
The first interesting functions we differentiate in calculus are the monomials:
<blockquote class="box-fact" markdown="1">
<div class="title" markdown="1">
Derivative of <span class="math-inline" markdown="0">\(x^n\)</span>
</div>

<div class="math-block" markdown="0">
\[
\frac{d}{dx}x^n=nx^{n-1}
\]
</div>

</blockquote>

Naturally, let's start with them for the discrete case as well. For example, take 

<div class="math-block" markdown="0">
\[
\Delta x=(x+1)-x=1
\]
</div>
So far so good. Let's move up a degree.

<div class="math-block" markdown="0">
\[
\Delta x^2=(x+1)^2-x^2=2x+1
\]
</div>

Huh. That's not quite the same as in the continuous version. Strange. Let's try a few more:

<div class="math-block" markdown="0">
\[
\Delta x^3=(x+1)^3-x^3=3x^2+3x+1
\]
</div>


<div class="math-block" markdown="0">
\[
\Delta x^4=(x+1)^4-x^4=4x^3+6x^2+4x+1
\]
</div>

Okay, this is really not going as well as we expected, given that all the properties of operators seemed to match up pretty well before. Let's take a step back and think through what goes differently in the discrete versus the continuous case.

Typically, we do the proof using the binomial theorem, <span class="math-inline" markdown="0">\((a+b)^n=\sum_{k=0}^{n} \binom{n}{k}a^k b^{n-k}\)</span>.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Derivative of <span class="math-inline" markdown="0">\(x^n\)</span>
</div>

<div class="math-block" markdown="0">
\[
\frac{d}{dx}x^n=\lim_{ h \to 0 } \frac{(x+h)^n-x^n}{h}=\lim_{ h \to 0 } \frac{1}{h}\sum_{k=1}^{n}\binom{n}{k}h^k x^{n-k}=\binom{n}{1}x^{n-1}=nx^{n-1}
\]
</div>

</blockquote>

The crucial step here is that <span class="math-inline" markdown="0">\(\frac{1}{h}\cdot h^k=h^{k-1}\to 0\)</span> as <span class="math-inline" markdown="0">\(h\to 0\)</span> for any <span class="math-inline" markdown="0">\(k>1\)</span>. But in the discrete case, this doesn't happen: the terms survive as <span class="math-inline" markdown="0">\(h=1\)</span>. So for <span class="math-inline" markdown="0">\(\Delta x^n=(x+1)^n-x^n\)</span> we have that the coefficients <span class="math-inline" markdown="0">\((x+1)^n\)</span> are literally just generated by the binomial expansion in Pascal's triangle, and we delete the highest degree due to <span class="math-inline" markdown="0">\(-x^n\)</span>.


<div class="math-block" markdown="0">
\[
\begin{array}{|c||c|c|c|c|c|c|}
\hline
\textbf{n} & \mathbf{x^0} & \mathbf{x^1} & \mathbf{x^2} & \mathbf{x^3} & \mathbf{x^4} & \mathbf{x^5} \\
\hline
\hline
\textbf{0} & 1 & & & & & \\
\hline
\textbf{1} & 1 & 1 & & & & \\
\hline
\textbf{2} & 1 & 2 & 1 & & & \\
\hline
\textbf{3} & 1 & 3 & 3 & 1 & & \\
\hline
\textbf{4} & 1 & 4 & 6 & 4 & 1 & \\
\hline
\textbf{5} & 1 & 5 & 10 & 10 & 5 & 1 \\
\hline
\end{array}
\]
</div>

_Binomial expansion of <span class="math-inline" markdown="0">\((x+1)^n\)</span> for <span class="math-inline" markdown="0">\(n=1,2,3,4,5\)</span>_

But having explored linearity so far, we know that we should be able to use linear combinations to construct a polynomial basis that would indeed satisfy an identity like <span class="math-inline" markdown="0">\(\frac{d}{dx}x^n=nx^{n-1}\)</span>. Let's try it out.

By exploiting the linearity of <span class="math-inline" markdown="0">\(\Delta\)</span>, let's construct polynomials <span class="math-inline" markdown="0">\(p_{n}(x)\)</span> so that <span class="math-inline" markdown="0">\(\Delta p_{n}(x)=np_{n-1}(x)\)</span>. From the table, we start at <span class="math-inline" markdown="0">\(\Delta x^0=\Delta 1=0 \implies p_{0}=1\)</span> from the constant rule. Also, <span class="math-inline" markdown="0">\(\Delta x=1=1x^0\)</span>, giving <span class="math-inline" markdown="0">\(p_{1}=x\)</span>. 

But the problem arises starting from quadratics, where <span class="math-inline" markdown="0">\(\Delta x^2=2x+1\)</span>. But <span class="math-inline" markdown="0">\(2x=\frac{d}{dx}x^2\)</span> is the part we are looking for, so let's isolate it and use the fundamental theorem of discrete calculus. With it, the equation <span class="math-inline" markdown="0">\(\Delta p_{n}=np_{n-1}\)</span> is equivalent to <span class="math-inline" markdown="0">\(p_{n}=\sum np_{n-1}\)</span>, i.e. we are computing its anti-difference = sum.

<div class="math-block" markdown="0">
\[
2x=\Delta x^2-1=\Delta x^2-\Delta x=\Delta (x^2-x)=\Delta [x(x-1)]=\Delta p_{2}
\]
</div>

Nice, <span class="math-inline" markdown="0">\(p_{2}=x(x-1)\)</span>. Let's try <span class="math-inline" markdown="0">\(\Delta p_{3}=3p_{2}=3x(x-1)=3x^2-3x\)</span>:

<div class="math-block" markdown="0">
\[
3x^2-3x=(\Delta x^3-3x-1)-3x=\Delta x^3-6x-1=\Delta x^3-3\Delta x(x-1)-\Delta x
\]
</div>

Hence, 

<div class="math-block" markdown="0">
\[
p_{3}=x^3-3x(x-1)-x=x(x^2-3x+3-1)=x(x^2-3x+2)=x(x-1)(x-2)
\]
</div>

Hold on, that seems like a suspicious pattern. So far, we got:

<div class="math-block" markdown="0">
\[
\begin{align}
p_{0} &= 1 \\
p_{1} &= x \\
p_{2} &= x(x-1) \\
p_{3} &= x(x-1)(x-2)
\end{align}
\]
</div>

It seems like our polynomials will look like <span class="math-inline" markdown="0">\(x(x-1)(x-2)(x-3)\dots\)</span> as we go on. 
<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Try one more!
</div>
As we did before, compute <span class="math-inline" markdown="0">\(p_{4}\)</span> from <span class="math-inline" markdown="0">\(4p_{3}=\Delta p_{4}\)</span> and verify that the pattern still holds: <span class="math-inline" markdown="0">\(p_{4}=x(x-1)(x-2)(x-3)\)</span>.
</blockquote>

Let's see if we can verify this. Define the falling power:
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Falling power <span class="math-inline" markdown="0">\(x^\underline{n}\)</span>
</div>

<div class="math-block" markdown="0">
\[
x^\underline{n}:=\underbrace{ x(x-1)(x-2)\dots(x-(n-1)) }_{ n\text{ terms} }
\]
</div>

By convention, <span class="math-inline" markdown="0">\(x^\underline{0}=1\)</span>.
</blockquote>

Now, let's try taking <span class="math-inline" markdown="0">\(\Delta x^\underline{n}\)</span>:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Forward difference of <span class="math-inline" markdown="0">\(x^\underline{n}\)</span>
</div>

<div class="math-block" markdown="0">
\[
\begin{align}
\Delta x^\underline{n}&=(x+1)\underbrace{ \textcolor{red}{x(x-1)\dots(x-n)} }_{ =x^\underline{n-1} }-\underbrace{ \textcolor{red}{x(x-1)(x-2)\dots(x-n)} }_{ =x^\underline{n-1} }(x-n+1) \\
&=x^\underline{n-1}[\cancel{ x }+\cancel{ 1 }-(\cancel{ x }-n+\cancel{ 1 })] \\
&=nx^\underline{n-1}
\end{align}
\]
</div>

</blockquote>

<blockquote class="box-corollary" markdown="1">
<div class="title" markdown="1">
Sum of <span class="math-inline" markdown="0">\(x^\underline{n}\)</span>
</div>
Summing and telescoping the identity <span class="math-inline" markdown="0">\(x^\underline{n}=\Delta \frac{1}{n+1} x^\underline{n+1}\)</span>,

<div class="math-block" markdown="0">
\[
\sum_{x=a}^{b-1} x^\underline{n}=\sum_{x=a}^{b-1}\Delta \frac{1}{n+1}x^\underline{n+1}=\frac{1}{n+1}b^\underline{n+1}-\frac{1}{n+1}a^\underline{n+1}
\]
</div>

</blockquote>

So we've found exactly the result we were looking for: when using the forward difference instead of the derivative, we have <span class="math-inline" markdown="0">\(\Delta x^\underline{n}=nx^\underline{n-1}\)</span> compared to <span class="math-inline" markdown="0">\(Dx^n=nx^{n-1}\)</span>. 

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Characterization of falling power basis <span class="math-inline" markdown="0">\(x^\underline{n}\)</span>
</div>
<span class="math-inline" markdown="0">\(x^\underline{n}\)</span> is the unique sequence of polynomials <span class="math-inline" markdown="0">\(p_{n}(x)\)</span> that satisfies

- <span class="math-inline" markdown="0">\(\Delta p_{n}(x)=p_{n}(x+1)-p_{n}(x)=np_{n-1}(x)\)</span>
- <span class="math-inline" markdown="0">\(p_{0}(x)=1\)</span>
- <span class="math-inline" markdown="0">\(n\ge 1 \implies p_{n}(0)=0\)</span> (only <span class="math-inline" markdown="0">\(p_{0}\)</span> has a non-zero constant term)
</blockquote>
 
<details class="box-proof" markdown="1">
<summary markdown="1">
Characterization of falling power basis <span class="math-inline" markdown="0">\(x^\underline{n}\)</span>
</summary>
1. Existence

<div class="math-block" markdown="0">
\[
\Delta x^\underline{n} =(x+1)^\underline{n}-x^\underline{n}=(x+1)x^\underline{n-1}-x^\underline{n-1}(x-n+1)=nx^\underline{n-1}
\]
</div>


<div class="math-block" markdown="0">
\[
x^\underline{0}=1
\]
</div>


<div class="math-block" markdown="0">
\[
n\ge 1 \implies 0^\underline{n}=0(-1)(-2)\dots(-n+1)=0
\]
</div>

2. Uniqueness
By the fundamental theorem of discrete calculus, sum <span class="math-inline" markdown="0">\(\sum_{x=0}^{k-1}\)</span> both sides of the condition <span class="math-inline" markdown="0">\(\Delta p_{n}(x)=np_{n-1}(x)\)</span> to get

<div class="math-block" markdown="0">
\[
p_{n}(k)\cancel{ -p_{n}(0) }=\sum_{x=0}^{k-1}np_{n-1}(x)
\]
</div>

for all <span class="math-inline" markdown="0">\(k\in \mathbb{Z}\)</span>. But choosing <span class="math-inline" markdown="0">\(\deg p_{n}+1\)</span> points is enough to uniquely determine <span class="math-inline" markdown="0">\(p_{n}\)</span>.
</details>

Once we have this, what can we do with it? Let's see the following problem.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Sum of <span class="math-inline" markdown="0">\(n\)</span> first squares
</div>

<div class="math-block" markdown="0">
\[
\sum_{k=1}^{n}k^2=1^2+2^2+3^2+\dots+n^2=\;?
\]
</div>

</blockquote>

Knowing that the basis of <span class="math-inline" markdown="0">\(n^\underline{r}\)</span> is very convenient for differences and sums, we could simply decompose <span class="math-inline" markdown="0">\(n^2\)</span> in terms of <span class="math-inline" markdown="0">\(\{ n^\underline{0},n^\underline{1},n^\underline{2} \}=\{ 1,n,n^\underline{2}\}\)</span>. Write <span class="math-inline" markdown="0">\(k^2=(k^2-k)+k=k^\underline{2}+k\)</span>, so that

<div class="math-block" markdown="0">
\[
\begin{align}
\sum_{k=1}^{n} k^2&=\sum_{k=0}^nk^2 \\
&=\sum_{k=0}^{n} k^\underline{2}+\sum_{k=0}^{n} k \\
&=\frac{1}{3}(n+1)^\underline{3}+\frac{1}{2}(n+1)^\underline{2} \\
&=\frac{\textcolor{red}{2}}{6}\textcolor{aqua}{(n+1)n}\textcolor{red}{(n-1)}+\frac{\textcolor{red}{3}}{6}\textcolor{aqua}{(n+1)n} \\
&=\boxed{ \frac{1}{6}n(n+1)(2n+1) } \\ 
\end{align}
\]
</div>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Sum of first <span class="math-inline" markdown="0">\(n\)</span> cubes
</div>

<div class="math-block" markdown="0">
\[
\sum_{k=1}^{n}k^3=1^3+2^3+3^3+\dots+n^3=\; ?
\]
</div>

</blockquote>

Here is a combinatorics problem. Try it out using the following lemma:

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
Difference and summation of binomial coefficient
</div>
The binomial coefficient is given by

<div class="math-block" markdown="0">
\[
\binom{x}{k}=\frac{x^\underline{k}}{k!}=\frac{x^\underline{k}}{k^\underline{k}}
\]
</div>

Then:
- Pascal's identity:

<div class="math-block" markdown="0">
\[
\frac{\Delta}{\Delta x} \binom{x}{k}=\binom{x+1}{k}-\binom{x}{k}=\binom{x}{k-1}
\]
</div>

- Hockey-stick identity:

<div class="math-block" markdown="0">
\[
\sum_{x=0}^{n-1}\binom{x}{k}=\binom{n}{k+1}
\]
</div>

</blockquote>

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Up and Down
</div>
How many 4-digit positive integers <span class="math-inline" markdown="0">\(d_{1}d_{2}d_{3}d_{4}\)</span> satisfy the inequalities <span class="math-inline" markdown="0">\(d_{1}>d_{2}<d_{3}>d_{4}\)</span>?
</blockquote>

<details class="box-solution" markdown="1">
<summary markdown="1">
Up and Down
</summary>
Note that the conditions make <span class="math-inline" markdown="0">\(d_{2},d_{4}\in \{ 0,1,2,\dots,9 \}\)</span> while <span class="math-inline" markdown="0">\(d_{1},d_{3}\in \{ 1,2,\dots,9 \}\)</span>, so we don't need to worry about leading digit <span class="math-inline" markdown="0">\(0\)</span>. We can encode our constraints into a big sum, counting 1 for each valid case:


<div class="math-block" markdown="0">
\[
\begin{align}
\sum_{d_{1}=1}^{9} \sum_{d_{2}=0}^{d_{1}-1}\sum_{d_{3}=d_{2}+1}^{9}\sum_{d_{4}=0}^{d_{3}-1}1
&=\sum_{d_{1}=1}^{9} \sum_{d_{2}=0}^{d_{1}-1}\sum_{d_{3}=d_{2}+1}^{9} d_{3} \\
&=\sum_{d_{1}=1}^{9} \sum_{d_{2}=0}^{d_{1}-1}\left(\underbrace{ \binom{10}{2} }_{ =45 }-\binom{d_{2}+1}{2}\right) \\
&=\sum_{d_{1}=1}^{9}\left(45d_{1}-\sum_{n=1}^{d_{1}}\binom{n}{2}\right) \\
&=\sum_{d_{1}=1}^{9}\left(45d_{1}-\left.\binom{n}{3}\right\vert_{\cancel{ 1 }}^{d_{1}+1}\right) \\
&=\sum_{d_{1}=1}^{9}\left( 45d_{1}-\binom{d_{1}+1}{3} \right)  \\
&=45\binom{10}{2}-\sum_{n=2}^{10} \binom{n}{3} \\
&=45^2-\left.\binom{n}{4}\right\vert_{\cancel{ 2 }}^{11} \\
&=2025-\binom{11}{4} \\
&=2025-330 \\
&=\boxed{ 1695 }
\end{align}
\]
</div>

</details>

#### Stirling Numbers
We have seen that the basis <span class="math-inline" markdown="0">\(x^\underline{n}\)</span> is easier to work with than <span class="math-inline" markdown="0">\(x^n\)</span> in the discrete case. So, how can we convert between the two in general? They are given by the following linear combinations:


<div class="math-block" markdown="0">
\[
x^\underline{n}=\sum_{k=0}^{n} s(n,k)x^k
\]
</div>


<div class="math-block" markdown="0">
\[
\newcommand{\bracenom}{\genfrac{\lbrace}{\rbrace}{0pt}{}} 
x^n=\sum_{k=0}^{n} \bracenom{n}{k}
\]
</div>

where <span class="math-inline" markdown="0">\(s(n,k)\)</span> are the *signed Stirling numbers of the first kind* and and <span class="math-inline" markdown="0">\(\newcommand{\bracenom}{\genfrac{\lbrace}{\rbrace}{0pt}{}} \bracenom{n}{k}\)</span> are the *unsigned Stirling numbers of the second kind*. They have many interpretations in combinatorics, but we won't go over them. To help remember them better, I will use the following notation suggested by Gemini 3.0 Pro that I found really great: <span class="math-inline" markdown="0">\(\binom{\text{Target}}{\text{Source}}\)</span>.

<div class="math-block" markdown="0">
\[
x^\underline{n}=\sum_{k=0}^{n}\binom{\underline{n}}{k}x^k
\]
</div>


<div class="math-block" markdown="0">
\[
x^n=\sum_{k=0}^{n}\binom{n}{\underline{k}}x^\underline{k}
\]
</div>

As you can see, the powers line up with the notation in the coefficients.
We can also use matrix notation for change of basis. Let <span class="math-inline" markdown="0">\(\mathbf{x^{\underline{n}}}\)</span> be the column vector of falling factorials and <span class="math-inline" markdown="0">\(\mathbf{x^n}\)</span> be the column vector of standard powers:


<div class="math-block" markdown="0">
\[
\mathbf{x^{\underline{n}}} = \begin{bmatrix} x^{\underline{0}} \\ x^{\underline{1}} \\ x^{\underline{2}} \\ \vdots \\ x^{\underline{n}} \end{bmatrix}, \quad
\mathbf{x^n} = \begin{bmatrix} x^0 \\ x^1 \\ x^2 \\ \vdots \\ x^n \end{bmatrix}
\]
</div>


We can represent the transformations as matrix multiplications involving lower-triangular matrices. For <span class="math-inline" markdown="0">\(x^\underline{n} \mapsto x^n\)</span>:


<div class="math-block" markdown="0">
\[
\begin{bmatrix} x^{\underline{0}} \\ x^{\underline{1}} \\ x^{\underline{2}} \\ \vdots \\ x^{\underline{n}} \end{bmatrix}
=
\begin{bmatrix} 
\binom{\underline{0}}{0} & 0 & 0 & \cdots & 0 \\
\binom{\underline{1}}{0} & \binom{\underline{1}}{1} & 0 & \cdots & 0 \\
\binom{\underline{2}}{0} & \binom{\underline{2}}{1} & \binom{\underline{2}}{1} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\binom{\underline{n}}{0} & \binom{\underline{n}}{1} & \binom{\underline{n}}{1} & \cdots & \binom{\underline{n}}{n}
\end{bmatrix}
\begin{bmatrix} x^0 \\ x^1 \\ x^2 \\ \vdots \\ x^n \end{bmatrix}
\]
</div>

For <span class="math-inline" markdown="0">\(x^n \mapsto x^\underline{n}\)</span>:

<div class="math-block" markdown="0">
\[
\begin{bmatrix} x^{0} \\ x^{1} \\ x^{2} \\ \vdots \\ x^{n} \end{bmatrix}
=
\begin{bmatrix} 
\binom{0}{\underline{0}} & 0 & 0 & \cdots & 0 \\
\binom{1}{\underline{0}} & \binom{1}{\underline{1}} & 0 & \cdots & 0 \\
\binom{2}{\underline{0}} & \binom{2}{\underline{1}} & \binom{2}{\underline{2}} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\binom{n}{\underline{0}} & \binom{n}{\underline{1}} & \binom{n}{\underline{2}} & \cdots & \binom{n}{\underline{n}}
\end{bmatrix}
\begin{bmatrix} x^{\underline{0}} \\ x^{\underline{1}} \\ x^{\underline{2}} \\ \vdots \\ x^{\underline{n}} \end{bmatrix}
\]
</div>

By nature of change of basis, these matrices are inverses of each other. Therefore,

<div class="math-block" markdown="0">
\[
\sum_{k} \binom{\underline{n}}{k}\binom{k}{\underline{m}}=\delta_{nm}
\]
</div>

where <span class="math-inline" markdown="0">\(n=m\implies \delta_{nm}=1\)</span> else <span class="math-inline" markdown="0">\(n\ne m \implies \delta_{nm}=0\)</span> is the Kronecker delta.

Now that we have defined a notation for our basis conversions, how can we actually compute the coefficients? To solve this problem, we will introduce perhaps the most overpowered tool in all of discrete mathematics, the theory of [generating functions](https://math.mit.edu/~goemans/18310S15/generating-function-notes.pdf).

### Generating functions

<blockquote class="box-info" markdown="1">

A generating function is a clothesline on which we hang up a sequence of numbers for display.
- Herbert S. Wilf, [*generatingfunctionology*](https://www2.math.upenn.edu/~wilf/gfology2.pdf)
</blockquote>

[Video by 3Blue1Brown](https://www.youtube.com/watch?v=bOXCLR3Wric) on generating functions.

The core idea between generating functions is to take a sequence of numbers <span class="math-inline" markdown="0">\(a_{0},a_{1},a_{2},\dots\)</span> and attach them to a power of some dummy variable <span class="math-inline" markdown="0">\(x\)</span>.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Generating function
</div>
Given a sequence <span class="math-inline" markdown="0">\(a_{0},a_{1},a_{2},\dots\)</span>, its generating function <span class="math-inline" markdown="0">\(G(x)\)</span> is given by

<div class="math-block" markdown="0">
\[
G(x)=a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots=\sum_{n \ge 0}a_{n}x^n
\]
</div>

</blockquote>

The natural first question to ask is: are you crazy? Doesn't this just make everything even more complicated? On its own, yes, of course it accomplishes nothing. But the motivation behind this approach is that relationships concerning the sequence are often easier to encode using the generating functions, and we can then manipulate these generating functions algebraically. Let's see an example.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Geometric series
</div>
Suppose you have the sequence <span class="math-inline" markdown="0">\(1,1,1,1,\dots\)</span>. Its generating function is derive like this:

<div class="math-block" markdown="0">
\[
G(x)=1+x+x^2+\dots=\sum_{n \ge 0}x^n
\]
</div>


<div class="math-block" markdown="0">
\[
xG(x)=x+x^2+\dots=G(x)-1
\]
</div>


<div class="math-block" markdown="0">
\[
\boxed{ G(x)=\frac{1}{1-x} }
\]
</div>

Normally, in calculus, you would have to assert that this sum only converges for <span class="math-inline" markdown="0">\(\vert x\vert <1\)</span>. However, in the basic setting of generating functions in discrete math, we said in the beginning that <span class="math-inline" markdown="0">\(x\)</span> is a dummy variable, so it doesn't actually have to hold any meaning. Of course, as is common in mathematics, it is possible to draw connections, here between the worlds of discrete and continuous math, but we'll get to that later.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Reciprocal factorials
</div>
The sequence <span class="math-inline" markdown="0">\(\frac{1}{0!},\frac{1}{1!}, \frac{1}{2!}, \frac{1}{3!},\dots\)</span> has generating function

<div class="math-block" markdown="0">
\[
G(x)=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\dots=\sum_{n \ge 0} \frac{x^n}{n!}=e^x
\]
</div>

</blockquote>

So, from these examples alone, we see that generating functions can simplify to something simpler. But that doesn't solve our problems: to calculate the generating function, we already knew exactly what our sequence was, but the main challenge with sequences is usually to understand them given only some indirect relationships such as recurrence relations.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Fibonacci sequence
</div>
The famous Fibonacci sequence is defined recursively through the equation

<div class="math-block" markdown="0">
\[
f_{n+2}=f_{n+1}+f_{n}, \quad n\ge 0
\]
</div>

with initial values <span class="math-inline" markdown="0">\(f_{0}=f_{1}=1\)</span>. Now, we will walk through an example of how we can derive a closed-form solution for <span class="math-inline" markdown="0">\(f_{n}\)</span> through generating functions.

Let the generating function be <span class="math-inline" markdown="0">\(F(x)=\sum_{n\ge 0}f_{n}x^n\)</span>. Now, we try to recover this function in the original defining recurrence relation. Multiply both sides by <span class="math-inline" markdown="0">\(x^{n+2}\)</span> and sum <span class="math-inline" markdown="0">\(\sum_{n\ge 0}\)</span> to get:

<div class="math-block" markdown="0">
\[
\sum_{n\ge 0} f_{n+2}x^{n+2}=\sum_{n\ge 0} f_{n+1}x^{n+2}+\sum_{n\ge 0}f_{n}x^{n+2}
\]
</div>

We can re-index and pull out extra factors of <span class="math-inline" markdown="0">\(x\)</span> to obtain

<div class="math-block" markdown="0">
\[
\sum_{n\ge 2}f_{n}x^n=x\sum_{n\ge 1}f_{n}x^n+x^2\sum_{n\ge 0}f_{n}x^n
\]
</div>

Now we can express these in terms of <span class="math-inline" markdown="0">\(F(x)=1+x+\sum_{n\ge 2}f_{n}x^n=1+\sum_{n\ge 1}f_{n}x^n\)</span>:

<div class="math-block" markdown="0">
\[
F(x)-x-1=x(F(x)-1)+x^2F(x)
\]
</div>

Solving for <span class="math-inline" markdown="0">\(F(x)\)</span>, we get

<div class="math-block" markdown="0">
\[
F(x)(1-x-x^2)=1
\]
</div>


<div class="math-block" markdown="0">
\[
\boxed{ F(x)=\frac{1}{1-x-x^2} }
\]
</div>


Here, we started with a convenient indexing so we wouldn't encounter some trouble with bookkeeping. Now we get to the same answer slightly differently to learn how to deal with these troubles, this time starting with this re-indexed recurrence relation

<div class="math-block" markdown="0">
\[
f_{n}=f_{n-1}+f_{n-2},\quad n\ge 2
\]
</div>

again with <span class="math-inline" markdown="0">\(f_{0}=f_{1}=1\)</span>. Notice that we now have to deal more carefully with the edge cases, because our recurrence relation only works for <span class="math-inline" markdown="0">\(n\ge 2\)</span>. So, we are only allowed to sum over the equation for <span class="math-inline" markdown="0">\(\sum_{n\ge 2}\)</span> after multiplying by <span class="math-inline" markdown="0">\(x^n\)</span>:

<div class="math-block" markdown="0">
\[
\sum_{n \ge 2}f_{n}x^n=\sum_{n\ge 2}f_{n-1}x^n+\sum_{n\ge 2}f_{n-2}x^n
\]
</div>

In the end, it turns out to be the same as our original problem. If you tried to sum over <span class="math-inline" markdown="0">\(n\ge 0\)</span> when the equation only holds for <span class="math-inline" markdown="0">\(n\ge 2\)</span>, you would get the wrong answer. So you need to keep track of the problem constraints.
</blockquote>

Now that we have found a formula for <span class="math-inline" markdown="0">\(F(x)\)</span>, how do we derive a formula for <span class="math-inline" markdown="0">\(f_{n}\)</span>? The trick is to remember our first identity for the geometric series:

<div class="math-block" markdown="0">
\[
\frac{1}{1-rx}=1+rx+r^2x^2+r^3x^3+\dots
\]
</div>

So we know the closed-form for the <span class="math-inline" markdown="0">\(n\)</span>-th term a geometric series. But using partial fraction decomposition, we can precisely turn <span class="math-inline" markdown="0">\(F(x)=\frac{1}{1-x-x^2}\)</span> into a linear combination of geometric series. Let's force

<div class="math-block" markdown="0">
\[
F(x)=\frac{1}{1-x-x^2}=\frac{1}{(1-\alpha x)(1-\beta x)}=\frac{a}{1-\alpha x}+\frac{b}{1-\beta x}
\]
</div>

where <span class="math-inline" markdown="0">\(1-x-x^2=(1-\alpha x)(1-\beta x)\)</span>. Comparing coefficients, <span class="math-inline" markdown="0">\(\alpha\beta=-1,-(\alpha+\beta)=-1\)</span>, so after substituting <span class="math-inline" markdown="0">\(\beta=-\frac{1}{\alpha}\)</span> in <span class="math-inline" markdown="0">\(\alpha+\beta=1\)</span>, we get 

<div class="math-block" markdown="0">
\[
\alpha-\frac{1}{\alpha}=1\iff \alpha^2-\alpha-1=0
\]
</div>

which is the famous equation for the golden ratio: <span class="math-inline" markdown="0">\(\alpha=\phi=\frac{1+\sqrt{5}}{2},\beta=\psi=\frac{1-\sqrt{5}}{2}\)</span>. To solve for the partial fraction coefficients, so after clearing denominators, we apply linear operator of evaluation at <span class="math-inline" markdown="0">\(x=\frac{1}{\alpha}\)</span> and <span class="math-inline" markdown="0">\(x=\frac{1}{\beta}\)</span>, giving <span class="math-inline" markdown="0">\(a=\frac{1}{1-\frac{\beta}{\alpha}}=\frac{\alpha}{\alpha-\beta}=\frac{\alpha}{\sqrt{5}}\)</span>, <span class="math-inline" markdown="0">\(b=\frac{1}{1-\frac{\alpha}{\beta}}=\frac{\beta}{\beta-\alpha}=-\frac{\beta}{\sqrt{5}}\)</span>. 

<blockquote class="box-notation" markdown="1">
<div class="title" markdown="1">
Coefficient <span class="math-inline" markdown="0">\(a_{n}\)</span> of <span class="math-inline" markdown="0">\(x^n\)</span> in <span class="math-inline" markdown="0">\(G(x)\)</span>
</div>
For convenience, the notation

<div class="math-block" markdown="0">
\[
[x^n]G(x)=a_{n}
\]
</div>

is often used to extract the <span class="math-inline" markdown="0">\(n\)</span>-th term in a sequence from the coefficient of <span class="math-inline" markdown="0">\(x^n\)</span> in <span class="math-inline" markdown="0">\(G(x)\)</span> (note that <span class="math-inline" markdown="0">\([x^n]\)</span> is a linear functional).
</blockquote>

Hence, our closed-form solution comes from

<div class="math-block" markdown="0">
\[
\begin{align}
f_{n}&=[x^n]F(x) \\
&=[x^n] \frac{1}{1-x-x^2} \\
&=[x^n] \frac{1}{\sqrt{5}}\left(\frac{\phi}{1-\phi x}-\frac{\psi}{1-\psi x}\right) \\
&=[x^n] \frac{1}{\sqrt{5}} \phi \sum_{n\ge 0}\phi^n x^n-\psi \sum_{n\ge 0}\psi^n x^n \\
&=[x^n] \frac{1}{\sqrt{5}} \sum_{n\ge 0}\left( \phi^{n+1}-\psi^{n+1} \right) \\
&=\frac{\phi^{n+1}-\psi^{n+1}}{\sqrt{5}} \\
&=\boxed{ \frac{1}{\sqrt{5}}\left( \left( \frac{1+\sqrt{5}}{2} \right)^{n+1} - \left( \frac{1-\sqrt{5}}{2} \right)^{n+1}  \right)  }
\end{align}
\]
</div>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Geometric series via recurrence relation
</div>
Using a similar approach on the recursive relation

<div class="math-block" markdown="0">
\[
g_{n+1}=rg_{n},\quad n\ge 0
\]
</div>

with <span class="math-inline" markdown="0">\(g_{0}=a\)</span>, prove that <span class="math-inline" markdown="0">\(G(x)=\frac{a}{1-r}\)</span>.
</blockquote>

Another reason why generating functions are so useful is that you can create a factor of <span class="math-inline" markdown="0">\(n\)</span> in a recursion by taking the derivative. Here is an example where the dummy variable <span class="math-inline" markdown="0">\(x\)</span> does mean something, and we care about convergence.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Random problem on Twitter
</div>
Taken from @abakcus on X: https://x.com/abakcus/status/1990261853801611758
Find 
<div class="math-block" markdown="0">
\[
\sum_{n=1}^{\infty}\frac{n^3}{2^n}=\; ?
\]
</div>

</blockquote>

We use generating functions for this problem. Note that <span class="math-inline" markdown="0">\(n^3=n^\underline{3}+3n^\underline{2}+n^\underline{1}\)</span>. Write


<div class="math-block" markdown="0">
\[
f(x)=\sum_{n\ge 1}n^3 x^n=\sum_{n\ge 1}(n^\underline{3}+3n^\underline{2}+n^\underline{1})x^n
\]
</div>

which converges for <span class="math-inline" markdown="0">\(\vert x\vert <1\)</span>. 

<blockquote class="box-principle" markdown="1">
<div class="title" markdown="1">
Derivative
</div>
Using our convenient falling power basis for <span class="math-inline" markdown="0">\(n\)</span>, we decompose the sum by noting that <span class="math-inline" markdown="0">\(D^k x^n:=\left( \frac{d}{dx} \right)^k x^n=n^\underline{k}x^{n-k}\)</span>.

<div class="math-block" markdown="0">
\[
D^k \sum_{n\ge 0}a_{n}x^n=\sum_{n\ge 0}a_{n}D^kx^n= \sum_{n\ge 0}n^{\underline{k}}a_{n}x^{n-k}
\]
</div>

</blockquote>


<div class="math-block" markdown="0">
\[
f(x)=x^3 \sum_{n\ge 1}n^\underline{3}x^{n-3}+3x^2 \sum_{n\ge 1}n^\underline{2}x^{n-2}+x\sum_{n\ge 1}n^\underline{1}x^{n-1}
\]
</div>



<div class="math-block" markdown="0">
\[
f(x)=x^3 \sum_{n\ge 1} D^3 x^n+3x^2 \sum_{n\ge 1} D^2x^n+x\sum_{n\ge 1}Dx^n
\]
</div>

Interchanging the derivative and summation:

<div class="math-block" markdown="0">
\[
f(x)=x^3 D^3 \sum_{n\ge 1}x^n+3x^2 D^2 \sum_{n\ge 1}x^n+xD \sum_{n\ge 1}x^n
\]
</div>

Using the geometric series formula <span class="math-inline" markdown="0">\(\sum_{n \ge 1}x^n=x\sum_{n\ge 0} x^n=\frac{x}{1-x}\)</span>, we now just need to apply <span class="math-inline" markdown="0">\(D,D^2,D^3\)</span> on it then add everything together.

<div class="math-block" markdown="0">
\[
D \frac{x}{1-x}=D\left( \frac{x-1+1}{1-x} \right)=D\left( -1+\frac{1}{1-x} \right)=(1-x)^{-2}
\]
</div>


<div class="math-block" markdown="0">
\[
D^2 \frac{x}{1-x}=D (1-x)^{-2}=2(1-x)^{-3}
\]
</div>


<div class="math-block" markdown="0">
\[
D^3 \frac{x}{1-x}=D \; 2(1-x)^{-3}=6(1-x)^{-4}
\]
</div>

<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
Feynman's trick: differentiation under the integral sign
</div>
In the continuous case, the act of creating a generating function and manipulating it with derivatives is called *differentiation under the integral sign*, *Feynman's integration trick*.
We first turn our function <span class="math-inline" markdown="0">\(f(x)\)</span> into a generating function with parameter <span class="math-inline" markdown="0">\(\alpha\)</span>, and then integrate:

<div class="math-block" markdown="0">
\[
I(\alpha)=\int f(x,\alpha)\, dx
\]
</div>

Under suitable conditions for convergence,

<div class="math-block" markdown="0">
\[
I'(\alpha)=\frac{ \partial }{ \partial \alpha } \int f(x,\alpha)\,dx=\int \frac{ \partial }{ \partial \alpha } f(x,\alpha)\,dx
\]
</div>

</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Spot the pattern!
</div>
Can you find a closed-form expression for <span class="math-inline" markdown="0">\(D^n \frac{x}{1-x}\)</span>?
</blockquote>

Now, we can simply plug everything in.

<div class="math-block" markdown="0">
\[
f(x):=\sum_{n\ge 1}n^3 x^n=x^3\cdot 6(1-x)^{-4}+3x^2 \cdot 2(1-x)^{-3}+x(1-x)^{-2}
\]
</div>

Evaluating at <span class="math-inline" markdown="0">\(x=\frac{1}{2}\)</span> for the original sum <span class="math-inline" markdown="0">\(\sum_{n=1}^{\infty}\frac{n^3}{2^n}\)</span>, note that <span class="math-inline" markdown="0">\(\left( 1-\frac{1}{2} \right)^{-k}=2^k\)</span>, so <span class="math-inline" markdown="0">\(\left( \frac{1}{2} \right)^k \cdot (1-\frac{1}{2})^{-k-1}=2^{-k}\cdot 2^{k+1}=2\)</span>.

<div class="math-block" markdown="0">
\[
f\left( \frac{1}{2} \right)=6\cdot 2+3\cdot 2\cdot 2+2=\boxed{ 26 }
\]
</div>


<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
<span class="math-inline" markdown="0">\((xD)^n\)</span>
</div>
The operator of differentiation then multiplication by a polynomial <span class="math-inline" markdown="0">\(E=p(x)\cdot D\)</span> is sometimes called the Cauchy-Euler operator, defined for some polynomial <span class="math-inline" markdown="0">\(p(x)\)</span> and the differentiation operator <span class="math-inline" markdown="0">\(D:=\frac{d}{dx}\)</span>. Applied to a function <span class="math-inline" markdown="0">\(f(x)\)</span>, it gives <span class="math-inline" markdown="0">\(Ef(x)=p(x)f'(x)\)</span>.

Notably, the simplest example <span class="math-inline" markdown="0">\(E=xD\)</span> has eigenfunctions <span class="math-inline" markdown="0">\(x^n\)</span>, since <span class="math-inline" markdown="0">\(Ex^n=xDx^n=nx^n\)</span>. This matches the form in our earlier problem! Based on our steps, find a closed-form expression for <span class="math-inline" markdown="0">\((xD)^n\)</span> as a linear combination of the terms <span class="math-inline" markdown="0">\(x^k D^k\)</span>.
</blockquote>

Now that we have the powerful tool of generating functions in our pockets, let's go back to the question of figuring out how to compute the Stirling numbers.

<div class="math-block" markdown="0">
\[
x^\underline{n}=\sum_{k=0}^{n}\binom{\underline{n}}{k}x^k
\]
</div>


<div class="math-block" markdown="0">
\[
x^n=\sum_{k=0}^{n}\binom{n}{\underline{k}}x^\underline{k}
\]
</div>

Let's start by trying a recursive approach. <span class="math-inline" markdown="0">\(x^\underline{n+1}=x^\underline{n} (x-n)\)</span>, so

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{n+1}\binom{\underline{n+1}}{k}x^k=(x-n)\sum_{j=0}^{n}\binom{\underline{n}}{j}x^j=\sum_{i=0}^{n}\binom{\underline{n}}{i}x^{i+1}-n\sum_{j=0}^{n}\binom{\underline{n}}{j}x^{j}
\]
</div>

Take <span class="math-inline" markdown="0">\([x^k]\)</span> on both sides for <span class="math-inline" markdown="0">\(1\le k \le n\)</span>:

<div class="math-block" markdown="0">
\[
\binom{\underline{n+1}}{k}=\binom{\underline{n}}{k-1}-n\binom{\underline{n}}{k}
\]
</div>

Great, now we construct the generating function. Note that <span class="math-inline" markdown="0">\(\binom{\underline{n}}{k}=0\)</span> for <span class="math-inline" markdown="0">\(n<k\)</span>, since the polynomials <span class="math-inline" markdown="0">\(x^\underline{n},x^{n}\)</span> have the same degrees, high powers can't contribute, so let the generating function

<div class="math-block" markdown="0">
\[
A_{k}(y)=\sum_{n\ge 0}\binom{\underline{n}}{k}y^n=\sum_{n\ge k} \binom{\underline{n}}{k}y^n
\]
</div>

Now we multiply our recurrence relation by <span class="math-inline" markdown="0">\(y^{n+1}\)</span> and sum over <span class="math-inline" markdown="0">\(n\ge k\)</span>:

<div class="math-block" markdown="0">
\[
\sum_{n\ge k}\binom{\underline{n+1}}{k}y^{n+1}=y\sum_{n\ge k}\binom{\underline{n}}{k-1}y^{n}-ny\sum_{n\ge k}\binom{\underline{n}}{k}y^{n}
\]
</div>


Writing in terms of <span class="math-inline" markdown="0">\(A_{k}(y)\)</span>:

<div class="math-block" markdown="0">
\[
LHS=\sum_{n\ge k+1}\binom{\underline{n}}{k}y^n=A_{k}(y)-\underbrace{ \binom{\underline{k}}{k} }_{ =1 }y^{k}=A_{k}(y)-y^k
\]
</div>

where we know polynomials <span class="math-inline" markdown="0">\(y^\underline{k},y^k\)</span> have the same leading coefficient <span class="math-inline" markdown="0">\(1\)</span>, so <span class="math-inline" markdown="0">\(\binom{\underline{k}}{k}=1\)</span>.
Let the terms in the RHS be <span class="math-inline" markdown="0">\(RHS_{1}\)</span> and <span class="math-inline" markdown="0">\(RHS_{2}\)</span> in order:

<div class="math-block" markdown="0">
\[
RHS_{1}=y\left(A_{k-1}(y)-\underbrace{ \binom{\underline{k-1}}{k-1} }_{ =1 }y^{k-1}\right)=yA_{k-1}(y)-y^k
\]
</div>

The second term has a factor of <span class="math-inline" markdown="0">\(n\)</span>, so we need a derivative.
Note that 

<div class="math-block" markdown="0">
\[
A'_{k}(y)=\sum_{n\ge k}\binom{\underline{n}}{k}ny^{n-1}=\frac{n}{y}\sum_{n\ge k}\binom{\underline{n}}{k}y^n
\]
</div>



<div class="math-block" markdown="0">
\[
RHS_{2}=-y^2 A'_{k}(y)
\]
</div>

Altogether,

<div class="math-block" markdown="0">
\[
A_{k}(y)\cancel{ -y^k }=yA_{k-1}(y)\cancel{ -y^k }-y^2 A'_{k}(y)
\]
</div>


<div class="math-block" markdown="0">
\[
A_{k}(y)+y^2 A'_{k}(y)=yA_{k-1}
\]
</div>

This is a first-order ODE, and you can solve it using integrating factors, but it results in some very ugly algebra and a lot of integration. It turns out, there's a better approach to this problem.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Exponential Generating Function (EGF)
</div>
So far, we have only been working with something called the *ordinary generating function (OGF)*. For a sequence <span class="math-inline" markdown="0">\(a_{0},a_{1},a_{2},\dots\)</span>

<div class="math-block" markdown="0">
\[
OGF(x)=\sum_{n\ge 0}a_{n}x^{n}
\]
</div>

But for problems involving falling powers (which includes factorials, binomial coefficients, e.g. labeled set structures and permutations), the *exponential generating function (EGF)* often simplifies better:

<div class="math-block" markdown="0">
\[
EGF(x)=\sum_{n\ge 0}\frac{a_{n}}{n!}x^n
\]
</div>

If <span class="math-inline" markdown="0">\(a_{n}=1\)</span> for all <span class="math-inline" markdown="0">\(n\)</span>, then we recover the Taylor series of <span class="math-inline" markdown="0">\(e^x\)</span>:

<div class="math-block" markdown="0">
\[
\sum_{n\ge 0}\frac{x^n}{n!}=e^x
\]
</div>

as opposed to the geometric series <span class="math-inline" markdown="0">\(\sum_{n\ge 0}x^n=\frac{1}{1-x}\)</span> for the OGFs.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Derivative of EGF
</div>

<div class="math-block" markdown="0">
\[
A(x)=\sum_{n\ge 0}\frac{a_{n}}{n!}x^n=a_{0}+a_{1}x+\frac{a_{2}}{2!}x^2+\frac{a_{3}}{3!}x^3+\cdots
\]
</div>


<div class="math-block" markdown="0">
\[
A'(x)=a_{1}+a_{2}x+\frac{a_{3}}{2!}x^2+\cdots=\sum_{n\ge 1}\frac{a_{n}}{(n-1)!}x^{n-1}=\sum_{n\ge 0}\frac{a_{n+1}}{n!}x^n
\]
</div>

So the factor of <span class="math-inline" markdown="0">\(n\)</span> cancels, and the index just shifts by <span class="math-inline" markdown="0">\(1\)</span>. This simplifies algebra for ODEs with recursions like <span class="math-inline" markdown="0">\(a_{n}=na_{n-1}\)</span>.
</blockquote>

Let's try it out on our problem of computing Stirling coefficients.

 First, we need the binomial theorem.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Binomial theorem
</div>
Define the binomial coefficient

<div class="math-block" markdown="0">
\[
\binom{n}{k}=\frac{n^\underline{k}}{k^\underline{k}}=\frac{n^\underline{k}}{k!}
\]
</div>

Using the generating function

<div class="math-block" markdown="0">
\[
G(x,y)=\sum_{n\ge 0}\sum_{k\ge 0}\binom{n}{k}x^n y^k
\]
</div>

Prove by taking <span class="math-inline" markdown="0">\([x^n]\)</span> that

<div class="math-block" markdown="0">
\[
(y+1)^n=\sum_{k=0}^{n}\binom{n}{k}y^k
\]
</div>

Hint: note that <span class="math-inline" markdown="0">\(\binom{n}{0}=1,n\ge 0\)</span>, <span class="math-inline" markdown="0">\(\binom{0}{k}=0, k\ge 1\)</span>, 

<div class="math-block" markdown="0">
\[
\binom{n+1}{k+1}=\binom{n}{k+1}+\binom{n}{k},\quad n,k\ge 0
\]
</div>

Then, replacing <span class="math-inline" markdown="0">\(y=\frac{a}{b}\)</span>, prove the full binomial theorem:

<div class="math-block" markdown="0">
\[
(a+b)^n=\sum_{k=0}^{n}\binom{n}{k}a^k b^{n-k}
\]
</div>

If you more practice, you can also take <span class="math-inline" markdown="0">\([y^k]\)</span> to prove the Stars and Bars identity:

<div class="math-block" markdown="0">
\[
[x^n]\left( \frac{1}{1-x} \right)^{k}=\binom{n+k-1}{k-1}
\]
</div>

which by remembering <span class="math-inline" markdown="0">\(\frac{1}{1-x}=1+x+x^2+\cdots\)</span> is the number of nonnegative integer solutions to

<div class="math-block" markdown="0">
\[
x_{1}+x_{2}+\dots+x_{k}=n
\]
</div>

</blockquote>

Now, to recover an exponential generating function, we apply the clever identity 
<div class="math-block" markdown="0">
\[
a^b=e^{\ln(a^b)}=e^{b \ln a}
\]
</div>
 with <span class="math-inline" markdown="0">\(a=(x+1),b=n\)</span>. Define

<div class="math-block" markdown="0">
\[
F(x,n)=(x+1)^n=\sum_{k=0}^n \binom{n}{k}x^k=\sum_{k=0}^n \frac{n^\underline{k}}{k!}x^k
\]
</div>

From our definition <span class="math-inline" markdown="0">\(n^\underline{k}=\sum_{j=0}^{k}\binom{\underline{k}}{j}n^j\)</span>,

<div class="math-block" markdown="0">
\[
F(x,n)=\sum_{k=0}^{n}\frac{x^k}{k!}\sum_{j=0}^{k} \binom{\underline{k}}{j}n^j
\]
</div>

then we have

<div class="math-block" markdown="0">
\[
F(x,n)=e^{n\ln(x+1)}=\sum_{j\ge 0}\frac{(n \ln(x+1))^j}{j!}=\sum_{j\ge 0} \frac{(\ln(x+1))^j}{j!} n^j
\]
</div>

Comparing coefficients with <span class="math-inline" markdown="0">\([n^j]\)</span>, we get that

<div class="math-block" markdown="0">
\[
[n^j]F(x,n)=\sum_{k=0}^{n} \binom{\underline{k}}{j}\frac{x^k}{k!}=\frac{(\ln(x+1))^j}{j!}
\]
</div>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Stirling numbers of the second kind <span class="math-inline" markdown="0">\(\binom{n}{\underline{k}}\)</span>
</div>
Find a similar relationship for <span class="math-inline" markdown="0">\(\binom{n}{\underline{k}}\)</span> using the following trick:

<div class="math-block" markdown="0">
\[
e^{zt}=(1+(e^t-1))^z
\]
</div>

You should arrive at

<div class="math-block" markdown="0">
\[
\sum_{n\ge k} \binom{n}{\underline{k}}\frac{t^n}{n!}=\frac{(e^{t}-1)^k}{k!}
\]
</div>

</blockquote>

For solving discrete problems (sums), we usually convert from the power basis <span class="math-inline" markdown="0">\(x^k\)</span> to the falling power basis <span class="math-inline" markdown="0">\(x^\underline{k}\)</span> via the Stirling numbers of the second kind, so we'll derive an expression for them. We expand the RHS using binomial theorem and Taylor series of <span class="math-inline" markdown="0">\(e^x\)</span>.


<div class="math-block" markdown="0">
\[
\begin{align}
\sum_{n\ge k}\binom{n}{\underline{k}}\frac{t^n}{n!}
&=\frac{1}{k!} \sum_{j=0}^{k} \binom{k}{j}e^{jt}(-1)^{k-j} \\
&=\frac{1}{k!} \sum_{j=0}^{k} (-1)^{k-j}\binom{k}{j} \sum_{n \ge 0}\frac{(jt)^n}{n!}
\end{align}
\]
</div>

Comparing coefficients <span class="math-inline" markdown="0">\(\left[ \frac{t^n}{n!} \right]\)</span> of the EGF:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Closed-form solution for Stirling numbers of the second kind <span class="math-inline" markdown="0">\(\binom{n}{\underline{k}}\)</span>
</div>

<div class="math-block" markdown="0">
\[
\boxed{ \binom{n}{\underline{k}}=\frac{1}{k!}\sum_{j=0}^k (-1)^{k-j}\binom{k}{j}j^n }
\]
</div>

</blockquote>

So we can convert from <span class="math-inline" markdown="0">\(x^k\)</span> to <span class="math-inline" markdown="0">\(x^\underline{k}\)</span> basis fairly OK, but as for the Stirling numbers of the first kind <span class="math-inline" markdown="0">\(\binom{\underline{n}}{k}\)</span> for the inverse matrix, from [Wikipedia](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind#Explicit_formula):
>No one-sum formula for Stirling numbers of the first kind is currently known. A two-sum formula can be obtained using one of the [symmetric formulae for Stirling numbers](https://en.wikipedia.org/wiki/Stirling_number#Symmetric_formulae "Stirling number") in conjunction with the explicit formula for [Stirling numbers of the second kind](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind "Stirling numbers of the second kind").

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Taylor series of <span class="math-inline" markdown="0">\(\ln(1+x)\)</span>
</div>
Find the Taylor series of <span class="math-inline" markdown="0">\(\ln(1+x)\)</span>, given

<div class="math-block" markdown="0">
\[
\ln(1+x)=\int \frac{1}{1+x} \,dx
\]
</div>

and <span class="math-inline" markdown="0">\(\frac{1}{1+x}=\frac{1}{1-(-x)}=\sum_{n \ge 0}(-x)^n\)</span>.
</blockquote>

### Natural Discrete Base 2
In infinitesimal calculus, we define the *natural base* <span class="math-inline" markdown="0">\(e\)</span> by

<div class="math-block" markdown="0">
\[
\frac{d}{dx}e^x=e^x
\]
</div>

What is the natural basis <span class="math-inline" markdown="0">\(a\)</span> in the continuous world? It's a lot easier to solve:

<div class="math-block" markdown="0">
\[
\Delta a^n=a^n
\]
</div>


<div class="math-block" markdown="0">
\[
a^{n+1}-a^n=a^n\iff a^{n+1}=2a^n
\]
</div>

For the non-trivial solution <span class="math-inline" markdown="0">\(a\ne 0\)</span>, we divide both sides by <span class="math-inline" markdown="0">\(a^n\)</span> to find <span class="math-inline" markdown="0">\(a=2\)</span>. So actually, <span class="math-inline" markdown="0">\(2\)</span> is the natural basis for the discrete world. We expand on this in the next section.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
<span class="math-inline" markdown="0">\(\sum_{n=0}^{k} n 2^n\)</span>
</div>
You could use generating functions, of course. Here, just like in <span class="math-inline" markdown="0">\(\int_{a}^{b} xe^x \,dx\)</span>, we apply summation by parts.

<div class="math-block" markdown="0">
\[
\sum_{x=a}^{b-1} f(x) \Delta g(x)=\left.f(x)g(x)\right\vert_{a}^{b}-\sum_{x=a}^{b-1} \Delta f(x)g(x+1)
\]
</div>

Take <span class="math-inline" markdown="0">\(f(n)=n=n^\underline{1},g(n)=2^n\)</span>, so <span class="math-inline" markdown="0">\(\Delta f(n)=1, \Delta g(n)=2^n\)</span>.

<div class="math-block" markdown="0">
\[
\begin{align}
\sum_{n=0}^{k} n 2^n&=\left.\left(n\cdot 2^n\right)\right\vert_{0}^{k+1}-\sum_{n=0}^{k}1 \cdot 2^{n+1} \\
&=(k+1)2^{k+1}-(2^{k+2}-2) \\
&=\boxed{ (k-1)2^{k+1}+2 }
\end{align}
\]
</div>

</blockquote>

### Newton series
Given the defining equation

<div class="math-block" markdown="0">
\[
e^x=\frac{d}{dx}e^x
\]
</div>

If we assume that <span class="math-inline" markdown="0">\(e^x\)</span> is analytic, i.e. we can write it as a power series, then we are generating a sequence defined by <span class="math-inline" markdown="0">\(Dx^n=nx^{n-1},n\ne 0\)</span>:

<div class="math-block" markdown="0">
\[
\begin{align}
a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots&=\frac{d}{dx}(a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots) \\
&=a_{1}+2a_{2}x+3a_{3}x^2+\dots
\end{align}
\]
</div>

Comparing <span class="math-inline" markdown="0">\(x^n\)</span>, we get the recurrence <span class="math-inline" markdown="0">\(a_{n+1}=na_{n}\)</span>, <span class="math-inline" markdown="0">\(a_{0}=e^{0}=1\)</span>, or <span class="math-inline" markdown="0">\(a_{n}=n!\)</span> giving the famous Taylor series

<div class="math-block" markdown="0">
\[
e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
\]
</div>

The Taylor series is an expansion in the basis <span class="math-inline" markdown="0">\(x^k\)</span> and <span class="math-inline" markdown="0">\(D=\frac{d}{dx}\)</span>: 

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Taylor series
</div>

<div class="math-block" markdown="0">
\[
f(x)=\sum_{n=0}^\infty \frac{(D^nf)(a)}{n!}(x-a)^n
\]
</div>

</blockquote>

But lesser know is the notion of Newton series, which is simply a series in the basis of <span class="math-inline" markdown="0">\(x^\underline{k}\)</span> and <span class="math-inline" markdown="0">\(\Delta\)</span>. It is also called the Gregory-Newton forward difference interpolation formula because it interpolates at integer points based on <span class="math-inline" markdown="0">\(\Delta\)</span>.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Newton series
</div>

<div class="math-block" markdown="0">
\[
f(x)=\sum_{n=0}^{\infty} \frac{(\Delta^n f)(a)}{n!}(x-a)^\underline{n}=\sum_{n=0}^{\infty} \binom{x-a}{k}(\Delta^n f)(a)
\]
</div>

</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Newton series of <span class="math-inline" markdown="0">\(2^x\)</span>
</div>
Based on the change of basis through the Stirling numbers, the Newton series of <span class="math-inline" markdown="0">\(2^x\)</span> in <span class="math-inline" markdown="0">\(x^\underline{k}\)</span> looks exactly like <span class="math-inline" markdown="0">\(e^x\)</span> in <span class="math-inline" markdown="0">\(x^k\)</span> since just as <span class="math-inline" markdown="0">\(D^n e^x=e^x\)</span> with <span class="math-inline" markdown="0">\(e^0=1\)</span>, <span class="math-inline" markdown="0">\(\Delta^n 2^x=2^x\)</span>, and <span class="math-inline" markdown="0">\(2^0=1\)</span>, so <span class="math-inline" markdown="0">\(2^x\)</span> encodes the sequence <span class="math-inline" markdown="0">\(1,1,1,1,\dots\)</span> in an EGF in <span class="math-inline" markdown="0">\(x^\underline{k}\)</span>:

<div class="math-block" markdown="0">
\[
2^x=\sum_{n=0}^\infty \frac{1}{n!}x^\underline{n}=1+x^\underline{1}+\frac{x^\underline{2}}{2!}+\frac{x^\underline{3}}{3!}+\dots
\]
</div>

</blockquote>

We can also derive it in a funny way using operator calculus. Let <span class="math-inline" markdown="0">\(E=e^D\)</span> be the shift operator <span class="math-inline" markdown="0">\(Ef(x)=f(x+1)\)</span>, so that <span class="math-inline" markdown="0">\(E=I+\Delta\)</span> for the identity operator <span class="math-inline" markdown="0">\(If(x)=f(x)\)</span>. Then Newton series for integers comes from the (extended) binomial theorem:

<div class="math-block" markdown="0">
\[
\begin{align}
f(x)&=E^x f(0) \\
&=(I+\Delta)^x f(0) \\
&=\sum_{k\ge 0} \binom{x}{k}\Delta^k f(0)
\end{align}
\]
</div>

For an example application, you can use this formula to solve for sequences that you know have a polynomial solution of degree <span class="math-inline" markdown="0">\(n\)</span> with <span class="math-inline" markdown="0">\(n+1\)</span> points.

Here is a quick worked example and an exercise based on the logic that <span class="math-inline" markdown="0">\(f(x) = \sum_{k=0}^\infty \frac{(\Delta^k f)(0)}{k!} x^{\underline{k}}\)</span>.

While <span class="math-inline" markdown="0">\(2^x\)</span> is an infinite series (because the differences never go to zero), this method is most powerful for finding **polynomial closed forms** for integer sequences, where the differences eventually become zero.

### Example Application: Find the closed form

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Problem:** Find the lowest degree polynomial <span class="math-inline" markdown="0">\(f(x)\)</span> that generates the sequence: <span class="math-inline" markdown="0">\(1, 6, 15, 28, \dots\)</span> for <span class="math-inline" markdown="0">\(x=0, 1, 2, 3 \dots\)</span>
</div>
**Step 1: Build the Difference Table**
Just as a Taylor series requires derivatives at <span class="math-inline" markdown="0">\(x=0\)</span>, the Newton series requires the forward differences at <span class="math-inline" markdown="0">\(x=0\)</span> (the first number in each row).


<div class="math-block" markdown="0">
\[
\begin{array}{r|cccccl}
x & f(x) & \Delta & \Delta^2 & \Delta^3 & \dots \\
\hline
0 & \mathbf{1} & & & & \\
& & \mathbf{5} & & & \\
1 & 6 & & \mathbf{4} & & \\
& & 9 & & \mathbf{0} & \leftarrow \text{Stops here} \\
2 & 15 & & 4 & & \\
& & 13 & & & \\
3 & 28 & & & & \\
\end{array}
\]
</div>


**Step 2: Apply the Formula**
We take the "diagonal" values <span class="math-inline" markdown="0">\((\mathbf{1}, \mathbf{5}, \mathbf{4}, \mathbf{0})\)</span> as our coefficients for <span class="math-inline" markdown="0">\(\frac{x^\underline{k}}{k!}\)</span>.


<div class="math-block" markdown="0">
\[
f(x) = 1 \cdot \frac{x^\underline{0}}{0!} + 5 \cdot \frac{x^\underline{1}}{1!} + 4 \cdot \frac{x^\underline{2}}{2!}
\]
</div>


**Step 3: Convert to Standard Polynomials**
Recall that <span class="math-inline" markdown="0">\(x^\underline{1} = x\)</span> and <span class="math-inline" markdown="0">\(x^\underline{2} = x(x-1)\)</span>.


<div class="math-block" markdown="0">
\[
\begin{aligned}
f(x) &= 1(1) + 5(x) + \frac{4}{2} x(x-1) \\
&= 1 + 5x + 2(x^2 - x) \\
&= 1 + 5x + 2x^2 - 2x \\
&= \mathbf{2x^2 + 3x + 1}
\end{aligned}
\]
</div>


*(Check: If <span class="math-inline" markdown="0">\(x=3\)</span>, <span class="math-inline" markdown="0">\(2(9) + 9 + 1 = 28\)</span>. It works.)*
</blockquote>

### Example: Formula for the Sum of Cubes

This is a classic application of the method. Because the sum of polynomials of degree <span class="math-inline" markdown="0">\(d\)</span> results in a polynomial of degree <span class="math-inline" markdown="0">\(d+1\)</span>, we know ahead of time that the sum of cubes (degree 3) will result in a degree 4 polynomial. This guarantees the difference table will eventually hit zero.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Problem:** Find the formula for <span class="math-inline" markdown="0">\(S(x) = \sum_{i=1}^x i^3 = 1^3 + 2^3 + \dots + x^3\)</span>.
</div>
**Step 1: Generate the Sequence and Difference Table**
We must include <span class="math-inline" markdown="0">\(x=0\)</span> (the empty sum, which is 0) to anchor the Newton series properly.

*   <span class="math-inline" markdown="0">\(x=0, S=0\)</span>
*   <span class="math-inline" markdown="0">\(x=1, S=1\)</span>
*   <span class="math-inline" markdown="0">\(x=2, S=1+8=9\)</span>
*   <span class="math-inline" markdown="0">\(x=3, S=9+27=36\)</span>
*   <span class="math-inline" markdown="0">\(x=4, S=36+64=100\)</span>


<div class="math-block" markdown="0">
\[
\begin{array}{r|cccccl}
x & S(x) & \Delta & \Delta^2 & \Delta^3 & \Delta^4 & \dots \\
\hline
0 & \mathbf{0} & & & & & \\
& & \mathbf{1} & & & & \\
1 & 1 & & \mathbf{7} & & & \leftarrow (8-1) \\
& & 8 & & \mathbf{12} & & \leftarrow (19-7) \\
2 & 9 & & 19 & & \mathbf{6} & \leftarrow (18-12) \\
& & 27 & & 18 & & \\
3 & 36 & & 37 & & 0 & \leftarrow \text{Stops!} \\
& & 64 & & & & \\
4 & 100 & & & & &
\end{array}
\]
</div>


**Step 2: Apply the Newton Series Formula**
The diagonal values are <span class="math-inline" markdown="0">\(\mathbf{0, 1, 7, 12, 6}\)</span>.

<div class="math-block" markdown="0">
\[
S(x) = 0 \frac{x^\underline{0}}{0!} + 1 \frac{x^\underline{1}}{1!} + 7 \frac{x^\underline{2}}{2!} + 12 \frac{x^\underline{3}}{3!} + 6 \frac{x^\underline{4}}{4!}
\]
</div>


**Step 3: Simplify**
Evaluate the factorials: <span class="math-inline" markdown="0">\(1! = 1, 2!=2, 3!=6, 4!=24\)</span>.


<div class="math-block" markdown="0">
\[
S(x) = x + \frac{7}{2}x(x-1) + \frac{12}{6}x(x-1)(x-2) + \frac{6}{24}x(x-1)(x-2)(x-3)
\]
</div>


This is a valid formula, but to show it matches the standard textbook formula, we simplify algebraically.

1.  **Linear/Quadratic terms:** <span class="math-inline" markdown="0">\(x + 3.5(x^2-x) = 3.5x^2 - 2.5x\)</span>
2.  **Cubic term:** <span class="math-inline" markdown="0">\(2(x^3 - 3x^2 + 2x) = 2x^3 - 6x^2 + 4x\)</span>
3.  **Quartic term:** <span class="math-inline" markdown="0">\(\frac{1}{4}(x^4 - 6x^3 + 11x^2 - 6x)\)</span>

Summing these up:
*   <span class="math-inline" markdown="0">\(x^4\)</span>: <span class="math-inline" markdown="0">\(\frac{1}{4}x^4\)</span>
*   <span class="math-inline" markdown="0">\(x^3\)</span>: <span class="math-inline" markdown="0">\(-\frac{6}{4}x^3 + 2x^3 = \frac{1}{2}x^3\)</span>
*   <span class="math-inline" markdown="0">\(x^2\)</span>: <span class="math-inline" markdown="0">\(\frac{11}{4}x^2 - 6x^2 + 3.5x^2 = \frac{1}{4}x^2\)</span>
*   <span class="math-inline" markdown="0">\(x^1\)</span>: <span class="math-inline" markdown="0">\(-\frac{6}{4}x + 4x - 2.5x = 0\)</span>


<div class="math-block" markdown="0">
\[
S(x) = \frac{1}{4}x^4 + \frac{1}{2}x^3 + \frac{1}{4}x^2 = \frac{x^2}{4}(x^2 + 2x + 1)
\]
</div>



<div class="math-block" markdown="0">
\[
\mathbf{S(x) = \left[ \frac{x(x+1)}{2} \right]^2}
\]
</div>


This confirms the famous identity that <span class="math-inline" markdown="0">\(\sum i^3 = (\sum i)^2\)</span>.
</blockquote>

<blockquote class="box-exercise" markdown="1">

**Problem:**
Using the Newton Series method, find the polynomial <span class="math-inline" markdown="0">\(f(x)\)</span> for the following sequence (where <span class="math-inline" markdown="0">\(x=0, 1, 2\dots\)</span>):

<div class="math-block" markdown="0">
\[
3, 5, 9, 15, 23, \dots
\]
</div>


1. Construct the difference table to find <span class="math-inline" markdown="0">\(\Delta^n f(0)\)</span>.
2. Write out the Newton series using falling factorials (<span class="math-inline" markdown="0">\(x^{\underline{k}}\)</span>).
3. Simplify it into a standard polynomial form (<span class="math-inline" markdown="0">\(ax^2+bx+c\)</span>).
</blockquote>

<details class="box-tip" markdown="1">
<summary markdown="1">
**Click for Solution**
</summary>
**1. The Difference Table:**
*   Sequence (<span class="math-inline" markdown="0">\(f\)</span>): <span class="math-inline" markdown="0">\(3, 5, 9, 15, 23\)</span>
*   <span class="math-inline" markdown="0">\(\Delta^1\)</span>: <span class="math-inline" markdown="0">\(2, 4, 6, 8\)</span>
*   <span class="math-inline" markdown="0">\(\Delta^2\)</span>: <span class="math-inline" markdown="0">\(2, 2, 2\)</span>  (Constant!)
*   <span class="math-inline" markdown="0">\(\Delta^3\)</span>: <span class="math-inline" markdown="0">\(0\)</span>

The coefficients at <span class="math-inline" markdown="0">\(x=0\)</span> are **3, 2, 2**.

**2. The Newton Series:**

<div class="math-block" markdown="0">
\[
f(x) = 3 \frac{x^\underline{0}}{0!} + 2 \frac{x^\underline{1}}{1!} + 2 \frac{x^\underline{2}}{2!}
\]
</div>


**3. Simplification:**

<div class="math-block" markdown="0">
\[
\begin{aligned}
f(x) &= 3(1) + 2(x) + \frac{2}{2}x(x-1) \\
&= 3 + 2x + x^2 - x \\
&= \mathbf{x^2 + x + 3}
\end{aligned}
\]
</div>

</details>

### Umbral Calculus

#### Warmup

A *convolution* <span class="math-inline" markdown="0">\(f \ast g\)</span> is a process where "multiplying" objects adds their "degrees". A good example is polynomials.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Polynomial multiplication
</div>
If we multiply two quadratics <span class="math-inline" markdown="0">\(A(x)\cdot B(x)=C(x)\)</span>

<div class="math-block" markdown="0">
\[
\begin{gather}
(a_{0}+a_{1}x+a_{2}x^2)(b_{0}+b_{1}x+b_{2}x^2) \\
=a_{0}b_{0}+(a_{0}b_{1}+b_{0}a_{1})x+(a_{0}b_{2}+a_{1}b_{1}+a_{2}b_{0})x^2+(a_{1}b_{2}+a_{2}b_{1})x^3+a_{2}b_{2}x^4
\end{gather}
\]
</div>


As you can see, the subscripts add together to the degree encoded by the monomial <span class="math-inline" markdown="0">\(x^d\)</span>, e.g. <span class="math-inline" markdown="0">\(a_{0}b_{1}\)</span> gets assigned to <span class="math-inline" markdown="0">\(x^{0+1}=x\)</span>. So <span class="math-inline" markdown="0">\(c_{n}=\sum_{k}a_{k}b_{n-k}\)</span> are the coefficients of <span class="math-inline" markdown="0">\(C(x)\)</span>.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Dice rolls
</div>
Another example is the addition of random variables: let <span class="math-inline" markdown="0">\(X,Y\)</span> be the outcomes of a standard 6-sided die. Then <span class="math-inline" markdown="0">\(X+Y\)</span> is given by the convolution of the <span class="math-inline" markdown="0">\(X\)</span> and <span class="math-inline" markdown="0">\(Y\)</span>:

<div class="math-block" markdown="0">
\[
\begin{array}{c|cccccc}
{}_{X} \setminus ^{Y} & \mathbf{1} & \mathbf{2} & \mathbf{3} & \mathbf{4} & \mathbf{5} & \mathbf{6} \\
\hline
\mathbf{1} & 2 & 3 & 4 & 5 & 6 & 7 \\
\mathbf{2} & 3 & 4 & 5 & 6 & 7 & 8 \\
\mathbf{3} & 4 & 5 & 6 & 7 & 8 & 9 \\
\mathbf{4} & 5 & 6 & 7 & 8 & 9 & 10 \\
\mathbf{5} & 6 & 7 & 8 & 9 & 10 & 11 \\
\mathbf{6} & 7 & 8 & 9 & 10 & 11 & 12 \\
\end{array}
\]
</div>

The number outcomes for <span class="math-inline" markdown="0">\(X+Y\)</span> amounts to "counting along the off-diagonal". For example, finding the number of ways for <span class="math-inline" markdown="0">\(X+Y=6\)</span>, you get <span class="math-inline" markdown="0">\(5\)</span>:

<div class="math-block" markdown="0">
\[
\begin{array}{c|cccccc}
{}_{X} \setminus ^{Y} & \mathbf{1} & \mathbf{2} & \mathbf{3} & \mathbf{4} & \mathbf{5} & \mathbf{6} \\
\hline
\mathbf{1} & 2 & 3 & 4 & 5 & \textcolor{red}{6} & 7 \\
\mathbf{2} & 3 & 4 & 5 & \textcolor{red}{6} & 7 & 8 \\
\mathbf{3} & 4 & 5 & \textcolor{red}{6} & 7 & 8 & 9 \\
\mathbf{4} & 5 & \textcolor{red}{6} & 7 & 8 & 9 & 10 \\
\mathbf{5} & \textcolor{red}{6} & 7 & 8 & 9 & 10 & 11 \\
\mathbf{6} & 7 & 8 & 9 & 10 & 11 & 12 \\
\end{array}
\]
</div>

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Discrete convolution
</div>
Define OGFs for sequences <span class="math-inline" markdown="0">\((a_{i})_{i},(b_{j})_{j}\)</span>.

<div class="math-block" markdown="0">
\[
A(x)=\sum_{i=i_{0}}^{i_{1}}a_{i}x^i=a_{i_{0}}x^{i_{0}}+a_{i_{0}+1}x^{i_{0}+1}+\dots+a_{i_{1}}x^{i_{1}}
\]
</div>


<div class="math-block" markdown="0">
\[
B(x)=\sum_{j=j_{0}}^{j_{1}}b_{j}x^j=b_{j_{0}}x^{j_{0}}+b_{j_{0}+1}x^{j_{0}+1}+\dots+b_{j_{1}}x^{j_{1}}
\]
</div>

Define

<div class="math-block" markdown="0">
\[
C(x)=A(x)B(x)
\]
</div>

Then we have

<div class="math-block" markdown="0">
\[
\begin{align}
C(x)&=\left( \sum_{i=i_{0}}^{i_{1}}a_{i}x^i \right)\left( \sum_{j=j_{0}}^{j_{1}} b_{j}x^j \right) \\
&=(a_{i_{0}}x^{i_{0}}+\dots+a_{i_{1}}x^{i_{1}})(b_{j_{0}}x^{j_{0}}+\dots+b_{j_{1}}x^{j_{1}}) \\
&=a_{i_{0}}b_{j_{0}}x^{i_{0}+j_{0}}+(a_{i_{0}}b_{j_{0}+1}+a_{i_{0}+1}b_{j_{0}})x^{i_{0}+j_{0}+1}+\dots+a_{i_{1}}b_{j_{1}}x^{i_{1}+j_{1}} \\
&=\sum_{n=i_{0}+j_{0}}^{i_{1}+j_{1}}x^n \left( \sum_{k=\max(i_{0},j_{0})}^{\min(i_{1},j_{1})} a_{k}b_{n-k}\right) 
\end{align}
\]
</div>

We define the *discrete convolution* of <span class="math-inline" markdown="0">\((a_{i})_{i},(b_{j})_{j}\)</span>  as a sequence <span class="math-inline" markdown="0">\((c_{n})_{n}=(a_{i})_{i} \ast (b_{j})_{j}\)</span>, with

<div class="math-block" markdown="0">
\[
c_{n}=[x^n]C(x)=\sum_{k=\max(i_{0},j_{0})}^{\min(i_{1},j_{1})} a_{k}b_{n-k}
\]
</div>

for <span class="math-inline" markdown="0">\(i_{0}+j_{0}\le n\le i_{1}+j_{1}\)</span>.
</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Vandermonde's convolution identity
</div>
Comparing coefficients of <span class="math-inline" markdown="0">\((x+1)^{m+n}=(x+1)^m (x+1)^n\)</span>, prove

<div class="math-block" markdown="0">
\[
\binom{m+n}{m}=\sum_{j=0}^{k}\binom{n}{j}\binom{m}{k-j}
\]
</div>

</blockquote>

<blockquote class="box-solution" markdown="1">
<div class="title" markdown="1">
Vandermonde's convolution identity
</div>
Here is an alternative solution using operator calculus again. First, we prove that the binomial theorem holds with falling powers:

<div class="math-block" markdown="0">
\[
(x+y)^\underline{n}=\sum_{k=0}^{n}x^\underline{k} y^{\underline{n-k}}
\]
</div>

Expand into the Newton series by <span class="math-inline" markdown="0">\((x+y)^\underline{n}=E^yx^\underline{n}\)</span>:

<div class="math-block" markdown="0">
\[
(x+y)^\underline{n}=(\Delta+I)^y x^\underline{n}=\sum_{k\ge 0}\binom{y}{k}\Delta^k x^\underline{n}
\]
</div>

Compute <span class="math-inline" markdown="0">\(\Delta^k x^\underline{n}=n^\underline{k}x^\underline{n-k}\)</span>, so

<div class="math-block" markdown="0">
\[
(x+y)^\underline{n}=\sum_{k\ge 0} \frac{y^\underline{k}}{k!} n^\underline{k} x^\underline{n-k}=\sum_{k=0}^{n}\binom{n}{k}y^\underline{k}x^\underline{n-k}
\]
</div>

Now, dividing both sides by <span class="math-inline" markdown="0">\(n! =n^\underline{k}(n-k)!\)</span> proves the identity.
</blockquote>

It's very interesting that the binomial theorem also holds in the discrete world with falling powers instead of regular powers. But this is no coincidence, it turns out to hold for many more identities (but not all!). As you might guess from the recurring theme, we will view this with the perspective of linearity. With this warmup in mind, let's get introduced to Umbral calculus. 

Throughout this guide, we have seen a recurring theme: discrete math behaves eerily like continuous math if you simply swap the components correctly.

| Continuous                                                      | Discrete                                                                          |
| :-------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| Power <span class="math-inline" markdown="0">\(x^n\)</span>     | Falling Power <span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span> |
| Derivative <span class="math-inline" markdown="0">\(D\)</span>  | Difference <span class="math-inline" markdown="0">\(\Delta\)</span>               |
| Integral <span class="math-inline" markdown="0">\(\int\)</span> | Sum <span class="math-inline" markdown="0">\(\sum\)</span>                        |
| Base <span class="math-inline" markdown="0">\(e\)</span>        | Base <span class="math-inline" markdown="0">\(2\)</span>                          |

Historically, 19th-century mathematicians (like Sylvester and Cayley) noticed they could prove difficult identities regarding number sequences by pretending the indices were exponents. They called this **Umbral Calculus** (from the Latin *umbra*, meaning "shadow"), because these techniques were "shadowy" like dark arts.

For a long time, this was considered "black magic" that lacked rigor. However, in the 1970s, Gian-Carlo Rota formalized this theory using **linear functionals**, turning it into a rigorous and overpowered algebraic tool.

#### The "Umbral Trick": Linear Functionals

To understand why the discrete world mirrors the continuous one, we must define the machinery that allows us to treat indices like exponents.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Linear Functional
</div>
Let <span class="math-inline" markdown="0">\(P\)</span> be the vector space of all polynomials in the variable <span class="math-inline" markdown="0">\(z\)</span>. A **linear functional** is a map <span class="math-inline" markdown="0">\(L: P \to \mathbb{K}\)</span> (where <span class="math-inline" markdown="0">\(\mathbb{K}\)</span> is a field like <span class="math-inline" markdown="0">\(\mathbb{R}\)</span> or <span class="math-inline" markdown="0">\(\mathbb{C}\)</span>) such that for any polynomials <span class="math-inline" markdown="0">\(p(z), q(z)\)</span> and constants <span class="math-inline" markdown="0">\(c_1, c_2\)</span>:

<div class="math-block" markdown="0">
\[
L(c_1 \cdot p(z) + c_2 \cdot q(z)) = c_1 \cdot L(p(z)) + c_2 \cdot L(q(z))
\]
</div>

Essentially, <span class="math-inline" markdown="0">\(L\)</span> is a machine that "eats" a polynomial and spits out a scalar.
</blockquote>

The most common linear functional is simply **evaluation**. For example, if <span class="math-inline" markdown="0">\(L\)</span> is "evaluate at <span class="math-inline" markdown="0">\(z=0\)</span>", then <span class="math-inline" markdown="0">\(L(z+3) = 3\)</span>. (Verify that this is really is linear as an exercise). However, the power of Umbral calculus comes from defining functionals based on sequences.

**The Isomorphism:**
Suppose we have a sequence of numbers <span class="math-inline" markdown="0">\(a_0, a_1, a_2, \dots\)</span>. We can define a linear functional <span class="math-inline" markdown="0">\(L\)</span> by specifying its action on the basis vectors <span class="math-inline" markdown="0">\(z^n\)</span>:

<div class="math-block" markdown="0">
\[
L(z^n) = a_n
\]
</div>

Because <span class="math-inline" markdown="0">\(L\)</span> is linear, knowing how it acts on <span class="math-inline" markdown="0">\(z^n\)</span> tells us how it acts on *any* polynomial. This formalizes the "black magic" notation where we write <span class="math-inline" markdown="0">\(a^n\)</span> during algebraic manipulation and then "lower the index" to <span class="math-inline" markdown="0">\(a_n\)</span> at the very end.

#### Problem 1: Vandermonde's Convolution Identity

A classic problem in combinatorics is proving **Vandermonde's Identity**:

<div class="math-block" markdown="0">
\[
\sum_{k=0}^n \binom{r}{k} \binom{s}{n-k} = \binom{r+s}{n}
\]
</div>

Standard proofs involve combinatorial counting arguments (choosing a committee from two groups of people). However, in Umbral calculus, this is simply a consequence of the fact that our falling factorial basis behaves exactly like a binomial expansion.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
The Discrete Binomial Theorem
</div>
Just as <span class="math-inline" markdown="0">\((a+b)^n = \sum \binom{n}{k} a^k b^{n-k}\)</span>, the falling powers satisfy:

<div class="math-block" markdown="0">
\[
(x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
\]
</div>

</blockquote>

**Proof via Linear Functionals:**
We will prove this by mapping the standard binomial theorem to the discrete world.
Start with two variables <span class="math-inline" markdown="0">\(a, b\)</span> and the standard identity:

<div class="math-block" markdown="0">
\[
(a+b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}
\]
</div>

We define two linear functionals, <span class="math-inline" markdown="0">\(L_x\)</span> and <span class="math-inline" markdown="0">\(L_y\)</span>, that act as "basis changers" from standard powers to falling powers:
*   <span class="math-inline" markdown="0">\(L_x(a^k) = x^{\underline{k}}\)</span>
*   <span class="math-inline" markdown="0">\(L_y(b^k) = y^{\underline{k}}\)</span>
*   Let <span class="math-inline" markdown="0">\(L = L_x L_y\)</span> be the joint functional acting on products.

**1. Apply <span class="math-inline" markdown="0">\(L\)</span> to the Right Hand Side:**

<div class="math-block" markdown="0">
\[
L\left( \sum_{k=0}^n \binom{n}{k} a^k b^{n-k} \right) = \sum_{k=0}^n \binom{n}{k} L_x(a^k) L_y(b^{n-k}) = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
\]
</div>

This matches the desired expansion.

**2. Apply <span class="math-inline" markdown="0">\(L\)</span> to the Left Hand Side:**
To evaluate <span class="math-inline" markdown="0">\(L((a+b)^n)\)</span>, we look at the **Exponential Generating Functions (EGF)**.
The EGF for the standard basis is <span class="math-inline" markdown="0">\(e^{at}\)</span>.
The EGF for the falling factorial basis is <span class="math-inline" markdown="0">\((1+t)^x\)</span>, because <span class="math-inline" markdown="0">\(\sum \frac{x^{\underline{n}}}{n!}t^n = (1+t)^x\)</span>.
Therefore, the functional <span class="math-inline" markdown="0">\(L_x\)</span> is defined by the mapping <span class="math-inline" markdown="0">\(L_x(e^{at}) = (1+t)^x\)</span>.

Applying this to the joint exponential:

<div class="math-block" markdown="0">
\[
L(e^{(a+b)t}) = L(e^{at} e^{bt}) = L_x(e^{at})L_y(e^{bt}) = (1+t)^x (1+t)^y = (1+t)^{x+y}
\]
</div>

We know that <span class="math-inline" markdown="0">\((1+t)^{x+y}\)</span> is the generating function for the sequence <span class="math-inline" markdown="0">\((x+y)^{\underline{n}}\)</span>. Thus, by comparing coefficients of <span class="math-inline" markdown="0">\(t^n/n!\)</span>, we conclude:

<div class="math-block" markdown="0">
\[
L((a+b)^n) = (x+y)^{\underline{n}}
\]
</div>


Equating LHS and RHS proves the Discrete Binomial Theorem.

**Deriving Vandermonde's Identity:**
Now that we have <span class="math-inline" markdown="0">\((x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}\)</span>, we simply divide both sides by <span class="math-inline" markdown="0">\(n!\)</span>:

<div class="math-block" markdown="0">
\[
\frac{(x+y)^{\underline{n}}}{n!} = \sum_{k=0}^n \frac{1}{n!} \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
\]
</div>

Using the identity <span class="math-inline" markdown="0">\(\binom{n}{k} = \frac{n!}{k!(n-k)!}\)</span> and rearranging terms:

<div class="math-block" markdown="0">
\[
\frac{(x+y)^{\underline{n}}}{n!} = \sum_{k=0}^n \frac{x^{\underline{k}}}{k!} \frac{y^{\underline{n-k}}}{(n-k)!}
\]
</div>

Substituting <span class="math-inline" markdown="0">\(\frac{z^{\underline{m}}}{m!} = \binom{z}{m}\)</span>, we recover the identity:

<div class="math-block" markdown="0">
\[
\boxed{ \binom{x+y}{n} = \sum_{k=0}^n \binom{x}{k} \binom{y}{n-k} }
\]
</div>


#### Problem 2: Bernoulli Numbers and Sums of Powers

The "Killer App" of Umbral Calculus is solving for sums of powers, <span class="math-inline" markdown="0">\(\sum_{k=0}^{m-1} k^p\)</span>.
While we know <span class="math-inline" markdown="0">\(\sum k = \frac{m(m-1)}{2}\)</span>, formulas for <span class="math-inline" markdown="0">\(\sum k^{10}\)</span> are tedious to find manually. Umbral calculus gives us **Faulhaber's Formula** for the sum of first natural number powers almost for free.

First, we define the **Bernoulli Numbers** <span class="math-inline" markdown="0">\(B_n\)</span> using an implicit Umbral recurrence. We pretend <span class="math-inline" markdown="0">\(B\)</span> is a variable, write the relation, and then "lower the shadow" (convert powers <span class="math-inline" markdown="0">\(B^k\)</span> to indices <span class="math-inline" markdown="0">\(B_k\)</span> by a linear functional <span class="math-inline" markdown="0">\(L: B^k \mapsto B_{k}\)</span>).

**Definition:** <span class="math-inline" markdown="0">\(B_0 = 1\)</span>, and for <span class="math-inline" markdown="0">\(n > 1\)</span>:

<div class="math-block" markdown="0">
\[
L ((B+1)^n - B^n) = 0
\]
</div>

*(Note: This expands to <span class="math-inline" markdown="0">\(L\left( \sum_{k=0}^{n} \binom{n}{k} B^k - B^n \right) = 0\)</span>, or <span class="math-inline" markdown="0">\(\sum_{k=0}^{n-1} \binom{n}{k} B_k = 0\)</span>).*

Now, consider the problem of integration. In standard calculus, <span class="math-inline" markdown="0">\(\int x^n dx = \frac{x^{n+1}}{n+1}\)</span>.
In discrete calculus, we are looking for the sum, which is the anti-difference <span class="math-inline" markdown="0">\(\Delta^{-1}\)</span>. It turns out the Bernoulli numbers allow us to integrate standard powers <span class="math-inline" markdown="0">\(x^n\)</span> directly.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
Faulhaber’s Formula (Umbral Form)
</div>

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{m-1} k^n = \frac{1}{n+1} \left( (m+B)^{n+1} - B^{n+1} \right)
\]
</div>

</blockquote>

<details class="box-example" markdown="1">
<summary markdown="1">
Sum of First <span class="math-inline" markdown="0">\(m\)</span> Squares (<span class="math-inline" markdown="0">\(n=2\)</span>)
</summary>
We want to compute <span class="math-inline" markdown="0">\(\sum_{k=0}^{m-1} k^2\)</span>. Using the formula with <span class="math-inline" markdown="0">\(n=2\)</span>:

<div class="math-block" markdown="0">
\[
\Sigma = \frac{1}{3} \left( (m+B)^3 - B^3 \right)
\]
</div>

Expand the binomial <span class="math-inline" markdown="0">\((m+B)^3 = m^3 + 3m^2B + 3mB^2 + B^3\)</span>:

<div class="math-block" markdown="0">
\[
\Sigma = \frac{1}{3} \left( (m^3 + 3m^2B^1 + 3mB^2 + B^3) - B^3 \right)
\]
</div>


<div class="math-block" markdown="0">
\[
\Sigma = \frac{1}{3} m^3 + m^2 B^1 + m B^2
\]
</div>

Now we "lower the shadow." We interpret <span class="math-inline" markdown="0">\(B^1\)</span> as <span class="math-inline" markdown="0">\(B_1\)</span> and <span class="math-inline" markdown="0">\(B^2\)</span> as <span class="math-inline" markdown="0">\(B_2\)</span>.
From the definition of Bernoulli numbers:
*   <span class="math-inline" markdown="0">\(B_0 = 1\)</span>
*   <span class="math-inline" markdown="0">\((B+1)^2 - B^2 = 0 \implies B_0 + 2B_1 = 0 \implies 1 + 2B_1 = 0 \implies \mathbf{B_1 = -1/2}\)</span>
*   <span class="math-inline" markdown="0">\((B+1)^3 - B^3 = 0 \implies B_0 + 3B_1 + 3B_2 = 0 \implies 1 - \frac{3}{2} + 3B_2 = 0 \implies \mathbf{B_2 = 1/6}\)</span>

Substitute these values back:

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{m-1} k^2 = \frac{1}{3}m^3 - \frac{1}{2}m^2 + \frac{1}{6}m
\]
</div>

This is exactly the standard formula <span class="math-inline" markdown="0">\(\frac{m(m-1)(2m-1)}{6}\)</span> (adjusted for the summation limit <span class="math-inline" markdown="0">\(m-1\)</span>).
</details>

### Bernoulli Polynomials
We have seen two parallel worlds:
1.  **Continuous:** <span class="math-inline" markdown="0">\(D x^n = n x^{n-1}\)</span> is easy, so we integrate <span class="math-inline" markdown="0">\(x^n\)</span> easily.
2.  **Discrete:** <span class="math-inline" markdown="0">\(\Delta x^{\underline{n}} = n x^{\underline{n-1}}\)</span> is easy, so we sum <span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span> easily.

But what if we stubbornly want to sum standard powers <span class="math-inline" markdown="0">\(x^n\)</span> without converting to falling powers? We need a sequence of polynomials <span class="math-inline" markdown="0">\(P_n(x)\)</span> that satisfies a "mixed" property:

<div class="math-block" markdown="0">
\[
\Delta P_n(x) = n x^{n-1}
\]
</div>

If we found such polynomials, we could sum <span class="math-inline" markdown="0">\(x^n\)</span> immediately by telescoping: <span class="math-inline" markdown="0">\(\sum k^n = \frac{P_{n+1}(k)}{n+1}\)</span>.
These are exactly the **Bernoulli Polynomials**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Bernoulli Polynomials (Umbral Definition)
</div>
Using the same umbral variable <span class="math-inline" markdown="0">\(B\)</span> (where <span class="math-inline" markdown="0">\(B^k \to B_k\)</span>) from the previous section, we define the Bernoulli polynomial <span class="math-inline" markdown="0">\(B_n(x)\)</span> as:

<div class="math-block" markdown="0">
\[
B_n(x) := (B + x)^n = \sum_{k=0}^n \binom{n}{k} B_k x^{n-k}
\]
</div>

</blockquote>

This definition reveals their superpower. In the continuous world, they are an **Appell sequence**, meaning their derivative lowers their degree:

<div class="math-block" markdown="0">
\[
\frac{d}{dx} B_n(x) = \frac{d}{dx} (B+x)^n = n(B+x)^{n-1} = n B_{n-1}(x)
\]
</div>


But in the discrete world, they satisfy an even more important property.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
The Difference Property
</div>

<div class="math-block" markdown="0">
\[
\Delta B_n(x) = n x^{n-1}
\]
</div>

</blockquote>

<blockquote class="box-proof" markdown="1">

Using the Umbral definition <span class="math-inline" markdown="0">\(B_n(x) = (B+x)^n\)</span>:

<div class="math-block" markdown="0">
\[
\begin{align}
\Delta B_n(x) &= (B+x+1)^n - (B+x)^n \\
&= \sum_{k=0}^n \binom{n}{k} (B+1)^k x^{n-k} - \sum_{k=0}^n \binom{n}{k} B^k x^{n-k} \\
&= \sum_{k=0}^n \binom{n}{k} x^{n-k} \left[ (B+1)^k - B^k \right]
\end{align}
\]
</div>

Recall the definition of Bernoulli numbers: <span class="math-inline" markdown="0">\((B+1)^k - B^k\)</span> is <span class="math-inline" markdown="0">\(0\)</span> for all <span class="math-inline" markdown="0">\(k > 1\)</span>.
For <span class="math-inline" markdown="0">\(k=0\)</span>, the term is <span class="math-inline" markdown="0">\(1-1=0\)</span>.
The only term that survives is <span class="math-inline" markdown="0">\(k=1\)</span>, where <span class="math-inline" markdown="0">\((B+1)^1 - B^1 = 1\)</span>.
Thus, the sum collapses to the term where <span class="math-inline" markdown="0">\(k=1\)</span>:

<div class="math-block" markdown="0">
\[
\Delta B_n(x) = \binom{n}{1} x^{n-1} (1) = n x^{n-1}
\]
</div>

</blockquote>

#### The General Summation Formula
Because <span class="math-inline" markdown="0">\(\Delta \frac{B_{n+1}(x)}{n+1} = x^n\)</span>, we can integrate (sum) <span class="math-inline" markdown="0">\(x^n\)</span> directly. This gives us the explicit form of Faulhaber's formula for any range <span class="math-inline" markdown="0">\([0, m)\)</span>:


<div class="math-block" markdown="0">
\[
\sum_{k=0}^{m-1} k^n = \sum_{k=0}^{m-1} \Delta \left( \frac{B_{n+1}(k)}{n+1} \right) = \boxed{ \frac{B_{n+1}(m) - B_{n+1}(0)}{n+1} }
\]
</div>


Since <span class="math-inline" markdown="0">\(B_{n+1}(0) = B_{n+1}\)</span> (the Bernoulli number), this matches our previous Umbral formula.

<details class="box-example" markdown="1">
<summary markdown="1">
Sum of first <span class="math-inline" markdown="0">\(m-1\)</span> squares
</summary>
Let's find <span class="math-inline" markdown="0">\(\sum_{k=0}^{m-1} k^2\)</span> using the polynomial <span class="math-inline" markdown="0">\(B_3(x)\)</span>.

First, construct <span class="math-inline" markdown="0">\(B_3(x)\)</span> using <span class="math-inline" markdown="0">\(B_0=1, B_1=-1/2, B_2=1/6\)</span>:

<div class="math-block" markdown="0">
\[
\begin{align}
B_3(x) &= \sum_{k=0}^3 \binom{3}{k} B_k x^{3-k} \\
&= \binom{3}{0}B_0 x^3 + \binom{3}{1}B_1 x^2 + \binom{3}{2}B_2 x + \binom{3}{3}B_3 \\
&= 1 \cdot x^3 + 3(-\frac{1}{2})x^2 + 3(\frac{1}{6})x + B_3 \\
&= x^3 - \frac{3}{2}x^2 + \frac{1}{2}x + B_3
\end{align}
\]
</div>

Now apply the formula <span class="math-inline" markdown="0">\(\sum k^2 = \frac{B_3(m) - B_3(0)}{3}\)</span>. Note that <span class="math-inline" markdown="0">\(B_3(0)\)</span> is just the constant term <span class="math-inline" markdown="0">\(B_3\)</span>, so they cancel out:

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{m-1} k^2 = \frac{1}{3} \left( x^3 - \frac{3}{2}x^2 + \frac{1}{2}x \right) = \frac{m^3}{3} - \frac{m^2}{2} + \frac{m}{6}
\]
</div>

Which is the correct result.
</details>
### Bonus: The Continuous Operator Algebra

To wrap up, let's look at how these operator techniques apply to the continuous world. throughout this text, we saw that the discrete derivative <span class="math-inline" markdown="0">\(\Delta\)</span> and the falling power <span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span> satisfy the relationship <span class="math-inline" markdown="0">\(\Delta x^{\underline{n}} = n x^{\underline{n-1}}\)</span>.

In the continuous world, we have the derivative <span class="math-inline" markdown="0">\(D = \frac{d}{dx}\)</span> and the position operator <span class="math-inline" markdown="0">\(x\)</span> (multiplying by <span class="math-inline" markdown="0">\(x\)</span>). The fundamental relationship between them is the **commutator**:

<div class="math-block" markdown="0">
\[
[D, x] = Dx - xD = 1
\]
</div>

(This is because <span class="math-inline" markdown="0">\(D(xf) - x(Df) = (f + xf') - xf' = f\)</span>).

This non-commutative structure generates the **Hermite Polynomials** (specifically the "probabilist" convention <span class="math-inline" markdown="0">\(He_n(x)\)</span> used in statistics and Brownian motion), which are the continuous analogs to the discrete falling factorials.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Probabilist's Hermite Polynomials
</div>
Instead of using a Taylor expansion, we can define these polynomials purely through an operator acting on the constant <span class="math-inline" markdown="0">\(1\)</span>:

<div class="math-block" markdown="0">
\[
He_n(x) := (x - D)^n \cdot 1
\]
</div>


For example:
- <span class="math-inline" markdown="0">\(He_0(x) = 1\)</span>
- <span class="math-inline" markdown="0">\(He_1(x) = (x - D)1 = x\)</span>
- <span class="math-inline" markdown="0">\(He_2(x) = (x - D)x = x^2 - Dx = x^2 - 1\)</span>
- <span class="math-inline" markdown="0">\(He_3(x) = (x - D)(x^2 - 1) = x(x^2 - 1) - D(x^2 - 1) = x^3 - x - 2x = x^3 - 3x\)</span>
</blockquote>

Does this structure look familiar? It should. In the discrete case, <span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span> is the eigenbasis for the operator <span class="math-inline" markdown="0">\(x\Delta\)</span>. In the continuous case, <span class="math-inline" markdown="0">\(He_n(x)\)</span> allows us to solve differential equations involving <span class="math-inline" markdown="0">\(e^{-x^2/2}\)</span> purely through algebra.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
The Hermite Recurrence Relation
</div>
Using **only** the operator definition <span class="math-inline" markdown="0">\(He_n(x) = (x-D)^n \cdot 1\)</span> and the commutator <span class="math-inline" markdown="0">\([D, x] = 1\)</span>, prove the famous three-term recurrence relation:

<div class="math-block" markdown="0">
\[
He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
\]
</div>


**Hint:**
1. Write <span class="math-inline" markdown="0">\(He_{n+1} = (x-D)(x-D)^n \cdot 1\)</span>.
2. Expand the product. You will need to figure out how to swap <span class="math-inline" markdown="0">\(D\)</span> and <span class="math-inline" markdown="0">\((x-D)^n\)</span>.
3. Prove the lemma that <span class="math-inline" markdown="0">\([D, (x-D)^n] = n(x-D)^{n-1}\)</span> (which looks suspiciously like a power rule!).
</blockquote>

<details class="box-solution" markdown="1">
<summary markdown="1">
**Click for Solution**
</summary>
We start with the definition:

<div class="math-block" markdown="0">
\[
He_{n+1}(x) = (x - D) \underbrace{(x - D)^n \cdot 1}_{He_n(x)}
\]
</div>


<div class="math-block" markdown="0">
\[
He_{n+1}(x) = x He_n(x) - D He_n(x)
\]
</div>


Now we need to evaluate <span class="math-inline" markdown="0">\(D He_n(x)\)</span>. Let's find the commutator of <span class="math-inline" markdown="0">\(D\)</span> with the operator <span class="math-inline" markdown="0">\(A = (x-D)\)</span>.
Note that <span class="math-inline" markdown="0">\([D, x-D] = [D, x] - [D, D] = 1 - 0 = 1\)</span>.

Since <span class="math-inline" markdown="0">\(D\)</span> commutes with itself, the commutator <span class="math-inline" markdown="0">\([D, A^n]\)</span> follows the derivative rule for operators (similar to <span class="math-inline" markdown="0">\(d/dx(f^n) = n f^{n-1} f'\)</span>):

<div class="math-block" markdown="0">
\[
[D, (x-D)^n] = n(x-D)^{n-1} [D, (x-D)] = n(x-D)^{n-1} \cdot 1
\]
</div>


So, we have the identity:

<div class="math-block" markdown="0">
\[
D(x-D)^n = (x-D)^n D + n(x-D)^{n-1}
\]
</div>


Apply this to the constant function <span class="math-inline" markdown="0">\(1\)</span> (remembering that <span class="math-inline" markdown="0">\(D \cdot 1 = 0\)</span>):

<div class="math-block" markdown="0">
\[
D He_n(x) = D(x-D)^n \cdot 1 = \left( (x-D)^n D + n(x-D)^{n-1} \right) \cdot 1
\]
</div>


<div class="math-block" markdown="0">
\[
D He_n(x) = 0 + n \underbrace{(x-D)^{n-1} \cdot 1}_{He_{n-1}(x)}
\]
</div>


<div class="math-block" markdown="0">
\[
D He_n(x) = n He_{n-1}(x)
\]
</div>


Substituting this back into our first equation:

<div class="math-block" markdown="0">
\[
\boxed{ He_{n+1}(x) = x He_n(x) - n He_{n-1}(x) }
\]
</div>


This exercise confirms that <span class="math-inline" markdown="0">\(He_n(x)\)</span> behaves in the continuous "operator world" exactly like the falling powers <span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span> behave in the discrete world, just with a different "creation operator" (<span class="math-inline" markdown="0">\(x-D\)</span> instead of multiplication by <span class="math-inline" markdown="0">\(x\)</span>).
</details>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Appell sequence
</div>
Prove that 

<div class="math-block" markdown="0">
\[
\frac{d}{dx} He_{n}(x)=nHe_{n-1}(x)
\]
</div>

</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Shift operator
</div>
Show that

<div class="math-block" markdown="0">
\[
e^{tD}xe^{-tD}=x+t
\]
</div>

Then prove for suitable <span class="math-inline" markdown="0">\(f\)</span>

<div class="math-block" markdown="0">
\[
(e^{tD}f)(x)=f(x+t)
\]
</div>

Hint: take <span class="math-inline" markdown="0">\(\frac{d}{dt} (e^{tD}xe^{-tD})\)</span> using <span class="math-inline" markdown="0">\([D,x]=1\)</span>
</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Ladder operator
</div>
Define operators

<div class="math-block" markdown="0">
\[
a=\frac{1}{\sqrt{2}}(x+D), \qquad a^{\dagger}=\frac{1}{\sqrt{ 2 }}(x-D)
\]
</div>

Use only <span class="math-inline" markdown="0">\([D,x]=1\)</span> to prove <span class="math-inline" markdown="0">\([a,a^{\dagger}]=1\)</span>.
Hint: expand <span class="math-inline" markdown="0">\(aa^{\dagger},a^{\dagger}a\)</span> in <span class="math-inline" markdown="0">\(x,D\)</span>.
</blockquote>

### Conclusion

Discrete Calculus is not just a "low-resolution" version of Continuous Calculus. It is a parallel universe with its own basis functions (<span class="math-inline" markdown="0">\(x^{\underline{n}}\)</span>), operators (<span class="math-inline" markdown="0">\(\Delta, \Sigma\)</span>), and constants (<span class="math-inline" markdown="0">\(B_n\)</span>).

By understanding the mapping between these worlds—using tools like Stirling numbers to change bases, and linear functionals to manipulate "shadows"—you can solve complex combinatorial summations and difference equations as easily as you solve integrals in standard calculus.

When you face a problem in discrete math involving sums of polynomials, binomial coefficients, or recursive sequences, ask yourself: **"What would this look like if I wrote it in the falling factorial basis?"** Usually, the answer is "a lot simpler."

---

**Exercise 1: Umbral Evaluation**
Let <span class="math-inline" markdown="0">\(L\)</span> be a linear functional defined on the basis <span class="math-inline" markdown="0">\(z^n\)</span> such that <span class="math-inline" markdown="0">\(L(z^n) = n!\)</span> (This corresponds to the Gamma function, <span class="math-inline" markdown="0">\(\int_0^\infty t^n e^{-t} dt\)</span>).
Compute <span class="math-inline" markdown="0">\(L((z+1)^2)\)</span> in two steps:
1. Expand the polynomial <span class="math-inline" markdown="0">\((z+1)^2\)</span> algebraically.
2. Apply <span class="math-inline" markdown="0">\(L\)</span> to the resulting terms.

**Exercise 2: Proving the Difference Rule**
Using the Discrete Binomial Theorem <span class="math-inline" markdown="0">\((x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}\)</span>, prove that:

<div class="math-block" markdown="0">
\[
\Delta (x^{\underline{n}}) = n x^{\underline{n-1}}
\]
</div>

*Hint: Recall that <span class="math-inline" markdown="0">\(\Delta f(x) = f(x+1) - f(x)\)</span>. Set <span class="math-inline" markdown="0">\(y=1\)</span> in the expansion.*

**Exercise 3: Sum of Integers**
Use the Umbral Faulhaber Formula to derive the sum of the first <span class="math-inline" markdown="0">\(m\)</span> integers (<span class="math-inline" markdown="0">\(n=1\)</span>).

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{m-1} k = \frac{1}{2} \left( (m+B)^2 - B^2 \right)
\]
</div>

Remember that <span class="math-inline" markdown="0">\(B_1 = -1/2\)</span>. Does your result match <span class="math-inline" markdown="0">\(\frac{m(m-1)}{2}\)</span>?

---
## Exercises

### Part 1: The Difference Operator and Falling Powers

**Exercise 1: The Discrete Product Rule**
Recall the discrete product rule: <span class="math-inline" markdown="0">\(\Delta(u v) = u \Delta v + E(v) \Delta u\)</span>, where <span class="math-inline" markdown="0">\(E(v) = v(x+1)\)</span>.
Calculate <span class="math-inline" markdown="0">\(\Delta(x \cdot 2^x)\)</span> in two ways:
1.  By applying the definition <span class="math-inline" markdown="0">\(\Delta f(x) = f(x+1) - f(x)\)</span> directly.
2.  By applying the discrete product rule with <span class="math-inline" markdown="0">\(u=x\)</span> and <span class="math-inline" markdown="0">\(v=2^x\)</span>.

**Exercise 2: The Discrete Power Rule**
We know that <span class="math-inline" markdown="0">\(\Delta x^\underline{n} = n x^\underline{n-1}\)</span>.
Using the linearity of <span class="math-inline" markdown="0">\(\Delta\)</span>, find the forward difference of the polynomial:

<div class="math-block" markdown="0">
\[
f(x) = 3x^\underline{4} - 5x^\underline{2} + 7
\]
</div>


**Exercise 3: Converting Bases (Stirling Numbers)**
To differentiate (difference) standard powers like <span class="math-inline" markdown="0">\(x^3\)</span>, we usually convert them to falling powers first.
1.  Express <span class="math-inline" markdown="0">\(x^3\)</span> as a linear combination of falling powers (<span class="math-inline" markdown="0">\(Ax^\underline{3} + Bx^\underline{2} + Cx^\underline{1}\)</span>) using algebraic manipulation or Stirling numbers of the second kind.
2.  Use your result to calculate <span class="math-inline" markdown="0">\(\Delta x^3\)</span>. Verify that your answer matches <span class="math-inline" markdown="0">\((x+1)^3 - x^3\)</span>.

---

### Part 2: Discrete Integration (Summation)

**Exercise 4: Sum of Cubes**
Using the method of falling powers, calculate the closed-form formula for the sum of the first <span class="math-inline" markdown="0">\(n\)</span> cubes:

<div class="math-block" markdown="0">
\[
\sum_{k=0}^{n} k^3
\]
</div>

*Hint: Use your conversion from Exercise 3, then apply the Fundamental Theorem of Discrete Calculus: <span class="math-inline" markdown="0">\(\sum_{0}^{n} x^\underline{k} \delta x = \left[ \frac{x^\underline{k+1}}{k+1} \right]_0^{n+1}\)</span>.*

**Exercise 5: The Geometric Series**
In continuous calculus, <span class="math-inline" markdown="0">\(\int e^{ax} dx = \frac{1}{a}e^{ax}\)</span>.
In discrete calculus, we know <span class="math-inline" markdown="0">\(\Delta c^x = c^x(c-1)\)</span>.
1.  Use the antidifference concept to find a closed form for <span class="math-inline" markdown="0">\(\sum_{k=0}^{n-1} 3^k\)</span>.
2.  Generalize this to find the standard formula for a geometric series <span class="math-inline" markdown="0">\(\sum_{k=0}^{n-1} r^k\)</span>.

---

### Part 3: Newton Series and Interpolation

**Exercise 6: Finding the Function**
You are given a sequence generated by a polynomial function <span class="math-inline" markdown="0">\(f(n)\)</span>:

<div class="math-block" markdown="0">
\[
0, 6, 24, 60, 120, \dots
\]
</div>

(for <span class="math-inline" markdown="0">\(n=0, 1, 2, 3, 4 \dots\)</span>)

1.  Construct a **difference table** (calculate <span class="math-inline" markdown="0">\(\Delta f\)</span>, <span class="math-inline" markdown="0">\(\Delta^2 f\)</span>, etc.) until the differences become constant.
2.  Use the Newton Series formula to write <span class="math-inline" markdown="0">\(f(n)\)</span> as a sum of falling powers:

    <div class="math-block" markdown="0">
\[
f(n) = \sum_{k=0}^\infty \frac{(\Delta^k f)(0)}{k!} n^\underline{k}
\]
    </div>

3.  Convert your answer back to standard powers (<span class="math-inline" markdown="0">\(n^3, n^2\)</span>, etc.) to find the standard polynomial form.

---

### Part 4: Conceptual Proofs

**Exercise 7: The Pascal's Triangle Identity**
The binomial coefficient is defined as <span class="math-inline" markdown="0">\(\binom{n}{k} = \frac{n^\underline{k}}{k!}\)</span>.
Using the property <span class="math-inline" markdown="0">\(\Delta n^\underline{k} = k n^\underline{k-1}\)</span>, prove that:

<div class="math-block" markdown="0">
\[
\Delta \binom{n}{k} = \binom{n}{k-1}
\]
</div>

Then, write out what this equation means in terms of <span class="math-inline" markdown="0">\(n\)</span> and <span class="math-inline" markdown="0">\(k\)</span> using the definition <span class="math-inline" markdown="0">\(\Delta f(n) = f(n+1)-f(n)\)</span>. Does it look like a familiar identity from Pascal's Triangle?

**Exercise 8: Summation by Parts (Discrete Integration by Parts)**
The integration by parts formula is <span class="math-inline" markdown="0">\(\int u dv = uv - \int v du\)</span>.
Derive the **Summation by Parts** formula by summing the discrete product rule identity (<span class="math-inline" markdown="0">\(\Delta(uv) = u \Delta v + E(v) \Delta u\)</span>).
*Result should look like:*

<div class="math-block" markdown="0">
\[
\sum_{k=a}^{b-1} u_k \Delta v_k = [u_k v_k]_a^b - \sum_{k=a}^{b-1} v_{k+1} \Delta u_k
\]
</div>


---

### Solutions / Hints

1. <span class="math-inline" markdown="0">\(2^x(x+2)\)</span>.
2. <span class="math-inline" markdown="0">\(12x^\underline{3} - 10x^\underline{1}\)</span>.
3. <span class="math-inline" markdown="0">\(x^3 = x^\underline{3} + 3x^\underline{2} + x^\underline{1}\)</span>. Thus <span class="math-inline" markdown="0">\(\Delta x^3 = 3x^\underline{2} + 6x^\underline{1} + 1\)</span>.
4. <span class="math-inline" markdown="0">\(\left(\frac{n(n+1)}{2}\right)^2\)</span>.
5. <span class="math-inline" markdown="0">\(\sum r^k = \frac{r^n - 1}{r-1}\)</span>.
6. The terms are <span class="math-inline" markdown="0">\(n^3 - n\)</span> (or <span class="math-inline" markdown="0">\(1n^\underline{3} + 3n^\underline{2} + 2n^\underline{1}\)</span> before conversion).
7. This recovers <span class="math-inline" markdown="0">\(\binom{n+1}{k} - \binom{n}{k} = \binom{n}{k-1}\)</span>, which is Pascal's Identity.
8. Sum both sides of <span class="math-inline" markdown="0">\(\Delta(u_k v_k) = u_k \Delta v_k + v_{k+1} \Delta u_k\)</span>. Rearrange to isolate <span class="math-inline" markdown="0">\(\sum u \Delta v\)</span>. Note that the boundary term is <span class="math-inline" markdown="0">\([u_k v_k]_a^b\)</span>.

## References and Further Reading

{% bibliography --file posts/2025-11-26-discrete-calculus/discrete-calculus.bib %}

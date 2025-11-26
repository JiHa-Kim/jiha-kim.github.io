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

$$
\frac{d}{dt}w_{t}=-\nabla_{w_{t}} f(w_{t})
$$

for some weights \\(w_{t}\\) evolving through training time \\(t\\) under a loss \\(f\\).
We approximate by discretizing the derivative to a *finite difference quotient*.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Forward difference operator
</div>
The forward difference operator

$$
\Delta_{h}f(t):=f(t+h)-f(t)
$$

</blockquote>

Note that by definition of a derivative,


$$
\frac{dw_{t}}{dt}=\lim_{ h \to 0 } \frac{w_{t+h}-w_{t}}{h}=\lim_{ h \to 0 } \frac{\Delta_{h}w_{t}}{\Delta_{h}t}
$$

So in the discrete case, we use a finite \\(h\\) instead of the limit as \\(h\to 0\\). This gives us the finite difference quotient \\(\frac{\Delta_{h}w_{t}}{\Delta_{h}t}=\frac{w_{t+h}-w_{t}}{h}\\).

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Gradient descent algorithm
</div>
Discretizing the gradient flow ODE through the forward difference operator is called forward Euler integration, and recovers the gradient descent algorithm.
We replace \\(\frac{d}{dt}\\) by \\(\frac{\Delta_{h}}{\Delta_{h}t}\\) in the gradient flow ODE:

$$
\frac{\Delta_{h}}{\Delta_{h}t}w_{t}=-\nabla f(w_{t})=\frac{w_{t+h}-w_{t}}{h}
$$

Solving for \\(w_{t+h}\\) gives the explicit form:

$$
w_{t+h}=w_{t}-h\nabla f(w_{t})
$$

</blockquote>

Indeed, discrete calculus is more far-reaching than you might think. Now, we can index discrete objects using the natural numbers. Hence, we will often think of time as steps \\(t=1,2,3,\dots\\) and analyze the specific case with normalized increment to \\(h=1\\) so that \\(\Delta f(x)=f(x+1)-f(x)\\). 

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Forward difference operator \\(\Delta\\)
</div>

$$
\Delta f(x):=f(x+1)-f(x)
$$


(Note: the notation \\(\Delta\\) clashes with the symbol commonly used for the Laplacian, so be careful to avoid confusion.)
</blockquote>

This has the convenient effect that the denominator in the finite difference quotient just cancels, so we only need to keep track of the difference itself:

$$
\frac{\Delta f(x)}{\Delta x}=\frac{f(x+1)-f(x)}{(x+1)-x}=\frac{f(x+1)-f(x)}{1}=\Delta f(x)
$$


### Difference and Sum Operators
From the section above, the discrete analog to the derivative is the difference \\(\Delta\\). Then what is the integral? Straightforwardly, it is simply the sum operator \\(\sum\\). Then, some results in discrete calculus seem very familiar to their infinitesimal counterparts.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Properties of forward difference and sum
</div>
- Linearity: for constants \\(a,b\\), 

$$
\Delta (af+bg)=a\Delta f+b\Delta g
$$


$$
\sum(af+bg)=a\sum f+b\sum g
$$

- Constant rule: for a constant \\(c\\), \\(\Delta c=0\\).
- Fundamental theorem of discrete calculus:
1. Telescoping: 
$$
\sum_{k=a}^{b}\Delta f(k)=f(b+1)-f(a)
$$

2. This doesn't really have a name, maybe we should call it "microscoping" in spirit of the other part, since we zoom in on a single term. 
$$
\frac{\Delta}{\Delta n} \sum_{k=a}^{n-1}f(k)=f(n)
$$

- Compared to the infinitesimal case, where we have \\(\int_{[a,b)}=\int_{[a,b]}\\) so \\(\int_{[a,b)} \frac{df}{dx} \,dx=f(b)-f(a)\\), the distinction does matter in the discrete case, where we sum \\(\Delta f(k)\\) over the integer interval \\([a,n)=\{ a, \dots, n-1 \}\\) to recover \\(f(n)\\).
However, in the case of the product rule, the discrete version differs from its infinitesimal counterpart, because we don't discard the second-order term:
- Discrete product rule: \\(\Delta (fg)=f\Delta g+\Delta f g+\Delta f \Delta g=\\)
- Compare to continuous product rule: \\(d(fg)=(f+df)(g+dg)-fg=f\,dg+g\,df+\cancel{ df\,dg }\\)
</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Summation by parts
</div>
- The product rule can be simplified to 

$$
\Delta(fg)(x)=f(x+1)\Delta g(x)+\Delta f(x)g(x)=\Delta f(x)g(x+1)+f(x)\Delta g(x)
$$

Re-arranging and summing both sides gives summation by parts:

$$
\sum_{x=a}^{b-1} f(x) \Delta g(x)=f(b)g(b)-f(a)g(a)-\sum_{x=a}^{b-1} \Delta f(x)g(x+1)
$$

</blockquote>

When I first learned about the fundamental theorems of calculus, I just thought of them as mechanical rules to apply to crunch algebra. But the discrete versions seem so obvious, and they explain what's really going on: because we are accumulating changes, everything along your path will eventually cancel out, leaving only the endpoints relevant. 

A similar logic generalizes in several dimensions if you think about walking along a square, or a cube. In the continuous case, Green's theorem and Stokes' generalized integration theorem basically say that a change in one direction will be cancelled by a change in another direction unless you lie along the boundary.

This image from [Wikipedia](https://en.wikipedia.org/wiki/Discrete_calculus#Discrete_differential_forms:_cochains) sums it well:
![Paths in the interior cancel each other out](https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Stokes_patch.svg/2560px-Stokes_patch.svg.png)

### Polynomials: Falling Powers
The first interesting functions we differentiate in calculus are the monomials:
<blockquote class="box-fact" markdown="1">
<div class="title" markdown="1">
Derivative of \\(x^n\\)
</div>

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

</blockquote>

Naturally, let's start with them for the discrete case as well. For example, take 

$$
\Delta x=(x+1)-x=1
$$
So far so good. Let's move up a degree.

$$
\Delta x^2=(x+1)^2-x^2=2x+1
$$

Huh. That's not quite the same as in the continuous version. Strange. Let's try a few more:

$$
\Delta x^3=(x+1)^3-x^3=3x^2+3x+1
$$


$$
\Delta x^4=(x+1)^4-x^4=4x^3+6x^2+4x+1
$$

Okay, this is really not going as well as we expected, given that all the properties of operators seemed to match up pretty well before. Let's take a step back and think through what goes differently in the discrete versus the continuous case.

Typically, we do the proof using the binomial theorem, \\((a+b)^n=\sum_{k=0}^{n} \binom{n}{k}a^k b^{n-k}\\).

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Derivative of \\(x^n\\)
</div>

$$
\frac{d}{dx}x^n=\lim_{ h \to 0 } \frac{(x+h)^n-x^n}{h}=\lim_{ h \to 0 } \frac{1}{h}\sum_{k=1}^{n}\binom{n}{k}h^k x^{n-k}=\binom{n}{1}x^{n-1}=nx^{n-1}
$$

</blockquote>

The crucial step here is that \\(\frac{1}{h}\cdot h^k=h^{k-1}\to 0\\) as \\(h\to 0\\) for any \\(k>1\\). But in the discrete case, this doesn't happen: the terms survive as \\(h=1\\). So for \\(\Delta x^n=(x+1)^n-x^n\\) we have that the coefficients \\((x+1)^n\\) are literally just generated by the binomial expansion in Pascal's triangle, and we delete the highest degree due to \\(-x^n\\).


$$
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
$$

_Binomial expansion of \\((x+1)^n\\) for \\(n=1,2,3,4,5\\)_

But having explored linearity so far, we know that we should be able to use linear combinations to construct a polynomial basis that would indeed satisfy an identity like \\(\frac{d}{dx}x^n=nx^{n-1}\\). Let's try it out.

By exploiting the linearity of \\(\Delta\\), let's construct polynomials \\(p_{n}(x)\\) so that \\(\Delta p_{n}(x)=np_{n-1}(x)\\). From the table, we start at \\(\Delta x^0=\Delta 1=0 \implies p_{0}=1\\) from the constant rule. Also, \\(\Delta x=1=1x^0\\), giving \\(p_{1}=x\\). 

But the problem arises starting from quadratics, where \\(\Delta x^2=2x+1\\). But \\(2x=\frac{d}{dx}x^2\\) is the part we are looking for, so let's isolate it and use the fundamental theorem of discrete calculus. With it, the equation \\(\Delta p_{n}=np_{n-1}\\) is equivalent to \\(p_{n}=\sum np_{n-1}\\), i.e. we are computing its anti-difference = sum.

$$
2x=\Delta x^2-1=\Delta x^2-\Delta x=\Delta (x^2-x)=\Delta [x(x-1)]=\Delta p_{2}
$$

Nice, \\(p_{2}=x(x-1)\\). Let's try \\(\Delta p_{3}=3p_{2}=3x(x-1)=3x^2-3x\\):

$$
3x^2-3x=(\Delta x^3-3x-1)-3x=\Delta x^3-6x-1=\Delta x^3-3\Delta x(x-1)-\Delta x
$$

Hence, 

$$
p_{3}=x^3-3x(x-1)-x=x(x^2-3x+3-1)=x(x^2-3x+2)=x(x-1)(x-2)
$$

Hold on, that seems like a suspicious pattern. So far, we got:

$$
\begin{align}
p_{0} &= 1 \\
p_{1} &= x \\
p_{2} &= x(x-1) \\
p_{3} &= x(x-1)(x-2)
\end{align}
$$

It seems like our polynomials will look like \\(x(x-1)(x-2)(x-3)\dots\\) as we go on. 
<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Try one more!
</div>
As we did before, compute \\(p_{4}\\) from \\(4p_{3}=\Delta p_{4}\\) and verify that the pattern still holds: \\(p_{4}=x(x-1)(x-2)(x-3)\\).
</blockquote>

Let's see if we can verify this. Define the falling power:
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Falling power \\(x^\underline{n}\\)
</div>

$$
x^\underline{n}:=\underbrace{ x(x-1)(x-2)\dots(x-(n-1)) }_{ n\text{ terms} }
$$

By convention, \\(x^\underline{0}=1\\).
</blockquote>

Now, let's try taking \\(\Delta x^\underline{n}\\):

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Forward difference of \\(x^\underline{n}\\)
</div>

$$
\begin{align}
\Delta x^\underline{n}&=(x+1)\underbrace{ \textcolor{red}{x(x-1)\dots(x-n)} }_{ =x^\underline{n-1} }-\underbrace{ \textcolor{red}{x(x-1)(x-2)\dots(x-n)} }_{ =x^\underline{n-1} }(x-n+1) \\
&=x^\underline{n-1}[\cancel{ x }+\cancel{ 1 }-(\cancel{ x }-n+\cancel{ 1 })] \\
&=nx^\underline{n-1}
\end{align}
$$

</blockquote>

<blockquote class="box-corollary" markdown="1">
<div class="title" markdown="1">
Sum of \\(x^\underline{n}\\)
</div>
Summing and telescoping the identity \\(x^\underline{n}=\Delta \frac{1}{n+1} x^\underline{n+1}\\),

$$
\sum_{x=a}^{b-1} x^\underline{n}=\sum_{x=a}^{b-1}\Delta \frac{1}{n+1}x^\underline{n+1}=\frac{1}{n+1}b^\underline{n+1}-\frac{1}{n+1}a^\underline{n+1}
$$

</blockquote>

So we've found exactly the result we were looking for: when using the forward difference instead of the derivative, we have \\(\Delta x^\underline{n}=nx^\underline{n-1}\\) compared to \\(Dx^n=nx^{n-1}\\). 

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Characterization of falling power basis \\(x^\underline{n}\\)
</div>
\\(x^\underline{n}\\) is the unique sequence of polynomials \\(p_{n}(x)\\) that satisfies

- \\(\Delta p_{n}(x)=p_{n}(x+1)-p_{n}(x)=np_{n-1}(x)\\)
- \\(p_{0}(x)=1\\)
- \\(n\ge 1 \implies p_{n}(0)=0\\) (only \\(p_{0}\\) has a non-zero constant term)
</blockquote>
 
<details class="box-proof" markdown="1">
<summary markdown="1">
Characterization of falling power basis \\(x^\underline{n}\\)
</summary>
1. Existence

$$
\Delta x^\underline{n} =(x+1)^\underline{n}-x^\underline{n}=(x+1)x^\underline{n-1}-x^\underline{n-1}(x-n+1)=nx^\underline{n-1}
$$


$$
x^\underline{0}=1
$$


$$
n\ge 1 \implies 0^\underline{n}=0(-1)(-2)\dots(-n+1)=0
$$

2. Uniqueness
By the fundamental theorem of discrete calculus, sum \\(\sum_{x=0}^{k-1}\\) both sides of the condition \\(\Delta p_{n}(x)=np_{n-1}(x)\\) to get

$$
p_{n}(k)\cancel{ -p_{n}(0) }=\sum_{x=0}^{k-1}np_{n-1}(x)
$$

for all \\(k\in \mathbb{Z}\\). But choosing \\(\deg p_{n}+1\\) points is enough to uniquely determine \\(p_{n}\\).
</details>

Once we have this, what can we do with it? Let's see the following problem.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Sum of \\(n\\) first squares
</div>

$$
\sum_{k=1}^{n}k^2=1^2+2^2+3^2+\dots+n^2=\;?
$$

</blockquote>

Knowing that the basis of \\(n^\underline{r}\\) is very convenient for differences and sums, we could simply decompose \\(n^2\\) in terms of \\(\{ n^\underline{0},n^\underline{1},n^\underline{2} \}=\{ 1,n,n^\underline{2}\}\\). Write \\(k^2=(k^2-k)+k=k^\underline{2}+k\\), so that

$$
\begin{align}
\sum_{k=1}^{n} k^2&=\sum_{k=0}^nk^2 \\
&=\sum_{k=0}^{n} k^\underline{2}+\sum_{k=0}^{n} k \\
&=\frac{1}{3}(n+1)^\underline{3}+\frac{1}{2}(n+1)^\underline{2} \\
&=\frac{\textcolor{red}{2}}{6}\textcolor{aqua}{(n+1)n}\textcolor{red}{(n-1)}+\frac{\textcolor{red}{3}}{6}\textcolor{aqua}{(n+1)n} \\
&=\boxed{ \frac{1}{6}n(n+1)(2n+1) } \\ 
\end{align}
$$

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Sum of first \\(n\\) cubes
</div>

$$
\sum_{k=1}^{n}k^3=1^3+2^3+3^3+\dots+n^3=\; ?
$$

</blockquote>

Here is a combinatorics problem. Try it out using the following lemma:

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
Difference and summation of binomial coefficient
</div>
The binomial coefficient is given by

$$
\binom{x}{k}=\frac{x^\underline{k}}{k!}=\frac{x^\underline{k}}{k^\underline{k}}
$$

Then:
- Pascal's identity:

$$
\frac{\Delta}{\Delta x} \binom{x}{k}=\binom{x+1}{k}-\binom{x}{k}=\binom{x}{k-1}
$$

- Hockey-stick identity:

$$
\sum_{x=0}^{n-1}\binom{x}{k}=\binom{n}{k+1}
$$

</blockquote>

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Up and Down
</div>
How many 4-digit positive integers \\(d_{1}d_{2}d_{3}d_{4}\\) satisfy the inequalities \\(d_{1}>d_{2}<d_{3}>d_{4}\\)?
</blockquote>

<details class="box-solution" markdown="1">
<summary markdown="1">
Up and Down
</summary>
Note that the conditions make \\(d_{2},d_{4}\in \{ 0,1,2,\dots,9 \}\\) while \\(d_{1},d_{3}\in \{ 1,2,\dots,9 \}\\), so we don't need to worry about leading digit \\(0\\). We can encode our constraints into a big sum, counting 1 for each valid case:


$$
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
$$

</details>

#### Stirling Numbers
We have seen that the basis \\(x^\underline{n}\\) is easier to work with than \\(x^n\\) in the discrete case. So, how can we convert between the two in general? They are given by the following linear combinations:


$$
x^\underline{n}=\sum_{k=0}^{n} s(n,k)x^k
$$


$$
\newcommand{\bracenom}{\genfrac{\lbrace}{\rbrace}{0pt}{}} 
x^n=\sum_{k=0}^{n} \bracenom{n}{k}
$$

where \\(s(n,k)\\) are the *signed Stirling numbers of the first kind* and and \\(\newcommand{\bracenom}{\genfrac{\lbrace}{\rbrace}{0pt}{}} \bracenom{n}{k}\\) are the *unsigned Stirling numbers of the second kind*. They have many interpretations in combinatorics, but we won't go over them. To help remember them better, I will use the following notation suggested by Gemini 3.0 Pro that I found really great: \\(\binom{\text{Target}}{\text{Source}}\\).

$$
x^\underline{n}=\sum_{k=0}^{n}\binom{\underline{n}}{k}x^k
$$


$$
x^n=\sum_{k=0}^{n}\binom{n}{\underline{k}}x^\underline{k}
$$

As you can see, the powers line up with the notation in the coefficients.
We can also use matrix notation for change of basis. Let \\(\mathbf{x^{\underline{n}}}\\) be the column vector of falling factorials and \\(\mathbf{x^n}\\) be the column vector of standard powers:


$$
\mathbf{x^{\underline{n}}} = \begin{bmatrix} x^{\underline{0}} \\ x^{\underline{1}} \\ x^{\underline{2}} \\ \vdots \\ x^{\underline{n}} \end{bmatrix}, \quad
\mathbf{x^n} = \begin{bmatrix} x^0 \\ x^1 \\ x^2 \\ \vdots \\ x^n \end{bmatrix}
$$


We can represent the transformations as matrix multiplications involving lower-triangular matrices. For \\(x^\underline{n} \mapsto x^n\\):


$$
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
$$

For \\(x^n \mapsto x^\underline{n}\\):

$$
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
$$

By nature of change of basis, these matrices are inverses of each other. Therefore,

$$
\sum_{k} \binom{\underline{n}}{k}\binom{k}{\underline{m}}=\delta_{nm}
$$

where \\(n=m\implies \delta_{nm}=1\\) else \\(n\ne m \implies \delta_{nm}=0\\) is the Kronecker delta.

Now that we have defined a notation for our basis conversions, how can we actually compute the coefficients? To solve this problem, we will introduce perhaps the most overpowered tool in all of discrete mathematics, the theory of [generating functions](https://math.mit.edu/~goemans/18310S15/generating-function-notes.pdf).

### Generating functions

<blockquote class="box-info" markdown="1">

A generating function is a clothesline on which we hang up a sequence of numbers for display.
- Herbert S. Wilf, [*generatingfunctionology*](https://www2.math.upenn.edu/~wilf/gfology2.pdf)
</blockquote>

[Video by 3Blue1Brown](https://www.youtube.com/watch?v=bOXCLR3Wric) on generating functions.

The core idea between generating functions is to take a sequence of numbers \\(a_{0},a_{1},a_{2},\dots\\) and attach them to a power of some dummy variable \\(x\\).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Generating function
</div>
Given a sequence \\(a_{0},a_{1},a_{2},\dots\\), its generating function \\(G(x)\\) is given by

$$
G(x)=a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots=\sum_{n \ge 0}a_{n}x^n
$$

</blockquote>

The natural first question to ask is: are you crazy? Doesn't this just make everything even more complicated? On its own, yes, of course it accomplishes nothing. But the motivation behind this approach is that relationships concerning the sequence are often easier to encode using the generating functions, and we can then manipulate these generating functions algebraically. Let's see an example.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Geometric series
</div>
Suppose you have the sequence \\(1,1,1,1,\dots\\). Its generating function is derive like this:

$$
G(x)=1+x+x^2+\dots=\sum_{n \ge 0}x^n
$$


$$
xG(x)=x+x^2+\dots=G(x)-1
$$


$$
\boxed{ G(x)=\frac{1}{1-x} }
$$

Normally, in calculus, you would have to assert that this sum only converges for \\(\vert x\vert <1\\). However, in the basic setting of generating functions in discrete math, we said in the beginning that \\(x\\) is a dummy variable, so it doesn't actually have to hold any meaning. Of course, as is common in mathematics, it is possible to draw connections, here between the worlds of discrete and continuous math, but we'll get to that later.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Reciprocal factorials
</div>
The sequence \\(\frac{1}{0!},\frac{1}{1!}, \frac{1}{2!}, \frac{1}{3!},\dots\\) has generating function

$$
G(x)=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\dots=\sum_{n \ge 0} \frac{x^n}{n!}=e^x
$$

</blockquote>

So, from these examples alone, we see that generating functions can simplify to something simpler. But that doesn't solve our problems: to calculate the generating function, we already knew exactly what our sequence was, but the main challenge with sequences is usually to understand them given only some indirect relationships such as recurrence relations.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Fibonacci sequence
</div>
The famous Fibonacci sequence is defined recursively through the equation

$$
f_{n+2}=f_{n+1}+f_{n}, \quad n\ge 0
$$

with initial values \\(f_{0}=f_{1}=1\\). Now, we will walk through an example of how we can derive a closed-form solution for \\(f_{n}\\) through generating functions.

Let the generating function be \\(F(x)=\sum_{n\ge 0}f_{n}x^n\\). Now, we try to recover this function in the original defining recurrence relation. Multiply both sides by \\(x^{n+2}\\) and sum \\(\sum_{n\ge 0}\\) to get:

$$
\sum_{n\ge 0} f_{n+2}x^{n+2}=\sum_{n\ge 0} f_{n+1}x^{n+2}+\sum_{n\ge 0}f_{n}x^{n+2}
$$

We can re-index and pull out extra factors of \\(x\\) to obtain

$$
\sum_{n\ge 2}f_{n}x^n=x\sum_{n\ge 1}f_{n}x^n+x^2\sum_{n\ge 0}f_{n}x^n
$$

Now we can express these in terms of \\(F(x)=1+x+\sum_{n\ge 2}f_{n}x^n=1+\sum_{n\ge 1}f_{n}x^n\\):

$$
F(x)-x-1=x(F(x)-1)+x^2F(x)
$$

Solving for \\(F(x)\\), we get

$$
F(x)(1-x-x^2)=1
$$


$$
\boxed{ F(x)=\frac{1}{1-x-x^2} }
$$


Here, we started with a convenient indexing so we wouldn't encounter some trouble with bookkeeping. Now we get to the same answer slightly differently to learn how to deal with these troubles, this time starting with this re-indexed recurrence relation

$$
f_{n}=f_{n-1}+f_{n-2},\quad n\ge 2
$$

again with \\(f_{0}=f_{1}=1\\). Notice that we now have to deal more carefully with the edge cases, because our recurrence relation only works for \\(n\ge 2\\). So, we are only allowed to sum over the equation for \\(\sum_{n\ge 2}\\) after multiplying by \\(x^n\\):

$$
\sum_{n \ge 2}f_{n}x^n=\sum_{n\ge 2}f_{n-1}x^n+\sum_{n\ge 2}f_{n-2}x^n
$$

In the end, it turns out to be the same as our original problem. If you tried to sum over \\(n\ge 0\\) when the equation only holds for \\(n\ge 2\\), you would get the wrong answer. So you need to keep track of the problem constraints.
</blockquote>

Now that we have found a formula for \\(F(x)\\), how do we derive a formula for \\(f_{n}\\)? The trick is to remember our first identity for the geometric series:

$$
\frac{1}{1-rx}=1+rx+r^2x^2+r^3x^3+\dots
$$

So we know the closed-form for the \\(n\\)-th term a geometric series. But using partial fraction decomposition, we can precisely turn \\(F(x)=\frac{1}{1-x-x^2}\\) into a linear combination of geometric series. Let's force

$$
F(x)=\frac{1}{1-x-x^2}=\frac{1}{(1-\alpha x)(1-\beta x)}=\frac{a}{1-\alpha x}+\frac{b}{1-\beta x}
$$

where \\(1-x-x^2=(1-\alpha x)(1-\beta x)\\). Comparing coefficients, \\(\alpha\beta=-1,-(\alpha+\beta)=-1\\), so after substituting \\(\beta=-\frac{1}{\alpha}\\) in \\(\alpha+\beta=1\\), we get 

$$
\alpha-\frac{1}{\alpha}=1\iff \alpha^2-\alpha-1=0
$$

which is the famous equation for the golden ratio: \\(\alpha=\phi=\frac{1+\sqrt{5}}{2},\beta=\psi=\frac{1-\sqrt{5}}{2}\\). To solve for the partial fraction coefficients, so after clearing denominators, we apply linear operator of evaluation at \\(x=\frac{1}{\alpha}\\) and \\(x=\frac{1}{\beta}\\), giving \\(a=\frac{1}{1-\frac{\beta}{\alpha}}=\frac{\alpha}{\alpha-\beta}=\frac{\alpha}{\sqrt{5}}\\), \\(b=\frac{1}{1-\frac{\alpha}{\beta}}=\frac{\beta}{\beta-\alpha}=-\frac{\beta}{\sqrt{5}}\\). 

<blockquote class="box-notation" markdown="1">
<div class="title" markdown="1">
Coefficient \\(a_{n}\\) of \\(x^n\\) in \\(G(x)\\)
</div>
For convenience, the notation

$$
[x^n]G(x)=a_{n}
$$

is often used to extract the \\(n\\)-th term in a sequence from the coefficient of \\(x^n\\) in \\(G(x)\\) (note that \\([x^n]\\) is a linear functional).
</blockquote>

Hence, our closed-form solution comes from

$$
\begin{align}
f_{n}&=[x^n]F(x) \\
&=[x^n] \frac{1}{1-x-x^2} \\
&=[x^n] \frac{1}{\sqrt{5}}\left(\frac{\phi}{1-\phi x}-\frac{\psi}{1-\psi x}\right) \\
&=[x^n] \frac{1}{\sqrt{5}} \phi \sum_{n\ge 0}\phi^n x^n-\psi \sum_{n\ge 0}\psi^n x^n \\
&=[x^n] \frac{1}{\sqrt{5}} \sum_{n\ge 0}\left( \phi^{n+1}-\psi^{n+1} \right) \\
&=\frac{\phi^{n+1}-\psi^{n+1}}{\sqrt{5}} \\
&=\boxed{ \frac{1}{\sqrt{5}}\left( \left( \frac{1+\sqrt{5}}{2} \right)^{n+1} - \left( \frac{1-\sqrt{5}}{2} \right)^{n+1}  \right)  }
\end{align}
$$

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Geometric series via recurrence relation
</div>
Using a similar approach on the recursive relation

$$
g_{n+1}=rg_{n},\quad n\ge 0
$$

with \\(g_{0}=a\\), prove that \\(G(x)=\frac{a}{1-r}\\).
</blockquote>

Another reason why generating functions are so useful is that you can create a factor of \\(n\\) in a recursion by taking the derivative. Here is an example where the dummy variable \\(x\\) does mean something, and we care about convergence.

<blockquote class="box-problem" markdown="1">
<div class="title" markdown="1">
Random problem on Twitter
</div>
Taken from @abakcus on X: https://x.com/abakcus/status/1990261853801611758
Find 
$$
\sum_{n=1}^{\infty}\frac{n^3}{2^n}=\; ?
$$

</blockquote>

We use generating functions for this problem. Note that \\(n^3=n^\underline{3}+3n^\underline{2}+n^\underline{1}\\). Write


$$
f(x)=\sum_{n\ge 1}n^3 x^n=\sum_{n\ge 1}(n^\underline{3}+3n^\underline{2}+n^\underline{1})x^n
$$

which converges for \\(\vert x\vert <1\\). 

<blockquote class="box-principle" markdown="1">
<div class="title" markdown="1">
Derivative
</div>
Using our convenient falling power basis for \\(n\\), we decompose the sum by noting that \\(D^k x^n:=\left( \frac{d}{dx} \right)^k x^n=n^\underline{k}x^{n-k}\\).

$$
D^k \sum_{n\ge 0}a_{n}x^n=\sum_{n\ge 0}a_{n}D^kx^n= \sum_{n\ge 0}n^{\underline{k}}a_{n}x^{n-k}
$$

</blockquote>


$$
f(x)=x^3 \sum_{n\ge 1}n^\underline{3}x^{n-3}+3x^2 \sum_{n\ge 1}n^\underline{2}x^{n-2}+x\sum_{n\ge 1}n^\underline{1}x^{n-1}
$$



$$
f(x)=x^3 \sum_{n\ge 1} D^3 x^n+3x^2 \sum_{n\ge 1} D^2x^n+x\sum_{n\ge 1}Dx^n
$$

Interchanging the derivative and summation:

$$
f(x)=x^3 D^3 \sum_{n\ge 1}x^n+3x^2 D^2 \sum_{n\ge 1}x^n+xD \sum_{n\ge 1}x^n
$$

Using the geometric series formula \\(\sum_{n \ge 1}x^n=x\sum_{n\ge 0} x^n=\frac{x}{1-x}\\), we now just need to apply \\(D,D^2,D^3\\) on it then add everything together.

$$
D \frac{x}{1-x}=D\left( \frac{x-1+1}{1-x} \right)=D\left( -1+\frac{1}{1-x} \right)=(1-x)^{-2}
$$


$$
D^2 \frac{x}{1-x}=D (1-x)^{-2}=2(1-x)^{-3}
$$


$$
D^3 \frac{x}{1-x}=D \; 2(1-x)^{-3}=6(1-x)^{-4}
$$

<blockquote class="box-remark" markdown="1">
<div class="title" markdown="1">
Feynman's trick: differentiation under the integral sign
</div>
In the continuous case, the act of creating a generating function and manipulating it with derivatives is called *differentiation under the integral sign*, *Feynman's integration trick*.
We first turn our function \\(f(x)\\) into a generating function with parameter \\(\alpha\\), and then integrate:

$$
I(\alpha)=\int f(x,\alpha)\, dx
$$

Under suitable conditions for convergence,

$$
I'(\alpha)=\frac{ \partial }{ \partial \alpha } \int f(x,\alpha)\,dx=\int \frac{ \partial }{ \partial \alpha } f(x,\alpha)\,dx
$$

</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Spot the pattern!
</div>
Can you find a closed-form expression for \\(D^n \frac{x}{1-x}\\)?
</blockquote>

Now, we can simply plug everything in.

$$
f(x):=\sum_{n\ge 1}n^3 x^n=x^3\cdot 6(1-x)^{-4}+3x^2 \cdot 2(1-x)^{-3}+x(1-x)^{-2}
$$

Evaluating at \\(x=\frac{1}{2}\\) for the original sum \\(\sum_{n=1}^{\infty}\frac{n^3}{2^n}\\), note that \\(\left( 1-\frac{1}{2} \right)^{-k}=2^k\\), so \\(\left( \frac{1}{2} \right)^k \cdot (1-\frac{1}{2})^{-k-1}=2^{-k}\cdot 2^{k+1}=2\\).

$$
f\left( \frac{1}{2} \right)=6\cdot 2+3\cdot 2\cdot 2+2=\boxed{ 26 }
$$


<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
\\((xD)^n\\)
</div>
The operator of differentiation then multiplication by a polynomial \\(E=p(x)\cdot D\\) is sometimes called the Cauchy-Euler operator, defined for some polynomial \\(p(x)\\) and the differentiation operator \\(D:=\frac{d}{dx}\\). Applied to a function \\(f(x)\\), it gives \\(Ef(x)=p(x)f'(x)\\).

Notably, the simplest example \\(E=xD\\) has eigenfunctions \\(x^n\\), since \\(Ex^n=xDx^n=nx^n\\). This matches the form in our earlier problem! Based on our steps, find a closed-form expression for \\((xD)^n\\) as a linear combination of the terms \\(x^k D^k\\).
</blockquote>

Now that we have the powerful tool of generating functions in our pockets, let's go back to the question of figuring out how to compute the Stirling numbers.

$$
x^\underline{n}=\sum_{k=0}^{n}\binom{\underline{n}}{k}x^k
$$


$$
x^n=\sum_{k=0}^{n}\binom{n}{\underline{k}}x^\underline{k}
$$

Let's start by trying a recursive approach. \\(x^\underline{n+1}=x^\underline{n} (x-n)\\), so

$$
\sum_{k=0}^{n+1}\binom{\underline{n+1}}{k}x^k=(x-n)\sum_{j=0}^{n}\binom{\underline{n}}{j}x^j=\sum_{i=0}^{n}\binom{\underline{n}}{i}x^{i+1}-n\sum_{j=0}^{n}\binom{\underline{n}}{j}x^{j}
$$

Take \\([x^k]\\) on both sides for \\(1\le k \le n\\):

$$
\binom{\underline{n+1}}{k}=\binom{\underline{n}}{k-1}-n\binom{\underline{n}}{k}
$$

Great, now we construct the generating function. Note that \\(\binom{\underline{n}}{k}=0\\) for \\(n<k\\), since the polynomials \\(x^\underline{n},x^{n}\\) have the same degrees, high powers can't contribute, so let the generating function

$$
A_{k}(y)=\sum_{n\ge 0}\binom{\underline{n}}{k}y^n=\sum_{n\ge k} \binom{\underline{n}}{k}y^n
$$

Now we multiply our recurrence relation by \\(y^{n+1}\\) and sum over \\(n\ge k\\):

$$
\sum_{n\ge k}\binom{\underline{n+1}}{k}y^{n+1}=y\sum_{n\ge k}\binom{\underline{n}}{k-1}y^{n}-ny\sum_{n\ge k}\binom{\underline{n}}{k}y^{n}
$$


Writing in terms of \\(A_{k}(y)\\):

$$
LHS=\sum_{n\ge k+1}\binom{\underline{n}}{k}y^n=A_{k}(y)-\underbrace{ \binom{\underline{k}}{k} }_{ =1 }y^{k}=A_{k}(y)-y^k
$$

where we know polynomials \\(y^\underline{k},y^k\\) have the same leading coefficient \\(1\\), so \\(\binom{\underline{k}}{k}=1\\).
Let the terms in the RHS be \\(RHS_{1}\\) and \\(RHS_{2}\\) in order:

$$
RHS_{1}=y\left(A_{k-1}(y)-\underbrace{ \binom{\underline{k-1}}{k-1} }_{ =1 }y^{k-1}\right)=yA_{k-1}(y)-y^k
$$

The second term has a factor of \\(n\\), so we need a derivative.
Note that 

$$
A'_{k}(y)=\sum_{n\ge k}\binom{\underline{n}}{k}ny^{n-1}=\frac{n}{y}\sum_{n\ge k}\binom{\underline{n}}{k}y^n
$$



$$
RHS_{2}=-y^2 A'_{k}(y)
$$

Altogether,

$$
A_{k}(y)\cancel{ -y^k }=yA_{k-1}(y)\cancel{ -y^k }-y^2 A'_{k}(y)
$$


$$
A_{k}(y)+y^2 A'_{k}(y)=yA_{k-1}
$$

This is a first-order ODE, and you can solve it using integrating factors, but it results in some very ugly algebra and a lot of integration. It turns out, there's a better approach to this problem.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Exponential Generating Function (EGF)
</div>
So far, we have only been working with something called the *ordinary generating function (OGF)*. For a sequence \\(a_{0},a_{1},a_{2},\dots\\)

$$
OGF(x)=\sum_{n\ge 0}a_{n}x^{n}
$$

But for problems involving falling powers (which includes factorials, binomial coefficients, e.g. labeled set structures and permutations), the *exponential generating function (EGF)* often simplifies better:

$$
EGF(x)=\sum_{n\ge 0}\frac{a_{n}}{n!}x^n
$$

If \\(a_{n}=1\\) for all \\(n\\), then we recover the Taylor series of \\(e^x\\):

$$
\sum_{n\ge 0}\frac{x^n}{n!}=e^x
$$

as opposed to the geometric series \\(\sum_{n\ge 0}x^n=\frac{1}{1-x}\\) for the OGFs.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Derivative of EGF
</div>

$$
A(x)=\sum_{n\ge 0}\frac{a_{n}}{n!}x^n=a_{0}+a_{1}x+\frac{a_{2}}{2!}x^2+\frac{a_{3}}{3!}x^3+\cdots
$$


$$
A'(x)=a_{1}+a_{2}x+\frac{a_{3}}{2!}x^2+\cdots=\sum_{n\ge 1}\frac{a_{n}}{(n-1)!}x^{n-1}=\sum_{n\ge 0}\frac{a_{n+1}}{n!}x^n
$$

So the factor of \\(n\\) cancels, and the index just shifts by \\(1\\). This simplifies algebra for ODEs with recursions like \\(a_{n}=na_{n-1}\\).
</blockquote>

Let's try it out on our problem of computing Stirling coefficients.

 First, we need the binomial theorem.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Binomial theorem
</div>
Define the binomial coefficient

$$
\binom{n}{k}=\frac{n^\underline{k}}{k^\underline{k}}=\frac{n^\underline{k}}{k!}
$$

Using the generating function

$$
G(x,y)=\sum_{n\ge 0}\sum_{k\ge 0}\binom{n}{k}x^n y^k
$$

Prove by taking \\([x^n]\\) that

$$
(y+1)^n=\sum_{k=0}^{n}\binom{n}{k}y^k
$$

Hint: note that \\(\binom{n}{0}=1,n\ge 0\\), \\(\binom{0}{k}=0, k\ge 1\\), 

$$
\binom{n+1}{k+1}=\binom{n}{k+1}+\binom{n}{k},\quad n,k\ge 0
$$

Then, replacing \\(y=\frac{a}{b}\\), prove the full binomial theorem:

$$
(a+b)^n=\sum_{k=0}^{n}\binom{n}{k}a^k b^{n-k}
$$

If you more practice, you can also take \\([y^k]\\) to prove the Stars and Bars identity:

$$
[x^n]\left( \frac{1}{1-x} \right)^{k}=\binom{n+k-1}{k-1}
$$

which by remembering \\(\frac{1}{1-x}=1+x+x^2+\cdots\\) is the number of nonnegative integer solutions to

$$
x_{1}+x_{2}+\dots+x_{k}=n
$$

</blockquote>

Now, to recover an exponential generating function, we apply the clever identity 
$$
a^b=e^{\ln(a^b)}=e^{b \ln a}
$$
 with \\(a=(x+1),b=n\\). Define

$$
F(x,n)=(x+1)^n=\sum_{k=0}^n \binom{n}{k}x^k=\sum_{k=0}^n \frac{n^\underline{k}}{k!}x^k
$$

From our definition \\(n^\underline{k}=\sum_{j=0}^{k}\binom{\underline{k}}{j}n^j\\),

$$
F(x,n)=\sum_{k=0}^{n}\frac{x^k}{k!}\sum_{j=0}^{k} \binom{\underline{k}}{j}n^j
$$

then we have

$$
F(x,n)=e^{n\ln(x+1)}=\sum_{j\ge 0}\frac{(n \ln(x+1))^j}{j!}=\sum_{j\ge 0} \frac{(\ln(x+1))^j}{j!} n^j
$$

Comparing coefficients with \\([n^j]\\), we get that

$$
[n^j]F(x,n)=\sum_{k=0}^{n} \binom{\underline{k}}{j}\frac{x^k}{k!}=\frac{(\ln(x+1))^j}{j!}
$$

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Stirling numbers of the second kind \\(\binom{n}{\underline{k}}\\)
</div>
Find a similar relationship for \\(\binom{n}{\underline{k}}\\) using the following trick:

$$
e^{zt}=(1+(e^t-1))^z
$$

You should arrive at

$$
\sum_{n\ge k} \binom{n}{\underline{k}}\frac{t^n}{n!}=\frac{(e^{t}-1)^k}{k!}
$$

</blockquote>

For solving discrete problems (sums), we usually convert from the power basis \\(x^k\\) to the falling power basis \\(x^\underline{k}\\) via the Stirling numbers of the second kind, so we'll derive an expression for them. We expand the RHS using binomial theorem and Taylor series of \\(e^x\\).


$$
\begin{align}
\sum_{n\ge k}\binom{n}{\underline{k}}\frac{t^n}{n!}
&=\frac{1}{k!} \sum_{j=0}^{k} \binom{k}{j}e^{jt}(-1)^{k-j} \\
&=\frac{1}{k!} \sum_{j=0}^{k} (-1)^{k-j}\binom{k}{j} \sum_{n \ge 0}\frac{(jt)^n}{n!}
\end{align}
$$

Comparing coefficients \\(\left[ \frac{t^n}{n!} \right]\\) of the EGF:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
Closed-form solution for Stirling numbers of the second kind \\(\binom{n}{\underline{k}}\\)
</div>

$$
\boxed{ \binom{n}{\underline{k}}=\frac{1}{k!}\sum_{j=0}^k (-1)^{k-j}\binom{k}{j}j^n }
$$

</blockquote>

So we can convert from \\(x^k\\) to \\(x^\underline{k}\\) basis fairly OK, but as for the Stirling numbers of the first kind \\(\binom{\underline{n}}{k}\\) for the inverse matrix, from [Wikipedia](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind#Explicit_formula):
>No one-sum formula for Stirling numbers of the first kind is currently known. A two-sum formula can be obtained using one of the [symmetric formulae for Stirling numbers](https://en.wikipedia.org/wiki/Stirling_number#Symmetric_formulae "Stirling number") in conjunction with the explicit formula for [Stirling numbers of the second kind](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind "Stirling numbers of the second kind").

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Taylor series of \\(\ln(1+x)\\)
</div>
Find the Taylor series of \\(\ln(1+x)\\), given

$$
\ln(1+x)=\int \frac{1}{1+x} \,dx
$$

and \\(\frac{1}{1+x}=\frac{1}{1-(-x)}=\sum_{n \ge 0}(-x)^n\\).
</blockquote>

### Natural Discrete Base 2
In infinitesimal calculus, we define the *natural base* \\(e\\) by

$$
\frac{d}{dx}e^x=e^x
$$

What is the natural basis \\(a\\) in the continuous world? It's a lot easier to solve:

$$
\Delta a^n=a^n
$$


$$
a^{n+1}-a^n=a^n\iff a^{n+1}=2a^n
$$

For the non-trivial solution \\(a\ne 0\\), we divide both sides by \\(a^n\\) to find \\(a=2\\). So actually, \\(2\\) is the natural basis for the discrete world. We expand on this in the next section.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
\\(\sum_{n=0}^{k} n 2^n\\)
</div>
You could use generating functions, of course. Here, just like in \\(\int_{a}^{b} xe^x \,dx\\), we apply summation by parts.

$$
\sum_{x=a}^{b-1} f(x) \Delta g(x)=\left.f(x)g(x)\right\vert_{a}^{b}-\sum_{x=a}^{b-1} \Delta f(x)g(x+1)
$$

Take \\(f(n)=n=n^\underline{1},g(n)=2^n\\), so \\(\Delta f(n)=1, \Delta g(n)=2^n\\).

$$
\begin{align}
\sum_{n=0}^{k} n 2^n&=\left.\left(n\cdot 2^n\right)\right\vert_{0}^{k+1}-\sum_{n=0}^{k}1 \cdot 2^{n+1} \\
&=(k+1)2^{k+1}-(2^{k+2}-2) \\
&=\boxed{ (k-1)2^{k+1}+2 }
\end{align}
$$

</blockquote>

### Newton series
Given the defining equation

$$
e^x=\frac{d}{dx}e^x
$$

If we assume that \\(e^x\\) is analytic, i.e. we can write it as a power series, then we are generating a sequence defined by \\(Dx^n=nx^{n-1},n\ne 0\\):

$$
\begin{align}
a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots&=\frac{d}{dx}(a_{0}+a_{1}x+a_{2}x^2+a_{3}x^3+\dots) \\
&=a_{1}+2a_{2}x+3a_{3}x^2+\dots
\end{align}
$$

Comparing \\(x^n\\), we get the recurrence \\(a_{n+1}=na_{n}\\), \\(a_{0}=e^{0}=1\\), or \\(a_{n}=n!\\) giving the famous Taylor series

$$
e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
$$

The Taylor series is an expansion in the basis \\(x^k\\) and \\(D=\frac{d}{dx}\\): 

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Taylor series
</div>

$$
f(x)=\sum_{n=0}^\infty \frac{(D^nf)(a)}{n!}(x-a)^n
$$

</blockquote>

But lesser know is the notion of Newton series, which is simply a series in the basis of \\(x^\underline{k}\\) and \\(\Delta\\). It is also called the Gregory-Newton forward difference interpolation formula because it interpolates at integer points based on \\(\Delta\\).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Newton series
</div>

$$
f(x)=\sum_{n=0}^{\infty} \frac{(\Delta^n f)(a)}{n!}(x-a)^\underline{n}=\sum_{n=0}^{\infty} \binom{x-a}{k}(\Delta^n f)(a)
$$

</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Newton series of \\(2^x\\)
</div>
Based on the change of basis through the Stirling numbers, the Newton series of \\(2^x\\) in \\(x^\underline{k}\\) looks exactly like \\(e^x\\) in \\(x^k\\) since just as \\(D^n e^x=e^x\\) with \\(e^0=1\\), \\(\Delta^n 2^x=2^x\\), and \\(2^0=1\\), so \\(2^x\\) encodes the sequence \\(1,1,1,1,\dots\\) in an EGF in \\(x^\underline{k}\\):

$$
2^x=\sum_{n=0}^\infty \frac{1}{n!}x^\underline{n}=1+x^\underline{1}+\frac{x^\underline{2}}{2!}+\frac{x^\underline{3}}{3!}+\dots
$$

</blockquote>

We can also derive it in a funny way using operator calculus. Let \\(E=e^D\\) be the shift operator \\(Ef(x)=f(x+1)\\), so that \\(E=I+\Delta\\) for the identity operator \\(If(x)=f(x)\\). Then Newton series for integers comes from the (extended) binomial theorem:

$$
\begin{align}
f(x)&=E^x f(0) \\
&=(I+\Delta)^x f(0) \\
&=\sum_{k\ge 0} \binom{x}{k}\Delta^k f(0)
\end{align}
$$

For an example application, you can use this formula to solve for sequences that you know have a polynomial solution of degree \\(n\\) with \\(n+1\\) points.

Here is a quick worked example and an exercise based on the logic that \\(f(x) = \sum_{k=0}^\infty \frac{(\Delta^k f)(0)}{k!} x^{\underline{k}}\\).

While \\(2^x\\) is an infinite series (because the differences never go to zero), this method is most powerful for finding **polynomial closed forms** for integer sequences, where the differences eventually become zero.

### Example Application: Find the closed form

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Problem:** Find the lowest degree polynomial \\(f(x)\\) that generates the sequence: \\(1, 6, 15, 28, \dots\\) for \\(x=0, 1, 2, 3 \dots\\)
</div>
**Step 1: Build the Difference Table**
Just as a Taylor series requires derivatives at \\(x=0\\), the Newton series requires the forward differences at \\(x=0\\) (the first number in each row).


$$
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
$$


**Step 2: Apply the Formula**
We take the "diagonal" values \\((\mathbf{1}, \mathbf{5}, \mathbf{4}, \mathbf{0})\\) as our coefficients for \\(\frac{x^\underline{k}}{k!}\\).


$$
f(x) = 1 \cdot \frac{x^\underline{0}}{0!} + 5 \cdot \frac{x^\underline{1}}{1!} + 4 \cdot \frac{x^\underline{2}}{2!}
$$


**Step 3: Convert to Standard Polynomials**
Recall that \\(x^\underline{1} = x\\) and \\(x^\underline{2} = x(x-1)\\).


$$
\begin{aligned}
f(x) &= 1(1) + 5(x) + \frac{4}{2} x(x-1) \\
&= 1 + 5x + 2(x^2 - x) \\
&= 1 + 5x + 2x^2 - 2x \\
&= \mathbf{2x^2 + 3x + 1}
\end{aligned}
$$


*(Check: If \\(x=3\\), \\(2(9) + 9 + 1 = 28\\). It works.)*
</blockquote>

### Example: Formula for the Sum of Cubes

This is a classic application of the method. Because the sum of polynomials of degree \\(d\\) results in a polynomial of degree \\(d+1\\), we know ahead of time that the sum of cubes (degree 3) will result in a degree 4 polynomial. This guarantees the difference table will eventually hit zero.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Problem:** Find the formula for \\(S(x) = \sum_{i=1}^x i^3 = 1^3 + 2^3 + \dots + x^3\\).
</div>
**Step 1: Generate the Sequence and Difference Table**
We must include \\(x=0\\) (the empty sum, which is 0) to anchor the Newton series properly.

*   \\(x=0, S=0\\)
*   \\(x=1, S=1\\)
*   \\(x=2, S=1+8=9\\)
*   \\(x=3, S=9+27=36\\)
*   \\(x=4, S=36+64=100\\)


$$
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
$$


**Step 2: Apply the Newton Series Formula**
The diagonal values are \\(\mathbf{0, 1, 7, 12, 6}\\).

$$
S(x) = 0 \frac{x^\underline{0}}{0!} + 1 \frac{x^\underline{1}}{1!} + 7 \frac{x^\underline{2}}{2!} + 12 \frac{x^\underline{3}}{3!} + 6 \frac{x^\underline{4}}{4!}
$$


**Step 3: Simplify**
Evaluate the factorials: \\(1! = 1, 2!=2, 3!=6, 4!=24\\).


$$
S(x) = x + \frac{7}{2}x(x-1) + \frac{12}{6}x(x-1)(x-2) + \frac{6}{24}x(x-1)(x-2)(x-3)
$$


This is a valid formula, but to show it matches the standard textbook formula, we simplify algebraically.

1.  **Linear/Quadratic terms:** \\(x + 3.5(x^2-x) = 3.5x^2 - 2.5x\\)
2.  **Cubic term:** \\(2(x^3 - 3x^2 + 2x) = 2x^3 - 6x^2 + 4x\\)
3.  **Quartic term:** \\(\frac{1}{4}(x^4 - 6x^3 + 11x^2 - 6x)\\)

Summing these up:
*   \\(x^4\\): \\(\frac{1}{4}x^4\\)
*   \\(x^3\\): \\(-\frac{6}{4}x^3 + 2x^3 = \frac{1}{2}x^3\\)
*   \\(x^2\\): \\(\frac{11}{4}x^2 - 6x^2 + 3.5x^2 = \frac{1}{4}x^2\\)
*   \\(x^1\\): \\(-\frac{6}{4}x + 4x - 2.5x = 0\\)


$$
S(x) = \frac{1}{4}x^4 + \frac{1}{2}x^3 + \frac{1}{4}x^2 = \frac{x^2}{4}(x^2 + 2x + 1)
$$



$$
\mathbf{S(x) = \left[ \frac{x(x+1)}{2} \right]^2}
$$


This confirms the famous identity that \\(\sum i^3 = (\sum i)^2\\).
</blockquote>

<blockquote class="box-exercise" markdown="1">

**Problem:**
Using the Newton Series method, find the polynomial \\(f(x)\\) for the following sequence (where \\(x=0, 1, 2\dots\\)):

$$
3, 5, 9, 15, 23, \dots
$$


1. Construct the difference table to find \\(\Delta^n f(0)\\).
2. Write out the Newton series using falling factorials (\\(x^{\underline{k}}\\)).
3. Simplify it into a standard polynomial form (\\(ax^2+bx+c\\)).
</blockquote>

<details class="box-tip" markdown="1">
<summary markdown="1">
**Click for Solution**
</summary>
**1. The Difference Table:**
*   Sequence (\\(f\\)): \\(3, 5, 9, 15, 23\\)
*   \\(\Delta^1\\): \\(2, 4, 6, 8\\)
*   \\(\Delta^2\\): \\(2, 2, 2\\)  (Constant!)
*   \\(\Delta^3\\): \\(0\\)

The coefficients at \\(x=0\\) are **3, 2, 2**.

**2. The Newton Series:**

$$
f(x) = 3 \frac{x^\underline{0}}{0!} + 2 \frac{x^\underline{1}}{1!} + 2 \frac{x^\underline{2}}{2!}
$$


**3. Simplification:**

$$
\begin{aligned}
f(x) &= 3(1) + 2(x) + \frac{2}{2}x(x-1) \\
&= 3 + 2x + x^2 - x \\
&= \mathbf{x^2 + x + 3}
\end{aligned}
$$

</details>

### Umbral Calculus

#### Warmup

A *convolution* \\(f \ast g\\) is a process where "multiplying" objects adds their "degrees". A good example is polynomials.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Polynomial multiplication
</div>
If we multiply two quadratics \\(A(x)\cdot B(x)=C(x)\\)

$$
\begin{gather}
(a_{0}+a_{1}x+a_{2}x^2)(b_{0}+b_{1}x+b_{2}x^2) \\
=a_{0}b_{0}+(a_{0}b_{1}+b_{0}a_{1})x+(a_{0}b_{2}+a_{1}b_{1}+a_{2}b_{0})x^2+(a_{1}b_{2}+a_{2}b_{1})x^3+a_{2}b_{2}x^4
\end{gather}
$$


As you can see, the subscripts add together to the degree encoded by the monomial \\(x^d\\), e.g. \\(a_{0}b_{1}\\) gets assigned to \\(x^{0+1}=x\\). So \\(c_{n}=\sum_{k}a_{k}b_{n-k}\\) are the coefficients of \\(C(x)\\).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
Dice rolls
</div>
Another example is the addition of random variables: let \\(X,Y\\) be the outcomes of a standard 6-sided die. Then \\(X+Y\\) is given by the convolution of the \\(X\\) and \\(Y\\):

$$
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
$$

The number outcomes for \\(X+Y\\) amounts to "counting along the off-diagonal". For example, finding the number of ways for \\(X+Y=6\\), you get \\(5\\):

$$
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
$$

</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Discrete convolution
</div>
Define OGFs for sequences \\((a_{i})_{i},(b_{j})_{j}\\).

$$
A(x)=\sum_{i=i_{0}}^{i_{1}}a_{i}x^i=a_{i_{0}}x^{i_{0}}+a_{i_{0}+1}x^{i_{0}+1}+\dots+a_{i_{1}}x^{i_{1}}
$$


$$
B(x)=\sum_{j=j_{0}}^{j_{1}}b_{j}x^j=b_{j_{0}}x^{j_{0}}+b_{j_{0}+1}x^{j_{0}+1}+\dots+b_{j_{1}}x^{j_{1}}
$$

Define

$$
C(x)=A(x)B(x)
$$

Then we have

$$
\begin{align}
C(x)&=\left( \sum_{i=i_{0}}^{i_{1}}a_{i}x^i \right)\left( \sum_{j=j_{0}}^{j_{1}} b_{j}x^j \right) \\
&=(a_{i_{0}}x^{i_{0}}+\dots+a_{i_{1}}x^{i_{1}})(b_{j_{0}}x^{j_{0}}+\dots+b_{j_{1}}x^{j_{1}}) \\
&=a_{i_{0}}b_{j_{0}}x^{i_{0}+j_{0}}+(a_{i_{0}}b_{j_{0}+1}+a_{i_{0}+1}b_{j_{0}})x^{i_{0}+j_{0}+1}+\dots+a_{i_{1}}b_{j_{1}}x^{i_{1}+j_{1}} \\
&=\sum_{n=i_{0}+j_{0}}^{i_{1}+j_{1}}x^n \left( \sum_{k=\max(i_{0},j_{0})}^{\min(i_{1},j_{1})} a_{k}b_{n-k}\right) 
\end{align}
$$

We define the *discrete convolution* of \\((a_{i})_{i},(b_{j})_{j}\\)  as a sequence \\((c_{n})_{n}=(a_{i})_{i} \ast (b_{j})_{j}\\), with

$$
c_{n}=[x^n]C(x)=\sum_{k=\max(i_{0},j_{0})}^{\min(i_{1},j_{1})} a_{k}b_{n-k}
$$

for \\(i_{0}+j_{0}\le n\le i_{1}+j_{1}\\).
</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Vandermonde's convolution identity
</div>
Comparing coefficients of \\((x+1)^{m+n}=(x+1)^m (x+1)^n\\), prove

$$
\binom{m+n}{m}=\sum_{j=0}^{k}\binom{n}{j}\binom{m}{k-j}
$$

</blockquote>

<blockquote class="box-solution" markdown="1">
<div class="title" markdown="1">
Vandermonde's convolution identity
</div>
Here is an alternative solution using operator calculus again. First, we prove that the binomial theorem holds with falling powers:

$$
(x+y)^\underline{n}=\sum_{k=0}^{n}x^\underline{k} y^{\underline{n-k}}
$$

Expand into the Newton series by \\((x+y)^\underline{n}=E^yx^\underline{n}\\):

$$
(x+y)^\underline{n}=(\Delta+I)^y x^\underline{n}=\sum_{k\ge 0}\binom{y}{k}\Delta^k x^\underline{n}
$$

Compute \\(\Delta^k x^\underline{n}=n^\underline{k}x^\underline{n-k}\\), so

$$
(x+y)^\underline{n}=\sum_{k\ge 0} \frac{y^\underline{k}}{k!} n^\underline{k} x^\underline{n-k}=\sum_{k=0}^{n}\binom{n}{k}y^\underline{k}x^\underline{n-k}
$$

Now, dividing both sides by \\(n! =n^\underline{k}(n-k)!\\) proves the identity.
</blockquote>

It's very interesting that the binomial theorem also holds in the discrete world with falling powers instead of regular powers. But this is no coincidence, it turns out to hold for many more identities (but not all!). As you might guess from the recurring theme, we will view this with the perspective of linearity. With this warmup in mind, let's get introduced to Umbral calculus. 

Throughout this guide, we have seen a recurring theme: discrete math behaves eerily like continuous math if you simply swap the components correctly.

| Continuous          | Discrete                              |
| :------------------ | :------------------------------------ |
| Power \\(x^n\\)     | Falling Power \\(x^{\underline{n}}\\) |
| Derivative \\(D\\)  | Difference \\(\Delta\\)               |
| Integral \\(\int\\) | Sum \\(\sum\\)                        |
| Base \\(e\\)        | Base \\(2\\)                          |

Historically, 19th-century mathematicians (like Sylvester and Cayley) noticed they could prove difficult identities regarding number sequences by pretending the indices were exponents. They called this **Umbral Calculus** (from the Latin *umbra*, meaning "shadow"), because these techniques were "shadowy" like dark arts.

For a long time, this was considered "black magic" that lacked rigor. However, in the 1970s, Gian-Carlo Rota formalized this theory using **linear functionals**, turning it into a rigorous and overpowered algebraic tool.

#### The "Umbral Trick": Linear Functionals

To understand why the discrete world mirrors the continuous one, we must define the machinery that allows us to treat indices like exponents.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Linear Functional
</div>
Let \\(P\\) be the vector space of all polynomials in the variable \\(z\\). A **linear functional** is a map \\(L: P \to \mathbb{K}\\) (where \\(\mathbb{K}\\) is a field like \\(\mathbb{R}\\) or \\(\mathbb{C}\\)) such that for any polynomials \\(p(z), q(z)\\) and constants \\(c_1, c_2\\):

$$
L(c_1 \cdot p(z) + c_2 \cdot q(z)) = c_1 \cdot L(p(z)) + c_2 \cdot L(q(z))
$$

Essentially, \\(L\\) is a machine that "eats" a polynomial and spits out a scalar.
</blockquote>

The most common linear functional is simply **evaluation**. For example, if \\(L\\) is "evaluate at \\(z=0\\)", then \\(L(z+3) = 3\\). (Verify that this is really is linear as an exercise). However, the power of Umbral calculus comes from defining functionals based on sequences.

**The Isomorphism:**
Suppose we have a sequence of numbers \\(a_0, a_1, a_2, \dots\\). We can define a linear functional \\(L\\) by specifying its action on the basis vectors \\(z^n\\):

$$
L(z^n) = a_n
$$

Because \\(L\\) is linear, knowing how it acts on \\(z^n\\) tells us how it acts on *any* polynomial. This formalizes the "black magic" notation where we write \\(a^n\\) during algebraic manipulation and then "lower the index" to \\(a_n\\) at the very end.

#### Problem 1: Vandermonde's Convolution Identity

A classic problem in combinatorics is proving **Vandermonde's Identity**:

$$
\sum_{k=0}^n \binom{r}{k} \binom{s}{n-k} = \binom{r+s}{n}
$$

Standard proofs involve combinatorial counting arguments (choosing a committee from two groups of people). However, in Umbral calculus, this is simply a consequence of the fact that our falling factorial basis behaves exactly like a binomial expansion.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
The Discrete Binomial Theorem
</div>
Just as \\((a+b)^n = \sum \binom{n}{k} a^k b^{n-k}\\), the falling powers satisfy:

$$
(x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
$$

</blockquote>

**Proof via Linear Functionals:**
We will prove this by mapping the standard binomial theorem to the discrete world.
Start with two variables \\(a, b\\) and the standard identity:

$$
(a+b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}
$$

We define two linear functionals, \\(L_x\\) and \\(L_y\\), that act as "basis changers" from standard powers to falling powers:
*   \\(L_x(a^k) = x^{\underline{k}}\\)
*   \\(L_y(b^k) = y^{\underline{k}}\\)
*   Let \\(L = L_x L_y\\) be the joint functional acting on products.

**1. Apply \\(L\\) to the Right Hand Side:**

$$
L\left( \sum_{k=0}^n \binom{n}{k} a^k b^{n-k} \right) = \sum_{k=0}^n \binom{n}{k} L_x(a^k) L_y(b^{n-k}) = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
$$

This matches the desired expansion.

**2. Apply \\(L\\) to the Left Hand Side:**
To evaluate \\(L((a+b)^n)\\), we look at the **Exponential Generating Functions (EGF)**.
The EGF for the standard basis is \\(e^{at}\\).
The EGF for the falling factorial basis is \\((1+t)^x\\), because \\(\sum \frac{x^{\underline{n}}}{n!}t^n = (1+t)^x\\).
Therefore, the functional \\(L_x\\) is defined by the mapping \\(L_x(e^{at}) = (1+t)^x\\).

Applying this to the joint exponential:

$$
L(e^{(a+b)t}) = L(e^{at} e^{bt}) = L_x(e^{at})L_y(e^{bt}) = (1+t)^x (1+t)^y = (1+t)^{x+y}
$$

We know that \\((1+t)^{x+y}\\) is the generating function for the sequence \\((x+y)^{\underline{n}}\\). Thus, by comparing coefficients of \\(t^n/n!\\), we conclude:

$$
L((a+b)^n) = (x+y)^{\underline{n}}
$$


Equating LHS and RHS proves the Discrete Binomial Theorem.

**Deriving Vandermonde's Identity:**
Now that we have \\((x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}\\), we simply divide both sides by \\(n!\\):

$$
\frac{(x+y)^{\underline{n}}}{n!} = \sum_{k=0}^n \frac{1}{n!} \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}
$$

Using the identity \\(\binom{n}{k} = \frac{n!}{k!(n-k)!}\\) and rearranging terms:

$$
\frac{(x+y)^{\underline{n}}}{n!} = \sum_{k=0}^n \frac{x^{\underline{k}}}{k!} \frac{y^{\underline{n-k}}}{(n-k)!}
$$

Substituting \\(\frac{z^{\underline{m}}}{m!} = \binom{z}{m}\\), we recover the identity:

$$
\boxed{ \binom{x+y}{n} = \sum_{k=0}^n \binom{x}{k} \binom{y}{n-k} }
$$


#### Problem 2: Bernoulli Numbers and Sums of Powers

The "Killer App" of Umbral Calculus is solving for sums of powers, \\(\sum_{k=0}^{m-1} k^p\\).
While we know \\(\sum k = \frac{m(m-1)}{2}\\), formulas for \\(\sum k^{10}\\) are tedious to find manually. Umbral calculus gives us **Faulhaber's Formula** for the sum of first natural number powers almost for free.

First, we define the **Bernoulli Numbers** \\(B_n\\) using an implicit Umbral recurrence. We pretend \\(B\\) is a variable, write the relation, and then "lower the shadow" (convert powers \\(B^k\\) to indices \\(B_k\\) by a linear functional \\(L: B^k \mapsto B_{k}\\)).

**Definition:** \\(B_0 = 1\\), and for \\(n > 1\\):

$$
L ((B+1)^n - B^n) = 0
$$

*(Note: This expands to \\(L\left( \sum_{k=0}^{n} \binom{n}{k} B^k - B^n \right) = 0\\), or \\(\sum_{k=0}^{n-1} \binom{n}{k} B_k = 0\\)).*

Now, consider the problem of integration. In standard calculus, \\(\int x^n dx = \frac{x^{n+1}}{n+1}\\).
In discrete calculus, we are looking for the sum, which is the anti-difference \\(\Delta^{-1}\\). It turns out the Bernoulli numbers allow us to integrate standard powers \\(x^n\\) directly.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
Faulhaber’s Formula (Umbral Form)
</div>

$$
\sum_{k=0}^{m-1} k^n = \frac{1}{n+1} \left( (m+B)^{n+1} - B^{n+1} \right)
$$

</blockquote>

<details class="box-example" markdown="1">
<summary markdown="1">
Sum of First \\(m\\) Squares (\\(n=2\\))
</summary>
We want to compute \\(\sum_{k=0}^{m-1} k^2\\). Using the formula with \\(n=2\\):

$$
\Sigma = \frac{1}{3} \left( (m+B)^3 - B^3 \right)
$$

Expand the binomial \\((m+B)^3 = m^3 + 3m^2B + 3mB^2 + B^3\\):

$$
\Sigma = \frac{1}{3} \left( (m^3 + 3m^2B^1 + 3mB^2 + B^3) - B^3 \right)
$$


$$
\Sigma = \frac{1}{3} m^3 + m^2 B^1 + m B^2
$$

Now we "lower the shadow." We interpret \\(B^1\\) as \\(B_1\\) and \\(B^2\\) as \\(B_2\\).
From the definition of Bernoulli numbers:
*   \\(B_0 = 1\\)
*   \\((B+1)^2 - B^2 = 0 \implies B_0 + 2B_1 = 0 \implies 1 + 2B_1 = 0 \implies \mathbf{B_1 = -1/2}\\)
*   \\((B+1)^3 - B^3 = 0 \implies B_0 + 3B_1 + 3B_2 = 0 \implies 1 - \frac{3}{2} + 3B_2 = 0 \implies \mathbf{B_2 = 1/6}\\)

Substitute these values back:

$$
\sum_{k=0}^{m-1} k^2 = \frac{1}{3}m^3 - \frac{1}{2}m^2 + \frac{1}{6}m
$$

This is exactly the standard formula \\(\frac{m(m-1)(2m-1)}{6}\\) (adjusted for the summation limit \\(m-1\\)).
</details>

### Bernoulli Polynomials
We have seen two parallel worlds:
1.  **Continuous:** \\(D x^n = n x^{n-1}\\) is easy, so we integrate \\(x^n\\) easily.
2.  **Discrete:** \\(\Delta x^{\underline{n}} = n x^{\underline{n-1}}\\) is easy, so we sum \\(x^{\underline{n}}\\) easily.

But what if we stubbornly want to sum standard powers \\(x^n\\) without converting to falling powers? We need a sequence of polynomials \\(P_n(x)\\) that satisfies a "mixed" property:

$$
\Delta P_n(x) = n x^{n-1}
$$

If we found such polynomials, we could sum \\(x^n\\) immediately by telescoping: \\(\sum k^n = \frac{P_{n+1}(k)}{n+1}\\).
These are exactly the **Bernoulli Polynomials**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Bernoulli Polynomials (Umbral Definition)
</div>
Using the same umbral variable \\(B\\) (where \\(B^k \to B_k\\)) from the previous section, we define the Bernoulli polynomial \\(B_n(x)\\) as:

$$
B_n(x) := (B + x)^n = \sum_{k=0}^n \binom{n}{k} B_k x^{n-k}
$$

</blockquote>

This definition reveals their superpower. In the continuous world, they are an **Appell sequence**, meaning their derivative lowers their degree:

$$
\frac{d}{dx} B_n(x) = \frac{d}{dx} (B+x)^n = n(B+x)^{n-1} = n B_{n-1}(x)
$$


But in the discrete world, they satisfy an even more important property.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
The Difference Property
</div>

$$
\Delta B_n(x) = n x^{n-1}
$$

</blockquote>

<blockquote class="box-proof" markdown="1">

Using the Umbral definition \\(B_n(x) = (B+x)^n\\):

$$
\begin{align}
\Delta B_n(x) &= (B+x+1)^n - (B+x)^n \\
&= \sum_{k=0}^n \binom{n}{k} (B+1)^k x^{n-k} - \sum_{k=0}^n \binom{n}{k} B^k x^{n-k} \\
&= \sum_{k=0}^n \binom{n}{k} x^{n-k} \left[ (B+1)^k - B^k \right]
\end{align}
$$

Recall the definition of Bernoulli numbers: \\((B+1)^k - B^k\\) is \\(0\\) for all \\(k > 1\\).
For \\(k=0\\), the term is \\(1-1=0\\).
The only term that survives is \\(k=1\\), where \\((B+1)^1 - B^1 = 1\\).
Thus, the sum collapses to the term where \\(k=1\\):

$$
\Delta B_n(x) = \binom{n}{1} x^{n-1} (1) = n x^{n-1}
$$

</blockquote>

#### The General Summation Formula
Because \\(\Delta \frac{B_{n+1}(x)}{n+1} = x^n\\), we can integrate (sum) \\(x^n\\) directly. This gives us the explicit form of Faulhaber's formula for any range \\([0, m)\\):


$$
\sum_{k=0}^{m-1} k^n = \sum_{k=0}^{m-1} \Delta \left( \frac{B_{n+1}(k)}{n+1} \right) = \boxed{ \frac{B_{n+1}(m) - B_{n+1}(0)}{n+1} }
$$


Since \\(B_{n+1}(0) = B_{n+1}\\) (the Bernoulli number), this matches our previous Umbral formula.

<details class="box-example" markdown="1">
<summary markdown="1">
Sum of first \\(m-1\\) squares
</summary>
Let's find \\(\sum_{k=0}^{m-1} k^2\\) using the polynomial \\(B_3(x)\\).

First, construct \\(B_3(x)\\) using \\(B_0=1, B_1=-1/2, B_2=1/6\\):

$$
\begin{align}
B_3(x) &= \sum_{k=0}^3 \binom{3}{k} B_k x^{3-k} \\
&= \binom{3}{0}B_0 x^3 + \binom{3}{1}B_1 x^2 + \binom{3}{2}B_2 x + \binom{3}{3}B_3 \\
&= 1 \cdot x^3 + 3(-\frac{1}{2})x^2 + 3(\frac{1}{6})x + B_3 \\
&= x^3 - \frac{3}{2}x^2 + \frac{1}{2}x + B_3
\end{align}
$$

Now apply the formula \\(\sum k^2 = \frac{B_3(m) - B_3(0)}{3}\\). Note that \\(B_3(0)\\) is just the constant term \\(B_3\\), so they cancel out:

$$
\sum_{k=0}^{m-1} k^2 = \frac{1}{3} \left( x^3 - \frac{3}{2}x^2 + \frac{1}{2}x \right) = \frac{m^3}{3} - \frac{m^2}{2} + \frac{m}{6}
$$

Which is the correct result.
</details>
### Bonus: The Continuous Operator Algebra

To wrap up, let's look at how these operator techniques apply to the continuous world. throughout this text, we saw that the discrete derivative \\(\Delta\\) and the falling power \\(x^{\underline{n}}\\) satisfy the relationship \\(\Delta x^{\underline{n}} = n x^{\underline{n-1}}\\).

In the continuous world, we have the derivative \\(D = \frac{d}{dx}\\) and the position operator \\(x\\) (multiplying by \\(x\\)). The fundamental relationship between them is the **commutator**:

$$
[D, x] = Dx - xD = 1
$$

(This is because \\(D(xf) - x(Df) = (f + xf') - xf' = f\\)).

This non-commutative structure generates the **Hermite Polynomials** (specifically the "probabilist" convention \\(He_n(x)\\) used in statistics and Brownian motion), which are the continuous analogs to the discrete falling factorials.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Probabilist's Hermite Polynomials
</div>
Instead of using a Taylor expansion, we can define these polynomials purely through an operator acting on the constant \\(1\\):

$$
He_n(x) := (x - D)^n \cdot 1
$$


For example:
- \\(He_0(x) = 1\\)
- \\(He_1(x) = (x - D)1 = x\\)
- \\(He_2(x) = (x - D)x = x^2 - Dx = x^2 - 1\\)
- \\(He_3(x) = (x - D)(x^2 - 1) = x(x^2 - 1) - D(x^2 - 1) = x^3 - x - 2x = x^3 - 3x\\)
</blockquote>

Does this structure look familiar? It should. In the discrete case, \\(x^{\underline{n}}\\) is the eigenbasis for the operator \\(x\Delta\\). In the continuous case, \\(He_n(x)\\) allows us to solve differential equations involving \\(e^{-x^2/2}\\) purely through algebra.

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
The Hermite Recurrence Relation
</div>
Using **only** the operator definition \\(He_n(x) = (x-D)^n \cdot 1\\) and the commutator \\([D, x] = 1\\), prove the famous three-term recurrence relation:

$$
He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
$$


**Hint:**
1. Write \\(He_{n+1} = (x-D)(x-D)^n \cdot 1\\).
2. Expand the product. You will need to figure out how to swap \\(D\\) and \\((x-D)^n\\).
3. Prove the lemma that \\([D, (x-D)^n] = n(x-D)^{n-1}\\) (which looks suspiciously like a power rule!).
</blockquote>

<details class="box-solution" markdown="1">
<summary markdown="1">
**Click for Solution**
</summary>
We start with the definition:

$$
He_{n+1}(x) = (x - D) \underbrace{(x - D)^n \cdot 1}_{He_n(x)}
$$


$$
He_{n+1}(x) = x He_n(x) - D He_n(x)
$$


Now we need to evaluate \\(D He_n(x)\\). Let's find the commutator of \\(D\\) with the operator \\(A = (x-D)\\).
Note that \\([D, x-D] = [D, x] - [D, D] = 1 - 0 = 1\\).

Since \\(D\\) commutes with itself, the commutator \\([D, A^n]\\) follows the derivative rule for operators (similar to \\(d/dx(f^n) = n f^{n-1} f'\\)):

$$
[D, (x-D)^n] = n(x-D)^{n-1} [D, (x-D)] = n(x-D)^{n-1} \cdot 1
$$


So, we have the identity:

$$
D(x-D)^n = (x-D)^n D + n(x-D)^{n-1}
$$


Apply this to the constant function \\(1\\) (remembering that \\(D \cdot 1 = 0\\)):

$$
D He_n(x) = D(x-D)^n \cdot 1 = \left( (x-D)^n D + n(x-D)^{n-1} \right) \cdot 1
$$


$$
D He_n(x) = 0 + n \underbrace{(x-D)^{n-1} \cdot 1}_{He_{n-1}(x)}
$$


$$
D He_n(x) = n He_{n-1}(x)
$$


Substituting this back into our first equation:

$$
\boxed{ He_{n+1}(x) = x He_n(x) - n He_{n-1}(x) }
$$


This exercise confirms that \\(He_n(x)\\) behaves in the continuous "operator world" exactly like the falling powers \\(x^{\underline{n}}\\) behave in the discrete world, just with a different "creation operator" (\\(x-D\\) instead of multiplication by \\(x\\)).
</details>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Appell sequence
</div>
Prove that 

$$
\frac{d}{dx} He_{n}(x)=nHe_{n-1}(x)
$$

</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Shift operator
</div>
Show that

$$
e^{tD}xe^{-tD}=x+t
$$

Then prove for suitable \\(f\\)

$$
(e^{tD}f)(x)=f(x+t)
$$

Hint: take \\(\frac{d}{dt} (e^{tD}xe^{-tD})\\) using \\([D,x]=1\\)
</blockquote>

<blockquote class="box-exercise" markdown="1">
<div class="title" markdown="1">
Ladder operator
</div>
Define operators

$$
a=\frac{1}{\sqrt{2}}(x+D), \qquad a^{\dagger}=\frac{1}{\sqrt{ 2 }}(x-D)
$$

Use only \\([D,x]=1\\) to prove \\([a,a^{\dagger}]=1\\).
Hint: expand \\(aa^{\dagger},a^{\dagger}a\\) in \\(x,D\\).
</blockquote>

### Conclusion

Discrete Calculus is not just a "low-resolution" version of Continuous Calculus. It is a parallel universe with its own basis functions (\\(x^{\underline{n}}\\)), operators (\\(\Delta, \Sigma\\)), and constants (\\(B_n\\)).

By understanding the mapping between these worlds—using tools like Stirling numbers to change bases, and linear functionals to manipulate "shadows"—you can solve complex combinatorial summations and difference equations as easily as you solve integrals in standard calculus.

When you face a problem in discrete math involving sums of polynomials, binomial coefficients, or recursive sequences, ask yourself: **"What would this look like if I wrote it in the falling factorial basis?"** Usually, the answer is "a lot simpler."

---

**Exercise 1: Umbral Evaluation**
Let \\(L\\) be a linear functional defined on the basis \\(z^n\\) such that \\(L(z^n) = n!\\) (This corresponds to the Gamma function, \\(\int_0^\infty t^n e^{-t} dt\\)).
Compute \\(L((z+1)^2)\\) in two steps:
1. Expand the polynomial \\((z+1)^2\\) algebraically.
2. Apply \\(L\\) to the resulting terms.

**Exercise 2: Proving the Difference Rule**
Using the Discrete Binomial Theorem \\((x+y)^{\underline{n}} = \sum_{k=0}^n \binom{n}{k} x^{\underline{k}} y^{\underline{n-k}}\\), prove that:

$$
\Delta (x^{\underline{n}}) = n x^{\underline{n-1}}
$$

*Hint: Recall that \\(\Delta f(x) = f(x+1) - f(x)\\). Set \\(y=1\\) in the expansion.*

**Exercise 3: Sum of Integers**
Use the Umbral Faulhaber Formula to derive the sum of the first \\(m\\) integers (\\(n=1\\)).

$$
\sum_{k=0}^{m-1} k = \frac{1}{2} \left( (m+B)^2 - B^2 \right)
$$

Remember that \\(B_1 = -1/2\\). Does your result match \\(\frac{m(m-1)}{2}\\)?

---
## Exercises

### Part 1: The Difference Operator and Falling Powers

**Exercise 1: The Discrete Product Rule**
Recall the discrete product rule: \\(\Delta(u v) = u \Delta v + E(v) \Delta u\\), where \\(E(v) = v(x+1)\\).
Calculate \\(\Delta(x \cdot 2^x)\\) in two ways:
1.  By applying the definition \\(\Delta f(x) = f(x+1) - f(x)\\) directly.
2.  By applying the discrete product rule with \\(u=x\\) and \\(v=2^x\\).

**Exercise 2: The Discrete Power Rule**
We know that \\(\Delta x^\underline{n} = n x^\underline{n-1}\\).
Using the linearity of \\(\Delta\\), find the forward difference of the polynomial:

$$
f(x) = 3x^\underline{4} - 5x^\underline{2} + 7
$$


**Exercise 3: Converting Bases (Stirling Numbers)**
To differentiate (difference) standard powers like \\(x^3\\), we usually convert them to falling powers first.
1.  Express \\(x^3\\) as a linear combination of falling powers (\\(Ax^\underline{3} + Bx^\underline{2} + Cx^\underline{1}\\)) using algebraic manipulation or Stirling numbers of the second kind.
2.  Use your result to calculate \\(\Delta x^3\\). Verify that your answer matches \\((x+1)^3 - x^3\\).

---

### Part 2: Discrete Integration (Summation)

**Exercise 4: Sum of Cubes**
Using the method of falling powers, calculate the closed-form formula for the sum of the first \\(n\\) cubes:

$$
\sum_{k=0}^{n} k^3
$$

*Hint: Use your conversion from Exercise 3, then apply the Fundamental Theorem of Discrete Calculus: \\(\sum_{0}^{n} x^\underline{k} \delta x = \left[ \frac{x^\underline{k+1}}{k+1} \right]_0^{n+1}\\).*

**Exercise 5: The Geometric Series**
In continuous calculus, \\(\int e^{ax} dx = \frac{1}{a}e^{ax}\\).
In discrete calculus, we know \\(\Delta c^x = c^x(c-1)\\).
1.  Use the antidifference concept to find a closed form for \\(\sum_{k=0}^{n-1} 3^k\\).
2.  Generalize this to find the standard formula for a geometric series \\(\sum_{k=0}^{n-1} r^k\\).

---

### Part 3: Newton Series and Interpolation

**Exercise 6: Finding the Function**
You are given a sequence generated by a polynomial function \\(f(n)\\):

$$
0, 6, 24, 60, 120, \dots
$$

(for \\(n=0, 1, 2, 3, 4 \dots\\))

1.  Construct a **difference table** (calculate \\(\Delta f\\), \\(\Delta^2 f\\), etc.) until the differences become constant.
2.  Use the Newton Series formula to write \\(f(n)\\) as a sum of falling powers:

    $$
f(n) = \sum_{k=0}^\infty \frac{(\Delta^k f)(0)}{k!} n^\underline{k}
    $$

3.  Convert your answer back to standard powers (\\(n^3, n^2\\), etc.) to find the standard polynomial form.

---

### Part 4: Conceptual Proofs

**Exercise 7: The Pascal's Triangle Identity**
The binomial coefficient is defined as \\(\binom{n}{k} = \frac{n^\underline{k}}{k!}\\).
Using the property \\(\Delta n^\underline{k} = k n^\underline{k-1}\\), prove that:

$$
\Delta \binom{n}{k} = \binom{n}{k-1}
$$

Then, write out what this equation means in terms of \\(n\\) and \\(k\\) using the definition \\(\Delta f(n) = f(n+1)-f(n)\\). Does it look like a familiar identity from Pascal's Triangle?

**Exercise 8: Summation by Parts (Discrete Integration by Parts)**
The integration by parts formula is \\(\int u dv = uv - \int v du\\).
Derive the **Summation by Parts** formula by summing the discrete product rule identity (\\(\Delta(uv) = u \Delta v + E(v) \Delta u\\)).
*Result should look like:*

$$
\sum_{k=a}^{b-1} u_k \Delta v_k = [u_k v_k]_a^b - \sum_{k=a}^{b-1} v_{k+1} \Delta u_k
$$


---

### Solutions / Hints

1. \\(2^x(x+2)\\).
2. \\(12x^\underline{3} - 10x^\underline{1}\\).
3. \\(x^3 = x^\underline{3} + 3x^\underline{2} + x^\underline{1}\\). Thus \\(\Delta x^3 = 3x^\underline{2} + 6x^\underline{1} + 1\\).
4. \\(\left(\frac{n(n+1)}{2}\right)^2\\).
5. \\(\sum r^k = \frac{r^n - 1}{r-1}\\).
6. The terms are \\(n^3 - n\\) (or \\(1n^\underline{3} + 3n^\underline{2} + 2n^\underline{1}\\) before conversion).
7. This recovers \\(\binom{n+1}{k} - \binom{n}{k} = \binom{n}{k-1}\\), which is Pascal's Identity.
8. Sum both sides of \\(\Delta(u_k v_k) = u_k \Delta v_k + v_{k+1} \Delta u_k\\). Rearrange to isolate \\(\sum u \Delta v\\). Note that the boundary term is \\([u_k v_k]_a^b\\).

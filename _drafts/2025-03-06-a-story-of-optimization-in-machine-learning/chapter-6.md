---
layout: post
title: "A Story of Optimization in ML: Chapter 6 - Generalizing Euclidean Distance with Bregman Divergences"
description: "Chapter 6 explores how Bregman divergences generalize Euclidean distance for differerently suited geometries."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["Bregman divergence", "non-Euclidean optimization", "Bregman projection", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/mirror_descent_chapter6.gif # Placeholder image path
  alt: "A visual metaphor for mirror descent in non-Euclidean geometry" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

### Diverging from the Norm: Bregman Divergence

It holds for a $$\lambda$$-Lipschitz smooth function that

$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

Interpreting this geometrically, it tells us that the difference between $$f(x)$$ and the linear approximation of $$f$$ at $$y$$ is bounded by a paraboloid with a curvature of $$\lambda$$.

[Quadratic bound on difference between function and linear approximation in 2D]

As we will see, this quantity on the left-hand side is very interesting. To proceed, we will take a detour along the way.

### Optimality of conditional expectation

$$\ell^2$$ loss makes the conditional expectation the optimal predictor, but this does not hold for $$\ell^1$$, which results in the median. In fact, the following is true:

> **Exercise.**  
> Let $$X$$ be an $$\mathbb{R}^n$$-valued random variable and $$Y$$ be another random variable (possibly vector-valued) on the same probability space. Define the $$\ell^p$$ loss for a vector $$x \in \mathbb{R}^n$$ by
> 
> $$
> \|x\|_p^p := \sum_{i=1}^n |x_i|^p.
> $$
> 
> For a given $$p > 1$$ with $$p \neq 2$$, consider the problem of finding a predictor $$\hat{Y}$$ (which may depend on $$Y$$) that minimizes the expected loss
> 
> $$
> \min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
> 
> Demonstrate that, in general, the conditional expectation is not the optimal predictor under the $$\ell^p$$ loss. That is, show that
> 
> $$
> \mathbb{E}[X|Y] \ne \arg\min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
>  
> *Hint:*  
> 1. Use the linearity of expectation to decompose the multivariate problem into $$n$$ independent univariate problems—one for each coordinate.  
> 2. For each coordinate 
> $$i$$, consider the function $$f_i(a) = \mathbb{E}\bigl[|X_i - a|^p \mid Y\bigr]$$ and assume that $$p>1$$ so that the loss is differentiable almost everywhere.  
> 3. Differentiate $$f_i(a)$$ with respect to $$a$$ under the expectation to obtain the first-order optimality condition:  
>    $$
>    \mathbb{E}\Bigl[\operatorname{sgn}(X_i-a)|X_i-a|^{p-1} \mid Y\Bigr] = \mathbb{E}\Bigl[(X_i-a)|X_i-a|^{p-2} \mid Y\Bigr] = 0.
>    $$
> 4. Note that for $$p=2$$ this condition simplifies to  
>    $$
>    \mathbb{E}[X_i - a \mid Y] = 0,
>    $$
>    yielding $$a = \mathbb{E}[X_i|Y]$$; however, for $$p \neq 2$$ the optimal $$a$$ will generally differ from $$\mathbb{E}[X_i|Y]$$.

So this gives rise to the natural question: what other losses beyond squared Euclidean distance ($$\ell^2$$) will make the conditional expectation the optimal predictor? 

This question is answered in [Banerjee et al. (2005)](https://ieeexplore.ieee.org/document/1459065) as a Bregman divergence.

> **Definition. Bregman Divergence**
> Let $$\phi:\mathbb{R}^n\to\mathbb{R}$$ be a strictly convex and differentiable function. The Bregman divergence between two points $$x$$ and $$y$$ is defined as  
> 
> $$  
> D_\phi(x\,\|\,y) = \phi(x) - \left[\phi(y) + \langle \nabla \phi(y), x - y \rangle\right].  
> $$  

Examples taken from [Nielsen and Nock (2008)](https://www.researchgate.net/publication/224460161_Sided_and_Symmetrized_Bregman_Centroids) (definitely worth a read):

### Table: Common Univariate Bregman Divergences $$ D_F(p||q) $$ for Creating Separable Bregman Divergences

$$
\begin{array}{|c|c|c|c|c|}
\hline
\text{Domain } \mathcal{X} & \text{Function } F(x) & \text{Gradient } F'(x) & \text{Inverse Gradient } (F'(x))^{-1} & \text{Divergence } D_F(p||q) \\
\hline
\mathbb{R} & \begin{array}{c} \text{Squared function} \\ x^2 \end{array} & 2x & \frac{x}{2} & \begin{array}{c} (p-q)^2 \\ \text{(Squared loss)} \end{array} \\
\hline
\mathbb{R}_+, \alpha \in \mathbb{N}, \alpha > 1 & \begin{array}{c} \text{Norm-like} \\ x^\alpha \end{array} & \alpha x^{\alpha - 1} & \left( \frac{x}{\alpha} \right)^{\frac{1}{\alpha-1}} & p^\alpha + (\alpha - 1)q^\alpha - \alpha p q^{\alpha -1} \\
\hline
\mathbb{R}^+ & \begin{array}{c} \text{Unnormalized Shannon entropy} \\ x \log x - x \end{array} & \log x & \exp(x) & \begin{array}{c} p \log \frac{p}{q} - p + q \\ \text{(Kullback-Leibler divergence, I-divergence)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Exponential function} \\ \exp x \end{array} & \exp x & \log x & \begin{array}{c} \exp(p) - (p-q+1)\exp(q) \\ \text{(Exponential loss)} \end{array} \\
\hline
\mathbb{R}^+_* & \begin{array}{c} \text{Burg entropy} \\ -\log x \end{array} & -\frac{1}{x} & -\frac{1}{x} & \begin{array}{c} \frac{p}{q} - \log \frac{p}{q} - 1 \\ \text{(Itakura-Saito divergence)} \end{array} \\
\hline
[0,1] & \begin{array}{c} \text{Bit entropy} \\ x \log x + (1-x) \log (1-x) \end{array} & \log \frac{x}{1-x} & \frac{\exp x}{1+\exp x} & \begin{array}{c} p \log \frac{p}{q} + (1-p) \log \frac{1-p}{1-q} \\ \text{(Logistic loss)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Dual bit entropy} \\ \log(1+\exp x) \end{array} & \frac{\exp x}{1+\exp x} & \log \frac{x}{1-x} & \begin{array}{c} \log \frac{1+\exp p}{1+\exp q} - (p-q) \frac{\exp q}{1+\exp q} \\ \text{(Dual logistic loss)} \end{array} \\
\hline
[-1,1] & \begin{array}{c} \text{Hellinger-like function} \\ -\sqrt{1-x^2} \end{array} & \frac{x}{\sqrt{1-x^2}} & \frac{x}{\sqrt{1+x^2}} & \begin{array}{c} \frac{1-pq}{\sqrt{1-q^2}} - \sqrt{1-p^2} \\ \text{(Hellinger-like divergence)} \end{array} \\
\hline
\end{array}
$$

> **Exercise: Non-Negativity and Uniqueness of Zero**  
> 
> **(a)** Prove that $$D_\phi(x\,\|\,y) \geq 0$$ for all $$x,y\in\mathbb{R}^n$$.  
> **(b)** Show that $$D_\phi(x\,\|\,y)=0$$ if and only if $$x=y$$.  
> *Hint:* Use the strict convexity of $$\phi$$ and consider the first-order Taylor expansion of $$\phi$$ at the point $$y$$.

> **Exercise: Bregman Divergence for the Kullback–Leibler (KL) Divergence**  
> Consider the function  
> 
> $$  
> \phi(x) = \sum_{i=1}^n x_i \log x_i - x_i,  
> $$  
> 
> defined on the probability simplex (with the usual convention that $$0\log0=0$$).  
> **(a)** Show that the Bregman divergence induced by $$\phi$$, 
>  
> $$
> D_\phi(x\,\|\,y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y \rangle,  
> $$
> 
> reduces to the KL divergence between $$x$$ and $$y$$.  
> **(b)** Verify explicitly that the divergence is non-negative and zero if and only if $$x=y$$.  
> *Hint:* Compute the gradient $$\nabla \phi(y)$$ and substitute it back into the expression for $$D_\phi(x\,\|\,y)$$.

> **Exercise: Bregman Projections and Proximal Mappings**  
> In many optimization algorithms (such as mirror descent), the update step is formulated as a Bregman projection.  
> **(a)** Given a closed convex set $$\mathcal{C}\subseteq\mathbb{R}^n$$ and a point $$z\in\mathbb{R}^n$$, define the Bregman projection of $$z$$ onto $$\mathcal{C}$$ as  
> 
> $$  
> \operatorname{proj}_{\mathcal{C}}^\phi(z) = \arg\min_{x\in\mathcal{C}} D_\phi(x\,\|\,z).  
> $$  
> 
> Show that when $$\phi(x)=\frac{1}{2}\|x\|_2^2$$, the Bregman projection reduces to the standard Euclidean projection onto $$\mathcal{C}$$.  
> **(b)** Discuss how this concept is connected to the proximal mapping defined earlier through the Moreau envelope. Generalize this concept to a generalize Bregman divergence.
> *Hint:* Recall that the Euclidean proximal mapping for a function $$g$$ is given by  
> 
> $$  
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y}\left\{ g(y) + \frac{1}{2\eta}\|y-v\|_2^2 \right\}.  
> $$

> **Exercise.** [Banarjee et al. (2004)](https://www.researchgate.net/publication/224754032_Optimal_Bregman_prediction_and_Jensen's_equality)
> Define the conditional Bregman information of a random variable $$X$$ for a strictly convex differentable function $$\phi : \mathbb{R}^n \to \mathbb{R}$$ as
>
> $$
> I_{\phi}(X|\mathcal{G}) := \mathbb{E}[D_\phi(x\,\|\,E[X|\mathcal{G}])|\mathcal{G}]
> $$
>
> where $$D_\phi(x\,\|\,y) := \phi(x) - (\phi(y) + \langle \nabla \phi(y), x-y \rangle)$$ is the Bregman divergence under $$\phi$$ from $$y$$ to $$x$$.
>
> Prove that 
> $$I_{\phi}(X|\mathcal{G}) \geq 0$$ for all $$X$$ and $$\phi$$. Then, show Jensen's inequality in the following form:
> 
> $$
> \mathbb{E}[\phi(X)|\mathcal{G}] = \phi(\mathbb{E}[X|\mathcal{G}]) + I_{\phi}(X|\mathcal{G}).
> $$

---

**Further Reading:**

*   [Banerjee et al. (2005) - On the Optimality of Conditional Expectation as a Bregman Predictor](https://ieeexplore.ieee.org/document/1459065)
*   [Nielsen and Nock (2008) - The Sided and Symmetrized Bregman Centroids](https://www.researchgate.net/publication/224460161_Sided_and_Symmetrized_Bregman_Centroids)
*   [Hazan (2019) - Lecture Notes: Optimization for Machine Learning](https://arxiv.org/abs/1909.03550)

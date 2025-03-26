---
layout: post
title: What Do Loss Functions Mean?
date: 2025-03-25 20:32 -0400
description: Beyond just measuring error, loss functions encode deep assumptions about our data and the goals of learning. Let's explore their hidden meanings.
image: /assets/img/loss_functions_concept.png # Suggest using a relevant image here
category: Machine Learning
tags: [loss functions, machine learning, information theory, statistics, optimization, bregman divergence]
math: true
tikz: true
---

## Introduction

You might feel "at a loss" when trying to understand the math behind loss functions. They are often presented without much context, leaving their origins and motivations unclear. Why square the error in regression? Why use that peculiar log-likelihood formula for classification?

This blog post explores the intuition and deeper meanings behind common loss functions. As hinted in the title, we'll find that the concept of the **mean** (or expectation) provides a surprisingly central viewpoint for understanding why these functions take the forms they do and what goals they implicitly pursue.

---

### 0. Warm Up: Linear Regression & A First Look at Loss

One of the simplest settings where loss functions appear is linear regression. Imagine we have data consisting of input features $$x$$ and target values $$y$$, and we want to model their relationship with a line: $$\hat{y} = w^T x + b$$. Here, $$w$$ and $$b$$ are the parameters (weights and bias) we need to learn.

The standard approach is to find the parameters that minimize the **Sum of Squared Errors (SSE)** between the predicted values $$\hat{y}_i$$ and the true values $$y_i$$ across all $$n$$ data points in our dataset $$\mathcal{D} = \{(x_1, y_1), \dots, (x_n, y_n)\}$$:

$$
\min_{w,b} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \min_{w,b} \sum_{i=1}^n (y_i - (w^T x_i + b))^2 = \min_{w,b} \|y - \hat{y}\|_2^2
$$

This is often called the **L2 loss**. Minimizing the SSE is equivalent to minimizing the **Mean Squared Error (MSE)**,

$$
\min_{w,b} \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \min_{w,b} \frac{1}{n} \|y - \hat{y}\|_2^2
$$

since they only differ by a constant factor $$1/n$$. This minimization problem is convex and can be solved efficiently, for instance, using gradient descent or even analytically via the Normal Equations.

Why squared error? Common justifications include:
1.  **Nonnegative Penalty:** Squaring ensures errors are always non-negative, $$ (y_i - \hat{y}_i)^2 \ge 0 $$, with zero error only when prediction matches the target perfectly.
2.  **Differentiability:** The squared error is smooth and easily differentiable, which is convenient for gradient-based optimization.
3.  **Penalizing Large Errors:** Squaring magnifies larger errors more than smaller ones (e.g., an error of 2 contributes 4 to the sum, while an error of 4 contributes 16). This pushes the model to avoid significant mistakes.

While useful, these points don't fully capture the deeper reasons. To explore those, let's first formalize what we mean by a loss function.

> **Definition (Loss Function):**
> In supervised learning, we seek a model $$f_\theta: \mathcal{X} \to \mathcal{Y}$$ parameterized by $$\theta \in \Theta$$, mapping inputs $$x \in \mathcal{X}$$ to outputs $$\hat{y} \in \mathcal{Y}$$.
>
> 1.  A **Pointwise Loss Function** (or Cost Function) $$ \ell: \mathcal{Y} \times \mathcal{Y} \to [0, \infty) $$ measures the discrepancy between a *single* true target value $$y$$ and its corresponding prediction $$\hat{y}$$. Typically, $$\ell(y, y) = 0$$ and $$\ell(y, \hat{y}) \ge 0$$.
>
> 2.  Given a dataset $$\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$$, the **Empirical Loss** (or Objective Function) $$ L_{emp}: \Theta \to [0, \infty) $$ aggregates the pointwise losses over the dataset, quantifying the overall model performance for parameters $$\theta$$. It is typically the arithmetic mean (or expectation):
> 
>     $$
>     L_{emp}(\theta; \mathcal{D}) = \frac{1}{N} \sum_{i=1}^N \ell(y_i, f_\theta(x_i))
>     $$
>
> The process of **training** involves finding the parameters $$\theta^*$$ that minimize this empirical loss:
> 
> $$
> \theta^* = \arg\min_{\theta \in \Theta} L_{emp}(\theta; \mathcal{D})
> $$
> 
> *(Often, a regularization term is added to $$L_{emp}$$ to prevent overfitting).*

Now that we have a clearer definition, let's return to the question: why specific forms for $$\ell(y, \hat{y})$$? We'll start by connecting the familiar squared error to the concept of the mean.

---

### 1. The Mean: A Central Point of Reference

The arithmetic mean is perhaps the most fundamental statistic. For a set of numbers $$\{y_1, \dots, y_N\}$$, it's simply their sum divided by the count: $$\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$. In probability theory, this generalizes to the **expected value** (or expectation) of a random variable $$Y$$, denoted $$E[Y]$$, representing its probability-weighted average value.

What makes the mean so special? Let's reconsider the squared error criterion. Suppose we have a set of data points $$\{y_1, \dots, y_N\}$$, and we want to find a *single constant value* $$c$$ that is "closest" to all these points. If we define "closest" using the sum of squared differences, our objective is to find the $$c$$ that solves:

$$
\min_{c \in \mathbb{R}} J(c) \quad \text{where} \quad J(c) = \sum_{i=1}^N (y_i - c)^2
$$

This is an unconstrained optimization problem. Since $$J(c)$$ is a convex quadratic function (a parabola opening upwards), we can find the minimizing input (argument) of a quadratic equation $$a_2 x^2 + a_1 x + a_0$$ as $$\frac{-a_1}{2a_2}$$.

Thus, expanding:

$$
J(c) = \sum_{i=1}^N (y_i - c)^2 = \sum_{i=1}^N (y_i^2 - 2yc + c^2) = N c^2 - 2\sum_{i=1}^N y_ic + \sum_{i=1}^N y_i^2
$$

Then, we find the optimal $$c^\ast$$:

$$
c^\ast = \frac{1}{N} \sum_{i=1}^N y_i.
$$

The value $$c^*$$ that minimizes the sum of squared differences is precisely the **arithmetic mean** of the data points, $$\bar{y}$$!

**What does this mean?** It tells us that the mean is the optimal "summary" or "representative point" for a dataset *if* our criterion for optimality is minimizing squared deviations. This provides our first deep insight into the L2 loss: **Minimizing squared error is intrinsically linked to finding the mean.**

This connection extends to random variables. If $$Y$$ is a random variable, the constant $$c$$ that minimizes the **expected squared error** $$E[(Y - c)^2]$$ is the expected value $$c = E[Y]$$. The minimum value achieved is $$E[(Y - E[Y])^2]$$, which is the definition of the **Variance** of $$Y$$.

---

#### A Geometric Perspective: The Mean as a Projection

There's also a powerful geometric interpretation of the mean using **orthogonal projection** and the **Pythagorean theorem**.

Think of the data vector $$ y = (y_1, y_2, \dots, y_N)^T $$ as a point in $$ \mathbb{R}^N $$. Now consider the 1-dimensional subspace of $$ \mathbb{R}^N $$ that consists of all constant vectors—those of the form $$ (c, c, \dots, c)^T $$. This subspace is spanned by the all-ones vector $$ \mathbf{1} = (1, 1, \dots, 1)^T $$.

Projecting $$ y $$ onto this subspace finds the constant vector $$ \hat{y} $$ that is closest to $$ y $$ in Euclidean (L2) distance. That projection turns out to be:

$$
\hat{y} = \frac{\langle y, \mathbf{1} \rangle}{\|\mathbf{1}\|^2} \mathbf{1} = \left( \frac{1}{N} \sum_{i=1}^N y_i \right) \cdot \mathbf{1} = (\bar{y}, \bar{y}, \dots, \bar{y})^T.
$$

So the mean $$ \bar{y} $$ naturally emerges as the coefficient of this projection. It's not just an average—it's the **best constant approximation** to $$ y $$ under squared error, viewed as a projection in a vector space.

The **Pythagorean theorem** then explains the squared error:

$$
\|y\|_2^2 = \|\hat{y}\|_2^2 + \|y - \hat{y}\|_2^2,
$$

where $$ \|y - \hat{y}\|_2^2 $$ is the sum of squared residuals—the loss we’re minimizing.
<!-- 
<script type="text/tikz">
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\begin{tikzpicture}[scale=1.2, thick]
  % Axes
  \draw[->] (-0.5,0) -- (4.5,0) node[anchor=west] {\footnotesize $\text{span}(\mathbf{1})$};
  \draw[->] (0,-0.5) -- (0,3.5) node[anchor=south east] {\footnotesize $\mathbb{R}^n$};

  % y vector
  \draw[->, blue, very thick] (0,0) -- (2.2,2.8) node[anchor=south west] {\footnotesize $y$};

  % projection (onto horizontal axis)
  \draw[->, orange, very thick] (0,0) -- (3.3,0) node[anchor=north] {\footnotesize $\hat{y}$};

  % residual vector
  \draw[->, red, dashed, thick] (3.3,0) -- (2.2,2.8) node[midway, right=2pt] {\footnotesize $y - \hat{y}$};

  % right angle indicator
  \draw (3.1,0) -- (3.1,0.2) -- (2.9,0.2);
\end{tikzpicture}
\end{document}
</script> -->

This shows that squared loss has deep geometric roots: minimizing it is equivalent to orthogonally projecting $$ y $$ onto a subspace, and the mean arises as the optimal point in that subspace. This perspective will resurface again when we look at linear regression more generally.

This fundamental property sets the stage for understanding L2 loss in more complex modeling scenarios.

*(Next section will likely cover the L2 loss perspectives: Probabilistic/MLE, Conditional Expectation, Geometric)*

---

## Further Reading

- [Reid (2013) - Meet the Bregman Divergences](https://mark.reid.name/blog/meet-the-bregman-divergences.html)
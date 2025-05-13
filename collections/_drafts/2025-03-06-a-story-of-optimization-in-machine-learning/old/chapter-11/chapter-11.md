---
layout: post
title: "A Story of Optimization in ML: Chapter 11 - Curved Statistics: The Fisher Information Matrix and the Geometry of Learning"
description: "Chapter 11 introduces the Fisher Information Matrix, tracing its origins in statistics and showing how it naturally defines a geometry on parameter spaces, leading to deeper insights in optimization."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["information geometry", "Fisher information", "natural gradient", "statistical estimation", "Riemannian geometry"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/fisher_information_chapter11.gif
  alt: "Visual metaphor for curved statistical spaces and Fisher information"
date: 2025-03-22 14:00 +0000
math: true
---

## Chapter 11: Curved Statistics - The Fisher Information Matrix and the Geometry of Learning

Imagine walking blindfolded across an unfamiliar landscape. You can’t see far, but by carefully feeling the slope under your feet, you can infer the shape of the ground.

In statistics and machine learning, the landscape is the space of probability distributions. Our "slope" is how sensitive those distributions are to the parameters we’re trying to learn. But what if the terrain isn’t flat? What if the space of parameters is **curved**?

Enter the **Fisher Information Matrix**—an elegant tool that quantifies not only how precisely we can estimate parameters but also reveals the intrinsic geometry of statistical models.

---

## 1. Estimation and Precision: Where It All Began

Suppose you have data sampled from a distribution \( p(x \mid \theta) \), but you don't know the true parameter \( \theta \). You want to construct an estimator \( \hat{\theta}(x) \) that guesses \( \theta \) based on the data.

Naturally, you'd like your estimate to be **precise and unbiased**. But is there a fundamental limit to how precisely we can estimate \( \theta \)?

This question intrigued **R.A. Fisher** in the early 20th century, and his answer was the concept of **Fisher Information**.

---

## 2. Measuring Sensitivity: The Fisher Information Matrix

The key intuition:

> **If small changes in \( \theta \) lead to large changes in the likelihood function, then we should be able to estimate \( \theta \) more precisely.**

### Definition:

For a scalar parameter \( \theta \):

\[
F(\theta) = \mathbb{E}_{x \sim p(x \mid \theta)} \left[ \left( \frac{\partial}{\partial \theta} \log p(x \mid \theta) \right)^2 \right].
\]

For a vector parameter \( \theta \in \mathbb{R}^n \), the **Fisher Information Matrix (FIM)** is:

\[
F(\theta) = \mathbb{E}_{x \sim p(x \mid \theta)} \left[ \nabla_\theta \log p(x \mid \theta) \, \nabla_\theta \log p(x \mid \theta)^\top \right].
\]

---

### 2.1. Example: The Bernoulli Distribution

Consider a coin flip modeled as:

\[
p(x \mid \theta) = \theta^x (1 - \theta)^{1 - x}, \quad x \in \{0, 1\}, \quad 0 < \theta < 1.
\]

Compute:

\[
\frac{\partial}{\partial \theta} \log p(x \mid \theta) = \frac{x}{\theta} - \frac{1 - x}{1 - \theta}.
\]

Then:

\[
F(\theta) = \mathbb{E}_{x} \left[ \left( \frac{x}{\theta} - \frac{1 - x}{1 - \theta} \right)^2 \right] = \frac{1}{\theta(1 - \theta)}.
\]

**Insight:**

- Fisher information grows large when \( \theta \to 0 \) or \( \theta \to 1 \), where the outcome becomes almost deterministic.
- It's smallest at \( \theta = 0.5 \), where randomness is maximized.

---

## 3. Cramér-Rao Bound: Precision Has Limits

Fisher Information isn't just abstract—it quantifies a **fundamental limit** on estimation precision.

The **Cramér-Rao bound** tells us:

\[
\mathrm{Var}(\hat{\theta}) \geq \frac{1}{F(\theta)}.
\]

This means:

- No unbiased estimator can achieve variance smaller than \( 1/F(\theta) \).
- **Higher Fisher information → Tighter bound → Greater precision.**

It’s a ceiling imposed by nature, not by our choice of estimator.

---

## 4. From Statistics to Geometry: A Riemannian View

Here’s where the story takes a geometric turn.

Fisher observed that **Fisher Information behaves like a metric**, defining how "far apart" two distributions are when you vary \( \theta \).

In Euclidean space, distance is:

\[
ds^2 = dx^\top dx.
\]

In the space of probability distributions, Fisher introduced:

\[
ds^2 = d\theta^\top F(\theta) d\theta.
\]

This is a **Riemannian metric**—meaning, the parameter space forms a **curved manifold**, with the Fisher Information Matrix capturing its local geometry.

---

## 5. Information Geometry: The Statistical Manifold

In the 1980s, **Shun'ichi Amari** and others formalized these insights into **Information Geometry**:

- **Statistical Manifold:**  
  The space of all probability distributions parameterized by \( \theta \).

- **Fisher-Rao Metric:**  
  The Fisher Information Matrix defines the metric tensor, measuring distances between distributions.

- **Geodesics & Curvature:**  
  Optimization paths and inference trajectories follow curved lines (geodesics) respecting this metric.

---

## 6. Key Properties of the Fisher Information Matrix

Beyond intuition, the Fisher Information Matrix has **powerful mathematical properties** that make it indispensable in both statistics and optimization.

---

### 6.1. Relation to the Hessian of Log-Likelihood

For differentiable densities, Fisher Information is directly related to the curvature of the log-likelihood:

\[
F(\theta) = -\mathbb{E}_{x \sim p(x \mid \theta)} \left[ \nabla^2_\theta \log p(x \mid \theta) \right].
\]

**Why?**

Starting with:

\[
\nabla_\theta \log p(x \mid \theta) = \frac{1}{p(x \mid \theta)} \nabla_\theta p(x \mid \theta),
\]

the second derivative gives:

\[
\nabla^2_\theta \log p(x \mid \theta) = \frac{\nabla^2_\theta p(x \mid \theta)}{p(x \mid \theta)} - \nabla_\theta \log p(x \mid \theta) \, \nabla_\theta \log p(x \mid \theta)^\top.
\]

Taking expectations cancels the first term (under regularity conditions), yielding:

\[
F(\theta) = \mathbb{E} \left[ \nabla_\theta \log p(x \mid \theta) \, \nabla_\theta \log p(x \mid \theta)^\top \right] = -\mathbb{E} \left[ \nabla^2_\theta \log p(x \mid \theta) \right].
\]

---

### 6.2. Observed vs Expected Fisher Information

- **Expected Fisher Information:**  
  The average curvature over the data distribution.
  
- **Observed Fisher Information:**  
  Evaluates the negative Hessian at the specific observed data:

  \[
  I_{\text{obs}}(\theta) = -\nabla^2_\theta \log p(x_{\text{obs}} \mid \theta).
  \]

In large samples, both align, but the observed version is often used in practical estimation (e.g., standard errors).

---

### 6.3. Additivity: Independent Samples

Fisher Information **adds up across independent data points**:

\[
F_{\text{total}}(\theta) = \sum_{i=1}^n F_i(\theta).
\]

This underlies:

- Consistency of MLEs.
- Asymptotic normality (MLE converges to Gaussian with covariance \( F(\theta)^{-1} \)).

---

### 6.4. Invariance Under Reparameterization

**Fisher Information transforms correctly under smooth, invertible changes of variables.**

If \( \theta = h(\phi) \), then:

\[
F_\phi(\phi) = J(\phi)^\top F_\theta(h(\phi)) J(\phi),
\]

where \( J(\phi) = \frac{\partial h(\phi)}{\partial \phi} \).

**Consequence:**

- Fisher Information is **intrinsic**—it depends only on the family of distributions, not how we parameterize them.
- This makes optimization methods like **Natural Gradient Descent** coordinate-invariant.

---

## 7. Applications Beyond Estimation

The Fisher Information Matrix shows up everywhere:

- **Statistical Inference:**  
  MLE consistency, asymptotic variances, hypothesis testing.

- **Bayesian Inference:**  
  Jeffreys prior is derived from \( \sqrt{\det F(\theta)} \).

- **Optimization:**  
  Natural gradient methods (deep learning, variational inference, reinforcement learning).

- **Second-Order Methods:**  
  Approximating the Hessian efficiently via Fisher Information (e.g., K-FAC optimizer).

---

## 8. Conclusion: Statistics Meets Geometry

The Fisher Information Matrix began as a tool to quantify estimation precision. But it revealed something much deeper:

- It defines a **curved geometry** on parameter spaces.
- It connects to the **Hessian of log-likelihood**, providing curvature information.
- It enjoys beautiful properties like **additivity** and **invariance**, making it robust in both theory and practice.

Most importantly, it teaches us that in optimization, not all steps are created equal—the shape of the space matters.

---

## Exercises

> **Exercise 1:** Derive the Fisher Information Matrix for a univariate Gaussian with unknown mean and known variance.

> **Exercise 2:** Show explicitly that for the Bernoulli distribution, the Fisher Information adds linearly over \( n \) i.i.d. samples.

> **Exercise 3:** Prove the invariance of Fisher Information under smooth reparameterizations.

---

## Further Reading

- Fisher, R.A. (1922) – *On the Mathematical Foundations of Theoretical Statistics*
- Amari, S. – *Information Geometry and Its Applications*
- Nielsen, F. – *An Elementary Introduction to Information Geometry*
- Martens, J. (2020) – [New Insights and Perspectives on the Natural Gradient Method](https://jmlr.org/papers/volume21/17-678/17-678.pdf)

---

*In the next chapter, we’ll show how this geometric viewpoint connects Natural Gradient Descent with Mirror Descent—two methods that, at their core, exploit the same principles of curved optimization spaces.*

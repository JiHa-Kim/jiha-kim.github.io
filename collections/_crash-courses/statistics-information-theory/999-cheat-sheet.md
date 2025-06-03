---
title: "Statistics & Info Theory Cheat Sheet: Key Formulas & Definitions"
date: 2025-06-02 10:00 -0400 # Date after Part 2
course_index: 999 # To place it at the end of this specific crash course
series_index: 0 # Indicates it's part of a crash course collection
description: "A quick reference guide with key formulas and definitions from the Statistics and Information Theory crash course for machine learning."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Cheat Sheet
- Statistics
- Information Theory
- Probability
- Formulas
- Definitions
- Crash Course
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  Never introduce any non-existant path, like an image.
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

## Introduction

This cheat sheet provides a quick reference to the key definitions, formulas, and concepts covered in the Statistics & Info Theory Part 1: Statistical Foundations for ML and Statistics & Info Theory Part 2: Information Theory Essentials for ML crash course posts. Use it as a quick reminder or for reviewing core ideas.

## Part 1: Statistical Foundations Recap

### Basic Probability

| Concept                | Formula / Definition                                                      |
| :--------------------- | :------------------------------------------------------------------------ |
| **Conditional Prob.**  | $$P(A \vert B) = \frac{P(A \cap B)}{P(B)}$$, for $$P(B) > 0$$             |
| **Independence**       | $$A, B$$ indep. if $$P(A \cap B) = P(A)P(B)$$, or $$P(A \vert B) = P(A)$$ |
| **Bayes' Theorem**     | $$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}$$                          |
| **Law of Total Prob.** | $$P(B) = \sum_i P(B \vert A_i)P(A_i)$$ for partition $$\{A_i\}$$          |

### Random Variables (RVs)

| Concept                       | Discrete RV                                 | Continuous RV                                               |
| :---------------------------- | :------------------------------------------ | :---------------------------------------------------------- |
| **Description**               | Probability Mass Function (PMF): $$p_X(x)$$ | Probability Density Function (PDF): $$f_X(x)$$              |
| **PMF/PDF Properties**        | $$\sum_x p_X(x) = 1$$, $$p_X(x) \ge 0$$     | $$\int_{-\infty}^{\infty} f_X(x) dx = 1$$, $$f_X(x) \ge 0$$ |
| **CDF $$F_X(x)=P(X \le x)$$** | $$F_X(x) = \sum_{k \le x} p_X(k)$$          | $$F_X(x) = \int_{-\infty}^x f_X(t) dt$$                     |

### Expectation, Variance, Covariance

| Concept                             | Formula / Definition                                                                                                                      |
| :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Expected Value $$E[X]$$**         | Discrete: $$\sum_x x p_X(x)$$. Continuous: $$\int x f_X(x) dx$$                                                                           |
| **Variance $$Var(X)$$**             | $$E[(X - E[X])^2] = E[X^2] - (E[X])^2$$                                                                                                   |
| **Standard Deviation $$\sigma_X$$** | $$\sqrt{Var(X)}$$                                                                                                                         |
| **Covariance $$Cov(X,Y)$$**         | $$E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]$$                                                                                                |
| **Correlation $$\rho_{X,Y}$$**      | $$\frac{Cov(X,Y)}{\sigma_X \sigma_Y}$$, where $$-1 \le \rho_{X,Y} \le 1$$                                                                 |
| **Covariance Matrix $$\Sigma$$**    | For RV vector $$\mathbf{X}$$, $$\Sigma = E[(\mathbf{X} - E[\mathbf{X}])(\mathbf{X} - E[\mathbf{X}])^T]$$. $$\Sigma_{ij} = Cov(X_i, X_j)$$ |

### Common Probability Distributions

| Distribution            | Type       | Parameters                     | PMF / PDF $$p(x;\cdot)$$ or $$f(x;\cdot)$$                                                                                           | Mean $$E[X]$$        | Variance $$Var(X)$$    |
| :---------------------- | :--------- | :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------- | :--------------------- |
| **Bernoulli**           | Discrete   | $$p$$                          | $$p^x (1-p)^{1-x}$$ for $$x \in \{0,1\}$$                                                                                            | $$p$$                | $$p(1-p)$$             |
| **Binomial**            | Discrete   | $$n, p$$                       | $$\binom{n}{x} p^x (1-p)^{n-x}$$                                                                                                     | $$np$$               | $$np(1-p)$$            |
| **Categorical**         | Discrete   | $$\mathbf{p}=(p_1,\dots,p_K)$$ | $$P(X=k) = p_k$$                                                                                                                     | (Vector)             | (Cov Matrix)           |
| **Poisson**             | Discrete   | $$\lambda$$                    | $$\frac{\lambda^x e^{-\lambda}}{x!}$$                                                                                                | $$\lambda$$          | $$\lambda$$            |
| **Uniform**             | Continuous | $$a, b$$                       | $$\frac{1}{b-a}$$ for $$x \in [a,b]$$                                                                                                | $$\frac{a+b}{2}$$    | $$\frac{(b-a)^2}{12}$$ |
| **Normal (Gaussian)**   | Continuous | $$\mu, \sigma^2$$              | $$\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$                                                                   | $$\mu$$              | $$\sigma^2$$           |
| **Multivariate Normal** | Continuous | $$\boldsymbol{\mu}, \Sigma$$   | $$\frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$ | $$\boldsymbol{\mu}$$ | $$\Sigma$$             |
| **Exponential**         | Continuous | $$\lambda$$                    | $$\lambda e^{-\lambda x}$$ for $$x \ge 0$$                                                                                           | $$1/\lambda$$        | $$1/\lambda^2$$        |

### Important Theorems

| Theorem                         | Summary                                                                                                                                                                                                            |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Law of Large Numbers (LLN)**  | Sample mean $$\bar{X}_n$$ converges to true mean $$E[X]$$ as $$n \to \infty$$.                                                                                                                                     |
| **Central Limit Theorem (CLT)** | Sum/average of many i.i.d. RVs (with finite mean/variance) tends towards a normal distribution, regardless of original distribution. $$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$ |

### Statistical Estimation

| Concept                                | Definition / Formula                                               |
| :------------------------------------- | :----------------------------------------------------------------- |
| **Estimator Bias**                     | $$Bias(\hat{\theta}) = E[\hat{\theta}] - \theta$$                  |
| **Estimator MSE**                      | $$MSE(\hat{\theta}) = Var(\hat{\theta}) + (Bias(\hat{\theta}))^2$$ |
| **Likelihood $$L(\theta; D)$$**        | $$L(\theta; D) = \prod_{i=1}^n p(x_i; \theta)$$ (for i.i.d. data)  |
| **Log-Likelihood $$\ell(\theta; D)$$** | $$\ell(\theta; D) = \sum_{i=1}^n \log p(x_i; \theta)$$             |
| **MLE $$\hat{\theta}_{MLE}$$**         | $$\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta; D)$$         |

## Part 2: Information Theory Essentials Recap

### Entropy Measures

| Concept                                        | Formula / Definition                                     | Notes                                                                  |
| :--------------------------------------------- | :------------------------------------------------------- | :--------------------------------------------------------------------- |
| **Shannon Entropy $$H(X)$$** (Discrete)        | $$-\sum_{x \in \mathcal{X}} p(x) \log_b p(x)$$           | Units: bits (b=2), nats (b=e). $$H(X) \ge 0$$.                         |
| **Differential Entropy $$h(X)$$** (Continuous) | $$-\int_{\mathcal{X}} f(x) \log f(x) dx$$                | Can be negative. Units usually nats.                                   |
| **Joint Entropy $$H(X,Y)$$**                   | $$-\sum_{x,y} p(x,y) \log p(x,y)$$ (discrete)            | Total uncertainty of pair $$(X,Y)$$.                                   |
| **Conditional Entropy $$H(Y \vert X)$$**       | $$-\sum_{x,y} p(x,y) \log p(y \vert x) = H(X,Y) - H(X)$$ | Remaining uncertainty in $$Y$$ given $$X$$. $$H(Y \vert X) \le H(Y)$$. |
| **Chain Rule for Entropy**                     | $$H(X,Y) = H(X) + H(Y \vert X)$$                         | Generalizes to multiple variables.                                     |

### Information Measures and Divergences

| Concept                                 | Formula / Definition                                                                                      | Key Properties                                                             |
| :-------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| **Mutual Information $$I(X;Y)$$**       | $$H(X) - H(X \vert Y)$$ $$= H(X) + H(Y) - H(X,Y)$$ $$= \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$   | $$I(X;Y) \ge 0$$. $$I(X;Y)=0 \iff X,Y$$ independent. Symmetric.            |
| **KL Divergence $$D_{KL}(P \Vert Q)$$** | Discrete: $$\sum_x p(x) \log \frac{p(x)}{q(x)}$$ <br> Continuous: $$\int p(x) \log \frac{p(x)}{q(x)} dx$$ | $$D_{KL}(P \Vert Q) \ge 0$$. $$D_{KL}(P \Vert Q)=0 \iff P=Q$$. Asymmetric. |
| **Cross-Entropy $$H(P,Q)$$**            | Discrete: $$-\sum_x p(x) \log q(x)$$ <br> Continuous: $$-\int p(x) \log q(x) dx$$                         | $$H(P,Q) = H(P) + D_{KL}(P \Vert Q)$$. Common loss function.               |

### Fisher Information

| Concept                                                   | Definition / Formula                                                                                                                                                                                                                                                                                                                              |
| :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Score Function $$\mathbf{s}(\boldsymbol{\theta}; x)$$** | $$\nabla_{\boldsymbol{\theta}} \log p(x; \boldsymbol{\theta})$$                                                                                                                                                                                                                                                                                   |
| **Fisher Information Matrix $$I(\boldsymbol{\theta})$$**  | $$E_{X \sim p(x;\boldsymbol{\theta})}\left[ (\nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta})) (\nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta}))^T \right]$$ <br> OR <br> $$-E_{X \sim p(x;\boldsymbol{\theta})}\left[ \nabla^2_{\boldsymbol{\theta}} \log p(x; \boldsymbol{\theta}) \right]$$ (under regularity conditions) |
| **Cramér-Rao Lower Bound (CRLB)**                         | For unbiased estimator $$\hat{\boldsymbol{\theta}}$$, $$Cov(\hat{\boldsymbol{\theta}}) \succeq I(\boldsymbol{\theta})^{-1}$$                                                                                                                                                                                                                      |
| **Local KL-FIM Relation**                                 | $$D_{KL}(p(\cdot;\boldsymbol{\theta}) \Vert p(\cdot;\boldsymbol{\theta} + d\boldsymbol{\theta})) \approx \frac{1}{2} d\boldsymbol{\theta}^T I(\boldsymbol{\theta}) d\boldsymbol{\theta}$$                                                                                                                                                         |

---

This cheat sheet is intended as a condensed summary. For detailed explanations, derivations, and examples, please refer to the full posts in the crash course.

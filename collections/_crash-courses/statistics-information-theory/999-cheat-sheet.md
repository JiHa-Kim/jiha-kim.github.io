---
title: "Statistics & Info Theory Cheat Sheet: Key Formulas & Definitions"
date: 2025-06-02 10:00 -0400 # Date after Part 2
sort_index: 999 # To place it at the end of this specific crash course
sort_index: 0 # Indicates it's part of a crash course collection
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

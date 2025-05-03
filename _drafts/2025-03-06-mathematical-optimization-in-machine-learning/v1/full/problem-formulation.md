---
layout: post
title: "What Problem is Machine Learning Trying to Solve? A Formal Look"
date: 2025-03-06 12:00:00 +0000
description: "Formalizing the core optimization problem in supervised ML, its pitfalls, and why it often works."
image: assets\2025-03-06-mathematical-optimization-in-machine-learning\thumbnail.png
categories:
- Machine Learning
- Mathematical Optimization
tags: ["Formalism", "Supervised Learning", "Empirical Risk Minimization", "Optimization", "Overfitting", "Generalization", "Statistical Learning Theory"]
math: true
---

Welcome to the first post in our series on optimization theory in machine learning! Before diving into *how* we optimize, we need to understand *what* we're optimizing. Machine learning often feels like magic – how does a machine learn to spot cats or translate languages just from examples? While the results can seem magical, the underlying process, especially in **supervised learning**, revolves around solving a well-defined mathematical problem. Let's unpack that problem.

**The Goal: Learning to Generalize**

At its heart, machine learning aims to create systems that **generalize**. We want them to learn patterns from past experiences (data) and use those patterns to make accurate predictions or decisions on new, unseen situations. Instead of us writing explicit rules for every possible scenario, the machine should learn the rules itself.

Think about **spam email detection**:

*   **Experience (Data):** We have a collection of emails, each labeled "Spam" or "Not Spam," along with features like words used, sender, time sent, etc.
*   **Goal:** Create a function that takes a *new*, unlabeled email's features and correctly predicts if it's spam.

The core challenge is moving from the *limited set of examples* we have to a rule that works well on the *potentially infinite stream of future emails*.

**The Ideal World: Minimizing Error on All Possible Data (True Risk)**

Imagine there's a perfect, underlying truth governing the relationship between inputs (email features) and outputs (spam/not spam). Let's formalize this ideal scenario:

1.  **Input Space ($$ \mathcal{X} $$):** The set of all possible inputs (e.g., feature vectors for emails). Often $$ \mathcal{X} \subseteq \mathbb{R}^d $$.
2.  **Output Space ($$ \mathcal{Y} $$):** The set of all possible outputs (e.g., $$ \{0, 1\} $$ for Spam/Not Spam, or $$ \mathbb{R} $$ for predicting house prices).
3.  **Unknown True Distribution ($$ P(X, Y) $$):** A probability distribution governing how input-output pairs $$ (x, y) $$ are generated in the real world. **Crucially, we don't know $$ P $$.**
4.  **Hypothesis Space ($$ \mathcal{H} $$):** The set of *candidate functions* (models, predictors) $$ f: \mathcal{X} \to \mathcal{Y} $$ we are willing to consider. This is *our choice*. Examples: linear models, decision trees, specific neural networks.
5.  **Loss Function ($$ L $$):** A function $$ L: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0} $$ measuring the cost of predicting $$ \hat{y} = f(x) $$ when the true answer is $$ y $$. Common examples:
    *   **0-1 Loss** (classification): $$ L(\hat{y}, y) = \mathbb{I}(\hat{y} \neq y) $$
    *   **Squared Loss** (regression): $$ L(\hat{y}, y) = (\hat{y} - y)^2 $$

Ideally, we'd find the function $$ f $$ in our chosen set $$ \mathcal{H} $$ that has the lowest average loss over *all possible data points* drawn from the true distribution $$ P $$. This average loss is the **Expected Risk** or **True Risk**, $$ R(f) $$:

$$
R(f) = \mathbb{E}_{(X,Y) \sim P} [L(f(X), Y)] = \int_{\mathcal{X} \times \mathcal{Y}} L(f(x), y) \, dP(x, y)
$$

The ultimate, theoretical goal is to find the best function $$ f^* $$ in our hypothesis space:

$$
f^* = \arg \min_{f \in \mathcal{H}} R(f)
$$

**The Big Problem:** We can't calculate $$ R(f) $$ because we don't know the true distribution $$ P(X, Y) $$! We only have a limited sample.

**The Practical Approach: Minimizing Error on Our Data (Empirical Risk)**

Since we can't access $$ P $$, we use the data we *do* have as a substitute.

1.  **Training Data ($$ D $$):** We're given $$ n $$ examples $$ D = \{(x_1, y_1), \dots, (x_n, y_n)\} $$. We assume these are drawn independently and identically distributed (i.i.d.) from the unknown $$ P $$.

Instead of minimizing the *true* risk, we calculate the average loss of a function $$ f $$ *on our training data*. This is the **Empirical Risk**, $$ \hat{R}_D(f) $$:

$$
\hat{R}_D(f) = \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i)
$$

This leads us to the central principle of much of supervised machine learning: **Empirical Risk Minimization (ERM)**.

**The ERM Problem:** Find the function $$ \hat{f} $$ in our hypothesis space $$ \mathcal{H} $$ that minimizes the empirical risk:

$$
\boxed{
\hat{f} = \arg \min_{f \in \mathcal{H}} \hat{R}_D(f) = \arg \min_{f \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i)
}
$$

**This minimization problem is precisely where optimization theory enters the picture.** The job of a machine learning *algorithm* is often to solve (or approximately solve) this optimization problem to find $$ \hat{f} $$.

**The Hope: Why Minimizing Empirical Risk Might Work**

We take a leap of faith: we hope that the function $$ \hat{f} $$ that performs best on our *training data* (minimizes $$ \hat{R}_D(f) $$) will also perform well on *new, unseen data* (have low $$ R(f) $$).

Why might this be true? Intuitively, if our training sample $$ D $$ is large and representative of the true distribution $$ P $$, then the empirical risk $$ \hat{R}_D(f) $$ should be a good approximation of the true risk $$ R(f) $$. Statistical learning theory provides the formal backing for this intuition. This is also why we use separate test sets – to estimate how well $$ \hat{f} $$ actually generalizes.

<blockquote class="prompt-warning" markdown="1">
I should note that it is not immediately obvious that the arithmetic mean of the sample data is the best approximation of the population data. Luckily, this is the case for the mean, but not for the variance.

A somewhat surprising fact in statistics shows that the best approximation to population variance with sample data, called the sample variance, is not the same as the variance applied on the sample data.

Given a sample $$X_1, \dots, X_n$$ from the population, the sample variance is:

$$
\sigma^2_\text{sample} = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2 = \frac{n-1}{n} \text{Var}_\text{data} (X)
$$

where 

$$
\bar{X} = \mathbb{E}_\text{data}[X] = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

denotes the sample mean.
</blockquote>

**Recap: Knowns vs. Unknowns**

*   **We Know:** Training Data ($$D$$), Input/Output Spaces ($$\mathcal{X}, \mathcal{Y}$$), Hypothesis Space ($$\mathcal{H}$$), Loss Function ($$L$$).
*   **We Don't Know:** True Distribution ($$P$$), True Risk ($$R(f)$$), Best Possible Function ($$f^*$$), True Performance of our Learned Function ($$R(\hat{f})$$).

**When ERM Fails: The Danger Zone**

Simply minimizing empirical risk isn't foolproof. Our learned function $$ \hat{f} $$ might look great on the training data but fail miserably in the real world. Key failure modes include:

This ERM framework is the backbone of much of ML, but it's not a magic bullet. Minimizing loss on the training data doesn't automatically guarantee success on new data. Here are some ways it can fail:

1.  **Overfitting:** This is perhaps the most classic failure mode. The model learns the training data *too well*, capturing noise and idiosyncrasies specific to the sample $$ D $$, rather than the underlying pattern from $$ P $$.
    *   **Example:** Imagine trying to "learn" the outcome of a fair six-sided die. In the true process, we generate, say, 6 samples. Our training data is $$ D = \{(\text{input } 1, \text{output } 1), (\text{input } 2, \text{output } 2), (\text{input } 4, \text{output } 4)\} $$. A function $$ \hat{f} $$ could achieve **zero empirical risk** ($$ \hat{R}_D(\hat{f}) = 0 $$) by simply memorizing these three pairs and maybe predicting "0" for inputs 3, 5, and 6. This function is perfect on $$ D $$, but its true risk $$ R(\hat{f}) $$ will be high (likely $$ 3/6 = 1/2 $$ using 0-1 loss, assuming inputs 1-6 are equally likely) because it fails completely on unseen inputs. The model didn't learn the concept of a die roll; it just memorized the sample. This often happens when the hypothesis space $$ \mathcal{H} $$ is too complex relative to the amount of data $$ n $$.

2.  **Bad or Biased Data:** The ERM principle relies heavily on the assumption that the training data $$ D $$ is a representative sample (i.i.d.) from the true distribution $$ P $$ we care about. If the sampling process is flawed, the empirical risk $$ \hat{R}_D(f) $$ might be a poor estimate of the true risk $$ R(f) $$.
    *   **Example:** Suppose we want to learn a function related to the **ReLU** activation, $$ g(x) = \max(0, x) $$. But, due to a faulty sensor or data collection process, our training data $$ D $$ only contains examples where $$ x < -0.1 $$. In this case, all $$ y_i = g(x_i) $$ values in $$ D $$ will be 0. An ERM algorithm might learn the function $$ \hat{f}(x) = 0 $$ for all $$ x $$. This function has zero empirical risk on $$ D $$. However, it will be completely wrong for any future data point where $$ x > 0 $$. The learned model reflects the bias in the data, not the true underlying function.

3.  **Fundamentally Hard Problems:** Sometimes, the underlying relationship $$ P(X, Y) $$ or the optimal function $$ f^* $$ might be inherently difficult to approximate, even with lots of data.
    *   **Example:** Consider a function $$ f: \mathbb{R} \to \{0, 1\} $$ defined as $$ f(x) = 1 $$ if $$ x $$ is irrational and $$ f(x) = 0 $$ if $$ x $$ is rational. Even if we could sample infinitely many points $$ (x_i, y_i) $$ according to some distribution over $$ x $$, it's extremely hard to find a function in typical hypothesis spaces (like polynomials or standard neural networks) that captures this wildly discontinuous behavior. The structure is too complex or "pathological" for standard learning algorithms to grasp from samples.

These pitfalls show that blindly minimizing $$ \hat{R}_D(f) $$ isn't enough. We need strategies to ensure our solution generalizes.

**Why ERM Often Succeeds (The Theory Behind the Practice)**

Despite the dangers, ERM is incredibly successful. Why does minimizing loss on a sample often lead to good performance on unseen data? It's a combination of statistical foundations, careful model design, and the nature of our optimization methods.

1.  **Statistics: The Sample Reflects Reality (Mostly):**
    *   **Law of Large Numbers (LLN) Intuition:** For any *single* function $$ f $$, as the sample size $$ n $$ grows, its empirical risk $$ \hat{R}_D(f) $$ converges to its true risk $$ R(f) $$.
    *   **Uniform Convergence:** Statistical learning theory goes further. For "well-behaved" (not overly complex) hypothesis spaces $$ \mathcal{H} $$, the empirical risk converges to the true risk *uniformly* across all functions $$ f \in \mathcal{H} $$ as $$ n $$ increases. This means the entire landscape of empirical risk $$ \hat{R}_D(\cdot) $$ starts to look like the true risk landscape $$ R(\cdot) $$. This is crucial because ERM involves *searching* this landscape. The complexity of $$ \mathcal{H} $$ (measured by concepts like VC dimension or Rademacher complexity) determines how much data $$ n $$ is needed for this approximation to hold reliably.

2.  **Controlling Complexity: Priors and Regularization:**
    *   **Choosing $$ \mathcal{H} $$ (Hard Prior):** Simply selecting a *specific* hypothesis space (e.g., linear models) is a strong choice. We are essentially saying solutions *must* come from this set, encoding a belief about the form of the solution. Simpler $$ \mathcal{H} $$ generally require less data to avoid overfitting.
    *   **Regularization (Soft Prior):** Instead of just minimizing $$ \hat{R}_D(f) $$, we often minimize a modified objective: $$ \hat{R}_D(f) + \lambda \Omega(f) $$. The term $$ \Omega(f) $$ penalizes complexity (e.g., large weights), and $$ \lambda $$ controls the trade-off. This encourages simpler solutions *within* $$ \mathcal{H} $$ that still fit the data reasonably well. This is closely related to Bayesian MAP estimation, where $$ \Omega(f) $$ acts like a prior belief favoring simpler models (e.g., L2 regularization corresponds to a Gaussian prior on weights). It's a way to inject preference for simpler solutions, implementing Occam's Razor.

3.  **The Power of Algorithms & Big Data:**
    *   **Big Data:** With massive datasets ($$ n \to \infty $$), the empirical risk becomes a very accurate proxy for the true risk. The data "shouts louder" than any prior beliefs (regularization becomes less critical), allowing complex models ($$ \mathcal{H} $$) to be trained reliably.
    *   **Optimization Algorithms (e.g., SGD):** The algorithms used to solve the ERM problem often have beneficial properties. Stochastic Gradient Descent (SGD), widely used in deep learning, processes data point by point (or in mini-batches). It can be viewed through the lens of **Stochastic Approximation (SA)**. The gradient computed on a small batch is a noisy estimate of the *true* risk gradient $$ \nabla R(f) $$. So, conceptually, SGD takes noisy steps towards minimizing the *true risk* directly, rather than just the fixed empirical risk $$ \hat{R}_D(f) $$. The noise in SGD can also act as a form of implicit regularization, often favoring flatter minima that generalize better. Theories like **Online-to-Batch (O2B) conversion** provide formal links showing that the iterative process of algorithms like SGD can lead directly to solutions with good generalization guarantees ($$ R(f) $$ is low).

**Conclusion: Setting the Stage for Optimization**

So, what problem is machine learning trying to solve? At its core, supervised ML seeks a function that generalizes well from observed data to unseen data. The ideal target is minimizing the **True Risk**, but this is impossible as we don't know the true data distribution. The practical approach is **Empirical Risk Minimization (ERM)**: minimizing the average loss on the available training data.

**Finding the function $$ \hat{f} $$ that minimizes this empirical risk $$ \hat{R}_D(f) $$ is the fundamental *optimization problem* that learning algorithms need to solve.**

While pitfalls like overfitting exist, ERM often works well due to statistical convergence properties (especially uniform convergence), explicit constraints and priors (choice of $$ \mathcal{H} $$, regularization), the sheer amount of data available today, and sometimes, the beneficial implicit biases of the optimization algorithms themselves (like SGD).

Understanding this ERM framework is crucial because it defines the objective function landscape that our optimization algorithms will navigate. In the next posts in this series, we will dive into *how* these optimization algorithms work to find the minimum of the empirical risk and tackle the challenges involved.

---


## Further Reading and References

The concepts discussed here form the foundation of statistical learning theory and modern machine learning practice. For readers interested in exploring these topics in more detail, the following resources are highly recommended:

**Core Statistical Learning Theory & Practice:**

1.  **Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.**
    *   Focuses specifically on the theoretical foundations. Excellent, clear explanations of PAC learning, ERM, hypothesis space complexity (VC dimension, Rademacher complexity), uniform convergence bounds (Ch 4-6), and the theory behind generalization. Crucial for understanding *why* ERM works from a classical perspective.

2.  **Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.**
    *   Often considered the "bible" of statistical learning. Provides a comprehensive and mathematically rigorous treatment of many supervised learning methods, loss functions, model assessment, regularization (Ch 3), and the bias-variance trade-off. Discusses the ERM principle extensively.

3.  **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning: with Applications in R*. Springer.**
    *   A more accessible introduction compared to *The Elements*. Covers core concepts like ERM, overfitting, cross-validation, and regularization (Ridge, Lasso) with less mathematical depth but excellent intuition and practical R examples. Great starting point.

**Probabilistic/Bayesian Perspective:**

4.  **Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.**
    *   Provides a thorough introduction from a probabilistic perspective. Beautifully explains the connection between loss functions and likelihoods (Ch 1), and explicitly derives regularization (like L2) as MAP estimation using Gaussian priors (Ch 3). Excellent for understanding the Bayesian interpretation.

5.  **Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.**
    *   Another comprehensive textbook with a strong probabilistic focus. Similar to Bishop, it provides detailed explanations of the Bayesian approach, likelihoods, priors, and MAP estimation in the context of various models.

**Online Learning, Optimization, and Generalization:**

6.  **Orabona, F. (2019). *A Modern Introduction to Online Learning*. arXiv:1912.13213.** [Latest version available on arXiv]
    *   An excellent, modern monograph focusing on online convex optimization and regret minimization. It rigorously covers algorithms like Online Mirror Descent and Follow-The-Regularized-Leader. **Chapter 3 specifically details Online-to-Batch conversions**, providing a theoretical link between the low regret of online algorithms (like those related to SGD) and the low generalization error (true risk) of the resulting predictors. Essential reading for the SA/Online/O2B perspective on generalization.

7.  **Cutkosky, A. (2019). Online-to-Batch Conversions via Coarse-Graining. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97:1483-1492.** (And related works like "Anytime Online-to-Batch, Optimism and Acceleration", PMLR 97:1446-1454, 2019).
    *   These papers delve into specific theoretical results and techniques for online-to-batch conversions, showing how the performance of an online algorithm translates into guarantees for batch learning (i.e., generalization on the underlying distribution). They offer deeper insights into the mechanics mentioned in point #6 of the main text.

These resources collectively cover the statistical foundations of ERM (uniform convergence), the Bayesian interpretation (priors via regularization), and the optimization-centric view (SA, SGD, and online-to-batch guarantees) explaining why minimizing empirical risk, often through sophisticated algorithms, leads to models that generalize well to unseen data.

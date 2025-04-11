---
layout: post
title: "What Problem is Machine Learning Trying to Solve? A Formal Look"
date: 2025-03-06
description: "Formalizing the core optimization problem in supervised ML, its pitfalls, and why it often works."
image: assets\2025-03-06-mathematical-optimization-in-machine-learning\thumbnail.png
categories:
- Machine Learning
- Mathematical Optimization
tags: ["Formalism", "Supervised Learning", "Empirical Risk Minimization", "Optimization", "Overfitting", "Generalization", "Statistical Learning Theory"]
math: true
---

Before actually exploring the theory behind optimization in machine learning, we have to first understand what problem we are even optimizing to begin with. This is something that was explored significantly, but is often glossed over in more modern texts. Let's break it down.

Machine learning feels like magic sometimes. How can a computer learn to identify cat pictures, translate languages, or detect spam emails just by looking at examples? While the applications are diverse, much of machine learning, especially **supervised learning**, boils down to solving a specific, well-defined mathematical problem. Let's break down exactly what that problem is.

**The Goal: Learning from Experience**

At its core, machine learning aims to **generalize** from past observations (data) to make accurate predictions or decisions on new, unseen situations. Instead of hand-crafting rules for every possibility, we want the machine to *learn* the underlying patterns itself.

Consider a classic example: **spam email detection**.

*   **Experience (Data):** We have a collection of emails, each labeled as "Spam" or "Not Spam". For each email, we also have features like the words used, sender information, time of day, etc.
*   **Desired Outcome:** We want a function that takes the features of a *new*, unlabeled email and correctly predicts whether it's spam or not.

The challenge is moving from the finite set of *labeled* emails we have to a rule that works well on the potentially infinite stream of *future* emails.

**The Ideal, But Unreachable, Target: Minimizing True Risk**

Mathematically, we imagine there's some underlying truth governing the relationship between inputs (email features) and outputs (spam/not spam). Let's define the components:

1.  **Input Space ($$ \mathcal{X} $$):** The set of all possible inputs. For our emails, an input $$ x \in \mathcal{X} $$ is a vector representing its features (e.g., word counts, sender domain). Often, $$ \mathcal{X} $$ is a subset of $$ d $$-dimensional real space, $$ \mathcal{X} \subseteq \mathbb{R}^d $$.
2.  **Output Space ($$ \mathcal{Y} $$):** The set of all possible outputs. For spam detection (binary classification), $$ \mathcal{Y} = \{0, 1\} $$ (where 0 might mean "Not Spam" and 1 means "Spam"). For predicting house prices (regression), it might be $$ \mathcal{Y} = \mathbb{R} $$.
3.  **Unknown True Distribution ($$ P(X, Y) $$):** This is a joint probability distribution over $$ \mathcal{X} \times \mathcal{Y} $$. We assume nature generates input-output pairs $$ (x, y) $$ according to this distribution. **Crucially, we do not know $$ P $$.**
4.  **Hypothesis Space ($$ \mathcal{H} $$):** This is the set of *possible functions* (or models, predictors, hypotheses) $$ f: \mathcal{X} \to \mathcal{Y} $$ that we are willing to consider. This is a choice we make. Examples include:
    *   The set of all linear classifiers.
    *   The set of all decision trees up to depth 5.
    *   The set of all functions representable by a specific neural network architecture.
5.  **Loss Function ($$ L $$):** A function $$ L: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0} $$ that measures the "cost" or "error" when the true label is $$ y $$ and our function predicts $$ \hat{y} = f(x) $$. A common choice for classification is the **0-1 Loss**:

    $$
    L(\hat{y}, y) = \mathbb{I}(\hat{y} \neq y) = \begin{cases} 1 & \text{if } \hat{y} \neq y \\ 0 & \text{if } \hat{y} = y \end{cases}
    $$

    For regression, the **Squared Loss** is common:

    $$
    L(\hat{y}, y) = (\hat{y} - y)^2
    $$

Ideally, we want to find the function $$ f $$ within our chosen hypothesis space $$ \mathcal{H} $$ that minimizes the expected loss over *all possible data points* drawn from the true distribution $$ P $$. This is called the **Expected Risk** or **True Risk**, $$ R(f) $$:

$$
R(f) = \mathbb{E}_{(X,Y) \sim P} [L(f(X), Y)] = \int_{\mathcal{X} \times \mathcal{Y}} L(f(x), y) \, dP(x, y)
$$

The ultimate goal is to find the best function $$ f^* $$ in our set $$ \mathcal{H} $$:

$$
f^* = \arg \min_{f \in \mathcal{H}} R(f)
$$

**The Problem:** We can't calculate $$ R(f) $$ because we don't know $$ P(X, Y) $$!

**The Practical Solution: Empirical Risk Minimization (ERM)**

Since we can't access the true distribution $$ P $$, we use the data we *do* have as a proxy.

1.  **Training Data ($$ D $$):** We are given a set of $$ n $$ examples $$ D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\} $$. We assume these are drawn independently and identically distributed (i.i.d.) from the unknown true distribution $$ P $$. $$ (x_i, y_i) \sim P(X, Y) $$ for $$ i=1, \dots, n $$.

We can calculate the average loss of a function $$ f $$ *on our training data*. This is called the **Empirical Risk**, $$ \hat{R}_D(f) $$:

$$
\hat{R}_D(f) = \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i)
$$

The core idea of most supervised machine learning is **Empirical Risk Minimization (ERM)**. Instead of minimizing the unknown true risk $$ R(f) $$, we minimize the empirical risk $$ \hat{R}_D(f) $$ which we *can* compute:

$$
\hat{f} = \arg \min_{f \in \mathcal{H}} \hat{R}_D(f)
$$

The learning algorithm's job is typically to perform this minimization (or an approximation of it) to find $$ \hat{f} $$.

**The Leap of Faith (and Statistical Theory)**

The fundamental hope is that the function $$ \hat{f} $$ that performs best on the *training data* (minimizes empirical risk) will also perform well on *unseen data* (has low true risk). Intuitively, if our training sample $$ D $$ is large and representative enough of the true distribution $$ P $$, then $$ \hat{R}_D(f) $$ should be a good approximation of $$ R(f) $$. This is the reason why we split our data into training and testing sets: to evaluate how well our model generalizes to unseen data.

**Summary: Knowns and Unknowns**

Let's recap what we work with and what we're trying to find:

**Knowns:**

1.  The **Training Data** $$ D = \{(x_i, y_i)\}_{i=1}^n $$.
2.  The **Input Space** $$ \mathcal{X} $$ and **Output Space** $$ \mathcal{Y} $$.
3.  The chosen **Hypothesis Space** $$ \mathcal{H} $$ (our set of candidate functions).
4.  The chosen **Loss Function** $$ L(\hat{y}, y) $$ (how we measure error).

**Unknowns:**

1.  The **True Data Generating Distribution** $$ P(X, Y) $$.
2.  The **True Risk** $$ R(f) $$ for any given function $$ f $$.
3.  The **Optimal Function** $$ f^* $$ that minimizes the true risk within $$ \mathcal{H} $$.
4.  The **Actual Performance** of our learned function $$ \hat{f} $$ on future, unseen data (i.e., $$ R(\hat{f}) $$).

**When Things Go Wrong: The Limits of ERM**

This ERM framework is the backbone of much of ML, but it's not a magic bullet. Minimizing loss on the training data doesn't automatically guarantee success on new data. Here are some ways it can fail:

1.  **Overfitting:** This is perhaps the most classic failure mode. The model learns the training data *too well*, capturing noise and idiosyncrasies specific to the sample $$ D $$, rather than the underlying pattern from $$ P $$.
    *   **Example:** Imagine trying to "learn" the outcome of a fair six-sided die. Our training data is $$ D = \{(\text{input } 1, \text{output } 1), (\text{input } 2, \text{output } 2), (\text{input } 4, \text{output } 4)\} $$. A function $$ \hat{f} $$ could achieve **zero empirical risk** ($$ \hat{R}_D(\hat{f}) = 0 $$) by simply memorizing these three pairs and maybe predicting "0" for inputs 3, 5, and 6. This function is perfect on $$ D $$, but its true risk $$ R(\hat{f}) $$ will be high (likely $$ 3/6 = 1/2 $$ using 0-1 loss, assuming inputs 1-6 are equally likely) because it fails completely on unseen inputs. The model didn't learn the concept of a die roll; it just memorized the sample. This often happens when the hypothesis space $$ \mathcal{H} $$ is too complex relative to the amount of data $$ n $$.

2.  **Bad or Biased Data:** The ERM principle relies heavily on the assumption that the training data $$ D $$ is a representative sample (i.i.d.) from the true distribution $$ P $$ we care about. If the sampling process is flawed, the empirical risk $$ \hat{R}_D(f) $$ might be a poor estimate of the true risk $$ R(f) $$.
    *   **Example:** Suppose we want to learn a function related to the **ReLU** activation, $$ g(x) = \max(0, x) $$. But, due to a faulty sensor or data collection process, our training data $$ D $$ only contains examples where $$ x < -0.1 $$. In this case, all $$ y_i = g(x_i) $$ values in $$ D $$ will be 0. An ERM algorithm might learn the function $$ \hat{f}(x) = 0 $$ for all $$ x $$. This function has zero empirical risk on $$ D $$. However, it will be completely wrong for any future data point where $$ x > 0 $$. The learned model reflects the bias in the data, not the true underlying function.

3.  **Fundamentally Hard Problems:** Sometimes, the underlying relationship $$ P(X, Y) $$ or the optimal function $$ f^* $$ might be inherently difficult to approximate, even with lots of data.
    *   **Example:** Consider a function $$ f: \mathbb{R} \to \{0, 1\} $$ defined as $$ f(x) = 1 $$ if $$ x $$ is irrational and $$ f(x) = 0 $$ if $$ x $$ is rational. Even if we could sample infinitely many points $$ (x_i, y_i) $$ according to some distribution over $$ x $$, it's extremely hard to find a function in typical hypothesis spaces (like polynomials or standard neural networks) that captures this wildly discontinuous behavior. The structure is too complex or "pathological" for standard learning algorithms to grasp from samples.

These examples highlight that blindly minimizing empirical risk is not enough. We need to consider the complexity of our model ($$ \mathcal{H} $$), the quality and quantity of our data ($$ D $$), and the nature of the underlying problem ($$ P $$).

**Why ERM (and its practical approximations) Often Succeeds: Bridging the Gap Between Sample and Reality**

Empirical Risk Minimization (ERM) – minimizing loss on the available data $$D$$ – is the cornerstone of modern machine learning and is remarkably successful in practice. Why doesn't optimizing on a limited sample typically lead to poor performance when the model encounters new, unseen data from the true distribution $$P$$? The reasons lie in a combination of statistical principles, careful model design, implicit/explicit prior beliefs, the nature of the optimization algorithms used, and often, the sheer scale of data.

1.  **The Sample as a Mirror (Law of Large Numbers Intuition):** For any *single*, fixed function $$ f $$, the Law of Large Numbers tells us that its average loss on a large random sample $$ D $$ (its empirical risk $$ \hat{R}_D(f) $$) will be very close to its average loss over the entire distribution $$ P $$ (its true risk $$ R(f) $$). The empirical risk $$ \hat{R}_D(f) $$ acts like a sample average, approximating the true average $$ R(f) $$. However, ERM involves *searching* for the best $$f$$.

2.  **Uniform Convergence: Taming the Entire Hypothesis Space:** ERM *searches* through an entire space $$ \mathcal{H} $$ of functions. Statistical learning theory provides **uniform convergence bounds**, showing that for hypothesis spaces $$ \mathcal{H} $$ that are not "infinitely complex" (e.g., have finite VC dimension or bounded Rademacher complexity), with high probability, the empirical risk $$ \hat{R}_D(f) $$ is close to the true risk $$ R(f) $$ *simultaneously for all functions $$ f \in \mathcal{H} $$*, provided the sample size $$ n $$ is large enough. This ensures the empirical risk landscape largely reflects the true risk landscape across the entire space we are searching.

    $$
    P\left( \sup_{f \in \mathcal{H}} \vert \hat{R}_D(f) - R(f) \vert \le \epsilon(n, \mathcal{H}, \delta) \right) \ge 1 - \delta
    $$

    If $$ \hat{f}_{ERM} = \arg \min_{f \in \mathcal{H}} \hat{R}_D(f) $$ and $$ f^* = \arg \min_{f \in \mathcal{H}} R(f) $$, this uniformity allows us to bound the excess risk: $$ R(\hat{f}_{ERM}) - R(f^*) \le 2 \sup_{f \in \mathcal{H}} \vert \hat{R}_D(f) - R(f) \vert $$.

3.  **Controlling Complexity ($$ \mathcal{H} $$) - A Hard Prior:** The error term $$ \epsilon $$ in the uniform convergence bound depends crucially on the **complexity** of $$ \mathcal{H} $$. Choosing a *specific* $$ \mathcal{H} $$ (like linear models, decision trees of a certain depth, or a particular neural network architecture) is itself a strong modeling assumption. It acts like placing a **"hard" prior belief**: we restrict our search to functions within this chosen set, effectively assigning zero prior probability to functions outside it. A simpler $$ \mathcal{H} $$ implies a stronger prior belief in simplicity and typically leads to better bounds (smaller $$ \epsilon $$ for a given $$ n $$), reducing the risk of overfitting.

4.  **Regularization: Injecting Soft Prior Beliefs:** Instead of just minimizing empirical risk $$ \hat{R}_D(f) $$, we often minimize a regularized objective:

    $$
    \hat{f} = \arg \min_{f \in \mathcal{H}} \left( \hat{R}_D(f) + \lambda \Omega(f) \right)
    $$

    Here, $$ \Omega(f) $$ penalizes function complexity (e.g., norm of weights), and $$ \lambda $$ controls the trade-off. This is often mathematically equivalent to **Maximum A Posteriori (MAP)** estimation in Bayesian inference.
    *   **The Bayesian Link:** MAP seeks $$ f $$ maximizing $$ p(f \vert D) \propto p(D \vert f) p(f) $$. Minimizing the regularized objective often corresponds to maximizing $$ \log p(D \vert f) + \log p(f) $$, where $$ \hat{R}_D(f) $$ relates to the negative log-likelihood $$-\log p(D|f)$$ and $$ \lambda \Omega(f) $$ relates to the negative log-prior $$ -\log p(f) $$.
    *   **Examples:** L2 regularization ($$ \Omega(f) = \Vert w \Vert_2^2 $$) corresponds to a Gaussian prior; L1 regularization ($$ \Omega(f) = \Vert w \Vert_1 $$) corresponds to a Laplacian prior (promoting sparsity).
    *   **Interpretation:** Regularization injects a **"soft" prior belief** favouring simpler functions (as defined by $$ \Omega $$) *within* $$ \mathcal{H} $$. It guides optimization towards solutions that fit the data reasonably well *and* align with our prior notion of plausibility (e.g., smaller weights), helping to prevent overfitting by implementing a form of Occam's Razor.

5.  **The Blessing of Big Data:** As $$ n $$ increases, the empirical risk $$ \hat{R}_D(f) $$ becomes a more accurate estimate of $$ R(f) $$. In the regularized objective, the data term $$ \hat{R}_D(f) $$ (or likelihood in MAP) increasingly dominates the regularization term $$ \lambda \Omega(f) $$ (or prior). With massive datasets, the data "speaks loudly," allowing us to reliably learn complex functions even within complex $$ \mathcal{H} $$ because the empirical evidence strongly points towards the true underlying patterns.

6.  **Optimization Algorithms: Stochastic Approximation (SA) and Online-to-Batch Guarantees:** While ERM defines the *objective* ($$ \hat{R}_D(f) $$ or its regularized version), the *algorithms* used to minimize it, especially for large datasets, provide another crucial lens:
    *   **The Goal & Batch Approach:** We want to minimize the *true risk* $$ R(f) $$. Standard ERM approximates $$ R(f) $$ with $$ \hat{R}_D(f) $$ and then minimizes the latter, often using Batch Gradient Descent which computes the gradient over the entire dataset $$ D $$: $$ \nabla_\theta \hat{R}_D(f_\theta) = \frac{1}{n} \sum_{i=1}^n \nabla_\theta L(f_\theta(x_i), y_i) $$. This is computationally expensive for large $$ n $$.
    *   **Stochastic Gradient Descent (SGD):** This algorithm takes a different route. At each step $$ t $$, it approximates the true gradient using just **one** or a **mini-batch** of training example(s) $$ (x_t, y_t) $$, sampled from $$ D $$ (approximating drawing from $$ P $$): $$ g_t(\theta_t) = \nabla_\theta L(f_{\theta_t}(x_t), y_t) $$ (or an average over a mini-batch). The update is $$ \theta_{t+1} = \theta_t - \eta_t g_t(\theta_t) $$.
    *   **The SA Connection:** SGD is a classic Stochastic Approximation algorithm. The stochastic gradient $$ g_t(\theta_t) $$ is a *noisy but unbiased estimate* of the true gradient $$ \nabla_\theta R(f_{\theta_t}) $$ (under i.i.d. sampling and assuming expectation/gradient swap): $$ \mathbb{E}[g_t(\theta_t)] \approx \nabla_\theta R(f_{\theta_t}) $$. Conceptually, SGD directly takes steps to minimize the *true risk* using noisy gradient information from samples, rather than explicitly minimizing the fixed empirical risk $$ \hat{R}_D(f) $$.
    *   **Online-to-Batch (O2B) Conversions:** This perspective connects directly to **online learning theory**. Algorithms like SGD, when viewed as processing a sequence of examples (even if sampled from a fixed dataset $$D$$), resemble online learning algorithms. O2B theorems (e.g., Cesa-Bianchi et al., 2004; Cutkosky, 2019; Orabona, 2019, Chapter 3) provide guarantees that link the performance of an online algorithm (often measured by its *regret* – how much worse it performs compared to the best fixed function in hindsight over the sequence) to the *generalization error* (expected true risk) of a predictor derived from it (e.g., the average of its iterates, or its final iterate).
        *   **Significance:** O2B bounds often show that if an online algorithm achieves low regret (meaning it learns effectively from the sequence), the resulting predictor will have low true risk $$R(f)$$. This provides an *alternative theoretical justification* for why algorithms like SGD lead to good generalization, especially in convex settings. Instead of relying solely on uniform convergence over the entire function class $$ \mathcal{H} $$ before optimization starts, O2B analyzes the *dynamics of the learning process itself*. It shows that the *path* taken by the optimizer, driven by sequential data, can directly lead to a solution that performs well on the underlying distribution $$P$$.
    *   **Benefits of the SA / Online View:**
        *   **Direct (Approximate) Optimization of True Risk:** SGD aims for the true risk minimum.
        *   **Computational Efficiency:** Cheap updates enable massive datasets.
        *   **Implicit Regularization:** SGD's noise can prefer flatter minima, improving generalization.
        *   **Strong Theoretical Guarantees:** Convergence proofs from SA theory and generalization guarantees from O2B theory explain *why* these practical algorithms succeed in finding solutions with low true risk.

**In Summary:** ERM and its practical implementations succeed due to a confluence of factors. The sample data reflects the true distribution (LLN, Uniform Convergence). We constrain the search space ($$\mathcal{H}$$) and inject prior beliefs (regularization/MAP) to guide the search towards plausible solutions. Crucially, the optimization algorithms often used (like SGD) can be viewed as stochastic approximation schemes that directly, albeit noisily, optimize the true risk. Online-to-batch conversion theory further solidifies this by showing that the online learning process inherent in these algorithms can itself guarantee good generalization performance, complementing or sometimes replacing arguments based purely on the properties of the function class and the empirical risk objective.

**Conclusion**

So, what is machine learning trying to do from a mathematical standpoint? Supervised learning uses a finite dataset $$ D $$ to find a function $$ \hat{f} $$ within a chosen class $$ \mathcal{H} $$ that minimizes the average error on $$ D $$ (empirical risk $$ \hat{R}_D(f) $$). This serves as a proxy for the ultimate, but unreachable, goal of minimizing the error on all possible future data (true risk $$ R(f) $$). While pitfalls like overfitting exist, the principle of ERM is often successful. This success stems from a combination of factors: the statistical convergence of empirical to true risk (especially uniform convergence), the constraining effect of choosing a hypothesis space (a hard prior), the explicit preference for simplicity via regularization (a soft prior, often equivalent to Bayesian MAP), the power of large datasets, and sometimes beneficial biases of optimization algorithms. These elements collectively bridge the gap between learning from the past and predicting the future with remarkable effectiveness.

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
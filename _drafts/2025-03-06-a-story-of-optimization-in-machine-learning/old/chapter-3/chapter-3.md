---
layout: post
title: "A Story of Optimization in ML: Chapter 3 - Adapting to the Landscape"
description: "Chapter 3 delves into adaptive learning rates, exploring how methods like AdaGrad, RMSProp, and Adam dynamically adjust step sizes during optimization for more efficient descent."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["gradient descent", "adaptive learning rate", "AdaGrad", "RMSProp", "Adam", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/adaptive_lr_chapter3.gif # Placeholder image path
  alt: "Visual representation of adaptive learning rates in optimization" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 3: Adapting to the Landscape - Parameter-Wise Learning Rates

In the previous chapter, we saw how momentum helps smooth out the noisy descent of Stochastic Gradient Descent (SGD), leading to faster and more stable convergence. But even with momentum, we’ve been using a **single, fixed learning rate** for *all* parameters throughout the entire optimization process.

Is this always the best approach?  Imagine navigating a diverse landscape – not just a single hill, but a terrain with flat plains, steep ravines, and rocky patches all interconnected. Would you use the same stride length everywhere?

Probably not. On flat ground, you might take long, confident strides.  But approaching a steep ravine, you'd shorten your steps, becoming more cautious.  On uneven, rocky terrain, you might need to adjust your stride with every step, carefully placing your feet.

Similarly, in machine learning, the loss landscape is rarely uniform.  Different parameters in our model can have vastly different sensitivities and live in regions of the loss landscape with varying curvatures and gradient magnitudes.  Using a single, global learning rate for all parameters can be inefficient, and even detrimental.

Consider these challenges with a fixed learning rate:

*   **One-size-fits-all is rarely optimal:** A learning rate that's good for one set of parameters or one stage of training might be too large or too small for others.
*   **Oscillations vs. Slow Progress:** If the learning rate is too large, some parameters might oscillate wildly, especially those with large gradients. If it's too small, progress for other parameters, especially those with consistently small gradients, might be painfully slow.
*   **Uneven Terrain:**  In some directions of the loss landscape, the terrain might be very steep, requiring smaller steps to avoid overshooting. In other directions, it might be much flatter, allowing for larger steps to accelerate progress.

The solution? **Adaptive learning rates.**  The idea is to dynamically adjust the learning rate during optimization, adapting it to the specific characteristics of the loss landscape and the behavior of each parameter.  Instead of a single, global learning rate, we aim for **parameter-wise adaptive learning rates** – giving each parameter its own, dynamically adjusted step size.

Let's explore some key adaptive learning rate methods, starting with **AdaGrad (Adaptive Gradient Algorithm)**.

**AdaGrad: Learning from History**

AdaGrad's core idea is to adapt the learning rate for each parameter based on the **historical sum of squared gradients** for that parameter.  Parameters that have consistently received large gradients in the past are deemed to be in regions of the loss landscape that are steep or sensitive.  For these parameters, AdaGrad *decreases* the learning rate, making future steps more cautious. Conversely, parameters that have received small gradients in the past are considered to be in flatter regions, and their learning rates are kept relatively larger, allowing for faster progress.

In AdaGrad, the update rule becomes:

$$
\begin{aligned}
G_t &= \sum_{i=1}^t g_i \odot g_i  \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{aligned}
$$

Here, $$g_i$$ is the gradient at iteration $$i$$, $$\odot$$ denotes element-wise multiplication, and $$G_t$$ is a diagonal matrix where each diagonal element accumulates the sum of squared gradients for the corresponding parameter up to iteration $$t$$.  $$\epsilon$$ is a small smoothing term (e.g., 1e-8) to prevent division by zero.

Notice how the learning rate is now **parameter-specific** and **time-dependent**.  For each parameter $$j$$, the effective learning rate becomes $$\frac{\eta}{\sqrt{G_{t,jj} + \epsilon}}$$, where $$G_{t,jj}$$ is the sum of squared gradients for parameter $$j$$ up to iteration $$t$$.

AdaGrad is particularly effective in scenarios with **sparse data** or **sparse gradients**.  Features that appear infrequently will have smaller accumulated squared gradients, and thus larger learning rates, allowing them to learn more quickly when they *do* appear.

However, AdaGrad has a significant drawback: its **learning rates are constantly decreasing**, as the sum of squared gradients $$G_t$$ is always increasing.  This aggressive and ever-decreasing learning rate can sometimes cause training to **stall prematurely**, especially in deep neural networks, where training can take a very long time.

**RMSProp: Forgetting the Distant Past**

To address AdaGrad's vanishing learning rate problem, **RMSProp (Root Mean Square Propagation)** was developed.  RMSProp modifies AdaGrad by using an **exponentially decaying average** of squared gradients instead of the cumulative sum. This gives more weight to recent gradients and "forgets" gradients from the distant past.  This "fading memory" allows RMSProp to adapt to non-stationary objectives and prevent the learning rate from vanishing too quickly.

The RMSProp update rule is:

$$
\begin{aligned}
G_t &= \beta G_{t-1} + (1-\beta) g_t \odot g_t \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{aligned}
$$

Here, $$\beta$$ is a decay rate (typically around 0.9), controlling the "memory" of the squared gradient average.  A higher $$\beta$$ means a longer memory, closer to AdaGrad's behavior.

RMSProp is often more robust than AdaGrad, especially for training deep neural networks. Its adaptive learning rates can still adjust to parameter-specific sensitivities, but the fading memory prevents premature stagnation caused by overly aggressive learning rate decay.

**Adam: The Best of Both Worlds (Momentum + RMSProp)**

Finally, let's consider **Adam (Adaptive Moment Estimation)**, one of the most widely used optimizers in deep learning.  Adam can be seen as combining the best features of **momentum** (from Chapter 2) and **RMSProp**.  It uses momentum to accelerate descent and reduce oscillations, *and* it uses adaptive learning rates based on the exponentially weighted average of squared gradients, like RMSProp.

The Adam update rule is:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t  \quad \text{(Momentum term)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t \quad \text{(RMSProp-like adaptation)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(Bias correction for momentum)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(Bias correction for adaptation)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
\end{aligned}
$$

Adam maintains two exponentially moving averages:

*   $$m_t$$:  Estimate of the **first moment** of the gradient (the mean), similar to momentum.
*   $$v_t$$: Estimate of the **second moment** of the gradient (uncentered variance), used for adaptive learning rates, similar to RMSProp.

The bias correction terms $$\hat{m}_t$$ and $$\hat{v}_t$$ are used to correct for the initialization bias of these moving averages, especially in the early iterations.

Adam often works very well out-of-the-box and is a popular default choice for many deep learning tasks. It combines the benefits of both momentum and adaptive learning rates, leading to fast convergence and robustness across a wide range of architectures and datasets.

While AdaGrad, RMSProp, and Adam are among the most well-known adaptive optimizers, the field is constantly evolving.  Variants like **AMSGrad**, **AdamW**, and **LAMB** have been developed to address specific issues or improve performance in particular scenarios.

**Choosing the Right Optimizer: An Empirical Art**

It's important to remember that there is **no single "best" optimizer** that works optimally for every problem.  The choice of optimizer often becomes an empirical decision, guided by the specific characteristics of your data, model architecture, and loss landscape.

Factors to consider when choosing an optimizer:

*   **Sparsity of Data/Gradients:** AdaGrad and methods that adapt well to sparse gradients can be beneficial.
*   **Non-Stationarity of Objective:** RMSProp and Adam with their fading memory are often better suited for non-stationary objectives, common in online learning and deep learning.
*   **Computational Cost:** Adaptive methods generally have a slightly higher per-parameter computational cost than basic Gradient Descent or Momentum due to the need to maintain and update per-parameter statistics.
*   **Hyperparameter Tuning:** Adaptive methods often have more hyperparameters to tune (e.g., $$\beta_1, \beta_2$$ in Adam), although the default values often work reasonably well.

In practice, it's often a good idea to experiment with a few different optimizers (e.g., Adam, RMSProp, and Momentum) and compare their performance on your specific task.

In the next chapter, we'll shift our focus from adapting learning rates to handling a different kind of challenge: **non-smooth loss landscapes**. We'll explore techniques that go beyond gradients and allow us to optimize functions even when they are not differentiable everywhere.

---

> **Exercise: Convergence Analysis of Adaptive Gradient Methods on a Quadratic Loss**
>
> Consider the quadratic loss function
>
> $$
> L(x) = \frac{1}{2}\|x-x^\ast\|^2,
> $$
>
> where $$ x^\ast $$ is the unique minimizer.
>
> In this exercise, we analyze a simplified version of the **RMSProp** update applied to this loss.
>
> **(a)** Consider the following RMSProp-inspired update rules:
>
> $$
> \begin{aligned}
> v_{k+1} &= \beta\, v_k + (1-\beta)(x_k-x^\ast)^2, \\
> x_{k+1} &= x_k - \frac{\eta}{\sqrt{v_{k+1}+\epsilon}} (x_k-x^\ast),
> \end{aligned}
> $$
>
> where $$ \eta > 0 $$ is the nominal step size, $$ \beta \in [0,1) $$ is a decay parameter, and $$ \epsilon > 0 $$ is a small constant to prevent division by zero.
>
> Define the error as $$ e_k = x_k-x^\ast. $$  
> *Derive the recurrence relation for the error $$ e_k $$ in terms of $$ e_{k-1} $$, $$ \eta $$, $$ \beta $$, and $$ v_k $$.*
>
> **(b)** Assume that initially $$ e_0 > 0 $$ (i.e., the error remains positive) and that $$ v_0 $$ is set such that the updates are well-behaved.  
> *Under what conditions on $$ \eta $$ and $$ \beta $$ will the error sequence $$ \{e_k\} $$ decrease (i.e., $$ |e_{k+1}| < |e_k| $$) for all $$ k $$?*
>
> **(c)** The adaptive factor
>
> $$
> \frac{1}{\sqrt{v_{k+1}+\epsilon}}
> $$
>
> dynamically adjusts the effective learning rate.  
> *Discuss how this term modifies the effective step size compared to using a constant step size $$ \eta $$, and explain the role of $$ \epsilon $$ in ensuring numerical stability.*
>
> **(d)** Reflect on the following discussion points:
>
> 1. *Compare the convergence behavior of this RMSProp update with that of standard gradient descent applied to the same quadratic loss. How does the adaptive adjustment help in scenarios where the curvature of the loss might vary?*
>
> 2. *What are the potential advantages and drawbacks of having a memory term (via $$ \beta $$) that incorporates past gradient information? Consider the cases when $$ \beta $$ is set very close to 1 versus when it is significantly smaller.*
>
> 3. *Can you identify situations (for instance, in non-quadratic or non-smooth losses) where the adaptive updates might not track the “ideal” continuous-time dynamics as effectively as standard gradient descent?*
>
> *Hint:*  
> - For part **(a)**, start by substituting $$ e_k = x_k-x^\ast $$ into the update rule for $$ x_{k+1} $$ and express the new error in terms of $$ e_k $$ and $$ v_{k+1} $$.
> - For part **(b)**, consider what bounds on $$ \frac{\eta}{\sqrt{v_{k+1}+\epsilon}} $$ would guarantee contraction of the error.
> - For part **(c)**, think about how the running average $$ v_{k+1} $$ influences the effective learning rate over iterations.

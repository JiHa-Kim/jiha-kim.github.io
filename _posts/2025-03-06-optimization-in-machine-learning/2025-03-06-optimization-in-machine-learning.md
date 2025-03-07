---
layout: post
title: "Optimization in Machine Learning: From Gradient Descent to Modern Variants"
description: "A intuitive exploration of optimization theory, connecting optimization algorithms to physics-inspired frameworks and generalizations."
categories: ["Machine Learning", "Optimization"]
tags: ["gradient descent", "optimization", "Bregman divergence", "Riemannian geometry"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/GradientFlowVsDescent.gif
  alt: "Gradient flow vs. gradient descent"
date: 2025-03-06 02:45 +0000
math: true
---

## Introduction

Machine learning boils down to one core task: take a loss function, estimating how bad a model’s predictions are, and tweak the model’s parameters to make that loss as small as possible.

The go-to method for this is gradient descent (GD): start at some point in parameter space, find the direction of steepest increase (the gradient), step the opposite way, and keep going.

It’s straightforward, almost mechanical, but its story is deeper than it seems. Blindly taking the standard formulation is inefficient and leads to many problems such as poor convergence.

Instead of being handed down formulas and algorithms out of the blue, the goal of this blog post is to leverage our intuition from physical and mathematical scenarios. We explore them, simplify them, or extend them to help us better understand and construct the theory behind optmization algorithms.

What's the point of converting our formulations from machine learning into physics? Well, physics is a very extensively studied field, long before the emergence of machine learning. Hence, we can leverage a very rich collection of mathematical knowledge, intuition and results from physics to improve our understanding of optimization algorithms.

Also, as humans have evolved to adapt to the physical world, our intuition of physical phenomena has been refined over and over so that we can have a good understanding and feel for the world around us. If we can think of optimization in terms of physics, then we can picture and have a clearer sense for predicting behavior.

## Gradient Flows: Motivation
We will see that gradient descent emerges naturally as a discrete approximation to an autonomous ordinary differential equation (ODE) describing the flow of a potential function, called the gradient flow.

Let's start with a familiar physical scenario. Let's imagine that we plot the loss function's graph so that we end up with some kind of loss landscape. Then we can imagine that if we place it on the ground, it should hopefully roll toward a valley that is a local minimum. 

However, you can also see from this picture that this path might be quite inefficient.

[Ball rolling down unevenly sloped terrain]

If we imagine a very round slide, then it might take a huge detour around a small hill to end up at the endpoint, which takes extra computation unnecessarily:

[Ball on a long slide around a small hill]

Similarly, we can have concerns that it might not even overcome hills at all and get stuck in very shallow valleys, so that the loss is still very high globally speaking.

[Ball trapped in local minimum (shallow valley)]

These points are important to keep in mind, as they drive lots of the motivation behind optimization algorithms. However, for now, we'll nevertheless put these concerns aside and start our investigation with this simple scenario, because it is still a highly powerful algorithm in well-behaved situations.

### A Marble in a Bowl

Suppose our loss landscape is well-behaved enough so that there are some fairly deep bowl-shaped valleys, and the ball will not get trapped in saddle points. Then, it should be able to roll without too much trouble into one of these valleys, and give us a fairly good solution.

Let's model this movement of a particle with position $$x(t)$$ at time $$t$$ in a potential field $$V(x)$$ using Newtonian mechanics. For simplicity, we will avoid stochastic terms and model a deterministic system. Let's consider an energy-preserving system:

$$
\sum F = 0
$$

$$
m\ddot{x}(t) + \gamma \dot{x}(t) + \nabla V(x(t)) = 0
$$

where $$\dot{x}$$ denotes the time derivative of $$x$$. We have some parameters $$m$$ for the mass and $$\gamma$$ for the friction coefficient.

Let's further simplify the model. A second-order ODE would be harder to work with than a first-order one, so let's eliminate the second-order term from our model. Specifically, we look at the case of overdamped Langevin equation. In many fluid or highly viscous regimes, the inertial term $$m \ddot x(t)$$ is negligible compared to the friction term $$\gamma \dot x(t)$$. Dropping this acceleration leads to

$$\dot x(t) = -\frac{1}{\gamma}\nabla V(x(t))$$

This indeed kind of looks like our familiar gradient descent algorithm. Let's set $$\gamma=1$$ for simplicity, and substitute the loss $$L(x(t))$$ for the potential:

$$\frac{dx(t)}{dt} = -L(x(t))$$

This equation is referred to as the **gradient flow ODE**. 

To see the connection, consider discretizing the continuous gradient flow with a forward Euler scheme. Given a small time step $$ \eta $$, we approximate:

$$
\dot{x}(t) \approx \frac{x(t+\eta) - x(t)}{\eta}.
$$

Substituting into the gradient flow equation:

$$
\frac{x(t+\eta) - x(t)}{\eta} = -\nabla L(x(t)) \quad \Longrightarrow \quad x(t+\eta) = x(t) - \eta \nabla L(x(t)).
$$

This is exactly the gradient descent update rule we started with. By taking the limit as $$ \eta \to 0 $$, we recover the continuous-time dynamics, which gives us a new perspective to study.

> **Exercise: Deriving Gradient Descent from Gradient Flow**  
> Starting from the gradient flow ODE  
> 
> $$
> \frac{dx(t)}{dt} = -\nabla L(x(t)),
> $$ 
> 
> use the forward Euler discretization
> 
> $$
> \frac{x(t+\eta) - x(t)}{\eta} \approx -\nabla L(x(t))
> $$
> 
> to derive the standard gradient descent update
> 
> $$
> x_{k+1} = x_k - \eta \nabla L(x_k).
> $$
> 
> *Hint:* Thinking in terms of numerical simulations for ODEs, explain each step and discuss the role of the step size $$ \eta $$ in controlling the approximation accuracy.

### A Concrete Example: Mean Squared Error (MSE)

> **Exercise: Exponential Convergence in a Quadratic Bowl**  
> Consider the mean squared error loss
> 
> $$
> L(x) = \frac{1}{2}\|x-x^\ast\|^2,
> $$
> 
> where $$ x^\ast $$ is the unique minimizer.  
> **(a)** Show that the gradient flow
> 
> $$
> \dot{x}(t) = -(x(t)-x^\ast)
> $$
> 
> has the solution
> 
> $$
> x(t) = x^\ast + (x(0)-x^\ast)e^{-t}.
> $$
> 
> **(b)** Explain why this solution demonstrates exponential convergence to the minimizer.
> > **Exponential Convergence:**  
> > A sequence $$ \{x(t)\} $$ or trajectory $$ x(t) $$ is said to converge exponentially to a limit $$ x^\ast $$ if there exist constants $$ C > 0 $$ and $$ \alpha > 0 $$ such that
> > 
> > $$
> > \|x(t) - x^\ast\| \leq C e^{-\alpha t} \quad \text{for all } t \geq 0.
> > $$ 
> **(c)** Now consider the discrete gradient descent update with a fixed step size \( \eta \):
> 
> \[
> x_{k+1} = x_k - \eta (x_k - x^\ast).
> \]
> 
> Show that the error evolves as
> \[
> 
> \|x_{k} - x^\ast\| = |1-\eta|^k \|x_0-x^\ast\|,
> \]
> 
> and deduce the condition on \( \eta \) (in terms of its magnitude) under which the discrete update converges exponentially.  
> 
> **(d)** Reflect on the following questions:  
> 1. How does the convergence rate \( |1-\eta| \) compare to the continuous rate \( e^{-1} \) when \( \eta \) is small?  
> 2. What are the potential pitfalls if \( \eta \) is chosen too large or too small in the discrete case?  
> 3. Can you identify scenarios where the discrete updates may fail to mimic the continuous dynamics, even if the continuous gradient flow converges exponentially?  
>  
> *Hint:* Consider evaluating the discrete convergence factor \( |1-\eta| \) for different choices of \( \eta \), and compare these values with the ideal continuous decay rate \( e^{-1} \) over a unit time interval.

We might also be worried about the convergence of the gradient flow ODE into a saddle point, but we can fortunately demonstrate that this is improbable.

> **Exercise: Investigating the Instability of Saddle Points**  
> Consider a twice-differentiable function \( L: \mathbb{R}^n \to \mathbb{R} \) and let \( x^\ast \) be a critical point (i.e., \( \nabla L(x^\ast) = 0 \)). Suppose that the Hessian \( H = \nabla^2 L(x^\ast) \) has both positive and negative eigenvalues, meaning \( x^\ast \) is a saddle point.  
>
> **(a)** Linearize the gradient flow dynamics around \( x^\ast \) by writing  
> \[
> \dot{y}(t) = -H\,y(t),
> \]  
> where \( y(t) = x(t) - x^\ast \). Explain how the eigenvalues of \( H \) influence the behavior of \( y(t) \).  
>
> **(b)** Show that even though \( \nabla L(x^\ast) = 0 \), a small perturbation in the direction corresponding to a negative eigenvalue of \( H \) will grow over time, thereby illustrating the instability of the saddle point under gradient flow.  
>
> **(c)** Discuss why the standard gradient descent update  
> \[
> x_{k+1} = x_k - \eta \nabla L(x_k)
> \]  
> might exhibit erratic behavior when initialized near a saddle point, and how this contrasts with the behavior near a local minimum.
>
> *Hint:* Consider the exponential behavior \( y(t) \approx e^{-\lambda t} y(0) \) in each eigendirection and relate this to the choice of step size \( \eta \) in the discrete case.

## Rigorous Foundations: Lyapunov Stability

Now, this playing around is fun, but are we really justified in creating this continuous formulation? Let's take a more theoretical perspective through Lyapunov’s Stability Theorem.

To understand why our gradient flow leads to convergence, we turn to **Lyapunov’s Stability Theorem**. This theorem provides a systematic way to prove that an equilibrium point of a dynamical system is stable by constructing an appropriate Lyapunov function. In our context, the equilibrium is the minimum $$ x^\ast $$ of the loss function $$ L(x) $$.

### Lyapunov Functions: The Core Idea

A Lyapunov function $$ V(x) $$ is a scalar function that serves as an "energy measure" for the system. For a system described by

$$
\dot{x}(t) = f(x(t)),
$$

a function $$ V : \mathbb{R}^n \to \mathbb{R} $$ is a Lyapunov function if it satisfies:

1. **Positive Definiteness:**

   $$
   V(x) > 0 \quad \text{for all } x \neq x^\ast \quad \text{and} \quad V(x^\ast) = 0.
   $$

2. **Negative Definiteness of the Derivative:**
   
   $$
   \dot{V}(x) = \nabla V(x)^\top f(x) < 0 \quad \text{for all } x \neq x^\ast.
   $$

If these conditions hold, the equilibrium $$ x^\ast $$ is stable, and trajectories $$ x(t) $$ converge toward $$ x^\ast $$.

### Constructing a Lyapunov Function for Gradient Flow

For our gradient flow:

$$
\dot{x}(t) = -\nabla L(x(t)),
$$

a natural candidate is the loss function itself (or a shifted version). Define

$$
V(x) = L(x) - L(x^\ast),
$$

where $$ x^\ast $$ is the minimizer of $$ L $$. Notice that $$V(x)$$ has the same differentiability properties as $$L(x)$$.

> **Exercise:** Assuming that $$x*$$ is unique, show that $$V(x)$$ is a valid Lyapunov function for the gradient flow.

## A Majorization-Minimization Perspective For Convex Lipschitz Smooth Functions

An alternative way to understand gradient descent is through the lens of **majorization-minimization (MM)**. The MM framework seeks to solve a complex optimization problem by instead iteratively minimizing a surrogate (upper bound) that is easier to handle. 

### The Variational Problem

Assume that the loss function $$ L(x) $$ is convex and $$\lambda$$-Lipschitz smooth, i.e.:

> A function $$L: \mathbb{R}^d \to \mathbb{R}$$ is $$\lambda$$-Lipschitz smooth if for all $$x,y\in \mathbb{R}^d$$, the gradient satisfies
> 
> $$
> \|\nabla L(y) - \nabla L(x)\| \le \lambda \|y-x\|.
> $$

This condition implies the following upper bound. For any $$ x $$ and $$ y $$ in the domain, we have:

$$
L(y) \leq L(x) + \nabla L(x)^\top (y-x) + \frac{\lambda}{2}\|y-x\|^2.
$$

The proof is in the [appendix](#appendix-compact-proof-of-the-quadratic-upper-bound). The right-hand side of the inequality acts as an upper bound (or surrogate) for $$ L(y) $$ around the point $$ x $$. This quadratic function is tight at $$ y = x $$ and easy to minimize with respect to $$ y $$.

### Formulating the Surrogate Minimization

At the current iterate $$ x_k $$, we define the surrogate function:

$$
Q(y; x_k) = L(x_k) + \nabla L(x_k)^\top (y-x_k) + \frac{\lambda}{2}\|y-x_k\|^2.
$$

Since the inequality above holds for all points $$y$$, it should also hold at the minimum: $$\arg\min_y L(y) \le \arg\min_y Q(y; x_k)$$.

The Majorization-Minimization (MM) principle suggests that instead of minimizing $$ L(y) $$ directly, we can minimize this upper bound:

$$
x_{k+1} = \arg\min_{y} Q(y; x_k).
$$

Intuitively, think of our loss landscape again. This surface might be somewhat jagged and annoying to work with. Instead, majorization-minimization tells us "let's lay a nicely shaped and artificially constructed tarp lying strictly above the landscape". In such a way, if our choice is fairly good, then since the tarp traps the landscape below, descending it will ensure that the loss is at least below it. How well this works depends on the sharpness of our bound (closeness of our tarp).

[Example majorization through surrogate function]

### Solving the Variational Problem

To find the minimizer, we differentiate $$ Q(y; x_k) $$ with respect to $$ y $$ and set the derivative to zero. For simplicity, let's start with the case of the $$\ell_2$$ norm $$\|x\|_2 = \sqrt{x^\top x}$$:

$$
\nabla_y Q(y; x_k) = \nabla L(x_k) + \lambda (y-x_k) = 0.
$$

Solving for $$ y $$ yields:

$$
y = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

Thus, the update rule becomes:

$$
x_{k+1} = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

This is exactly the gradient descent update with a fixed learning rate $$ \eta = \frac{1}{\lambda} $$.

## Proximal Methods: Generalizing to Non-Differentiable Losses

So far, we derived the familiar gradient descent update rule by discretizing the continuous gradient flow using the forward Euler method. However, the **backward Euler** method offers distinct advantages that are especially relevant in machine learning, while maintaining a single evaluation point per update.

### The Backward Euler Discretization

Recall our gradient flow ODE:

> Given a differentiable $$L: \mathbb{R}^d \to \mathbb{R}$$, the continuous-time dynamics are given by the gradient flow ODE
> $$
> \frac{dx(t)}{dt} = -\nabla L(x(t)).
> $$

Instead of approximating the derivative at the current iterate, backward Euler evaluates it at the future point:

$$
\frac{x_{k+1} - x_k}{\eta} = -\nabla L(x_{k+1}).
$$

Rearranging, we obtain the implicit update:

$$
x_{k+1} = x_k - \eta \, \nabla L(x_{k+1}).
$$

This implicitness means that the new iterate $$ x_{k+1} $$ is defined through an equation that involves itself, making the method more stable—especially in stiff or rapidly changing regions of the loss landscape.

In practice, it is of course harder to compute this, $$x_k$$ is dependent on itself here. Hence, you have to solve a nonlinear equation to get $$x_{k+1}$$. However, as we are about to see, it is possible to transform this into a more tractable form to serve for theoretical analysis.

### Variational Formulation and Regularization

The backward Euler update, after some similar re-arrangement, can be interpreted as the first-order optimality condition of the minimization problem

$$
x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}.
$$

Here, the quadratic term $$\frac{1}{2\eta}\|y - x_k\|^2$$ serves two important roles:

1. **Regularization:** In machine learning, adding regularization terms is a common strategy to avoid overfitting and to stabilize optimization. This quadratic term naturally penalizes large deviations from the current iterate $$ x_k $$, thereby acting as an implicit regularizer. It forces the update to remain close to $$ x_k $$ unless the loss $$ L(y) $$ strongly suggests a different direction. This is particularly advantageous when dealing with noisy data or highly non-linear loss landscapes.

2. **Stability:** By preventing large steps, this term also dampens oscillations in regions where the loss function changes rapidly. In essence, it “smooths out” the update dynamics, much like adding momentum or using adaptive step sizes in other optimization methods.

Notice that this form does not contain the gradient of $$L$$ anywhere! This is because the gradient is implicitly included in the minimization problem.

We assumed throughout the previous examples that $$L$$ behaved nicely, namely that it should be differentiable. But this is not always the case: take, for instance

$$\text{ReLU}(x) := \max(0, x).$$

This function is extensively used as an activation function for multi-layer perceptrons (a.k.a. feed-forward neural networks). It is not differentiable at $$x=0$$.

### Extension to Composite Optimization: Proximal Methods

The variational interpretation paves the way to tackle composite optimization problems, where the loss can be split as

$$
L(x) = f(x) + g(x),
$$

with $$ f(x) $$ being smooth and $$ g(x) $$ possibly non-smooth (often representing regularization, such as an $$ \ell_1 $$ penalty).

Using backward Euler ideas, we can decouple the treatment of the smooth and non-smooth parts. First, perform a gradient step on the smooth function:

$$
v = x_k - \eta \, \nabla f(x_k),
$$

and then apply a proximal step to incorporate the non-smooth regularizer $$ g(x) $$:

$$
x_{k+1} = \operatorname{prox}_{\eta g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
$$

This minimized quantity is referred to as the **Moreau envelope**.

> **Proximal Mapping:**  
> Given a proper, lower semicontinuous, convex function $$ g: \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ and a parameter $$ \eta > 0 $$, the proximal mapping of $$ g $$ is defined as  
> $$
> \operatorname{prox}_{\eta g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|^2 \right\}.
> $$  
> This operator finds a point $$ y $$ that balances minimizing $$ g $$ while remaining close to $$ v $$.

> **Moreau Envelope:**  
> The Moreau envelope of a function $$ g $$ with parameter $$ \eta > 0 $$ is given by  
> $$
> M_{\eta g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|^2 \right\}.
> $$  
> It provides a smooth approximation of $$ g $$, and its gradient is closely related to the proximal mapping, making it a powerful tool in optimization.

### Gradient Flows: Applications

Reformulating the steepest descent scenario into a continuous setting allows for a whole new branch of studying and development of theory in machine learning algorithms. The following is a short list of some interesting perspectives on machine learning that emerge from it.

- [Chen et al. (2020) - Better Parameter-free Stochastic Optimization with ODE Updates for Coin-Betting](https://arxiv.org/abs/2006.07507)
- [Sharrock and Nemeth (2023) - Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates](https://arxiv.org/abs/2301.11294)
- [Wibisono et al. (2016) - A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245)
- [Chen and Ewald (2024) - Gradient flow in parameter space is equivalent to linear interpolation in output space](https://arxiv.org/abs/2408.01517v1)
- [Romero and Benosman (2019) - Finite-Time Convergence of Continuous-Time Optimization Algorithms via Differential Inclusions](https://arxiv.org/abs/1912.08342)
- [Zhang et al. (2020) -  A Hessian-Free Gradient Flow (HFGF) Method for the Optimisation of Deep Learning Neural Networks](https://wenyudu.github.io/publication/hfgf_preproof.pdf)


## Appendix

### Appendix: Compact Proof of the Quadratic Upper Bound

Let $$ L: \mathbb{R}^n \to \mathbb{R} $$ be differentiable and $$\lambda$$-Lipschitz smooth, so that for any $$ x,y $$:

$$
\|\nabla L(y) - \nabla L(x)\| \leq \lambda \|y-x\|.
$$

**Goal:** Show that

$$
L(y) \leq L(x) + \nabla L(x)^\top (y-x) + \frac{\lambda}{2}\|y-x\|^2.
$$

**Proof:**

1. Define $$\phi(t) = L(x+t(y-x))$$ for $$ t \in [0,1] $$. Then $$\phi(0)=L(x)$$ and $$\phi(1)=L(y)$$.

2. By the chain rule,
   $$
   \phi'(t) = \nabla L(x+t(y-x))^\top (y-x).
   $$
   Using the Fundamental Theorem of Calculus:
   $$
   L(y)-L(x)=\int_0^1 \phi'(t) \, dt = \int_0^1 \nabla L(x+t(y-x))^\top (y-x) \, dt.
   $$

3. Add and subtract $$\nabla L(x)^\top (y-x)$$:
   $$
   L(y)-L(x)=\nabla L(x)^\top (y-x)+\int_0^1 \Bigl(\nabla L(x+t(y-x))-\nabla L(x)\Bigr)^\top (y-x) \, dt.
   $$

4. By the Lipschitz condition and Cauchy-Schwarz, for each $$ t $$:
   $$
   \left|\Bigl(\nabla L(x+t(y-x))-\nabla L(x)\Bigr)^\top (y-x)\right| \leq \lambda t\,\|y-x\|^2.
   $$
   Therefore,
   $$
   \int_0^1 \Bigl(\nabla L(x+t(y-x))-\nabla L(x)\Bigr)^\top (y-x) \, dt \leq \lambda \|y-x\|^2 \int_0^1 t\, dt = \frac{\lambda}{2}\|y-x\|^2.
   $$

5. Combining the results:
   $$
   L(y)-L(x) \leq \nabla L(x)^\top (y-x) + \frac{\lambda}{2}\|y-x\|^2.
   $$
   Rearranging gives the desired inequality.


## Further Reading

- [Kundu (2024) - Who's Adam and What's He Optimizing? | Deep Dive into Optimizers for Machine Learning!](https://www.youtube.com/watch?v=MD2fYip6QsQ)
- [Bernstein and Newhouse (2024) - Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
- [Zhang and Nemeth (2024) - Why Should We Care About Gradient Flows?](https://shusheng3927.github.io/posts/2024-09-13-WGF/)
- [Zhang (2024) - Gradient Flow and Its Applications in Statistical Learning](https://shusheng3927.github.io/files/grad_flow.pdf)
- [Orabona (2023) - A Modern Introduction to Online Learning](https://arxiv.org/abs/1912.13213)
- [Bach (2019) - Effortless optimization through gradient flows](https://francisbach.com/gradient-flows/)
- [Nielsen (2021) - Bregman divergences, dual information geometry, and generalized comparative convexity](https://franknielsen.github.io/BregmanDivergenceDualIGGenConvexity-25Nov2021.pdf)
- [Banerjee et al. (2005) - On the Optimality of Conditional Expectation as a Bregman Predictor](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1459065)
- [Mehta (2023) - Introduction to Online Learning (CSC 482A/581A)- Lecture 6](https://web.uvic.ca/~nmehta/online_learning_spring2023/lecture6.pdf)
- [Wikipedia - Bregman Divergence](https://en.wikipedia.org/wiki/Bregman_divergence)
- [Schiebinger (2019) -  Gradient Flow in Wasserstein Space](https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture16.pdf)
- [Fatir (2020) - Introduction to Gradient Flows in the 2-Wasserstein Space](https://abdulfatir.com/blog/2020/Gradient-Flows/)
- [Wibisono et al. (2016) - A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245)
- [Figalli (2022) - AN INTRODUCTION TO OPTIMAL TRANSPORT
  AND WASSERSTEIN GRADIENT FLOWS](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf)
- [d’Aspremont et al. (2021) - Acceleration Methods](https://arxiv.org/abs/2101.09545)
- [Xu (2024) - Gradient Flows: Modeling and Numerical
 Methods](https://www.birs.ca/iasm-workshops/2024/24w5504/files/10.24-01%20Chuanju%20Xu%20-%20for%20sharing.pdf)

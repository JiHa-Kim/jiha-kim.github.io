---
title: "Crash Course Cheat Sheet: Numerical Analysis for Optimization"
date: 2025-05-18
sort_index: 999
mermaid: false
description: "A quick reference guide for key concepts in numerical analysis relevant to optimization, covering ODE solvers and numerical linear algebra."
image: # placeholder
categories:
- Numerical Analysis
- Mathematical Optimization
tags:
- Cheat Sheet
- ODE Solvers
- Numerical Linear Algebra
- Optimization Methods
- Gradient Descent
- Newton's Method
- Condition Number
- Preconditioning
- Iterative Methods
---

This cheat sheet summarizes key concepts from Numerical Analysis relevant to understanding and developing optimization algorithms in Machine Learning. It covers essential topics from numerical methods for Ordinary Differential Equations (ODEs) and Numerical Linear Algebra (NLA).

## Numerical Methods for Ordinary Differential Equations (ODEs)

| Concept / Method                      | Key Formula / Description                                                                                                                          | Relevance to Optimization / Key Insight                                                                                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ODE & Initial Value Problem (IVP)** | ODE: $$\frac{dy}{dt} = f(t, y(t))$$. IVP: with $$y(t_0)=y_0$$.                                                                                     | Models continuous opt. paths like Gradient Flow ($$\dot{\theta} = -\nabla L(\theta)$$) or Heavy Ball ODE. Opt. algos are discretizations.                                  |
| **Discretization & Finite Diff.**     | Approx. solution at $$t_n$$ with step $$h$$. Fwd Diff: $$y'(t_n) \approx \frac{y_{n+1}-y_n}{h}$$.                                                  | Foundation for turning continuous ODE models into iterative algorithms. Step size $$h$$ often maps to learning rate $$\alpha$$.                                            |
| **Explicit Euler Method**             | $$y_{n+1} = y_n + h f(t_n, y_n)$$                                                                                                                  | Gradient Descent ($$\theta_{k+1} = \theta_k - \alpha \nabla L(\theta_k)$$) is Explicit Euler on Gradient Flow. 1st order global error ($$O(h)$$).                          |
| **Implicit Euler Method**             | $$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$                                                                                                          | Often better stability (e.g., A-stable), allowing larger steps. Requires solving for $$y_{n+1}$$ at each step.                                                             |
| **Stability of Numerical Methods**    | Errors don't cause divergence. Conditional (e.g., Explicit Euler: $$h$$ must be small) vs. Unconditional (e.g., Implicit Euler for some problems). | Explains why GD diverges if learning rate $$\alpha$$ ($$\approx h$$) is too large. Relates to max stable $$\alpha$$ (e.g., $$< 2/\lambda_{\text{max}}(H)$$ for quadratic). |
| **Systems & Higher-Order ODEs**       | Convert $$k^{\text{th}}$$-order ODE to system of $$k$$ first-order ODEs: $$\mathbf{\dot{z}} = \mathbf{F}(t, \mathbf{z})$$. Solve component-wise.   | Heavy Ball ODE ($$m\ddot{\theta} + \gamma \dot{\theta} + \nabla L(\theta) = 0$$) for momentum is 2nd order, converted to a system. Discretization yields momentum updates. |
| **Linear Multistep Methods (LMMs)**   | Use multiple past steps: $$\sum_{j=0}^{k} a_j y_{n+j} = h \sum_{j=0}^{k} b_j f(t_{n+j}, y_{n+j})$$.                                                | Polyak's momentum ($$\theta_{k+1} = \theta_k + \mu(\theta_k - \theta_{k-1}) - \eta \nabla L(\theta_k)$$) can be seen as a 2-step LMM.                                      |

## Numerical Linear Algebra (NLA) for Optimization

| Concept / Method                      | Key Formula / Description                                                                                                                               | Relevance to Optimization / Key Insight                                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Condition Number $$\kappa(A)$$**    | $$\kappa(A) = \Vert A \Vert \Vert A^{-1} \Vert$$. For SPD Hessian $$H$$, $$\kappa_2(H) = \frac{\lambda_{\text{max}}(H)}{\lambda_{\text{min}}(H)}$$.     | Measures problem sensitivity. High $$\kappa(H)$$ means ill-conditioned loss surface (long, narrow valleys), slows GD convergence. Factor $$\approx ((\kappa-1)/(\kappa+1))^2$$.                                 |
| **Solving Linear Systems $$Ax=b$$**   | Direct methods (LU, Cholesky for SPD; $$O(n^3)$$) for small/dense $$A$$. Iterative methods (CG, GMRES) for large/sparse $$A$$.                          | Core of Newton-type methods (solve $$H_k \Delta\theta_k = -\nabla L(\theta_k)$$). Iterative methods are crucial for large-scale optimization.                                                                   |
| **Conjugate Gradient (CG) Method**    | Iterative solver for SPD $$Ax=b$$. Generates $$A$$-conjugate search directions $$p_k$$. Requires matrix-vector products ($$Ap_k$$).                     | Solves Newton systems $$H_k \Delta\theta_k = -\nabla L(\theta_k)$$ efficiently without forming/inverting $$H_k$$. Convergence rate depends on $$\sqrt{\kappa(A)}$$; much faster than GD for ill-cond. problems. |
| **Preconditioning**                   | Transform $$Ax=b$$ to $$M^{-1}Ax = M^{-1}b$$ (or similar). $$M \approx A$$ and $$M^{-1}r$$ is easy to compute. Goal: $$\kappa(M^{-1}A) \ll \kappa(A)$$. | Speeds up iterative solvers like CG by improving effective condition number. Preconditioned CG (PCG) for faster Newton steps.                                                                                   |
| **Types of Preconditioners**          | Diagonal (Jacobi): $$M = \text{diag}(A)$$. Incomplete Cholesky (IC). Structured: Block-diag, Kronecker (e.g., K-FAC, Shampoo).                          | Diagonal preconditioning is the basis for adaptive methods (AdaGrad, RMSProp, Adam). Structured preconditioners for advanced optimizers.                                                                        |
| **Newton's Method (in Optimization)** | Step: $$\Delta\theta_k = -[H_k]^{-1} \nabla L(\theta_k)$$. Solves Hessian system $$H_k \Delta\theta_k = -\nabla L(\theta_k)$$.                          | Uses 2nd-order (curvature) info for potentially quadratic convergence. System usually solved with (P)CG for large problems.                                                                                     |
| **Quasi-Newton Methods**              | Approximate Hessian $$B_k \approx H_k$$ or inverse $$C_k \approx H_k^{-1}$$ using gradient info (e.g., BFGS). L-BFGS for large scale (limited memory).  | Avoids explicit Hessian computation/storage. L-BFGS uses past $$s_k, y_k$$ vectors to implicitly compute $$-C_k \nabla L(\theta_k)$$.                                                                           |
| **Adaptive Optimizers (Adam, etc.)**  | E.g., Adam: $$\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$. The term $$1/\sqrt{v_t}$$ scales learning rate per parameter.       | $$v_t$$ (running avg of squared gradients) acts as a diagonal estimate of preconditioning, attempting to normalize gradient steps. Improves conditioning locally.                                               |

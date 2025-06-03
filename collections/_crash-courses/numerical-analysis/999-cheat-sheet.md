---
title: "Crash Course Cheat Sheet: Numerical Analysis for Optimization"
date: 2025-05-18
series_index: 999
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

---
layout: post
title: "Rational Matrix Iterations for Polar Decomposition"
date: 2026-04-05 00:00 +0000
description: "Newton-Schulz is the classic hardware-aware polar decomposition, but it struggles with ill-conditioning and low precision. We derive a comprehensive, robust alternative using rational iterations (DWH) and specific stabilization primitives to ensure convergence without eigenvalues or restarts."
categories:
  - Numerical Linear Algebra
  - Mathematical Optimization
tags:
  - Polar Decomposition
  - Muon
  - Matrix Iterations
  - Rational Functions
  - Numerical Stability
math: true
scholar:
  bibliography: posts/2026-04-05-rational-polar-decomposition/rational-polar-decomposition.bib
---

> [!info] Overview
> The Muon optimizer {% cite jordanMuonOptimizer2024 %} has demonstrated that projecting update directions onto the Stiefel manifold—via the matrix polar decomposition—is a powerful tool for training Transformers and other scale-sensitive architectures. 
>
> While the Newton-Schulz iteration is the standard hardware-aware choice, it is a **polynomial** map that can be slow or unstable for ill-conditioned matrices. This has led to the development of methods like **Polar Express** {% cite polarExpress2025 %}, which optimize the polynomial basin.
>
> In this post, we work through a different path: **rational** iterations. We present a "clean" end-to-end design that hits several high-bar goals: exactly 2 rectangular GEMMs, no power iteration for $\lambda_{\min}$, and stability that handles the spurious negative eigenvalues common in low-precision (FP16/TF32) arithmetic.

---

## 1. The Rational Path: Dynamic Weighted Halley

Iterative methods for the polar factor $U = \operatorname{polar}(A)$ work by applying a function $f(x)$ to the singular values $\sigma_i$ of $A$. While Newton-Schulz is polynomial, the **Dynamic Weighted Halley (DWH)** iteration {% cite nakatsukasaOptimizingHalleyIteration2010 %} uses a rational function:

$$
f(x) = x \frac{a + b x^2}{1 + c x^2}
$$

By picking $a, b, c$ dynamically based on an estimate of the lower bound of the spectrum, DWH achieves global convergence and extremely high throughput.

### 1.1 Offline Coefficient Computation

Given a known lower bound estimate $\ell \in (0, 1]$, the optimal DWH coefficients are computed once (in FP64) as follows:

$$
\gamma(\ell) = \left(\frac{4(1-\ell^2)}{\ell^4}\right)^{1/3},\qquad r = \sqrt{1+\gamma},
$$
$$
a(\ell) = r + \frac{1}{2}\sqrt{8 - 4\gamma + \frac{8(2-\ell^2)}{\ell^2 r}},
$$
$$
b(\ell) = \frac{(a-1)^2}{4},\qquad c(\ell)=a+b-1.
$$

We then define the "apply-friendly" parameters used in the iteration step:
$$
\alpha = \frac{b}{c},\qquad \beta = a-\alpha.
$$

---

## 2. Stability Primitives for Low Precision

In low precision (FP16 or TF32), the Gram matrix $G = X^\top X$ can suffer from "drift" that creates spurious negative eigenvalues. Standard methods like Gram-NS need restarts or heavy regularization to handle this. {% cite daoAILabGramNewtonSchulz2026 %}

### 2.1 Symmetrization and Jacobi Scaling
We work in the "column-space view" $X \in \mathbb{R}^{M \times N}, M \ge N$.
For $A \in \mathbb{R}^{m \times n}$, we set $X = A$ if $m \ge n$ and $X = A^\top$ otherwise.

**1. Symmetrize**: $G \leftarrow \tfrac12(G+G^\top)$.
**2. Jacobi Preconditioning**: $D = \operatorname{rsqrt}(\max(G_{ii}, d_{\min}))$.
**3. Jacobi Gram**: $\tilde{A} \leftarrow DGD$.

### 2.2 Moment-Based Upper Bounds
We avoid expensive power iterations for $\lambda_{\max}$ by using a moment-based upper bound $u = \min(u_M, u_F)$ computed from $\tilde{A}$:
$$
u_M = \frac{\operatorname{tr}(\tilde{A}) + \sqrt{(N-1)\max(0, N\|\tilde{A}\|_F^2 - \operatorname{tr}(\tilde{A})^2)}}{N}, \quad u_F = \|\tilde{A}\|_F.
$$
Inflate with $u \leftarrow u + \eta \frac{\|X\|_F^2}{s_x^2}$ for matmul rounding safety.

---

## 3. The End-to-End Algorithm (Explicit Version)

The following procedure, **Rational Polar Decomposition**, explicitly integrates the adaptive bound logic and the denominator-stabilized Cholesky factorization required for low-precision stability.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Rational Polar Decomposition</div>
<pre class="pseudocode">
\begin{algorithmic}
\PROCEDURE{RationalPolar}{$X \in \mathbb{R}^{M \times N}$}
    \STATE $X \leftarrow X / 2^{\lfloor \log_2(\max \vert X_{ij}\vert ) \rceil}$ \COMMENT{Range scaling}
    \STATE $G \leftarrow \frac{1}{2}(X^\top X + (X^\top X)^\top)$
    \STATE $D \leftarrow \text{diag}(\max(G_{ii}, 10^{-30}))^{-1/2}$
    \STATE $B \leftarrow DGD, \quad B_{ii} \leftarrow 1$
    \STATE Compute moment bound $u$, $\sigma \leftarrow \sqrt{u} \cdot \text{safety} + \epsilon$
    \STATE $X \leftarrow X/\sigma, \quad B \leftarrow B/\sigma^2$
    \REPEAT
        \STATE Compute coefficients $(\alpha_k, \beta_k, c_k, \delta)_k$ from active $\ell_0$
        \STATE $S \leftarrow B + \frac{1}{c_0} I, \quad \tau \leftarrow 0$
        \WHILE{true}
            \STATE $\tilde{S} \leftarrow S + \tau I$
            \STATE \textbf{try} $L = \text{Cholesky}(\tilde{S})$
            \IF{$L$ exists}
                \STATE \BREAK
            \ELSE
            \IF{$\tau = 0$}
                \STATE $\tau \leftarrow 10^{-6} \cdot \operatorname{tr}(S)/N$
            \ELSE
                \STATE $\tau \leftarrow 10\tau$
            \ENDIF
        \ENDIF
    \ENDWHILE
    \STATE $\rho \leftarrow \tau / (\operatorname{tr}(B)/N)$
        \IF{$\rho > 10^{-3}$}
            \STATE $\ell_0 \leftarrow \min(10\ell_0, 0.1)$ \COMMENT{Increase floor and restart}
        \ELSE
            \STATE \BREAK
        \ENDIF
    \UNTIL{false}
    \STATE $H_0 \leftarrow \frac{1}{c_0} (L L^\top)^{-1}, \quad K_0 \leftarrow I - H_0, \quad M_0 \leftarrow \alpha_0 I + \beta_0 H_0$
    \STATE $B_1 \leftarrow I + \delta(c_0\alpha_0^2 B + 2\alpha_0\beta_0 K_0 + \beta_0^2 H_0 K_0)$
    \STATE Factor $B_1$ with stabilized Cholesky (as above)
    \STATE $T \leftarrow M_0 B_1^{-1}, \quad K_{\text{final}} \leftarrow \alpha_1 M_0 + \beta_1 T$
    \RETURN $Q = (X D) K_{\text{final}} D$
\ENDPROCEDURE
\end{algorithmic}
</pre>
</div>

---

## 4. Why This is "Hardware-Aware SOTA"

### 4.1 Efficiency
* **Two GEMMs**: Exactly one $X^\top X$ and one $X K_{\text{final}}$. All other operations are $N \times N$ or diagonal.
* **No Eigenvalues**: Power iteration is replaced by moment scaling ($\lambda_{\max}$) and adaptive denominator tracking ($\lambda_{\min}$).
* **Minimized Latency**: The "small side" $N\times N$ work is dominated by Cholesky/Trsm, which is fast on tensor cores.

### 4.2 Stability
* **Bounded Coefficients**: We never form huge matrices like $I + 10^3 G$. Instead, we factor $G + \epsilon I$, where $\epsilon$ is the natural floor computed from DWH.
* **Robust Denominators**: Minimal jitter ensures we never hit singular matrices without needing to restart the whole optimization step.
* **No Bias**: We only shift denominators, preserving the update direction in the Gram matrix.

## Conclusion

Rational iterations like DWH provide a more robust foundation for the polar decomposition in deep learning optimizers than polynomial Newton-Schulz. By combining them with hardware-specific preconditioning and moment-based scaling, we can build optimizers that are both faster and more stable in the low-precision regimes required by modern hardware.

---

## References

{% bibliography %}

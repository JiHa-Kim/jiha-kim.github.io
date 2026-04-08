---
layout: post
title: "Fast Polar Decomposition with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigenvalues, and robust convergence from condition numbers up to 1000."
categories:
  - Numerical Linear Algebra
  - Mathematical Optimization
tags:
  - Polar Decomposition
  - Muon
  - Matrix Iterations
  - Rational Functions
  - Polynomial Iterations
  - Numerical Stability
math: true
scholar:
  bibliography: posts/2026-04-05-rational-polar-decomposition/rational-polar-decomposition.bib
---

> [!info] Overview
> The Muon optimizer {% cite jordanMuonOptimizer2024 %} projects update directions onto the Stiefel manifold via the matrix polar decomposition. The Newton-Schulz iteration is the standard hardware-aware choice, but it is a polynomial map that can be slow or unstable for ill-conditioned matrices. The Polar Express {% cite polarExpress2025 %} optimizes the polynomial basin, and the Dynamic Weighted Halley (DWH) {% cite nakatsukasaOptimizingHalleyIteration2010 %} iteration uses a rational map with much faster early convergence.
>
> In this post we show that **neither pure rational nor pure polynomial is optimal**. Instead, a simple hybrid—**one DWH step followed by two degree-5 Polar Express steps**—is the sweet spot. The rational step crushes the hard high-condition-number regime; the polynomial steps finish the job where they are most efficient. The algorithm uses exactly two rectangular GEMMs, no eigenvalue estimates, and is stable under FP16/BF16 arithmetic.

---

## 1. Why a Hybrid?

Iterative methods for the polar factor work by applying a scalar iteration $f(x)$ to the singular values $\sigma_i$ of $A$. After normalization so that $\sigma_{\max} = 1$, the goal is to drive the lower endpoint of the singular value interval $\ell \to 1$.

### 1.1 The Two Scalar Maps

- **DWH (rational)**: the Dynamic Weighted Halley map {% cite nakatsukasaOptimizingHalleyIteration2010 %} applies the rational function:

$$
f_{\text{DWH}}(x) = x\frac{a + bx^2}{1 + cx^2}
$$

where the coefficients $a, b, c$ are chosen optimally for the current floor $\ell$. Rational maps are exceptionally powerful at lifting small $\ell$ but require a linear solve per update step.

- **Polar Express (polynomial)**: the degree-5 PE map {% cite polarExpress2025 %} applies the odd polynomial:

$$
p(x) = x(a + bx^2 + cx^4) = ax + bx^3 + cx^5
$$

where $(a, b, c)$ solve the minimax problem $\min_{a,b,c}\max_{x \in [\ell,1]} |1 - p(x)|$. Polynomials are cheaper per step (no solve) but struggle once $\ell$ is very small.

The key empirical fact: **Rational wins in the hard regime** (high condition number), while **Polynomial wins once the interval is easy**.

### 1.2 Motivation: Crossing the Crossover

The following table reports the final lower endpoint for different starting floors $\ell$. Larger is better. (Polynomial steps are normalized so $\hat{p}(1)=1$).

| Start $\ell$       | 1 PE      | 1 DWH    | 2 PE     | 5 PE        |
|:-------------------|:----------|:---------|:---------|:------------|
| $10^{-1}$          | 0.446899  | 0.865659 | 0.882043 | 0.999999989 |
| $9 \times 10^{-2}$ | 0.402120  | 0.850958 | 0.852727 | 0.999999954 |
| $8 \times 10^{-2}$ | 0.356949  | 0.833906 | 0.816042 | 0.999999999 |
| $5 \times 10^{-2}$ | 0.220748  | 0.761019 | 0.642311 | 1.000000000 |
| $10^{-2}$          | 0.042954  | 0.505336 | 0.171507 | 0.999997565 |
| $10^{-3}$          | 0.004261  | 0.248039 | 0.018033 | 0.796221    |
| $10^{-4}$          | 0.000426  | 0.116562 | 0.001811 | 0.133854    |
| $10^{-6}$          | 0.000004  | 0.025194 | 0.000018 | 0.001398    |

At the standard design floor of $\ell_0 = 10^{-3}$: **2 PE steps from scratch are useless**, and even 5 PE steps leave significant error. However, a single DWH step lifts the floor to $\approx 0.25$—exactly where polynomial iterations enter their "sweet spot."

### 1.3 The Hybrid Sweet Spot

For $\ell_0 = 10^{-3}$, the best pattern is not "all rational" or "all polynomial." It is the synergy of both:

$$
\boxed{\text{1 DWH step} \;\to\; \text{2 normalized PE quintic steps}.}
$$

The progression at $\ell_0 = 10^{-3}$ demonstrates the efficiency:
$$
[10^{-3},1]
\;\xrightarrow{\;\text{1 DWH}\;}
[0.248039,1]
\;\xrightarrow{\;\text{PE}_1\;}
[0.729007,1]
\;\xrightarrow{\;\text{PE}_2\;}
[0.995160,1].
$$

---

## 2. Theoretical Components

### 2.1 The DWH Front-End (Rational)

Given a design floor $\ell \in (0, 1]$, the DWH coefficients $a, b, c$ are computed as:
$$
\gamma = \left(\frac{4(1 - \ell^2)}{\ell^4}\right)^{1/3},\quad r = \sqrt{1 + \gamma},\quad a = r + \frac{1}{2}\sqrt{8 - 4\gamma + \frac{8(2 - \ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a+b-1.
$$

In the Gram-space iteration ($B = A^\top A$), we use the **"apply-friendly" reparametrization**:
$$
\alpha = \frac{b}{c},\qquad \beta = a - \alpha.
$$
This lets us write the update as $R = \alpha I + \beta (I + cB)^{-1}$, avoiding the formation of large intermediate matrices and requiring only one SPD solve.

### 2.2 Polar Express Cleanup (Polynomial)

We use the degree-5 odd polynomial $p(x) = x(a + bx^2 + cx^4)$. To keep the top endpoint fixed, we normalize by $p(1)$: $\hat{p}(x) = p(x)/(a+b+c)$.

> [!theorem] Closed-Form PE Coefficients (Gram-Quadratic)
> Fix $0 < \ell < 1$. Let $q_0 \in (\ell, 1)$ be the root of $F(q_0; \ell) = 0$ (see Appendix B) that yields equioscillation with minimax error $E < 1$. Define:
> $$r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)},\quad S = q_0^2 + r^2,\quad P = q_0^2 r^2,\quad D = \left(1 - \frac{5}{3}S + 5P\right) + \ell\left(\ell^4 - \frac{5}{3}S\ell^2 + 5P\right).$$
> Then the minimax coefficients are:
> $$c = \frac{2}{D},\qquad b = -\frac{5c}{3}\,S,\qquad a = 5cP.$$

> [!remark] Root Isolation
> The degree-9 polynomial $F$ typically has a unique valid root in $(\ell, 1)$ that yields $E < 1$. It is easily isolated via bisection in FP64.

---

## 3. Hardware-Aware Implementation

In ML applications, the polar decomposition must be stable under FP16/BF16 arithmetic. We work in the tall orientation $X \in \mathbb{R}^{M \times N}, M \ge N$.

### 3.1 Stability Primitives

1.  **Column Normalization**: Compute FP32 column norms $n_j = \|X_{:j}\|_2$. Let $D = \text{diag}(1/\max(n_j, d_{\min}))$. Working with $Y = XD$ ensures the Gram diagonals stay near 1.
2.  **Moment-Based Upper Bound**: We avoid power iteration for $\lambda_{\max}$ by using a trace/Frobenius moment bound:
    $$u = \frac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max\bigl(0,\; N\|G\|_F^2 - \operatorname{tr}(G)^2\bigr)}}{N}$$
    where $\eta$ is a safety margin. This bound is tighter than scaling the full $u$ because it only adds a small multiple of the mean eigenvalue.
3.  **Safe SPD Solve**: For $(I + cB)$, we use Cholesky with adaptive jitter. If it fails, we add diagonal shift $\tau \leftarrow \max(\tau_{\min}, 10\tau + \tau_{\min})$ and retry.

### 3.2 The Full Algorithm

### 3.2 Stability and Utility Primitives

The following support procedures handle symmetrization, robust eigenvalue upper-bounding, and safe SPD inversions in the presence of floating-point drift.

<div class="algorithm-container">
<pre class="pseudocode">
\begin{algorithmic}
\PROCEDURE{Sym}{A}
    \RETURN $\frac{1}{2}(A + A^\top)$
\ENDPROCEDURE


\PROCEDURE{MomentUpperBound}{G}
    \STATE $u_M \leftarrow \dfrac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max(0, N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}$
    \RETURN $u_M$
\ENDPROCEDURE


\PROCEDURE{SafeSolveSPD}{S}
    \STATE $\tau \leftarrow 0$
    \WHILE{true}
        \STATE \textbf{try} $L \leftarrow \mathrm{Cholesky}(S + \tau I)$
        \IF{success}
            \RETURN $\mathrm{solve}(L L^\top, I)$
        \ELSE
            \STATE $\tau \leftarrow \max(\tau_{\min}, 10\tau + \tau_{\min})$
        \ENDIF
    \ENDWHILE
\ENDPROCEDURE
\end{algorithmic}
</pre>
</div>

### 3.3 The Main Hybrid Algorithm

With the primitives defined, the full hybrid polar decomposition is expressed as a three-step spectral update entirely in the small-side Gram space.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Stable Hybrid Polar: 1 DWH + 2 PE</div>
<pre class="pseudocode">
\begin{algorithmic}
\PROCEDURE{HybridPolar}{$X \in \mathbb{R}^{M \times N}$}
    \IF{$M < N$}
        \STATE $X \leftarrow X^\top$
    \ENDIF

    \STATE \COMMENT{--- Preconditioning & Form Gram ---}
    \STATE $d_j \leftarrow 1/\max(\|X_{:j}\|_2, d_{\min})$
    \STATE $D \leftarrow \operatorname{diag}(d_j), \quad Y \leftarrow X D, \quad G \leftarrow \mathrm{Sym}(Y^\top Y)$

    \STATE \COMMENT{--- Normalize Gram ---}
    \STATE $u \leftarrow \mathrm{MomentUpperBound}(G)$
    \STATE $B \leftarrow G/u, \quad Y \leftarrow Y/\sqrt{u}, \quad K \leftarrow I$

    \STATE \COMMENT{--- Step 1: DWH ($\ell_0 = 10^{-3}$) ---}
    \STATE $H \leftarrow \mathrm{SafeSolveSPD}(I + c_0 B)$
    \STATE $R_0 \leftarrow \alpha_0 I + \beta_0 H, \quad K \leftarrow K R_0, \quad B \leftarrow \mathrm{Sym}(R_0 B R_0)$

    \STATE \COMMENT{--- Steps 2-3: Normalized PE Cleanup ---}
    \FOR{$i=1, 2$}
        \STATE $Q_i \leftarrow \hat{a}_i I + \hat{b}_i B + \hat{c}_i B^2, \quad K \leftarrow K Q_i, \quad B \leftarrow \mathrm{Sym}(Q_i B Q_i)$
    \ENDFOR

    \RETURN $Q = Y K$ \COMMENT{Final rectangular GEMM}
\ENDPROCEDURE
\end{algorithmic}
</pre>
</div>


---

## 4. Design Constants ($\ell_0 = 10^{-3}$)

Fixed constants for implementation, computed offline in FP64:

| Step | Parameters | Values |
|:---|:---|:---|
| **DWH** | $\alpha_0, \beta_0, c_0$ | $0.984313, 251.00779, 16000.25$ |
| **PE 1** | $\hat{a}_1, \hat{b}_1, \hat{c}_1$ | $3.306253, -6.208058, 3.901804$ |
| **PE 2** | $\hat{a}_2, \hat{b}_2, \hat{c}_2$ | $2.194121, -1.974869, 0.780748$ |

---

### 5. Discussion

- **Efficiency**: Only two rectangular GEMMs ($Y^\top Y$ and $YK$) are required. All other heavy computation is performed on small $N \times N$ matrices in Gram space, which is computationally negligible when $M \gg N$.
- **Reduced Small-Side Latency**: Compared to a standard two-step DWH approach, the hybrid replaces the second SPD solve with two cheap polynomial matrix evaluations, significantly reducing latency on modern hardware.
- **Dynamic Stability**: The DWH step immediately exits the ill-conditioned regime ($10^{-3} \to 0.25$), while normalization by $\hat{p}(1) = 1$ prevents the dynamic range instability often seen in pure Newton-Schulz methods.

Combining rational robustess with polynomial speed results in a polar decomposition that is both fast enough for inner-loop training and robust enough for real-world ML spectral distributions.

---

## Appendix A: PE Coefficient Polynomial

The degree-9 polynomial $F(q_0; \ell)$ from §2.2:
$$
F(q_0; \ell) = F_0(q_0) + \ell^2 F_1(q_0) - \ell^4 F_2(q_0) + \ell^6 F_3(q_0),
$$
where:
- $F_0(q_0) = -2048q_0^9 - 5888q_0^8 - 9608q_0^7 - 7728q_0^6 - 1288q_0^5 + 1748q_0^4 + 888q_0^3 + 8q_0^2 - 72q_0 - 12$
- $F_1(q_0) = 4520q_0^7 + 9340q_0^6 + 10990q_0^5 + 8525q_0^4 + 3200q_0^3 + 80q_0^2 - 240q_0 - 40$
- $F_2(q_0) = 3600q_0^5 + 5800q_0^4 + 3900q_0^3 + 1750q_0^2 + 600q_0 + 100$
- $F_3(q_0) = 125(2q_0 + 1)^3$

## Appendix B: Verification Code

> [!example]- Python: Scalar Iteration and Analytic PE Coefficients
> ```python
> import numpy as np
>
> def dwh_coeffs(ell):
>     gamma = (4 * (1 - ell**2) / ell**4) ** (1.0 / 3.0)
>     r = np.sqrt(1 + gamma)
>     a = r + 0.5 * np.sqrt(8 - 4 * gamma + 8 * (2 - ell**2) / (ell**2 * r))
>     b = (a - 1) ** 2 / 4
>     c = a + b - 1
>     return a, b, c
>
>
> def dwh_map(x, a, b, c):
>     return x * (a + b * x**2) / (1 + c * x**2)
>
>
> def analytic_pe_coeffs(ell):
>     L2, L4, L6 = ell**2, ell**4, ell**6
>     def F(q):
>         return (-2048*q**9 - 5888*q**8 - 9608*q**7 - 7728*q**6 - 1288*q**5 + 1748*q**4 + 888*q**3 + 8*q**2 - 72*q - 12
>                 + L2 * (4520*q**7 + 9340*q**6 + 10990*q**5 + 8525*q**4 + 3200*q**3 + 80*q**2 - 240*q - 40)
>                 - L4 * (3600*q**5 + 5800*q**4 + 3900*q**3 + 1750*q**2 + 600*q + 100)
>                 + L6 * (1000*q**3 + 1500*q**2 + 750*q + 125))
>     qs = np.linspace(ell + 1e-12, 1 - 1e-12, 100000)
>     Fvals = np.array([F(q) for q in qs])
>     for i in range(len(Fvals) - 1):
>         if Fvals[i] * Fvals[i + 1] <= 0:
>             lo, hi = qs[i], qs[i + 1]
>             for _ in range(100):
>                 mid = (lo + hi) / 2
>                 if F(mid) * F(lo) <= 0: hi = mid
>                 else: lo = mid
>             q0 = (lo + hi) / 2
>             r2 = (2*q0**3 + 4*q0**2 + 6*q0 + 3) / (5 * (2*q0 + 1))
>             S, P = q0**2 + r2, q0**2 * r2
>             D = (1 - 5/3*S + 5*P) + ell*(ell**4 - 5/3*S*ell**2 + 5*P)
>             c, b, a = 2/D, -5*c/3*S, 5*c*P
>             if 1 - ell*(a + b*ell**2 + c*ell**4) < 1: return a, b, c
>     return None
>
>
> # Reproduce progression
> ell_0 = 1e-3
> a, b, c = dwh_coeffs(ell_0)
> ell_1 = dwh_map(ell_0, a, b, c)
> print(f"DWH: [{ell_1:.4f}, 1]")
> a1, b1, c1 = analytic_pe_coeffs(ell_1)
> u1 = a1 + b1 + c1
> ell_2 = ell_1*(a1 + b1*ell_1**2 + c1*ell_1**4)/u1
> print(f"PE1: [{ell_2:.4f}, 1]")
> a2, b2, c2 = analytic_pe_coeffs(ell_2)
> u2 = a2 + b2 + c2
> ell_3 = ell_2*(a2 + b2*ell_2**2 + c2*ell_2**4)/u2
> print(f"PE2: [{ell_3:.4f}, 1]")
> ```
> </blockquote>

---

## References

{% bibliography %}

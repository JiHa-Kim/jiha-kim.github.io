---
layout: post
title: "Fast Polar Decomposition with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigendecomposition or power iteration, and robust convergence from condition numbers up to 1000."
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
> In this post we show that **neither pure rational nor pure polynomial is optimal**. Instead, a simple hybrid—**one DWH step followed by two degree-5 Polar Express steps**—is the sweet spot. The rational step crushes the hard high-condition-number regime; the polynomial steps finish the job where they are most efficient. The algorithm uses exactly two rectangular GEMMs, no eigendecomposition or power iteration, and is stable under FP16/BF16 arithmetic.

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

| Start $\ell$       | 1 PE     | 1 DWH    | 2 PE     | 5 PE        |
| :----------------- | :------- | :------- | :------- | :---------- |
| $10^{-1}$          | 0.446899 | 0.865659 | 0.882043 | 0.999999989 |
| $9 \times 10^{-2}$ | 0.402120 | 0.850958 | 0.852727 | 0.999999954 |
| $8 \times 10^{-2}$ | 0.356949 | 0.833906 | 0.816042 | 0.999999999 |
| $5 \times 10^{-2}$ | 0.220748 | 0.761019 | 0.642311 | 1.000000000 |
| $10^{-2}$          | 0.042954 | 0.505336 | 0.171507 | 0.999997565 |
| $10^{-3}$          | 0.004261 | 0.248039 | 0.018033 | 0.796221    |
| $10^{-4}$          | 0.000426 | 0.116562 | 0.001811 | 0.133854    |
| $10^{-6}$          | 0.000004 | 0.025194 | 0.000018 | 0.001398    |

{% include comparison_widget.html %}

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

{% include polar_widget.html %}

---

## 2. Theoretical Components

### 2.1 The DWH Front-End (Rational)

Given a design floor $\ell \in (0, 1]$, the DWH coefficients $a, b, c$ are computed as:
$$
\zeta = \left(\frac{4(1 - \ell^2)}{\ell^4}\right)^{1/3},\quad r = \sqrt{1 + \zeta},\quad a = r + \frac{1}{2}\sqrt{8 - 4\zeta + \frac{8(2 - \ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a+b-1.
$$

In the Gram-space iteration ($B = A^\top A$), we use the **"apply-friendly" reparametrization**:
$$
\alpha = \frac{b}{c},\qquad \beta = \frac{a - \alpha}{c},\qquad \gamma = \frac{1}{c}.
$$
This lets us write the update as $R = \alpha I + \beta (\gamma I + B)^{-1}$, completely removing the need for division or scalar divisions on the fly. Importantly, the constants $(\alpha, \beta, \gamma)$ are pre-computed entirely offline in FP64 and injected directly into the ML kernel as raw floats, guaranteeing consistency and avoiding numerical jitter at runtime.

### 2.2 Polar Express Cleanup (Polynomial)

We use the degree-5 odd polynomial $p(x) = x(a + bx^2 + cx^4)$. The closed-form coefficients below are for the unnormalized minimax polynomial $p$. In the actual iteration we keep the top endpoint fixed by normalizing with $p(1)$:
$$
\hat{p}(x) = \frac{p(x)}{p(1)} = \frac{p(x)}{a+b+c}.
$$

> [!theorem] Closed-Form PE Coefficients (Gram-Quadratic)
> Fix $0 < \ell < 1$. Let $q_0 \in (\ell, 1)$ be the root of $F(q_0; \ell) = 0$ (see Appendix A and B) that yields equioscillation with minimax error $E < 1$. Define:
> $$r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)},\quad S = q_0^2 + r^2,\quad P = q_0^2 r^2,\quad D = \left(1 - \frac{5}{3}S + 5P\right) + \ell\left(\ell^4 - \frac{5}{3}S\ell^2 + 5P\right).$$
> Then the minimax coefficients are:
> $$c = \frac{2}{D},\qquad b = -\frac{5c}{3}\,S,\qquad a = 5cP.$$


---

## 3. Hardware-Aware Implementation

In ML applications, the polar decomposition must be stable under FP16/BF16 arithmetic. We work in the tall orientation $X \in \mathbb{R}^{M \times N}, M \ge N$.

### 3.1 Stability Primitives

1.  **Robust Gram Accumulation (ColNorm)**: To alleviate precision errors and reduce dynamic range instability in low-precision (FP16/BF16) arithmetic, we temporarily normalize the columns of $X$ before forming the Gram matrix. This gives a safer accumulation path and also a better-conditioned matrix for the initial DWH solve, but the overall iteration must still target the original polar factor.
    - We compute column sums-of-squares in one pass, lower-clip them as $d_j = \max(\sum_i X_{ij}^2, \epsilon)$, and form $D = \operatorname{diag}(d^{-1/2})$, then accumulate the scaled Gram:
      $$
      \tilde{G} = (X D)^\top (X D).
      $$
    - Because $\tilde{G} = D X^\top X D$, the original Gram is
      $$
      G = D^{-1} \tilde{G} D^{-1}.
      $$
      Writing $\Delta = D^2 = \operatorname{diag}(1/d_j)$, the moment-bound statistics can be evaluated directly from the scaled Gram:
      $$
      \operatorname{tr}(G) = \sum_i d_i \tilde{G}_{ii}, \qquad \|G\|_F^2 = \sum_{i,j} d_i d_j \vert \tilde{G}_{ij} \vert^2.
      $$
      For the DWH front-end, we should not discard the scaling before Cholesky. Instead, if
      $$
      B = G/u,
      $$
      then
      $$
      (\gamma_0 I + B)^{-1}
      = u(\gamma_0 u I + G)^{-1}
      = u D (\gamma_0 u \Delta + \tilde{G})^{-1} D.
      $$
      So the Cholesky should be applied to the scaled SPD matrix $\gamma_0 u \Delta + \tilde{G}$, and only then mapped back to the original coordinates.
    This adds an element-wise pass over $X$, but it prevents the intermediate Gram accumulation from overflowing or losing critical precision bits while still preserving the original polar factor of $X$.
2.  **Moment-Based Upper Bound**: We avoid power iteration for $\lambda_{\max}$ by using a trace/Frobenius moment bound:
    $$u = \frac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max\bigl(0,\; N\|G\|_F^2 - \operatorname{tr}(G)^2\bigr)}}{N}$$
    where $\eta$ is a safety margin. This gives a cheap closed-form upper bound for $\lambda_{\max}(G)$ without needing a power iteration.

> [!proof]- Proof of the Moment Bound
> Assume the eigenvalues are ordered $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_N$. Let $\mu = \frac{1}{N} \operatorname{tr}(G)$ be the mean and $V = \sum (\lambda_k - \mu)^2$ be the total variance. Since $\sum (\lambda_k - \mu) = 0$, we have:
> $$
> \lambda_1 - \mu = \sum_{k=2}^N (\mu - \lambda_k)
> $$
> Squaring both sides and applying Cauchy-Schwarz:
> $$
> (\lambda_1 - \mu)^2 = \left( \sum_{k=2}^N 1 \cdot (\mu - \lambda_k) \right)^2 \le (N-1) \sum_{k=2}^N (\lambda_k - \mu)^2
> $$
> Substituting $\sum_{k=2}^N (\lambda_k - \mu)^2 = V - (\lambda_1 - \mu)^2$ gives:
> $$
> (\lambda_1 - \mu)^2 \le (N-1) \left( V - (\lambda_1 - \mu)^2 \right) \implies N(\lambda_1 - \mu)^2 \le (N-1)V
> $$
> Finally, substituting $V = \|G\|_F^2 - N\mu^2$ and taking the square root:
> $$
> \lambda_1 \le \mu + \sqrt{\frac{N-1}{N} V} = \frac{\operatorname{tr}(G) + \sqrt{(N-1)(N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}
> $$

3.  **Safe SPD Solve**: For $(\gamma_0 I + B)$, we use Cholesky with adaptive jitter. If it fails, we add diagonal shift $\tau \leftarrow \max(\tau_{\min}, 10\tau + \tau_{\min})$ and retry.
4.  **Stable Fused Updates for Gram Iterations**: Matrix iterations compute spectral updates via an intermediate factor $Q$. For example, DWH computes $R_0 = \alpha_0 I + \beta_0 H$, and PE computes $Q = a I + b B + c B^2$, both followed by $B \leftarrow Q B Q$ and $K \leftarrow K Q$. However, in low precision (BF16/FP16), adding a large $O(1)$ constant to the diagonal of a matrix can squash the precision of its small off-diagonal elements. We avoid this by defining the strictly-matrix part $Z$ (where $Z_0 = \beta_0 H$ for DWH and $Z_i = \hat{b}_i B + \hat{c}_i B^2$ for PE). We then apply a "fused" update without ever materializing the inflated diagonal:
    $$
    R_Z = a B + B Z, \quad B \leftarrow a R_Z + Z R_Z
    $$
    and apply the distributed update $K \leftarrow a K + K Z$ to the factor $K$. This preserves numerical dynamic range in the off-diagonals.

### 3.2 Stability and Utility Primitives

The following support procedures handle symmetrization, robust eigenvalue upper-bounding, and safe SPD inversions in the presence of floating-point drift.

<div class="algorithm-container">
```pseudo
def Sym($A$):
    return $\frac{1}{2}(A + A^\top)$

def ScaledMomentUpperBound($\tilde{G}, d$):
    $\operatorname{tr}(G) \leftarrow \sum_i d_i \tilde{G}_{ii}$
    $\|G\|_F^2 \leftarrow \sum_{i,j} d_i d_j \vert \tilde{G}_{ij} \vert^2$
    $u_M \leftarrow \dfrac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max(0, N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}$
    return $u_M$

def SafeSolveSPD($S$):
    $\tau \leftarrow 0$
    while true:
        try $L \leftarrow \mathrm{Cholesky}(S + \tau I)$
        if success:
            return $\mathrm{solve}(L L^\top, I)$
        else:
            $\tau \leftarrow \max(\tau_{\min}, 10\tau + \tau_{\min})$
```
</div>

### 3.3 The Main Hybrid Algorithm

With the primitives defined, the full hybrid polar decomposition is expressed as a three-step spectral update entirely in the small-side Gram space of the **original** matrix.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Stable Hybrid Polar: 1 DWH + 2 PE</div>
```pseudo
def HybridPolar($X \in \mathbb{R}^{M \times N}$):
    transposed = false
    if $M < N$:
        $X \leftarrow X^\top$
        transposed = true

    # --- ColNorm for safe accumulation and the first rational solve ---
    $d_j \leftarrow \max(\sum_i X_{ij}^2, \epsilon)$
    $D \leftarrow \operatorname{diag}(d_j^{-1/2})$
    $\Delta \leftarrow \operatorname{diag}(d_j^{-1})$
    $\tilde{G} \leftarrow$ @Sym($(X D)^\top (X D)$)
    $u \leftarrow$ @ScaledMomentUpperBound($\tilde{G}, d$)
    $G \leftarrow$ @Sym($D^{-1} \tilde{G} D^{-1}$)

    # --- Normalize original Gram ---
    $B \leftarrow G/u, \quad K \leftarrow \frac{1}{\sqrt{u}} I$

    # --- Step 1: DWH ($\ell_0 = 10^{-3}$) ---
    $H \leftarrow u D$ @SafeSolveSPD($\gamma_0 u \Delta + \tilde{G}$) $D$
    $Z_0 \leftarrow \beta_0 H, \quad R_{Z_0} \leftarrow \alpha_0 B + B Z_0$
    $B \leftarrow$ @Sym($\alpha_0 R_{Z_0} + Z_0 R_{Z_0}$), $\quad K \leftarrow \alpha_0 K + K Z_0$

    # --- Steps 2-3: Normalized PE Cleanup (Stable Form) ---
    for $i=1, 2$:
        $Z \leftarrow \hat{b}_i B + \hat{c}_i B^2$
        $R_Z \leftarrow \hat{a}_i B + B Z$
        $B \leftarrow$ @Sym($\hat{a}_i R_Z + Z R_Z$)
        $K \leftarrow \hat{a}_i K + K Z$

    $Q \leftarrow X K$
    if transposed:
        return $Q^\top$
    return $Q$ # Polar factor of the original matrix
```
</div>


---

## 4. Design Constants ($\ell_0 = 10^{-3}$)

Fixed constants for implementation, computed offline in FP64:

| Step     | Parameters                        | Values                                            |
| :------- | :-------------------------------- | :------------------------------------------------ |
| **DWH**  | $\alpha_0, \beta_0, \gamma_0$     | $0.984313239819, 0.015687740588, 0.000062499018$  |
| **PE 1** | $\hat{a}_1, \hat{b}_1, \hat{c}_1$ | $3.306253466518, -6.208057722422, 3.901804255904$ |
| **PE 2** | $\hat{a}_2, \hat{b}_2, \hat{c}_2$ | $2.194120920624, -1.974868730017, 0.780747809393$ |

---

### 5. Discussion

- **Numerical Robustness**: By performing column-wise normalization (ColNorm) for the Gram accumulation and carrying that scaling through the first Cholesky solve, we shield the algorithm from dynamic range overflows and precision loss in low-precision (BF16/FP16) arithmetic without changing the target polar factor.
- **Balanced Latency**: While pre-scaling adds one element-wise pass over the tall matrix $X$, the cost is offset by the reduction in small-side complexity compared to pure rational iterations. The algorithm still performs exactly two tall rectangular GEMMs.
- **Dynamic Stability**: The DWH step immediately exits the ill-conditioned regime ($10^{-3} \to 0.25$), while normalization by $\hat{p}(1) = 1$ prevents the dynamic range instability often seen in pure Newton-Schulz methods.

Combining rational robustness with polynomial speed results in a polar decomposition that is both fast enough for inner-loop training and robust enough for real-world ML spectral distributions.

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

## Appendix B: Isolation of the Minimax Root

The degree-9 equation $F(q_0; \ell) = 0$ is obtained by eliminating other variables and thus contains extraneous branches that do not correspond to the minimax solution.

Let $V = \operatorname{span}\{x, x^3, x^5\}$ and $e(x) = 1 - p(x)$ on $[\ell, 1]$. Since $V$ is a 3-dimensional Haar (Chebyshev) space on the positive interval, the best uniform approximant $p^* \in V$ is uniquely characterized by the existence of $4$ points $\ell \le x_0 < x_1 < x_2 < x_3 \le 1$ such that $e^*(x_j) = \pm E$ with alternating signs.

In our parametrization, the error $e(x)$ has critical points precisely at $q_0$ and $r$. A real root $q_0$ of $F(q_0;\ell) = 0$ is therefore the **unique** valid minimax solution if and only if it produces an ordering:
$$ \ell < q_0 < r < 1 $$
and alternating error signs:
$$ e(\ell) > 0, \quad e(q_0) < 0, \quad e(r) > 0, \quad e(1) < 0 $$
Since $e(x) = 1 - p(x)$, we isolate the root by rigorously verifying that $p(\ell)$ and $p(r)$ dip strictly underneath $1$ (yielding $E > 0$), whereas $p(q_0)$ and $p(1)$ push strictly above $1$ (yielding $-E < 0$). Extraneous roots will natively fail this geometric alternation condition!

## Appendix C: Verification Code

> [!example]- Python: Scalar Iteration and Analytic PE Coefficients
> ```python
> import numpy as np
>
> def dwh_coeffs(ell):
>     gamma_inner = (4 * (1 - ell**2) / ell**4) ** (1.0 / 3.0)
>     r = np.sqrt(1 + gamma_inner)
>     a = r + 0.5 * np.sqrt(8 - 4 * gamma_inner + 8 * (2 - ell**2) / (ell**2 * r))
>     b = (a - 1) ** 2 / 4
>     c = a + b - 1
>     
>     alpha = b / c
>     beta = (a - alpha) / c
>     gamma = 1 / c
>     return alpha, beta, gamma
>
>
> def dwh_map(x, alpha, beta, gamma):
>     return alpha * x + beta * x / (gamma + x**2)
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
>             c = 2/D
>             b = -5*c/3*S
>             a = 5*c*P
>             
>             # Chebyshev alternation check to reject extraneous branches
>             p_r = np.sqrt(r2) * (a + b*r2 + c*r2**2)
>             p_l = ell * (a + b*ell**2 + c*ell**4)
>             p_q0 = q0 * (a + b*q0**2 + c*q0**4)
>             p_1 = a + b + c
>             
>             r = np.sqrt(r2)
>             if r2 > 0 and ell < q0 < r < 1:
>                 E_ell, E_r = 1 - p_l, 1 - p_r
>                 E_q0, E_1 = p_q0 - 1, p_1 - 1
>                 errs = [E_ell, E_r, E_q0, E_1]
>                 if all(e > 0 for e in errs) and max(errs) - min(errs) < 1e-4:
>                     return a, b, c
>     return None
>
>
> # Reproduce progression
> ell_0 = 1e-3
> alpha, beta, gamma = dwh_coeffs(ell_0)
> ell_1 = dwh_map(ell_0, alpha, beta, gamma)
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


---

## References

{% bibliography %}

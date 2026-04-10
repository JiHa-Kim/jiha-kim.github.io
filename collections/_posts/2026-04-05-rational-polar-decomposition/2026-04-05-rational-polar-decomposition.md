---
layout: post
title: "Fast Polar Decomposition for Muon Optimizer with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigendecomposition or power iteration, and robust convergence from condition numbers up to 1000."
categories:
  - Numerical Linear Algebra
  - Mathematical Optimization
tags:
  - Polar Decomposition
  - Muon
  - Orthogonalization
  - Matrix Iterations
  - Rational Functions
  - Polynomial Iterations
  - Newton-Schulz
  - Dynamic Weighted Halley
  - Polar Express
  - Numerical Stability
math: true
scholar:
  bibliography: posts/2026-04-05-rational-polar-decomposition/rational-polar-decomposition.bib
---

> [!info] Overview
> We introduce an optimized hybrid algorithm for the matrix polar decomposition, tailored for machine learning optimizers like Muon. By combining the rapid early convergence of **Dynamic Weighted Halley (rational)** steps with the hardware efficiency of **Polar Express (polynomial)** cleanup, we achieve a robust, high-performance polar factor.

---

## 1. Background and Motivation

### 1.1 The Geometry of Gradients

In deep learning, gradients often exhibit **extreme anisotropy**—they are much larger in some directions than others. This ill-conditioning forces standard optimizers to use small learning rates or complex preconditioning to avoid instability.

Recently, the **Muon optimizer** {% cite jordanMuonOptimizer2024 %} has gained popularity by taking an aggressive stance: it completely discards the "stretch" (the magnitude of the gradient in different directions) and preserves only the "direction". By projecting the update onto the Stiefel manifold, Muon ensures that the update is purely orthogonal, which has shown remarkable empirical success in training large transformers.

### 1.2 Defining the Polar Decomposition

To formalize this "direction vs. stretch" split, we use the **polar decomposition**. Any matrix $A \in \mathbb{R}^{m \times n}$ (with $m \ge n$) can be factored as:
$$
A = QP
$$
where:
- $Q$ is the **direction** matrix (column-orthonormal, $Q^\top Q = I$).
- $P$ is a **positive semi-definite (non-negative definite)** stretch matrix (the "modulus").

This is intimately related to the **Thin SVD**. If $A = U \Sigma V^\top$ ($U \in \mathbb{R}^{m \times r}, \Sigma \in \mathbb{R}^{r \times r}, V \in \mathbb{R}^{n \times r}$ for $r = \operatorname{rank}(A)$) is the thin SVD, then its corresponding polar factor $Q$ is:
$$
Q = U V^\top
$$
while $P = V \Sigma V^\top$. Equivalently,

$$
Q = A (A^\top A)^{+/2} = (A A^\top)^{+/2} A, \quad P = (A^\top A)^{1/2}.
$$

where $(A^\top A)^{+/2}$ is the Moore-Penrose pseudoinverse of the modulus. 

The polar factor $Q$ is the "closest" orthonormal matrix to $A$ in the Frobenius norm, capturing the pure orientation of the transformation.

$$
Q = \underset{X^\top X = I}{\operatorname{argmin}} \|A - X\|_F
$$

### 1.3 Spectral Mapping

While the SVD provides a direct way to compute $Q$, it is globally synchronous and expensive on modern hardware. Instead, we can use the **Spectral Mapping Theorem**. Any matrix function $f(A)$ defined via its Taylor series (or rational form) acts directly on the singular values of $A$:
$$
A = U \operatorname{diag}(\sigma_i) V^\top \implies f(A) = U \operatorname{diag}(f(\sigma_i)) V^\top
$$
The crucial trick is to design a function $f(x)$ such that $f(\sigma_i) \approx 1$ for all $\sigma_i \in (0, 1]$. If we can find such a function, then $f(A) \approx U (I) V^\top = Q$, giving us the polar factor without ever computing an explicit eigendecomposition.

---

## 2. Why a Hybrid?

Iterative methods for the polar factor work by applying a scalar iteration $f(x)$ to the singular values $\sigma_i$ of $A$. In this post we track the actual interval of singular values, not just its ratio. The hybrid uses two distinct scalar regimes:

- DWH keeps the image inside $[\ell_{t+1}, 1]$.
- PE uses centered minimax maps whose image interval is $[1-E_t, 1+E_t]$.

If a final contractive or explicitly 1-Lipschitz output is desired, we can divide once at the very end by the final upper endpoint. That terminal normalization is optional and is **not** part of the intermediate PE iteration.

### 2.1 The Two Scalar Maps

- **DWH (rational)**: the Dynamic Weighted Halley map {% cite nakatsukasaOptimizingHalleyIteration2010 %} applies the rational function:

$$
f_{\text{DWH}}(x) = x\frac{a + bx^2}{1 + cx^2}
$$

where the coefficients $a, b, c$ are chosen optimally for the current floor $\ell$. Rational maps are exceptionally powerful at lifting small $\ell$ but require a linear solve per update step.

- **Polar Express (polynomial)**: the degree-5 PE map {% cite polarExpress2025 %} applies the odd polynomial:

$$
p(x) = x(a + bx^2 + cx^4) = ax + bx^3 + cx^5
$$

where $(a, b, c)$ solve the Polar Express minimax problem on the current tracked interval $[\ell_t, u_t]$. The key PE invariant is that the image interval is re-centered around $1$: after any finite-precision cushioning adjustment, the polynomial is rescaled so
$$
1 - p_t(\ell_t) = p_t(u_t) - 1,
$$
equivalently $p_t(\ell_t) + p_t(u_t) = 2$.

More generally, any non-unit upper endpoint can be absorbed directly into the next PE coefficients. If $q(y) = ay + by^3 + cy^5$ is designed on the normalized interval $[\lambda_t, 1]$ with $\lambda_t = \ell_t/u_t$, then the actual polynomial on $[\ell_t, u_t]$ is simply
$$
p_t(x) = q\left(\frac{x}{u_t}\right) = \frac{a}{u_t}x + \frac{b}{u_t^3}x^3 + \frac{c}{u_t^5}x^5.
$$
So there is no separate "recenter the matrix, then run PE" step in the final presentation. The scale is folded into the coefficients. In particular, the **first** PE step after DWH is already on $[\ell_1, 1]$, so there is no extra nontrivial scale factor there; the absorption matters starting with later PE steps once $u_t \neq 1$.

The key empirical fact: **Rational wins in the hard regime** (high condition number), while **Polynomial wins once the interval is easy**.

### 2.2 Motivation: Crossing the Crossover

The following widget plots the resulting **lower endpoint** after one or more scalar steps. For PE, the full image interval is centered around $1$; the chart shows only its lower edge. Larger values closer to $1$ are better (equivalently, smaller worst-case error on the lower side).

{% include comparison_widget.html %}

### 2.3 The Hybrid Sweet Spot

For $\ell_0 = 10^{-3}$, the best pattern is not "all rational" or "all polynomial." It is the synergy of both:

$$
\boxed{\text{1 DWH step} \;\to\; \text{2 PE quintic steps}.}
$$

The progression at $\ell_0 = 10^{-3}$ demonstrates the efficiency:
$$
[10^{-3},1]
\;\xrightarrow{\;\text{1 DWH}\;}
[0.248039,1]
\;\xrightarrow{\;\text{PE}_1\;}
[0.843267,1.156733]
\;\xrightarrow{\;\text{PE}_2\;}
[0.997574,1.002426].
$$

If we want a final bounded operator, we normalize only once at the end:
$$
\frac{1}{1.002426}[0.997574,1.002426] = [0.995160,1].
$$
That terminal normalization is the right place to enforce a spectral-norm or Lipschitz certificate. Doing it at every PE step would change the polynomial optimization problem and make the iteration less aggressive.

{% include polar_widget.html %}

---

## 3. Implementation and Performance

The hybrid polar decomposition targets speed and stability under low-precision (BF16/FP16) arithmetic.

### 3.1 Auxiliary Functions

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Helpers</span> Stability and Symmetry Primitives</div>
```pseudo
def Sym($A$): return $\frac{1}{2}(A + A^\top)$

def SafeCholesky($S, dtype, K_{\max}=6$):
    # Pivot-informed jittered Cholesky with monotone doubling backoff
    $S \leftarrow$ @Sym($S$), $\tau \leftarrow 0$
    $\tau_{\min} \leftarrow \epsilon_{\text{mach}}(dtype) \max(\text{tr}(S)/N, \max_i S_{ii}, 1)$
    for $t=1, \dots, K_{\max}$:
        try $L \leftarrow$ @Cholesky($S + \tau I$) while tracking $\pi_{\min}$
        if success: return ($L$, $\tau$)
        $\tau \leftarrow \max(\tau + \max(0, -\pi_{\min}) + \tau_{\min}, 2\tau, \tau_{\min})$
    fail
```
</div>

### 3.2 The Main Hybrid Algorithm

The following algorithm utilizes exactly two tall rectangular GEMMs. The DWH phase is the usual constrained rational step. The PE phase is expressed directly in absolute coordinates, with any non-unit PE upper endpoint absorbed into the next PE coefficients rather than handled by a separate rescaling kernel.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Optimized Hybrid Polar: 1 DWH + 2 PE</div>
```pseudo
def HybridPolar($X \in \mathbb{R}^{M \times N}$):
    # 1. Tall orientation check
    if $M < N$: return @HybridPolar($X^\top$)$^\top$

    dtype = fp32 # Precision for Cholesky/TRSM
    $\ell_{\text{fast}} = 0.15$ # Threshold for fast-path bypassing DWH

    # 2. Robust ColNorm & In-place Scaling (Saves O(MN) memory)
    # Accumulate col sums $d_j$ and total $\|X\|_F^2$ in FP64
    $d, \|X\|_F^2 \leftarrow \sum_i X_{ij}^2, \sum_j d_j$
    $\epsilon_{\text{col}} \leftarrow \epsilon_{\text{mach}}(dtype) \max(\|X\|_F^2/N, 1)$
    $d \leftarrow \max(d, \epsilon_{\text{col}})$
    $D \leftarrow \operatorname{diag}(d_j^{-1/2}), \quad \Delta \leftarrow \operatorname{diag}(d_j^{-1})$
    $X.mul\_(D)$ # Scale X in-place to avoid duplication

    # 3. Gram Accumulation & Moment Bound
    $\tilde{G} \leftarrow X^\top X$ # (via SYRK)
    $\operatorname{tr}(G) \leftarrow \|X\|_F^2, \quad \|G\|_F^2 \leftarrow d^\top \tilde{G}^{\circ 2} d$ # Vectorized
    $u \leftarrow \dfrac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max(0, N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}$
    $B \leftarrow (D^{-1} \tilde{G} D^{-1})/u$

    # 4. Step 1: DWH ($\ell_{0} = 10^{-3}$)
    $S \leftarrow \gamma_{0} u \Delta + \tilde{G}$
    $(L, \tau) \leftarrow$ @SafeCholesky($S$)
    $S_{\text{inv}} \leftarrow$ @cholesky_inverse($L$)
    
    if $\frac{1}{u \operatorname{tr}(S_{\text{inv}})} - \gamma_{0} > \ell_{\text{fast}}$:
        $K \leftarrow \frac{1}{\sqrt{u}} I$ # Fast-path: bypass DWH
    else:
        $H_{0} \leftarrow \gamma_{0} u D S_{\text{inv}} D$ # Bounded resolvent
        $H_{sq} \leftarrow$ @Sym($H_{0}^{2}$) # Half-FLOP SYRK
        $B \leftarrow g_{I} I + g_{B} B + g_{H} H_{0} + g_{H^2} H_{sq}$ # Zero-GEMM step
        $K \leftarrow \frac{1}{\sqrt{u}}(\alpha_{0} I + \beta_{0} H_{0})$
    $(\ell_1, u_1) \leftarrow$ scalar image interval after the DWH step # For the design floor, [0.248039, 1]

    # 5. Step 2: Centered PE Cleanup 1
    # Here u_1 = 1, so the "absorbed scale" is trivial in the first PE step.
    $\lambda_1 \leftarrow \ell_1 / u_1$
    $(a_1, b_1, c_1) \leftarrow$ @CenteredPECoeffs($\lambda_1$)
    $P_1(B) \leftarrow \frac{a_1}{u_1} I + \frac{b_1}{u_1^3} B + \frac{c_1}{u_1^5} B^2$
    $K \leftarrow K P_1(B)$
    $B \leftarrow$ @Sym($B P_1(B)^2$)
    $(\ell_2, u_2) \leftarrow (p_1(\ell_1), p_1(u_1))$

    # 6. Step 3: Centered PE Cleanup 2
    # Now u_2 > 1 in general, and that scale is folded into P_2(B).
    $\lambda_2 \leftarrow \ell_2 / u_2$
    $(a_2, b_2, c_2) \leftarrow$ @CenteredPECoeffs($\lambda_2$)
    $P_2(B) \leftarrow \frac{a_2}{u_2} I + \frac{b_2}{u_2^3} B + \frac{c_2}{u_2^5} B^2$
    $K \leftarrow K P_2(B)$
    $B \leftarrow$ @Sym($B P_2(B)^2$)
    $(\ell_3, u_3) \leftarrow (p_2(\ell_2), p_2(u_2))$

    # 7. Final Reconstitution
    $K \leftarrow \operatorname{diag}(d_i^{1/2})$ @Sym($K$) # Invert in-place scaling on K rows
    return $X K$ # Optional: divide once more by u_3 if a bounded final operator is required
```
</div>

Implementation note: treat $\tilde{G}, S, H, B$ as symmetric objects and use symmetric kernels conceptually (SYRK/SYMM/SYR2K). Only apply `@Sym(·)` at choke points (right before factorization, and optionally right before the final $Q \leftarrow X K$ matmul).

## 4. Design Constants ($\ell_0 = 10^{-3}$)

Fixed constants for implementation, computed offline in FP64:

| Step     | Parameters                    | Values                                                                      |
| :------- | :---------------------------- | :-------------------------------------------------------------------------- |
| **DWH**  | $g_I, g_B, g_H, g_{H^2}$      | 0.030883301527615, 0.968872554082809, 3.906861822017413, -3.937745123545028 |
|          | $\gamma_0, \alpha_0, \beta_0$ | 0.000062499017684, 0.984313239818915, 251.007791810856                      |
| **PE 1** | $A_1, B_1, C_1$               | 3.824452920237891, -7.181066039236940, 4.513346248799179                    |
| **PE 2** | $A_2, B_2, C_2$               | 1.901427287944732, -1.279060386064908, 0.377917707130065                    |

Here $p_t(x) = A_t x + B_t x^3 + C_t x^5$ is the **actual** centered PE polynomial used on the current absolute interval. For PE 1, this agrees with the normalized centered coefficients because $u_1 = 1$. For PE 2, the non-unit upper endpoint $u_2 = 1.156733\ldots$ from the previous PE step has already been absorbed into $(A_2, B_2, C_2)$.

---

## 5. Discussion

- **Numerical Robustness**: By performing column-wise normalization (ColNorm) for the Gram accumulation and carrying that scaling through the first Cholesky solve, we shield the algorithm from dynamic range overflows and precision loss in low-precision (BF16/FP16) arithmetic without changing the target polar factor.
- **Architectural FLOP Avoidance**: The DWH $W^{\top} W$ inversion is fully replaced by a Cholesky inverse via `potri` and scalar broadcast, while the vectorized Frobenius norm calculation $\|G\|_F^2 = d^\top \tilde{G}^{\circ 2} d$ replaces an $O(N^2)$ summation with a single tensor-friendly quadratic form. The PE phase then works entirely on the small $N \times N$ side.
- **Zero-Allocation $X$-Scaling**: The initial column normalization is applied in place on the tall matrix $X$, avoiding an extra $O(MN)$ buffer. The original scaling is restored later through the small-side factor $K$, so the memory savings do not change the target polar factor.
- **Dynamic Fast-Path Bypassing**: Because the explicitly materialized inverse $S_{\text{inv}} \approx (u(\gamma_{0} I + B))^{-1}$ contains strict spectral volume, its trace structurally enforces a computationally free lower bound: $\lambda_{\min} \ge \frac{1}{u \operatorname{tr}(S_{\text{inv}})} - \gamma_{0}$. If this dynamically verifiable trace bound exceeds the polynomial safety threshold $\ell_{\text{fast}}$, the kernel can safely skip the rational correction and move straight into the PE cleanup.
- **Balanced Latency**: While pre-scaling adds one element-wise pass over the tall matrix $X$, the cost is offset by the reduction in small-side complexity compared to pure rational iterations. The algorithm still performs exactly two tall rectangular GEMMs.
- **Dynamic Stability**: The DWH step immediately exits the ill-conditioned regime ($10^{-3} \to 0.25$). Each subsequent PE step balances its image interval around $1$, which is the right minimax geometry for the polynomial cleanup phase.
- **Absorbed PE Scaling**: Once a PE step produces an interval $[\ell_t, u_t]$ with $u_t \neq 1$, the next centered polynomial is applied as $q_t(x/u_t)$, so the scale is folded directly into the next PE coefficients. For the first PE step after DWH, this is trivial because $u_1 = 1$.
- **Final Certification**: If the returned update must satisfy a spectral-norm or Lipschitz bound, normalize once at the end by the final upper endpoint. That keeps the intermediate PE steps maximally aggressive while still providing the desired certificate.

Combining rational robustness with polynomial speed results in a polar decomposition that is both fast enough for inner-loop training and robust enough for real-world ML spectral distributions.

---

## 6. Technical Derivations

### 6.1 The Minimax Problem on a Tracked Interval

Both the DWH and Polar Express steps approximate the **matrix sign function**, but they solve different scalar optimization problems.

- In the **DWH/QDWH** phase, the rational map is constrained to remain in $[0,1]$ on the positive interval.
- In the **PE** phase, the polynomial map solves the unconstrained minimax problem on the current positive interval $[\ell_t, u_t]$ and therefore centers the image around $1$.

This difference is exactly why rational and polynomial steps play different roles in the hybrid.

On the positive interval, the target value is $1$, so the scalar optimization problem is
$$
\min_{f \in \mathcal{F}_{\text{odd}}} \max_{x \in [\ell_t, u_t]} \vert 1 - f(x) \vert,
$$
where $\mathcal{F}_{\text{odd}}$ is the space of candidate odd functions (rational or polynomial).

Because $f(x)$ and $\operatorname{sign}(x)$ are odd, the maximal error on the negative side mirrors the positive side. For a PE-style minimax optimum, the endpoint errors are balanced. In the simplest case on $[\ell, 1]$, this means
$$
1 - f(\ell) = f(1) - 1,
$$
so the image interval is centered around $1$. Equivalently, if $\ell_{t+1} = f(\ell_t)$ and $u_{t+1} = f(u_t)$, then the PE recurrence tracks intervals with $\ell_{t+1} + u_{t+1} = 2$. This is the sense in which the intermediate iterations are "centered around $1$."

If one instead solves a polynomial problem with an explicit no-overshoot constraint $f(x) \le 1$ from the outset, that defines a genuinely different constrained problem. That is **not** the primary Polar Express objective.

There is also an independent **input scaling** freedom: if an iterate has positive singular values in $[\ell_t, u_t]$, then multiplying the whole iterate by any scalar $\alpha > 0$ changes the interval to $[\alpha \ell_t, \alpha u_t]$ without changing the polar factor. Under the variable change $y = x/\alpha$, the polynomial optimization problem on $[\alpha \ell_t, \alpha u_t]$ is equivalent to the one on $[\ell_t, u_t]$. This is why any non-unit upper endpoint $u_t$ can be absorbed directly into the next PE coefficients instead of being handled by a separate explicit rescaling step. For the first PE step after DWH this absorption is trivial, because the DWH output already has upper endpoint $u_1 = 1$.

### 6.2 Optimality for Rational Iterations (DWH)

The Dynamic Weighted Halley (DWH) iteration utilizes a degree-$(3, 2)$ rational map. Because this specific rational form is chosen to be monotonically increasing on $[\ell, 1]$, its minimum strictly occurs at $f(\ell)$ and its maximum at $f(1)$.

For the DWH/QDWH problem, the no-overshoot constraint is active at the level of the applied map: the positive image is kept inside $[0,1]$, so the update quality is governed by how far it lifts the lower endpoint while keeping the top endpoint fixed at $f(1)=1$.

At the same time, this does **not** mean the rational phase is solving a weaker approximation problem than the two-sided sign approximation. In the Zolotarev/QDWH theory {% cite nakatsukasaOptimizingHalleyIteration2010 %}, the constrained positive-interval problem and the best rational approximation of $\operatorname{sign}(x)$ on $[-1,-\ell] \cup [\ell,1]$ are equivalent up to scaling. The type-$(3,2)$ DWH/QDWH step is exactly the corresponding scaled Zolotarev rational. So the rational phase is constrained per step, but it is still optimal within its rational family.

> [!theorem] Optimal DWH Coefficients
> For a design floor $\ell \in (0, 1]$, the DWH coefficients $a, b, c$ that minimize the minimax error are:
> $$
> \zeta = \left(\frac{4(1 - \ell^2)}{\ell^4}\right)^{1/3},\quad r = \sqrt{1 + \zeta}
> $$
> 
> $$
> a = r + \frac{1}{2}\sqrt{8 - 4\zeta + \frac{8(2 - \ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a+b-1.
> $$

In the Gram-space iteration ($B = A^\top A$), we use a **Resolvent-Basis Expansion** that guarantees exact stability while cutting the matrix multiplication FLOPs in half. Because the algebra generated by $B$ and the bounded resolvent $H_0 = (I + cB)^{-1} = \gamma(\gamma I + B)^{-1}$ closes linearly under the identity $B H_0 = (I - H_0)/c$, we can analytically expand the DWH update into a linear combination that avoids matrix cross-multiplications entirely:
$$
B \leftarrow g_I I + g_B B + g_H H_0 + g_{H^2} H_0^2
$$
By using this algebraic flattening, we compute the DWH update using only one symmetric matrix squaring, $H_0^2$. The scale-invariant coefficients $g_I, g_B, g_H, g_{H^2}$ are pre-computed in FP64, eliminating all dynamic-range runtime evaluation. The orientation factor $K$ is simultaneously updated via $(\alpha_0 I + \beta_0 H_0)$, where $\alpha_0 = b/c$ and $\beta_0 = a - b/c$.

### 6.3 Optimality for Polynomial Iterations (Polar Express)

We use the degree-5 odd polynomial $p(x) = x(a + bx^2 + cx^4)$. Unlike the monotonic DWH rational map, the PE optimal polynomial *equioscillates* on its tracked interval.

For exposition, we first describe the centered one-step problem on $[\ell, 1]$:
$$
\min_{a,b,c} \max_{x \in [\ell, 1]} \vert 1 - p(x) \vert = E
$$
By the Chebyshev Equioscillation Theorem, the optimal $p(x)$ oscillates between $1-E$ and $1+E$. So in the absolute one-step setting, the image interval becomes
$$
[p(\ell), p(1)] = [1-E, 1+E],
$$
which is centered around $1$ and has width $2E$.

In the full Polar Express recurrence, this balanced-endpoint property is maintained across iterations by tracking both $\ell_t$ and $u_t$. In the implementation code in Appendix G, after the finite-precision cushioning step the authors explicitly rescale the polynomial so that $1 - p(\ell_t) = p(u_t) - 1$, i.e. the image interval is re-centered around $1$ before the next step.

To apply such a centered polynomial on a general interval $[\ell_t, u_t]$, set $\lambda_t = \ell_t/u_t$ and solve for the normalized coefficients of
$$
q_t(y) = a_t y + b_t y^3 + c_t y^5
$$
on $[\lambda_t, 1]$. The corresponding polynomial on the original singular-value variable is
$$
p_t(x) = q_t\left(\frac{x}{u_t}\right) = \frac{a_t}{u_t}x + \frac{b_t}{u_t^3}x^3 + \frac{c_t}{u_t^5}x^5.
$$
This is the precise sense in which the DWH-to-PE recentering is "absorbed into the coefficients." For the first PE step after DWH, $u_1 = 1$, so this formula reduces to $p_1(x) = q_1(x)$; the nontrivial absorbed scaling starts with the second PE step.

> [!theorem] Closed-Form Centered PE Coefficients
> Fix $0 < \ell < 1$. Let $q_0 \in (\ell, 1)$ be the root of the following polynomial that yields equioscillation with minimax error $E < 1$:
> $$ F(q_0; \ell) = F_0(q_0) + \ell^2 F_1(q_0) - \ell^4 F_2(q_0) + \ell^6 F_3(q_0) $$
> where:
> - $F_0(q_0) = -2048q_0^9 - 5888q_0^8 - 9608q_0^7 - 7728q_0^6 - 1288q_0^5 + 1748q_0^4 + 888q_0^3 + 8q_0^2 - 72q_0 - 12$
> - $F_1(q_0) = 4520q_0^7 + 9340q_0^6 + 10990q_0^5 + 8525q_0^4 + 3200q_0^3 + 80q_0^2 - 240q_0 - 40$
> - $F_2(q_0) = 3600q_0^5 + 5800q_0^4 + 3900q_0^3 + 1750q_0^2 + 600q_0 + 100$
> - $F_3(q_0) = 125(2q_0 + 1)^3$
>
> Define geometric moments $S = q_0^2 + r^2, P = q_0^2 r^2,$ and $D$ via:
> $$r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}, \quad D = \left(1 - \frac{5}{3}S + 5P\right) + \ell\left(\ell^4 - \frac{5}{3}S\ell^2 + 5P\right)$$
> The centered minimax coefficients for $p(x) = ax + bx^3 + cx^5$ are then:
> $$c = \frac{2}{D},\qquad b = -\frac{5c}{3}\,S,\qquad a = 5cP.$$

The degree-9 equation $F(q_0; \ell) = 0$ contains extraneous branches. We isolate the **unique** valid root by verifying that it produces the required ordering $\ell < q_0 < r < 1$ and the correct Chebyshev alternation cycle.

<details class="box-proof" markdown="1">
<summary>Theory: Sketch of the Minimax Proof</summary>

The minimax odd polynomial $p(x) = ax + bx^3 + cx^5$ minimizes $\max_{x \in [\ell, 1]} |1 - p(x)|$. By the **Chebyshev Equioscillation Theorem**, since the basis $\{x, x^3, x^5\}$ has dimension 3, there exist 4 points $\ell \le x_0 < x_1 < x_2 < x_3 \le 1$ where the error $e(x) = 1 - p(x)$ attains its maximum magnitude $E$ with alternating signs.

For the optimal solution, these points must be $\ell, q_0, r, 1$, where $q_0$ and $r$ are internal critical points such that $p'(q_0) = 0$ and $p'(r) = 0$. The equioscillation conditions are:
1. $p(\ell) = 1 - E, \quad p(r) = 1 - E \implies p(\ell) = p(r)$
2. $p(q_0) = 1 + E, \quad p(1) = 1 + E \implies p(q_0) = p(1)$

Using the fact that $q_0, r$ are roots of $p'(x) = 5c x^4 + 3b x^2 + a = 0$, we substitute $a = 5cq_0^2 r^2$ and $b = -\frac{5c}{3}(q_0^2 + r^2)$. The condition $p(q_0) = p(1)$ yields $r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}$. Substituting this back into $p(\ell) = p(r)$ yields the polynomial $F(q_0; \ell) = 0$.

You can verify this entire derivation using the following symbolic checker:

```python
# /// script
# dependencies = ["sympy"]
# ///
import sympy

def verify_full_pe_derivation():
    # Define symbols
    a, b, c, ell, q0, r = sympy.symbols('a b c ell q0 r', real=True, positive=True)
    x = sympy.symbols('x')
    
    # Odd polynomial p(x) = ax + bx^3 + cx^5
    p = a*x + b*x**3 + c*x**5
    
    # 1. First-order optimality: p'(x) ~ 5c(x^2 - q0^2)(x^2 - r^2)
    a_expr = 5 * c * (q0**2 * r**2)
    b_expr = -sympy.Rational(5, 3) * c * (q0**2 + r**2)
    p_opt = p.subs({a: a_expr, b: b_expr})
    
    # 2. Equioscillation Condition A: p(q0) = p(1)
    eq_rsq = sympy.simplify(p_opt.subs(x, q0) - p_opt.subs(x, 1))
    r_sq = sympy.symbols('r_sq')
    r2_sol = sympy.solve(eq_rsq.subs(r**2, r_sq), r_sq)[0]
    
    # Verified target from §6.3
    target_r2 = (2*q0**3 + 4*q0**2 + 6*q0 + 3) / (10*q0 + 5)
    assert sympy.simplify(r2_sol - target_r2) == 0
    
    # 3. Equioscillation Condition B: p(ell) = p(r)
    expr_ell = p_opt.subs(x, ell) / c
    expr_r = p_opt.subs(x, r) / c
    final_eq = sympy.simplify((expr_ell**2 - expr_r**2).subs(r**2, r2_sol))
    num, _ = sympy.fraction(final_eq)
    
    # Target polynomial F(q0, ell) from §6.3
    F0 = -2048*q0**9 - 5888*q0**8 - 9608*q0**7 - 7728*q0**6 - 1288*q0**5 + 1748*q0**4 + 888*q0**3 + 8*q0**2 - 72*q0 - 12
    F1 = 4520*q0**7 + 9340*q0**6 + 10990*q0**5 + 8525*q0**4 + 3200*q0**3 + 80*q0**2 - 240*q0 - 40
    F2 = 3600*q0**5 + 5800*q0**4 + 3900*q0**3 + 1750*q0**2 + 600*q0 + 100
    F3 = 125*(2*q0 + 1)**3
    F_target = F0 + ell**2*F1 - ell**4*F2 + ell**6*F3
    
    assert sympy.simplify(num % F_target) == 0
    
    # 4. Centering: p(1) + p(ell) = 2
    D_post = (1 - sympy.Rational(5,3)*(q0**2 + r2_sol) + 5*(q0**2 * r2_sol)) + \
             ell*(ell**4 - sympy.Rational(5,3)*(q0**2 + r2_sol)*ell**2 + 5*(q0**2 * r2_sol))
    D_calc = (p_opt.subs(x, 1) + p_opt.subs(x, ell)).subs(r**2, r2_sol) / c
    assert sympy.simplify(D_calc - D_post) == 0
    print("All PE algebraic components verified successfully!")

if __name__ == "__main__":
    verify_full_pe_derivation()
```
</details>

### 6.4 The Limits of Composition (Polynomials vs. Zolotarev)

A natural question is whether we can construct high-degree optimal polynomials by simply composing lower-degree optimal ones. While this works for rational functions (Zolotarev rationals), it notably fails for polynomials.

#### 1) Structural Restriction: Composition is a Strict Subclass
Take the simplest nontrivial case: composing two odd cubics $p_1(x) = ax + bx^3$ and $p_2(x) = cx + dx^3$. The composition $P(x) = p_2(p_1(x))$ is an odd degree-9 polynomial:
$$
P(x) = A_1 x + A_3 x^3 + A_5 x^5 + A_7 x^7 + A_9 x^9
$$
However, its coefficients are not free. Expanding and eliminating $a, b, c, d$ from $A_5, A_7, A_9$ yields the strict identity:
$$
A_7^2 = 3 A_5 A_9
$$
Odd degree-9 polynomials that are compositions of two cubics live on a 4-parameter algebraic subset of the full 5-parameter space of odd degree-9 polynomials. The unique global minimax polynomial is generally not in this subset.

#### 2) Concrete Counterexample
Consider the approximation interval $[0.2, 1]$. We compare the **global optimal** degree-9 polynomial $p_9^{\ast}$ against a **greedy two-step composition** $P(x) = p_2(p_1(x))$ where each $p_t$ is the optimal degree-3 cubic for the current interval updated by $\ell_{t+1} \leftarrow p_t(\ell_t)$.

*   **Greedy Two-Step (Two Cubics):**
    $$ P(x) \approx 5.236x - 21.440x^3 + 41.654x^5 - 33.592x^7 + 9.030x^9 $$
    Maximum Error: $\approx 0.1114$
*   **Best Odd Degree-9 Polynomial:**
    $$ p_9^{\ast}(x) \approx 5.643x - 28.940x^3 + 74.524x^5 - 83.540x^7 + 33.393x^9 $$
    Maximum Error: $\approx 0.0801$

The global minimax polynomial is significantly better ($\approx 28\%$ lower error). The composed polynomial satisfies the $A_7^2 = 3 A_5 A_9$ constraint almost exactly, while the global optimal deviates by about $7\%$.

> [!info] Why Zolotarev is different
> Nakatsukasa and Freund {% cite nakatsukasaOptimizingHalleyIteration2010 %} highlight that for the sign function, high-degree Zolotarev minimax rationals can be obtained by appropriately composing low-degree ones. This "closure" under composition is a special property of the Zolotarev solution that does **not** carry over to polynomials.
> 
> This is exactly why Polar Express {% cite polarExpress2025 %} proves optimality specifically for their *composition-constrained* problem, and why we compute our high-degree coefficients directly via $F(q_0; \ell)$ rather than via composition.



## Appendix: Verification Code

> [!example]- Python: Scalar Iteration and Centered PE
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
> def centered_pe_coeffs(lam):
>     L2, L4, L6 = lam**2, lam**4, lam**6
>     def F(q):
>         return (-2048*q**9 - 5888*q**8 - 9608*q**7 - 7728*q**6 - 1288*q**5 + 1748*q**4 + 888*q**3 + 8*q**2 - 72*q - 12
>                 + L2 * (4520*q**7 + 9340*q**6 + 10990*q**5 + 8525*q**4 + 3200*q**3 + 80*q**2 - 240*q - 40)
>                 - L4 * (3600*q**5 + 5800*q**4 + 3900*q**3 + 1750*q**2 + 600*q + 100)
>                 + L6 * (1000*q**3 + 1500*q**2 + 750*q + 125))
>     qs = np.linspace(lam + 1e-12, 1 - 1e-12, 100000)
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
>             D = (1 - 5/3*S + 5*P) + lam*(lam**4 - 5/3*S*lam**2 + 5*P)
>             c = 2/D
>             b = -5*c/3*S
>             a = 5*c*P
>             
>             # Chebyshev alternation check to reject extraneous branches
>             p_r = np.sqrt(r2) * (a + b*r2 + c*r2**2)
>             p_l = lam * (a + b*lam**2 + c*lam**4)
>             p_q0 = q0 * (a + b*q0**2 + c*q0**4)
>             p_1 = a + b + c
>             
>             r = np.sqrt(r2)
>             if r2 > 0 and lam < q0 < r < 1:
>                 E_ell, E_r = 1 - p_l, 1 - p_r
>                 E_q0, E_1 = p_q0 - 1, p_1 - 1
>                 errs = [E_ell, E_r, E_q0, E_1]
>                 if all(e > 0 for e in errs) and max(errs) - min(errs) < 1e-4:
>                     return a, b, c, p_1
>     return None
>
>
> # Reproduce the hybrid progression in absolute coordinates
> ell_0 = 1e-3
> alpha, beta, gamma = dwh_coeffs(ell_0)
> ell_1 = dwh_map(ell_0, alpha, beta, gamma)
> u_1 = 1.0
> print(f"DWH: [{ell_1:.6f}, {u_1:.6f}]")
>
> # First centered PE step: u_1 = 1, so the absorbed scale is trivial
> a1, b1, c1, pe1_upper = centered_pe_coeffs(ell_1 / u_1)
> A1, B1, C1 = a1 / u_1, b1 / u_1**3, c1 / u_1**5
> ell_2 = A1*ell_1 + B1*ell_1**3 + C1*ell_1**5
> u_2 = A1*u_1 + B1*u_1**3 + C1*u_1**5
> assert abs(u_2 - pe1_upper) < 1e-12
> print(f"PE1 absolute: [{ell_2:.6f}, {u_2:.6f}]")
> print(f"PE1 actual coeffs: {A1:.15f}, {B1:.15f}, {C1:.15f}")
>
> # Second centered PE step: absorb the non-unit upper endpoint u_2 into the coefficients
> lambda_2 = ell_2 / u_2
> a2, b2, c2, pe2_upper = centered_pe_coeffs(lambda_2)
> A2, B2, C2 = a2 / u_2, b2 / u_2**3, c2 / u_2**5
> ell_3 = A2*ell_2 + B2*ell_2**3 + C2*ell_2**5
> u_3 = A2*u_2 + B2*u_2**3 + C2*u_2**5
> assert abs(u_3 - pe2_upper) < 1e-12
> print(f"PE2 absolute: [{ell_3:.6f}, {u_3:.6f}]")
> print(f"PE2 actual coeffs: {A2:.15f}, {B2:.15f}, {C2:.15f}")
> print(f"Optional final normalization: [{ell_3/u_3:.6f}, 1.000000]")
> ```


---

## References

{% bibliography %}

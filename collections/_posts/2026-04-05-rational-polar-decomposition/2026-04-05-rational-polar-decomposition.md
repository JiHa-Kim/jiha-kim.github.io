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

Iterative methods for the polar factor work by applying a scalar map $f(x)$ to the singular values $\sigma_i$ of $A$. The hybrid uses two complementary regimes:

- DWH uses a constrained rational step and keeps the image inside $[\ell_{t+1}, 1]$.
- PE uses a centered polynomial step and maps the current interval to $[1-E_t, 1+E_t]$.

Polynomials are more optimized for hardware than rational functions, but they are less stable. DWH is more stable and efficient for the hard early regime, but it is more expensive. The hybrid approach combines the strengths of both.

### 2.1 The Two Scalar Maps

- **DWH (rational)**: the Dynamic Weighted Halley map {% cite nakatsukasaOptimizingHalleyIteration2010 %} applies the rational function

$$
f_{\text{DWH}}(x) = x\frac{a + bx^2}{1 + cx^2}
$$

with coefficients chosen for the current floor $\ell$.

- **Polar Express (polynomial)**: the degree-5 PE map {% cite polarExpress2025 %} applies the odd polynomial

$$
p(x) = x(a + bx^2 + cx^4) = ax + bx^3 + cx^5
$$

with coefficients chosen on the current interval. PE is used only after the spectrum is already in the easy regime, where a polynomial update is cheaper than another rational step.

Operationally, the hybrid is simple: use DWH to escape the hard early regime, then use centered PE steps to finish. [Section 6](#section-6) gives the exact optimization problems and coefficient formulas.

### 2.2 Motivation: Crossing the Crossover

The following widget plots the resulting **lower endpoint** after one or more scalar steps. For PE, the full image interval is centered around $1$; the chart shows only its lower edge. Larger values closer to $1$ are better.

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

To show the final result, we can normalize the matrix by the upper endpoint to get a spectral-norm certificate.

$$
\frac{1}{1.002426}[0.997574,1.002426] = [0.995160,1]
$$

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

The implementation uses one DWH step followed by two PE steps and exactly two tall rectangular GEMMs. To optimize performance on well-conditioned matrices, we estimate the actual lower bound $\ell_{\text{est}}$ online and fetch specialized coefficients from a densely sweeped cache if the matrix is "easy" enough to warrant a more aggressive update.

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

    # 4. Step 1: DWH (Initial floor $\ell_{0} = 10^{-3}$)
    $S \leftarrow \gamma_{0} u \Delta + \tilde{G}$
    $(L, \tau) \leftarrow$ @SafeCholesky($S$)
    $S_{\text{inv}} \leftarrow$ @cholesky_inverse($L$)
    
    # Estimate actual lower bound and fetch specialized coefficients
    $\ell_{\text{est}}^2 \leftarrow \max(\ell_0^2, \frac{1}{u \operatorname{tr}(S_{\text{inv}})} - \gamma_{0})$
    
    if $\ell_{\text{est}}^2 > \ell_{\text{fast}}^2$:
        $K \leftarrow \frac{1}{\sqrt{u}} I$ # Fast-path: bypass DWH
    else:
        if $\ell_{\text{est}}^2 > \ell_0^2$:
            # Non-trivial improvement detected: fetch from densely sweeped cache
            $(g_I, g_B, g_H, g_{H^2}, \alpha_0, \beta_0) \leftarrow$ @FetchCachedCoeffs($\sqrt{\ell_{\text{est}}^2}$)

        $H_{0} \leftarrow \gamma_{0} u D S_{\text{inv}} D$ # Bounded resolvent
        $H_{sq} \leftarrow$ @Sym($H_{0}^{2}$) # Half-FLOP SYRK
        $B \leftarrow g_{I} I + g_{B} B + g_{H} H_{0} + g_{H^2} H_{sq}$ # Zero-GEMM step
        $K \leftarrow \frac{1}{\sqrt{u}}(\alpha_{0} I + \beta_{0} H_{0})$
    $(\ell_1, u_1) \leftarrow$ scalar image interval after the DWH step

    # 5. Step 2: Centered PE Cleanup 1
    $\lambda_1 \leftarrow \ell_1 / u_1$
    $(a_1, b_1, c_1) \leftarrow$ @CenteredPECoeffs($\lambda_1, u_1$)
    
    $B_{sq} \leftarrow$ @Sym($B^2$)
    $Z \leftarrow b_1 B + c_1 B_{sq}$         # Stable "non-identity" part
    $K \leftarrow @Sym(K Z + a_1 K)$          # Split evaluation for $K$
    $B \leftarrow @Sym(a_1^2 B + 2a_1 B Z + B Z^2)$ # Split evaluation for $B$
    
    $(\ell_2, u_2) \leftarrow (p_1(\ell_1), p_1(u_1))$

    # 6. Step 3: Centered PE Cleanup 2
    $\lambda_2 \leftarrow \ell_2 / u_2$
    $(a_2, b_2, c_2) \leftarrow$ @CenteredPECoeffs($\lambda_2, u_2$)
    
    $B_{sq} \leftarrow$ @Sym($B^2$)
    $Z \leftarrow b_2 B + c_2 B_{sq}$         # Uniform stable split default
    $K \leftarrow @Sym(K Z + a_2 K)$
    # Note: B is not needed after the terminal step unless tracking final error

    # 7. Final Reconstitution
    $K \leftarrow \operatorname{diag}(d_i^{1/2})$ @Sym($K$) # Invert in-place scaling on K rows
    return $X K$ # Optional: divide once more by u_3 if a bounded final operator is required
```
</div>

### 3.3 Numerical Stability in Low Precision {#numerical-stability-in-low-precision}

To ensure robust convergence in BF16/FP16, we adopt several stability "tricks" that prevent catastrophic cancellation and maintain the PSD property of the Gram objects.

> [!tip] Low-Precision Best Practices
> 
> 1. **Stable Gram Newton-Schulz Form**: Avoid explicitly adding a large identity to a half-precision matrix ($aI + Z$). Instead, compute the non-identity part $Z$ first and fold $a K$ into the accumulate path: $K \leftarrow K Z + a K$.
> 2. **Restart (Rebuild the Gram)**: Periodically reconstruct $B \leftarrow XX^\top$ (or $B \leftarrow K^\top (X^\top X) K$) to clear accumulated rounding errors and spurious negative eigenvalues.
> 3. **Deliberate Symmetry Enforcement**: Rounding error in SYMM/GEMM creates small asymmetries. We apply `@Sym(B) = 1/2(B+B^T)` at every sensitive choke point.
> 4. **Identity-Centered Evaluation**: When $B \approx I$, one can *optionally* evaluate in the basis $E = B - I$. This is primarily a late-stage micro-optimization for when the spectrum is in a tight band $[1-\delta, 1+\delta]$ and the accumulator path is in FP32. For $P(B) = aI + bB + cB^2$, this is:
>    $$ P(B) = (a+b+c)I + (b+2c)E + cE^2. $$
>    **Caution**: To avoid the cancellation trap, form $E$ and $E^2 = EE$ "honestly" in FP32; do not use the identity $E^2 = B^2 - 2B + I$ in half precision.
> 5. **Absorbing Scaling**: Absorbing the current upper endpoint $u_t$ into the next step's coefficients keeps the effective spectrum bounded and avoids nasty dynamic-range issues.
> 6. **Bounded-Resolvent Basis**: For the hard early regime, using $(I+cB)^{-1}$ prevents the large cross-multiplications that plague standard polynomial expansions. 

Implementation note: treat $\tilde{G}, S, H, B$ as symmetric objects and use symmetric kernels conceptually (SYRK/SYMM/SYR2K). Only apply `@Sym(·)` at choke points (right before factorization, and optionally right before the final $Q \leftarrow X K$ matmul).

> [!remark] Terminal Normalization
> If the returned update must satisfy a spectral-norm or 1-Lipschitz certificate, normalize once at the very end by the final upper endpoint $u_3$. For the running example,
> $$
> \frac{1}{1.002426}[0.997574,1.002426] = [0.995160,1].
> $$
> That terminal normalization is a certification step, not part of the intermediate PE iteration.

---

## 4. Design Constants ($\ell_0 = 10^{-3}$)

For the design floor $\ell_0 = 10^{-3}$, the implementation constants are precomputed offline in FP64. In the dynamic path, if $\ell_{\text{est}} > \ell_0$, the algorithm fetches a set of precomputed coefficients from a **densely sweeped cache** (a lookup table indexed by $\ell$). This allows the DWH and PE steps to be significantly more aggressive, leading to tighter final error bounds or potentially fewer iterations for easy matrices.

| Step     | Parameters                    | Values                                                                      |
| :------- | :---------------------------- | :-------------------------------------------------------------------------- |
| **DWH**  | $g_I, g_B, g_H, g_{H^2}$      | 0.030883301527615, 0.968872554082809, 3.906861822017413, -3.937745123545028 |
|          | $\gamma_0, \alpha_0, \beta_0$ | 0.000062499017684, 0.984313239818915, 251.007791810856                      |
| **PE 1** | $a_1, b_1, c_1$               | 3.824452920237891, -7.181066039236940, 4.513346248799179                    |
| **PE 2** | $a_2, b_2, c_2$               | 1.901427287944732, -1.279060386064908, 0.377917707130065                    |

These are the actual PE coefficients used by the implementation. The normalized centered coefficients and the absorbed-scaling relation are derived in Section 6.3.

---

## 5. Discussion

The design has three practical advantages.

- **Right tool for each regime**: DWH is used only where rational lifting matters, namely the hard early regime. Once the interval is already tight, PE gives cheaper cleanup on the small side.
- **Low-precision robustness**: Column normalization, bounded resolvents, and symmetric kernels keep the Gram-side computation numerically controlled. The use of **stable split evaluation** and **identity-centering** (see [Section 3.3](#numerical-stability-in-low-precision)) specifically targets the failure modes of BF16/FP16.
- **Clean certification boundary**: If the returned update must satisfy a spectral-norm or Lipschitz bound, that normalization happens once at the end. The iteration itself remains centered and aggressive until the last step.

This gives a method that is both practical for inner-loop training and faithful to the underlying scalar approximation problems.

---

## 6. Technical Derivations {#section-6}

### 6.1 Why Greedy Is Optimal

Both DWH and Polar Express approximate the **matrix sign function**. With $T$ remaining steps, the natural scalar problem is
$$
\min_{p_1,\dots,p_T \in \mathcal P_d^{\mathrm{odd}}}\ \max_{x \in [-u,-\ell] \cup [\ell,u]} \vert \operatorname{sign}(x) - (p_T \circ \cdots \circ p_1)(x) \vert.
$$
Because the target and the approximants are odd, this is equivalent to the positive-side problem
$$
\min_{p_1,\dots,p_T \in \mathcal P_d^{\mathrm{odd}}}\ \max_{x \in [\ell,u]} \vert 1 - (p_T \circ \cdots \circ p_1)(x) \vert.
$$

There is also a scale invariance. Writing $\lambda = \ell/u$, any odd degree-$d$ polynomial $q$ on $[\lambda,1]$ induces one on $[\ell,u]$ by
$$
p(x) = q\left(\frac{x}{u}\right).
$$
So the state is really the normalized interval parameter $\lambda$, not the two endpoints separately.

> [!theorem] Greedy Is Optimal
> Fix a degree $d$ and a number of remaining steps. Then the globally optimal first move is the one-step minimax polynomial on the current normalized interval $[\lambda,1]$. Repeating the same rule after each update is globally optimal.

> [!proof]- Greedy is Optimal
> We prove that the $T$-step minimax problem decomposes into $T$ independent one-step minimax problems.
> 
> 1. **State Space**: The state of the system is the normalized condition $\lambda = \ell/u \in (0, 1]$. Since the target (the sign function) and the approximants (odd polynomials) are scale-invariant, any map $p$ on $[\ell, u]$ with image $[\ell', u']$ is equivalent to a map on $[\lambda, 1]$ with normalized image $\lambda_{\text{next}} = \ell'/u'$.
> 
> 2. **Value Function**: Let $V_t(\lambda)$ be the minimum possible worst-case error achievable in $t$ remaining steps starting from $[\lambda, 1]$. By definition, $V_t(\lambda)$ is monotonically decreasing in $\lambda$; a tighter starting interval $(\lambda \to 1)$ always yields a smaller final error.
> 
> 3. **Bellman Optimality**: The optimal $(t+1)$-step error satisfies:
>    $$ V_{t+1}(\lambda) = \inf_{p} V_t\left( \lambda_{\text{next}}(p) \right) $$
>    where $\lambda_{\text{next}}(p) = \frac{\min p([\lambda, 1])}{\max p([\lambda, 1])}$. To minimize the tail error $V_t$, we must choose the first step $p$ to **maximize** $\lambda_{\text{next}}(p)$.
> 
> 4. **One-Step Optimizer**: For any odd polynomial $p$ with minimax error $E = \sup_{x \in [\lambda, 1]} |1 - p(x)|$, its image is contained in $[1-E, 1+E]$ by equioscillation. Thus:
>    $$ \lambda_{\text{next}}(p) \le \frac{1-E}{1+E} $$
>    Equality is achieved if and only if $p$ is the one-step minimax polynomial (which oscillates between $1-E$ and $1+E$).
> 
> 5. **Conclusion**: Since $f(E) = \frac{1-E}{1+E}$ is decreasing in $E$, the ratio $\lambda_{\text{next}}$ is maximized precisely when the one-step error $E$ is minimized. Therefore, the greedy move is globally optimal.

### 6.2 Why the Rational Reduction Works

The Dynamic Weighted Halley (DWH) iteration uses a degree-$(3, 2)$ rational map. Because this map is monotonically increasing on $[\ell, 1]$, its minimum occurs at $f(\ell)$ and its maximum at $f(1)$. This is why the rational problem can be reduced to a one-sided constrained problem without losing optimality.

> [!proof]- Why the Rational Reduction Is Valid
> For odd maps, the two-sided sign-approximation error mirrors across the origin, so only the positive interval matters. After normalizing so that the top endpoint satisfies $f(1)=1$, a monotone rational step is judged entirely by how far it lifts the lower endpoint while keeping the image inside $[0,1]$.
>
> Nakatsukasa and Freund {% cite nakatsukasaOptimizingHalleyIteration2010 %} show that, for the Zolotarev rational family, this one-sided constrained problem is equivalent up to scaling to the original two-sided sign-approximation problem. So in the rational case we can impose the no-overshoot constraint without giving up optimality.

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

### 6.3 Why the Polynomial Reduction Fails

For Polar Express we use the degree-5 odd polynomial
$$
p(x) = x(a + bx^2 + cx^4).
$$
Unlike the rational case, the optimal polynomial is not monotone on its tracked interval. It equioscillates. That is why the rational reduction does **not** carry over.

> [!proof]- Why the Polynomial Reduction Fails
> Let $p$ be the centered minimax polynomial on $[\ell,1]$, and suppose its image is
> $$
> [1-E,1+E]
> $$
> with $E > 0$. If we force a no-overshoot constraint by rescaling to
> $$
> \hat{p}(x) = \frac{p(x)}{1+E},
> $$
> then $\hat{p}(1)=1$ but
> $$
> \hat{p}(\ell) = \frac{1-E}{1+E},
> $$
> so the new worst-case error is
> $$
> 1 - \hat{p}(\ell) = \frac{2E}{1+E} > E.
> $$
> Thus the constrained one-sided polynomial problem is strictly worse than the centered minimax problem. It is not an equivalent reformulation.

So PE should be formulated directly as the centered one-step problem
$$
\min_{a,b,c} \max_{x \in [\ell, 1]} \vert 1 - p(x) \vert = E
$$
By the Chebyshev Equioscillation Theorem, the optimal polynomial oscillates between $1-E$ and $1+E$. In the one-step setting, the image interval is therefore
$$
[p(\ell), p(1)] = [1-E, 1+E],
$$
which is centered around $1$.

In the full Polar Express recurrence this balanced-endpoint property is maintained across iterations by tracking both $\ell_t$ and $u_t$. In the implementation code in Appendix G, after the finite-precision cushioning step the polynomial is explicitly rescaled so that
$$
1 - p(\ell_t) = p(u_t) - 1,
$$
before the next step.

To apply such a centered polynomial on a general interval $[\ell_t, u_t]$, set $\lambda_t = \ell_t/u_t$ and solve first for the normalized coefficients of
$$
q_t(y) = a_t y + b_t y^3 + c_t y^5
$$
on $[\lambda_t, 1]$. Then the actual polynomial on the original interval is
$$
p_t(x) = q_t\left(\frac{x}{u_t}\right) = \frac{a_t}{u_t}x + \frac{b_t}{u_t^3}x^3 + \frac{c_t}{u_t^5}x^5.
$$
This is the precise sense in which the PE scale is absorbed into the coefficients. For the first PE step after DWH, $u_1 = 1$, so nothing nontrivial happens yet. The first genuine absorbed scale appears in the second PE step.

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

The degree-9 equation $F(q_0; \ell) = 0$ contains extraneous branches. In practice, we keep the **unique** root that satisfies the correct ordering $\ell < q_0 < r < 1$ and the expected alternation pattern.

<details class="box-proof" markdown="1">
<summary>Theory: Sketch of the Centered PE Derivation</summary>

The minimax odd polynomial $p(x) = ax + bx^3 + cx^5$ minimizes $\max_{x \in [\ell, 1]} |1 - p(x)|$. By the **Chebyshev Equioscillation Theorem**, since the basis $\{x, x^3, x^5\}$ has dimension 3, the optimal error must alternate at four points. For the degree-5 odd optimum, those points are
$$
\ell,\ q_0,\ r,\ 1,
$$
where $q_0$ and $r$ are interior critical points.

The equioscillation conditions are:
1. $p(\ell) = 1 - E, \quad p(r) = 1 - E \implies p(\ell) = p(r)$
2. $p(q_0) = 1 + E, \quad p(1) = 1 + E \implies p(q_0) = p(1)$

Because $q_0$ and $r$ are roots of $p'(x) = 5cx^4 + 3bx^2 + a$, we can write
$$
a = 5cq_0^2r^2,\qquad b = -\frac{5c}{3}(q_0^2 + r^2).
$$
The condition $p(q_0) = p(1)$ then yields
$$
r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}.
$$
Substituting this into $p(\ell) = p(r)$ gives the scalar equation $F(q_0;\ell)=0$. Finally, imposing the centering condition $p(1)+p(\ell)=2$ gives the denominator $D$ and therefore the closed-form coefficients above.

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

A natural question is whether high-degree optimal polynomials can be built by composing lower-degree optimal ones. For Zolotarev rationals the answer is yes. For polynomials it is no.

The obstruction is structural. Even the composition of two odd cubics,
$$
p_1(x) = ax + bx^3,\qquad p_2(x) = cx + dx^3,
$$
produces only a restricted subset of degree-9 odd polynomials:
$$
P(x) = p_2(p_1(x)) = A_1 x + A_3 x^3 + A_5 x^5 + A_7 x^7 + A_9 x^9.
$$
Its coefficients are not free. Expanding and eliminating $a, b, c, d$ from $A_5, A_7, A_9$ yields the identity
$$
A_7^2 = 3 A_5 A_9
$$
so compositions of two cubics occupy a proper algebraic subset of the full degree-9 odd polynomial space. A generic global minimax polynomial will not lie in that subset.

This is not just a structural curiosity. On the interval $[0.2,1]$, the **greedy two-step cubic composition**
$$
P(x) = p_2(p_1(x))
$$
has maximum error about $0.1114$, while the **best odd degree-9 polynomial** has maximum error about $0.0801$.

*   **Greedy Two-Step (Two Cubics):**
    $$ P(x) \approx 5.236x - 21.440x^3 + 41.654x^5 - 33.592x^7 + 9.030x^9 $$
    Maximum Error: $\approx 0.1114$
*   **Best Odd Degree-9 Polynomial:**
    $$ p_9^{\ast}(x) \approx 5.643x - 28.940x^3 + 74.524x^5 - 83.540x^7 + 33.393x^9 $$
    Maximum Error: $\approx 0.0801$

So the composed polynomial is materially worse. This is why polynomial coefficients must be recomputed step by step rather than generated by composition.

> [!info] Why Zolotarev is different
> Nakatsukasa and Freund {% cite nakatsukasaOptimizingHalleyIteration2010 %} highlight that for the sign function, high-degree Zolotarev minimax rationals can be obtained by appropriately composing low-degree ones. This "closure" under composition is a special property of the Zolotarev solution that does **not** carry over to polynomials.
> 
> This is exactly why Polar Express {% cite polarExpress2025 %} proves optimality specifically for their *composition-constrained* problem, and why we compute our high-degree coefficients directly via $F(q_0; \ell)$ rather than via composition.



### 6.5 Summary Table

| Aspect                                  | Rational: DWH / Zolotarev                        | Polynomial: PE                                                         |
| :-------------------------------------- | :----------------------------------------------- | :--------------------------------------------------------------------- |
| Optimal shape                           | Monotone on $[\ell,1]$                           | Equioscillating on the tracked interval                                |
| Reduction to one-sided constrained form | Exact up to scaling                              | Not exact                                                              |
| Composition of optima                   | Optimal class is closed under composition        | Optimal class is not closed under composition                          |
| Global scale                            | Usually written with upper endpoint fixed at $1$ | Any non-unit upper endpoint can be absorbed into the next coefficients |
| Offline design                          | Closed-form rational formulas                    | Interval-dependent equioscillation and root-finding                    |

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
> a1_eff, b1_eff, c1_eff = a1 / u_1, b1 / u_1**3, c1 / u_1**5
> ell_2 = a1_eff*ell_1 + b1_eff*ell_1**3 + c1_eff*ell_1**5
> u_2 = a1_eff*u_1 + b1_eff*u_1**3 + c1_eff*u_1**5
> assert abs(u_2 - pe1_upper) < 1e-12
> print(f"PE1 absolute: [{ell_2:.6f}, {u_2:.6f}]")
> print(f"PE1 actual coeffs: {a1_eff:.15f}, {b1_eff:.15f}, {c1_eff:.15f}")
>
> # Second centered PE step: absorb the non-unit upper endpoint u_2 into the coefficients
> lambda_2 = ell_2 / u_2
> a2, b2, c2, pe2_upper = centered_pe_coeffs(lambda_2)
> a2_eff, b2_eff, c2_eff = a2 / u_2, b2 / u_2**3, c2 / u_2**5
> ell_3 = a2_eff*ell_2 + b2_eff*ell_2**3 + c2_eff*ell_2**5
> u_3 = a2_eff*u_2 + b2_eff*u_2**3 + c2_eff*u_2**5
> assert abs(u_3 - pe2_upper) < 1e-12
> print(f"PE2 absolute: [{ell_3:.6f}, {u_3:.6f}]")
> print(f"PE2 actual coeffs: {a2_eff:.15f}, {b2_eff:.15f}, {c2_eff:.15f}")
> print(f"Optional final normalization: [{ell_3/u_3:.6f}, 1.000000]")
> ```


---

## References

{% bibliography %}

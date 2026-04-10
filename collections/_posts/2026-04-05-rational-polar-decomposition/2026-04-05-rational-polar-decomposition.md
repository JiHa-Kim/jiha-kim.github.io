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

> [!table] Design Constants for $\ell_0 = 10^{-3}$
> | Step     | Parameters                    | Values                                                                      |
> | :------- | :---------------------------- | :-------------------------------------------------------------------------- |
> | **DWH**  | $g_I, g_B, g_H, g_{H^2}$      | 0.030883301527615, 0.968872554082809, 3.906861822017413, -3.937745123545028 |
> |          | $\gamma_0, \alpha_0, \beta_0$ | 0.000062499017684, 0.984313239818915, 251.007791810856                      |
> | **PE 1** | $a_1, b_1, c_1$               | 3.824452920237891, -7.181066039236940, 4.513346248799179                    |
> | **PE 2** | $a_2, b_2, c_2$               | 1.901427287944732, -1.279060386064908, 0.377917707130065                    |

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

Both DWH and Polar Express are scalar maps applied to singular values. After rescaling so the active singular-value interval is $[\ell,1]$, a single step applies an odd map $g$ to every $\sigma\in[\ell,1]$. The post-step interval is
$$
g([\ell,1])=[m,M],\qquad m:=\min_{x\in[\ell,1]} g(x),\quad M:=\max_{x\in[\ell,1]} g(x).
$$
Because the polar/sign target is scale-invariant up to a final normalization, it is natural to renormalize after each step by dividing by the upper edge $M$, so the state stays in the canonical form $[\ell,1]$. Under this normalization, the next state is the single number
$$
\ell_+ \;=\;\frac{m}{M}\in(0,1].
$$
Thus the step quality is completely determined by how much it improves the floor $\ell\mapsto \ell_+$.

Let $\mathcal F_d$ denote an allowed one-step family (odd degree-$d$ polynomials, or a fixed rational type), assumed closed under positive scaling. For $t\ge 0$, define the value function
$$
V_t(\ell)\;:=\;\sup\{\ell_t:\exists\ g_1,\dots,g_t\in\mathcal F_d\ \text{with}\ \ell_{i+1}=T(\ell_i;g_{i+1}),\ \ell_0=\ell\},
$$
where the transition $T(\ell;g)$ is
$$
T(\ell;g):=\frac{\min_{x\in[\ell,1]} g(x)}{\max_{x\in[\ell,1]} g(x)}.
$$

> [!lemma] Monotonicity of the value function
> For each $t$, $V_t(\ell)$ is nondecreasing in $\ell$.

> [!proof]-
> If $\ell_1\le \ell_2$, then $[\ell_2,1]\subseteq[\ell_1,1]$. Any admissible sequence of maps acting on $[\ell_2,1]$ is also admissible on $[\ell_1,1]$, and normalization by the upper edge preserves the inclusion ordering. Therefore the best achievable floor starting from $\ell_2$ cannot be smaller than the best achievable floor starting from $\ell_1$.

> [!theorem] Greedy is optimal for floor maximization
> Fix a step family $\mathcal F_d$ and an integer $T\ge 1$. An optimal $T$-step strategy is obtained by choosing, at each state $\ell$, a map $g\in\mathcal F_d$ that maximizes the one-step floor $T(\ell;g)$, and then repeating this rule on the updated state.

> [!proof]-
> The Bellman recursion is
> $$
> V_{t+1}(\ell)=\sup_{g\in\mathcal F_d} V_t\!\left(T(\ell;g)\right).
> $$
> By the lemma, $V_t$ is nondecreasing, so the maximizing choice of $g$ is any map that maximizes $T(\ell;g)$. Applying the same argument at every subsequent state yields an optimal policy by induction.

This establishes the structural reason both DWH and PE can be designed one step at a time: once the step family and normalization convention are fixed, the globally best sequence is obtained by repeatedly solving the one-step floor-maximization problem on the current interval.

### 6.2 The Bounded Max-Min Framework

For the degree-$(3,2)$ odd rational family used by DWH,
$$
f(x)=x\frac{a+bx^2}{1+cx^2},\qquad a,b,c>0,
$$
DWH is derived by directly optimizing the floor under a global no-overshoot constraint. This is the bounded max-min viewpoint used in the DWH/QDWH line and in the Zolotarev-based Zolo-pd formulation.

#### 6.2.1 Scaling equivalence of minimax and bounded max-min

Let $\mathcal F$ be any class of odd approximants closed under positive scaling. Consider the centered minimax problem on $[\ell,1]$,
$$
E_* \;=\; \inf_{R\in\mathcal F}\ \max_{x\in[\ell,1]} |1-R(x)|,
$$
and the bounded max-min problem,
$$
m_* \;=\; \sup_{S\in\mathcal F}\ \min_{x\in[\ell,1]} S(x)
\quad\text{s.t.}\quad
0\le S(x)\le 1\ \ \forall x\in[0,1].
$$

> [!lemma] Scaling equivalence: centered minimax vs bounded max-min
> The optimal values satisfy
> $$
> m_*=\frac{1-E_*}{1+E_*}
> \qquad\Longleftrightarrow\qquad
> E_*=\frac{1-m_*}{1+m_*}.
> $$
> In particular, the induced interval contraction $[\ell,1]\mapsto[m_*,1]$ is the same whether one solves the centered minimax problem and then rescales, or solves the bounded max-min problem directly.

> [!proof]-
> (1) From centered minimax to bounded max-min: If $R\in\mathcal F$ achieves centered error $E$, then on $[\ell,1]$,
> $$
> 1-E\le R(x)\le 1+E.
> $$
> Define $S(x)=R(x)/(1+E)$. Then $S\in\mathcal F$, $S(x)\le 1$ on $[0,1]$, and
> $$
> \min_{x\in[\ell,1]} S(x)\ge \frac{1-E}{1+E}.
> $$
> Taking the infimum over $R$ yields $m_*\ge (1-E_*)/(1+E_*)$.
>
> (2) From bounded max-min to centered minimax: Let $S\in\mathcal F$ be feasible with $m=\min_{[\ell,1]}S$ and $\max_{[0,1]}S=1$ (this can be assumed at optimum by scaling up until the constraint is tight). Set $\alpha=2/(1+m)$ and $R(x)=\alpha S(x)$. Then on $[\ell,1]$, $R(x)\in[\alpha m,\alpha]$ and
> $$
> \alpha-1 = 1-\alpha m = \frac{1-m}{1+m},
> $$
> so $\max_{[\ell,1]}|1-R(x)|\le (1-m)/(1+m)$. Taking the supremum over feasible $S$ yields $E_*\le (1-m_*)/(1+m_*)$.
>
> Combining both directions gives the stated identities.

The scaling equivalence is the mechanism behind the common practice of moving between a floor-maximized bounded map $S$ and a centered equioscillating map $R$: the normalization changes, but the condition-number update $\kappa_+=1/m_*$ is preserved.

#### 6.2.2 One-step design as a bounded max-min problem

With the canonical normalization $f(1)=1$ (equivalently $c=a+b-1$), a stable one-step update is specified by the global constraint
$$
0<f(x)\le 1\qquad\forall x\in[\ell,1],
$$
and the objective is to maximize the next floor
$$
m(f)=\min_{x\in[\ell,1]} f(x).
$$

> [!lemma] Error under no-overshoot
> If $f(1)=1$ and $0<f(x)\le 1$ for all $x\in[\ell,1]$, then
> $$
> \sup_{x\in[\ell,1]}|1-f(x)| = 1-\min_{x\in[\ell,1]} f(x)=1-m(f).
> $$

> [!proof]-
> Under the constraint $f(x)\le 1$, the quantity $1-f(x)$ is nonnegative on $[\ell,1]$, so
> $$
> \sup_{x\in[\ell,1]}|1-f(x)|
> =
> \sup_{x\in[\ell,1]}(1-f(x))
> =
> 1-\inf_{x\in[\ell,1]} f(x)
> =
> 1-m(f).
> $$

> [!proposition] Finite characterization of the floor
> For any continuous $f$, the minimum on $[\ell,1]$ is attained either at the endpoint $x=\ell$ or at an interior critical point $x\in(\ell,1)$ where $f'(x)=0$. Hence
> $$
> m(f)=\min\Bigl\{f(\ell),\ \min_{x\in(\ell,1):\,f'(x)=0} f(x)\Bigr\}.
> $$

This reduces the one-step optimization to controlling $f(\ell)$ and the values at any interior stationary minima $x_M\in(\ell,1)$. For the DWH family, the optimum lies on a one-parameter boundary, and the maximized floor is obtained by equalizing the active minima (endpoint and/or interior), yielding closed-form coefficients.

> [!theorem] Optimal DWH coefficients
> For a design floor $\ell\in(0,1]$, the DWH coefficients $(a,b,c)$ that solve the bounded max-min problem are:
> $$
> \zeta=\left(\frac{4(1-\ell^2)}{\ell^4}\right)^{1/3},\qquad r=\sqrt{1+\zeta},
> $$
> $$
> a = r + \frac{1}{2}\sqrt{8 - 4\zeta + \frac{8(2 - \ell^2)}{\ell^2 r}}, \qquad
> b = \frac{(a-1)^2}{4}, \qquad
> c = a+b-1.
> $$

In the Gram-space iteration ($B=A^\top A$), the DWH step is evaluated efficiently in a bounded-resolvent basis. Let $H_0=(I+cB)^{-1}=\gamma(\gamma I+B)^{-1}$. Using the identity $BH_0=(I-H_0)/c$, the update can be expanded into
$$
B \leftarrow g_I I + g_B B + g_H H_0 + g_{H^2} H_0^2,
$$
so the step requires only one symmetric squaring $H_0^2$. The orientation factor $K$ is updated via
$$
K \leftarrow K(\alpha_0 I+\beta_0 H_0),
\qquad
\alpha_0=\frac{b}{c},\quad \beta_0=a-\frac{b}{c}.
$$

### 6.3 Composition and the Zolotarev Advantage

The bounded max-min and centered minimax formulations are interchangeable for any scale-closed odd family $\mathcal F$. In practice, the two families are often presented differently:

- Rationals (DWH/Zolotarev) are naturally written in the bounded form $0<f\le 1$, which aligns with resolvent-based evaluation and interval tracking $[\ell,1]\mapsto[m,1]$.
- Polynomials (PE) are naturally written in centered form $p([\ell,1])\subseteq[1-E,1+E]$, which aligns with monomial evaluation and symmetric error control.

The structural difference exploited by Zolo-pd is not the existence of a one-sided formulation, but a special closure property of Zolotarev minimax rationals: high-type minimax solutions can be obtained by composing low-type minimax maps.

> [!info] Zolotarev composition reduction
> Zolotarev sign approximants satisfy a composition law: composing $k$ suitably scaled type-$(2r+1,2r)$ Zolotarev maps yields a scaled Zolotarev map of type $\bigl((2r+1)^k,(2r+1)^k-1\bigr)$. This exact closure under composition is what lets Zolo-pd reach very high effective types in very few iterations.

A separate, purely stability-oriented point is that endpoint-only constraints do not control interior behavior.

> [!caution] The endpoint trap
> Endpoint normalization alone does not bound the interior. Consider the odd cubics
> $$
> p_k(x)=(1+k)x-kx^3,\qquad k>0,
> $$
> which satisfy $p_k(1)=1$. For any $\ell\in(0,1)$,
> $$
> p_k(\ell)=\ell+k(\ell-\ell^3)\xrightarrow[k\to\infty]{}\infty,
> $$
> and $p_k$ has an interior maximizer at $x=\sqrt{(1+k)/(3k)}\approx 1/\sqrt{3}$ with $p_k(x)=\Theta(k)$.
>
> Stable step design therefore requires a global constraint (bounded max-min) or a global uniform-error bound (centered minimax), not an endpoint-only condition.

For Polar Express, the degree-5 odd polynomial
$$
p(x)=ax+bx^3+cx^5
$$
is obtained by solving the centered minimax problem on the current normalized interval, producing the equioscillating optimal coefficients.

> [!theorem] Closed-form centered PE coefficients
> Fix $0<\ell<1$. The centered minimax coefficients for $p(x)=ax+bx^3+cx^5$ on $[\ell,1]$ are uniquely determined by the interior equioscillation root $q_0\in(\ell,1)$ of the degree-9 polynomial
> $$
> F(q_0;\ell)=F_0(q_0)+\ell^2 F_1(q_0)-\ell^4 F_2(q_0)+\ell^6 F_3(q_0)=0,
> $$
> where $F_0,\dots,F_3$ are the pre-defined algebraic moments. The coefficients are
> $$
> c=\frac{2}{D},\qquad b=-\frac{5c}{3}(q_0^2+r^2),\qquad a=5cq_0^2r^2,
> $$
> with auxiliary parameters $r^2$ and $D$ derived from the equioscillation conditions.

> [!proof]- Derivation sketch
> By Chebyshev alternation, the degree-5 odd minimax error alternates at four points on $[\ell,1]$: the endpoints $\ell,1$ and two interior critical points $q_0,r$.
>
> 1. The critical points satisfy $p'(x)=5cx^4+3bx^2+a=0$ at $x^2=q_0^2$ and $x^2=r^2$, yielding $a$ and $b$ in terms of $c,q_0,r$.
> 2. The equal-upper-peak condition $p(q_0)=p(1)$ determines $r^2$ as a rational function of $q_0$:
> $$
> r^2=\frac{2q_0^3+4q_0^2+6q_0+3}{5(2q_0+1)}.
> $$
> 3. Substituting $r^2(q_0)$ into the lower-peak matching $p(\ell)=p(r)$ produces the characteristic equation $F(q_0;\ell)=0$.
> 4. The centering $p(1)+p(\ell)=2$ fixes the remaining scale through $D$, hence $c$.

### 6.4 The Structural Limits of Composition

A natural question is whether high-degree optimal polynomials can be built by composing lower-degree optimal ones. Unlike Zolotarev rationals, odd polynomials are not closed under composition in a way that preserves minimax optimality.

> [!lemma] The composition obstruction
> The set of degree-9 odd polynomials representable as a composition of two odd cubics, $P(x)=p_2(p_1(x))$, is a proper algebraic subset of the space of degree-9 odd polynomials. In particular, the coefficients of any such composition satisfy
> $$
> A_7^2 = 3A_5A_9.
> $$
> Consequently, a generic minimax degree-9 odd polynomial does not admit such a representation.

> [!proof]
> Let $p_1(x)=ax+bx^3$ and $p_2(x)=cx+dx^3$. Expanding,
> $$
> \begin{aligned}
> P(x)
> &=c(ax+bx^3)+d(ax+bx^3)^3 \\
> &=(ca)x+(cb+da^3)x^3+(3da^2b)x^5+(3dab^2)x^7+(db^3)x^9.
> \end{aligned}
> $$
> Let $A_5=3da^2b$, $A_7=3dab^2$, and $A_9=db^3$. Then
> $$
> A_7^2=(3dab^2)^2=9d^2a^2b^4
> \qquad\text{and}\qquad
> 3A_5A_9=3(3da^2b)(db^3)=9d^2a^2b^4,
> $$
> hence $A_7^2=3A_5A_9$ holds identically for all cubic compositions.

This obstruction is not merely algebraic: it limits approximation quality because the composed family cannot reach the full best-approximation degrees of freedom.

### 6.5 Summary Table

| Aspect                                   | Rational: DWH / Zolotarev                          | Polynomial: PE                                        |
| :--------------------------------------- | :------------------------------------------------- | :---------------------------------------------------- |
| Canonical one-step objective             | Bounded max-min: maximize $\min f$ s.t. $0<f\le 1$ | Centered minimax: minimize $\|1-p\|_\infty$           |
| Scaling equivalence (minimax vs max-min) | Holds for any scale-closed odd family              | Holds for any scale-closed odd family                 |
| Composition of optima                    | Zolotarev class closed under composition           | Not closed under composition (generic obstruction)    |
| Basis and evaluation                     | Resolvent basis $(I+cB)^{-1}$ and one squaring     | Monomial basis with stable split / identity-centering |
| Offline design                           | Closed-form rational formulas                      | Interval-dependent equioscillation and root-finding   |

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

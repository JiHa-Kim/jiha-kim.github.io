---
layout: post
title: "Fast Tight Spectral-Norm Bounds"
date: 2026-04-28 12:38 +0000
description: "A GPU-friendly algorithm for tight certified singular-value endpoint bounds using scaled Gram matrices, SYRKs, Frobenius reductions, and a small scalar moment problem."
categories:
  - Numerical Linear Algebra
  - Matrix Computations
tags:
  - Spectral Norm
  - Moment Bounds
  - Gram Matrix
  - Numerical Stability
  - Matrix Norms
  - Complex Matrices
math: true
---

Inspired by discussions with [@YouJiacheng](https://x.com/YouJiacheng/status/2039781806652969054) and the Schatten-norm trick discussed by [@Jianlin_S](https://kexue.fm/archives/10922).

> [!info] Overview
> This post gives a hardware-efficient method to bound the maximum singular value of a dense matrix $X$. The main matrix computations are symmetric-rank-k (SYRK) matrix operations, which are well-suited for GPUs:
> $$
> G=X^HX
> \quad [O(mn^2)\text{ SYRK}],
> \qquad
> S=G^2
> \quad [O(n^3)\text{ SYRK}].
> $$
> These two products give the first four trace power sums using only reductions and Frobenius inner products:
> $$
> s_1=\operatorname{tr}(G)=\|X\|_F^2,
> \qquad
> s_2=\operatorname{tr}(G^2)=\|G\|_F^2=\operatorname{tr}(S),
> $$
> $$
> s_3=\operatorname{tr}(G^3)=\langle G,S\rangle_F,
> \qquad
> s_4=\operatorname{tr}(G^4)=\|S\|_F^2.
> $$
> Here
> $$
> s_k=\sum_i\sigma_i^{2k},
> $$
> so these are power sums of the squared singular values. The reductions cost $O(mn)$ for $s_1$ and $O(n^2)$ for the Gram-level quantities.
>
> The scalar part uses these power sums directly as moment constraints on the spectrum. The moment constraints themselves give certified bounds on $\sigma_{\max}(X)$, and the same maximum-endpoint primitive bounds $\sigma_{\min}(X)$ through a shifted inverse.

> [!notation] Conjugate Transpose
> Throughout, $A^H$ denotes the conjugate transpose (Hermitian transpose). For real matrices, $A^H=A^\top$.

---

## 1. Singular-Value Endpoints

> [!notation] SVD, Orientation, and Rank
> Let $X\in\mathbb{F}^{m\times n}$, with $\mathbb{F}\in\{\mathbb{R},\mathbb{C}\}$, and let $r=\operatorname{rank}(X)$. We assume $m\ge n$ without loss of generality; if $m<n$, replace $X$ by $X^H$ since $X$ and $X^H$ have the same singular values.
>
> The thin SVD is
> $$
> X=U_r\Sigma_rV_r^H,
> \qquad
> \Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),
> \qquad
> \sigma_1\ge\cdots\ge\sigma_r>0.
> $$
>
> In this tall orientation, full column rank means $r=n$. If $r<n$, the smallest singular value as a map from $\mathbb{F}^n$ to $\mathbb{F}^m$ is zero.

> [!definition] Endpoint Quantities
> The matrix spectral norm is the Schatten-$\infty$ norm. To avoid confusing it with the vector Euclidean norm, this post writes it as $\|\cdot\|_{S_\infty}$ and reserves $\|\cdot\|_2$ for vectors:
> $$
> \|X\|_{S_\infty}
> =
> \sigma_{\max}(X)
> =
> \max_{\|v\|_2=1}\|Xv\|_2
> =
> \sigma_1.
> $$
>
> The minimum singular value is
> $$
> \sigma_{\min}(X)
> =
> \begin{cases}
> \sigma_n, & r=n,\\
> 0, & r<n,
> \end{cases}
> \qquad
> \sigma_{\min}(X)=\min_{\|v\|_2=1}\|Xv\|_2.
> $$

> [!proposition] Schatten-$\infty$ Interpretation
> For $p\ge1$,
> $$
> \|X\|_{S_p}=\left(\sum_{i=1}^r\sigma_i^p\right)^{1/p},
> \qquad
> \sigma_{\max}(X)=\|X\|_{S_\infty}
> =
> \lim_{p\to\infty}\|X\|_{S_p}.
> $$

> [!proof]- Proof of the Schatten-$p$ Limit
> Let $M=\sigma_1=\sigma_{\max}(X)$. Since $\sigma_i\le M$,
> $$
> M^p\le\sum_{i=1}^r\sigma_i^p\le rM^p.
> $$
> Taking $p$th roots gives
> $$
> M
> \le
> \left(\sum_{i=1}^r\sigma_i^p\right)^{1/p}
> \le
> r^{1/p}M.
> $$
> Since $r^{1/p}\to1$, the squeeze theorem gives the result.

> [!lemma] Unitary Symmetry Selects the Euclidean Norm
> A norm $N$ on $\mathbb{F}^n$ is invariant under every unitary map,
> $$
> N(Ux)=N(x)
> \qquad
> (U^HU=I,\ x\in\mathbb{F}^n),
> $$
> if and only if it is a positive multiple of the Euclidean norm:
> $$
> N=c\|\cdot\|_2
> $$
> for some $c>0$.
>
> This is a statement about norms on $\mathbb{F}^n$. Matrix unitarily invariant norms form a larger family, including Schatten norms and Ky Fan norms.

> [!proof]-
> The case $x=0$ is automatic. For $x\ne0$, set $q_1=x/\|x\|_2$. Extend $q_1$ to an orthonormal basis $q_1,\ldots,q_n$ of $\mathbb{F}^n$, and let
> $$
> Q=\begin{bmatrix}q_1 & q_2 & \cdots & q_n\end{bmatrix}.
> $$
> Then $Q$ is unitary, and
> $$
> Q^Hx
> =
> (q_1^Hx,\ldots,q_n^Hx)^T
> =
> (\|x\|_2,0,\ldots,0)^T
> =
> \|x\|_2e_1,
> $$
> so $U=Q^H$ is unitary and sends $x$ to $\|x\|_2e_1$. By unitary invariance and homogeneity,
> $$
> N(x)=N(Ux)=N(\|x\|_2e_1)=\|x\|_2N(e_1).
> $$
> Thus $N=c\|\cdot\|_2$ with $c=N(e_1)>0$.
>
> Conversely, every positive multiple of $\|\cdot\|_2$ is unitary invariant because
> $$
> \|Ux\|_2^2=x^HU^HUx=x^Hx=\|x\|_2^2.
> $$

> [!lemma] Trace, Power Sums, and Frobenius Reductions
> For compatible matrices whose product is square, trace is cyclic:
> $$
> \operatorname{tr}(A_1A_2\cdots A_\ell)
> =
> \operatorname{tr}(A_2\cdots A_\ell A_1),
> $$
> and therefore trace powers are invariant under similarity:
> $$
> \operatorname{tr}\left((C^{-1}AC)^k\right)=\operatorname{tr}(A^k)
> \qquad
> (C\text{ invertible},\ k\ge1).
> $$
>
> In particular, if $H=H^H$ has eigenvalues $\lambda_1,\ldots,\lambda_n$, then
> $$
> \operatorname{tr}(H^k)=\sum_{i=1}^n\lambda_i^k.
> $$
> Applying this to $G=X^HX$ gives
> $$
> \operatorname{tr}(G^k)=\sum_i\sigma_i^{2k}.
> $$
>
> The Frobenius inner product is the trace inner product:
> $$
> \langle A,B\rangle_F
> =
> \operatorname{tr}(A^HB)
> =
> \sum_{i,j}\overline{A_{ij}}B_{ij}.
> $$
> With $S=G^2$, the concrete reductions used in the algorithm are
> $$
> \operatorname{tr}(G)=\|X\|_F^2,
> \qquad
> \operatorname{tr}(G^2)=\|G\|_F^2=\operatorname{tr}(S),
> $$
> $$
> \operatorname{tr}(G^3)=\langle G,S\rangle_F,
> \qquad
> \operatorname{tr}(G^4)=\|S\|_F^2.
> $$
>
> Finally, unitary factors preserve the Schatten-$\infty$ and Frobenius norms:
> $$
> \|UAV\|_{S_\infty}=\|A\|_{S_\infty},
> \qquad
> \|UAV\|_F=\|A\|_F.
> $$
> For real matrices, read "unitary" as "orthogonal."

> [!proof]- Proof
> For two compatible matrices $A\in\mathbb{F}^{a\times b}$ and $B\in\mathbb{F}^{b\times a}$,
> $$
> \operatorname{tr}(AB)
> =
> \sum_{i=1}^a\sum_{j=1}^b A_{ij}B_{ji}
> =
> \sum_{j=1}^b\sum_{i=1}^a B_{ji}A_{ij}
> =
> \operatorname{tr}(BA).
> $$
> Repeating this step gives cyclic invariance. Since $(C^{-1}AC)^k=C^{-1}A^kC$,
> $$
> \operatorname{tr}\left((C^{-1}AC)^k\right)
> =
> \operatorname{tr}(C^{-1}A^kC)
> =
> \operatorname{tr}(A^k).
> $$
>
> If $H=H^H$, the spectral theorem gives $H=Q\Lambda Q^H$ with $Q$ unitary and
> $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$. Therefore
> $$
> \operatorname{tr}(H^k)
> =
> \operatorname{tr}(Q\Lambda^kQ^H)
> =
> \operatorname{tr}(\Lambda^k)
> =
> \sum_{i=1}^n\lambda_i^k.
> $$
>
> The SVD gives $X=U_r\Sigma_rV_r^H$, so
> $$
> G=X^HX=V_r\Sigma_r^2V_r^H,
> $$
> with $n-r$ additional zero eigenvalues. Thus the eigenvalues of $G$ are $\sigma_1^2,\ldots,\sigma_r^2$ plus zeros, and
> $$
> \operatorname{tr}(G^k)=\sum_i\sigma_i^{2k}.
> $$
>
> The Frobenius identity is the diagonal sum of $A^HB$:
> $$
> \operatorname{tr}(A^HB)
> =
> \sum_j (A^HB)_{jj}
> =
> \sum_j\sum_i \overline{A_{ij}}B_{ij}
> =
> \sum_{i,j}\overline{A_{ij}}B_{ij}.
> $$
> Since $G=G^H$ and $S=G^2=S^H$,
> $$
> \operatorname{tr}(G)=\operatorname{tr}(X^HX)=\|X\|_F^2,
> \qquad
> \operatorname{tr}(G^2)=\operatorname{tr}(G^HG)=\|G\|_F^2,
> $$
> $$
> \operatorname{tr}(G^3)=\operatorname{tr}(G^HS)=\langle G,S\rangle_F,
> \qquad
> \operatorname{tr}(G^4)=\operatorname{tr}(S^HS)=\|S\|_F^2.
> $$
>
> Finally, unitary matrices preserve Euclidean length. Thus
> $$
> \|UAV\|_{S_\infty}
> =
> \max_{\|x\|_2=1}\|UAVx\|_2
> =
> \max_{\|y\|_2=1}\|Ay\|_2
> =
> \|A\|_{S_\infty},
> $$
> and cyclic trace gives
> $$
> \|UAV\|_F^2
> =
> \operatorname{tr}(V^HA^HAV)
> =
> \operatorname{tr}(A^HA)
> =
> \|A\|_F^2.
> $$

> [!fact] Gram Reduction and Inverse Trick
> Let $G=X^HX\succeq0$. Its eigenvalues are $\sigma_1^2,\ldots,\sigma_r^2$ plus $n-r$ zeros, hence
> $$
> \boxed{\sigma_{\max}(X)=\sqrt{\lambda_{\max}(G)}.}
> $$
>
> If $r=n$, then $G\succ0$ and
> $$
> \lambda_{\min}(G)=\frac{1}{\lambda_{\max}(G^{-1})}.
> $$
> Therefore a maximum-eigenvalue upper bound also gives a minimum-singular-value lower bound. In finite precision, use $A=G+\rho I$ with $\rho>0$: if $\lambda_{\max}(A^{-1})\le u$, then
> $$
> \lambda_{\min}(G)\ge\frac1u-\rho.
> $$
>
> Thus the maximum endpoint is the primitive. The post derives a fast upper bound for $\lambda_{\max}(X^HX)$; Section 5 applies the same primitive to a shifted inverse for $\sigma_{\min}(X)$.

> [!problem] Computational Target
> Given dense $X$, return a certified interval
> $$
> l\le\sigma_{\max}(X)\le u
> $$
> using one or two GEMM/SYRK-like matrix operations and small scalar post-processing. When $X$ has full column rank, also optionally return $\sigma_{\min}(X)\ge\ell$ via the shifted-inverse path.

---

## 2. Why the Certificate Must Be an Upper Bound

> [!warning] SVD and Power Iteration Miss Different Requirements
> A dense SVD computes singular vectors and all singular values; here the normalization step only needs one scalar $u\ge\sigma_{\max}(X)$. That is too much factorization work for one endpoint.
>
> Power iteration is cheaper and refineable, but its basic output has the opposite certificate:
> $$
> \|Xv\|_2\le\sigma_{\max}(X)
> \qquad
> (\|v\|_2=1).
> $$
> This is a lower bound. It estimates the endpoint, but it does not certify that scaling by the estimate puts the spectrum inside the required interval.

> [!proof]- Power Iteration Produces a Lower Bound
> Let $G=X^HX\succeq0$ and let $\|v\|_2=1$.
> $$
> \|Xv\|_2^2=v^HGv\le\lambda_{\max}(G)=\sigma_{\max}(X)^2.
> $$
>
> Thus $\|Xv\|_2$ is certified from below. Even if iteration improves $v$, a lower bound $\ell\le\sigma_{\max}(X)$ does not imply safe normalization:
> $$
> \ell\le\sigma_{\max}(X)
> \quad\not\Longrightarrow\quad
> \sigma_{\max}(X/\ell)\le1.
> $$

> [!caution] GPU Mismatch
> Power iteration is a dependency chain of matrix-vector or matrix-block-vector products, normalizations, and reductions. That structure is natural for sparse problems. For dense GPU workloads, one or two large GEMM/SYRK-style kernels usually expose more parallel work and higher arithmetic intensity.

> [!principle] Scaling into the Unit Interval
> A certified upper bound has the correct direction:
> $$
> u\ge\sigma_{\max}(X)
> \quad\Longrightarrow\quad
> 0\le\sigma_i(X/u)=\frac{\sigma_i(X)}{u}\le1.
> $$
>
> If $\hat u<\sigma_{\max}(X)$, then $\sigma_{\max}(X/\hat u)>1$. Any iteration whose proof assumes $\sigma_i\le1$ may start outside its convergence domain.

> [!example] Spectral-Function Iterations
> If $X=U\operatorname{diag}(\sigma_i)V^H$, a spectral map has the form
> $$
> f(X)=U\operatorname{diag}(f(\sigma_i))V^H.
> $$
>
> Polynomial and rational iterations for polar decompositions, inverse square roots, and spectral filters often prove convergence only on an interval:
> $$
> \sigma_i(Y)\in[\ell,1]
> \quad\Longrightarrow\quad
> g_k(\sigma_i(Y))\to f(\sigma_i(Y)).
> $$
>
> The upper certificate enforces the endpoint $1$ after scaling $Y=X/u$. A lower certificate for $\sigma_{\max}(X)$ reports the slack $u/l$; the shifted-inverse path gives an actual floor for $\sigma_{\min}(X)$ when the algorithm needs one.

---

## 3. Scaled Gram Moment Reduction

> [!summary] The Reduction
> Let $G=X^HX$ and $\mu_1=\operatorname{tr}(G)$. If $\mu_1=0$, then $X=0$ and $\sigma_{\max}(X)=0$. Otherwise define
> $$
> P=\frac{G}{\mu_1},
> \qquad
> m_k=\operatorname{tr}(P^k).
> $$
> If $p_1\ge\cdots\ge p_n\ge0$ are the eigenvalues of $P$, then $\sum_i p_i=1$ and
> the trace power-sum identity from Section 1 gives
> $$
> m_k=\sum_i p_i^k.
> $$
> Also,
> $$
> \sigma_{\max}(X)=\sqrt{\mu_1p_1}.
> $$
> Therefore the matrix problem reduces to a scalar endpoint problem:
> $$
> p_1\le\beta
> \quad\Longrightarrow\quad
> \boxed{\sigma_{\max}(X)\le\sqrt{\mu_1\beta}.}
> $$

> [!algorithm] Scaling and Moment Pipeline
> Compute column norms $c_j=\|X_{:j}\|_2$ and choose power-of-two column scales
> $$
> d_j=
> \begin{cases}
> 2^{\operatorname{round}(\log_2 c_j)}, & c_j>0,\\
> 1, & c_j=0.
> \end{cases}
> $$
>
> With $D=\operatorname{diag}(d_j)$, form $Z=XD^{-1}$ and $B=Z^HZ$. Then choose $\alpha=2^{2q}\ge\max_j c_j^2$, define $r_j=d_j/\sqrt{\alpha}$ and $R=\operatorname{diag}(r_j)$, and form
> $$
> T=RBR.
> $$
>
> Since $T=G/\alpha$, set $t_1=\operatorname{tr}(T)=\mu_1/\alpha$ and, for $k\ge2$,
> $$
> r_k=\operatorname{tr}(T^k),
> \qquad
> m_k=\operatorname{tr}(P^k)=\frac{r_k}{t_1^k}.
> $$
>
> A scalar certificate $p_1\le\beta$ becomes
> $$
> \boxed{\sigma_{\max}(X)\le\sqrt{\alpha t_1\beta}.}
> $$
>
> The first scale makes the columns entering the Gram comparable. The second scale makes powers of $T$, not powers of the raw Gram $G$, the numerical object. Because both scales are powers of two, the scaling multiplications are exact in binary floating point, ignoring overflow and underflow.

> [!proof]- Scaled Gram Identity and Entry Bound
> Since $Z=XD^{-1}$ and $D$ is real positive diagonal,
>
> $$
> B=Z^HZ=(XD^{-1})^H(XD^{-1})
> =D^{-1}X^HXD^{-1}
> =D^{-1}GD^{-1}.
> $$
>
> Thus $DBD=G$. Since $R=D/\sqrt{\alpha}$,
>
> $$
> T=RBR=\frac{DBD}{\alpha}=\frac{G}{\alpha}.
> $$
>
> The entry bound follows from the Gram Cauchy-Schwarz inequality:
> $$
> |G_{ij}|^2\le G_{ii}G_{jj}.
> $$
> Because $G_{jj}=c_j^2\le\alpha$, every entry of $T$ satisfies
> $$
> |T_{ij}|=\frac{|G_{ij}|}{\alpha}
> \le
> \frac{\sqrt{G_{ii}G_{jj}}}{\alpha}
> \le1.
> $$
>
> Therefore $T$ is Hermitian positive semidefinite, satisfies $T=G/\alpha$, and has entries bounded in magnitude by one.
>
> After this reduction, the scalar solver only sees $n$ and the normalized moments $m_2,m_3,m_4,\ldots$.

---

## 4. Moment Bounds and Certificates

> [!summary] Bound Hierarchy
> | Path | Extra GEMMs after $B=Z^HZ$ | Output |
> | :--- | :---: | :--- |
> | Two moments | 0 | fast upper bound $\beta_2$ |
> | Four moments | 1 | tighter upper bound $\beta_4$ |
> | Four moments plus support test | 1 | interval $\ell_4\le p_1\le\beta_4$ |

> [!important] Why Not Chase One Very High Moment?
> Since $p_1=\lim_{k\to\infty}m_k^{1/k}$, one could try to approximate $p_1$ by repeated squaring:
> $$
> T,\ T^2,\ T^4,\ T^8,\ldots
> \quad\Longrightarrow\quad
> \operatorname{tr}(T^{2^s}).
> $$
> In exact arithmetic this is a Schatten-norm route to the endpoint. In low precision, the matrix powers accumulate rounding error quickly, and each squaring amplifies any loss of symmetry, scaling error, or cancellation in the trace.
>
> The moment approach uses the information more efficiently: instead of betting on one high-degree power sum, it uses several low-degree constraints $(m_2,m_3,m_4)$ simultaneously and solves the scalar feasibility problem they imply. This keeps the matrix work shallow while still producing a certified endpoint bound.

### 4.1 Two Moments: One Gram

> [!proposition] Two-Moment Bound
> With $m_2=\operatorname{tr}(P^2)=r_2/t_1^2$ and $r_2=\operatorname{tr}(T^2)=\|T\|_F^2$, the sharp two-moment upper bound is
> $$
> \boxed{
> \beta_2
> =
> \frac1n+
> \sqrt{
> \frac{n-1}{n}
> \left(
> m_2-\frac1n
> \right)
> }.
> }
> $$
>
> Therefore $\sigma_{\max}(X)\le\sqrt{\alpha t_1\beta_2}$.

> [!proof]- Derivation of the Two-Moment Bound
> Let $t=p_1$ and $N=n-1$. The residual eigenvalues have total mass $1-t$, so their squared mass is minimized when $p_2=\cdots=p_n=(1-t)/N$. Hence
>
> $$
> m_2-t^2
> =
> \sum_{i=2}^n p_i^2
> \ge
> N\left(\frac{1-t}{N}\right)^2
> =
> \frac{(1-t)^2}{N}.
> $$
>
> Rearranging gives
> $$
> N(m_2-t^2)\ge(1-t)^2
> \quad\Longleftrightarrow\quad
> nt^2-2t+1-(n-1)m_2\le0.
> $$
>
> Therefore $t$ is at most the larger root:
>
> $$
> t\le
> \frac{1+\sqrt{(n-1)(nm_2-1)}}{n}
> =
> \frac1n+
> \sqrt{
> \frac{n-1}{n}
> \left(m_2-\frac1n\right)
> }.
> $$
>
> Equality is attained by one top atom and $n-1$ equal residual atoms, so the bound is sharp given only $m_2$.

### 4.2 Four Moments: One Extra GEMM

> [!proposition] Four-Moment Harvest
> One extra square Gram multiply, $S=T^2$, gives three moments:
> $$
> r_2=\operatorname{tr}(S),
> \qquad
> r_3=\operatorname{tr}(T^3)=\langle T,S\rangle_F,
> \qquad
> r_4=\operatorname{tr}(T^4)=\|S\|_F^2.
> $$
> Normalize by $m_k=r_k/t_1^k$ for $k=2,3,4$. For complex matrices, $\langle T,S\rangle_F=\operatorname{tr}(T^HS)=\sum_{ij}\overline{T_{ij}}S_{ij}$.

> [!proof]- Moment Identities from $S=T^2$
> Since $T$ is Hermitian, $S=T^2$ is Hermitian positive semidefinite.
>
> $$
> \operatorname{tr}(S)=\operatorname{tr}(T^2)=r_2,
> \qquad
> \langle T,S\rangle_F=\operatorname{tr}(T^HS)=\operatorname{tr}(T^3)=r_3,
> $$
> and
> $$
> \|S\|_F^2=\operatorname{tr}(S^HS)=\operatorname{tr}(S^2)=\operatorname{tr}(T^4)=r_4.
> $$

### 4.3 The Four-Moment Upper Bound

> [!proposition] Four-Moment Upper Bound
> For a candidate top value $t$, remove one atom of size $t$:
> $$
> s_0=n-1,
> \qquad
> s_k=m_k-t^k
> \quad(k=1,2,3,4).
> $$
> If $t=p_1$, then the residual eigenvalues lie in $[0,t]$, so the residual moments must satisfy
> $$
> M_0(t)=
> \begin{pmatrix}
> s_0 & s_1 & s_2\\
> s_1 & s_2 & s_3\\
> s_2 & s_3 & s_4
> \end{pmatrix}
> \succeq0,
> $$
> and
> $$
> M_1(t)=
> \begin{pmatrix}
> t s_1-s_2 & t s_2-s_3\\
> t s_2-s_3 & t s_3-s_4
> \end{pmatrix}
> \succeq0.
> $$
>
> Therefore
> $$
> \boxed{
> \beta_4
> =
> \sup\{t\in[0,\beta_2]:M_0(t)\succeq0,\ M_1(t)\succeq0\}.
> }
> $$
>
> Then $p_1\le\beta_4$ and $\sigma_{\max}(X)\le\sqrt{\alpha t_1\beta_4}$.

> [!proof]- Why the Residual Test Gives an Upper Bound
> At the true endpoint $t=p_1$, the residual atomic measure on $p_2,\ldots,p_n$ has moments
> $$
> s_k=\sum_{i=2}^n p_i^k=m_k-p_1^k
> \quad(k=1,2,3,4),
> \qquad
> s_0=n-1.
> $$
> For every quadratic $q(x)=a+bx+cx^2$,
> $$
> \sum_{i=2}^n q(p_i)^2\ge0.
> $$
> Expanding gives $M_0(p_1)\succeq0$:
>
> $$
> \begin{pmatrix}a&b&c\end{pmatrix}
> M_0(p_1)
> \begin{pmatrix}a\\b\\c\end{pmatrix}
> \ge0.
> $$
>
> Since $0\le p_i\le p_1$, every linear $q(x)=a+bx$ also satisfies
> $$
> \sum_{i=2}^n p_i(p_1-p_i)q(p_i)^2\ge0.
> $$
> Expanding gives $M_1(p_1)\succeq0$:
>
> $$
> \begin{pmatrix}a&b\end{pmatrix}
> M_1(p_1)
> \begin{pmatrix}a\\b\end{pmatrix}
> \ge0.
> $$
> Thus the true $p_1$ is feasible in the set defining $\beta_4$, so $p_1\le\beta_4$.

> [!algorithm]- Scalar Solver Details
> The scalar solver should not use iterative bisection. Feasibility changes only when one of the fixed low-degree feasibility polynomials changes sign. Isolate their real roots, add endpoints $0,\beta_2$, sort the resulting constant-size list, and test one midpoint in each interval. The $M_1(t)\succeq0$ tests are
> $$
> U(t)=t-m_2,
> \qquad
> W(t)=tm_3-m_4,
> \qquad
> Q(t)
> =
> (m_3-m_2^2)t^2
> +(m_2m_3-m_4)t
> +(m_2m_4-m_3^2).
> $$
>
> The $M_0(t)\succeq0$ principal-minor tests are
> $$
> B_0(t)=m_2-t^2,
> \qquad
> F_0(t)=m_4-t^4,
> \qquad
> A(t)=(n-1)m_2-1+2t-nt^2,
> $$
> $$
> C(t)=(n-1)m_4-m_2^2+2m_2t^2-nt^4,
> \qquad
> E(t)=(m_2-t^2)(m_4-t^4)-(m_3-t^3)^2,
> $$
> and
> $$
> \begin{aligned}
> D(t)
> ={}&
> (1-nm_2)t^4
> +2(nm_3-m_2)t^3\\
> &+
> (3m_2^2-2m_3-nm_4)t^2\\
> &+
> 2(m_4-m_2m_3)t\\
> &+
> (n-1)(m_2m_4-m_3^2)
> -m_2^3+2m_2m_3-m_4.
> \end{aligned}
> $$
> where $D(t)=\det M_0(t)$. Feasibility is tested by nonnegativity of these polynomials and the actual PSD checks at roots.
>
> All polynomial degrees are at most four. In practice this is a fixed-size scalar kernel: closed-form roots for linear/quadratic pieces, a robust quartic routine or companion-matrix eigenvalue solve for the quartics, fixed-size stack arrays, and no heap allocation.

> [!proof]- Derivation of the Polynomial Tests
> For $M_1(t)$, the diagonal entries are
> $$
> t s_1-s_2=t(1-t)-(m_2-t^2)=t-m_2=U(t),
> \qquad
> t s_3-s_4=t(m_3-t^3)-(m_4-t^4)=tm_3-m_4=W(t).
> $$
> Its determinant is
> $$
> (t s_1-s_2)(t s_3-s_4)-(t s_2-s_3)^2
> =
> (t-m_2)(tm_3-m_4)-(tm_2-m_3)^2,
> $$
> which expands to $Q(t)$.
>
> For $M_0(t)$, the nontrivial $2\times2$ principal minors are
> $$
> s_0s_2-s_1^2
> =
> (n-1)(m_2-t^2)-(1-t)^2
> =
> A(t),
> $$
>
> $$
> s_0s_4-s_2^2
> =
> (n-1)(m_4-t^4)-(m_2-t^2)^2
> =
> C(t),
> $$
>
> and
>
> $$
> s_2s_4-s_3^2
> =
> (m_2-t^2)(m_4-t^4)-(m_3-t^3)^2
> =
> E(t).
> $$
>
> Finally,
> $$
> \det M_0
> =
> s_0s_2s_4+2s_1s_2s_3-s_0s_3^2-s_2^3-s_1^2s_4.
> $$
>
> Substituting $s_0=n-1$ and $s_k=m_k-t^k$ gives the stated quartic $D(t)$.

### 4.4 A Posteriori Slack Certificate

> [!proposition] Two-Sided Certificate
> If all normalized eigenvalues lie in $[0,t]$, then $\sum_i p_i(t-p_i)q(p_i)^2\ge0$ for every linear $q$. With moments through degree four, this is the $2\times2$ PSD test
> $$
> K(t)=
> \begin{pmatrix}
> t-m_2 & tm_2-m_3\\
> tm_2-m_3 & tm_3-m_4
> \end{pmatrix}.
> $$
> Equivalently, test $t-m_2\ge0$, $tm_3-m_4\ge0$, and
> $$
> q_{\mathrm{low}}(t)
> =
> (t-m_2)(tm_3-m_4)-(tm_2-m_3)^2.
> $$
>
> The lower certificate is the single support endpoint
> $$
> \boxed{
> \ell_4
> =
> \inf\{t\in[0,1]:K(t)\succeq0\}.
> }
> $$
> Then
> $$
> \ell_4\le p_1\le\beta_4,
> \qquad
> 1\le
> \frac{\sqrt{\alpha t_1\beta_4}}{\sigma_{\max}(X)}
> \le
> \sqrt{\frac{\beta_4}{\ell_4}}.
> $$
> Report the a posteriori relative slack $\sqrt{\beta_4/\ell_4}-1$. No separate lower-bound menu is needed: the lower-right entry of $K(t)$ already enforces the strongest elementary ratio check $t\ge m_4/m_3$.
>
> Practically, start at $\ell_0=\max\{m_2,m_4/m_3\}$. If $K(\ell_0)\succeq0$, then $\ell_4=\ell_0$. Otherwise $\ell_4$ is the next real root of the quadratic $q_{\mathrm{low}}(t)$ in $[\ell_0,1]$ that passes the same $2\times2$ PSD test.

> [!proof]- Why the Support Test Gives a Lower Bound
> Let $t=p_1$. Since $0\le p_i\le p_1$, for every linear $q(x)=a+bx$,
> $$
> \sum_i p_i(p_1-p_i)q(p_i)^2\ge0.
> $$
> Expanding gives
> $$
> \begin{pmatrix}a&b\end{pmatrix}
> K(p_1)
> \begin{pmatrix}a\\b\end{pmatrix}
> \ge0.
> $$
> The entries are exactly
> $$
> \sum_i p_i(p_1-p_i)=p_1-m_2,
> \qquad
> \sum_i p_i^2(p_1-p_i)=p_1m_2-m_3,
> \qquad
> \sum_i p_i^3(p_1-p_i)=p_1m_3-m_4.
> $$
> Hence $K(p_1)\succeq0$. Therefore the smallest $t$ satisfying $K(t)\succeq0$ is at most $p_1$.

### 4.5 A Crude Dimension-Only Scale Check

> [!fact] Crude Universal Bound
> The fourth moment alone gives
> $$
> p_1\ge m_4^{1/3},
> \qquad
> \beta_4\le m_4^{1/4},
> \qquad
> \frac{\beta_4}{p_1}\le m_4^{-1/12}.
> $$
> Since $m_4\ge1/n^3$,
> $$
> \boxed{
> \frac{\beta_4}{p_1}\le n^{1/4},
> \qquad
> \frac{\sqrt{\mu_1\beta_4}}{\sigma_{\max}(X)}\le n^{1/8}.
> }
> $$

> [!example] Worst-Worst Scale for $n=16384$
> If $n=16384=2^{14}$, then
> $$
> n^{1/4}=2^{3.5}\approx 11.31.
> $$
> So the dimension-only eigenvalue overestimate is at most about $11.31\times$.
>
> For the Schatten-$\infty$ norm,
> $$
> n^{1/8}=2^{1.75}\approx 3.36.
> $$
> So even this deliberately pessimistic norm overestimate is at most about $3.36\times$, corresponding to relative slack at most about $2.36$. The actual a posteriori certificate is usually much tighter.

> [!proof]- Derivation of the Crude Bound
> Since $p_1$ is the largest normalized eigenvalue and $\sum_i p_i=1$,
> $$
> m_4=\sum_i p_i^4\le p_1^3\sum_i p_i=p_1^3,
> $$
> so $p_1\ge m_4^{1/3}$. Feasibility of an upper endpoint also requires $\beta_4^4\le m_4$, hence $\beta_4\le m_4^{1/4}$ and $\beta_4/p_1\le m_4^{-1/12}$.
>
> Jensen's inequality gives the dimension-only floor:
> $$
> \frac1n\sum_i p_i^4
> \ge
> \left(\frac1n\sum_i p_i\right)^4
> =
> \frac{1}{n^4}.
> $$
> Multiplying by $n$ gives $m_4\ge1/n^3$.

### 4.6 Why Decaying Spectra Give Tight Bounds

> [!proposition] Decay-Ratio Tightness
> Let the normalized eigenvalues be sorted as $p_1\ge p_2\ge\cdots\ge0$, with $\sum_i p_i=1$. Set $\theta=p_1$ and write the relative tail ratios
> $$
> a_i=\frac{p_i}{\theta},
> \qquad
> a_1=1,
> \qquad
> 0\le a_i\le1.
> $$
> Define the high-power tail sums
> $$
> R_k=\sum_{i\ge2}a_i^k.
> $$
> Then $m_k=\theta^k(1+R_k)$, and the certified interval obeys
> $$
> 0\le\beta_4-\theta
> \le
> \theta\left((1+R_4)^{1/4}-1\right)
> \le
> \frac{\theta R_4}{4},
> $$
> while
> $$
> 0\le\theta-\ell_4
> \le
> \theta-\frac{m_4}{m_3}
> =
> \theta\frac{R_3-R_4}{1+R_3}
> =
> \theta\frac{\sum_{i\ge2}a_i^3(1-a_i)}{1+R_3}.
> $$
> Hence
> $$
> \boxed{
> \beta_4-\ell_4
> \le
> \frac{\theta R_4}{4}
> +
> \theta\frac{R_3-R_4}{1+R_3}.
> }
> $$
> The important quantities are not the dimension itself, but the high-power tail sums $R_3$ and $R_4$. Fast spectral decay makes these small. The upper estimate here is deliberately conservative: it only uses $m_4$, while the implemented scalar solver also uses $m_2$ and $m_3$.

> [!proof]- Decay-Ratio Derivation
> Since $p_i=\theta a_i$ and $a_1=1$,
> $$
> m_k=\sum_i p_i^k
> =
> \theta^k\sum_i a_i^k
> =
> \theta^k(1+R_k).
> $$
>
> For the upper endpoint, any feasible largest atom $t$ must satisfy $t^4\le m_4$. Therefore
> $$
> \beta_4\le m_4^{1/4}
> =
> \theta(1+R_4)^{1/4}.
> $$
> Concavity gives $(1+x)^{1/4}\le1+x/4$ for $x\ge0$, hence
> $$
> \beta_4-\theta
> \le
> \theta\left((1+R_4)^{1/4}-1\right)
> \le
> \frac{\theta R_4}{4}.
> $$
>
> For the lower certificate, the support test includes $\ell_4\ge m_4/m_3$. Using the ratio form of the moments,
> $$
> \frac{m_4}{m_3}
> =
> \theta\frac{1+R_4}{1+R_3}.
> $$
> Thus
> $$
> \theta-\frac{m_4}{m_3}
> =
> \theta\left(1-\frac{1+R_4}{1+R_3}\right)
> =
> \theta\frac{R_3-R_4}{1+R_3}.
> $$
> Since $R_3-R_4=\sum_{i\ge2}a_i^3(1-a_i)$, the lower gap is controlled by a weighted tail defect. It is small when the high-power tail is small, and it also vanishes for exact top multiplicities. Adding the upper and lower gaps gives the stated interval bound.

> [!corollary] Simple Ratio Bound
> If every tail ratio satisfies $a_i\le q<1$ for $i\ge2$, and $\delta=1-\theta$ is the total tail mass, then
> $$
> \beta_4-\theta\le\frac{\delta q^3}{4},
> \qquad
> \theta-\ell_4\le\delta q^2,
> \qquad
> \beta_4-\ell_4\le\delta\left(q^2+\frac{q^3}{4}\right).
> $$

> [!proof]- Ratio-Bound Derivation
> Let $R_1=\sum_{i\ge2}a_i$. Since $\sum_i p_i=1$, the tail mass is $\delta=\theta R_1$. If $a_i\le q$, then
> $$
> R_3=\sum_{i\ge2}a_i^3\le q^2R_1,
> \qquad
> R_4=\sum_{i\ge2}a_i^4\le q^3R_1.
> $$
> Substitute these into the decay-ratio bound:
> $$
> \beta_4-\theta\le\frac{\theta R_4}{4}\le\frac{\delta q^3}{4},
> \qquad
> \theta-\ell_4\le\theta R_3\le\delta q^2.
> $$

> [!example] Geometric Decay
> For an ideal geometric profile $a_i=q^{i-1}$, $i\ge1$, on an infinite tail,
> $$
> \theta=1-q,
> \qquad
> R_k=\frac{q^k}{1-q^k}.
> $$
> The lower ratio certificate has the exact gap
> $$
> \theta-\frac{m_4}{m_3}
> =
> \frac{(1-q)^2q^3}{1-q^4},
> $$
> while the upper gap satisfies
> $$
> \beta_4-\theta
> \le
> \frac{(1-q)q^4}{4(1-q^4)}.
> $$
> Thus the interval closes like $O(q^3)$ as the decay ratio $q$ becomes small. For a finite geometric tail, replace $R_k$ by $q^k(1-q^{k(n-1)})/(1-q^k)$.

### 4.7 Scalar Solver Widget

{% include spectral_moment_bounds_widget.html %}

---

## 5. Minimum Singular Value by Shifted Inverse Moments

> [!summary] Reuse the Maximum-Endpoint Primitive
> The inverse trick uses the same moment solver on a different positive definite matrix. The input is the scaled Gram $T=G/\alpha$ and the shifted inverse
> $$
> A=T+\rho I\succ0,
> \qquad
> H=A^{-1}.
> $$

> [!algorithm] Shifted Inverse Moment Reduction
> Let $\tau_i=\lambda_i(G)/\alpha$ and $h_i=1/(\tau_i+\rho)$. Compute inverse moments
> $$
> \eta_k=\operatorname{tr}(H^k)=\sum_i h_i^k,
> \qquad
> q_i=\frac{h_i}{\eta_1}.
> $$
> If the moment solver returns $q_1\le\beta_{\mathrm{inv}}$, then
> $$
> \boxed{
> \lambda_{\min}(G)
> \ge
> \alpha\left(
> \frac{1}{\eta_1\beta_{\mathrm{inv}}}
> -\rho
> \right)
> }
> $$
> and
> $$
> \boxed{
> \sigma_{\min}(X)
> \ge
> \sqrt{
> \max\left\{
> 0,
> \alpha\left(
> \frac{1}{\eta_1\beta_{\mathrm{inv}}}
> -\rho
> \right)
> \right\}
> }.
> }
> $$

> [!proof]- Shifted Inverse Moment Bound
> The eigenvalues of $H=A^{-1}$ are
> $$
> h_i=\frac{1}{\tau_i+\rho}.
> $$
> The largest inverse eigenvalue corresponds to the smallest original eigenvalue:
> $$
> h_{\max}
> =
> \frac{1}{\tau_{\min}+\rho}.
> $$
> Since $q_i=h_i/\eta_1$, the upper certificate $q_1\le\beta_{\mathrm{inv}}$ gives
> $$
> h_{\max}\le\eta_1\beta_{\mathrm{inv}}.
> $$
> Taking reciprocals,
> $$
> \tau_{\min}+\rho
> =
> \frac1{h_{\max}}
> \ge
> \frac{1}{\eta_1\beta_{\mathrm{inv}}}.
> $$
> Finally, $\lambda_{\min}(G)=\alpha\tau_{\min}$.

> [!caution] Precision
> The inverse path is conditioning-sensitive. Use fp32/fp64 factorizations and scalar reductions; avoid fp16 for the shifted inverse moments.

---

## 6. Practical Implementation

> [!summary] Implementation Split
> The implementation has one matrix pipeline and one scalar pipeline:
> $$
> X\to Z=XD^{-1}\to B=Z^HZ\to T=RBR\to S=T^2\to(m_2,m_3,m_4),
> $$
> followed by the scalar endpoint solves for $\ell_4,\beta_4$. Matrix work is GEMM/SYRK-like; scalar work is tiny and should run in high precision.

### 6.1 Helpers

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Helpers</span> Scaling, Symmetry, and Two-Moment Formula</div>
```pseudo
def Sym($A$):
    return $\frac12(A+A^H)$

def Pow2Round($c$):
    if $c=0$: return $1$
    return $2^{\operatorname{round}(\log_2 c)}$

def Pow2CeilEven($a$):
    # returns $\alpha=2^{2q}\ge a$
    if $a=0$: return $(1,1)$
    $q \leftarrow \left\lceil \frac12\log_2 a\right\rceil$
    return $(2^{2q},2^q)$  # $(\alpha,\sqrt{\alpha})$

def Beta2($m_2,n$):
    $v \leftarrow \max(0, m_2-\frac1n)$
    return $\frac1n+\sqrt{\frac{n-1}{n}v}$
```
</div>

### 6.2 Scaled Gram Setup

> [!summary] Shared Setup
> The same scaled Gram setup feeds the two-moment path, the four-moment path, and the shifted-inverse path.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Scaled Gram Setup</div>
```pseudo
def ScaledGramSetup($X\in\mathbb{F}^{m\times n}$):
    # Column norm squares; accumulate in fp32 or fp64.
    for $j=1,\ldots,n$:
        $c_j^2 \leftarrow \sum_i |X_{ij}|^2$

    $\mu_1 \leftarrow \sum_j c_j^2$
    if $\mu_1=0$:
        return ZeroMatrixCase

    # Exact power-of-two column scales.
    for $j=1,\ldots,n$:
        $d_j \leftarrow$ @Pow2Round($\sqrt{c_j^2}$)

    $D \leftarrow \operatorname{diag}(d_1,\ldots,d_n)$
    $Z \leftarrow XD^{-1}$  # preferably fused into the Gram input path

    $B \leftarrow Z^HZ$

    # Global power-of-two scaling.
    $(\alpha,\sqrt{\alpha}) \leftarrow$ @Pow2CeilEven($\max_j c_j^2$)
    for $j=1,\ldots,n$:
        $r_j \leftarrow d_j/\sqrt{\alpha}$

    $R \leftarrow \operatorname{diag}(r_1,\ldots,r_n)$
    $T \leftarrow RBR$  # elementwise: $T_{ij}=r_iB_{ij}r_j$
    $T \leftarrow$ @Sym($T$)

    $t_1 \leftarrow \operatorname{tr}(T)$
    return $(T,t_1,\alpha)$
```
</div>

> [!check] Invariant
> In exact arithmetic, Algorithm 1 returns
> $$
> T=\frac{X^HX}{\alpha},
> \qquad
> t_1=\frac{\operatorname{tr}(X^HX)}{\alpha}.
> $$
> The raw Gram $G=X^HX$ never has to be materialized.

### 6.3 End-to-End Spectral Norm Bound

> [!tip] Cost Switch
> Use `order = 2` for the one-Gram path and `order = 4` for the one-extra-GEMM path.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 2</span> Spectral Norm Upper Bound</div>
```pseudo
def SpectralNormUpperBound($X,\operatorname{order}=4$):
    setup $\leftarrow$ @ScaledGramSetup($X$)
    if setup is ZeroMatrixCase:
        return upper = $0$, lower_cert = $0$, rel_slack = $0$

    $(T,t_1,\alpha) \leftarrow$ setup
    $n \leftarrow$ number of columns of $X$

    # Two-moment path.
    if order = 2:
        $r_2 \leftarrow \|T\|_F^2$
        $m_2 \leftarrow r_2/t_1^2$
        $\beta_2 \leftarrow$ @Beta2($m_2,n$)
        $u \leftarrow \sqrt{\alpha t_1\beta_2}$
        return upper = $u$

    # Four-moment path.
    $S \leftarrow$ @Sym($T^2$)

    # Fused reduction over $T$ and $S$.
    $r_2 \leftarrow \operatorname{tr}(S)$
    $r_3 \leftarrow \operatorname{Re}\langle T,S\rangle_F$
    $r_4 \leftarrow \|S\|_F^2$

    $m_2 \leftarrow r_2/t_1^2$
    $m_3 \leftarrow r_3/t_1^3$
    $m_4 \leftarrow r_4/t_1^4$

    $\beta_2 \leftarrow$ @Beta2($m_2,n$)
    $\beta_4 \leftarrow$ @FourMomentUpper($m_2,m_3,m_4,n,\beta_2$)
    $\ell_4 \leftarrow$ @FourMomentLower($m_2,m_3,m_4$)

    $u \leftarrow \sqrt{\alpha t_1\beta_4}$
    $\ell \leftarrow \sqrt{\alpha t_1\ell_4}$
    $\mathrm{rel\_slack} \leftarrow \sqrt{\beta_4/\ell_4}-1$

    return upper = $u$, lower_cert = $\ell$, rel_slack = $\mathrm{rel\_slack}$
```
</div>

> [!tip] What to Return
> For production use, return both the upper bound and the certificate:
> $$
> \sqrt{\alpha t_1\ell_4}
> \le
> \sigma_{\max}(X)
> \le
> \sqrt{\alpha t_1\beta_4}.
> $$
> The scalar $\sqrt{\beta_4/\ell_4}-1$ is the a posteriori relative slack.

### 6.4 Scalar Four-Moment Solvers

> [!tip] Scalar Precision
> The scalar solver is not a performance bottleneck, but it should still be written as a constant-size kernel: fp64 coefficients, fixed-size root buffers, closed-form linear/quadratic roots, robust quartic roots, and direct minor tests instead of matrix eigensolvers.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 3</span> Four-Moment Upper Endpoint</div>
```pseudo
def FourMomentUpper($m_2,m_3,m_4,n,\beta_2$):
    # Constant-size root enumeration
    polys $\leftarrow \{A,B_0,C,D,E,F_0,U,W,Q\}$
    roots $\leftarrow \{0,\beta_2\}$

    for $p$ in polys:
        roots $\leftarrow$ roots $\cup$ @RealRootsInInterval($p,0,\beta_2$)

    roots $\leftarrow$ @SortUnique($roots$)
    $\beta \leftarrow 0$

    for each adjacent pair $(a,b)$ in roots:
        $z \leftarrow (a+b)/2$
        if @ResidualPSD($z,m_2,m_3,m_4,n$):
            $\beta \leftarrow \max(\beta,b)$

    # Also test roots themselves for numerical safety.
    for $z$ in roots:
        if @ResidualPSD($z,m_2,m_3,m_4,n$):
            $\beta \leftarrow \max(\beta,z)$

    return $\beta$

def ResidualPSD($t,m_2,m_3,m_4,n$):
    # Equivalent to $M_0(t)\succeq0$ and $M_1(t)\succeq0$.
    return @AllNonnegative($B_0(t),F_0(t),A(t),C(t),E(t),D(t),U(t),W(t),Q(t)$)
```
</div>

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 4</span> Four-Moment Lower Certificate</div>
```pseudo
def FourMomentLower($m_2,m_3,m_4$):
    # Direct 2x2 PSD support endpoint; no interval scan needed.
    $q_{\mathrm{low}}(t) \leftarrow (t-m_2)(tm_3-m_4)-(tm_2-m_3)^2$
    $\ell_0 \leftarrow \max(m_2,\ m_4/m_3)$
    if @SupportPSD($\ell_0,m_2,m_3,m_4$):
        return $\ell_0$

    roots $\leftarrow$ @RealRootsInInterval($q_{\mathrm{low}},\ell_0,1$)
    candidates $\leftarrow$ @SortUnique($roots \cup \{1\}$)
    return first $z$ in candidates with @SupportPSD($z,m_2,m_3,m_4$)

def SupportPSD($t,m_2,m_3,m_4$):
    $a \leftarrow t-m_2$
    $b \leftarrow tm_2-m_3$
    $c \leftarrow tm_3-m_4$
    return $a\ge0$ and $c\ge0$ and $ac-b^2\ge0$
```
</div>

### 6.5 Minimum Singular Value Routine

> [!caution] Inverse Path Precision
> The minimum singular value path reuses the scaled Gram but switches to inverse moments. Use higher precision here.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 5</span> Shifted-Inverse Minimum Singular Value Lower Bound</div>
```pseudo
def MinSingularLowerBound($X,\rho,\operatorname{order}=4$):
    setup $\leftarrow$ @ScaledGramSetup($X$)
    if setup is ZeroMatrixCase:
        return lower = $0$

    $(T,t_1,\alpha) \leftarrow$ setup
    $n \leftarrow$ number of columns of $X$

    # Use fp32/fp64 factorization; avoid fp16 here.
    $A \leftarrow$ @Sym($T+\rho I$)
    $L \leftarrow$ @Cholesky($A$)

    # Conceptually form $H=A^{-1}$; implementations may use triangular solves.
    $H \leftarrow A^{-1}$

    $\eta_1 \leftarrow \operatorname{tr}(H)$
    $\eta_2 \leftarrow \|H\|_F^2$

    if order = 2:
        $m_2^{\mathrm{inv}} \leftarrow \eta_2/\eta_1^2$
        $\beta_{\mathrm{inv}} \leftarrow$ @Beta2($m_2^{\mathrm{inv}},n$)
    else:
        $H_2 \leftarrow$ @Sym($H^2$)
        $\eta_2 \leftarrow \operatorname{tr}(H_2)$
        $\eta_3 \leftarrow \operatorname{Re}\langle H,H_2\rangle_F$
        $\eta_4 \leftarrow \|H_2\|_F^2$

        $m_2^{\mathrm{inv}} \leftarrow \eta_2/\eta_1^2$
        $m_3^{\mathrm{inv}} \leftarrow \eta_3/\eta_1^3$
        $m_4^{\mathrm{inv}} \leftarrow \eta_4/\eta_1^4$

        $\beta_2^{\mathrm{inv}} \leftarrow$ @Beta2($m_2^{\mathrm{inv}},n$)
        $\beta_{\mathrm{inv}} \leftarrow$ @FourMomentUpper($m_2^{\mathrm{inv}},m_3^{\mathrm{inv}},m_4^{\mathrm{inv}},n,\beta_2^{\mathrm{inv}}$)

    $\tau_{\min}^{\mathrm{lower}} \leftarrow \frac{1}{\eta_1\beta_{\mathrm{inv}}}-\rho$
    $\lambda_{\min}^{\mathrm{lower}} \leftarrow \alpha\max(0,\tau_{\min}^{\mathrm{lower}})$

    return lower = $\sqrt{\lambda_{\min}^{\mathrm{lower}}}$
```
</div>

> [!tip] Precision Choices
> For $B=Z^HZ$, a good fast path is fp16 or bf16 inputs with fp32 accumulation, or TF32/fp32 when $X$ is already fp32 and range matters.
>
> For $S=T^2$, store $T$ and $S$ in fp32. Casting $T$ to fp16 or bf16 for tensor-core multiplication can be reasonable because $|T_{ij}|\le1$, but the output should remain fp32. In extreme rank-one cases, entries of $S$ can be $O(n)$.
>
> For scalar reductions and the moment solver, fp64 is preferred. If fp64 reductions are too expensive, use fp32 pairwise or compensated reductions, then run the scalar solver in fp64.

> [!check] Debug Check
> If $S=T^2$ is already formed, use
> $$
> r_2=\operatorname{tr}(S)
> $$
> rather than doing a second full Frobenius reduction over $T$.
>
> Occasionally compare this with
> $$
> \|T\|_F^2.
> $$
> A large discrepancy points to matmul or reduction error.

> [!summary] Final Path
> $$
> X
> \rightarrow
> Z=XD^{-1}
> \rightarrow
> B=Z^HZ
> \rightarrow
> T=RBR=G/\alpha
> \rightarrow
> S=T^2
> \rightarrow
> m_2,m_3,m_4
> \rightarrow
> \beta_4
> \rightarrow
> \sqrt{\alpha t_1\beta_4}.
> $$
>
> The important choices are: scale before the Gram, scale before powers, compute powers of $T$ instead of $G$, solve the scalar moment problem in high precision, and report the two-sided certificate when available.

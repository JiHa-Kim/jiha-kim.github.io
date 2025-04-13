---
math: true
---

Plan for Convex Analysis

1. Convex Sets
2. Convex Functions
3. Subgradients and Subdifferential Calculus
4. Convex Optimization
5. Convex Duality

Okay, this is a fantastic topic for a blog series! Convex analysis and optimization are foundational for understanding *why* many machine learning algorithms work and how to design new ones. Here's an extensive list of concepts, theorems, and connections, structured logically for a potential series.

I'll structure this list assuming multiple blog posts. You can pick and choose, reorder, or combine based on the desired depth for each post.

---

**Blog Series Outline: Convex Analysis & Optimization for ML**

**Goal:** Build a strong foundation in convex sets, functions, duality, and optimization algorithms, highlighting equivalences and preparing for ML applications.

**Part 1: The Basics - Sets and Geometry**

1.  **Introduction:**
    *   Why study convexity? (Guarantees in optimization, uniqueness of solutions, efficient algorithms, prevalence in ML).
    *   Geometric intuition: "Bowl shape", no "dents" or "holes".
    *   Brief mention of convex vs. non-convex optimization challenges.
2.  **Affine Sets:**
    *   Definition: $$ S = \{ x \in \mathbb{R}^n \mid x = \theta x_1 + (1-\theta) x_2 \text{ for some } x_1, x_2 \in S, \forall \theta \in \mathbb{R} \} $$
    *   Connection to linear subspaces + translation.
    *   Affine combinations: $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \sum \theta_i = 1 $$.
    *   Affine hull (`aff S`).
    *   Examples: Lines, planes, hyperplanes, solutions to $$ Ax = b $$.
3.  **Convex Sets:**
    *   Definition: $$ S = \{ x \in \mathbb{R}^n \mid x = \theta x_1 + (1-\theta) x_2 \text{ for } x_1, x_2 \in S, \forall \theta \in [0, 1] \} $$
    *   Geometric interpretation: Line segment between any two points lies within the set.
    *   Convex combinations: $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \sum \theta_i = 1 $$ and $$ \theta_i \ge 0 $$.
    *   Convex hull (`conv S`): Smallest convex set containing S. Carathéodory's Theorem (any point in `conv S` is a convex combination of at most n+1 points from S).
    *   Examples:
        *   Empty set, single point, $$ \mathbb{R}^n $$.
        *   Hyperplanes: $$ \{ x \mid a^T x = b \} $$
        *   Halfspaces: $$ \{ x \mid a^T x \le b \} $$, $$ \{ x \mid a^T x \ge b \} $$
        *   Norm balls: $$ \{ x \mid \Vert x - x_c \Vert \le r \} $$ (for any norm!).
        *   Ellipsoids: $$ \{ x \mid (x - x_c)^T P^{-1} (x - x_c) \le 1 \} $$ (P positive definite).
        *   Polyhedra: Intersection of finite number of halfspaces/hyperplanes $$ \{ x \mid Ax \preceq b, Cx = d \} $$. (Includes intervals, rays, simplexes).
4.  **Cones:**
    *   Definition: $$ S = \{ x \in \mathbb{R}^n \mid x \in S \implies \theta x \in S \text{ for all } \theta \ge 0 \} $$.
    *   Convex Cones: Cone that is also convex. Equivalently, $$ x_1, x_2 \in S, \theta_1, \theta_2 \ge 0 \implies \theta_1 x_1 + \theta_2 x_2 \in S $$.
    *   Conic combinations: $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \theta_i \ge 0 $$.
    *   Conic hull (`cone S`).
    *   Examples:
        *   Non-negative orthant: $$ \mathbb{R}^n_+ $$.
        *   Norm cones (Second-order cone / Lorentz cone / ice-cream cone): $$ \{ (x, t) \in \mathbb{R}^{n+1} \mid \Vert x \Vert_2 \le t \} $$.
        *   Positive Semidefinite cone: $$ \mathbb{S}^n_+ $$ (set of symmetric PSD matrices).
5.  **Topological Properties (Briefly):**
    *   Interior (`int S`), Relative Interior (`relint S`), Boundary (`bd S`), Closure (`cl S`).
    *   Importance of `relint` for constraint qualifications later.
6.  **Operations Preserving Convexity (Sets):**
    *   Intersection: $$ \cap_i S_i $$ is convex if all $$ S_i $$ are convex.
    *   Affine transformations: $$ f(x) = Ax + b $$. If S is convex, $$ f(S) $$ and $$ f^{-1}(S) $$ are convex.
    *   Perspective transformation: $$ P(x, t) = x/t $$ (domain $$ t > 0 $$).
    *   Linear-fractional functions: $$ f(x) = (Ax+b)/(c^Tx+d) $$.
    *   Minkowski Sum: $$ S_1 + S_2 = \{ x_1 + x_2 \mid x_1 \in S_1, x_2 \in S_2 \} $$.

    ***Cheat Table Idea 1: Operations Preserving Set Convexity***

**Part 2: Convex Functions - Properties and Characterizations**

1.  **Definition of Convex Functions:**
    *   Domain `dom f` must be a convex set.
    *   Jensen's Inequality: $$ f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y) $$ for $$ x, y \in \text{dom } f, \theta \in [0, 1] $$.
    *   Geometric Interpretation: Chord lies above the graph.
    *   Epigraph Characterization: $$ \text{epi } f = \{ (x, t) \mid x \in \text{dom } f, f(x) \le t \} $$. *f is convex iff epi f is a convex set.*
2.  **Strict Convexity:**
    *   $$ f(\theta x + (1-\theta) y) < \theta f(x) + (1-\theta) f(y) $$ for $$ x, y \in \text{dom } f, x \ne y, \theta \in (0, 1) $$.
    *   Importance: Guarantees uniqueness of minimum *point*.
3.  **Strong Convexity:**
    *   Definition: There exists $$ m > 0 $$ such that $$ g(x) = f(x) - \frac{m}{2} \Vert x \Vert_2^2 $$ is convex.
    *   Equivalent condition (if differentiable): $$ f(y) \ge f(x) + \nabla f(x)^T (y-x) + \frac{m}{2} \Vert y - x \Vert_2^2 $$.
    *   Equivalent condition (if twice differentiable): $$ \nabla^2 f(x) \succeq m I $$ (Hessian eigenvalues $$ \ge m $$).
    *   Importance: Guarantees existence and uniqueness of minimum, key for linear convergence rates in optimization.
    *   Relationship: Strong Convexity => Strict Convexity => Convexity.
4.  **Concave Functions:**
    *   Definition: $$ -f $$ is convex. Jensen's inequality reversed. Hypograph is convex.
5.  **First-Order Condition (for differentiable f):**
    *   *f is convex iff* $$ f(y) \ge f(x) + \nabla f(x)^T (y-x) $$ for all $$ x, y \in \text{dom } f $$.
    *   Geometric Interpretation: Tangent line is a global underestimator.
6.  **Second-Order Condition (for twice differentiable f):**
    *   *f is convex iff* $$ \nabla^2 f(x) \succeq 0 $$ (Hessian is positive semidefinite) for all $$ x \in \text{dom } f $$.
    *   Strict Convexity: $$ \nabla^2 f(x) \succ 0 $$ (usually sufficient, not necessary unless dom f is open).
    *   Strong Convexity: $$ \nabla^2 f(x) \succeq m I $$ for some $$ m > 0 $$.

    ***Cheat Table Idea 2: Equivalent Conditions for Convex Functions (Jensen's, Epigraph, 1st Order, 2nd Order)***

7.  **Examples of Convex Functions:**
    *   Affine functions: $$ f(x) = a^T x + b $$ (both convex and concave).
    *   Quadratic functions: $$ f(x) = \frac{1}{2} x^T P x + q^T x + r $$ (convex iff $$ P \succeq 0 $$).
    *   Norms: $$ f(x) = \Vert x \Vert $$ (any valid norm).
    *   Max function: $$ f(x) = \max \{ x_1, \dots, x_n \} $$.
    *   Log-sum-exp: $$ f(x) = \log(\sum_{i=1}^n e^{x_i}) $$ (smooth approximation of max).
    *   Negative Logarithm: $$ f(x) = -\log x $$ (on $$ \mathbb{R}_{++} $$).
    *   Negative Entropy: $$ f(x) = \sum x_i \log x_i $$ (on $$ \mathbb{R}^n_{++} $$).
    *   Matrix Fractional: $$ f(X, y) = y^T X^{-1} y $$ (on $$ \mathbb{S}^n_{++} \times \mathbb{R}^n $$).
    *   Log-determinant: $$ f(X) = \log \det X $$ (concave on $$ \mathbb{S}^n_{++} $$).
8.  **Sublevel Sets:**
    *   $$ \alpha $$-sublevel set: $$ C_\alpha = \{ x \in \text{dom } f \mid f(x) \le \alpha \} $$.
    *   *If f is convex, then all its sublevel sets are convex.* (Converse not true).
9.  **Quasiconvex Functions:**
    *   Definition: Domain is convex and all sublevel sets $$ C_\alpha $$ are convex.
    *   Equivalent: $$ f(\theta x + (1-\theta) y) \le \max\{f(x), f(y)\} $$.
    *   Broader class than convex functions. Still useful in some optimization contexts (e.g., bisection method).

**Part 3: Operations Preserving Function Convexity & More Geometry**

1.  **Operations Preserving Convexity (Functions):**
    *   Non-negative weighted sum: $$ \sum w_i f_i(x) $$ where $$ w_i \ge 0 $$ and $$ f_i $$ are convex.
    *   Composition with affine map: $$ g(x) = f(Ax+b) $$ is convex if f is convex.
    *   Pointwise maximum/supremum: $$ f(x) = \max_{i} \{ f_i(x) \} $$ or $$ f(x) = \sup_{y \in C} g(x, y) $$ is convex if $$ f_i(x) $$ (or $$ g(x, y) $$ for fixed y) are convex.
    *   Composition (general): $$ h(x) = f(g(x)) $$. If f is convex and non-decreasing, and g is convex. (More general rules exist).
    *   Perspective function: $$ g(x, t) = t f(x/t) $$ is convex if f is convex (dom $$ t > 0 $$).
    *   Infimal convolution / Partial minimization: $$ h(x) = \inf_{y} g(x, y) $$ is convex if $$ g(x, y) $$ is jointly convex in (x, y).

    ***Cheat Table Idea 3: Operations Preserving Function Convexity***

2.  **Separation Theorems:**
    *   Separating Hyperplane Theorem: Two disjoint convex sets $$ C, D $$ can be separated by a hyperplane $$ a^T x = b $$ (i.e., $$ a^T x \le b $$ for $$ x \in C $$ and $$ a^T x \ge b $$ for $$ x \in D $$).
    *   Strict separation: Requires conditions like one set compact, one closed, or distance > 0.
    *   Supporting Hyperplane Theorem: For a convex set S, at any boundary point $$ x_0 \in \text{bd } S $$, there exists a supporting hyperplane (i.e., $$ a \ne 0 $$ such that $$ a^T x \le a^T x_0 $$ for all $$ x \in S $$).
    *   Geometric intuition and importance for duality/optimality conditions.
3.  **Cones Revisited:**
    *   Dual Cone: $$ K^* = \{ y \mid y^T x \ge 0 \text{ for all } x \in K \} $$.
    *   Properties: $$ K^* $$ is always a closed convex cone. $$ K^{**} = \text{cl}(\text{conv}(K \cup \{0\})) $$. If K is a closed convex cone, $$ K^{**} = K $$.
    *   Self-dual cones: $$ K = K^* $$ (e.g., non-negative orthant, PSD cone, second-order cone).
    *   Generalized Inequalities: Define $$ x \preceq_K y \iff y - x \in K $$.

**Part 4: Convex Optimization Problems**

1.  **Standard Form:**
    Minimize $$ f_0(x) $$
    Subject to $$ f_i(x) \le 0, \quad i=1, \dots, m $$
    $$ h_j(x) = 0, \quad j=1, \dots, p $$
    *   Convex Problem: $$ f_0, \dots, f_m $$ are convex, $$ h_j $$ are affine ($$ h_j(x) = a_j^T x - b_j $$).
    *   Feasible set is convex.
2.  **Key Property: Local Optima are Global Optima.**
    *   Proof using definition of convexity or first-order condition.
    *   If $$ f_0 $$ is strictly convex, the global minimum *point* is unique (if it exists).
3.  **Optimality Conditions (Unconstrained):**
    *   Differentiable $$ f_0 $$: $$ x^* $$ is optimal iff $$ \nabla f_0(x^*) = 0 $$.
    *   Non-differentiable $$ f_0 $$: $$ x^* $$ is optimal iff $$ 0 \in \partial f_0(x^*) $$ (using subgradients - see Part 5).
4.  **Examples of Convex Problems:**
    *   Linear Programming (LP): Affine objective, affine inequalities/equalities.
    *   Quadratic Programming (QP): Convex quadratic objective, affine inequalities/equalities.
    *   Quadratically Constrained Quadratic Programming (QCQP): Convex quadratic objective, convex quadratic inequalities.
    *   Second-Order Cone Programming (SOCP): Linear objective, affine and second-order cone constraints.
    *   Semidefinite Programming (SDP): Linear objective, affine and positive semidefinite cone constraints.
    *   Hierarchy: LP $$ \subset $$ QP $$ \subset $$ SOCP $$ \subset $$ SDP.
    *   Geometric Programming (GP): Can be transformed into convex form.
5.  **Equivalen Problem Transformations:**
    *   Change of variables.
    *   Introducing slack variables.
    *   Epigraph form: Minimizing $$ t $$ subject to $$ f_0(x) \le t $$ and original constraints.
    *   Eliminating equality constraints.

**Part 5: Duality - The Other Side of the Coin**

1.  **The Lagrangian:**
    *   $$ L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x) $$.
    *   Lagrange multipliers / dual variables: $$ \lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p $$. Require $$ \lambda \succeq 0 $$.
2.  **Lagrange Dual Function:**
    *   $$ g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) $$ (where $$ \mathcal{D} $$ is the intersection of domains).
    *   *Key Property: g is always concave*, regardless of the convexity of the primal problem.
    *   Provides lower bounds: For any $$ \lambda \succeq 0 $$ and any feasible x, $$ g(\lambda, \nu) \le f_0(x) $$.
3.  **Lagrange Dual Problem:**
    *   Maximize $$ g(\lambda, \nu) $$
    *   Subject to $$ \lambda \succeq 0 $$.
    *   This is a convex optimization problem (maximizing a concave function).
4.  **Weak Duality:**
    *   Optimal value of dual $$ d^* \le $$ Optimal value of primal $$ p^* $$. Always holds.
    *   Duality Gap: $$ p^* - d^* \ge 0 $$.
5.  **Strong Duality:**
    *   $$ d^* = p^* $$. Does *not* always hold for convex problems.
    *   Constraint Qualifications: Conditions ensuring strong duality.
        *   Slater's Condition: There exists a strictly feasible point $$ \tilde{x} $$ such that $$ f_i(\tilde{x}) < 0 $$ for all $$ i $$ and $$ h_j(\tilde{x}) = 0 $$ for all $$ j $$. (If some $$ f_i $$ are affine, strict inequality is not needed for them).
        *   Refined Slater's Condition (using relative interior for non-polyhedral constraints).
    *   Strong duality often holds for LP, QP, SOCP, SDP under mild conditions.
6.  **Karush-Kuhn-Tucker (KKT) Conditions:**
    *   Necessary conditions for optimality under strong duality. Sufficient under convexity.
    *   Assume strong duality holds, $$ x^* $$ is primal optimal, $$ (\lambda^*, \nu^*) $$ is dual optimal.
    *   1. Primal Feasibility: $$ f_i(x^*) \le 0, h_j(x^*) = 0 $$.
    *   2. Dual Feasibility: $$ \lambda^* \succeq 0 $$.
    *   3. Complementary Slackness: $$ \lambda_i^* f_i(x^*) = 0 $$ for all $$ i $$.
    *   4. Stationarity: $$ \nabla f_0(x^*) + \sum \lambda_i^* \nabla f_i(x^*) + \sum \nu_j^* \nabla h_j(x^*) = 0 $$ (gradient of Lagrangian w.r.t. x is zero).
    *   Interpretation: Complementary slackness means if a constraint is inactive ($$ f_i(x^*) < 0 $$), its dual variable must be zero ($$ \lambda_i^* = 0 $$). If a dual variable is positive ($$ \lambda_i^* > 0 $$), the constraint must be active ($$ f_i(x^*) = 0 $$).
    *   Extension to non-differentiable case using subgradients.

    ***Cheat Table Idea 4: Summary of Duality (Weak, Strong, Slater's, KKT)***

7.  **Sensitivity Analysis:** Interpretation of dual variables as shadow prices ($$ \lambda_i^* \approx -\frac{\partial p^*}{\partial b_i} $$ if $$ f_i(x) = \tilde{f}_i(x) - b_i $$).

**Part 6: Subgradients and Non-Smooth Convexity**

1.  **Motivation:** Many important functions in ML are non-differentiable (e.g., hinge loss, L1 norm).
2.  **Subgradient Definition:**
    *   A vector $$ g $$ is a subgradient of a convex function f at x if $$ f(y) \ge f(x) + g^T (y-x) $$ for all y.
    *   Compare to first-order condition for differentiable functions. Geometrically, $$ g $$ defines a supporting hyperplane to `epi f` at $$ (x, f(x)) $$.
3.  **Subdifferential:**
    *   $$ \partial f(x) = \{ g \mid g \text{ is a subgradient of } f \text{ at } x \} $$.
    *   Properties: $$ \partial f(x) $$ is a closed, convex set. It is non-empty if x is in the relative interior of `dom f`.
    *   If f is differentiable at x, $$ \partial f(x) = \{ \nabla f(x) \} $$.
4.  **Examples:**
    *   Absolute value: $$ f(x) = |x| $$. $$ \partial f(0) = [-1, 1] $$. $$ \partial f(x) = \{ \text{sgn}(x) \} $$ for $$ x \ne 0 $$.
    *   L1 norm: $$ f(x) = \Vert x \Vert_1 $$.
    *   Max function: $$ f(x) = \max \{ f_1(x), \dots, f_k(x) \} $$. $$ \partial f(x) = \text{conv} \{ \nabla f_i(x) \mid f_i(x) = f(x) \} $$ (if $$ f_i $$ are differentiable).
5.  **Subgradient Calculus:** Rules for computing subgradients (sum rule, scaling, composition with affine maps, pointwise max).
6.  **Optimality Condition (Revisited):**
    *   For unconstrained minimization of convex f, $$ x^* $$ is optimal iff $$ 0 \in \partial f(x^*) $$.
7.  **KKT Conditions with Subgradients:** Replace Stationarity condition with $$ 0 \in \partial f_0(x^*) + \sum \lambda_i^* \partial f_i(x^*) + \sum \nu_j^* \nabla h_j(x^*) $$.

**Part 7: Algorithms for Convex Optimization**

1.  **Gradient Descent (for differentiable f):**
    *   Update rule: $$ x_{k+1} = x_k - \alpha_k \nabla f(x_k) $$.
    *   Step size selection ($$ \alpha_k $$): Constant, diminishing, exact line search, backtracking line search.
    *   Convergence:
        *   Convex f + Lipschitz gradient: $$ O(1/k) $$ convergence rate (function value).
        *   Strongly convex f + Lipschitz gradient: Linear convergence $$ O(c^k) $$ ($$ c < 1 $$).
2.  **Subgradient Method (for non-differentiable f):**
    *   Update rule: $$ x_{k+1} = x_k - \alpha_k g_k $$, where $$ g_k \in \partial f(x_k) $$.
    *   Step size selection: Crucial! Typically diminishing, non-summable but square-summable (e.g., $$ \alpha_k = a / (b+k) $$ or $$ a / \sqrt{k} $$). Cannot use line search easily.
    *   Convergence: Slower, $$ O(1/\sqrt{k}) $$ or $$ O(\log k / k) $$ for function value (best value found so far). Not a descent method (function value might increase).
3.  **Newton's Method (for twice differentiable f):**
    *   Update rule: $$ x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k) $$.
    *   Interpretation: Minimizing second-order Taylor approximation.
    *   Convergence: Quadratic convergence near optimum, but expensive per iteration (Hessian inversion). Requires $$ \nabla^2 f(x) \succ 0 $$. Damped Newton method for global convergence.
4.  **Accelerated Methods:**
    *   Nesterov's Accelerated Gradient (NAG): Adds "momentum".
    *   Convergence: Optimal rate $$ O(1/k^2) $$ for convex, $$ O(\sqrt{\kappa} \log(1/\epsilon)) $$ iterations for strongly convex (where $$ \kappa $$ is condition number).
5.  **Coordinate Descent:**
    *   Update one coordinate (or block of coordinates) at a time, minimizing along that direction.
    *   Efficient if coordinate-wise minimization is cheap. Works well for some ML problems (e.g., Lasso).
6.  **Stochastic Gradient Descent (SGD):**
    *   For objectives of the form $$ f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x) $$ (common in ML - Empirical Risk Minimization).
    *   Update rule: $$ x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k) $$ (using gradient of a *single* or mini-batch sample $$ i_k $$).
    *   Noisy gradient estimate, but much cheaper iterations for large N.
    *   Convergence: Slower than GD per iteration, but often faster overall for large datasets. Requires diminishing step sizes. Variants: SGD with momentum, Adagrad, RMSprop, Adam.
7.  **Proximal Algorithms:**
    *   Proximal Operator: $$ \text{prox}_{\alpha f}(z) = \arg \min_x ( f(x) + \frac{1}{2\alpha} \Vert x - z \Vert_2^2 ) $$.
    *   Proximal Gradient Descent: For minimizing $$ f(x) + g(x) $$ where f is smooth, g is convex but possibly non-smooth (e.g., L1 regularization). Update: $$ x_{k+1} = \text{prox}_{\alpha_k g}(x_k - \alpha_k \nabla f(x_k)) $$. Combines gradient step on f with proximal step on g. Also known as ISTA (Iterative Shrinkage-Thresholding Algorithm) for L1. FISTA is accelerated version.
    *   Alternating Direction Method of Multipliers (ADMM): For problems like minimize $$ f(x) + g(z) $$ subject to $$ Ax + Bz = c $$. Breaks problem down.
8.  **Interior Point Methods:**
    *   Handle inequality constraints using barrier functions (e.g., $$ -\log(-f_i(x)) $$).
    *   Solve sequence of unconstrained problems using Newton's method.
    *   Polynomial time complexity for many problem classes (LP, SOCP, SDP). Very effective for medium-sized, structured problems.

    ***Cheat Table Idea 5: Algorithm Comparison (Type, Use Case, Convergence Rate, Cost/Iteration)***

**Part 8: Advanced Topics & ML Connections**

1.  **Fenchel Conjugate (Convex Conjugate):**
    *   $$ f^*(y) = \sup_x (y^T x - f(x)) $$.
    *   Properties: $$ f^* $$ is always convex. Fenchel-Young Inequality: $$ f(x) + f^*(y) \ge x^T y $$. If f is closed convex, $$ f^{**} = f $$.
    *   Connection to Lagrangian duality.
    *   Examples: conjugate of norm is indicator function of dual norm ball. Conjugate of quadratic.
2.  **Moreau Envelope & Proximal Operator:**
    *   $$ M_{\lambda f}(x) = \inf_z ( f(z) + \frac{1}{2\lambda} \Vert x - z \Vert_2^2 ) $$.
    *   Relationship: $$ \text{prox}_{\lambda f}(x) $$ is the minimizer z. $$ \nabla M_{\lambda f}(x) = (x - \text{prox}_{\lambda f}(x)) / \lambda $$.
    *   Smooths the function f.
3.  **Convex Relaxation:**
    *   Approximating non-convex problems with convex ones (e.g., relaxing integer constraints, using convex envelopes). L1 relaxation for sparse recovery (Compressed Sensing). SDP relaxation for MAXCUT.
4.  **Specific ML Applications:**
    *   **Linear Regression:** Least squares (Quadratic objective). Ridge Regression (Strongly convex QP). Lasso (Non-smooth convex, L1 penalty).
    *   **Support Vector Machines (SVM):** Hinge loss (Non-smooth convex). Can be formulated as QP. Dual formulation often used.
    *   **Logistic Regression:** Convex objective (log-likelihood with sigmoid). Often L1/L2 regularized.
    *   **Regularization:** L1 (sparsity), L2 (smoothness/stability), Elastic Net. Interpretation via proximal operators/subgradients.
    *   **Empirical Risk Minimization (ERM):** $$ \min_w \frac{1}{N} \sum L(y_i, f(x_i; w)) + \Omega(w) $$. Convex if Loss L and Regularizer $$ \Omega $$ are convex. SGD is often used.
    *   **Maximum Likelihood Estimation (MLE):** Often involves maximizing log-likelihood, which is concave (equivalent to minimizing convex negative log-likelihood) for exponential families.

---

**Suggestions for Blog Post Structure:**

*   **Start each post with:** What is it? Why is it important? Geometric intuition.
*   **Use lots of examples:** Simple 1D/2D examples first, then standard forms (norms, quadratics, etc.).
*   **Visualizations:** Graphs of functions, sets, epigraphs, separating hyperplanes, level sets, algorithm paths.
*   **Cheat Tables:** Dedicate sections or call-outs to summarize key equivalences and properties.
*   **Code Snippets (Optional):** Simple Python examples using NumPy/SciPy/CVXPY to illustrate concepts or algorithms.
*   **Connect back:** Explicitly mention how concepts (like strong convexity, subgradients, duality) impact algorithm choice and performance in ML contexts, even if detailed applications come later.
*   **Mathematical Rigor:** Balance intuition with precise definitions and theorems. State assumptions clearly (e.g., differentiability, closedness). Use the specified MathJax syntax diligently.

This list should give you plenty of material for a detailed and informative series! Good luck!

Okay, here is the extensive list of concepts, theorems, and connections for your convex analysis blog series, organized into tables for clean referencing.

Remember to add `markdown="1"` to the `<table>` tag if you are placing these tables inside other HTML elements in your Jekyll posts, although standard Markdown tables in Kramdown should parse the MathJax correctly.

---

**Part 1: Convex Sets - The Geometric Foundation**

| Concept               | Definition / Statement                                                                                                                                                                           | Geometric Intuition                                            | Examples / Importance                                                                                                                                          |
| :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Affine Set**        | A set $$ S $$ such that the line through any two distinct points in $$ S $$ is contained in $$ S $$. <br> $$ x_1, x_2 \in S, \theta \in \mathbb{R} \implies \theta x_1 + (1-\theta) x_2 \in S $$ | A flat (subspace + translation).                               | Solutions to $$ Ax=b $$, lines, planes, hyperplanes. Basis for defining dimension.                                                                             |
| **Affine Hull**       | `aff S`: Smallest affine set containing $$ S $$. Set of all affine combinations of points in $$ S $$.                                                                                            | The "flat" generated by the set $$ S $$.                       | Defines the dimension of a convex set.                                                                                                                         |
| **Convex Set**        | A set $$ S $$ such that the line *segment* between any two points in $$ S $$ is contained in $$ S $$. <br> $$ x_1, x_2 \in S, \theta \in [0, 1] \implies \theta x_1 + (1-\theta) x_2 \in S $$    | No "dents" or "holes". A "filled-in" shape.                    | Intervals, disks, $$ \mathbb{R}^n $$, polyhedra, norm balls, ellipsoids, PSD cone. Fundamental building block.                                                 |
| **Convex Hull**       | `conv S`: Smallest convex set containing $$ S $$. Set of all convex combinations of points in $$ S $$.                                                                                           | The shape formed by stretching a rubber band around $$ S $$.   | Carathéodory's Thm: any point in `conv S` is a convex combo of $$ \le n+1 $$ points. Important for optimization theory.                                        |
| **Cone**              | A set $$ S $$ such that for any $$ x \in S $$, the ray $$ \{ \theta x \mid \theta \ge 0 \} $$ is in $$ S $$.                                                                                     | Set containing all positive scalings of its points.            | $$ \mathbb{R}^n_+ $$, lines through origin.                                                                                                                    |
| **Convex Cone**       | A set that is both convex and a cone. Closed under positive linear combinations. <br> $$ x_1, x_2 \in S, \theta_1, \theta_2 \ge 0 \implies \theta_1 x_1 + \theta_2 x_2 \in S $$                  | Cone with a convex cross-section.                              | Non-negative orthant $$ \mathbb{R}^n_+ $$, Second-order cone $$ \mathcal{Q}^n $$, PSD cone $$ \mathbb{S}^n_+ $$. Crucial for Cone Programming (LP, SOCP, SDP). |
| **Conic Hull**        | `cone S`: Smallest convex cone containing $$ S $$. Set of all conic combinations.                                                                                                                | Cone "generated" by $$ S $$.                                   | Used in defining generalized inequalities.                                                                                                                     |
| **Hyperplane**        | $$ \{ x \mid a^T x = b \} $$ for $$ a \ne 0 $$.                                                                                                                                                  | A flat of dimension $$ n-1 $$.                                 | Separates space, fundamental for separation theorems. Affine, thus convex.                                                                                     |
| **Halfspace**         | $$ \{ x \mid a^T x \le b \} $$ or $$ \{ x \mid a^T x \ge b \} $$.                                                                                                                                | Everything on one side of a hyperplane.                        | Building blocks of polyhedra. Convex.                                                                                                                          |
| **Polyhedron**        | Intersection of finitely many halfspaces and hyperplanes. $$ \{ x \mid Ax \preceq b, Cx = d \} $$.                                                                                               | Shape with flat faces (possibly unbounded).                    | Feasible sets of LPs, QPs. Convex.                                                                                                                             |
| **Norm Ball**         | $$ \{ x \mid \Vert x - x_c \Vert \le r \} $$.                                                                                                                                                    | Ball defined by a norm (Euclidean, $$ L_1 $$, $$ L_\infty $$). | Always convex. Used in regularization, robust optimization.                                                                                                    |
| **Ellipsoid**         | $$ \{ x \mid (x - x_c)^T P^{-1} (x - x_c) \le 1 \} $$, $$ P \succ 0 $$.                                                                                                                          | Stretched/rotated ball.                                        | Convex. Used in optimization algorithms (ellipsoid method), state estimation.                                                                                  |
| **PSD Cone**          | $$ \mathbb{S}^n_+ = \{ X \in \mathbb{S}^n \mid X \succeq 0 \} $$.                                                                                                                                | Cone of symmetric positive semidefinite matrices.              | Convex cone. Basis for Semidefinite Programming (SDP), critical in control theory, combinatorial opt relaxation.                                               |
| **Relative Interior** | `relint S`: Interior relative to the affine hull of $$ S $$.                                                                                                                                     | The "middle" of the set, ignoring ambient dimensions.          | Needed for constraint qualifications (Slater's condition).                                                                                                     |

---

**Cheat Table 1: Operations Preserving Set Convexity**

| Operation                 | Description                                                                     | Resulting Set | Condition                           |
| :------------------------ | :------------------------------------------------------------------------------ | :------------ | :---------------------------------- |
| **Intersection**          | $$ \cap_{\alpha} S_\alpha $$                                                    | Convex        | Each $$ S_\alpha $$ is convex.      |
| **Affine Function**       | Image: $$ f(S) = \{ Ax+b \mid x \in S \} $$ <br> Inverse Image: $$ f^{-1}(S) $$ | Convex        | $$ S $$ is convex, f is affine.     |
| **Perspective Function**  | $$ P(S, t) = \{ x/t \mid (x, t) \in S, t > 0 \} $$                              | Convex        | $$ S $$ is convex.                  |
| **Linear-Frac. Func.**    | Image/Inverse Image under $$ f(x) = (Ax+b)/(c^Tx+d) $$                          | Convex        | $$ S $$ is convex (domain careful). |
| **Minkowski Sum**         | $$ S_1 + S_2 = \{ x_1 + x_2 \mid x_1 \in S_1, x_2 \in S_2 \} $$                 | Convex        | $$ S_1, S_2 $$ are convex.          |
| **Scalar Multiplication** | $$ \alpha S = \{ \alpha x \mid x \in S \} $$                                    | Convex        | $$ S $$ is convex.                  |
| **Cartesian Product**     | $$ S_1 \times S_2 = \{ (x_1, x_2) \mid x_1 \in S_1, x_2 \in S_2 \} $$           | Convex        | $$ S_1, S_2 $$ are convex.          |

---

**Part 2 & 3: Convex Functions - Properties and Operations**

| Concept               | Definition / Statement                                                                                                                                                                 | Geometric Intuition                                                 | Importance / Optimality                                                                                                                                  |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Convex Function**   | Domain `dom f` is convex, and $$ f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y) $$ for $$ x, y \in \text{dom } f, \theta \in [0, 1] $$. (Jensen's Inequality)            | Chord between $$ (x, f(x)) $$ and $$ (y, f(y)) $$ lies above graph. | Guarantees local minima are global. Makes optimization tractable.                                                                                        |
| **Epigraph**          | $$ \text{epi } f = \{ (x, t) \mid x \in \text{dom } f, f(x) \le t \} $$.                                                                                                               | Set of points lying on or above the graph of f.                     | $$ f $$ is convex $$ \iff $$ $$ \text{epi } f $$ is a convex set. Fundamental equivalence.                                                               |
| **Strict Convexity**  | Jensen's inequality holds strictly ($$ < $$) for $$ x \ne y, \theta \in (0, 1) $$.                                                                                                     | Graph is strictly "curved up".                                      | Guarantees uniqueness of the global minimum *point* (if one exists).                                                                                     |
| **Strong Convexity**  | $$ f(x) - \frac{m}{2} \Vert x \Vert_2^2 $$ is convex for some $$ m > 0 $$. <br> ($$ \iff f(y) \ge f(x) + \nabla f(x)^T (y-x) + \frac{m}{2} \Vert y - x \Vert_2^2 $$ if differentiable) | Function grows at least quadratically away from minimum.            | Guarantees existence & uniqueness of minimum. Leads to faster (linear) convergence rates for many algorithms (e.g., Gradient Descent). Ridge regression. |
| **Concave Function**  | $$ -f $$ is convex. Jensen's inequality reversed ($$ \ge $$).                                                                                                                          | Chord lies below the graph.                                         | Maximizing concave function = Minimizing convex function. Log-likelihood often concave.                                                                  |
| **Quasiconvex Func.** | Domain is convex, and all sublevel sets $$ \{ x \mid f(x) \le \alpha \} $$ are convex. <br> ($$ \iff f(\theta x + (1-\theta) y) \le \max\{f(x), f(y)\} $$)                             | Level sets are convex.                                              | Weaker than convex. Still allows some specialized optimization (e.g., bisection).                                                                        |
| **Sublevel Set**      | $$ C_\alpha = \{ x \in \text{dom } f \mid f(x) \le \alpha \} $$.                                                                                                                       | Set of points where function is below a value $$ \alpha $$.         | If $$ f $$ is convex, all $$ C_\alpha $$ are convex. Useful for visualizing functions, analyzing feasibility.                                            |

---

**Cheat Table 2: Equivalent Conditions for Function Convexity**

| Condition Type             | Statement                                                                                                            | Requirements                   | Geometric View                                          |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------- | :----------------------------- | :------------------------------------------------------ |
| **0th Order (Def)**        | Jensen's Inequality: <br> $$ f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y) $$                         | $$ \theta \in [0,1] $$         | Chord above graph                                       |
| **0th Order (Epigraph)**   | $$ \text{epi } f $$ is a convex set.                                                                                 |                                | Region above graph is convex                            |
| **1st Order**              | $$ f(y) \ge f(x) + \nabla f(x)^T (y-x) $$                                                                            | $$ f $$ differentiable         | Tangent line is global underestimator                   |
| **2nd Order**              | $$ \nabla^2 f(x) \succeq 0 $$ (Hessian is positive semidefinite)                                                     | $$ f $$ twice diff.            | Curvature is non-negative in all directions             |
| **Restriction to Lines**   | The function $$ g(t) = f(x+tv) $$ is convex in $$ t $$ for all $$ x \in \text{dom } f $$ and all directions $$ v $$. |                                | Function is convex along every line intersecting domain |
| **Strong Convexity (1st)** | $$ f(y) \ge f(x) + \nabla f(x)^T (y-x) + \frac{m}{2} \Vert y-x \Vert_2^2 $$                                          | $$ f $$ diff., $$ m>0 $$       | Quadratic lower bound grows from tangent                |
| **Strong Convexity (2nd)** | $$ \nabla^2 f(x) \succeq m I $$                                                                                      | $$ f $$ twice diff., $$ m>0 $$ | Eigenvalues of Hessian are $$ \ge m $$                  |

---

**Cheat Table 3: Operations Preserving Function Convexity**

| Operation                | Resulting Function $$ h $$                                            | Conditions                                                                                                                                 | Example                                                                                  |
| :----------------------- | :-------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| **Non-negative Sum**     | $$ h(x) = \sum_{i=1}^k w_i f_i(x) $$                                  | $$ f_i $$ convex, $$ w_i \ge 0 $$.                                                                                                         | $$ \Vert x \Vert_2^2 + \lambda \Vert x \Vert_1 $$ (Elastic Net penalty)                  |
| **Affine Composition**   | $$ h(x) = f(Ax+b) $$                                                  | $$ f $$ convex.                                                                                                                            | $$ \Vert Ax-b \Vert_2^2 $$ (Least squares)                                               |
| **Pointwise Max/Sup**    | $$ h(x) = \max_{i} f_i(x) $$ <br> $$ h(x) = \sup_{y \in C} g(x, y) $$ | $$ f_i $$ convex. <br> $$ g(x, y) $$ convex in $$ x $$ for each $$ y \in C $$.                                                             | Hinge loss $$ \max(0, 1-y f(x)) $$. Dual function.                                       |
| **Composition (Scalar)** | $$ h(x) = f(g(x)) $$                                                  | $$ f $$ convex & non-decreasing, $$ g $$ convex. <br> OR $$ f $$ convex & non-increasing, $$ g $$ concave. <br> (More general rules exist) | $$ e^{x^2} $$. $$ (\Vert x \Vert_2^2)^2 $$ (if $$ \Vert x \Vert_2^2 \ge 0 $$ is domain). |
| **Perspective**          | $$ h(x, t) = t f(x/t) $$                                              | $$ f $$ convex, domain $$ t > 0 $$.                                                                                                        | Quadratic-over-linear $$ x^T x / t $$.                                                   |
| **Partial Minimization** | $$ h(x) = \inf_{y} g(x, y) $$                                         | $$ g(x, y) $$ jointly convex in $$ (x, y) $$.                                                                                              | Distance to a convex set $$ \inf_{y \in C} \Vert x-y \Vert $$.                           |

---

**Part 3 Addendum: Separation and Duality Preliminaries**

| Concept                       | Statement                                                                                                                                                              | Geometric Intuition                                                  | Importance                                                                         |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Separating Hyperplane Thm** | Two disjoint convex sets $$ C, D $$ can be separated by a hyperplane $$ a^T x = b $$ (i.e., $$ a^T x \le b $$ for $$ x \in C $$, $$ a^T x \ge b $$ for $$ x \in D $$). | Can draw a line/plane between two disjoint convex shapes.            | Foundation for duality theory, optimality conditions (Farkas' Lemma).              |
| **Supporting Hyperplane Thm** | For a convex set $$ S $$, at any boundary point $$ x_0 \in \text{bd } S $$, there exists $$ a \ne 0 $$ s.t. $$ a^T x \le a^T x_0 $$ for all $$ x \in S $$.             | Can touch a convex set with a hyperplane at any boundary point.      | Relates to subgradients, used in proofs of optimality conditions.                  |
| **Dual Cone**                 | $$ K^* = \{ y \mid y^T x \ge 0 \text{ for all } x \in K \} $$.                                                                                                         | Cone of vectors forming non-negative inner product with K's vectors. | Defines generalized inequalities, appears in KKT conditions for cone programming.  |
| **Self-Dual Cone**            | A cone K such that $$ K^* = K $$.                                                                                                                                      | The cone is its own dual.                                            | $$ \mathbb{R}^n_+ $$, SOCP cone, PSD cone are self-dual. Simplifies some analysis. |

---

**Part 4: Convex Optimization Problems**

| Concept                     | Description                                                                                                                     | Key Properties                                                                                 | Examples                                                                                |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| **Convex Problem**          | Minimize $$ f_0(x) $$ s.t. $$ f_i(x) \le 0 $$ ($$ f_i $$ convex), $$ Ax = b $$ ($$ h_j(x)=a_j^Tx-b_j $$ affine).                | Feasible set is convex. Local minimum is global minimum. Set of optimal points is convex.      | LP, QP, QCQP, SOCP, SDP, Geometric Programming (GP).                                    |
| **Linear Program (LP)**     | Minimize $$ c^T x + d $$ s.t. $$ Gx \preceq h, Ax = b $$.                                                                       | Polyhedral feasible set. Solved efficiently (Simplex, Interior Point).                         | Resource allocation, network flows.                                                     |
| **Quadratic Program (QP)**  | Minimize $$ \frac{1}{2} x^T P x + q^T x + r $$ s.t. $$ Gx \preceq h, Ax = b $$. ($$ P \succeq 0 $$ for convex QP)               | Ellipsoidal level sets (if $$ P \succ 0 $$). Solved efficiently.                               | SVM (dual), portfolio optimization, least squares with linear constraints.              |
| **SOCP**                    | Minimize $$ f^T x $$ s.t. $$ \Vert A_i x + b_i \Vert_2 \le c_i^T x + d_i, Fx = g $$.                                            | More general than LP, QP.                                                                      | Robust LP, Truss design, antenna array design.                                          |
| **SDP**                     | Minimize $$ \text{tr}(CX) $$ s.t. $$ \text{tr}(A_i X) = b_i, X \succeq 0 $$.                                                    | Most general practical class. Involves PSD cone constraints.                                   | Control theory, relaxation of combinatorial problems (MAXCUT), structural optimization. |
| **Optimality (Unconstr.)**  | If $$ f_0 $$ differentiable, $$ \nabla f_0(x^*) = 0 $$. <br> If non-differentiable, $$ 0 \in \partial f_0(x^*) $$ (see Part 6). | Necessary and sufficient for convex $$ f_0 $$.                                                 | Basis for many algorithms.                                                              |
| **Problem Transformations** | Epigraph form, introducing slack variables, eliminating equality constraints, change of variables.                              | Can convert problems to equivalent standard forms solvable by specific algorithms or software. | Converting QCQP to SOCP/SDP. Transforming GP to convex form.                            |

---

**Part 5: Duality**

| Concept                | Definition / Statement                                                                                                                                                                                                                                                                                                                        | Property / Interpretation                                                                                                                                                                                                    |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lagrangian**         | $$ L(x, \lambda, \nu) = f_0(x) + \sum \lambda_i f_i(x) + \sum \nu_j (a_j^T x - b_j) $$. ($$ \lambda \succeq 0 $$)                                                                                                                                                                                                                             | Combines objective and constraints weighted by dual variables $$ \lambda, \nu $$.                                                                                                                                            |
| **Dual Function**      | $$ g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) $$.                                                                                                                                                                                                                                                                          | Always concave (even if primal is not convex). Provides lower bounds on primal optimal value $$ p^* $$: $$ g(\lambda, \nu) \le p^* $$ for $$ \lambda \succeq 0 $$.                                                           |
| **Dual Problem**       | Maximize $$ g(\lambda, \nu) $$ subject to $$ \lambda \succeq 0 $$.                                                                                                                                                                                                                                                                            | Always a convex optimization problem. Optimal value $$ d^* $$.                                                                                                                                                               |
| **Weak Duality**       | $$ d^* \le p^* $$.                                                                                                                                                                                                                                                                                                                            | Always holds. Duality gap = $$ p^* - d^* \ge 0 $$.                                                                                                                                                                           |
| **Strong Duality**     | $$ d^* = p^* $$.                                                                                                                                                                                                                                                                                                                              | Holds for convex problems under Constraint Qualifications (e.g., Slater's). Allows solving primal via dual. Fundamental for KKT conditions and algorithm design (e.g., barrier methods).                                     |
| **Slater's Condition** | (For convex problem) Strong duality holds if there exists a strictly feasible $$ x $$ (i.e., $$ f_i(x) < 0 $$ for non-affine $$ f_i $$, $$ Ax=b $$).                                                                                                                                                                                          | A sufficient condition for strong duality. Easy to check for many problems.                                                                                                                                                  |
| **KKT Conditions**     | Necessary conditions for optimality at $$ x^*, \lambda^*, \nu^* $$ under strong duality (sufficient if primal is convex): <br> 1. Primal Feasibility <br> 2. Dual Feasibility ($$ \lambda^* \succeq 0 $$) <br> 3. Complementary Slackness ($$ \lambda_i^* f_i(x^*) = 0 $$) <br> 4. Stationarity ($$ \nabla_x L(x^*, \lambda^*, \nu^*) = 0 $$) | Relate primal and dual optimal solutions. Provide checkable optimality certificate. Basis for some algorithms (e.g., IPMs implicitly solve KKT). Complementary slackness: active constraints can have $$ \lambda_i^* > 0 $$. |
| **Fenchel Conjugate**  | $$ f^*(y) = \sup_x (y^T x - f(x)) $$.                                                                                                                                                                                                                                                                                                         | Always convex. $$ f^{**} = f $$ if $$ f $$ closed convex. Fenchel-Young: $$ f(x) + f^*(y) \ge x^T y $$. Closely related to Legendre transform and dual function.                                                             |

---

**Cheat Table 4: Duality Summary**

| Feature                   | Primal Problem                                                                                                     | Dual Problem                                                                                                   | Relationship                                                                                                                          |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Goal**                  | Minimize $$ f_0(x) $$ s.t. $$ f_i(x) \le 0, Ax=b $$                                                                | Maximize $$ g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) $$ s.t. $$ \lambda \succeq 0 $$                        | Dual provides lower bound on primal.                                                                                                  |
| **Variables**             | $$ x \in \mathbb{R}^n $$                                                                                           | $$ \lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p $$                                                           | Dual variables correspond to primal constraints.                                                                                      |
| **Convexity**             | Requires $$ f_0, f_i $$ convex.                                                                                    | Always convex (maximize concave $$ g $$).                                                                      | Dual is useful even if primal is not convex (provides bound).                                                                         |
| **Optimal Value**         | $$ p^* $$                                                                                                          | $$ d^* $$                                                                                                      | Weak Duality: $$ d^* \le p^* $$. Strong Duality ($$ d^* = p^* $$) requires convex primal + Constraint Qualification (e.g., Slater's). |
| **Optimality Conditions** | For convex primal + strong duality, $$ x^* $$ is optimal iff KKT conditions hold with some $$ \lambda^*, \nu^* $$. | For convex primal + strong duality, $$ \lambda^*, \nu^* $$ are optimal iff KKT conditions hold with $$ x^* $$. | KKT links primal feasibility, dual feasibility, complementary slackness, and gradient of Lagrangian.                                  |

---

**Part 6: Subgradients - Handling Non-Smoothness**

| Concept                  | Definition / Statement                                                                                                                                 | Geometric Intuition                                                                                                              | Importance / Role                                                                                                                                                                                               |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Subgradient**          | Vector $$ g $$ such that $$ f(y) \ge f(x) + g^T (y-x) $$ for all $$ y $$.                                                                              | $$ g $$ defines slope of a supporting hyperplane to `epi f` at $$ (x, f(x)) $$. Generalizes gradient.                            | Allows defining optimality conditions and algorithms for non-differentiable convex functions (L1 norm, hinge loss).                                                                                             |
| **Subdifferential**      | $$ \partial f(x) = \{ g \mid g \text{ is a subgradient of } f \text{ at } x \} $$.                                                                     | The set of all possible subgradients at $$ x $$. A single vector if $$ f $$ is differentiable, a set otherwise (e.g., at kinks). | $$ \partial f(x) $$ is closed, convex, non-empty (under mild conditions). Captures local behavior of non-smooth function.                                                                                       |
| **Subgradient Calculus** | Rules for computing subdifferentials of sums, compositions, max, etc. (e.g., $$ \partial (f_1+f_2) = \partial f_1 + \partial f_2 $$ under conditions). | -                                                                                                                                | Allows finding subgradients for complex functions built from simpler ones.                                                                                                                                      |
| **Optimality Condition** | $$ x^* $$ minimizes convex $$ f $$ iff $$ 0 \in \partial f(x^*) $$.                                                                                    | At the minimum, a horizontal supporting hyperplane exists.                                                                       | Generalizes $$ \nabla f(x^*) = 0 $$. Foundation for subgradient methods.                                                                                                                                        |
| **KKT (Non-smooth)**     | Stationarity condition becomes $$ 0 \in \partial f_0(x^*) + \sum \lambda_i^* \partial f_i(x^*) + \sum \nu_j^* a_j $$.                                  | Gradient condition replaced by subdifferential inclusion.                                                                        | Extends KKT theory to cover important non-smooth problems like Lasso.                                                                                                                                           |
| **Proximal Operator**    | $$ \text{prox}_{\alpha f}(z) = \arg \min_x ( f(x) + \frac{1}{2\alpha} \Vert x - z \Vert_2^2 ) $$                                                       | A point "near" $$ z $$ that compromises between minimizing $$ f $$ and staying close to $$ z $$. Resolvent of $$ \partial f $$.  | Key component in proximal algorithms (ISTA, FISTA, ADMM) for structured non-smooth problems (e.g., $$ f+g $$ where $$ f $$ smooth, $$ g $$ non-smooth). Often has closed form (e.g., soft-thresholding for L1). |
| **Moreau Envelope**      | $$ M_{\lambda f}(x) = \inf_z ( f(z) + \frac{1}{2\lambda} \Vert x - z \Vert_2^2 ) $$. Value of the prox argmin problem.                                 | A smoothed approximation of $$ f $$.                                                                                             | Always differentiable with gradient related to proximal operator. Connects prox to gradient descent on a smoothed function.                                                                                     |

---

**Part 7: Algorithms for Convex Optimization**

| Algorithm                     | Update Rule / Idea                                                                                             | Use Case                                                                                         | Convergence (Typical)                                                              | Pros / Cons                                                                                                                                       |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Gradient Descent (GD)**     | $$ x_{k+1} = x_k - \alpha_k \nabla f(x_k) $$                                                                   | Differentiable $$ f $$.                                                                          | $$ O(1/k) $$ (convex), Linear $$ O(c^k) $$ (strongly convex).                      | Simple, robust with line search. Can be slow.                                                                                                     |
| **Subgradient Method**        | $$ x_{k+1} = x_k - \alpha_k g_k $$, $$ g_k \in \partial f(x_k) $$                                              | Non-differentiable $$ f $$.                                                                      | $$ O(1/\sqrt{k}) $$ or $$ O(\log k / k) $$ (best point).                           | Handles non-smooth $$ f $$. Slow convergence, not a descent method, requires careful step size choice.                                            |
| **Accelerated GD (Nesterov)** | $$ x_{k+1} = y_k - \alpha_k \nabla f(y_k) $$, $$ y_k $$ uses momentum.                                         | Differentiable $$ f $$.                                                                          | Optimal: $$ O(1/k^2) $$ (convex), Linear $$ O(\sqrt{\kappa} c^k) $$ (str. convex). | Faster convergence than GD. Slightly more complex.                                                                                                |
| **Newton's Method**           | $$ x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k) $$                                                     | Twice differentiable $$ f $$, $$ \nabla^2 f \succ 0 $$.                                          | Quadratic near optimum.                                                            | Very fast near optimum. Expensive per iteration (Hessian inverse), needs modifications (damping) for global convergence.                          |
| **Coordinate Descent**        | Minimize $$ f $$ along one coordinate (or block) at a time.                                                    | $$ f $$ where coordinate updates are cheap.                                                      | Can be fast, depends on coupling.                                                  | Simple, efficient if applicable (e.g., Lasso). Convergence requires care (e.g., separability or random order).                                    |
| **Stochastic GD (SGD)**       | $$ x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k) $$ (uses gradient of one sample/batch $$ f_i $$).              | Large sums / expectations: $$ f(x) = \mathbb{E}[F(x, \xi)] \approx \frac{1}{N}\sum f_i(x) $$.    | $$ O(1/\sqrt{k}) $$ or $$ O(1/k) $$ (average iterate / str. convex).               | Cheap iterations for large datasets. Noisy, slower theoretical convergence per iteration, needs diminishing step size. Many variants (Adam, etc). |
| **Proximal Gradient (ISTA)**  | $$ x_{k+1} = \text{prox}_{\alpha_k g}(x_k - \alpha_k \nabla f(x_k)) $$                                         | Composite objective $$ F(x) = f(x) + g(x) $$ ($$ f $$ smooth, $$ g $$ non-smooth prox-friendly). | $$ O(1/k) $$ (like GD). FISTA (accelerated): $$ O(1/k^2) $$.                       | Handles structured non-smooth problems (e.g., L1 reg). Relies on cheap prox operator.                                                             |
| **ADMM**                      | Alternating updates for $$ \min f(x) + g(z) $$ s.t. $$ Ax+Bz=c $$. Uses augmented Lagrangian.                  | Distributed opt, problems decomposable into simpler parts with linear coupling.                  | Often linear, depends on problem.                                                  | Flexible, handles constraints, good for distributed settings. Tuning parameters can be tricky.                                                    |
| **Interior Point Methods**    | Use barrier functions for inequalities, solve sequence of equality-constrained problems using Newton's method. | LP, QP, SOCP, SDP, GP. Medium-scale, high accuracy needed.                                       | Polynomial time complexity for many classes.                                       | Fast, reliable for standard forms. Not ideal for very large scale, less effective for general non-structured convex problems.                     |

---

This tabular format should provide a good overview and reference point for the concepts in your blog series. Remember to elaborate on each point with definitions, intuition, proofs (where appropriate), and examples in the actual blog posts.
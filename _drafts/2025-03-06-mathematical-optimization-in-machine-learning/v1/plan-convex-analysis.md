---
math: true
---

Plan for Convex Analysis

1. Convex Sets
2. Convex Functions
3. Subgradients and Subdifferential Calculus
4. Convex Optimization
5. Convex Duality

Okay, here is the detailed content outline, meticulously reformatted to follow your specified Kramdown MathJax syntax rules.

---

**Revised & Detailed Content Outline: Convex Analysis & Optimization Series**

**Part 1: The Geometric Foundation - Convex Sets**

1.  **Introduction:**
    *   Motivation: Why is convexity essential in optimization and machine learning? (Predictability, global optima, efficient algorithms, duality).
    *   Contrast: Convex vs. Non-convex problems - the landscape metaphor (smooth valley vs. rugged terrain).
    *   Overview of the series goals.
2.  **Affine Sets:**
    *   Definition: A set $$S$$ is affine if the line through any two points in $$S$$ lies in $$S$$. $$ x_1, x_2 \in S, \theta \in \mathbb{R} \implies \theta x_1 + (1-\theta) x_2 \in S $$.
    *   Affine Combinations: $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \sum \theta_i = 1 $$.
    *   Affine Hull ($$ \text{aff } S $$): The smallest affine set containing $$S$$; the set of all affine combinations of points in $$S$$.
    *   Geometric Interpretation: Flats (subspaces shifted from the origin).
    *   Examples: Solutions to $$ Ax=b $$, points, lines, planes, hyperplanes.
    *   Affine Dimension: Dimension of the affine hull minus 1.
3.  **Convex Sets:**
    *   Definition: A set $$S$$ is convex if the line segment between any two points in $$S$$ lies in $$S$$. $$ x_1, x_2 \in S, \theta \in [0, 1] \implies \theta x_1 + (1-\theta) x_2 \in S $$.
    *   Convex Combinations: $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \sum \theta_i = 1 $$ and $$ \theta_i \ge 0 $$.
    *   Convex Hull ($$ \text{conv } S $$): The smallest convex set containing $$S$$; the set of all convex combinations of points in $$S$$.
    *   **Carathéodory's Theorem:** Any point in $$ \text{conv } S $$ (where $$ S \subseteq \mathbb{R}^n $$) can be expressed as a convex combination of at most $$n+1$$ points from $$S$$.
    *   Examples: Intervals, disks, hyperplanes $$ \{x \mid a^T x = b\} $$, halfspaces $$ \{x \mid a^T x \le b\} $$, polyhedra (intersection of finite halfspaces/hyperplanes, $$ \{x \mid Ax \preceq b, Cx = d\} $$), norm balls $$ \{x \mid \Vert x - x_c \Vert \le r\} $$ (for *any* norm), ellipsoids $$ \{x \mid (x-x_c)^T P^{-1} (x-x_c) \le 1, P \succ 0\} $$.
4.  **Cones:**
    *   Definition: A set $$S$$ is a cone if for every $$ x \in S $$ and $$ \theta \ge 0 $$, we have $$ \theta x \in S $$.
    *   Convex Cones: A set that is both convex and a cone. Equivalent definition: closed under positive linear combinations ($$ x_1, x_2 \in S, \theta_1, \theta_2 \ge 0 \implies \theta_1 x_1 + \theta_2 x_2 \in S $$).
    *   Conic Combinations (Non-negative linear combinations): $$ \sum_{i=1}^k \theta_i x_i $$ where $$ \theta_i \ge 0 $$.
    *   Conic Hull ($$ \text{cone } S $$): Smallest convex cone containing $$S$$; set of all conic combinations.
    *   Important Examples:
        *   Non-negative orthant: $$ \mathbb{R}^n_+ = \{x \in \mathbb{R}^n \mid x_i \ge 0 \} $$.
        *   Second-Order Cone (SOC) / Lorentz Cone / Ice Cream Cone: $$ \mathcal{Q}^n = \{ (x, t) \in \mathbb{R}^{n+1} \mid \Vert x \Vert_2 \le t \} $$.
        *   Positive Semidefinite (PSD) Cone: $$ \mathbb{S}^n_+ = \{ X \in \mathbb{S}^n \mid X \succeq 0 \} $$ (set of symmetric PSD matrices).
5.  **Topological Properties of Convex Sets:**
    *   Interior ($$ \text{int } S $$): Largest open set contained in S.
    *   Closure ($$ \text{cl } S $$): Smallest closed set containing S. (Closure of a convex set is convex).
    *   Boundary ($$ \text{bd } S $$): $$ \text{cl } S \setminus \text{int } S $$.
    *   Relative Interior ($$ \text{relint } S $$): Interior relative to the affine hull of S. ($$ x \in \text{relint } S \iff \exists \epsilon > 0, B(x, \epsilon) \cap \text{aff } S \subseteq S $$).
    *   Importance: $$ \text{relint} $$ is crucial for constraint qualifications in duality (Slater's condition) and for subgradient properties. $$ \text{relint } S $$ is non-empty if $$ S $$ is non-empty.
6.  **Operations Preserving Set Convexity:**
    *   Intersection: The intersection of any collection of convex sets is convex.
    *   Affine Mappings: If $$S$$ is convex and $$ f(x) = Ax+b $$, then the image $$ f(S) = \{ f(x) \mid x \in S \} $$ and inverse image $$ f^{-1}(S) = \{ x \mid f(x) \in S \} $$ are convex.
    *   Perspective Function: $$ P: \mathbb{R}^{n+1} \to \mathbb{R}^n $$ with $$ P(x, t) = x/t $$ (domain $$ \text{dom } P = \mathbb{R}^n \times \mathbb{R}_{++} $$). If $$ C \subseteq \text{dom } P $$ is convex, then $$ P(C) $$ is convex.
    *   Linear-Fractional Function: $$ f(x) = (Ax+b)/(c^Tx+d) $$. Preserves convexity on domain where $$ c^Tx+d > 0 $$.
    *   Minkowski Sum: $$ S_1 + S_2 = \{ x_1 + x_2 \mid x_1 \in S_1, x_2 \in S_2 \} $$. Convex if $$ S_1, S_2 $$ are convex.
7.  **Separation and Supporting Hyperplane Theorems:**
    *   **Separating Hyperplane Theorem:** Let $$C, D$$ be non-empty disjoint convex sets. Then there exists $$ a \ne 0 $$ and $$ b $$ such that $$ a^T x \le b $$ for all $$ x \in C $$ and $$ a^T x \ge b $$ for all $$ x \in D $$. The hyperplane $$ \{x \mid a^T x = b\} $$ separates $$ C $$ and $$ D $$.
    *   **Strict Separation:** If $$C$$ is closed, $$D$$ is compact, and $$ C \cap D = \emptyset $$, then there exists $$ a \ne 0, b $$ such that $$ a^T x < b $$ for $$ x \in C $$ and $$ a^T x > b $$ for $$ x \in D $$. (Other conditions also yield strict separation).
    *   **Supporting Hyperplane Theorem:** Let $$S$$ be a non-empty convex set. For any $$ x_0 \in \text{bd } S $$, there exists a supporting hyperplane to $$S$$ at $$x_0$$; i.e., $$ a \ne 0 $$ such that $$ a^T x \le a^T x_0 $$ for all $$ x \in S $$.
    *   Geometric intuition and significance for optimality conditions and duality proofs.

**Part 2: Convex Functions & The Power of Subgradients**

1.  **Defining Convex Functions:**
    *   Domain: $$ \text{dom } f $$ must be a convex set.
    *   **Jensen's Inequality:**

        $$
        f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)
        $$

        for all $$ x, y \in \text{dom } f, \theta \in [0, 1] $$.
    *   **Epigraph Characterization:** The epigraph $$ \text{epi } f = \{ (x, t) \mid x \in \text{dom } f, f(x) \le t \} $$ is a convex set if and only if $$ f $$ is a convex function.
    *   First-Order Condition (Differentiable Case): If $$ f $$ is differentiable, $$ f $$ is convex iff $$ \text{dom } f $$ is convex and

        $$
        f(y) \ge f(x) + \nabla f(x)^T (y-x)
        $$

        for all $$ x, y \in \text{dom } f $$. (Tangent line lies below the function).
    *   Second-Order Condition (Twice Differentiable Case): If $$ f $$ is twice differentiable, $$ f $$ is convex iff $$ \text{dom } f $$ is convex and its Hessian is positive semidefinite: $$ \nabla^2 f(x) \succeq 0 $$ for all $$ x \in \text{dom } f $$.
    *   Restriction to Lines: $$ f $$ is convex iff the function $$ g(t) = f(x+tv) $$ is convex in $$ t $$ for all $$ x \in \text{dom} f $$ and all directions $$ v $$ such that $$ x+tv \in \text{dom} f $$.
    *   Examples: Affine ($$a^Tx+b$$), Norms ($$\Vert x \Vert$$), Max ($$\max(x_1, ..., x_n)$$), Quadratic ($$ \frac{1}{2} x^T P x + q^T x + r $$ iff $$ P \succeq 0 $$), Log-sum-exp ($$\log \sum e^{x_i}$$), Negative Logarithm ($$-\log x$$ on $$ \mathbb{R}_{++} $$), Negative Entropy ($$ \sum x_i \log x_i $$ on $$ \mathbb{R}^n_{++} $$).
2.  **Subgradients and Subdifferentials:**
    *   Motivation: Generalizing derivatives for non-smooth functions (e.g., $$\vert x \vert$$, $$ \max(0, x) $$, $$ \Vert x \Vert_1 $$) which are common in ML.
    *   **Subgradient Definition:** A vector $$ g \in \mathbb{R}^n $$ is a subgradient of a convex function $$ f $$ at a point $$ x \in \text{dom } f $$ if

        $$
        f(y) \ge f(x) + g^T (y-x)
        $$

        for all $$ y \in \text{dom } f $$.
    *   Geometric Interpretation: The affine function $$ h(y) = f(x) + g^T(y-x) $$ is a global underestimator for $$ f $$ and is exact at $$ x $$. The hyperplane $$ z = f(x) + g^T(y-x) $$ supports $$ \text{epi } f $$ at $$ (x, f(x)) $$.
    *   **Subdifferential $$ \partial f(x) $$:** The set of all subgradients of $$ f $$ at $$ x $$.
    *   **Properties of Subdifferential:**
        *   $$ \partial f(x) $$ is a closed, convex set (possibly empty).
        *   If $$ x \in \text{relint}(\text{dom } f) $$, then $$ \partial f(x) $$ is non-empty and bounded.
        *   If $$ f $$ is differentiable at $$ x $$, then $$ \partial f(x) = \{ \nabla f(x) \} $$. The subgradient is unique and equals the gradient.
    *   Examples:
        *   $$ f(x) = \vert x \vert $$: $$ \partial f(0) = [-1, 1] $$, $$ \partial f(x) = \{\text{sgn}(x)\} $$ for $$ x \ne 0 $$.
        *   $$ f(x) = \max(0, x) $$ (ReLU): $$ \partial f(0) = [0, 1] $$, $$ \partial f(x) = \{0\} $$ for $$ x < 0 $$, $$ \partial f(x) = \{1\} $$ for $$ x > 0 $$.
        *   $$ f(x) = \Vert x \Vert_1 $$: $$ \partial f(0) = \{ g \mid \Vert g \Vert_\infty \le 1 \} $$ (unit box).
        *   Indicator function $$ I_C(x) $$ of a convex set C: $$ \partial I_C(x) $$ is the normal cone to C at x (if $$ x \in C $$), empty otherwise. Denote as $$ N_C(x) $$.
3.  **Generalized Convexity Conditions via Subgradients:**
    *   **Convexity:** A function $$ f $$ with an open domain is convex iff $$ \partial f(x) $$ is non-empty at every $$ x \in \text{dom } f $$.
    *   **Monotonicity of Subgradient:** $$ f $$ is convex iff its subgradient mapping $$ \partial f $$ is monotone: $$ (g_x - g_y)^T (x - y) \ge 0 $$ for any $$ x, y $$ and $$ g_x \in \partial f(x), g_y \in \partial f(y) $$. (For differentiable case: $$ (\nabla f(x) - \nabla f(y))^T(x-y) \ge 0 $$).
    *   **Optimality Condition (Fundamental):** For a convex function $$ f $$, $$ x^\ast $$ minimizes $$ f $$ over $$ \mathbb{R}^n $$ (or $$ \text{dom } f $$ if open) if and only if $$ 0 \in \partial f(x^\ast) $$.
4.  **Strict and Strong Convexity (Generalized View):**
    *   **Strict Convexity:**
        *   Definition: $$ f(\theta x + (1-\theta) y) < \theta f(x) + (1-\theta) f(y) $$ for $$ x \ne y, \theta \in (0, 1) $$.
        *   Subgradient Characterization: Convex $$ f $$ is strictly convex iff for all $$ x \ne y $$, any $$ g \in \partial f(x) $$ satisfies

            $$
            f(y) > f(x) + g^T(y-x)
            $$

        *   Gradient Characterization (Differentiable): $$ f(y) > f(x) + \nabla f(x)^T(y-x) $$ for $$ x \ne y $$. Or, if twice differentiable, $$ \nabla^2 f(x) \succ 0 $$ is sufficient (not necessary if domain is not open).
        *   Implication: Minimizer (if exists) is unique.
    *   **Strong Convexity:**
        *   Definition: There exists $$ m > 0 $$ such that $$ g(x) = f(x) - \frac{m}{2} \Vert x \Vert_2^2 $$ is convex.
        *   Subgradient Characterization: Convex $$ f $$ is $$ m $$-strongly convex iff for all $$ x, y $$ and any $$ g \in \partial f(x) $$,

            $$
            f(y) \ge f(x) + g^T(y-x) + \frac{m}{2}\Vert y-x \Vert_2^2
            $$

        *   Alternative Subgradient Characterization: Convex $$ f $$ is $$ m $$-strongly convex iff $$ (g_x - g_y)^T(x-y) \ge m \Vert x-y \Vert_2^2 $$ for any $$ x,y $$ and $$ g_x \in \partial f(x), g_y \in \partial f(y) $$.
        *   Gradient Characterization (Differentiable):

            $$
            f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\Vert y-x \Vert_2^2
            $$

        *   Hessian Characterization (Twice Differentiable): $$ \nabla^2 f(x) \succeq m I $$.
        *   Implications: Guarantees existence and uniqueness of minimizer (if $$f$$ is closed, proper); leads to linear convergence rates for many algorithms.
5.  **Concave and Quasiconvex Functions:**
    *   **Concave Functions:** $$ f $$ is concave iff $$ -f $$ is convex. Jensen's reversed ($$ \ge $$). Subgradients defined similarly (or via $$ \partial f(x) = - \partial (-f)(x) $$). Hypograph $$ \{ (x, t) \mid t \le f(x) \} $$ is convex.
    *   **Quasiconvex Functions:** Domain is convex and all sublevel sets $$ S_\alpha = \{ x \in \text{dom } f \mid f(x) \le \alpha \} $$ are convex. Equivalent condition: $$ f(\theta x + (1-\theta) y) \le \max\{f(x), f(y)\} $$ for $$ \theta \in [0, 1] $$. (Weaker than convex).
    *   Quasiconcave: $$ -f $$ is quasiconvex. Superlevel sets are convex.

**Part 3: Subgradient Calculus & Function Operations**

1.  **Subgradient Calculus Rules (More Detail):**
    *   Non-negative Scaling: $$ \partial (\alpha f)(x) = \alpha \, \partial f(x) $$ for $$ \alpha > 0 $$.
    *   Sum Rule: If $$ f_1, \dots, f_k $$ are convex, and there exists a point in the intersection of the relative interiors of their domains, then $$ \partial (f_1 + \dots + f_k)(x) = \partial f_1(x) + \dots + \partial f_k(x) $$ (Minkowski sum of sets).
    *   Affine Composition: Let $$ h(x) = f(Ax+b) $$. If $$ f $$ is convex, then $$ \partial h(x) = A^T \partial f(Ax+b) $$.
    *   Pointwise Maximum: Let $$ f(x) = \max \{ f_1(x), \dots, f_k(x) \} $$ where $$ f_i $$ are convex. Let $$ I(x) = \{ i \mid f_i(x) = f(x) \} $$ be the set of active indices at $$ x $$. Then $$ \partial f(x) = \text{conv} \left( \bigcup_{i \in I(x)} \partial f_i(x) \right) $$.
    *   Pointwise Supremum: Let $$ f(x) = \sup_{y \in C} g(x, y) $$ where $$ g(x, y) $$ is convex in $$ x $$ for each $$ y \in C $$. Under certain conditions (e.g., $$ C $$ compact, $$ g $$ upper semi-continuous), **Danskin's Theorem** states that if $$ Y(x) = \{ y^\ast \in C \mid g(x, y^\ast) = f(x) \} $$ is the set of maximizing $$ y $$, then $$ \partial f(x) = \text{conv} \{ \partial_x g(x, y^\ast) \mid y^\ast \in Y(x) \} $$. (Requires $$ g $$ to be differentiable w.r.t $$ x $$ for simpler form).
    *   General Composition: $$ h(x) = f(g_1(x), ..., g_k(x)) $$. Chain rule is complex. If $$ f $$ is convex and non-decreasing in each argument, and $$ g_i $$ are convex, then $$ h $$ is convex. Chain rules for subgradients exist but are often inclusions or require strong assumptions.
    *   Partial Minimization (Infimal Convolution): If $$ g(x, y) $$ is jointly convex and closed, then $$ h(x) = \inf_y g(x, y) $$ is convex. If $$ y^\ast(x) $$ uniquely minimizes $$ g(x, \cdot) $$, then $$ \partial h(x) = \nabla_x g(x, y^\ast(x)) $$ if differentiable w.r.t x. More generally related to $$ \{ g_x \mid (g_x, 0) \in \partial g(x, y^\ast(x)) \} $$.
2.  **Operations Preserving Convexity (Functions):**
    *   Non-negative weighted sum: convex.
    *   Affine composition: convex.
    *   Pointwise max/supremum: convex.
    *   Composition (Scalar): $$ h(x) = f(g(x)) $$. Convex if $$ f $$ cvx/non-decr & $$ g $$ cvx; or $$ f $$ cvx/non-incr & $$ g $$ concave.
    *   Perspective Function: $$ g(x, t) = t f(x/t) $$ is convex if $$ f $$ is convex (domain $$ t>0 $$).
    *   Infimal Convolution: $$ (f_1 \square f_2)(x) = \inf_y (f_1(y) + f_2(x-y)) $$ is convex if $$ f_1, f_2 $$ convex. Related to Minkowski sum of epigraphs.
3.  **Calculus Examples:**
    *   Lasso Objective: $$ f(x) = \frac{1}{2}\Vert Ax-b \Vert_2^2 + \lambda \Vert x \Vert_1 $$. Use sum rule, gradient of quadratic part $$ A^T(Ax-b) $$, and subdifferential of L1 norm $$ \lambda \partial \Vert x \Vert_1 $$. $$ \partial f(x) = \{ A^T(Ax-b) + \lambda g \mid g \in \partial \Vert x \Vert_1 \} $$.
    *   SVM Hinge Loss (per sample): $$ f(w) = \max(0, 1 - y (w^T x + b)) $$. Use max rule and affine composition rule. $$ \partial f(w) $$ is $$ \{0\} $$ if $$ 1 - y(w^Tx+b) < 0 $$, $$ \{-y x\} $$ if $$ 1 - y(w^Tx+b) > 0 $$, and $$ \text{conv}\{0, -y x\} $$ (segment from $$ 0 $$ to $$ -y x $$) if $$ 1 - y(w^Tx+b) = 0 $$.

**Part 4: Convex Optimization Problems & Duality Introduction**

1.  **Standard Form:**
    Minimize $$ f_0(x) $$
    Subject to $$ f_i(x) \le 0, \quad i=1, \dots, m $$
    $$ Ax = b, \quad (\text{or } h_j(x) = a_j^T x - b_j = 0, j=1, \dots, p) $$
    *   Problem is convex if $$ f_0, f_1, \dots, f_m $$ are convex functions and equality constraints are affine.
    *   Domain: $$ \mathcal{D} = (\cap_{i=0}^m \text{dom } f_i) \cap \{ x \mid Ax=b \} $$.
    *   Feasible Set: $$ \{ x \in \mathcal{D} \mid f_i(x) \le 0 \text{ for all } i \} $$. This set is convex.
2.  **Optimality Property:** For a convex problem, any locally optimal point is globally optimal. The set of optimal points $$ X_{opt} $$ is convex. If $$ f_0 $$ is strictly convex, the optimal point (if it exists) is unique.
3.  **Problem Classification:**
    *   Linear Programming (LP): $$ f_0, f_i $$ are affine.
    *   Quadratic Programming (QP): $$ f_0 $$ convex quadratic, $$ f_i $$ affine.
    *   Quadratically Constrained QP (QCQP): $$ f_0, f_i $$ convex quadratic.
    *   Second-Order Cone Programming (SOCP): LP + SOC constraints $$ \Vert A_i x + b_i \Vert_2 \le c_i^T x + d_i $$.
    *   Semidefinite Programming (SDP): LP over the PSD cone $$ \mathbb{S}^n_+ $$. Minimize $$ \text{tr}(CX) $$ s.t. $$ \text{tr}(A_i X) = b_i, X \succeq 0 $$.
    *   Conic Programming: General form Minimize $$ c^T x $$ s.t. $$ Ax = b, x \in K $$ where K is a proper cone. LP, SOCP, SDP are special cases.
    *   Geometric Programming (GP): Can be transformed into convex form.
4.  **Lagrangian Duality Preliminaries:**
    *   **Lagrangian:**

        $$
        L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \nu^T (Ax - b)
        $$

        with dual variables $$ \lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p $$. Domain requires $$ \lambda \succeq 0 $$.
    *   **Lagrange Dual Function:** $$ g(\lambda, \nu) = \inf_{x \in \mathcal{D}_0} L(x, \lambda, \nu) $$, where $$ \mathcal{D}_0 = \cap_{i=0}^m \text{dom } f_i $$.
        *   Property: $$ g(\lambda, \nu) $$ is always a concave function (infimum of affine functions of $$ (\lambda, \nu) $$), regardless of the convexity of the primal problem.
        *   Lower Bound Property: For any $$ \lambda \succeq 0 $$ and feasible $$ x $$, $$ g(\lambda, \nu) \le f_0(x) $$. Proof: $$ g(\lambda, \nu) \le L(x, \lambda, \nu) = f_0(x) + \sum \lambda_i f_i(x) + \nu^T(Ax-b) \le f_0(x) $$.
    *   **Lagrange Dual Problem:**
        Maximize $$ g(\lambda, \nu) $$
        Subject to $$ \lambda \succeq 0 $$.
        *   This is always a convex optimization problem (maximizing a concave function subject to convex constraints $$ \lambda_i \ge 0 $$). Let the optimal value be $$ d^\ast $$.
    *   **Weak Duality:** $$ d^\ast \le p^\ast $$ (where $$ p^\ast $$ is the primal optimal value). Always holds.
    *   **Duality Gap:** $$ p^\ast - d^\ast \ge 0 $$.

**Part 5: Duality Theory & Generalized KKT Conditions**

1.  **Strong Duality:** When does the duality gap equal zero ($$ d^\ast = p^\ast $$)? Not always true, even for convex problems.
2.  **Constraint Qualifications (CQ):** Conditions that ensure strong duality for convex problems.
    *   **Slater's Condition:** For a convex problem, strong duality holds if there exists a point $$ \tilde{x} \in \text{relint } \mathcal{D} $$ that is strictly feasible for the non-affine inequality constraints: $$ f_i(\tilde{x}) < 0 $$ for $$ i $$ where $$ f_i $$ is not affine, $$ f_i(\tilde{x}) \le 0 $$ for affine $$ f_i $$, and $$ A\tilde{x} = b $$.
    *   Implication: If Slater's holds, $$ d^\ast = p^\ast $$, and if $$ p^\ast $$ is finite, the dual optimum is attained (there exists $$ (\lambda^\ast, \nu^\ast) $$ such that $$ g(\lambda^\ast, \nu^\ast) = d^\ast $$).
    *   Refinements exist (e.g., only need strict feasibility for non-affine constraints). Slater's often holds for LP, QP, SOCP, SDP problems encountered in practice.
3.  **Karush-Kuhn-Tucker (KKT) Conditions (Generalized):**
    *   Assume the primal problem is convex and Slater's condition holds (so strong duality holds and dual optimum is attained).
    *   Then $$ x^\ast $$ is primal optimal and $$ (\lambda^\ast, \nu^\ast) $$ is dual optimal **if and only if** they satisfy:
        1.  **Primal Feasibility:** $$ f_i(x^\ast) \le 0 $$ ($$ i=1..m $$), $$ Ax^\ast = b $$.
        2.  **Dual Feasibility:** $$ \lambda^\ast \succeq 0 $$.
        3.  **Complementary Slackness:** $$ \lambda_i^\ast f_i(x^\ast) = 0 $$ for all $$ i=1..m $$.
        4.  **Stationarity:**

            $$
            0 \in \partial f_0(x^\ast) + \sum_{i=1}^m \lambda_i^\ast \partial f_i(x^\ast) + A^T \nu^\ast
            $$

            (Requires subgradient sum rule).
    *   Interpretation:
        *   Complementary Slackness: If a constraint is inactive ($$ f_i(x^\ast) < 0 $$), its dual variable must be zero ($$ \lambda_i^\ast = 0 $$). If a dual variable is positive ($$ \lambda_i^\ast > 0 $$), its constraint must be active ($$ f_i(x^\ast) = 0 $$).
        *   Stationarity: The gradient (or a subgradient) of the objective is balanced by the gradients (subgradients) of the active constraints, weighted by the dual variables.
    *   Sufficiency: If the primal problem is convex, and $$ (x^\ast, \lambda^\ast, \nu^\ast) $$ satisfy the KKT conditions, then $$ x^\ast $$ and $$ (\lambda^\ast, \nu^\ast) $$ are primal and dual optimal. (CQ needed for necessity).
4.  **Fenchel Duality:**
    *   **Convex Conjugate (Fenchel Conjugate):**

        $$
        f^\ast(y) = \sup_{x \in \text{dom } f} (y^T x - f(x))
        $$

    *   Properties: $$ f^\ast $$ is always convex and closed. If $$ f $$ is proper, closed, convex, then $$ f^{\ast\ast} = f $$.
    *   **Fenchel-Young Inequality:**

        $$
        f(x) + f^\ast(y) \ge x^T y
        $$

        Equality holds iff $$ y \in \partial f(x) $$.
    *   **Fenchel Duality Theorem (Simple form):** Consider $$ p^\ast = \inf_x \{ f(x) + h(x) \} $$. The Fenchel dual problem is $$ d^\ast = \sup_y \{ -f^\ast(y) - h^\ast(-y) \} $$. Weak duality $$ d^\ast \le p^\ast $$ always holds. Strong duality $$ p^\ast = d^\ast $$ holds if, e.g., $$ f, h $$ are convex and $$ \text{relint}(\text{dom } f) \cap \text{relint}(\text{dom } h) \ne \emptyset $$.
    *   Connection to Lagrange Duality: Lagrange duality for $$ \inf f_0(x) $$ s.t. $$ Ax=b $$ can be derived from Fenchel duality using $$ f(x) = f_0(x) $$ and $$ h(x) = I_{\{b\}}(Ax) $$ (indicator function).

**Part 6: Algorithms - Gradients & Subgradients**

1.  **First-Order Methods Overview:** Algorithms using gradient or subgradient information.
2.  **Gradient Descent (for smooth $$ f $$):**
    *   Update:

        $$
        x_{k+1} = x_k - \alpha_k \nabla f(x_k)
        $$

    *   Step Size $$ \alpha_k $$: Fixed small, diminishing, or line search (e.g., backtracking).
    *   **Backtracking Line Search:** Start with $$ \alpha = \bar{\alpha} $$, while $$ f(x_k - \alpha \nabla f(x_k)) > f(x_k) - c \alpha \Vert \nabla f(x_k) \Vert_2^2 $$, update $$ \alpha := \beta \alpha $$. (Typical $$ c \in (0, 0.5), \beta \in (0, 1) $$). Guarantees sufficient decrease.
    *   Convergence (Assumes $$ f $$ convex, $$ \nabla f $$ L-Lipschitz):
        *   $$ f(x_k) - p^\ast \le O(1/k) $$.
        *   If $$ f $$ is $$ m $$-strongly convex: $$ \Vert x_k - x^\ast \Vert^2 \le O(c^k) $$, $$ f(x_k) - p^\ast \le O(c^k) $$ (Linear/Geometric Convergence). Rate depends on condition number $$ L/m $$.
3.  **Subgradient Method (for non-smooth convex $$ f $$):**
    *   Update:

        $$
        x_{k+1} = x_k - \alpha_k g_k
        $$

        where $$ g_k $$ is *any* vector in $$ \partial f(x_k) $$.
    *   Key Challenge: $$ -g_k $$ is not necessarily a descent direction ($$ f(x_{k+1}) $$ may be $$ > f(x_k) $$). Cannot use standard line search.
    *   Step Size Rules (Critical): Must be chosen beforehand. Common choices:
        *   Constant step length: $$ \alpha_k = \alpha $$. Convergence requires $$ \alpha $$ small, converges to sublevel set.
        *   Diminishing: $$ \alpha_k \to 0 $$. Requires $$ \sum \alpha_k = \infty $$ (guarantees progress) and often $$ \sum \alpha_k^2 < \infty $$ (dampens noise). E.g., $$ \alpha_k = a / (b+k) $$ or $$ \alpha_k = a / \sqrt{k} $$.
    *   Convergence (Diminishing step size $$ a/\sqrt{k} $$):
        *   Tracks best value: $$ f_{\min, k} = \min_{0 \le i \le k} f(x_i) $$.
        *   $$ f_{\min, k} - p^\ast \le O(1/\sqrt{k}) $$. Slower than gradient descent.
        *   Distance: $$ \Vert x_k - x^\ast \Vert_2^2 $$ convergence rates also known.
4.  **Stochastic (Sub)Gradient Descent (SGD):**
    *   Context: Objective is a sum $$ f(x) = \frac{1}{N} \sum_{i=1}^N f_i(x) $$ or expectation $$ f(x) = \mathbb{E}_\xi [ F(x, \xi) ] $$.
    *   Update: Pick index $$ i_k $$ (or mini-batch) randomly. Compute $$ g_{i_k} \in \partial f_{i_k}(x_k) $$. Update

        $$
        x_{k+1} = x_k - \alpha_k g_{i_k}
        $$

        ($$ g_{i_k} $$ is an unbiased but noisy estimate of a subgradient of $$ f $$).
    *   Properties: Very cheap iterations for large N. High variance. Requires diminishing step sizes (e.g., $$ \alpha_k = a / (b + k) $$).
    *   Convergence: Typically $$ O(1/\sqrt{k}) $$ expected error for convex, $$ O(1/k) $$ for strongly convex (with appropriate step sizes). Can be faster than deterministic methods in terms of wall-clock time for large datasets.
    *   Variants: Momentum, AdaGrad, RMSProp, Adam adjust step sizes adaptively or incorporate momentum to improve practical performance.
5.  **Accelerated Methods (e.g., Nesterov):**
    *   Primarily for **smooth** convex functions. Incorporate "momentum" terms.
    *   Update involves extrapolation step $$ y_k = x_k + \beta_k (x_k - x_{k-1}) $$ and gradient step $$ x_{k+1} = y_k - \alpha_k \nabla f(y_k) $$.
    *   Convergence: Optimal rates for first-order methods on smooth problems: $$ f(x_k) - p^\ast \le O(1/k^2) $$ (convex), faster linear rate $$ O(c^k) $$ depending on $$ \sqrt{L/m} $$ (strongly convex).

**Part 7: Advanced Algorithms - Handling Structure & Constraints**

1.  **Proximal Algorithms:** Target problems of the form $$ \min f(x) + g(x) $$ where $$ f $$ is smooth convex, $$ g $$ is convex but possibly non-smooth and "simple" (prox is easy to compute).
    *   **Proximal Operator:**

        $$
        \text{prox}_{\alpha g}(z) = \arg \min_u \left( g(u) + \frac{1}{2\alpha} \Vert u - z \Vert_2^2 \right)
        $$

    *   Interpretation: Finds a point $$ u $$ that makes a tradeoff between minimizing $$ g $$ and staying close (in Euclidean distance, scaled by $$ \alpha $$) to the input point $$ z $$. Generalizes projection onto convex sets.
    *   **Key Property (Resolvent):** $$ p = \text{prox}_{\alpha g}(z) \iff \frac{1}{\alpha}(z - p) \in \partial g(p) $$.
    *   Examples:
        *   If $$ g(x) = \lambda \Vert x \Vert_1 $$, $$ \text{prox}_{\alpha g}(z) $$ is component-wise soft-thresholding: $$ S_{\lambda \alpha}(z)_i = \text{sgn}(z_i) \max(0, \vert z_i \vert - \lambda \alpha) $$.
        *   If $$ g(x) = I_C(x) $$ (indicator function of closed convex C), $$ \text{prox}_{\alpha g}(z) = \Pi_C(z) $$ (projection onto C).
    *   **Proximal Gradient Method (ISTA / PG M):**
        *   Update:

            $$
            x_{k+1} = \text{prox}_{\alpha_k g}(x_k - \alpha_k \nabla f(x_k))
            $$

            Combines a gradient step on the smooth part $$ f $$ with a proximal step on the non-smooth part $$ g $$. Uses step sizes $$ \alpha_k $$ often chosen via backtracking on $$ f $$.
        *   Convergence: $$ O(1/k) $$ similar to Gradient Descent.
    *   **Accelerated Proximal Gradient (FISTA):** Nesterov-style acceleration for proximal gradient.
        *   Convergence: $$ O(1/k^2) $$. Often significantly faster than ISTA.
    *   **Moreau Envelope (Moreau-Yosida regularization):**

        $$
        M_{\alpha g}(z) = \inf_u \left( g(u) + \frac{1}{2\alpha} \Vert u - z \Vert_2^2 \right)
        $$

        This is the value of the prox objective.
        *   Properties: $$ M_{\alpha g} $$ is convex and continuously differentiable even if $$ g $$ is not, with $$ \nabla M_{\alpha g}(z) = \frac{1}{\alpha}(z - \text{prox}_{\alpha g}(z)) $$. Provides a smooth approximation of $$ g $$.
2.  **Alternating Direction Method of Multipliers (ADMM):**
    *   Solves problems of the form: Minimize $$ f(x) + g(z) $$ subject to $$ Ax + Bz = c $$.
    *   Augmented Lagrangian:

        $$
        L_\rho(x, z, y) = f(x) + g(z) + y^T(Ax+Bz-c) + \frac{\rho}{2}\Vert Ax+Bz-c \Vert_2^2
        $$

    *   Iterates:
        *   $$ x_{k+1} = \arg \min_x L_\rho(x, z_k, y_k) $$
        *   $$ z_{k+1} = \arg \min_z L_\rho(x_{k+1}, z, y_k) $$
        *   $$ y_{k+1} = y_k + \rho (Ax_{k+1} + Bz_{k+1} - c) $$
    *   Effective when the $$ x $$ and $$ z $$ subproblems (often involving prox operators of $$ f $$ and $$ g $$) are much easier to solve than the original coupled problem. Widely used in statistics, ML, signal processing, often for distributed computation. Convergence properties depend on $$ \rho $$ and problem structure.
3.  **Coordinate Descent Methods:**
    *   Iteratively minimize the objective function along coordinate directions or coordinate hyperplanes (blocks).
    *   Update: For $$ i=1..n $$, $$ x_{k+1}^{(i)} = \arg \min_{u} f(x_{k+1}^{(1)}, ..., x_{k+1}^{(i-1)}, u, x_k^{(i+1)}, ..., x_k^{(n)}) $$.
    *   Effective if coordinate-wise minimization is cheap.
    *   Proximal Coordinate Descent: Can handle composite objectives $$ f(x) + \sum g_i(x_i) $$ where $$ g_i $$ are non-smooth but separable (like L1 norm). Combines coordinate updates with proximal steps. Often very efficient for Lasso, SVMs. Convergence requires care (e.g., randomizing order or specific structural assumptions).
4.  **Newton's Method and Interior Point Methods (IPM):**
    *   **Newton's Method (Unconstrained Smooth):**

        $$
        x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k)
        $$

        Requires positive definite Hessian. Quadratic local convergence. Damped Newton for globalization. Expensive iteration (Hessian calculation and system solve).
    *   **Interior Point Methods:** Primarily for constrained problems (LP, QP, SOCP, SDP, GP). Transform inequality constraints $$ f_i(x) \le 0 $$ into logarithmic barrier terms $$ -(1/t) \log(-f_i(x)) $$ added to the objective. Solve a sequence of unconstrained problems (using Newton's method on the barrier objective + equality constraints) as the barrier parameter $$ t \to \infty $$. Follows a "central path". Polynomial time complexity for many problem classes. Very effective for medium-scale structured problems.

**Part 8: ML Connections & Advanced Topics**

1.  **Empirical Risk Minimization (ERM):** Framework: $$ \min_w \frac{1}{N} \sum_{i=1}^N L(y_i, h(x_i; w)) + \Omega(w) $$.
    *   Loss $$ L $$ (e.g., Square Loss, Logistic Loss, Hinge Loss) measures prediction error.
    *   Regularizer $$ \Omega(w) $$ (e.g., L2: $$ \lambda \Vert w \Vert_2^2 $$, L1: $$ \lambda \Vert w \Vert_1 $$) controls complexity.
    *   Convexity depends on $$ L, h, \Omega $$. If convex, the tools apply. Non-smoothness in $$ L $$ (Hinge) or $$ \Omega $$ (L1) requires subgradient/proximal methods. SGD is standard for large $$ N $$.
2.  **Specific Models Revisited:**
    *   **Linear Regression:** L2 Loss (smooth QP). Ridge (L2 loss + L2 reg, strongly convex). Lasso (L2 loss + L1 reg, non-smooth convex -> ISTA/FISTA, CD).
    *   **Logistic Regression:** Logistic Loss (smooth convex) + L1/L2 regularization.
    *   **Support Vector Machines (SVM):** Hinge Loss ($$ \max(0, 1 - y_i w^T x_i) $$ - non-smooth convex) + L2 regularization. Often solved in dual form (smooth QP) or using specialized solvers (like Coordinate Descent on primal/dual).
    *   **Deep Learning:** Non-convex objective. Activation functions like ReLU ($$\max(0,z)$$) are non-smooth. Technically non-differentiable, but uses heuristic "subgradient" (0 or 1) in backpropagation. SGD with momentum/adaptive variants is the de facto standard, using a heuristic subgradient for ReLU. Convex analysis provides insight into components but not global guarantees.
3.  **Convex Relaxation:** Approximating hard (often NP-hard, non-convex) problems with tractable convex ones.
    *   Sparsity: Replace $$ \Vert x \Vert_0 $$ (non-convex) with $$ \Vert x \Vert_1 $$ (convex envelope on unit box). Basis for Compressed Sensing, Lasso feature selection.
    *   Low Rank Matrices: Replace $$ \text{rank}(X) $$ (non-convex) with Nuclear Norm $$ \Vert X \Vert_\ast $$ (sum of singular values, convex envelope on unit spectral norm ball). Used in matrix completion, robust PCA. Proximal methods (using SVD for prox) are common.
    *   Combinatorial Optimization: SDP relaxations (e.g., MAXCUT relaxation using Goemans-Williamson).
4.  **Other Advanced Concepts (Brief Mention):**
    *   **Gauge Functions & Polarity:** Generalizations of norms and dual norms.
    *   **Legendre Transform:** Close relationship between convex conjugate and Legendre transform for differentiable functions.
    *   **Bregman Divergences:** Measures "distance" based on convex functions, generalizing squared Euclidean distance. Used in Mirror Descent algorithm (alternative to subgradient/proximal methods).
    *   **Monotone Operators:** General theory encompassing subdifferentials of convex functions.

---

This version should now align completely with the specified syntax requirements for your Jekyll blog with Kramdown and MathJax.

Okay, let's meticulously apply the specified Kramdown MathJax syntax rules to the cheat tables. All mathematical content within the tables will use the inline `$$ ... $$` delimiters.

---

**Cheat Table 1: Convex Sets - Geometry, Operations, Separation**

| Concept                | Definition / Key Property                                                                                             | Examples / Notes                                                                                               |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Affine Set**         | Contains the line through any two points ($$ \theta \in \mathbb{R} $$).                                               | Subspaces + translation, solutions to $$ Ax=b $$.                                                              |
| **Affine Hull**        | $$ \text{aff } S $$: Smallest affine set containing $$ S $$.                                                          | Defines affine dimension.                                                                                      |
| **Convex Set**         | Contains the line *segment* between any two points ($$ \theta \in [0, 1] $$).                                         | Halfspaces, polyhedra, norm balls, ellipsoids, cones (SOC, PSD), intervals.                                    |
| **Convex Hull**        | $$ \text{conv } S $$: Smallest convex set containing $$ S $$. Set of convex combinations.                             | Carathéodory: $$ n+1 $$ points suffice for convex combo in $$ \mathbb{R}^n $$.                                 |
| **Cone**               | Closed under positive scaling ($$ \theta \ge 0 $$).                                                                   | Rays from origin.                                                                                              |
| **Convex Cone**        | Convex + Cone. Closed under conic combinations ($$ \theta_i \ge 0 $$).                                                | $$ \mathbb{R}^n_+ $$, $$ \mathcal{Q}^n $$ (SOC), $$ \mathbb{S}^n_+ $$ (PSD cone). Basis for Conic Programming. |
| **Interior**           | $$ \text{int } S $$: Largest open set within $$ S $$.                                                                 | Standard topology.                                                                                             |
| **Relative Interior**  | $$ \text{relint } S $$: Interior relative to $$ \text{aff } S $$.                                                     | Crucial for constraint qualifications (Slater's), subgradient properties. Non-empty if $$ S $$ is non-empty.   |
| **Closure**            | $$ \text{cl } S $$: Smallest closed set containing $$ S $$.                                                           | Closure of convex set is convex.                                                                               |
| **Boundary**           | $$ \text{bd } S $$: $$ \text{cl } S \setminus \text{int } S $$.                                                       | Where supporting hyperplanes "touch".                                                                          |
| **Preserving Ops.**    | Intersection, Affine maps (image/preimage), Perspective fn, Linear-Fractional fn, Minkowski Sum ($$ S_1+S_2 $$).      | Tools for proving set convexity.                                                                               |
| **Separating H-plane** | Disjoint convex $$ C, D \implies \exists a \ne 0, b $$ s.t. $$ a^Tx \le b \le a^Ty $$ for $$ x \in C, y \in D $$.     | Foundation for duality. Strict separation under stronger conditions (e.g., one compact, one closed).           |
| **Supporting H-plane** | Convex $$ S $$, $$ x_0 \in \text{bd } S \implies \exists a \ne 0 $$ s.t. $$ a^Tx \le a^Tx_0 $$ for all $$ x \in S $$. | Geometric basis for subgradients. Normal cone is set of supporting hyperplanes' normal vectors at a point.     |

---

**Cheat Table 2: Convex Functions & Subgradients - Core Properties & Equivalences**

| Concept                                 | Definition / Characterization                                                                                                                                                                                                        | Notes / Significance                                                                                                                                                                 |
| :-------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Convex Function $$ f $$**             | 1. $$ \text{dom } f $$ is convex. <br> 2. Jensen: $$ f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y) $$. <br> 3. Epigraph $$ \text{epi } f $$ is a convex set.                                                          | Fundamental. Local minima are global.                                                                                                                                                |
| **(Sub)gradient $$ g $$**               | Vector $$ g $$ s.t. $$ f(y) \ge f(x) + g^T (y-x) $$ for all $$ y \in \text{dom } f $$.                                                                                                                                               | Generalizes gradient to non-smooth convex functions. Defines global affine underestimator. $$ g $$ defines supporting hyperplane to $$ \text{epi } f $$ at $$ (x, f(x)) $$.          |
| **Subdifferential $$ \partial f(x) $$** | Set of all subgradients $$ g $$ of $$ f $$ at $$ x $$.                                                                                                                                                                               | Closed, convex set. Non-empty on $$ \text{relint}(\text{dom } f) $$. $$ \partial f(x) = \{ \nabla f(x) \} $$ if $$ f $$ differentiable at $$ x $$. Captures all "slopes" at $$ x $$. |
| **Examples $$ \partial f(x) $$**        | $$ \vert x \vert $$ at 0: $$ [-1, 1] $$. $$ \max(0,x) $$ at 0: $$ [0,1] $$. $$ \Vert x \Vert_1 $$ at 0: $$ \{ g \mid \Vert g \Vert_\infty \le 1 \} $$. $$ I_C(x) $$: Normal Cone $$ N_C(x) $$.                                       | Quantifies non-smoothness.                                                                                                                                                           |
| **Convexity (via Subgrad)**             | $$ f $$ convex (on open domain) $$ \iff \partial f(x) \ne \emptyset $$ for all $$ x $$. <br> $$ \iff \partial f $$ is monotone: $$ (g_x - g_y)^T(x-y) \ge 0 $$.                                                                      | Connects existence/properties of subgradients to convexity.                                                                                                                          |
| **Optimality Condition**                | $$ x^\ast $$ minimizes convex $$ f \iff 0 \in \partial f(x^\ast) $$.                                                                                                                                                                 | **The** fundamental optimality condition for *any* convex function, smooth or not. Geometrically: a horizontal supporting hyperplane exists.                                         |
| **Strict Convexity**                    | Jensen's inequality is strict ($$<$$) for $$ x \ne y, \theta \in (0,1) $$. <br> *Subgrad Equiv:* $$ f(y) > f(x) + g^T(y-x) $$ for $$ x \ne y, g \in \partial f(x) $$.                                                                | Guarantees uniqueness of minimizer (if one exists).                                                                                                                                  |
| **Strong Convexity ($$m$$)**            | $$ f(x) - \frac{m}{2}\Vert x \Vert_2^2 $$ is convex ($$ m>0 $$). <br> *Subgrad Equiv:* $$ f(y) \ge f(x) + g^T(y-x) + \frac{m}{2}\Vert y-x \Vert_2^2 $$. <br> *Monotonicity Equiv:* $$ (g_x-g_y)^T(x-y) \ge m \Vert x-y \Vert_2^2 $$. | Guarantees existence & uniqueness (if closed proper). Quadratic lower bound. Leads to linear convergence rates. Differentiable Equiv: $$ \nabla^2 f(x) \succeq mI $$.                |
| **Concave Function**                    | $$ -f $$ is convex. Jensen $$ \ge $$. Hypograph is convex.                                                                                                                                                                           | Maximize concave = Minimize convex.                                                                                                                                                  |
| **Quasiconvex Function**                | Sublevel sets $$ \{x \mid f(x) \le \alpha\} $$ are convex. <br> $$ f(\theta x + (1-\theta) y) \le \max\{f(x), f(y)\} $$.                                                                                                             | Weaker condition than convex. Still useful (e.g., bisection method).                                                                                                                 |

---

**Cheat Table 3: Subgradient Calculus & Function Operations**

| Operation                    | Function $$ h(x) $$         | Subdifferential $$ \partial h(x) $$ / Convexity                                                                                                          | Condition Notes                                                                                           |
| :--------------------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **Scaling** ($$\alpha > 0$$) | $$ \alpha f(x) $$           | $$ \partial h(x) = \alpha \, \partial f(x) $$. Convex if $$ f $$ convex.                                                                                 |                                                                                                           |
| **Sum**                      | $$ f_1(x) + f_2(x) $$       | $$ \partial h(x) = \partial f_1(x) + \partial f_2(x) $$ (Minkowski sum). Convex if $$ f_1, f_2 $$ convex.                                                | Requires domain condition (e.g., $$ \text{relint dom } f_1 \cap \text{relint dom } f_2 \ne \emptyset $$). |
| **Affine Composition**       | $$ f(Ax+b) $$               | $$ \partial h(x) = A^T \partial f(Ax+b) $$. Convex if $$ f $$ convex.                                                                                    |                                                                                                           |
| **Pointwise Maximum**        | $$ \max_{i=1..k} f_i(x) $$  | $$ \partial h(x) = \text{conv} ( \cup_{i \in I(x)} \partial f_i(x) ) $$. Convex if all $$ f_i $$ convex. $$ I(x) $$ are indices where $$ f_i(x)=h(x) $$. | Important for hinge loss, max pooling.                                                                    |
| **Pointwise Supremum**       | $$ \sup_{y \in C} g(x,y) $$ | Convex if $$ g(x, y) $$ convex in $$ x $$ for each $$ y $$. Subgradient via Danskin's Thm (under conditions).                                            | Defines dual function.                                                                                    |
| **Composition (Scalar)**     | $$ f(g(x)) $$               | Convex if $$ f $$ cvx/non-decr & $$ g $$ cvx; or $$ f $$ cvx/non-incr & $$ g $$ concave. Chain rule complex.                                             | Be careful with subgradient chain rule.                                                                   |
| **Perspective**              | $$ t f(x/t) $$              | Convex if $$ f $$ convex. (Domain $$ t>0 $$).                                                                                                            | E.g., quadratic-over-linear $$ \frac{x^T x}{t} $$.                                                        |
| **Partial Minimization**     | $$ \inf_{y} g(x, y) $$      | Convex if $$ g(x, y) $$ jointly convex. Subgradient related to $$ \partial_x g(x, y^\ast) $$ where $$ y^\ast $$ is minimizer.                            | Distance to a convex set. Basis for Infimal Convolution ($$ f_1 \square f_2 $$), Moreau Envelope.         |

---

**Cheat Table 4: Convex Optimization Problems & Duality**

| Concept                           | Definition / Formulation                                                                                                                                                                          | Key Property / Notes                                                                                                                                             |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Convex Problem**                | Min $$ f_0(x) $$ s.t. $$ f_i(x) \le 0 $$ ($$ f_i $$ convex), $$ Ax = b $$.                                                                                                                        | Feasible set is convex. Local optimum = Global optimum. Set of optima is convex.                                                                                 |
| **Problem Classes**               | LP, QP, QCQP, SOCP, SDP, GP, Conic Programming.                                                                                                                                                   | Standard forms with efficient solvers available (often via IPMs).                                                                                                |
| **Lagrangian $$ L $$**            | $$ f_0(x) + \sum \lambda_i f_i(x) + \nu^T (Ax - b) $$.                                                                                                                                            | Combines objective and constraints. Dual variables $$ \lambda \succeq 0, \nu $$ unconstrained.                                                                   |
| **Dual Function $$ g $$**         | $$ g(\lambda, \nu) = \inf_{x \in \text{dom } L} L(x, \lambda, \nu) $$.                                                                                                                            | Always concave (even if primal not convex). Provides lower bounds: $$ g(\lambda, \nu) \le p^\ast $$ for $$ \lambda \succeq 0 $$.                                 |
| **Dual Problem**                  | Maximize $$ g(\lambda, \nu) $$ subject to $$ \lambda \succeq 0 $$. Let optimal value be $$ d^\ast $$.                                                                                             | Always a convex optimization problem.                                                                                                                            |
| **Weak Duality**                  | $$ d^\ast \le p^\ast $$.                                                                                                                                                                          | Always holds. Duality Gap = $$ p^\ast - d^\ast \ge 0 $$.                                                                                                         |
| **Strong Duality**                | $$ d^\ast = p^\ast $$.                                                                                                                                                                            | Holds for convex problems under Constraint Qualification (CQ). Enables solving primal via dual, basis for KKT.                                                   |
| **Slater's Condition (CQ)**       | Convex problem + $$ \exists \tilde{x} \in \text{relint dom} $$ s.t. $$ f_i(\tilde{x}) < 0 $$ (for non-affine $$ f_i $$), $$ f_i(\tilde{x}) \le 0 $$ (for affine $$ f_i $$), $$ A\tilde{x} = b $$. | Sufficient for strong duality. Guarantees dual optimum is attained if $$ p^\ast $$ finite.                                                                       |
| **Convex Conjugate $$ f^\ast $$** | $$ f^\ast(y) = \sup_x (y^T x - f(x)) $$.                                                                                                                                                          | Always convex, closed. $$ f^{\ast\ast} = f $$ if $$ f $$ closed proper convex. $$ y \in \partial f(x) \iff x \in \partial f^\ast(y) \iff f(x)+f^\ast(y)=x^Ty $$. |
| **Fenchel Duality**               | Primal: $$ \inf_x f(x) + h(x) $$. Dual: $$ \sup_y -f^\ast(y) - h^\ast(-y) $$. Strong duality under domain overlap conditions.                                                                     | Alternative, often symmetric view of duality. Connects to Lagrange duality.                                                                                      |

---

**Cheat Table 5: KKT Conditions (Generalized)**

*Assume convex problem, strong duality holds, and $$ x^\ast, (\lambda^\ast, \nu^\ast) $$ are primal/dual optimal.*

| Condition Name         | Mathematical Statement                                                                                    | Interpretation                                                                                                                                          |
| :--------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Primal Feasibility** | 1. $$ f_i(x^\ast) \le 0 $$, $$ i=1..m $$ <br> 2. $$ Ax^\ast = b $$                                        | $$ x^\ast $$ satisfies all original constraints.                                                                                                        |
| **Dual Feasibility**   | 3. $$ \lambda^\ast \succeq 0 $$                                                                           | Lagrange multipliers for inequality constraints are non-negative.                                                                                       |
| **Compl. Slackness**   | 4. $$ \lambda_i^\ast f_i(x^\ast) = 0 $$, $$ i=1..m $$                                                     | Active constraints ($$ f_i=0 $$) can have $$ \lambda_i^\ast>0 $$. Inactive constraints ($$ f_i<0 $$) must have $$ \lambda_i^\ast=0 $$.                  |
| **Stationarity**       | 5. $$ 0 \in \partial f_0(x^\ast) + \sum_{i=1}^m \lambda_i^\ast \partial f_i(x^\ast) + A^T \nu^\ast $$     | Subgradient of Lagrangian w.r.t. $$ x $$ contains zero. Balances objective "force" against constraint "forces" at optimum. (Uses subgradient sum rule). |
| **Sufficiency:**       | If primal is convex and $$ (x^\ast, \lambda^\ast, \nu^\ast) $$ satisfy 1-5, they are primal/dual optimal. | KKT provides checkable certificate of optimality for convex problems.                                                                                   |
| **Necessity:**         | If primal is convex and CQ holds (e.g., Slater's), then primal/dual optimal solutions *must* satisfy 1-5. | Guarantees existence of multipliers satisfying the conditions.                                                                                          |

---

**Cheat Table 6: Algorithm Overview (Convex Optimization)**

| Algorithm                    | Handles Non-Smooth? | Mechanism                                                          | Target Problem / Notes                                                                         | Convergence (Typical Convex)         |
| :--------------------------- | :------------------ | :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :----------------------------------- |
| **Gradient Descent**         | No                  | $$ x_{k+1} = x_k - \alpha_k \nabla f(x_k) $$                       | Smooth $$ f $$. Use line search (e.g., backtracking).                                          | $$ O(1/k) $$, Linear (str. convex)   |
| **Subgradient Method**       | Yes                 | $$ x_{k+1} = x_k - \alpha_k g_k $$, $$ g_k \in \partial f(x_k) $$  | General convex $$ f $$. **Not** a descent method. Needs diminishing $$ \alpha_k $$.            | $$ O(1/\sqrt{k}) $$ (best objective) |
| **Accelerated GD (NAG)**     | No                  | Gradient + Momentum                                                | Smooth $$ f $$. Optimal rate for smooth first-order.                                           | $$ O(1/k^2) $$, Linear (str. convex) |
| **Proximal Gradient (ISTA)** | Yes (Structured)    | $$ x_{k+1} = \text{prox}_{\alpha g}(x_k - \alpha \nabla f(x_k)) $$ | $$ \min f(x)+g(x) $$ ($$ f $$ smooth, $$ g $$ prox-friendly). Use backtracking on $$ f $$.     | $$ O(1/k) $$                         |
| **FISTA**                    | Yes (Structured)    | Accelerated Proximal Gradient                                      | $$ \min f(x)+g(x) $$ ($$ f $$ smooth, $$ g $$ prox-friendly).                                  | $$ O(1/k^2) $$                       |
| **ADMM**                     | Yes (Structured)    | Alternating updates, Augmented Lagrangian                          | $$ \min f(x)+g(z) $$ s.t. $$ Ax+Bz=c $$. Good for distributed opt.                             | Depends, often Linear                |
| **Coordinate Descent**       | Yes (Separable)     | Minimize along coordinate axes / blocks                            | Works well if coordinate steps cheap (e.g., Lasso w/ L1). Prox-CD handles separable $$ g_i $$. | Depends on problem structure         |
| **Newton's Method**          | No                  | $$ x_{k+1} = x_k - H_k^{-1} \nabla f(x_k) $$                       | Smooth $$ f $$, $$ \nabla^2 f \succ 0 $$. Quadratic local convergence. Expensive iterations.   | Quadratic                            |
| **Interior Point (IPM)**     | Indirectly          | Barrier method + Newton steps                                      | LP, QP, SOCP, SDP, GP. High accuracy, structured problems.                                     | Polynomial                           |

---

**Cheat Table 7: Proximal Operator**

| Concept                                | Definition / Property                                                                                                                                                 | Notes / Examples                                                                                                   |
| :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Definition**                         | $$ \text{prox}_{\alpha g}(z) = \arg \min_u \left( g(u) + \frac{1}{2\alpha} \Vert u - z \Vert_2^2 \right) $$                                                           | Finds unique point balancing minimizing $$ g $$ and proximity to $$ z $$. Requires $$ g $$ closed proper convex.   |
| **Interpretation**                     | Generalized projection, denoising, regularization step.                                                                                                               | If $$ g=I_C $$, prox is projection $$ \Pi_C $$.                                                                    |
| **Resolvent Property**                 | $$ p = \text{prox}_{\alpha g}(z) \iff \frac{1}{\alpha}(z - p) \in \partial g(p) $$                                                                                    | Fundamental link between prox operator and subdifferential. $$ \text{prox} = (I + \alpha \partial g)^{-1} $$.      |
| **Firm Non-expansiveness**             | $$ \Vert \text{prox}_{\alpha g}(z_1) - \text{prox}_{\alpha g}(z_2) \Vert_2^2 \le (z_1 - z_2)^T (\text{prox}_{\alpha g}(z_1) - \text{prox}_{\alpha g}(z_2)) $$         | Implies non-expansiveness ($$ 1 $$-Lipschitz), ensures stability and convergence of proximal algorithms.           |
| **Moreau Decomposition**               | $$ z = \text{prox}_{\alpha g}(z) + \alpha \, \text{prox}_{(1/\alpha) g^\ast}(z/\alpha) $$                                                                             | Relates prox of function $$ g $$ to prox of its conjugate $$ g^\ast $$.                                            |
| **Moreau Envelope $$ M_{\alpha g} $$** | $$ M_{\alpha g}(z) = \inf_u ( g(u) + \frac{1}{2\alpha} \Vert u - z \Vert_2^2 ) $$ <br> $$ \nabla M_{\alpha g}(z) = \frac{1}{\alpha}(z - \text{prox}_{\alpha g}(z)) $$ | Smooths $$ g $$. Gradient involves prox operator. Proximal Point Algorithm is gradient descent on Moreau envelope. |
| **Example: L1 Norm**                   | $$ g(x) = \lambda \Vert x \Vert_1 \implies \text{prox}_{\alpha g}(z) = S_{\lambda \alpha}(z) $$ (Soft-thresholding)                                                   | $$ S_t(z)_i = \text{sgn}(z_i) \max(0, \vert z_i \vert - t) $$. Widely used in ISTA/FISTA for sparse problems.      |
| **Example: Indicator Fn**              | $$ g(x) = I_C(x) $$ (Indicator of closed convex $$ C $$) $$ \implies \text{prox}_{\alpha g}(z) = \Pi_C(z) $$ (Projection)                                             | Reduces proximal gradient to projected gradient.                                                                   |
| **Example: L2 Norm Sq.**               | $$ g(x) = \frac{\lambda}{2} \Vert x \Vert_2^2 \implies \text{prox}_{\alpha g}(z) = \frac{1}{1+\alpha \lambda} z $$                                                    | Simple shrinkage. Related to Ridge regression.                                                                     |

---

These tables now strictly follow the inline `$$ ... $$` math syntax and use LaTeX commands as requested. They should render correctly with MathJax in your Jekyll setup.
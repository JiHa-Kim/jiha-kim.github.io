**Overall Goal:** To provide a comprehensive understanding of convex analysis and optimization, building from fundamentals to advanced algorithms and ML applications, with a strong emphasis on the unifying role of subgradients and incorporating non-Euclidean perspectives via general norms and Bregman divergences.

**General Requirements for Each Post:**

*   Use Kramdown MathJax syntax as specified.
*   Include numerous visualizations (diagrams of sets, epigraphs, hyperplanes, algorithm steps).
*   Conclude each post with a "Cheat Sheet" summarizing key definitions, theorems, properties, and equivalences covered.
*   Maintain a balance between formal definitions/theorems and intuitive explanations.
*   Briefly mention prerequisites (Linear Algebra, Multivariate Calculus, basic Probability).

---

**Blog Series Plan: Convex Analysis & Optimization for ML**

**Part 1: The Bedrock - Geometry of Convex Sets & Norms**

*   **Title Suggestion:** Convex Analysis Part 1 - The Geometric Landscape of Optimization
*   **Description:** Introduces the fundamental geometric objects – convex sets – and the concept of norms beyond the standard Euclidean distance. Explores their properties, how they combine, and why their structure is crucial for optimization guarantees. Covers basic topology needed later.
*   **Key Concepts:**
    *   Affine Sets, Affine Combinations, Affine Hull.
    *   Convex Sets, Convex Combinations, Convex Hull (Carathéodory's Theorem).
    *   Cones, Convex Cones, Conic Combinations, Conic Hull.
    *   **Norms:** Definition (properties), Examples ($$ \ell_p $$, $$ \ell_\infty $$, Matrix norms: Operator, Frobenius, Nuclear), Norm Balls ($$ B_\Vert \cdot \Vert(c, r) $$). Convexity of norm balls.
    *   **Dual Norms:** Definition ($$ \Vert y \Vert_* = \sup_{\Vert x \Vert \le 1} y^T x $$), Properties ($$ \Vert \cdot \Vert_{**} = \Vert \cdot \Vert $$), Examples ($$ \ell_1/\ell_\infty $$, $$ \ell_2/\ell_2 $$, Frob/Frob, Op/Nuc), Generalized Cauchy-Schwarz ($$ \vert y^T x \vert \le \Vert y \Vert_* \Vert x \Vert $$).
    *   Key Set Examples: Hyperplanes, Halfspaces, Polyhedra, $$ \mathbb{R}^n_+ $$, Second-Order Cone ($$ \mathcal{Q}^n $$), PSD Cone ($$ \mathbb{S}^n_+ $$).
    *   Topological Properties: Open/Closed Sets, Interior, Closure, Boundary, Relative Interior ($$ \text{relint} $$), Compactness (Heine-Borel). Importance of closed sets & $$ \text{relint} $$. Calculus of $$ \text{relint} $$.
    *   Operations Preserving Convexity: Intersection, Affine Mappings, Perspective, Minkowski Sum.
    *   Separation Theorems: Separating Hyperplane Theorem (for disjoint sets), Supporting Hyperplane Theorem (at boundary points). Intuition and role.
    *   Projection onto Convex Sets: $$ P_C(x) = \arg\min_{z \in C} \Vert x - z \Vert_2^2 $$. Definition (Euclidean first), Uniqueness, Characterization ($$ (x - P_C(x))^T (z - P_C(x)) \le 0 $$).
    *   Normal Cone $$ N_C(x) $$ & Tangent Cone $$ T_C(x) $$: Geometric/Formal Definitions, Relationship (Polarity).
*   **Summary:** Establishes the geometric language of convexity, introduces general norms, and the crucial separation principles.

**Part 2: Charting the Landscape - Convex Functions, Subgradients & Generalized Properties**

*   **Title Suggestion:** Convex Analysis Part 2 - Functions That Play Nice (and Their Generalized Slopes)
*   **Description:** Defines convex functions and introduces the universal concept of subgradients. Establishes core properties and generalized notions of smoothness and strong convexity relative to arbitrary norms. Introduces Bregman divergence.
*   **Key Concepts:**
    *   Definition of Convex Functions: Jensen's Inequality, Epigraph characterization. Quasiconvexity/Pseudoconvexity (briefly, via sublevel sets).
    *   Examples: Affine, Quadratic ($$ P \succeq 0 $$), Norms, Max, Log-sum-exp, Indicator function ($$ I_C(x) $$).
    *   **Subgradients:** Motivation (non-smoothness), Definition ($$ f(y) \ge f(x) + g^T(y-x) $$), Geometric Interpretation (supporting hyperplanes to epigraph).
    *   **Subdifferential $$ \partial f(x) $$:** Definition (set of subgradients), Properties (closed, convex, non-empty on $$ \text{relint dom } f $$), Connection to Gradient ($$ \partial f(x) = \{ \nabla f(x) \} $$ if differentiable).
    *   Subgradient Examples: $$ \vert x \vert $$, ReLU, $$ \Vert x \Vert $$ (general norm: $$ \partial \Vert x \Vert = \{ g \mid \Vert g \Vert_* \le 1, g^T x = \Vert x \Vert \} $$ for $$ x \ne 0 $$), $$ I_C(x) $$ ($$ \partial I_C(x) = N_C(x) $$).
    *   Generalized First/Second Order Conditions: Subgradient definition vs $$ \nabla f(x) $$. Relation to Hessian ($$ \nabla^2 f(x) \succeq 0 $$).
    *   Subgradient Monotonicity: $$ (g_x - g_y)^T(x-y) \ge 0 $$.
    *   Strict Convexity: Definition, Subgradient Characterization.
    *   **Generalized Strong Convexity:** Definition ($$ m $$-strongly convex w.r.t. $$ \Vert \cdot \Vert $$ if $$ f(x) - \frac{m}{2}\Vert x \Vert^2 $$ convex), Equivalences ($$ f(y) \ge \dots + \frac{m}{2}\Vert y-x \Vert^2 $$; $$ (g_x-g_y)^T(x-y) \ge m \Vert x-y \Vert^2 $$). Dependence on norm.
    *   **Generalized Smoothness (Lipschitz Gradient):** Definition ($$ L $$-smooth w.r.t. $$ \Vert \cdot \Vert $$ if $$ \Vert \nabla f(x) - \nabla f(y) \Vert_* \le L \Vert x-y \Vert $$ using *dual norm*), Equivalence (Descent Lemma: $$ f(y) \le \dots + \frac{L}{2}\Vert y-x \Vert^2 $$). Dependence on norm.
    *   Directional Derivative: $$ f'(x; d) $$. Relation to subgradient: $$ f'(x; d) = \sup_{g \in \partial f(x)} g^T d $$.
    *   **Bregman Divergence:** Definition ($$ D_\phi(x, y) = \phi(x) - \phi(y) - \nabla \phi(y)^T(x-y) $$ for $$ \phi $$ strictly convex, diff.), Properties ($$ \ge 0 $$, $$ =0 $$ iff $$ x=y $$, not symmetric), Examples ($$ \phi = \frac{1}{2}\Vert \cdot \Vert_2^2 \implies D_\phi = \frac{1}{2}\Vert \cdot - \cdot \Vert_2^2 $$; $$ \phi = \sum x_i \log x_i \implies D_\phi = \text{KL} $$). Connection to Strong Convexity ($$ D_\phi \ge \frac{m}{2}\Vert \cdot - \cdot \Vert^2 $$).
*   **Summary:** Introduces convex functions, the powerful subgradient concept, and generalizes smoothness/strong convexity using norms and Bregman divergence.

**Part 3: Navigating the Terrain - Subgradient Calculus & Function Operations**

*   **Title Suggestion:** Convex Analysis Part 3 - The Calculus of Bumps and Slopes
*   **Description:** Delves into the rules for calculating subgradients/subdifferentials of complex functions built from simpler ones. Reinforces understanding of operations that preserve convexity.
*   **Key Concepts:**
    *   Subgradient Calculus Rules:
        *   Non-negative Scaling ($$ \alpha f $$)
        *   Sum ($$ f_1+f_2 $$, $$ \partial(f_1+f_2)(x) = \partial f_1(x) + \partial f_2(x) $$ under $$ \text{relint} $$ condition).
        *   Affine Composition ($$ f(Ax+b) $$, $$ \partial(f \circ A)(x) = A^T \partial f(Ax+b) $$).
        *   Pointwise Maximum ($$ \max f_i $$, $$ \partial (\max f_i)(x) = \text{conv}(\cup_{j \in \mathcal{I}(x)} \partial f_j(x)) $$).
        *   Pointwise Supremum ($$ \sup_y g(x,y) $$, Danskin's Theorem - simplified).
        *   Composition ($$ h(x)=f(g(x)) $$, Chain rule - state complexity, simple cases e.g. outer scalar).
        *   Partial Minimization ($$ \inf_y g(x,y) $$).
    *   Detailed Examples: Derive $$ \partial f(x) $$ for Lasso penalty ($$ \lambda \Vert x \Vert_1 $$), Hinge loss ($$ \max(0, 1-yt) $$), Huber loss.
    *   Operations Preserving Convexity Revisited (Max, Supremum, Affine Comp, etc.). Perspective Transform ($$ t f(x/t) $$).
    *   Infimal Convolution: $$ (f_1 \square f_2)(x) = \inf_y (f_1(y) + f_2(x-y)) $$. Properties ($$ (f_1 \square f_2)^\ast = f_1^\ast + f_2^\ast $$). Relation to Moreau Envelope.
*   **Summary:** Provides the tools to compute subgradients for functions encountered in practice, building upon basic function operations.

**Part 4: Finding the Bottom - Convex Problems & Optimality**

*   **Title Suggestion:** Convex Analysis Part 4 - Defining the Destination: Optimization Problems & Optimality
*   **Description:** Formally defines convex optimization problems. Establishes the crucial property that local minima are global and introduces the fundamental (sub)gradient-based conditions for optimality.
*   **Key Concepts:**
    *   Convex Optimization Problem: Standard form (minimize $$ f_0 $$ s.t $$ f_i(x) \le 0 $$, $$ Ax=b $$), Convexity requirements ($$ f_i $$ convex, equality constraints affine). Feasible set convexity. Epigraph formulation.
    *   The Global Optimality Property: Proof sketch. Convexity of optimal set $$ X_{opt} $$. Uniqueness under strict convexity.
    *   **Fundamental Optimality Condition (Unconstrained):** $$ x^\ast $$ minimizes convex $$ f \iff 0 \in \partial f(x^\ast) $$. Intuition.
    *   **Optimality Condition (Constrained - Simple Case):** $$ x^\ast $$ minimizes convex $$ f $$ over convex $$ C \iff x^\ast \in C $$ and $$ \exists g \in \partial f(x^\ast) $$ s.t. $$ g^T (y - x^\ast) \ge 0 $$ for all $$ y \in C $$. Geometric meaning: $$ -g \in N_C(x^\ast) $$.
    *   Introduction to Problem Classes: LP, QP, SOCP, SDP, Conic Programming.
*   **Summary:** Defines the target (convex problems) and the fundamental condition identifying a solution (zero subgradient / normal cone condition).

**Part 5: The Alternate Route - Lagrangian Duality**

*   **Title Suggestion:** Convex Analysis Part 5 - Duality: The Flip Side of the Optimization Coin
*   **Description:** Introduces the powerful concept of Lagrangian duality. Defines the Lagrangian, the dual function, and the dual problem, establishing the fundamental weak duality relationship.
*   **Key Concepts:**
    *   The Lagrangian $$ L(x, \lambda, \nu) = f_0(x) + \sum \lambda_i f_i(x) + \sum \nu_j h_j(x) $$.
    *   Lagrange Dual Function $$ g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) $$. Derivation, Concavity property, Interpretation as best lower bound given $$ (\lambda, \nu) $$.
    *   Geometric Interpretation (Supporting Hyperplane to value set).
    *   Lower Bound Property: $$ g(\lambda, \nu) \le p^\ast $$ for $$ \lambda \succeq 0 $$.
    *   The Dual Problem: Maximize $$ g(\lambda, \nu) $$ subject to $$ \lambda \succeq 0 $$. Convexity (Concave maximization).
    *   Weak Duality: $$ d^\ast \le p^\ast $$. Duality Gap $$ p^\ast - d^\ast $$. Examples where gap can be non-zero.
    *   Examples: Derive the dual of a simple LP, QP, Lasso, SVM.
*   **Summary:** Introduces the dual problem, providing lower bounds on the optimal value via the Lagrangian.

**Part 6: Bridging the Gap - Strong Duality, KKT & Fenchel**

*   **Title Suggestion:** Convex Analysis Part 6 - When Worlds Collide: Strong Duality, KKT, and Conjugacy
*   **Description:** Explores conditions under which the duality gap closes (strong duality). Derives and interprets the Karush-Kuhn-Tucker (KKT) conditions using subgradients. Introduces the Fenchel conjugate and Fenchel duality.
*   **Key Concepts:**
    *   Strong Duality: $$ d^\ast = p^\ast $$.
    *   Constraint Qualifications: Slater's Condition (existence of strictly feasible point in $$ \text{relint} $$), Refined Slater for affine constraints. Role in guaranteeing strong duality for convex problems.
    *   **Generalized KKT Conditions:** Derived assuming strong duality:
        1.  Primal Feasibility: $$ f_i(x^\ast) \le 0, h_j(x^\ast) = 0 $$.
        2.  Dual Feasibility: $$ \lambda^\ast \succeq 0 $$.
        3.  Complementary Slackness: $$ \lambda_i^\ast f_i(x^\ast)=0 $$.
        4.  **Stationarity:** $$ 0 \in \partial f_0(x^\ast) + \sum \lambda_i^\ast \partial f_i(x^\ast) + \sum \nu_j^\ast \nabla h_j(x^\ast) $$.
    *   Interpretation of KKT: Necessary under CQ, Sufficient for convex problems. Geometric meaning (esp. stationarity connecting to Part 4).
    *   **Fenchel Conjugate:** $$ f^\ast(y) = \sup_x (y^T x - f(x)) $$. Properties ($$ f^{\ast\ast}=f $$ for closed convex proper f). Fenchel-Young inequality ($$ f(x)+f^\ast(y) \ge x^T y $$). Duality relationship: $$ y \in \partial f(x) \iff x \in \partial f^\ast(y) \iff f(x)+f^\ast(y) = x^T y $$.
    *   Conjugate Examples: Quadratics, Indicator function ($$ I_C^\ast = \sigma_C $$ support function), Norm ($$ (\Vert \cdot \Vert)^\ast = I_{B_{\Vert \cdot \Vert_*}} $$), Squared Norm ($$ (\frac{1}{2}\Vert \cdot \Vert^2)^\ast = \frac{1}{2}\Vert \cdot \Vert_*^2 $$). Support Function $$ \sigma_C(y) $$.
    *   **Fenchel Duality:** Primal $$ \inf_x f(x)+h(Ax) $$ vs Dual $$ \sup_y -f^\ast(-A^T y) - h^\ast(y) $$. Connection to Lagrange duality. Example (e.g., Ridge Regression).
    *   Sensitivity Analysis (briefly): $$ \lambda_i^\ast \approx - \frac{\partial p^\ast}{\partial b_i} $$ interpretation.
*   **Summary:** Connects primal and dual problems via strong duality and KKT conditions, introducing the powerful Fenchel conjugate framework.

**Part 7: The Journey Downhill - Gradient-Based Algorithms & Mirror Descent**

*   **Title Suggestion:** Convex Analysis Part 7 - The Descent Begins: Gradient, Subgradient & Mirror Methods
*   **Description:** Starts exploring iterative methods to find the minimum. Covers gradient descent (smooth), subgradient method (non-smooth), and introduces Mirror Descent for handling non-Euclidean geometries. Includes stochastic variants.
*   **Key Concepts:**
    *   **Gradient Descent (GD):** Update rule ($$ x_{k+1} = x_k - \alpha_k \nabla f(x_k) $$), Step size choices (fixed, backtracking line search). Convergence Analysis (Rates like $$ O(1/k) $$, linear $$ O(c^k) $$ under $$ L $$-smoothness, $$ m $$-strong convexity w.r.t. $$ \Vert \cdot \Vert_2 $$). Assumptions. Condition number $$ L/m $$.
    *   **Subgradient Method:** Update rule ($$ x_{k+1} = x_k - \alpha_k g_k, g_k \in \partial f(x_k) $$), *Not* a descent method. Step size rules (diminishing: $$ \sum \alpha_k = \infty, \sum \alpha_k^2 < \infty $$). Convergence Analysis ($$ O(1/\sqrt{k}) $$ rate for $$ f_{\text{best}} - f^\ast $$ under bounded subgradients/domain w.r.t. $$ \Vert \cdot \Vert_2 $$).
    *   **Stochastic (Sub)Gradient Descent (SGD):** Motivation (large sums/expectations), Update rule (using $$ g_k $$ estimate from sample/mini-batch), Step size importance. Convergence (briefly). Variants (Momentum, Adam - mention intuition).
    *   **(Optional) Accelerated Gradient Descent (Nesterov):** Briefly introduce for smooth Euclidean case ($$ O(1/k^2) $$ rate).
    *   **Mirror Descent (MD):** Motivation (non-Euclidean geometry). Algorithm using Mirror Map $$ \phi $$ & Bregman Divergence $$ D_\phi $$: $$ x_{k+1} = \arg\min_{x \in C} \{ \alpha_k g_k^T x + D_\phi(x, x_k) \} $$ or dual update form. Interpretation (GD is MD with $$ \phi = \frac{1}{2}\Vert \cdot \Vert_2^2 $$). Examples (Exponentiated Gradient for KL divergence/simplex). Convergence depends on $$ \phi $$ match.
*   **Summary:** Introduces fundamental iterative algorithms for smooth and non-smooth problems, including SGD and the non-Euclidean Mirror Descent framework.

**Part 8: Sophisticated Navigation - Proximal Algorithms & Operator Splitting**

*   **Title Suggestion:** Convex Analysis Part 8 - Advanced Tools: Proximal Methods and Operator Splitting
*   **Description:** Covers more advanced algorithms, particularly proximal methods crucial for structured (composite) non-smooth problems common in ML. Introduces ADMM.
*   **Key Concepts:**
    *   **Proximal Operator (Euclidean):** $$ \text{prox}_{\alpha g}(z) = \arg\min_x ( g(x) + \frac{1}{2\alpha} \Vert x-z \Vert_2^2 ) $$. Interpretation (generalized projection/denoising). Properties (Firm non-expansiveness). Resolvent of subdifferential ($$ (I+\alpha \partial g)^{-1} $$). Moreau Decomposition ($$ z = \text{prox}_f(z) + \text{prox}_{f^\ast}(z) $$). Moreau Envelope.
    *   Prox Examples: $$ \ell_1 $$ (Soft Thresholding), $$ \ell_2 $$ (Shrinkage), Indicator $$ I_C $$ (Projection $$ P_C $$), $$ \ell_0 $$ (Hard Thresholding - non-convex prox).
    *   **Proximal Gradient Method (ISTA/PGM):** For $$ \min f(x)+g(x) $$ ($$ f $$ smooth, $$ g $$ prox-friendly). Update: $$ x_{k+1} = \text{prox}_{\alpha_k g}(x_k - \alpha_k \nabla f(x_k)) $$. Convergence ($$ O(1/k) $$ rate under $$ L $$-smooth $$ f $$).
    *   **Accelerated Proximal Gradient (FISTA):** Update rule (momentum). Convergence ($$ O(1/k^2) $$ rate).
    *   **Alternating Direction Method of Multipliers (ADMM):** For $$ \min f(x)+g(z) $$ s.t $$ Ax+Bz=c $$. Augmented Lagrangian, Update steps (x-min, z-min, dual update). Use cases (distributed opt, consensus). Convergence conditions (mention).
    *   **Coordinate Descent (CD):** Basic idea. Proximal variant for separable non-smooth terms. Convergence conditions (mention dependency on coupling).
    *   **Generalized Proximal Operators (Bregman Prox):** Briefly define $$ \text{prox}^\phi_{\alpha g}(z) = \arg\min_x ( g(x) + \frac{1}{\alpha} D_\phi(x, z) ) $$. Mention Bregman Proximal Gradient as generalization of ISTA.
*   **Summary:** Focuses on powerful methods like Proximal Gradient and ADMM suitable for composite optimization problems prevalent in ML.

**Part 9: Reaching the Destination - Convex Relaxation & ML Applications**

*   **Title Suggestion:** Convex Analysis Part 9 - The Payoff: Convexity in Machine Learning
*   **Description:** Demonstrates the practical power by discussing convex relaxation techniques for hard problems and showing how the developed convex optimization tools underpin the guarantees and algorithms for core machine learning models.
*   **Key Concepts:**
    *   **Convex Relaxation:** Idea of replacing non-convex problems with tractable convex surrogates.
        *   Sparsity: $$ \Vert x \Vert_0 $$ vs $$ \Vert x \Vert_1 $$. Connection to Lasso, Compressed Sensing (mention RIP).
        *   Low Rank: $$ \text{rank}(X) $$ vs $$ \Vert X \Vert_\ast $$ (Nuclear Norm). Connection to Matrix Completion (mention incoherence).
        *   SDP Relaxations (e.g., MAXCUT).
    *   **Machine Learning Applications:** Frame as Empirical Risk Minimization $$ \min_w \frac{1}{n}\sum L(y_i, f(x_i; w)) + R(w) $$.
        *   Ridge Regression: L2 Loss + L2 Regularizer. Convex QP. Solvers: Direct, GD, CG.
        *   Lasso: L2 Loss + L1 Regularizer. Convex. Solvers: ISTA, FISTA, Coordinate Descent, ADMM.
        *   Logistic Regression: Logistic Loss (smooth convex) + L1/L2 Regularizer. Convex. Solvers: GD/L-BFGS (smooth), ISTA/FISTA (L1).
        *   Support Vector Machines (SVM): Hinge Loss (non-smooth convex) + L2 Regularizer. Convex. Primal solvers (Subgradient, SGD, CD). Dual formulation (QP). KKT conditions.
        *   Matrix Factorization / Completion: Non-convex vs Nuclear Norm Relaxation (convex).
        *   Deep Learning: Mention non-convexity of overall problem, but role of convex components (ReLU activations, convex loss layers), subgradient heuristics, optimizers (SGD variants, Adam).
    *   **Optimization Software:** Briefly mention CVXPY, CVXR, MOSEK, Gurobi, highlighting their role for standard problems vs implementing iterative methods for large-scale/custom problems.
*   **Summary:** Showcases how convex analysis enables tractable formulations (relaxations) and provides algorithms with guarantees for fundamental ML models.

---
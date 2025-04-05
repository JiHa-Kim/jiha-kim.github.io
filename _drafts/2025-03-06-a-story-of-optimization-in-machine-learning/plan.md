Optimization theory in ML
1. short description of modern optimizers (gradient descent, heavy ball, RMSProp, Adagrad, Adam, AdamW, etc.)
2. Returning to roots in physics: overview of Newtonian mechanics (vectors) vs Lagrangian mechanics (scalars)
3. gradient flow ODE, forward Euler discretization = gradient descent
4. Legendre transform (Lagrangian vs Hamiltonian), hint to convex duality
5. Present bra-ket notation, Einstein summation (covariant vs contravariant components)
6. Basics of convex optimization (duality, barrier, KKT conditions, etc.)
7. Variational formulation of gradient flow, backward Euler discretization: proximal point algorithm (special case: projected gradient descent), Moreau envelope
8. Proximal gradient descent, mirror descent, Bregman divergences (defer to other post)
9.  Preconditioning, whitening as a special case of mirror descent with a quadratic mirror map (quadratic form generates Mahalanobis distance)
10. Momentum, Nesterov momentum, accelerated gradient descent, etc.
11. FAdam: Adam approximates diagonal Fisher information matrix
12. Shampoo, Muon
13. Online learning: online convex optimization, effects of noise and stochastic gradient descent
14. Adam as FTRL
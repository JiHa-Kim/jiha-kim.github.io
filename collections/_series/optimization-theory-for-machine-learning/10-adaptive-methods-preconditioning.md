

- preconditioning: make landscape nicer "flat, straight" for better convergence
- scalefree
  - want positive homogeneity overall degree 0 (scale-free)
  - have positive homoegeneity of degree 1 w.r.t. loss in gradient (linear)
  - want positive homogeneity of degree -1 w.r.t. loss in learning rate
  - ideal example: Newton's method is affine-invariant (independent of basis)
- ODE formulation (augmented) of Adagrad: yields $$\sqrt{\Delta t}$$

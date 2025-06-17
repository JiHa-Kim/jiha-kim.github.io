- [ ] argue why Muon is effective (gradient aligns with input), justify duality map
- [ ] metrized deep learning and local approximation
- [ ] replace `series_index` and `course_index` by `sort_index`
- [ ] dual norm visualization
- [ ] notes on operator theory

Re-structure plan
- [ ] overhaul: elementary functional analysis then tensor calculus then matrix analysis
- [ ] elementary functional analysis:
    1. Vector spaces and Dual spaces
      - [ ] abstract definitions: vector/linear space, algebraic dual space
      - [ ] vector vs covectors: transformation rules
      - [ ] introduce bra-ket notation: type checking
    2. Inner products
      - [ ] Hilbert spaces
      - [ ] Riesz representation theorem: inner product space isometric to dual space,
            unique vector-covector pairing (isomorphism)
    3. Norms and Dual Norms
      - [ ] Banach spaces
      - [ ] Hahn-Banach theorem
      - [ ] Duality mapping

...
1.  Soft inductive biases (regularization)
   - `Functional Analysis`
   - `Tensor Calculus`
   - `Matrix Analysis`
   - `Differential Geometry`
2.  Adaptive methods and preconditioning
3.  Momentum
   - `Statistics and Information Theory`
   - `Information Geometry`
4.  Adam optimizer, info geo view: diagonal Fisher information approximation
   - `Variational Calculus`
   - `Convex Analysis`
   - `Online Learning`
5.  Adam optimizer, online learning view: Discounted Follow-The-Regularized-Leader
   - `Matrix Norms` (part of `Functional Analysis`)
6.  Metrized deep learning (Iso/IsoAdam, Shampoo, Muon)
7.  Parameter-free optimization



## Legend 
x=done
*=revise
(no prefix)=not started

## x Crash courses

- x Linear Algebra  
  └─ x Multivariable Calculus  
      ├─ x Functional Analysis & Matrix Norms  
      │   ├─ x Tensor Calculus  
      │   │   ├─ x Differential Geometry  
      │   │   └─ x Statistics and Information Theory  
      │   │       └─ x Information Geometry (requires both above)  
      │   └─ x Variational Calculus  
      │       └─ x Convex Analysis  
      │           └─ x Online Learning  
      └─ x Numerical Analysis  

## Series

x 1. Introduction to basic mathematical optimization
x 2. Iterative methods: gradient-free vs. gradient-based optimization
x 3. Desirable properties of optimizers
x 4. Speedrun of common gradient-based ML optimizers
x 5. Problem formalization
x 6. Gradient descent and gradient flow
x 7. Challenges of high-dimensional non-convex optimization in deep learning
x 8. Stochastic Gradient Descent and effects of randomness
x 9. Adaptive methods and preconditioning
x 10. Momentum
x 11. Soft inductive biases (regularization)
x 12. Adam optimizer, info geo view: diagonal Fisher information approximation
x 13. Adam optimizer, online learning view: Discounted Follow-The-Regularized-Leader
* 14. Metrized deep learning (Iso/IsoAdam, Shampoo, Muon)
15. Parameter-free optimization
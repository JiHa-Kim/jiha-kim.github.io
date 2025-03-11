Below is a derivation that formalizes mirror descent from a differential geometric viewpoint by explicitly linking it to Riemannian geometry. The key idea is to view the algorithm as a gradient descent method on a manifold whose metric is induced by a strictly convex function.

---

### 1. Setting Up the Manifold

Let \( \mathcal{M} \) be the open convex domain where our optimization is performed. Choose a strictly convex, twice-differentiable function  
\[
\psi : \mathcal{M} \to \mathbb{R},
\]  
which serves as the distance‐generating function. This function defines a Riemannian metric on \( \mathcal{M} \) via its Hessian:
\[
g(x) = \nabla^2 \psi(x).
\]
This metric endows \( \mathcal{M} \) with a notion of “local geometry.” In other words, near each point \( x \), small displacements are measured according to the quadratic form induced by \( g(x) \).

---

### 2. Riemannian Gradient Flow

On a Riemannian manifold, the gradient of a smooth function \( f: \mathcal{M} \to \mathbb{R} \) is defined so that for any tangent vector \( v \) we have
\[
Df(x)[v] = \langle \mathrm{grad}_g f(x),\, v \rangle_{g(x)},
\]
where the inner product is given by
\[
\langle u, v \rangle_{g(x)} = u^\top \nabla^2 \psi(x)\, v.
\]
A standard result from differential geometry is that the Riemannian gradient satisfies
\[
\mathrm{grad}_g f(x) = \nabla^2 \psi(x)^{-1} \nabla f(x).
\]
Thus, the natural (continuous-time) gradient flow on the manifold is given by:
\[
\dot{x}(t) = -\,\mathrm{grad}_g f(x(t)) = -\,\nabla^2 \psi(x(t))^{-1} \nabla f(x(t)).
\]
This is the geometric analogue of steepest descent, adapted to the metric \( g(x) \).

---

### 3. Discretization: From Continuous Flow to Mirror Descent

If one were to discretize the above continuous flow using an explicit Euler scheme with step size \(\eta\), the update becomes
\[
x_{t+1} \approx x_t - \eta\, \nabla^2 \psi(x_t)^{-1} \nabla f(x_t).
\]
While this expression suggests a second-order object (the Hessian of \(\psi\)) is involved, note that the discretization is only first order in \(\eta\).

However, there is an alternative—and very elegant—way to view this update using the mirror map. The mirror map is defined by
\[
\nabla \psi: \mathcal{M} \to \mathcal{M}^*,
\]
which is invertible since \(\psi\) is strictly convex. By defining the dual variable
\[
y = \nabla \psi(x),
\]
we can reinterpret the dynamics. In the dual space, the natural gradient flow update becomes a standard (Euclidean) gradient descent step.

---

### 4. Change of Coordinates: The Mirror Map

Let’s rewrite the update in the dual coordinates. Starting with the discretized Riemannian flow,
\[
x_{t+1} \approx x_t - \eta\, \nabla^2 \psi(x_t)^{-1} \nabla f(x_t),
\]
we apply \(\nabla \psi\) to both sides. Using a first-order (linear) approximation,
\[
\nabla \psi(x_{t+1}) \approx \nabla \psi(x_t) + \nabla^2 \psi(x_t)(x_{t+1} - x_t).
\]
Substitute the update:
\[
\nabla \psi(x_{t+1}) \approx \nabla \psi(x_t) - \eta\, \nabla f(x_t).
\]
Defining \( y_t = \nabla \psi(x_t) \), we get a simple update in the dual space:
\[
y_{t+1} = y_t - \eta\, \nabla f(x_t).
\]
Finally, to recover the primal iterate \( x_{t+1} \), we invert the mirror map:
\[
x_{t+1} = (\nabla \psi)^{-1}(y_{t+1}).
\]

This derivation shows that mirror descent is equivalent to a change of coordinates:  
1. **Map to the dual space:** \( y_t = \nabla \psi(x_t) \).  
2. **Take a standard gradient step:** \( y_{t+1} = y_t - \eta\, \nabla f(x_t) \).  
3. **Map back to the primal space:** \( x_{t+1} = (\nabla \psi)^{-1}(y_{t+1}) \).

---

### 5. Formalizing the Connection

Notice that although the derivation begins with the Riemannian gradient flow, the discretization step involves only a first-order Taylor expansion of \(\nabla \psi\) (a linear approximation). This is why mirror descent is fundamentally a first-order method despite having its roots in second-order geometry:
- The geometry (the metric \( \nabla^2 \psi(x) \)) determines the change of coordinates.
- The update in the dual space is simply a linear (first-order) gradient step.
- The overall procedure amounts to using the local geometry to make a more informed step while still relying only on the first-order derivative \( \nabla f \).

---

### 6. Conclusion

In summary, by formalizing the problem on a Riemannian manifold with metric \( \nabla^2 \psi(x) \), we see that:
- The natural gradient flow on this manifold is
  \[
  \dot{x}(t) = -\nabla^2 \psi(x(t))^{-1} \nabla f(x(t)).
  \]
- A first-order Euler discretization combined with a linearization of the mirror map yields
  \[
  \nabla \psi(x_{t+1}) = \nabla \psi(x_t) - \eta\, \nabla f(x_t).
  \]
- This is exactly the mirror descent update, interpreted as a change of coordinates to the dual space where a standard gradient descent step is performed.

Thus, mirror descent is naturally derived as the discretization of a Riemannian gradient flow, with the key approximation being the first-order Taylor expansion used to relate the dual coordinates \( \nabla \psi(x) \) across iterates. This formalizes the familiar observation that mirror descent is “just” a change of coordinates on the Euclidean gradient step, adapted to the geometry defined by \( \psi \).

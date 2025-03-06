---
layout: post
title: Automatic Stochastic Differentiation
categories:
  - Applied Mathematics
  - Stochastic Calculus
tags:
  - Automatic Differentiation
  - Dual Numbers
  - Ito Calculus
  - Ito Lemma
  - Stratonovich Calculus
date: 2025-02-21 05:41 +0000
math: true
---

## Introduction

Calculus can be quite tedious when computed symbolically by hand. In many modern applications (for example, in machine learning), automatic differentiation is used to efficiently compute derivatives. The key idea is that when performing differentiation, only first‐order (linear) infinitesimal terms survive; higher-order infinitesimals vanish. In other words, if we denote an infinitesimal by \(dt\), then any term like \(dt^k\) with \(k>1\) is negligible in the limit.

From the definition of the derivative,
\[
\lim_{dt\to 0}\frac{f(t+dt)-f(t)}{dt},
\]
we see that for each small increment, we only need to keep track of the function’s value and its linear change.

This is formalized by working in the ring of dual numbers,
\[
\mathbb{R}[\epsilon]/\epsilon^2,
\]
where we think of \(\epsilon\) as our infinitesimal with the property \(\epsilon^2 = 0\) (analogous to \(dt^2 \to 0\)).

For example, consider the product of two dual numbers:
\[
(a+a'\epsilon)(b+b'\epsilon)=ab+(ab'+a'b)\epsilon+a'b'\epsilon^2.
\]
Since \(\epsilon^2=0\), the term \(a'b'\epsilon^2\) vanishes, leaving
\[
(a+a'\epsilon)(b+b'\epsilon)=ab+(ab'+a'b)\epsilon.
\]
This naturally encodes the product rule: \((ab)' = a b' + a' b\).

In stochastic calculus, however, a key difference arises: the quadratic variation of a Brownian motion \(B_t\) is nonzero. In fact, if we define
\[
\Delta B_t := B_t - B_0 \sim N(0,t),
\]
then symbolically we have
\[
dB_t^2 = dt.
\]
This means that when differentiating functions of a Brownian path, second-order terms cannot be dropped as in the deterministic case. To capture this behavior algebraically, we use a larger number system. Instead of \(\mathbb{R}[\epsilon]/\epsilon^2\), we work in
\[
\mathbb{R}[\epsilon]/\epsilon^3,
\]
with the following interpretation:
- We set the “base differential” \(dB_t \sim \epsilon\) (so that the linear term corresponds to the Brownian increment).
- Since \(dB_t^2 \sim dt\), we interpret \(\epsilon^2 \sim dt\).
- All higher-order terms vanish: \(\epsilon^3 = 0\).

Then, for a stochastic differential equation such as
\[
dX_t = \mu\, dt + \sigma\, dB_t,
\]
we write
\[
dX_t = \sigma\,\epsilon + \mu\,\epsilon^2.
\]
Functions \(f(X_t,t)\) are differentiated by writing
\[
df(X_t,t) = f(X_t+dX_t, t+dt) - f(X_t,t)
\]
and extracting the coefficients of \(\epsilon\) and \(\epsilon^2\).

For example, using this framework you can derive Itô’s lemma.

### Deriving Itô’s Lemma

Using \(\mathbb{R}[\epsilon]/\epsilon^3\) with
\[
\epsilon \sim dB_t,\quad \epsilon^2 \sim dt,\quad \epsilon^3=0,
\]
and writing
\[
dX_t = \mu\,dt + \sigma\,dB_t = \mu\,\epsilon^2 + \sigma\,\epsilon,
\]
for a function \(f(X_t,t)\) we have
\[
f(X_t+dX_t, t+dt) = f\Bigl(X_t+\sigma\epsilon+\mu\epsilon^2,\,t+\epsilon^2\Bigr).
\]
A Taylor expansion up to \(\epsilon^2\) gives:
\[
f(X_t+dX_t,t+dt) = f + f_x (\sigma\epsilon+\mu\epsilon^2) + f_t\,\epsilon^2 + \frac{1}{2} f_{xx} (\sigma\epsilon+\mu\epsilon^2)^2,
\]
and since
\[
(\sigma\epsilon+\mu\epsilon^2)^2 = \sigma^2\epsilon^2,
\]
this simplifies to
\[
f(X_t+dX_t,t+dt) = f + f_x\sigma\epsilon + \Bigl(f_x\mu+f_t+\frac{1}{2}f_{xx}\sigma^2\Bigr)\epsilon^2.
\]
Thus, we identify
\[
df = f_x\sigma\,\epsilon + \Bigl(f_t+f_x\mu+\frac{1}{2}f_{xx}\sigma^2\Bigr)\epsilon^2,
\]
which, upon interpreting \(\epsilon\) as \(dB_t\) and \(\epsilon^2\) as \(dt\), gives the familiar Itô’s lemma:
\[
df(X_t,t) = f_x\,\sigma\,dB_t + \Bigl(f_t+f_x\mu+\frac{1}{2}f_{xx}\sigma^2\Bigr)dt.
\]

## Stochastic Calculus as Algebra with Dual Numbers

In deterministic automatic differentiation we work with \(\mathbb{R}[\epsilon]/\epsilon^2\) (since \(\epsilon^2=0\)). For stochastic calculus, because of the nonzero quadratic variation of Brownian motion, we extend our system to
\[
\mathbb{R}[\epsilon]/\epsilon^3,
\]
with:
- \(\epsilon \sim dB_t\) (the Brownian increment), with the property that \(\epsilon^2 \sim dt\),
- \(\epsilon^3 = 0\) (all higher-order terms vanish).

Then, as noted, the SDE
\[
dX_t = \mu\,dt + \sigma\,dB_t
\]
is encoded as
\[
dX_t = \sigma\,\epsilon + \mu\,\epsilon^2.
\]
For a function \(f(X_t,t)\), the differential is computed as
\[
df(X_t,t) = f(X_t+dX_t, t+dt) - f(X_t,t),
\]
and the coefficients of \(\epsilon\) and \(\epsilon^2\) give the \(dB_t\) and \(dt\) components respectively. This algebraic viewpoint immediately yields Itô’s lemma as shown above.

## Stratonovich Equivalence

In many applications—especially in physics—the chain rule in its classical form is desirable. In Itô calculus the chain rule acquires an extra correction term. In Stratonovich calculus the definition is modified so that the chain rule holds as in ordinary calculus.

To motivate this, note that when using the Itô discretization, the Taylor expansion of a function \(f(X_t)\) yields
\[
df = f'(X_t)dX_t + \frac{1}{2} f''(X_t)(dX_t)^2,
\]
with \((dB_t)^2\sim dt\) so that the second-order term is nonzero. In order to recover the familiar chain rule,
\[
df = f'(X_t)\circ dX_t,
\]
we modify the increment by using a midpoint (or symmetric) evaluation. In our algebraic language this means that we define the stochastic increment in the Stratonovich sense as
\[
dX = \sigma\Bigl(\epsilon+\frac{1}{2}\epsilon^2\Bigr) + \mu\,\epsilon^2.
\]
Notice that now
\[
dX = \sigma\,\epsilon + \Bigl(\frac{1}{2}\sigma+\mu\Bigr)\epsilon^2.
\]
When you Taylor expand
\[
f\Bigl(X+\sigma\epsilon+ \Bigl(\frac{1}{2}\sigma+\mu\Bigr)\epsilon^2\Bigr),
\]
you obtain
\[
f(X) + f'(X)\,\sigma\epsilon + \Bigl[f'(X)\Bigl(\frac{1}{2}\sigma+\mu\Bigr)+\frac{1}{2} f''(X)\sigma^2\Bigr]\epsilon^2.
\]
Interpreting \(\epsilon\) as \(dB_t\) and \(\epsilon^2\) as \(dt\), this means
\[
df = f'(X)\sigma\,dB_t + \Bigl[f'(X)\Bigl(\frac{1}{2}\sigma+\mu\Bigr)+\frac{1}{2} f''(X)\sigma^2\Bigr]dt.
\]
By design, the Stratonovich integral is defined so that the chain rule appears in its usual form:
\[
df = f'(X)\circ dX,
\]
which means that the \(dt\) component is taken to be simply
\[
f'(X)\Bigl(\frac{1}{2}\sigma+\mu\Bigr)dt,
\]
with no additional correction. In other words, the extra term \(\frac{1}{2}f''(X)\sigma^2\,dt\) that appears in the Itô expansion is exactly canceled by the modification in the discretization scheme. This is why many texts write the conversion between the Itô and Stratonovich differentials as
\[
dX_{\text{Itô}} = dX_{\text{Strat}} + \frac{1}{2}\sigma\,\sigma'\,dt,
\]
where the extra term accounts for the curvature of the function \(f\).

## Why It’s Neat

Encoding both Itô and Stratonovich calculus in the algebraic system \(\mathbb{R}[\epsilon]/\epsilon^3\) (using \(\epsilon\) for \(dB_t\) and \(\epsilon^2\) for \(dt\)) makes stochastic calculus more accessible to computer algebra systems. In this unified framework the differences between the two interpretations are captured simply by a slight modification of the differential \(dX\). In the Stratonovich formulation one uses
\[
dX = \sigma\Bigl(\epsilon+\frac{1}{2}\epsilon^2\Bigr) + \mu\,\epsilon^2,
\]
which is equivalent to applying a midpoint rule. As a result, the usual (deterministic) chain rule holds without any extra correction.

Here is a complete example using Python and SymPy for the function \(f(X)=X^2\).

```py
from sympy import symbols, expand, collect

X, mu, sigma = symbols('X mu sigma')
e = symbols('e')
# Itô increment: dX = sigma*e + mu*e**2
dX = sigma * e + mu * e**2
X_dual = X + dX

f_X = X**2
f_X_dX = expand((X_dual)**2)
df = f_X_dX - f_X
df = collect(df, e).subs(e**3, 0)  # Truncate terms of order 3 and higher
print(df)  # 2*X*sigma*e + (2*X*mu + sigma**2)*e**2
```

This outputs:
\[
df = 2X\sigma\,\epsilon + \Bigl(2X\mu+\sigma^2\Bigr)\epsilon^2,
\]
which we interpret as
\[
df = 2X\sigma\,dB_t + \Bigl(2X\mu+\sigma^2\Bigr)dt,
\]
matching Itô’s lemma (with the term \(\sigma^2dt\) coming from \((dB_t)^2 = dt\)).

## Stratonovich Example

For the Stratonovich version, we adjust the increment:
\[
dX = \sigma\Bigl(\epsilon+\frac{1}{2}\epsilon^2\Bigr) + \mu\,\epsilon^2.
\]
Then, for any smooth function \(f(X)\), a Taylor expansion yields
\[
f(X+dX)= f(X) + f'(X)\sigma\epsilon + \left[f'(X)\Bigl(\frac{1}{2}\sigma+\mu\Bigr) + \frac{1}{2}f''(X)\sigma^2\right]\epsilon^2.
\]
Interpreting \(\epsilon\) as \(dB_t\) and \(\epsilon^2\) as \(dt\), the differential becomes
\[
df = f'(X)\sigma\,dB_t + \left[f'(X)\Bigl(\frac{1}{2}\sigma+\mu\Bigr)+\frac{1}{2}f''(X)\sigma^2\right]dt.
\]
In the Stratonovich framework the chain rule is taken to hold in its usual form:
\[
df = f'(X)\circ dX,
\]
which implies that the \(dt\) term is interpreted as \(f'(X)(\frac{1}{2}\sigma+\mu)dt\) (i.e. without the extra \(\frac{1}{2}f''(X)\sigma^2\,dt\) term). The difference between the Itô and Stratonovich interpretations is precisely the contribution of that extra term, which is “absorbed” by the modified discretization in the Stratonovich case.

In one dimension, notice that the derivative of \(\sigma\) can be written as

\[
\sigma'(X_t) = \sigma(X_t) \, \frac{d}{dX}\log\sigma(X_t),
\]

so that

\[
\frac{1}{2}\sigma(X_t)\sigma'(X_t) = \frac{1}{2}\sigma(X_t)^2 \, \frac{d}{dX}\log\sigma(X_t).
\]

Thus, the conversion term \(\frac{1}{2}\sigma\sigma'\) is equivalently written as

\[
\frac{1}{2}\sigma^2 \nabla\log\sigma,
\]

where \(\nabla\log\sigma\) is simply the derivative of \(\log\sigma\) in one dimension. This is a compact way to express the impact of the state-dependent noise amplitude on the drift when switching from the Itô to the Stratonovich interpretation.

## Complete Dual Number Example

Below is a more complete Python example implementing dual numbers to derive Itô’s lemma for \(f(X)=X^2\):

```py
import math

class DualNumber:
    def __init__(self, real, e=0, e2=0):
        self.real = real  # Standard real part
        self.e = e        # Coefficient for ε (Brownian increment)
        self.e2 = e2      # Coefficient for ε² (time increment)

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real + other.real,
                self.e + other.e,
                self.e2 + other.e2
            )
        else:
            return DualNumber(
                self.real + other,
                self.e,
                self.e2
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            real = self.real * other.real
            e = self.real * other.e + self.e * other.real
            # Here we neglect terms of order 3 and higher (ε³=0)
            e2 = self.real * other.e2 + self.e * other.e + self.e2 * other.real
            return DualNumber(real, e, e2)
        else:
            return DualNumber(
                self.real * other,
                self.e * other,
                self.e2 * other
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __repr__(self):
        return f"DualNumber({self.real} + {self.e} ε + {self.e2} ε²)"

def exp(d):
    if isinstance(d, DualNumber):
        a = d.real
        b = d.e
        c = d.e2
        exp_a = math.exp(a)
        real = exp_a
        e_term = exp_a * b
        e2_term = exp_a * (b**2 / 2 + c)
        return DualNumber(real, e_term, e2_term)
    else:
        return math.exp(d)

# Example usage: derive Itô's lemma for f(X)=X²
X = 5.0       # Current value of X
mu = 2.0      # Drift coefficient (dX = mu dt + sigma dB)
sigma = 3.0   # Diffusion coefficient

# Represent dX as a dual number: sigma ε + mu ε²
dX = DualNumber(0, sigma, mu)
X_dual = DualNumber(X) + dX  # X + dX

# Compute f(X + dX) = (X + dX)²
f_X_sq = X_dual * X_dual

# Subtract f(X) to get df
df_dual = f_X_sq - DualNumber(X**2)

print("Differential df:")
print(f"df = ({df_dual.e2}) dt + ({df_dual.e}) dB")

# Verify with Itô's formula
expected_dt = 2 * X * mu + sigma**2
expected_dB = 2 * X * sigma
print("\nExpected from Itô's lemma:")
print(f"df = ({expected_dt}) dt + ({expected_dB}) dB")
```

The output is:
```
Differential df:
df = (29.0) dt + (30.0) dB

Expected from Itô's lemma:
df = (29.0) dt + (30.0) dB
```

This confirms that our dual number framework reproduces Itô’s lemma correctly.

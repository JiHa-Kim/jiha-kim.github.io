---
layout: post
title: Automatic Stochastic Differentiation
date: 2025-02-21 05:41 +0000
math: true
---

# Stochastic Calculus as Algebra with Dual Numbers

Stochastic calculus becomes algebraic with dual numbers. Deterministic derivatives use $$ \mathbb{R}[\epsilon]/\epsilon^2 $$ (where $$ \epsilon^2 = 0 $$). For stochastic calculus, extend to $$ \mathbb{R}[\epsilon]/\epsilon^3 $$:
- Brownian motion $$ dB_t \sim \epsilon $$, with $$ \langle \epsilon^2 \rangle = dt $$,
- Time $$ dt \sim \epsilon^2 $$, and $$ \epsilon^3 = 0 $$,
- $$ dX_t = \mu \, dt + \sigma \, dB_t $$ becomes $$ \sigma \epsilon + \mu \epsilon^2 $$.

For $$ f(X_t) $$, $$ df = f(X_t + dX_t) - f(X_t) $$ extracts the $$ dB_t $$- and $$ dt $$-coefficients from the $$ \epsilon $$- and $$ \epsilon^2 $$-terms.

## Itô’s Framework

Using SymPy for $$ f(X) = X^2 $$:

```py
from sympy import symbols, expand, collect

X, mu, sigma = symbols('X mu sigma')
e = symbols('e')
dX = sigma * e + mu * e**2
X_dual = X + dX

f_X = X**2
f_X_dX = expand((X_dual)**2)
df = f_X_dX - f_X
df = collect(df, e).subs(e**3, 0)  # Truncate O(e^3)
print(df)  # 2*X*sigma*e + (2*X*mu + sigma**2)*e**2
```

This gives:

$$
df = 2X \sigma \epsilon + (2X \mu + \sigma^2) \epsilon^2,
$$

or:

$$
df = (2X \mu + \sigma^2) dt + 2X \sigma dB_t,
$$

matching Itô’s lemma, with $$ \sigma^2 dt $$ from $$ dB_t^2 = dt $$.

## Stratonovich Equivalence

Stratonovich uses the midpoint rule for $$ X_t $$. Adjust the increment to $$ dX = \sigma (\epsilon + \frac{1}{2} \epsilon^2) + \mu \epsilon^2 $$. For any smooth $$ f(X) $$, Taylor expand:

$$
f(X + \sigma \epsilon + (\frac{1}{2} \sigma + \mu) \epsilon^2) = f(X) + f'(X) \sigma \epsilon + \left( f'(X) (\frac{1}{2} \sigma + \mu) + \frac{1}{2} f''(X) \sigma^2 \right) \epsilon^2,
$$

so:

$$
df = f'(X) \sigma \epsilon + f'(X) (\frac{1}{2} \sigma + \mu) \epsilon^2 + \frac{1}{2} f''(X) \sigma^2 \epsilon^2.
$$

The $$ \epsilon $$-term is $$ f'(X) \sigma dB_t $$, and the $$ \epsilon^2 $$-term, after averaging, lacks Itô’s extra $$ \frac{1}{2} f''(X) \sigma^2 dt $$, matching Stratonovich’s ordinary chain rule.

## Why It’s Neat

Encoding Itô and Stratonovich in $$ \mathbb{R}[\epsilon]/\epsilon^3 $$—with $$ \epsilon + \frac{1}{2} \epsilon^2 $$ for the latter—makes stochastic calculus algebraic and computer-friendly. A unified tool for theory and practice!

Here is a more complete example:

```py
import math

class DualNumber:
    def __init__(self, real, e=0, e2=0):
        self.real = real  # Represents the standard real part
        self.e = e        # Coefficient for e (Brownian increment)
        self.e2 = e2      # Coefficient for e² (time increment)

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
            e2 = (self.real * other.e2 + self.e * other.e +
                  self.e2 * other.real)
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
        return f"DualNumber({self.real} + {self.e} e + {self.e2} e²)"

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

# Example usage to derive Itô's lemma for f(t, X) = X²
X = 5.0    # Current value of X
mu = 2.0   # Drift coefficient (dX = mu dt + sigma dB)
sigma = 3.0  # Diffusion coefficient

# Represent dX as a dual number: sigma e + mu e²
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

Output
```
Differential df:
df = (29.0) dt + (30.0) dB

Expected from Itô's lemma:
df = (29.0) dt + (30.0) dB
```

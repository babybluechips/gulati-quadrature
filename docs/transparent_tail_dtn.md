# Exact transparent closure of autonomous shell tails

## 1. Problem and scope

Consider a compact boundary discretization attached to a uniform cylindrical
or conic end. After Fourier transformation in the periodic variable, a
nearest-shell discretization of the end has one scalar equation per angular
mode,

\[
-u x_{j-1}+d_k(\lambda)x_j-u x_{j+1}=0,
\qquad j\geq 1,
\]

where `u > 0` is the shell coupling and \(\lambda\) is the PDE resolvent
parameter. This document eliminates that *structured tail*. It does not turn
an unresolved arbitrary CAD surface into a machine-accurate continuum
discretization. In particular, it removes exterior truncation error but leaves
geometry, chart, local quadrature, and transition-region errors unchanged.

The implementation is
[`transparent_tail.py`](../src/inverse_shape/transparent_tail.py). It uses the
project's own radix-two FFT and stores only generated mode symbols.

## 2. Fixed-point Schur closure

Eliminating the last shell of a finite tail updates its first Schur pivot by

\[
\Phi_k(\sigma)=d_k-\frac{u^2}{\sigma}.
\]

An autonomous semi-infinite tail is invariant under removal of its first
shell. Its pivot therefore satisfies

\[
\Sigma_*^2-d_k\Sigma_*+u^2=0.
\]

Write

\[
w_k+w_k^{-1}=\frac{d_k}{u}
\]

and select the branch \(|w_k|<1\). Then

\[
\boxed{\Sigma_*(k,\lambda)=\frac{u}{w_k}}.
\]

Substitution proves the identity directly:

\[
d_k-\frac{u^2}{\Sigma_*}
=u(w_k+w_k^{-1})-uw_k
=\frac{u}{w_k}
=\Sigma_*.
\]

There are four related symbols. They must not be conflated:

| Quantity | Symbol | Use |
|---|---:|---|
| first-tail Schur pivot | \(u/w_k\) | terminal pivot in shell elimination |
| self-energy | \(uw_k=u^2/\Sigma_*\) | term inserted into the retained system |
| interface flux DtN | \(u(1-w_k)\) | flux for trace \(x_0\) |
| logarithmic generator | \(-\log w_k\) | discrete half-Laplacian symbol |

For a boundary trace \(x_0=f_k\), the decaying tail is
\(x_j=w_k^j f_k\). Hence the first edge flux is
\(u(x_0-x_1)=u(1-w_k)f_k\), proving the flux row in the table.

## 3. Exact cross-ratio certificate

The second fixed point is \(\Sigma_-=uw_k\). For a strictly stable mode define

\[
\chi_k(\sigma)
=\frac{\sigma-\Sigma_*}{\sigma-\Sigma_-}.
\]

### Theorem 1 (global linearization)

For every \(\sigma\) away from the pole and the repelling fixed point,

\[
\boxed{\chi_k(\Phi_k(\sigma))=w_k^2\chi_k(\sigma)}.
\]

**Proof.** Since

\[
\Phi_k(\sigma)-\Sigma_\pm
=\frac{(d_k-\Sigma_\pm)\sigma-u^2}{\sigma}
=\frac{u^2(\sigma-\Sigma_\pm)}
        {\Sigma_\pm\sigma},
\]

division of the `+` expression by the `-` expression gives
\(\Sigma_-/\Sigma_*=w_k^2\). No local expansion is used. \(\square\)

Let a Dirichlet-truncated tail contain \(L\) shells. Its terminal pivot is
\(\Sigma_1=d_k\), and
\(\Sigma_L=\Phi_k^{L-1}(d_k)\). Therefore

\[
c_L=w_k^{2(L-1)}\chi_k(d_k),
\qquad
\Sigma_L=\frac{\Sigma_*-c_L\Sigma_-}{1-c_L}.
\]

This yields the exact error identity

\[
\Sigma_L-\Sigma_*
=\frac{c_L(\Sigma_*-\Sigma_-)}{1-c_L}
\]

and the certified bound

\[
|\Sigma_L-\Sigma_*|
\leq
\frac{|c_L|\,|\Sigma_*-\Sigma_-|}{1-|c_L|},
\qquad |c_L|<1.
\]

The code reports both the exact reconstructed value and this bound. At a
double fixed point, such as the Laplace zero mode, \(w=1\) and the map is
parabolic. There the exact finite-depth laws are instead

\[
\Sigma_L=u\frac{L+1}{L},\qquad
|\Sigma_L-u|=\frac{u}{L},\qquad
\Lambda_L=\frac{u}{L+1}.
\]

Thus a universal geometric claim would be false at threshold. The exact cap
still gives the correct limiting flux, zero, without marching through shells.

## 4. Cylinder symbol and continuum limit

For the unit-ratio five-point cylinder,

\[
\frac{d_k}{u}
=2+4\sin^2\left(\frac{\pi k}{N_\theta}\right).
\]

Set \(s_k=|\sin(\pi k/N_\theta)|\). The identity

\[
2+4s_k^2
=2\cosh\bigl(2\operatorname{arsinh}s_k\bigr)
\]

gives

\[
w_k=e^{-2\operatorname{arsinh}s_k},
\qquad
\boxed{
\log\frac{\Sigma_*(k)}{u}
=-\log w_k
=2\operatorname{arsinh}s_k}.
\]

For fixed integer \(k\) and increasing \(N_\theta\),

\[
-\log w_k
=\frac{2\pi|k|}{N_\theta}
+O\!\left(\frac{|k|^3}{N_\theta^3}\right).
\]

After division by the angular mesh spacing, this tends to \(|k|\), the circle
half-Laplacian symbol. This is an exact discrete symbol identity followed by a
standard low-frequency limit; it is not an equality between an arbitrary CAD
graph and its continuum DtN map.

## 5. Spectral parameters

The diagonal used by the implementation is

\[
d_k(\lambda)
=2u+4u\gamma\sin^2(\pi k/N_\theta)+\lambda.
\]

The public factory uses the following dimensionless shifts:

| Problem | Shift \(\lambda\) |
|---|---:|
| Laplace / Poisson tail | \(0\) |
| screened Poisson | \(\kappa^2\) |
| heat resolvent | \(s\), \(\Re s>0\) |
| Helmholtz resolvent | \(-\kappa^2+i\eta\), \(\eta>0\) |
| causal wave resolvent | \((\eta+i\omega)^2\), \(\eta>0\) |

On an undamped propagating band both roots have unit modulus. A unique decaying
branch then does not exist. The API rejects that case and requires a positive
limiting-absorption or Laplace damping. This makes the branch convention part
of the input rather than a hidden numerical choice.

## 6. Residue-class sectors

If a deformed tail coefficient has Fourier support in \(b\mathbb Z\), modal
multiplication changes a Fourier label only by a multiple of \(b\). Hence the
de-aliased space decomposes as

\[
\mathcal H=\bigoplus_{r=0}^{b-1}
\mathcal H_r,
\qquad
\mathcal H_r=\operatorname{span}\{e^{ik\theta}:k\equiv r\pmod b\}.
\]

`residue_class_sectors` generates this partition from signed Fourier labels.
The de-aliasing qualification matters: cyclic wrap on an \(N\)-point FFT does
not preserve \(k\bmod b\) when \(b\nmid N\). The compiler must either pad the
product or restrict it to an alias-safe band before treating the sectors as
exact independent blocks.

## 7. Golden arithmetic checksum

At \(d/u=3\),

\[
w=\varphi^{-2},\qquad
\Sigma_*/u=\varphi^2,\qquad
w^2=\varphi^{-4}.
\]

Starting with a one-shell Dirichlet pivot, the exact rational convergents are

\[
\frac{\Sigma_L}{u}
=\frac{F_{2L+2}}{F_{2L}},
\qquad
\frac{\Sigma_L}{u}-\varphi^2
=\frac{\varphi^{-2L}}{F_{2L}}.
\]

For `L=4`, this gives `55/21`; `55 = F_10`. The integer numerator and
denominator, floating recurrence, fixed point, and closed error law are all
recorded by `golden_tail_certificate`. This is a regression checksum for the
general Riccati implementation, not a separate physical assumption.

## 8. Certified nonautonomous transition

Suppose a finite transition into the autonomous end has
\(d_s=d+e_s\) and \(u_s=u+f_s\). Terminate it with the exact autonomous
pivot \(a_L=\Sigma_*\), then sweep

\[
a_s=d+e_s-\frac{(u+f_s)^2}{a_{s+1}}.
\]

For \(b=\Sigma_*\), direct subtraction gives

\[
|a_s-b|
\leq
\underbrace{\left(
|e_s|+\frac{|(u+f_s)^2-u^2|}{|b|}
\right)}_{\delta_s}
+
\underbrace{\frac{|u+f_s|^2}{|a_{s+1}|\,|b|}}_{L_s}
|a_{s+1}-b|.
\]

Backward composition therefore produces the computable certificate

\[
|a_0-b|
\leq
\delta_0+L_0\delta_1+\cdots+
(L_0\cdots L_{L-2})\delta_{L-1}.
\]

`perturbed_transition_certificate` computes the pivots, every \(L_s\), every
\(\delta_s\), and the propagated bound. This is an exact finite transition
audit. For an infinite summably perturbed end, the same inequality converges
when the tail admits a uniform \(L_s\leq q<1\) and
\(\sum_s|\delta_s|<\infty\). The implementation does not infer those two
hypotheses from arbitrary CAD data.

## 9. Cost

For \(K\) right-hand sides, \(N_\theta\) angular values, and \(L\) explicit
shells:

| Method | Setup | Repeated application | Working storage | Tail error |
|---|---:|---:|---:|---:|
| direct tridiagonal shells | none | \(O(KN_\theta L+KN_\theta\log N_\theta)\) | \(O(N_\theta+L)\) when modes stream | finite-boundary error |
| compiled finite-tail symbol | \(O(N_\theta L)\) | \(O(KN_\theta\log N_\theta)\) | \(O(N_\theta)\) | finite-boundary error |
| fixed-point cap | \(O(N_\theta)\) | \(O(KN_\theta\log N_\theta)\) | \(O(N_\theta)\) | zero for the autonomous tail |

No method in the production implementation stores an
\((N_\theta L)\times(N_\theta L)\) matrix. The cap's gain is removal of `L`
from setup and application, together with exact removal of the autonomous-tail
truncation error.

## 10. Relation to held-out CAD error

The NASA display gallery solves a compressed finite operator on 24 to 42
nodes. Machine-scale residuals there show that retained manufactured channels
and the finite equations are solved accurately. The separate held-out CAD
campaign compiles 48 to 155 vertices, applies a degree-three correction space
to a degree-four continuum harmonic, and reports errors from approximately
`1.1e1` to `2.0e2`. Those quantities answer different questions.

The transparent cap can remove a cylindrical or conic *tail* attached to such
a model. It cannot recover degree-four geometry or operator content discarded
by a coarse CAD atlas. Reaching small held-out continuum error requires
refining the surface representation, increasing the correction space, and
certifying the transition from the irregular core to the autonomous end.

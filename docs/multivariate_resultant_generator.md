# Multivariate resultant generator for peeled jets

## Finite-part identity

For distinct nodes \(X_j\in\mathbb R^3\), put

\[
q_j(x)=\lVert x-X_j\rVert^2,
\qquad
D(x)=\prod_j q_j(x),
\]

and, for arbitrary residues \(a_j\),

\[
N_a(x)=\sum_j a_j\prod_{k\ne j}q_k(x).
\]

Then \(N_a/D=\sum_j a_j/q_j\). The quotient has a quadratic self-pole at each \(X_i\). Write

\[
D=q_iP_i,
\qquad
N_a=a_iP_i+q_iM_{a,i}.
\]

The desired peeled value is

\[
\frac{M_{a,i}(X_i)}{P_i(X_i)}
=\sum_{j\ne i}\frac{a_j}{\lVert X_i-X_j\rVert^2}.
\]

In local coordinates \(h=x-X_i\), \(q_i=\lVert h\rVert^2\). In three dimensions,

\[
\Delta D(X_i)=6P_i(X_i),
\qquad
\Delta^2D(X_i)=20\Delta P_i(X_i),
\]

and

\[
\Delta N_a(X_i)=a_i\Delta P_i(X_i)+6M_{a,i}(X_i).
\]

Eliminating \(P_i\) and \(M_{a,i}\) gives the implemented formula

\[
\boxed{
S_a(i)=
\frac{\Delta N_a(X_i)-a_i\Delta^2D(X_i)/20}
     {\Delta D(X_i)}
}.
\]

Using channels \(a=w\) and \(a=wf\) gives

\[
(Q_2f)_i=f_iS_w(i)-S_{wf}(i).
\]

## Sparse product tree

`MultivariateResultantPeeledJetQJet` builds \(D\), \(N_w\), and \(N_{wf}\) together. If left and right tree bundles are \((D_L,N_{a,L})\) and \((D_R,N_{a,R})\), the merge is

\[
D=D_LD_R,
\qquad
N_a=N_{a,L}D_R+D_LN_{a,R}.
\]

All channels at one node are divided by the same coefficient norm. The unknown common scale cancels from the finite-part quotient. Coefficient convolution and derivative traces use compensated accumulation. Coordinates are translated and scaled before compilation.

The implementation stores sparse monomial dictionaries only. It never stores pair distances or an operator matrix.

## Support and audit protocol

A generic product of \(N\) quadratic factors occupies the three-variable total-degree simplex

\[
\#\{\alpha\in\mathbb N^3:|\alpha|\le2N\}
=\binom{2N+3}{3}=\Theta(N^3).
\]

Therefore a global resultant does not by itself give a subquadratic arbitrary-surface representation. Fast multivariate multipoint evaluation is nearly linear in the coefficient input size, but here that generic coefficient input is already cubic. The available theoretical algorithms are also not practical floating-point replacements for the local QJet FFT. See [Ghosh et al., *Fast Numerical Multivariate Multipoint Evaluation*](https://arxiv.org/abs/2304.01191) and [Bhargava et al., *Fast Multivariate Multipoint Evaluation Over All Finite Fields*](https://arxiv.org/abs/2205.00342).

The production protocol is consequently strict:

1. Stop compilation if any polynomial exceeds `support_budget`.
2. Evaluate the second/fourth derivative finite-part formula.
3. Audit all targets by default against an independent exact peeled sum.
4. If support or numerical audit fails, discard every polynomial and repay by an exact matrix-free pair stream.

`audit_mode="sampled"` and `audit_mode="none"` are available for experiments, but only `full` performs the default all-node audit.

## Measured boundary

On the deterministic 3D test family from `N=4` through `N=16`:

- denominator support fits exponent `2.826`;
- resultant runtime fits exponent `4.530`;
- streamed direct runtime fits exponent `2.132` over this small range;
- the peeled audit error grows from `1.36e-15` at `N=4` to `1.90e-10` at `N=16`;
- the strict `5e-13` production audit retains the resultant through `N=8` and repays larger tested cases.

The formula is useful for small configurations, exact symbolic inputs, and geometries whose quotient relation keeps support sparse. It does not establish an unconditional fast solver on generic two-dimensional surfaces. Such a solver still requires either a nested Riesz/FMM generator or quotient-ideal structure strong enough to reduce the coordinate ring to near-linear dimension.

```sh
PYTHONPATH=src python3 scripts/multivariate_resultant_benchmark.py
```

See the [generated benchmark report](../outputs/multivariate_resultant/report.md).

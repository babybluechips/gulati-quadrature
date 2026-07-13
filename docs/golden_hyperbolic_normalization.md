# Golden hyperbolic normalization

## Result

The golden point supplies a canonical gauge for scale-phase charts. It does
not make every Euclidean chord kernel translation invariant. Its useful role
is to remove the Möbius, scale, and branch freedom before the QJet operator is
compiled.

Write an axisymmetric meridian point as

\[
w=z+ir\in\mathbb H,\qquad r>0.
\]

Choose a point and oriented tangent. There is a unique oriented
\(g\in\mathrm{PSL}_2(\mathbb R)\) which sends that frame to the point \(i\)
with positive vertical tangent. Define

\[
\boxed{
\tau(w)=\sqrt5\,\frac{g(w)-i}{g(w)+i}.
}
\]

This is a biholomorphic map

\[
\mathbb H\longrightarrow\{\tau:|\tau|<\sqrt5\}.
\]

It is rational and therefore has no logarithm branch. The equivalent strip
coordinate is

\[
\xi=\Log(-ig(w)),\qquad
\tau=\sqrt5\tanh\frac{\xi}{2},
\]

where \(-ig(w)\) is always in the right half-plane. Hence this logarithm is
single-valued on the whole chart.

## Why the golden point is optimal

Let \(m\ge3\) be an integer trace and put

\[
\mu_m=\operatorname{arcosh}\frac m2.
\]

Normalize the two geodesic points at signed distance \(\pm\mu_m\) to
\(\pm1\). The scaled Cayley disk then has radius

\[
R_m=\coth\frac{\mu_m}{2}
=\sqrt{\frac{m+2}{m-2}}.
\]

This is strictly decreasing for integer \(m\ge3\). Therefore the first
hyperbolic integer trace, \(m=3\), uniquely maximizes the analytic disk around
the computational interval:

\[
\mu_3=2\log\phi,\qquad R_3=\sqrt5.
\]

The largest Bernstein ellipse contained in that disk has parameter

\[
\varrho_3=R_3+\sqrt{R_3^2-1}
=\sqrt5+2=\phi^3.
\]

Thus the coordinate contribution to a degree-\(p\) analytic interpolation
has geometric factor

\[
\varrho_3^{-p}=\phi^{-3p}.
\]

At \(p=24\), this is `8.97e-16`. The measured relative interpolation error
for the extremal geodesic map is `6.63e-15`; order 28 is at floating-point
roundoff. Among integer-trace gauges, trace three needs the fewest modes.

This is the precise numerical optimality claim. It concerns the guaranteed
analytic collar of the normalized coordinate. Additional singularities of the
physical geometry can reduce the usable collar and must still be certified.

## Exact metric formulas

The upper-half-plane metric is preserved exactly. In golden coordinates,

\[
\cosh d(\tau_i,\tau_j)
=1+\frac{10|\tau_i-\tau_j|^2}
{(5-|\tau_i|^2)(5-|\tau_j|^2)}.
\]

The pseudohyperbolic distance and mode decay are

\[
\delta_{ij}
=\left|
\frac{\sqrt5(\tau_i-\tau_j)}
{5-\overline{\tau_i}\tau_j}
\right|,qquad
e^{-d_{ij}}=\frac{1-\delta_{ij}}{1+\delta_{ij}}.
\]

No `acosh`, logarithm, or matrix of pair distances is needed.

On the normalized geodesic,

\[
\xi=\pm2\log\phi\quad\Longleftrightarrow\quad\tau=\pm1.
\]

The decay from the center to either endpoint is

\[
e^{-2\log\phi}=\phi^{-2},
\]

and its inverse-square quotient ratio is \(\phi^{-4}\). These are exactly the
radial and Fourier constants already used by the golden Joukowski compiler.

## Tetration normalization

The golden tetration fixed-point multiplier used here is

\[
\mu_T=\frac1{2\phi}.
\]

It fixes the same radial scale through

\[
\phi^{-2}=4\mu_T^2.
\]

This identity selects the gauge. The geometric map itself is the branch-free
PSL(2,R)/Cayley map above. It is not raw complex tetration, and the
trace-three matrix flow must not be identified with the tetration flow. This
distinction agrees with the scope warning in the golden-flow source.

## Three-jet closure

The normalization itself has a closed holomorphic three-jet calculus. Write

\[
F(w)=\frac{aw+b}{cw+d},\qquad \Delta=ad-bc.
\]

Then

\[
F'(w)=\frac{\Delta}{(cw+d)^2},\qquad
F''(w)=\frac{-2c\Delta}{(cw+d)^3},\qquad
F'''(w)=\frac{6c^2\Delta}{(cw+d)^4}.
\]

For an arbitrary analytic source jet \((w,w',w'',w''')\),

\[
(F\circ w)'=F'w',
\]

\[
(F\circ w)''=F''(w')^2+F'w'',
\]

\[
(F\circ w)'''=F'''(w')^3+3F''w'w''+F'w'''.
\]

`GoldenHyperbolicFrame` stores four complex Mobius coefficients and applies
these formulas in `transform_jet` and `inverse_transform_jet`. This is the
universal local part: any supplied analytic boundary chart or isothermal
surface chart uses the same four-coefficient normalization without finite
differences or remeshing.

For a meridian \((r(u),z(u))\), hyperbolic speed is

\[
v=\frac{\sqrt{r'^2+z'^2}}r.
\]

The jets \((r,r',r'',r''')\) and \((z,z',z'',z''')\) determine
\((v,v',v'')\). Therefore they determine the complete third-order chain rule
from \(u\) to hyperbolic arclength \(s\):

\[
f_s=\frac{f'}v,
\]

\[
f_{ss}=\frac{f''}{v^2}-\frac{f'v'}{v^3},
\]

\[
f_{sss}
=\frac{f'''}{v^3}
-\frac{3f''v'}{v^4}
-\frac{f'v''}{v^4}
+\frac{3f'(v')^2}{v^5}.
\]

The golden real coordinate on a patch is

\[
t=\sqrt5\tanh\frac{s-s_c}{2}.
\]

Applying the same chain rule once more yields radius and height three-jets in
\(t\). No finite differencing or per-solve remeshing is used.

`GoldenHyperbolicJetAtlas` integrates the hyperbolic speed once during
compilation, partitions the meridian into spans no longer than

\[
2\mu_3=4\log\phi,
\]

and stores a degree-seven endpoint-Hermite generator on each patch. A patch
then generates uniform computational nodes directly in its golden interval.

## Production complexity contract

Production axisymmetric and scale-phase objects do not contain quadratic
reference methods. Independent streamed pair sums used by tests are in
`inverse_shape.testing.reference_pairwise`, which production modules never
import.

For fixed interpolation order \(p\), a compiled chart applies in

\[
O(p^2N+N\log n_\theta)
\]

time and uses \(O(p^2N)\) persistent storage. Compilation enforces this
structure: at most `64N` pair-recursion visits, `16N` static block records,
and `64N` exact local pairs are allowed per meridional mode. Exceeding a
budget raises a chart-subdivision error. There is no quadratic repayment or
dense fallback in a production object.

## Numerical audit

The reproducible campaign gives:

| audit | result |
|---|---:|
| frame inverse, seven shape families | `<=5.47e-16` |
| hyperbolic distance covariance | `<=2.22e-16` |
| strip/disk formula agreement | `<=4.44e-16` |
| long geodesic atlas Q error | `4.06e-16` |
| polynomial meridian atlas Q error | `9.92e-16` |
| corrugated meridian atlas Q error | `1.65e-14` |

The sampling experiment separates normalization from relabeling. With a
`963:1` hyperbolic node-spacing ratio, applying new coordinates to the old
nodes remains near `1.24e-12`. Generating nodes uniformly from the golden
chart reduces the spacing ratio to `1.245` and restores `6.69e-15` error.
Uniform hyperbolic arclength gives `4.97e-15`. The golden chart has comparable
accuracy while additionally fixing the interval and analytic collar
algebraically.

Run:

```sh
PYTHONPATH=src python3 scripts/golden_hyperbolic_benchmark.py
```

Machine-readable tables are under `outputs/golden_hyperbolic/`.

## Scope

The construction is universal for oriented frames in the hyperbolic upper
half-plane and gives a finite atlas for every regular positive-radius
meridian of finite hyperbolic length. A cusp or axis crossing is a separate
Mellin/Joukowski endpoint chart.

For a general two-dimensional surface in three dimensions, local isothermal
coordinates admit the same disk normalization for the DtN principal symbol.
Computing those coordinates and repaying the extrinsic layer-potential
remainder are separate operations. The golden gauge does not make an
arbitrary extrinsic chord kernel globally convolutional.

# Scale-Phase Pullback Demo

This is a visual demonstration of the scale-phase pullback used in the Cauchy/Q calculus.  It uses a funky closed piecewise-cubic boundary to make the geometry visibly non-conic.

Important limitation: this demo does not solve the exterior Riemann map.  It shows the mechanism and cost ledger.  In production, the map `Phi: exterior(Gamma) -> exterior(D)` is computed once per geometry, then reused for all targets.

## Complex Coordinates

The exterior disk coordinate is

```text
w = Phi(z) = exp(rho + i theta) = exp(rho) exp(i theta).
```

The two coordinates have separate jobs:

```text
rho   = log |w|      scale / distance from boundary in conformal units
theta = arg w        phase / boundary correspondence
```

De Moivre supplies the Fourier characters:

```text
(exp(i theta))^k = exp(i k theta).
```

The exterior harmonic continuation of a boundary mode is the same phase with real damping:

```text
exp(i k theta) -> exp(-|k| rho) exp(i k theta).
```

So the target enters through only two scalars, `rho` and `theta`, after the one-time pullback map has been built.

## Modal Evaluation

For a single-layer density with Fourier coefficients `sigma_hat[k]`, the circle-model exterior evaluation is

```text
I(w) = -pi sum_{k != 0} sigma_hat[k] exp(-|k| rho) exp(i k theta) / |k|
       + 2 pi sigma_hat[0] rho.
```

This is the diagonal scale flow `T_rho = exp(-rho |D|)` plus a phase character.  Scale does not rotate phase; phase does not change scale.

## Borrow-Compute-Repay Cost Ledger

| Stage | One-time per geometry | Per target | Scaling note |
|---|---|---|---|
| map theta<->s | iter x O(n log n) | - | once |
| phase resample | O(n log n) | - | once |
| density FFT | O(n log n) / density | - | amortized across targets |
| scale flow | - | O(n) diagonal | target only touches rho |
| phase synthesis | - | O(n) modal sum | target only touches theta |

Measured toy timing on this demo:

```text
boundary samples              256
modes each side               48
target count                  100000
arclength resample            6.879 ms
density FFT                   0.106 ms
modal target evaluation       28.551 ms
microseconds per target       0.285510
```

The Riemann-map solve is intentionally not hidden in this timing.  It belongs in the one-time geometry column.

## Pseudocode

```text
precompute_geometry(Gamma):
  z_j = arclength_sample(Gamma, n)
  theta_j = FourierNewtonExteriorMap(Gamma)  # one-time, not per target
  R = resample_density_to_phase_grid(theta_j)
  return PullbackQJet(z_j, theta_j, R)

precompute_density(sigma):
  sigma_hat = FFT(R sigma)
  return sigma_hat

evaluate_target(x, sigma_hat):
  w = Phi(x)
  rho = log(abs(w)); theta = arg(w)
  return sum_{k != 0} -pi*sigma_hat[k]*exp(-abs(k)*rho)*exp(i*k*theta)/abs(k)
```

## Files

- Figure: `scale_phase_pullback_demo.png`
- JSON payload: `scale_phase_pullback_demo.json`

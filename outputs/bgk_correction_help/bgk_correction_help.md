# BGK Correction Help

This is the concrete place where the BGK continuity correction helps the attached DtN/harmonic notes.

![BGK correction help](/Users/rick/Documents/New project 2/outputs/bgk_correction_help/bgk_correction_help.png)

## What Is Being Corrected

`dtn_bgk_merge (9).pdf` identifies the rescaled cycle DtN spectrum

```text
mu[n,k] = k (1 - k/n)
```

and says that the half-integer endpoint term of its spectral zeta sum is the BGK constant. `harmonic_sp (41).pdf` says the same thing in harmonic-language: when a continuum harmonic object is sampled on a finite cyclic grid, the residual is a zeta-coded sampling defect.

The correction does not replace Q. It repays the finite lattice after the principal DtN/Q symbol has already been borrowed:

```text
phase:       de Moivre/Fourier modes
operator:    pi |D| or the generated chord-QJet
metric:      pullback Jacobian J(theta)
sampling:    BGK/Hurwitz-zeta endpoint repayment
```

## Endpoint Model

The square-root endpoint channel is the local model behind the BGK rung:

```text
h sum_{j=1}^n (j h)^(-1/2) - integral_0^1 x^(-1/2) dx
  = zeta(1/2) h^(1/2) + O(h).
```

Since `zeta(1/2) = -1.4603545088095868`, the BGK barrier constant is

```text
beta_BGK = -zeta(1/2) / sqrt(2 pi)
         = 0.5825971579390108.
```

In this run the raw endpoint error fits power `0.496` in `h`; after subtracting the BGK rung it fits power `1.000`. At `n=262144`, the raw error is `0.002850` and the corrected error is `1.907e-06`.

## DtN Spectral Zeta Model

For the cycle DtN spectrum at `s=1/2`, the bulk integral is `pi sqrt(n)`. The endpoint residual is

```text
S_n(1/2) - pi sqrt(n) -> 2 zeta(1/2).
```

So the BGK-corrected bulk approximation is

```text
S_n(1/2) ~= pi sqrt(n) + 2 zeta(1/2).
```

At `n=131072`, the bulk-only endpoint residual is `-2.920711`. After adding `2 zeta(1/2)`, the residual drops to `-1.586e-06`. The corrected residual over the last five levels fits power `1.000` in `1/n`.

## Practical Meaning For Q

BGK helps when the Q/DtN computation has the right principal operator but the finite boundary sampling still sees a square-root endpoint or survival/barrier channel. It removes the leading half-integer sampling debt. For polygons and sharper corners, the same slot is filled by the Kondrat'ev/Hurwitz rule `zeta(1-lambda,beta) h^lambda`; BGK is the crack/square-root special case `lambda=1/2`.

## Artifacts

- endpoint CSV: `/Users/rick/Documents/New project 2/outputs/bgk_correction_help/endpoint_bgk_correction.csv`
- DtN CSV: `/Users/rick/Documents/New project 2/outputs/bgk_correction_help/dtn_spectral_bgk_correction.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/bgk_correction_help/bgk_correction_help.json`
- figure: `/Users/rick/Documents/New project 2/outputs/bgk_correction_help/bgk_correction_help.png`

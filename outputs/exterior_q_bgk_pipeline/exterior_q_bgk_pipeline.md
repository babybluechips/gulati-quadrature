# Exterior Q + BGK Correction Pipeline

This combines the slick exterior-map Q path with the BGK continuity ledger.

![exterior Q BGK pipeline](/Users/rick/Documents/New project 2/outputs/exterior_q_bgk_pipeline/exterior_q_bgk_pipeline.png)

## Pipeline

```text
physical boundary Gamma
  -> borrow exterior circle coordinate w via psi
  -> compute log singularity by Q_{S1}^{-1} Fourier weights 1/k
  -> repay analytic quotient log|G(w,z)| by trapezoidal
  -> repay discrete monitoring endpoint beta_BGK sqrt(h) by radial Q derivative ladder
```

`beta_BGK = 0.5825971579390108`.

The test constructs a raw discrete-monitoring target at `rho + beta sqrt(h)`, then applies the zeta/Taylor BGK repayment

```text
I_p = sum_{j=0}^p (-beta sqrt(h))^j / j! * d_rho^j I(rho + beta sqrt(h)).
```

If the correction is load-bearing, raw error should scale like `h^1/2`, first-order BGK like `h`, second-order like `h^3/2`, and higher orders should descend to the quadrature floor.

## Scaling Summary

| Shape | raw exponent | BGK-1 exponent | BGK-2 exponent | raw err | BGK-1 err | BGK-2 err | BGK-4 err | BGK-8 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | `0.500` | `1.000` | `1.500` | `1.701e-03` | `6.067e-08` | `2.092e-12` | `1.345e-15` | `1.345e-15` |
| golden_ellipse | `0.500` | `1.000` | `1.508` | `8.176e-05` | `3.055e-09` | `1.230e-13` | `4.712e-15` | `4.712e-15` |
| three_petal_laurent | `0.500` | `1.000` | `1.499` | `9.976e-04` | `5.243e-08` | `3.840e-12` | `2.607e-15` | `2.607e-15` |
| perturbed_ellipse | `0.500` | `1.000` | `1.500` | `5.878e-04` | `2.722e-08` | `1.794e-12` | `4.261e-16` | `4.261e-16` |

## Interpretation

The exterior Q split removes the near-singular quadrature problem. The BGK layer then removes the discrete-monitoring endpoint error as a local bridge expansion. In this benchmark the uncorrected endpoint displacement has the expected square-root law, first-order BGK has the expected first-order law in `h`, second-order BGK has the expected `h^3/2` law, and order 4-8 reaches the fp64 quadrature floor on the exact Laurent-map cases.

This is still the analytic-boundary path. Polygon corners need the same outer structure plus the corner continuity/Kondrat'ev ledger because the exterior map loses analytic regularity at corner preimages.

## Artifacts

- Rows CSV: `/Users/rick/Documents/New project 2/outputs/exterior_q_bgk_pipeline/exterior_q_bgk_pipeline_rows.csv`
- Summary CSV: `/Users/rick/Documents/New project 2/outputs/exterior_q_bgk_pipeline/exterior_q_bgk_pipeline_summary.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/exterior_q_bgk_pipeline/exterior_q_bgk_pipeline.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/exterior_q_bgk_pipeline/exterior_q_bgk_pipeline.png`

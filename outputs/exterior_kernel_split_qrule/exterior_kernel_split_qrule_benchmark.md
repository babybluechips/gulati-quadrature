# Exterior Kernel-Splitting Q Rule Benchmark

This tries the method from `optimal_quadrature_v63_public`: pull the layer-potential kernel to the exterior circle, evaluate the singular circle term by the Q/Fourier primitive, and integrate only the analytic correction by trapezoidal rule.

![exterior kernel split benchmark](/Users/rick/Documents/New project 2/outputs/exterior_kernel_split_qrule/exterior_kernel_split_qrule_benchmark.png)

## Why It Gets Machine Precision

For a known exterior Laurent map `psi(w)`, the kernel is split as

```text
log|psi(w_x)-psi(e^{i theta})| = log|w_x-e^{i theta}| + log|G(w_x,e^{i theta})|.
```

The first term is the universal circle singularity. Its Fourier series has weights `1/k`, exactly the inverse spectral weight of `Q_{S1} = pi |D|`. The second term is analytic and non-singular, so trapezoidal convergence is governed by the conformal collar, not by target distance.

This benchmark uses finite Laurent maps, so the exterior map is exact and no Riemann-map solver error is hidden in the result. No dense Q matrix is built.

## n=512 Headline

| Shape | offset | direct trap err | spectral Q split err |
|---|---:|---:|---:|
| circle | `1e-01` | `7.042e-15` | `1.531e-16` |
| circle | `1e-03` | `3.980e-02` | `<1e-16` |
| circle | `1e-06` | `6.555e-02` | `<1e-16` |
| circle | `1e-08` | `6.554e-02` | `1.877e-16` |
| golden_ellipse | `1e-01` | `1.867e-15` | `3.111e-15` |
| golden_ellipse | `1e-03` | `1.919e-03` | `2.069e-15` |
| golden_ellipse | `1e-06` | `3.239e-03` | `1.151e-15` |
| golden_ellipse | `1e-08` | `3.239e-03` | `1.231e-14` |
| three_petal_laurent | `1e-01` | `1.083e-14` | `2.041e-15` |
| three_petal_laurent | `1e-03` | `2.582e-02` | `3.352e-15` |
| three_petal_laurent | `1e-06` | `4.294e-02` | `1.286e-15` |
| three_petal_laurent | `1e-08` | `4.293e-02` | `3.673e-15` |
| perturbed_ellipse | `1e-01` | `6.259e-15` | `7.944e-15` |
| perturbed_ellipse | `1e-03` | `1.471e-02` | `2.130e-16` |
| perturbed_ellipse | `1e-06` | `2.501e-02` | `8.378e-15` |
| perturbed_ellipse | `1e-08` | `2.501e-02` | `4.941e-15` |

## Caveat

This is the slick regime: analytic boundary plus accurate exterior map. For true polygon corners the same paper predicts algebraic degradation because the conformal map is only Holder at corner preimages. That is exactly where the corner continuity/Kondrat'ev ledger remains necessary.

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/exterior_kernel_split_qrule/exterior_kernel_split_qrule_benchmark.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/exterior_kernel_split_qrule/exterior_kernel_split_qrule_benchmark.json`
- figure: `/Users/rick/Documents/New project 2/outputs/exterior_kernel_split_qrule/exterior_kernel_split_qrule_benchmark.png`

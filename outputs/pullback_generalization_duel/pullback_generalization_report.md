# Riemann versus cosecant pullback generalization

## Verdict

The cosecant route is the viable principal-channel generalization. It is exact on the circle, removes the universal inverse-square diagonal singularity on every regular closed curve, and survives azimuthal reduction of the three-dimensional inverse-cube kernel. Sparse local repayment improves it without storing a dense matrix.

The Riemann manuscript does not provide an arbitrary-3D fast path. Its boundary is a one-dimensional loop in a two-dimensional Riemann surface, whereas the 3D volume DtN operator acts on a two-dimensional boundary. In addition, a generic arclength-parametrized prime-form or chord kernel depends on both boundary points, not only their lag. The lag-averaged experiment below is the best circulant projection and still leaves a nonzero geometry residual.

## Structural checks

- planar action cases: `54`
- axisymmetric 3D cases: `9`
- circle cosecant maximum action error: `5.944e-14`
- circle best-lag maximum action error: `1.387e-13`
- smallest noncircle best-lag far-kernel residual: `4.323e-03`
- largest off-circle prime-form/chord far defect: `3.749e-01`
- smooth high-mode median cosecant error: `1.596e-02`
- after three exact local repayment bands: `1.327e-02`
- finest measured logarithmic coefficient: `0.24996685` (target `0.25`)
- dense matrices or pair tables stored: `no`

## Planar high-mode comparison at n=256

| shape | regularity | cosecant | cosecant + 3 bands | best lag-only | prime/chord far defect |
|---|---|---:|---:|---:|---:|
| circle | smooth | `1.373e-14` | `7.877e-15` | `1.117e-14` | `1.502e-16` |
| golden_ellipse | smooth | `4.634e-04` | `4.587e-04` | `5.211e-07` | `1.294e-03` |
| starfish_5 | smooth | `2.218e-02` | `2.053e-02` | `5.274e-03` | `6.713e-02` |
| asymmetric_fourier | smooth | `7.888e-03` | `7.207e-03` | `3.778e-03` | `2.608e-02` |
| peanut | smooth_concave | `4.726e-03` | `4.565e-03` | `2.094e-03` | `1.332e-02` |
| double_concave | smooth_concave | `1.136e-02` | `1.027e-02` | `4.206e-03` | `3.830e-02` |
| rounded_square | low_regular | `1.613e-03` | `1.525e-03` | `9.868e-05` | `4.870e-03` |
| cardioid_cusp | cusp | `6.375e-01` | `1.860e-02` | `6.363e-01` | `3.598e-01` |
| square | corners | `8.281e-02` | `1.631e-02` | `7.809e-02` | `1.345e-01` |

## Axisymmetric 3D meridional reduction

For a periodic meridian,

```text
r' integral |X-X'|^-3 dtheta' ~ 2 / |s-s'|^2.
```

The subtraction leaves a logarithmic local channel. For a cylinder of radius R,

```text
A_R(h) - 2/h^2 = log(8R/|h|)/(4R^2) - 3/(8R^2) + O(h^2 log|h|).
```

This produces the same cosecant principal symbol after periodization. The table compares that FFT operator with the full streamed ring-pair calculation for azimuthal mode zero.
The meridional probe is the refinement-relative high mode `n_s/4-1`; this is a principal-symbol stress test, not a fixed-mode convergence table.

| geometry | n_s | mode | nodes | cosecant | + 3 bands | first solve | repeated apply |
|---|---:|---:|---:|---:|---:|---:|---:|
| slender_torus | 32 | 7 | 4096 | `3.709e-01` | `1.898e-03` | `2.7x` | `1046.5x` |
| slender_torus | 64 | 15 | 8192 | `6.540e-01` | `1.649e-03` | `5.3x` | `2372.1x` |
| slender_torus | 128 | 31 | 16384 | `8.239e-01` | `6.241e-03` | `10.8x` | `4987.5x` |
| moderate_torus | 32 | 7 | 4096 | `2.215e-02` | `1.178e-02` | `2.7x` | `1051.7x` |
| moderate_torus | 64 | 15 | 8192 | `1.546e-01` | `5.898e-03` | `5.3x` | `2376.3x` |
| moderate_torus | 128 | 31 | 16384 | `4.473e-01` | `2.017e-03` | `10.4x` | `4439.9x` |
| near_horn_torus | 32 | 7 | 4096 | `4.361e-02` | `2.624e-02` | `2.7x` | `1048.0x` |
| near_horn_torus | 64 | 15 | 8192 | `3.275e-02` | `1.540e-02` | `5.5x` | `2360.8x` |
| near_horn_torus | 128 | 31 | 16384 | `2.016e-01` | `7.152e-03` | `10.7x` | `4162.5x` |

## Complexity

| route | setup | apply | retained storage | status |
|---|---:|---:|---:|---|
| global Riemann/lag-only proxy | `O(n^2)` | `O(n log n)` | `O(n)` | not exact off homogeneous loops |
| cosecant principal | `O(n)` | `O(n log n)` | `O(n)` | exact singular channel |
| cosecant + b local bands | `O(bn)` | `O(n log n + bn)` | `O(n + bn)` | implemented |
| complete arbitrary geometry | hierarchical | target `O(n log n)` | target `O(n)` | log-local and far certificates required |

## Consequence

Use the cosecant pullback as the borrowed principal operator. Repay the diagonal and corner/cusp and logarithmic reduced channels with local product integration. Compute the nonlocal geometry with a certified multipole or H-matrix layer. For non-axisymmetric 3D surfaces, the corresponding local normal form is the two-dimensional periodic Riesz kernel with symbol |xi|, not a one-dimensional cosecant kernel.

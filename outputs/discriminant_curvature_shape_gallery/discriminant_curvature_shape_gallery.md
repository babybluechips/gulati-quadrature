# Discriminant Curvature Shape Gallery

This gallery shows the same discriminant-curvature mechanism on smooth, conic, nonconvex, cusped, polygonal, airfoil, stealth-like, and rational/asymptotic shape charts.

![shape gallery](/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/discriminant_curvature_shape_gallery.png)

## Rule

Every panel starts from generated complex boundary samples `z_j`. The plotted scalar is

```text
q_j = sum_{k != j} |z_j - z_k|^(-2).
```

This is the diagonal discriminant curvature of Q. It is computed by the borrow-compute-repay chord protocol without storing the dense Q matrix.

## What To Read

- Smooth shapes show distributed curvature modulation.
- Cusp and airfoil charts show sharp spikes where the tangent from `psi(exp(i theta))` vanishes.
- Polygons show vertex channels where the tangent jumps.
- The rational chart shows the quotient/asymptote carrier with the residual touch/cross points still visible through chord curvature.

## Strongest Hard-Part Contrasts

| shape | family | max/median | q90/q10 | hard part |
|---|---|---:|---:|---|
| joukowski_airfoil | Joukowski cusp | `9.870e+08` | `2.924e+01` | critical-point cusp from zeta+a^2/zeta |
| cardioid_one_cusp | cusp | `1.115e+08` | `4.477e+01` | one vanishing tangent; one tangent cone |
| nephroid_two_cusps | multi-cusp | `1.526e+07` | `6.589e+01` | two vanishing tangents |
| astroid_four_cusps | multi-cusp | `1.336e+07` | `3.027e+02` | four tangent-cone endpoints |
| double_concave_stealth | concave polygon | `3.062e+00` | `1.126e+00` | symmetric concave vertex scattering |
| flower_nonconvex | smooth nonconvex | `2.690e+00` | `2.625e+00` | oscillatory metric/chord modulation |
| rounded_square | smooth high-curvature | `1.644e+00` | `7.518e+00` | large but finite curvature bands |
| star_polygon | re-entrant polygon | `1.607e+00` | `1.194e+00` | alternating convex/re-entrant corner channel |
| golden_ellipse | smooth conic | `1.388e+00` | `1.724e+00` | metric modulation from closed-form Laurent pullback |
| square_polygon | polygonal corners | `1.119e+00` | `1.034e+00` | four tangent jumps / Kondrat'ev vertices |
| rational_asymptote_chart | open rational chart | `1.006e+00` | `1.029e+00` | Q asymptote plus R/D touch-cross residual |
| circle | smooth normal form | `1.000e+00` | `1.000e+00` | none; constant chord-curvature density |

## Artifacts

- summary CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/discriminant_curvature_shape_gallery_summary.csv`
- samples CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/discriminant_curvature_shape_gallery_samples.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/discriminant_curvature_shape_gallery.json`
- figure: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/discriminant_curvature_shape_gallery.png`

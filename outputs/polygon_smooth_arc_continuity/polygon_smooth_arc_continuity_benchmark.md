# Polygon Smooth-Arc Continuity Benchmark

This tests the polygonal quadrature ledger requested here: represent the polygon by smooth arcs for the principal calculation, then repay the continuity/corner defect locally.

![polygon smooth arc benchmark](/Users/rick/Documents/New project 2/outputs/polygon_smooth_arc_continuity/polygon_smooth_arc_continuity_benchmark.png)

## Protocol

For each polygon vertex `v_i`, the benchmark cuts back adjacent edges and inserts a C1 quadratic tangent arc. The smooth model is then integrated as line-plus-arc primitives. The continuity correction adds back

```text
exact incoming edge stub + exact outgoing edge stub - smooth corner arc.
```

That is the polygon version of the BGK bookkeeping: compute on the continuous/smooth model, then repay the discrete boundary crossing/corner defect. The ledger also records the Kondrat'ev exponent `lambda = pi / omega` and the Hurwitz endpoint factor `zeta(1-lambda, 1/2)` at each corner.

The calculation stores no dense Q matrix here; the boundary is carried by primitive generators plus a local corner ledger.

`beta_BGK = 0.5825971579390108` is the square-root endpoint special case.

## Latest Near-Corner Results

| Shape | Raw polygon err | Smooth arc err | Corrected err | Smooth/corrected | Raw/corrected | Continuity delta |
|---|---:|---:|---:|---:|---:|---:|
| square | `<1e-15` | `3.858e-02` | `<1e-15` | `>=3.86e+13x` | `floor` | `1.559e-01` |
| l_notch | `<1e-15` | `2.793e-02` | `<1e-15` | `>=2.79e+13x` | `floor` | `1.376e-01` |
| star | `1.202e-15` | `3.816e-02` | `1.322e-15` | `28860737408724.00x` | `0.91x` | `7.049e-02` |
| stealth_double_concave | `6.093e-15` | `7.252e-02` | `1.259e-14` | `5758456607776.42x` | `0.48x` | `-7.928e-02` |

## Near-Corner Convergence

| Shape | Panels | Smooth arc err | Corrected err |
|---|---:|---:|---:|
| square | `24` | `3.858e-02` | `6.373e-15` |
| square | `48` | `3.858e-02` | `<1e-15` |
| square | `96` | `3.858e-02` | `<1e-15` |
| square | `192` | `3.858e-02` | `<1e-15` |
| l_notch | `24` | `2.793e-02` | `5.245e-09` |
| l_notch | `48` | `2.793e-02` | `1.087e-09` |
| l_notch | `96` | `2.793e-02` | `<1e-15` |
| l_notch | `192` | `2.793e-02` | `<1e-15` |
| star | `24` | `3.816e-02` | `3.293e-06` |
| star | `48` | `3.816e-02` | `2.851e-06` |
| star | `96` | `3.816e-02` | `4.612e-11` |
| star | `192` | `3.816e-02` | `1.322e-15` |
| stealth_double_concave | `24` | `7.230e-02` | `2.179e-04` |
| stealth_double_concave | `48` | `7.252e-02` | `4.953e-08` |
| stealth_double_concave | `96` | `7.252e-02` | `1.312e-10` |
| stealth_double_concave | `192` | `7.252e-02` | `1.259e-14` |

## What This Shows

- The smooth-arc representation is useful as a principal continuous model, but it changes the local corner ledger.
- The continuity correction is local and geometry-derived; it replaces the rounded corner contribution with the exact two-edge corner contribution.
- The same bookkeeping slot carries the BGK square-root endpoint constant for barriers and the Kondrat'ev/Hurwitz exponent for polygon corners.

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/polygon_smooth_arc_continuity/polygon_smooth_arc_continuity_benchmark.csv`
- corner ledger CSV: `/Users/rick/Documents/New project 2/outputs/polygon_smooth_arc_continuity/polygon_smooth_arc_corner_ledger.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/polygon_smooth_arc_continuity/polygon_smooth_arc_continuity_benchmark.json`
- figure: `/Users/rick/Documents/New project 2/outputs/polygon_smooth_arc_continuity/polygon_smooth_arc_continuity_benchmark.png`

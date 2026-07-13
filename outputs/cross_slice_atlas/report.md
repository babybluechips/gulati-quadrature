# Static curved cross-slice atlas benchmark

The local singular channel is the exact complete-slice Joukowski/cycle operator. Adjacent cross slices use phase-difference/modulation FFT charts. Separated product patches use symmetric adaptive-cross factors, and terminal failures are streamed exactly. No distance or operator matrix is stored.

## Accuracy against the independent streamed reference

| shape | p | nodes | atlas rel. error | old quadrupole rel. error | warm ms | direct ms |
|---|---:|---:|---:|---:|---:|---:|
| circular_cylinder | 2 | 192 | 5.966e-16 | 6.507e-04 | 10.707 | 7.729 |
| circular_cylinder | 3 | 192 | 7.771e-16 | 1.344e-03 | 9.716 | 7.965 |
| elliptic_taper | 2 | 192 | 4.738e-16 | 7.828e-04 | 20.193 | 7.835 |
| elliptic_taper | 3 | 192 | 4.853e-16 | 9.754e-04 | 10.166 | 8.026 |
| bent_tube | 2 | 192 | 5.190e-16 | 1.046e-03 | 19.601 | 7.778 |
| bent_tube | 3 | 192 | 4.221e-16 | 1.157e-03 | 10.378 | 7.721 |
| twisted_ellipse | 2 | 192 | 4.858e-16 | 8.365e-04 | 28.434 | 7.900 |
| twisted_ellipse | 3 | 192 | 4.194e-16 | 1.146e-03 | 10.386 | 7.900 |
| toroidal_bundle | 2 | 192 | 7.779e-14 | 1.406e-03 | 21.575 | 7.880 |
| toroidal_bundle | 3 | 192 | 3.698e-14 | 2.333e-03 | 10.018 | 7.522 |
| aircraft_body | 2 | 192 | 1.779e-15 | 1.036e-03 | 24.959 | 7.888 |
| aircraft_body | 3 | 192 | 1.080e-15 | 1.154e-03 | 10.600 | 7.973 |

## Scaling and storage

| nodes | compile ms | warm ms | direct ms | rel. error | low-rank pair fraction | exact residual pair fraction | factor entries | N^2 entries avoided |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 43.352 | 15.963 | 3.829 | 1.227e-15 | 0.000 | 0.750 | 0 | 16384 |
| 256 | 278.437 | 42.213 | 13.966 | 7.278e-16 | 0.000 | 0.875 | 0 | 65536 |
| 512 | 1068.079 | 122.693 | 51.862 | 5.784e-15 | 0.065 | 0.873 | 6912 | 262144 |
| 1024 | 3031.693 | 378.694 | 204.008 | 2.071e-14 | 0.190 | 0.778 | 56576 | 1048576 |

## Cost statement

For retained blocks the work is `O(sum_b r_b(m_b+n_b))`; local phase charts cost `O(L_local N log n_theta)` and exact terminal repayment costs `O(P_near)`. Thus a bounded-rank, bounded-neighbor atlas is `O(r N log N + L_local N log n_theta + N n_theta)` with `O(N + sum_b r_b(m_b+n_b))` storage. Arbitrary folded geometry can force rank growth or many exact terminal pairs, so this implementation does not claim an unconditional subquadratic worst-case bound.

Fitted warm-apply exponent on this four-size run: `1.524`. Streamed direct exponent: `1.910`.

The ACA residual is sampled, not a continuum theorem. The reported errors use the independent streamed all-pairs action, while the exact pair-partition checksum verifies that every cross-slice pair is represented once.

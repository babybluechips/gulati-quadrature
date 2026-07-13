# Scale-phase geometry acceptance audit

| shape | sampled max | exhaustive max | exhaustive RMS | accepted |
|---|---:|---:|---:|:---:|
| exact_flat_annulus | 4.316e-15 | 4.316e-15 | 5.107e-16 | yes |
| straight_cylinder | 1.000e+00 | 1.000e+00 | 6.201e-01 | no |
| tapered_circular_cone | 9.484e-01 | 9.484e-01 | 6.263e-01 | no |
| bent_circular_tube | 1.000e+00 | 1.000e+00 | 7.430e-01 | no |
| twisted_ellipse | 1.000e+00 | 1.000e+00 | 6.995e-01 | no |
| toroidal_bundle | 1.000e+00 | 1.000e+00 | 8.993e-01 | no |
| aircraft_bundle | 1.000e+00 | 1.000e+00 | 7.294e-01 | no |

The flat annulus is the exact exponential normal form, including rigid translation. The other surfaces require a geometry repayment; applying the Cauchy core to them without that residual would change the physical operator.

The sampled pass is an `O(N log n_scale)` diagnostic. The fail-closed point-cloud wrapper defaults to the exhaustive streamed audit; structurally generated exponential charts use the exact core directly and need no pair audit.

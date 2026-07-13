# Multivariate peeled-resultant benchmark

The product tree carries `D`, `N_w`, and `N_wf` in a normalized sparse monomial basis. Every run uses a full independent peeled-sum audit and stores no pair matrix.

| N | resultant ms | direct ms | support D | total support | peeled audit error | strict pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 2.863e+00 | 4.750e-03 | 122 | 248 | 1.359e-15 | yes |
| 6 | 1.535e+01 | 1.058e-02 | 353 | 815 | 8.514e-15 | yes |
| 8 | 5.671e+01 | 2.096e-02 | 827 | 1977 | 7.996e-14 | yes |
| 10 | 1.536e+02 | 2.988e-02 | 1561 | 3879 | 9.161e-13 | no |
| 12 | 3.620e+02 | 4.812e-02 | 2622 | 6716 | 3.731e-12 | no |
| 14 | 7.887e+02 | 6.783e-02 | 4078 | 10684 | 3.329e-11 | no |
| 16 | 1.509e+03 | 9.092e-02 | 5993 | 15975 | 1.896e-10 | no |

The fitted denominator-support exponent is 2.826. The measured arithmetic exponent is 4.530, compared with 2.132 for the streamed reference over this small range.

The algebraic finite-part identity is correct, but a generic three-variable product fills a cubic monomial simplex and the monomial evaluation becomes ill-conditioned. With the strict `5.0e-13` audit, the tested resultant path is retained only through N=8; larger cases repay by the exact stream in production mode.

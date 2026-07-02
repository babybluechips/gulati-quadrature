# Symbol of Observation Certificate Audit

- source: `symbol_of_observation (1).pdf`
- sha256: `9ec1b0fc0917de004443564a5ae275b538fcc6a15ed022d43b23ed462b065b49`
- profile: `full`
- elapsed: `3.62 s`
- overall: `PASS`

| Certificate | Status | Key check |
|---|---:|---|
| C1 Spitzer | PASS | diff `3.983e-14`, E[M7] `-0.114199 + 3.274636i` |
| C2 strobe transfer | PASS | last error/Delta `1.205890` vs `|zeta(-1/2-14i)|=1.215649` |
| C3 blind mode | PASS | last |R|/Delta `1.215562` vs `1.225463` |
| C4 SRW anchor | PASS | c(4000) `-0.500000` vs `-0.5` |
| C5 sector theta_0.785398 | PASS | TV(3000) `1.189172` vs `1.189207` |
| C5 sector theta_1.047198 | PASS | TV(3000) `1.414125` vs `1.414214` |
| C6 unitary TV | PASS | TV/sqrt(k) `1.230578` vs Watson `1.217188` |
| C9 beta_1_1over4 | PASS | Im kappa `-0.374998` vs `-0.375000` |
| C9 beta_1_minus1over8 | PASS | Im kappa `-0.187498` vs `-0.187500` |
| C9 beta_half_0_half | PASS | Im kappa `-0.499981` vs `-0.500000` |

Interpretation: the PDF's executable claims survive the numerical audit in this profile. The strongest link to the Q/BGK pipeline is the shared endpoint ledger: uniform strobes and monitored boundary/path functionals both expose zeta-coded sampling defects rather than Monte Carlo noise.

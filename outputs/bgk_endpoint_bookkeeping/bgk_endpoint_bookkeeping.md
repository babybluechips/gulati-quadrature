# BGK Endpoint Bookkeeping

This verifies the point from `geometry_of_money (16).pdf`: BGK is the translator between continuously monitored crossing and discretely monitored crossing. It is the half-integer endpoint defect, not an arbitrary empirical shift.

![BGK endpoint bookkeeping](/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_endpoint_bookkeeping.png)

## Gaussian Walk Ledger

For a standard Gaussian walk, Spitzer's identity gives

```text
E max_{0<=k<=N} S_k = (1/sqrt(2 pi)) sum_{k=1}^N k^(-1/2)
                         = sqrt(2N/pi) + zeta(1/2)/sqrt(2 pi) + O(N^(-1/2)).
```

Therefore the continuous-to-discrete translator is

```text
beta_BGK = -zeta(1/2)/sqrt(2 pi)
         = 0.5825971579390108.
```

The Monte Carlo estimate is noisy at large `N` because the maximum itself fluctuates on the `sqrt(N)` scale. A representative row is `N=128`: exact Spitzer gives beta estimate `0.564978`, while Monte Carlo gives `0.597833` with stderr `0.061712` on the maximum itself.

The convergence proof is the exact Spitzer sum. At the largest exact row, `N=2048`, the beta estimate is `0.578190`. After the BGK subtraction the exact residual decays with fitted power `0.500` in `1/N`.

## Q/DtN Ledger

The same half-integer endpoint defect appears in the cycle Q/DtN spectral zeta sum:

```text
sum_{k=1}^{n-1} [k(1-k/n)]^(-1/2) = pi sqrt(n) + 2 zeta(1/2) + lower terms.
```

At `n=65536`, the raw endpoint residual is `-2.920712` and the BGK-corrected residual is `-0.000003`. The last five corrected levels fit power `1.000` in `1/n`.

## Chord-To-Arc Ledger

The inverse-square chord operator is the boundary version of the same bookkeeping. The continuum sees arc distance; the finite operator sees chord distance. The correction ladder records the defect between those two ledgers.

At `M=4096`, the relative arc-chord defect is `9.804571e-08`, while the inverse-square Q defect is `1.960914e-07`.

## Interpretation

In the heat/barrier language, continuous monitoring is the killed semigroup and discrete monitoring is the projected product. The missed Brownian-bridge crossings produce the `sqrt(h)` boundary-flux term. In Q language, that boundary flux is the DtN/Q operator; BGK is the endpoint translator that repays the discrete monitoring mesh.

So the full bookkeeping is:

```text
continuous heat/barrier operator
  -> discrete monitoring dates
  -> Brownian bridge endpoint defect
  -> zeta(1/2) / sqrt(2 pi)
  -> beta_BGK
  -> Q/DtN boundary flux correction
  -> chord-to-arc inverse-square repayment
```

## Artifacts

- Monte Carlo CSV: `/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_monte_carlo_spitzer.csv`
- Q/DtN endpoint CSV: `/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_q_dtn_endpoint.csv`
- chord/arc CSV: `/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_chord_arc_bookkeeping.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_endpoint_bookkeeping.json`
- figure: `/Users/rick/Documents/New project 2/outputs/bgk_endpoint_bookkeeping/bgk_endpoint_bookkeeping.png`

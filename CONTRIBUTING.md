# Contributing

The repo is organized around small, testable numerical primitives. Keep changes
scoped to one layer when possible:

- `geometry`: boundary sampling, normalization, curvature, star-shaped models;
- `operators`: Q/chord matrices and Hadamard finite-part extraction;
- `reconstruction`: polygon and low-mode reconstruction routines;
- `spectra`: heat-trace and spectral-zeta estimators;
- `cli`: file-oriented workflows.

## Local Checks

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
make test
make compile
make lint
```

`ruff` is an optional development dependency and is enforced in CI.

## Numerical Changes

For any numerical algorithm change, include a deterministic unit test that checks
one of:

- exact recovery on synthetic data;
- relative error below a stated tolerance;
- a clear failure mode for invalid input.

Prefer explicit shape checks and residual diagnostics over silent best-effort
behavior.

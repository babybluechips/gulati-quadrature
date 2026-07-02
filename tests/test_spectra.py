import numpy as np

from inverse_shape.spectra import fit_heat_trace_from_values, spectral_zeta


def test_heat_trace_fit_from_synthetic_values() -> None:
    t = np.geomspace(1e-3, 1e-1, 60)
    coeffs = np.array([0.8, -0.3, 0.2, -0.05, 0.01])
    exponents = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    values = sum(c * t**p for c, p in zip(coeffs, exponents, strict=True))
    fit = fit_heat_trace_from_values(t, values, max_order=2)
    assert fit.residual_norm < 1e-10
    assert np.allclose(fit.coefficients, coeffs, atol=1e-10)


def test_spectral_zeta_matches_direct_sum() -> None:
    eigenvalues = np.array([1.0, 4.0, 9.0, 16.0])
    s = 0.5 + 2.0j
    expected = sum(complex(lam) ** (-s) for lam in eigenvalues)
    assert abs(spectral_zeta(eigenvalues, s) - expected) < 1e-12

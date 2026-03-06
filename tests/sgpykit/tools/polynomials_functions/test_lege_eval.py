import pytest
import numpy as np

from sgpykit.tools.polynomials_functions import lege_eval, standard_lege_eval


# ----------------------------------------------------------------------
# Helper – weighted inner product on [a,b] with ρ = 1/(b‑a)
# ----------------------------------------------------------------------
def weighted_inner_product(f, g, a, b, n=10_001):
    """∫_a^b f(x)·g(x)·ρ dx  with ρ = 1/(b‑a)  (trapezoidal rule)."""
    x = np.linspace(a, b, n)
    rho = 1.0 / (b - a)
    return rho * np.trapezoid(f(x) * g(x), x)


# ----------------------------------------------------------------------
# Known closed‑form values (orthonormal version)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "k,a,b,x,expected",
    [
        # k = 0 → L₀(x) = 1  (already orthonormal)
        (0, -1.0, 1.0, np.array([0.0, 0.5, -0.3]), np.ones(3)),

        # k = 1 → L₁(x) = √3·(2·x‑a‑b)/(b‑a)
        (1, 0.0, 2.0,
         np.array([0.0, 1.0, 2.0]),
         np.sqrt(3) * (2 * np.array([0.0, 1.0, 2.0]) - 0.0 - 2.0) / (2.0 - 0.0)),

        # k = 2 on the canonical interval [-1,1]
        # orthonormal L₂(x) = √5·P₂(t)  with  t = (2x‑a‑b)/(b‑a)
        (2, -1.0, 1.0,
         np.array([-1.0, 0.0, 1.0]),
         np.sqrt(5) * standard_lege_eval(
             (2 * np.array([-1.0, 0.0, 1.0]) - (-1.0) - 1.0) / (1.0 - (-1.0)),
             2)),
    ],
)
def test_known_values(k, a, b, x, expected):
    """Check that ``lege_eval`` reproduces the orthonormal formulas for low orders."""
    out = lege_eval(x, k, a, b)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-14)


# ----------------------------------------------------------------------
# Orthogonality / normalisation on a generic interval
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "k1,k2,a,b",
    [
        (0, 0, -2.0, 5.0),
        (0, 1, -2.0, 5.0),
        (1, 1, -2.0, 5.0),
        (2, 3, -2.0, 5.0),
        (4, 4, -2.0, 5.0),
    ],
)
def test_orthonormality(k1, k2, a, b):
    """
    For orthonormal Legendre polynomials

        ∫_a^b L_k1(x) L_k2(x) ρ dx = δ_{k1,k2},

    with ρ = 1/(b‑a).
    """
    f = lambda x: lege_eval(x, k1, a, b)
    g = lambda x: lege_eval(x, k2, a, b)

    val = weighted_inner_product(f, g, a, b)

    if k1 == k2:
        assert np.isclose(val, 1.0, rtol=1e-6, atol=1e-8)
    else:
        assert np.isclose(val, 0.0, rtol=1e-6, atol=1e-8)


# ----------------------------------------------------------------------
# Vectorised input – shape must be preserved
# ----------------------------------------------------------------------
def test_vectorised_shape():
    """Check that the function works on a 2‑D array and keeps the shape."""
    a, b = 1.0, 4.0
    k = 3
    X = np.random.rand(5, 7) * (b - a) + a   # points inside [a,b]
    L = lege_eval(X, k, a, b)

    assert L.shape == X.shape
    # sanity: evaluate on a flattened view and compare
    L_flat = lege_eval(X.ravel(), k, a, b).reshape(X.shape)
    np.testing.assert_allclose(L, L_flat)


# ----------------------------------------------------------------------
# Consistency with the standard (non‑normalised) version
# ----------------------------------------------------------------------
def test_consistency_with_standard():
    """Undo the normalisation and recover the standard Legendre polynomial."""
    a, b = -1.0, 1.0
    k = 4
    x = np.linspace(-1, 1, 13)

    std = standard_lege_eval(x, k)                # P_k(t) on [-1,1]
    ortho = lege_eval(x, k, a, b)                 # orthonormal version

    # Remove the extra √2 factor and the norm factor √(2/(2k+1))
    recovered = ortho * np.sqrt(2 / (2 * k + 1)) / np.sqrt(2)

    np.testing.assert_allclose(recovered, std, rtol=1e-12, atol=1e-14)
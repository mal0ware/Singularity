"""Symbolic re-derivation of the Schwarzschild Christoffel symbols.

The simulator's geodesic integrator calls ``geodesic_rhs_schwarzschild`` in
``shared_shader/geodesic_math.h`` — a hand-coded evaluation of
``du^mu / dlambda = -Gamma^mu_{nu sigma} u^nu u^sigma`` with the non-zero
Christoffels of ``PHYSICS.md`` Section 3 substituted in.

A copy-paste error in any of those nine non-trivial terms would silently bend
every ray in every backend the same wrong way, so the bug would not be caught
by cross-backend equivalence tests. The guard is this file: we rebuild the
metric from first principles with SymPy, take derivatives, compute all 64
Christoffel components via the metric-compatibility formula, assemble the
right-hand side of the geodesic equation symbolically, and assert it is
identically equal (``sympy.simplify(a - b) == 0``) to the expression the
simulator is actually evaluating.

See ``docs/PHYSICS.md`` Sections 3 and 4 for the physics; runtime is under five
seconds on a laptop because the metric is diagonal.
"""

from __future__ import annotations

import pytest
import sympy as sp


def _schwarzschild_metric() -> tuple[sp.Matrix, tuple[sp.Symbol, ...], sp.Symbol]:
    """Build the Schwarzschild metric in ``(t, r, theta, phi)`` coordinates.

    Returns
    -------
    g : sp.Matrix
        4x4 covariant metric tensor ``g_{mu nu}``.
    coords : tuple of sp.Symbol
        The coordinate symbols in index order ``(t, r, theta, phi)``.
    rs : sp.Symbol
        Schwarzschild radius parameter ``r_s = 2 M``.
    """
    t, r, theta, phi = sp.symbols("t r theta phi", real=True, positive=False)
    rs = sp.symbols("rs", positive=True)
    f = 1 - rs / r  # lapse factor, PHYSICS.md Section 2
    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    return g, (t, r, theta, phi), rs


def _christoffels(g: sp.Matrix, coords: tuple[sp.Symbol, ...]) -> list[list[list[sp.Expr]]]:
    """Compute ``Gamma^mu_{nu sigma}`` for a diagonal (or general) metric.

    Uses the textbook metric-compatibility formula
    ``Gamma^mu_{nu sigma} = (1/2) g^{mu lambda}
        (d_nu g_{lambda sigma} + d_sigma g_{lambda nu} - d_lambda g_{nu sigma})``
    per ``PHYSICS.md`` Section 3. Result is indexed as ``gamma[mu][nu][sigma]``.
    """
    n = len(coords)
    g_inv = g.inv()
    gamma = [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            for sigma in range(n):
                acc = sp.Integer(0)
                for lam in range(n):
                    acc += g_inv[mu, lam] * (
                        sp.diff(g[lam, sigma], coords[nu])
                        + sp.diff(g[lam, nu], coords[sigma])
                        - sp.diff(g[nu, sigma], coords[lam])
                    )
                gamma[mu][nu][sigma] = sp.simplify(acc / 2)
    return gamma


def _symbolic_rhs(gamma: list[list[list[sp.Expr]]], u: tuple[sp.Symbol, ...]) -> list[sp.Expr]:
    """Assemble the geodesic RHS ``du^mu/dlambda = -Gamma^mu_{nu sigma} u^nu u^sigma``.

    Full 4x4 double sum over ``nu, sigma`` per component; SymPy naturally covers
    both orderings of any symmetric cross-term ``(nu, sigma)`` and ``(sigma, nu)``,
    which is what lets us compare directly against the hand-written kernel that
    uses the ``2 * Gamma`` symmetric form for off-diagonal Christoffels.
    """
    n = len(u)
    rhs = []
    for mu in range(n):
        acc = sp.Integer(0)
        for nu in range(n):
            for sigma in range(n):
                acc -= gamma[mu][nu][sigma] * u[nu] * u[sigma]
        rhs.append(sp.simplify(acc))
    return rhs


@pytest.mark.physics
def test_schwarzschild_christoffels_match_kernel() -> None:
    """Compare the SymPy-derived geodesic RHS against the hand-coded kernel.

    Each component of ``du^mu / dlambda`` is simplified symbolically and then
    differenced with the expression transcribed from
    ``geodesic_rhs_schwarzschild`` in ``shared_shader/geodesic_math.h``. A
    passing test means the two are identically equal as rational functions of
    ``(r, theta, rs, u^mu)``, which is stronger than any floating-point check.
    """
    g, coords, rs = _schwarzschild_metric()
    _, r, theta, _ = coords
    gamma = _christoffels(g, coords)

    # Four-velocity components as independent symbols; the RHS is algebraic in
    # them, not a function, so plain symbols suffice.
    ut, ur, utheta, uphi = sp.symbols("ut ur utheta uphi", real=True)
    rhs = _symbolic_rhs(gamma, (ut, ur, utheta, uphi))

    f = 1 - rs / r
    r2 = r**2
    sin_t = sp.sin(theta)
    cos_t = sp.cos(theta)

    # Hand-written expressions transcribed from geodesic_rhs_schwarzschild.
    # The cross-term in d(u^t)/dlambda is written here in the symmetric
    # "Gamma sums both orderings" form: the kernel has the coefficient
    # -(rs / (r^2 f)) which equals 2 * (-rs / (2 r^2 f)), matching the
    # SymPy sum over (nu, sigma) and (sigma, nu).
    expected = [
        -(rs / (r2 * f)) * ut * ur,
        (
            -(rs * f / (2 * r2)) * ut**2
            + (rs / (2 * r2 * f)) * ur**2
            + r * f * (utheta**2 + sin_t**2 * uphi**2)
        ),
        -2 * ur * utheta / r + sin_t * cos_t * uphi**2,
        -2 * ur * uphi / r - 2 * (cos_t / sin_t) * utheta * uphi,
    ]

    names = ("du^t/dlambda", "du^r/dlambda", "du^theta/dlambda", "du^phi/dlambda")
    for name, derived, hand in zip(names, rhs, expected, strict=True):
        diff = sp.simplify(derived - hand)
        assert diff == 0, (
            f"Mismatch in {name}:\n"
            f"  SymPy-derived: {sp.simplify(derived)}\n"
            f"  Hand-coded   : {sp.simplify(hand)}\n"
            f"  Difference   : {diff}\n"
            "The metric in PHYSICS.md Section 2 or the kernel in "
            "shared_shader/geodesic_math.h has drifted."
        )


@pytest.mark.physics
def test_schwarzschild_nonzero_christoffels_match_physics_md() -> None:
    """Spot-check the individual Christoffel components listed in PHYSICS.md Section 3.

    Catches a bug in which the geodesic RHS happens to match by coincidence
    (for instance, two sign errors that cancel in the double sum) but an
    individual ``Gamma^mu_{nu sigma}`` is still wrong. We check every non-zero
    component documented in the reference.
    """
    g, coords, rs = _schwarzschild_metric()
    _, r, theta, _ = coords
    gamma = _christoffels(g, coords)
    f = 1 - rs / r

    # Index order: (t, r, theta, phi) -> (0, 1, 2, 3).
    expected_nonzero = {
        (0, 0, 1): rs / (2 * r**2 * f),  # Gamma^t_{tr}
        (0, 1, 0): rs / (2 * r**2 * f),  # Gamma^t_{rt}
        (1, 0, 0): rs * f / (2 * r**2),  # Gamma^r_{tt}
        (1, 1, 1): -rs / (2 * r**2 * f),  # Gamma^r_{rr}
        (1, 2, 2): -r * f,  # Gamma^r_{theta theta}
        (1, 3, 3): -r * f * sp.sin(theta) ** 2,  # Gamma^r_{phi phi}
        (2, 1, 2): 1 / r,  # Gamma^theta_{r theta}
        (2, 2, 1): 1 / r,  # Gamma^theta_{theta r}
        (2, 3, 3): -sp.sin(theta) * sp.cos(theta),  # Gamma^theta_{phi phi}
        (3, 1, 3): 1 / r,  # Gamma^phi_{r phi}
        (3, 3, 1): 1 / r,  # Gamma^phi_{phi r}
        (3, 2, 3): sp.cos(theta) / sp.sin(theta),  # Gamma^phi_{theta phi}
        (3, 3, 2): sp.cos(theta) / sp.sin(theta),  # Gamma^phi_{phi theta}
    }

    for (mu, nu, sigma), expected in expected_nonzero.items():
        derived = gamma[mu][nu][sigma]
        diff = sp.simplify(derived - expected)
        assert diff == 0, (
            f"Gamma^{mu}_({nu},{sigma}) mismatch:\n"
            f"  SymPy: {sp.simplify(derived)}\n"
            f"  PHYSICS.md Section 3: {sp.simplify(expected)}\n"
            f"  Difference: {diff}"
        )

    # And the zero components really are zero: scan the full 64-entry tensor
    # and require anything not in the expected set to simplify to 0.
    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                if (mu, nu, sigma) in expected_nonzero:
                    continue
                assert sp.simplify(gamma[mu][nu][sigma]) == 0, (
                    f"Unexpected non-zero Christoffel "
                    f"Gamma^{mu}_({nu},{sigma}) = {sp.simplify(gamma[mu][nu][sigma])}"
                )

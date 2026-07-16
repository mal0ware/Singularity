# verification/test_kerr_ham_analytic.py
#
# SymPy cross-check of the analytic Hamiltonian partials used by
# shared_shader/kerr_hamilton.h (and the WGSL hand-port). The C++ RHS
# computes dp_r/dλ = -1/2 ∂(2H)/∂r and dp_θ/dλ = -1/2 ∂(2H)/∂θ from the
# grouped form
#
#   2H = B/(ΣΔ) + L_z²/(Σ sin²θ) + (Δ p_r² + p_θ²)/Σ
#   B  = -A E² + 4 M r a E L_z - a² L_z²
#
# with hand-derived quotient-rule partials. This test re-implements those
# exact formulas in Python and asserts they equal SymPy's direct symbolic
# derivative of 2H, evaluated over a parameter grid spanning the spins,
# radii, and polar angles the renderer visits. Tolerance is tight (1e-9
# relative in float64) because both sides are the same mathematics — any
# transcription error in the grouped form shows up enormously above it.
#
# Rationale: the earlier centred-finite-difference RHS was replaced because
# it cancels the polar centrifugal barrier into noise near θ = 0/π (the
# dotted axis seam). PHYSICS.md §7.3 / §11.

import itertools

import numpy as np
import pytest
import sympy as sp

# --- symbols -----------------------------------------------------------------
r, th, pr, pth, E, L, a, M = sp.symbols("r theta p_r p_theta E L a M", real=True, positive=False)

_s2 = sp.sin(th) ** 2
_c2 = sp.cos(th) ** 2
_Sigma = r**2 + a**2 * _c2
_Delta = r**2 - 2 * M * r + a**2
_A = (r**2 + a**2) ** 2 - a**2 * _Delta * _s2
_SD = _Sigma * _Delta

_twoH = (
    -_A * E**2 / _SD
    + 4 * M * r * a * E * L / _SD
    + (_Delta - a**2 * _s2) * L**2 / (_SD * _s2)
    + _Delta * pr**2 / _Sigma
    + pth**2 / _Sigma
)

_args = (r, th, pr, pth, E, L, a, M)
_ref_dr = sp.lambdify(_args, sp.diff(_twoH, r), "numpy")
_ref_dth = sp.lambdify(_args, sp.diff(_twoH, th), "numpy")


def _grouped_partials(rv, thv, prv, pthv, Ev, Lv, av, Mv):
    """Line-for-line mirror of the analytic partials in kerr_ham_rhs
    (shared_shader/kerr_hamilton.h). Keep in sync with the C++ and the WGSL
    port — this function IS the reference the shaders are checked against."""
    st = np.sin(thv)
    ct = np.cos(thv)
    s2 = st * st
    c2 = ct * ct
    r2 = rv * rv
    a2 = av * av
    Sigma = r2 + a2 * c2
    Delta = r2 - 2.0 * Mv * rv + a2
    A = (r2 + a2) ** 2 - a2 * Delta * s2
    SD = Sigma * Delta

    L2 = Lv * Lv
    B = -A * Ev * Ev + 4.0 * Mv * rv * av * Ev * Lv - a2 * L2
    Sig_r = 2.0 * rv
    Del_r = 2.0 * rv - 2.0 * Mv
    A_r = 4.0 * rv * (r2 + a2) - a2 * Del_r * s2
    B_r = -A_r * Ev * Ev + 4.0 * Mv * av * Ev * Lv
    sin2t = 2.0 * st * ct
    Sig_t = -a2 * sin2t
    B_t = a2 * Delta * sin2t * Ev * Ev
    Sigma2 = Sigma * Sigma
    Dp2 = Delta * prv * prv + pthv * pthv

    dH_dr = (
        B_r / SD
        - B * (Sig_r * Delta + Sigma * Del_r) / (SD * SD)
        - L2 * Sig_r / (Sigma2 * s2)
        + Del_r * prv * prv / Sigma
        - Dp2 * Sig_r / Sigma2
    )
    dH_dth = (
        B_t / SD
        - B * Sig_t / (Sigma2 * Delta)
        - L2 * (Sig_t * s2 + Sigma * sin2t) / (Sigma2 * s2 * s2)
        - Dp2 * Sig_t / Sigma2
    )
    return dH_dr, dH_dth


GRID = list(
    itertools.product(
        [2.6, 3.1, 6.0, 12.0, 40.0, 150.0],  # r/M
        [0.05, 0.4, 1.0, np.pi / 2, 2.4, np.pi - 0.05],  # theta
        [0.0, 0.5, 0.94, 0.998],  # a/M
        [-7.5, 0.0, 3.7],  # L_z
    )
)


@pytest.mark.parametrize("rv,thv,av,Lv", GRID)
def test_grouped_partials_match_sympy(rv, thv, av, Lv):
    Ev, Mv, prv, pthv = 1.0, 1.0, 0.6, 2.3
    ref_r = float(_ref_dr(rv, thv, prv, pthv, Ev, Lv, av, Mv))
    ref_t = float(_ref_dth(rv, thv, prv, pthv, Ev, Lv, av, Mv))
    got_r, got_t = _grouped_partials(rv, thv, prv, pthv, Ev, Lv, av, Mv)
    assert got_r == pytest.approx(ref_r, rel=1e-9, abs=1e-9)
    assert got_t == pytest.approx(ref_t, rel=1e-9, abs=1e-9)


def test_polar_barrier_is_repulsive_and_vanishes_for_zero_Lz():
    # The property the finite-difference RHS lost: just shy of the pole the
    # centrifugal barrier must push back toward the equator, and for
    # L_z = 0 the barrier term must vanish identically (not blow up).
    rv, av, Ev, Mv = 8.0, 0.9, 1.0, 1.0
    thv = np.pi - 0.01

    dH_dr, dH_dth = _grouped_partials(rv, thv, 0.0, 1.0, Ev, 0.05, av, Mv)
    assert dH_dth > 0.0  # dp_θ/dλ = -1/2 dH/dθ < 0: decelerates poleward motion

    dH_dr0, dH_dth0 = _grouped_partials(rv, thv, 0.0, 1.0, Ev, 0.0, av, Mv)
    assert np.isfinite(dH_dth0)
    assert abs(dH_dth0) < abs(dH_dth)

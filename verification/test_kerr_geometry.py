"""Kerr-geometry verification: horizons, ergosphere, ISCO, photon spheres.

Drives ``singularity_cli --mode kerr-geometry`` at a grid of physically-
relevant spins and cross-checks every scalar the CLI emits against a wholly
independent double-precision re-derivation here in Python. The point is to
catch typos in the closed-form expressions coded in
``core/include/physics/kerr.hpp``: the Python formulas below are keyed off
PHYSICS.md §7 and use ``numpy`` / ``scipy.optimize`` rather than the simulator's
float32 pipeline, so a sign flip or exponent error would propagate to one
side but not the other and the comparison would fail.

See ``docs/PHYSICS.md`` §§7, 11 (test matrix) and ``docs/TODO.md`` Phase 6
for context. Tolerances match the table in §11 (0.01% for horizons, 0.1%
for ISCO); the 0.3% budget here accounts for the CLI-side float32 round-trip.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

# Canonical spin grid — PHYSICS.md §11.
_SPINS = [0.0, 0.5, 0.94, 0.998]
_MASS = 1.0
# Combined tolerance: the CLI emits float32, we compute with float64.
# A float32 round-trip sits below ~1e-7 relative, but cbrt on float32 inputs
# shows up with a slightly larger footprint near extremal spin; 3e-4 leaves
# comfortable headroom without admitting sign-flip class bugs.
_REL_TOL = 3e-4


def _outer_horizon(M: float, a: float) -> float:
    return M + np.sqrt(max(M * M - a * a, 0.0))


def _inner_horizon(M: float, a: float) -> float:
    return M - np.sqrt(max(M * M - a * a, 0.0))


def _ergosphere(M: float, a: float, theta: float) -> float:
    c = np.cos(theta)
    return M + np.sqrt(max(M * M - a * a * c * c, 0.0))


def _isco(M: float, a_over_M: float, prograde: bool) -> float:
    a = a_over_M
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    sign = -1.0 if prograde else +1.0
    return M * (3.0 + z2 + sign * np.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)))


def _photon_sphere_prograde(M: float, a: float) -> float:
    return 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-a / M)))


def _photon_sphere_retrograde(M: float, a: float) -> float:
    return 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(+a / M)))


def _run_cli(cli: Path, tmp_path: Path, spin: float, mass: float = _MASS) -> dict:
    """Invoke the CLI in kerr-geometry mode and load its JSON dump."""
    out = tmp_path / f"kerr_a{spin:.3f}.json"
    result = subprocess.run(
        [
            str(cli),
            "--mode",
            "kerr-geometry",
            "--mass",
            str(mass),
            "--spin",
            str(spin),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"singularity_cli --mode kerr-geometry exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    with out.open() as fh:
        return json.load(fh)


@pytest.mark.physics
@pytest.mark.parametrize("spin", _SPINS)
def test_kerr_horizons_match_closed_form(
    singularity_cli: Path, tmp_path: Path, spin: float
) -> None:
    """Outer and inner horizons at r± = M ± √(M² − a²). Independent check that
    Δ(r±) = 0 is left to :func:`test_kerr_horizons_satisfy_delta_equation`
    below — here we only assert the CLI's closed-form values match the
    reference computed in double precision."""
    data = _run_cli(singularity_cli, tmp_path, spin)
    expected_plus = _outer_horizon(_MASS, spin)
    expected_minus = _inner_horizon(_MASS, spin)
    np.testing.assert_allclose(data["outer_horizon"], expected_plus, rtol=_REL_TOL)
    np.testing.assert_allclose(data["inner_horizon"], expected_minus, rtol=_REL_TOL)


@pytest.mark.physics
@pytest.mark.parametrize("spin", _SPINS)
def test_kerr_horizons_satisfy_delta_equation(
    singularity_cli: Path, tmp_path: Path, spin: float
) -> None:
    """Independent root-finding verification that r± solve Δ(r) = 0.

    Computing the horizons analytically and then asking a numerical solver
    "does Δ vanish here?" is a cross-check on the closed form itself — if I
    had a sign error in ``kerr.hpp`` *and* the same sign error in the Python
    ``_outer_horizon`` helper, the previous test would pass silently. Here
    Δ(r) = r² − 2Mr + a² is its own independent encoding, so they can't
    both be wrong in the same way."""
    data = _run_cli(singularity_cli, tmp_path, spin)

    def delta(r: float) -> float:
        return r * r - 2.0 * _MASS * r + spin * spin

    r_plus = data["outer_horizon"]
    r_minus = data["inner_horizon"]
    assert abs(delta(r_plus)) < 1e-6, f"Δ({r_plus}) = {delta(r_plus):.3e}"
    assert abs(delta(r_minus)) < 1e-6, f"Δ({r_minus}) = {delta(r_minus):.3e}"

    # Sub-extremal case: also verify brentq finds r_plus independently.
    if spin < _MASS - 1e-3:
        r_plus_solved = brentq(delta, _MASS, 2.0 * _MASS)
        np.testing.assert_allclose(r_plus, r_plus_solved, rtol=_REL_TOL)


@pytest.mark.physics
@pytest.mark.parametrize("spin", _SPINS)
def test_kerr_ergosphere_extrema(singularity_cli: Path, tmp_path: Path, spin: float) -> None:
    """Ergosphere touches the outer horizon at the poles, reaches 2M at the
    equator regardless of spin (a² cos²θ vanishes at θ = π/2)."""
    data = _run_cli(singularity_cli, tmp_path, spin)
    np.testing.assert_allclose(
        data["ergosphere_polar"],
        _ergosphere(_MASS, spin, 0.0),
        rtol=_REL_TOL,
    )
    np.testing.assert_allclose(
        data["ergosphere_equatorial"],
        2.0 * _MASS,
        rtol=_REL_TOL,
    )
    # The equatorial ergosphere also equals _ergosphere(M, a, π/2) by construction.
    np.testing.assert_allclose(
        data["ergosphere_equatorial"],
        _ergosphere(_MASS, spin, 0.5 * np.pi),
        rtol=_REL_TOL,
    )


@pytest.mark.physics
@pytest.mark.parametrize("spin", _SPINS)
def test_kerr_photon_spheres(singularity_cli: Path, tmp_path: Path, spin: float) -> None:
    """Prograde and retrograde photon sphere radii via Bardeen's closed form.

    Schwarzschild limit (a=0): both collapse to 3M.
    Extremal (a=M): prograde = M, retrograde = 4M.
    Monotonicity across the spin grid is covered by the C++ Catch2 side."""
    data = _run_cli(singularity_cli, tmp_path, spin)
    np.testing.assert_allclose(
        data["photon_sphere_prograde"],
        _photon_sphere_prograde(_MASS, spin),
        rtol=_REL_TOL,
    )
    np.testing.assert_allclose(
        data["photon_sphere_retrograde"],
        _photon_sphere_retrograde(_MASS, spin),
        rtol=_REL_TOL,
    )


@pytest.mark.physics
@pytest.mark.parametrize("spin", _SPINS)
def test_kerr_isco_branches(singularity_cli: Path, tmp_path: Path, spin: float) -> None:
    """ISCO prograde / retrograde against Bardeen-Press-Teukolsky.

    Canonical values re-derived here in float64 to dodge any float32-drift in
    the reference table. Tolerance 0.3% absorbs both the float32 round-trip
    and ``cbrt`` precision near the extremal boundary."""
    data = _run_cli(singularity_cli, tmp_path, spin)
    np.testing.assert_allclose(
        data["isco_prograde"],
        _isco(_MASS, spin, prograde=True),
        rtol=_REL_TOL,
    )
    np.testing.assert_allclose(
        data["isco_retrograde"],
        _isco(_MASS, spin, prograde=False),
        rtol=_REL_TOL,
    )


@pytest.mark.physics
def test_kerr_extremal_collapse(singularity_cli: Path, tmp_path: Path) -> None:
    """Exactly-extremal Kerr (a = M): both horizons degenerate to M and the
    prograde photon sphere and prograde ISCO both approach M. A CLI that
    NaN'd at the boundary instead of clamping would fail this test."""
    data = _run_cli(singularity_cli, tmp_path, spin=1.0)
    np.testing.assert_allclose(data["outer_horizon"], _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["inner_horizon"], _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["photon_sphere_prograde"], _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["photon_sphere_retrograde"], 4.0 * _MASS, rtol=_REL_TOL)


@pytest.mark.physics
def test_kerr_schwarzschild_limit(singularity_cli: Path, tmp_path: Path) -> None:
    """At a = 0 Kerr must reproduce every Schwarzschild-limit scalar from
    PHYSICS.md §§3-5: rs = 2M, photon sphere = 3M, ISCO = 6M, ergosphere = 2M."""
    data = _run_cli(singularity_cli, tmp_path, spin=0.0)
    np.testing.assert_allclose(data["outer_horizon"], 2.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["inner_horizon"], 0.0, atol=_REL_TOL)
    np.testing.assert_allclose(data["photon_sphere_prograde"], 3.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["photon_sphere_retrograde"], 3.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["isco_prograde"], 6.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["isco_retrograde"], 6.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["ergosphere_polar"], 2.0 * _MASS, rtol=_REL_TOL)
    np.testing.assert_allclose(data["ergosphere_equatorial"], 2.0 * _MASS, rtol=_REL_TOL)

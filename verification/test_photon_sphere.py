"""Photon-sphere closure test — PHYSICS.md §5.2 and §11 test matrix entry.

Drives ``singularity_cli --mode photon-orbit`` at ``r = 1.5 r_s`` (the
Schwarzschild photon sphere) and verifies the unstable circular null orbit
closes within 0.5% of its initial radius after one orbital period, and that
the azimuth ``φ`` advances monotonically by approximately ``2π``. A
Christoffel-sign error would push the orbit off this unstable equilibrium
quickly — over one period the drift would be catastrophic, far beyond the
0.5% budget.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest

_M = 1.0
_RS = 2.0 * _M
_R_PHOTON = 1.5 * _RS  # = 3 M
_CLOSURE_TOLERANCE = 0.005  # 0.5% per PHYSICS.md §11
# Kerr-side equatorial-leak tolerance. Tighter than _CLOSURE_TOLERANCE
# because θ=π/2 is an exact constant of motion for the initial conditions.
# 5e-4 rad = 0.029°. clang/-O3 on Apple silicon keeps drift under 1e-4;
# g++-14/-O3/-ffast-math on Ubuntu runners drifts to ~2.2e-4 at a/M=0.9.
_KERR_THETA_EQUATORIAL_TOLERANCE = 5e-4


def _run_orbit(
    cli: Path, tmp_path: Path, r_init: float, orbits: float = 1.0, h_step: float = 0.01
) -> list[dict]:
    out = tmp_path / f"orbit_r{r_init:.3f}.csv"
    result = subprocess.run(
        [
            str(cli),
            "--mode",
            "photon-orbit",
            "--mass",
            str(_M),
            "--r-init",
            str(r_init),
            "--orbits",
            str(orbits),
            "--h-step",
            str(h_step),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"singularity_cli exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    rows: list[dict] = []
    with out.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return rows


@pytest.mark.physics
def test_photon_sphere_closes_within_half_percent(singularity_cli: Path, tmp_path: Path) -> None:
    """Circular orbit at r = 1.5 r_s returns to its starting radius after one
    period, within the 0.5% budget PHYSICS.md §11 targets."""
    rows = _run_orbit(singularity_cli, tmp_path, _R_PHOTON, orbits=1.0)
    assert rows, "photon-orbit CLI produced no rows"
    r_init = rows[0]["r"]
    # Any escape or capture sentinel row (step = -1) would indicate the RK4
    # integrator crashed the orbit off the photon sphere within one period.
    for row in rows:
        assert (
            row["step"] >= 0
        ), f"orbit destabilised before one period — sentinel {row['step']} at r={row['r']}"
    r_final = rows[-1]["r"]
    rel_drift = abs(r_final - r_init) / r_init
    assert rel_drift <= _CLOSURE_TOLERANCE, (
        f"r drifted by {rel_drift * 100:.3f}% over one photon-sphere period "
        f"(r_init={r_init:.6f}, r_final={r_final:.6f}); tolerance is "
        f"{_CLOSURE_TOLERANCE * 100:.2f}%"
    )


@pytest.mark.physics
def test_photon_sphere_theta_stays_equatorial(singularity_cli: Path, tmp_path: Path) -> None:
    """θ = π/2 is a constant of the motion for an equatorial null orbit. Any
    leak into θ-motion would indicate a polar-Christoffel bug."""
    rows = _run_orbit(singularity_cli, tmp_path, _R_PHOTON, orbits=1.0)
    thetas = np.array([row["theta"] for row in rows if row["step"] >= 0])
    assert np.all(np.isfinite(thetas))
    max_deviation = np.max(np.abs(thetas - 0.5 * np.pi))
    assert max_deviation < 1e-5, f"θ drifted from π/2 by {max_deviation:.3e} over one period"


@pytest.mark.physics
def test_photon_sphere_phi_advances_monotonically(singularity_cli: Path, tmp_path: Path) -> None:
    """φ is strictly increasing for a prograde orbit and covers ~2π in one
    period. A backwards or stuck φ coordinate would suggest the u^φ
    initialisation or the φ-Christoffel evolution is wrong.
    """
    rows = _run_orbit(singularity_cli, tmp_path, _R_PHOTON, orbits=1.0)
    phis = np.array([row["phi"] for row in rows if row["step"] >= 0])
    diffs = np.diff(phis)
    assert np.all(diffs > 0), "φ not monotonically increasing"
    total = phis[-1] - phis[0]
    # One period should advance φ by ~2π; allow ±2% slack for the period
    # estimate (the RK4 step length doesn't divide the period evenly).
    assert (
        abs(total - 2.0 * np.pi) < 0.04 * np.pi
    ), f"φ advanced by {total:.4f} rad after one period, expected ~2π"


@pytest.mark.physics
def test_photon_orbit_at_3_radius_survives_multi_period(
    singularity_cli: Path, tmp_path: Path
) -> None:
    """Same orbit integrated for three periods — a stricter closure check
    that catches slow Christoffel drift invisible in a single period."""
    rows = _run_orbit(singularity_cli, tmp_path, _R_PHOTON, orbits=3.0, h_step=0.005)
    for row in rows:
        assert row["step"] >= 0, f"orbit destabilised within three periods — sentinel {row['step']}"
    r_init = rows[0]["r"]
    r_final = rows[-1]["r"]
    # Tolerance scales with orbit count but still tight — RK4 at h = 0.005
    # over three periods should stay within 1% of r_init.
    assert abs(r_final - r_init) / r_init < 0.01


def _run_kerr_orbit(cli: Path, tmp_path: Path, spin: float, h_step: float = 0.001) -> list[dict]:
    """Drive photon-orbit in Kerr mode at auto-selected r_ph_pro(a)."""
    out = tmp_path / f"kerr_orbit_a{spin:.3f}.csv"
    result = subprocess.run(
        [
            str(cli),
            "--mode",
            "photon-orbit",
            "--mass",
            str(_M),
            "--spin",
            str(spin),
            "--orbits",
            "1.0",
            "--h-step",
            str(h_step),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    rows: list[dict] = []
    with out.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return rows


@pytest.mark.physics
@pytest.mark.parametrize("spin", [0.3, 0.5, 0.7, 0.9])
def test_kerr_prograde_photon_orbit_closes(
    singularity_cli: Path, tmp_path: Path, spin: float
) -> None:
    """Kerr equatorial prograde null circular orbit at the auto-selected
    ``r_ph_pro(a)`` must close within 0.5% after one orbital period. This
    exercises the Hamiltonian-form Kerr integrator end-to-end — any drift
    here would indicate the momentum-derivative numerics or the
    coordinate-velocity formulas are off."""
    rows = _run_kerr_orbit(singularity_cli, tmp_path, spin)
    assert rows, "CLI produced no rows"
    for row in rows:
        assert row["step"] >= 0, f"Kerr orbit at a={spin} destabilised — sentinel {row['step']}"
    r_init = rows[0]["r"]
    r_final = rows[-1]["r"]
    rel_drift = abs(r_final - r_init) / r_init
    assert rel_drift <= _CLOSURE_TOLERANCE, (
        f"Kerr prograde photon orbit at a={spin}, r_init={r_init:.6f}: "
        f"r drifted by {rel_drift * 100:.3f}% over one period"
    )
    # θ must stay on the equator.
    thetas = np.array([row["theta"] for row in rows])
    assert np.max(np.abs(thetas - 0.5 * np.pi)) < _KERR_THETA_EQUATORIAL_TOLERANCE
    # φ monotonic (prograde, so increasing).
    phis = np.array([row["phi"] for row in rows])
    assert np.all(np.diff(phis) > 0)

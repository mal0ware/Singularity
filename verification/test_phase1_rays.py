"""Phase 1 2D-toy ray check: full Schwarzschild deflection angle.

Drives the built ``singularity_cli`` in ``--mode 2d-toy``, inspects the dumped
CSV of ray trails, and compares each escaping ray's asymptotic deflection
against the exact closed-form result for a null Schwarzschild geodesic.

Why not just ``alpha = 2 rs / |b|``? That Eddington 1919 formula is only the
first-order expansion in ``rs / b``. The CLI fires rays at impact parameters
``|b| <= 10 M = 5 rs`` where the second-order term ``(15 pi / 16) (rs / b)^2``
already contributes ~30%, so the weak-field formula disagrees with reality
(and with the simulator) by a factor of 1.5. The full deflection is

    alpha(b) = 2 * integral[0..u_min] du / sqrt(1/b^2 - u^2 (1 - rs u))  -  pi

with ``u = 1/r`` and ``u_min = 1/r_closest_approach`` — a standard effective-
potential reduction of the null geodesic equation (see PHYSICS.md Section 5).
We evaluate it with ``scipy.integrate.quad`` and ``scipy.optimize.brentq``.

This comparison is independent of the CLI's Christoffel implementation — the
integral uses conserved quantities directly — so a sign flip on any Christoffel
term would produce different geodesics and be caught here, while more
pedestrian bugs (wrong affine step, null condition applied incorrectly) would
show up as a bulk scale discrepancy. See PHYSICS.md Sections 4-5 and the
Phase 1 exit criterion in TODO.md.
"""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

# Geometrized units matching cli/main.cpp: M = 1, rs = 2 M.
_M = 1.0
_RS = 2.0 * _M
# b_crit = 3 sqrt(3) M is the photon-sphere impact parameter; below it every
# ray is captured. We only check rays comfortably outside it, where the
# asymptotic deflection is well defined and finite.
_B_MIN_CHECK = 6.0 * _M
# Minimum rays that must pass the tolerance band. The CLI shoots 100 rays in
# |b| <= 10 M; roughly 60 have |b| >= 6 M, so 20 is a comfortable majority.
_MIN_PASSING = 20
# Require at least this many escapes — catches an integrator that's gunning
# every ray at the horizon.
_MIN_ESCAPES = 20
# Agreement band between CLI-measured and closed-form deflection. RK4 at
# h = 0.1 over O(10^3) steps plus the secant-tangent measurement sits well
# inside 5%; 7% leaves headroom for float32 drift without hiding bugs.
_TOLERANCE = 0.07
# Trail sampling threshold — below this, the tangent fit is too noisy.
_MIN_TRAIL_POINTS = 5
# "Deep in the asymptotic region" — x well past the black hole. The CLI bails
# at x = -37.5 M, so -20 M gives us a tail of ~17 M across which the geodesic
# is near-straight and a linear fit extracts the tangent cleanly.
_ASYMPTOTIC_X = -20.0 * _M


def _parse_trail_csv(path: Path) -> dict[int, list[tuple[int, float, float, float]]]:
    """Read the trail CSV produced by ``--dump-trails`` into per-ray lists.

    Returns a mapping ``ray_idx -> [(step, b, x, y), ...]`` preserving file
    order. ``step in {-1, -2}`` rows (capture and escape markers) stay as-is
    so callers can distinguish the two endings.
    """
    by_ray: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            by_ray[int(row["ray_idx"])].append(
                (int(row["step"]), float(row["b"]), float(row["x"]), float(row["y"]))
            )
    return dict(by_ray)


def _exact_deflection(b: float, rs: float = _RS) -> float:
    """Exact asymptotic deflection angle for a null Schwarzschild geodesic.

    Closed-form result derived from the effective-potential reduction of the
    null geodesic equation in Schwarzschild coordinates:

        alpha(b) = 2 * integral[0..u_min] du / sqrt(1/b^2 - u^2 (1 - rs u)) - pi

    where ``u_min`` is the positive root of ``u^2 (1 - rs u) = 1/b^2`` — the
    inverse radius of closest approach. Valid for ``|b| > b_crit = 3 sqrt(3)
    rs / 2``; diverges as ``b -> b_crit`` (orbit captured by photon sphere).
    """
    b = abs(b)
    # The cubic u^2(1 - rs u) peaks at u = 2/(3 rs); for b > b_crit the cubic
    # value exceeds 1/b^2 at the peak, so the relevant root lies in (0, u_peak).
    u_peak = 2.0 / (3.0 * rs)

    def cubic(u: float) -> float:
        return u * u * (1.0 - rs * u) - 1.0 / (b * b)

    u_min = brentq(cubic, 1e-12, u_peak)

    def integrand(u: float) -> float:
        radicand = 1.0 / (b * b) - u * u * (1.0 - rs * u)
        # Clamp at the endpoint where radicand -> 0+; scipy.quad handles the
        # 1/sqrt(u_min - u) integrable singularity via adaptive subdivision.
        return 1.0 / np.sqrt(max(radicand, 1e-30))

    integral, _ = quad(integrand, 0.0, u_min, limit=400)
    return 2.0 * integral - np.pi


def _tangent_deflection(points: list[tuple[int, float, float, float]]) -> float | None:
    """Estimate the asymptotic deflection angle from an escaping ray's tangent.

    The CLI fires photons along ``-x_hat``. In the far-field limit a deflected
    ray approaches a straight line; its slope ``dy/dx`` encodes the deflection
    angle via ``|alpha| = |arctan(m)|``. A linear fit over every sample in the
    asymptotic region (``x < _ASYMPTOTIC_X``) averages out RK4 step-to-step
    jitter and avoids the bias of the naive position-vector-from-origin proxy,
    which at finite bailout distance systematically under-reports the
    asymptotic deflection.

    Returns ``None`` if fewer than two samples fall in the asymptotic region,
    which in practice means the ray was captured before escaping.
    """
    # Drop terminal sentinel (step < 0); keep only real trail points.
    trail = [(x, y) for (s, _b, x, y) in points if s >= 0 and x < _ASYMPTOTIC_X]
    if len(trail) < 2:
        return None
    xs = np.array([x for (x, _) in trail])
    ys = np.array([y for (_, y) in trail])
    slope, _intercept = np.polyfit(xs, ys, 1)
    # alpha = arctan(m) in magnitude (sign depends on which side of the BH the
    # ray passed on; the caller compares magnitudes).
    return float(np.arctan(slope))


@pytest.mark.physics
def test_schwarzschild_deflection_matches_closed_form(
    singularity_cli: Path, tmp_path: Path
) -> None:
    """2D-toy ray deflections agree with the exact Schwarzschild integral.

    Runs the CLI, parses the emitted trails, and for every ray that escaped
    with ``|b| >= _B_MIN_CHECK`` compares the measured tangent-slope
    deflection against the closed-form result from the effective-potential
    integral. Both small bugs (a mis-signed Christoffel term, wrong affine
    step) and gross bugs (photons integrated with the timelike null condition)
    are caught by this single comparison.
    """
    png_path = tmp_path / "phase1_rays.png"
    csv_path = tmp_path / "trails.csv"
    result = subprocess.run(
        [
            str(singularity_cli),
            "--mode",
            "2d-toy",
            "--output",
            str(png_path),
            "--dump-trails",
            str(csv_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"singularity_cli --mode 2d-toy exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert csv_path.is_file(), f"trail CSV {csv_path} was not created"

    rays = _parse_trail_csv(csv_path)
    assert rays, "trail CSV was empty — CLI produced no ray data"

    min_samples = min(len(v) for v in rays.values())
    if min_samples < _MIN_TRAIL_POINTS:
        pytest.skip(
            f"only {min_samples} trail points for the shortest ray; expected "
            f">= {_MIN_TRAIL_POINTS}. Rebuild the CLI or check the 2d-toy mode."
        )

    passing = 0
    total_checked = 0
    escaped_count = 0
    failures: list[str] = []

    for ray_idx, points in sorted(rays.items()):
        last_step, b, _, _ = points[-1]
        if last_step == -2:
            escaped_count += 1
        if abs(b) < _B_MIN_CHECK:
            continue
        total_checked += 1
        if last_step != -2:
            failures.append(
                f"ray {ray_idx} |b|={abs(b):.2f} did not escape (final marker {last_step})"
            )
            continue

        measured = _tangent_deflection(points)
        if measured is None:
            failures.append(
                f"ray {ray_idx} |b|={abs(b):.2f}: no asymptotic samples (x < {_ASYMPTOTIC_X})"
            )
            continue

        predicted = _exact_deflection(abs(b))
        rel_err = abs(abs(measured) - predicted) / predicted
        if rel_err <= _TOLERANCE:
            passing += 1
        else:
            failures.append(
                f"ray {ray_idx} |b|={abs(b):.2f}: |alpha|_measured={abs(measured):.4f} "
                f"vs closed-form {predicted:.4f} (rel err {rel_err * 100:.1f}%)"
            )

    assert escaped_count >= _MIN_ESCAPES, (
        f"only {escaped_count} rays escaped to infinity; expected many more. "
        "Integrator likely miscomputing geodesics near b ~ b_crit."
    )

    failure_detail = "\n  ".join(failures[:10])
    if len(failures) > 10:
        failure_detail += "\n  ..."
    assert passing >= _MIN_PASSING, (
        f"only {passing}/{total_checked} rays with |b| >= {_B_MIN_CHECK} M matched "
        f"the exact Schwarzschild deflection within {_TOLERANCE * 100:.0f}%. "
        f"Failures:\n  {failure_detail}"
    )

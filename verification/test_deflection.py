"""Eddington 1919 weak-field deflection at |b| >> rs.

PHYSICS.md §11 test matrix entry. Complements ``test_phase1_rays.py`` (which
matches the *exact* Schwarzschild integral at any b): here we verify the
textbook result that as the impact parameter grows, the deflection angle
asymptotes to the first-order Einstein expression

    alpha = 4 G M / (b c^2) = 2 r_s / |b|

which is what Eddington's 1919 solar eclipse expedition measured, what every
GR textbook opens with, and what a reader of the simulator's repo would
expect to see verified by name.

Two distinct checks:

1. **Absolute agreement** at |b| = 1000 M and 2000 M. At these impact
   parameters the second-order correction ``(15π/16)(rs/b)²`` is below 0.3%
   so Einstein's linear formula is within the 1% tolerance PHYSICS.md §11
   demands.

2. **1/b scaling law** across |b| in {100, 200, 400, 800} M. A bug that
   inflated or deflated ``rs`` uniformly would pass one absolute test but
   fail the scaling check (log-log slope -1 ± 5%). The second-order
   correction only perturbs the log-log slope by O((rs/b)²), which is
   below 10⁻³ in this range.
"""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

_M = 1.0
_RS = 2.0 * _M
# Large enough for 1% Einstein accuracy: (15π/32)(rs/b) < 0.01 → b > 94 rs.
_STRICT_B_VALUES_M = [1000.0, 2000.0]
# Moderate enough for the 1/b scaling fit to resolve cleanly without spending
# 100k RK4 steps per ray.
_SCALING_B_VALUES_M = [100.0, 200.0, 400.0, 800.0]
_EINSTEIN_TOLERANCE = 0.01
_SCALING_SLOPE_TOLERANCE = 0.05
# Keep only the last 25% of the outgoing trail for the slope fit — well
# clear of any residual near-BH curvature influence.
_ASYMPTOTIC_FRACTION = 0.75


def _run_2d_toy(
    cli: Path, tmp_path: Path, b_min: float, b_max: float, n_rays: int, max_steps: int
) -> Path:
    csv_path = tmp_path / f"rays_b_{b_min:+.1f}_{b_max:+.1f}.csv"
    result = subprocess.run(
        [
            str(cli),
            "--mode",
            "2d-toy",
            "--output",
            str(tmp_path / "unused.png"),
            "--dump-trails",
            str(csv_path),
            "--b-range",
            str(b_min),
            str(b_max),
            "--n-rays",
            str(n_rays),
            "--max-steps",
            str(max_steps),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"singularity_cli exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return csv_path


def _parse_trails(path: Path) -> dict[int, list[tuple[int, float, float, float]]]:
    by_ray: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            by_ray[int(row["ray_idx"])].append(
                (int(row["step"]), float(row["b"]), float(row["x"]), float(row["y"]))
            )
    return dict(by_ray)


def _asymptotic_tangent(points: list[tuple[int, float, float, float]]) -> float | None:
    trail = [(x, y) for (s, _b, x, y) in points if s >= 0]
    if len(trail) < 4:
        return None
    xs = np.array([x for (x, _) in trail])
    x_cutoff = xs.min() * _ASYMPTOTIC_FRACTION  # xs.min() is negative
    mask = xs < x_cutoff
    if mask.sum() < 2:
        return None
    xs_tail = xs[mask]
    ys_tail = np.array([y for (_, y) in trail])[mask]
    slope, _intercept = np.polyfit(xs_tail, ys_tail, 1)
    return float(np.arctan(slope))


def _deflection_at(cli: Path, tmp_path: Path, b_value_M: float) -> float:
    """Magnitude of the asymptotic deflection for rays fired at |b| = b_value_M.

    The CLI places rays at bin-midpoints of ``[b_min, b_max]``, so to target a
    single impact parameter we pinch the range into a 1% sliver around it and
    average the four rays it produces. Max-step budget scales linearly with b
    because the integrator has to traverse an x-range of ≈ 4|b| and uses a
    fixed affine step of 0.1.
    """
    eps = 0.01 * b_value_M
    # Steps needed: ray travels ~4|b| in coordinate distance at unit speed in
    # affine parameter, so 40|b| steps at h=0.1 plus 20% headroom for the
    # near-BH phase where coordinate velocity slows.
    max_steps = int(50 * b_value_M)
    csv_path = _run_2d_toy(
        cli,
        tmp_path,
        b_min=b_value_M - eps,
        b_max=b_value_M + eps,
        n_rays=4,
        max_steps=max_steps,
    )
    rays = _parse_trails(csv_path)
    alphas: list[float] = []
    for points in rays.values():
        if points[-1][0] != -2:
            continue  # escaped rays only; weak-field should have no captures
        alpha = _asymptotic_tangent(points)
        if alpha is not None:
            alphas.append(abs(alpha))
    assert len(alphas) >= 2, f"fewer than 2 usable rays near |b|={b_value_M}; CLI output broken?"
    return float(np.mean(alphas))


@pytest.mark.physics
@pytest.mark.parametrize("b_value_M", _STRICT_B_VALUES_M)
def test_weak_field_deflection_matches_einstein_1919(
    singularity_cli: Path, tmp_path: Path, b_value_M: float
) -> None:
    """At truly asymptotic |b|, measured deflection equals ``2 rs / |b|`` to 1%.

    At |b| = 1000 M the second-order correction is under 0.3%, leaving
    ample margin for RK4 step error and float32 drift."""
    measured = _deflection_at(singularity_cli, tmp_path, b_value_M)
    predicted = 2.0 * _RS / b_value_M
    rel_err = abs(measured - predicted) / predicted
    assert rel_err <= _EINSTEIN_TOLERANCE, (
        f"|b|={b_value_M} M: |alpha|_measured={measured:.6f} "
        f"vs Einstein {predicted:.6f} (rel err {rel_err * 100:.2f}%)"
    )


@pytest.mark.physics
def test_weak_field_deflection_scales_as_inverse_b(singularity_cli: Path, tmp_path: Path) -> None:
    """``alpha ∝ 1/b`` as predicted by the Einstein formula. Log-log slope
    across four b values must equal -1 within 5%.

    A bug that got the magnitude right at one b but wrong elsewhere (e.g. a
    rs² instead of rs factor) would fail the slope test even if it managed
    to hit a single absolute point. The scaling law is therefore an
    independent check on the ``2rs/b`` form of the Einstein formula."""
    bs = np.array(_SCALING_B_VALUES_M)
    alphas = np.array([_deflection_at(singularity_cli, tmp_path, float(b)) for b in bs])
    slope, _intercept = np.polyfit(np.log(bs), np.log(alphas), 1)
    assert abs(slope + 1.0) < _SCALING_SLOPE_TOLERANCE, (
        f"weak-field deflection scales as b^{slope:.3f}, expected b^-1 within "
        f"{_SCALING_SLOPE_TOLERANCE * 100:.0f}% (measurements: "
        f"{list(zip(bs.tolist(), alphas.tolist(), strict=True))})"
    )

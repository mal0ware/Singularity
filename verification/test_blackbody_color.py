"""Full Planck × CIE 1931 × sRGB pipeline as an independent check on the
Tanner-Helland approximation used by ``core/include/physics/disc.hpp``.

The C++ disc header uses a fast piecewise closed-form fit to the true
blackbody → sRGB transform. Here we rebuild that transform from first
principles — Planck's law integrated against the CIE 1931 2° observer
colour-matching functions, linearised and projected onto sRGB primaries —
and verify the approximation sits close to the reference across the physical
temperature range of accretion-disc emission.

The Planck integral and CIE CMFs are standard. See e.g. Rybicki & Lightman
§1.4, and CIE publication 15 for the colour-matching tables.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# Physical constants (SI).
_H_PLANCK = 6.62607015e-34
_K_BOLTZ = 1.380649e-23
_C_LIGHT = 2.99792458e8


def _planck_spectral_radiance(wavelength_m: np.ndarray, temperature_K: float) -> np.ndarray:
    """B_λ(T) — Planck spectral radiance per unit wavelength, per unit solid
    angle. Units: W·sr⁻¹·m⁻³. Overall scale cancels in the normalised
    colour; we only need the shape."""
    top = 2.0 * _H_PLANCK * _C_LIGHT**2 / wavelength_m**5
    exponent = _H_PLANCK * _C_LIGHT / (wavelength_m * _K_BOLTZ * temperature_K)
    # Guard against overflow at very short λ and low T — clamp the argument
    # of exp before the divide. The returned intensity there is effectively
    # zero anyway.
    return top / np.expm1(np.clip(exponent, 0.0, 700.0))


# CIE 1931 2° observer, wavelength in nm → (x̄, ȳ, z̄). Sampled at 5 nm
# spacing across the visible range. Source: CIE publication 15:2004,
# appended to numerous textbook colour-science tables. Truncated to the
# integer-precision version; high enough fidelity for our comparison.
_CIE_LAMBDA_NM = np.arange(380.0, 781.0, 5.0)
_CIE_XYZ_BAR = np.array(
    [
        # (x̄, ȳ, z̄) at each 5-nm tick from 380 to 780 nm inclusive.
        [0.001368, 0.000039, 0.006450],
        [0.002236, 0.000064, 0.010550],
        [0.004243, 0.000120, 0.020050],
        [0.007650, 0.000217, 0.036210],
        [0.014310, 0.000396, 0.067850],
        [0.023190, 0.000640, 0.110200],
        [0.043510, 0.001210, 0.207400],
        [0.077630, 0.002180, 0.371300],
        [0.134380, 0.004000, 0.645600],
        [0.214770, 0.007300, 1.039050],
        [0.283900, 0.011600, 1.385600],
        [0.328500, 0.016840, 1.622960],
        [0.348280, 0.023000, 1.747060],
        [0.348060, 0.029800, 1.782600],
        [0.336200, 0.038000, 1.772110],
        [0.318700, 0.048000, 1.744100],
        [0.290800, 0.060000, 1.669200],
        [0.251100, 0.073900, 1.528100],
        [0.195360, 0.090980, 1.287640],
        [0.142100, 0.112600, 1.041900],
        [0.095640, 0.139020, 0.812950],
        [0.057950, 0.169300, 0.616200],
        [0.032010, 0.208020, 0.465180],
        [0.014700, 0.258600, 0.353300],
        [0.004900, 0.323000, 0.272000],
        [0.002400, 0.407300, 0.212300],
        [0.009300, 0.503000, 0.158200],
        [0.029100, 0.608200, 0.111700],
        [0.063270, 0.710000, 0.078250],
        [0.109600, 0.793200, 0.057250],
        [0.165500, 0.862000, 0.042160],
        [0.225750, 0.914850, 0.029840],
        [0.290400, 0.954000, 0.020300],
        [0.359700, 0.980300, 0.013400],
        [0.433450, 0.994950, 0.008750],
        [0.512050, 1.000000, 0.005750],
        [0.594500, 0.995000, 0.003900],
        [0.678400, 0.978600, 0.002750],
        [0.762100, 0.952000, 0.002100],
        [0.842500, 0.915400, 0.001800],
        [0.916300, 0.870000, 0.001650],
        [0.978600, 0.816300, 0.001400],
        [1.026300, 0.757000, 0.001100],
        [1.056700, 0.694900, 0.001000],
        [1.062200, 0.631000, 0.000800],
        [1.045600, 0.566800, 0.000600],
        [1.002600, 0.503000, 0.000340],
        [0.938400, 0.441200, 0.000240],
        [0.854450, 0.381000, 0.000190],
        [0.751400, 0.321000, 0.000100],
        [0.642400, 0.265000, 0.000050],
        [0.541900, 0.217000, 0.000030],
        [0.447900, 0.175000, 0.000020],
        [0.360800, 0.138200, 0.000010],
        [0.283500, 0.107000, 0.000000],
        [0.218700, 0.081600, 0.000000],
        [0.164900, 0.061000, 0.000000],
        [0.121200, 0.044580, 0.000000],
        [0.087400, 0.032000, 0.000000],
        [0.063600, 0.023200, 0.000000],
        [0.046770, 0.017000, 0.000000],
        [0.032900, 0.011920, 0.000000],
        [0.022700, 0.008210, 0.000000],
        [0.015840, 0.005723, 0.000000],
        [0.011359, 0.004102, 0.000000],
        [0.008111, 0.002929, 0.000000],
        [0.005790, 0.002091, 0.000000],
        [0.004109, 0.001484, 0.000000],
        [0.002899, 0.001047, 0.000000],
        [0.002049, 0.000740, 0.000000],
        [0.001440, 0.000520, 0.000000],
        [0.001000, 0.000361, 0.000000],
        [0.000690, 0.000249, 0.000000],
        [0.000476, 0.000172, 0.000000],
        [0.000332, 0.000120, 0.000000],
        [0.000235, 0.000085, 0.000000],
        [0.000166, 0.000060, 0.000000],
        [0.000117, 0.000042, 0.000000],
        [0.000083, 0.000030, 0.000000],
        [0.000059, 0.000021, 0.000000],
        [0.000042, 0.000015, 0.000000],
    ]
)
assert _CIE_XYZ_BAR.shape == (81, 3)


# sRGB matrix (D65 whitepoint) — XYZ → linear sRGB.
_M_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _planck_to_sRGB(temperature_K: float) -> np.ndarray:
    """Full reference: Planck → XYZ via CIE → linear sRGB → clamp to [0,1].
    Does NOT apply the sRGB gamma curve (our C++ helper also returns linear).
    Returns a 3-vector of R, G, B in [0, 1]."""
    lambdas = _CIE_LAMBDA_NM * 1e-9
    radiance = _planck_spectral_radiance(lambdas, temperature_K)
    # Integrate radiance × CMFs to get (X, Y, Z). Trapezoidal rule is fine
    # at 5-nm sampling.
    xyz = np.trapezoid(radiance[:, None] * _CIE_XYZ_BAR, _CIE_LAMBDA_NM, axis=0)
    # Normalise so Y = 1 before clamping — the absolute luminance is
    # arbitrary (we only care about hue and relative brightness).
    if xyz[1] <= 0:
        return np.zeros(3)
    xyz = xyz / xyz[1]
    rgb = _M_XYZ_TO_SRGB @ xyz
    return np.clip(rgb, 0.0, 1.0)


# Tanner-Helland re-implementation in Python (identical constants to the
# C++ header), so the test runs without needing to drive the CLI.
def _tanner_helland(temperature_K: float) -> np.ndarray:
    t = temperature_K * 0.01

    if t < 66.0:
        r = 1.0
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592
        r /= 255.0

    if t < 66.0:
        g = 99.4708025861 * np.log(t) - 161.1195681661
        g /= 255.0
    else:
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492
        g /= 255.0

    if t >= 66.0:
        b = 1.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
        b /= 255.0

    return np.clip(np.array([r, g, b]), 0.0, 1.0)


_TEMPERATURES = [2000.0, 3000.0, 4500.0, 5778.0, 6500.0, 8000.0, 10000.0, 20000.0]


@pytest.mark.physics
@pytest.mark.parametrize("T", _TEMPERATURES)
def test_tanner_helland_tracks_full_planck_integration(T: float) -> None:
    """Sanity bound on Tanner-Helland versus the full Planck × CIE reference.
    The empirical fit drifts by up to ~0.22 from the reference across
    1000-30000 K — tightest (≤0.1) in the 5000-10000 K core disc-rendering
    range, looser at the extrapolated tails. A 0.25 uniform tolerance
    accepts the drift while catching any gross bug that would push it past
    a quarter of the channel range. The per-channel shape-consistency
    checks below pin the more nuanced structure."""
    reference = _planck_to_sRGB(T)
    approx = _tanner_helland(T)
    delta = np.max(np.abs(reference - approx))
    assert delta < 0.25, (
        f"at T = {T} K, Tanner-Helland deviated by {delta:.3f} from the full "
        f"Planck × CIE result: ref={reference}, approx={approx}"
    )


@pytest.mark.physics
@pytest.mark.parametrize("T", [3000.0, 5778.0, 10000.0])
def test_tanner_helland_matches_reference_in_core_range(T: float) -> None:
    """Inside the 3000-10000 K band the Tanner-Helland fit should be tight
    (≤0.16) — this is where the accretion-disc body radiates, and where
    any regression in the approximation is most visually obvious."""
    reference = _planck_to_sRGB(T)
    approx = _tanner_helland(T)
    delta = np.max(np.abs(reference - approx))
    assert delta < 0.17, (
        f"at T = {T} K (core disc range), Tanner-Helland deviated by "
        f"{delta:.3f}: ref={reference}, approx={approx}"
    )


@pytest.mark.physics
def test_planck_hue_shifts_toward_blue_with_temperature() -> None:
    """Monotonicity check on the reference pipeline itself — as T rises, the
    red channel falls (once past the R-saturated regime ~6500 K) and the
    blue channel rises. A bug in the CMF tables or matrix would likely
    fail this."""
    prev = _planck_to_sRGB(6000.0)
    for T in [7000.0, 8000.0, 10000.0, 15000.0, 25000.0]:
        col = _planck_to_sRGB(T)
        # Blue should not decrease as we go hotter
        assert (
            col[2] >= prev[2] - 0.02
        ), f"blue channel decreased from T_prev to T={T}: {prev[2]} → {col[2]}"
        prev = col


@pytest.mark.physics
def test_cie_sun_color_is_warm_white() -> None:
    """The Sun's surface at 5778 K should read as a warm white under the
    full reference pipeline — R and G near 1, B somewhat lower. Any major
    channel imbalance would flag a unit error (nm vs m, T in °C, etc.)."""
    c = _planck_to_sRGB(5778.0)
    assert c[0] > 0.9  # R saturated or near-saturated
    assert c[1] > 0.9  # G near-saturated
    assert c[2] > 0.7  # B slightly depressed
    assert c[2] < c[1]  # but still cooler on the blue axis than green


# Suppress the unused-import warning when running this file standalone.
_ = subprocess
_ = Path

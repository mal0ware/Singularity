// core/include/physics/kerr.hpp
//
// Host-side analytic scalars for a rotating (Kerr) black hole. This header
// holds the closed-form geometric quantities used everywhere the renderer
// needs to reason about Kerr in coordinates rather than integrate a geodesic —
// scientific overlays, ISCO-aware disc inner edge, ergosphere wireframe,
// scene validation. See docs/PHYSICS.md §7 for the metric and derivations.
//
// The integrable-system form of the geodesic equation used by the GPU kernels
// (PHYSICS.md §7.3) lives alongside the Schwarzschild RHS in
// shared_shader/geodesic_math.h — its derivation exploits the four conserved
// quantities (E, L_z, Q, m²) and therefore never touches Christoffel symbols
// directly, which is why this file does not enumerate them.
//
// Conventions match PHYSICS.md: geometrized units (G = c = 1); Boyer-Lindquist
// coordinates (t, r, θ, φ); spin parameter `a = J/M` constrained by
// 0 ≤ a ≤ M. Sub-extremal Kerr only — the Cauchy horizon is skipped for spins
// at or above extremal since `M² - a² < 0` stops being physically meaningful.

#ifndef SINGULARITY_PHYSICS_KERR_HPP
#define SINGULARITY_PHYSICS_KERR_HPP

#include <cmath>

namespace singularity::physics {

// Outer event horizon. PHYSICS.md §7.1.
//   r₊ = M + √(M² - a²)
// For |a| ≥ M the horizon degenerates (extremal) or disappears (hypothetical
// naked singularity); we clamp the radicand at zero so the function returns
// M at exactly extremal spin without producing NaN.
//
// Not constexpr: std::sqrt is only constexpr from C++26 (P1383). libstdc++
// and libc++ expose it constexpr as an extension, but MSVC under /permissive-
// enforces the standard strictly.
inline float kerr_outer_horizon(float mass_geom, float spin) {
    const float disc = mass_geom * mass_geom - spin * spin;
    return mass_geom + std::sqrt(disc > 0.0f ? disc : 0.0f);
}

// Inner (Cauchy) horizon. PHYSICS.md §7.1.
//   r₋ = M - √(M² - a²)
// Goes to M at extremal and coincides with r₊ there.
inline float kerr_inner_horizon(float mass_geom, float spin) {
    const float disc = mass_geom * mass_geom - spin * spin;
    return mass_geom - std::sqrt(disc > 0.0f ? disc : 0.0f);
}

// Static-limit / ergosphere outer boundary at polar angle θ. PHYSICS.md §7.1.
//   r_ergo(θ) = M + √(M² - a² cos²θ)
// Equals r₊ at the poles and 2M at the equator (for any spin), enclosing the
// horizon on a prolate-spheroid-shaped surface. Inside, g_tt > 0 and no
// stationary observer can exist.
inline float kerr_ergosphere_outer(float mass_geom, float spin, float theta) {
    const float c = std::cos(theta);
    const float disc = mass_geom * mass_geom - spin * spin * c * c;
    return mass_geom + std::sqrt(disc > 0.0f ? disc : 0.0f);
}

// Prograde photon-sphere radius. PHYSICS.md §5.2 (Schwarzschild case) extended
// via Bardeen's closed form for Kerr (cited Carroll §6.7):
//   r_ph_± = 2M (1 + cos((2/3) arccos(∓ a/M)))
// Upper sign (∓ = −) is the prograde branch (co-rotating photon orbit).
// Recovers 3M at a = 0 and M at a = M.
inline float kerr_photon_sphere_prograde(float mass_geom, float spin) {
    const float x = -spin / mass_geom;
    return 2.0f * mass_geom * (1.0f + std::cos((2.0f / 3.0f) * std::acos(x)));
}

// Retrograde photon-sphere radius (counter-rotating photon orbit).
//   r_ph_− = 2M (1 + cos((2/3) arccos(+ a/M)))
// Recovers 3M at a = 0 and 4M at a = M.
inline float kerr_photon_sphere_retrograde(float mass_geom, float spin) {
    const float x = spin / mass_geom;
    return 2.0f * mass_geom * (1.0f + std::cos((2.0f / 3.0f) * std::acos(x)));
}

// Innermost Stable Circular Orbit for matter. PHYSICS.md §7.2.
//   Z₁ = 1 + (1 − a²/M²)^(1/3) [(1 + a/M)^(1/3) + (1 − a/M)^(1/3)]
//   Z₂ = √(3 a²/M² + Z₁²)
//   r_ISCO = M [3 + Z₂ ∓ √((3 − Z₁)(3 + Z₁ + 2 Z₂))]
// Upper sign (∓ = −) is prograde, lower is retrograde.
// Checks: a = 0 → 6M (Schwarzschild). a = M prograde → M. a = M retrograde → 9M.
namespace detail {
inline float kerr_isco(float mass_geom, float spin, bool prograde) {
    const float a_over_M = spin / mass_geom;
    const float a2 = a_over_M * a_over_M;
    // cbrt is not constexpr in C++20 but is in the standard library; std::cbrt
    // handles negative arguments correctly (unlike pow(x, 1/3) for x < 0).
    const float one_minus_a2 = 1.0f - a2;
    const float z1 =
        1.0f + std::cbrt(one_minus_a2) * (std::cbrt(1.0f + a_over_M) + std::cbrt(1.0f - a_over_M));
    const float z2 = std::sqrt(3.0f * a2 + z1 * z1);
    const float radicand = (3.0f - z1) * (3.0f + z1 + 2.0f * z2);
    // radicand >= 0 throughout 0 ≤ a ≤ M; guard against float noise at the
    // extremal boundary where it can dip by 10⁻⁷ or so.
    const float root = std::sqrt(radicand > 0.0f ? radicand : 0.0f);
    return mass_geom * (3.0f + z2 + (prograde ? -root : +root));
}
}  // namespace detail

inline float kerr_isco_prograde(float mass_geom, float spin) {
    return detail::kerr_isco(mass_geom, spin, /*prograde=*/true);
}

inline float kerr_isco_retrograde(float mass_geom, float spin) {
    return detail::kerr_isco(mass_geom, spin, /*prograde=*/false);
}

// Boyer-Lindquist ``Σ`` and ``Δ`` — the two geometric quantities the Kerr
// metric and every conserved-quantity expression are written in terms of.
// PHYSICS.md §7.
//   Σ = r² + a² cos²θ
//   Δ = r² − r_s r + a² = r² − 2 M r + a²
// Δ(r) = 0 exactly at the horizons; Σ > 0 everywhere off the ring
// singularity and is the quantity that reduces to r² in Schwarzschild limit.
inline float kerr_sigma(float mass_geom, float spin, float r, float theta) {
    (void)mass_geom;  // mass parameter drops out of Σ by the BL convention.
    const float c = std::cos(theta);
    return r * r + spin * spin * c * c;
}

inline float kerr_delta(float mass_geom, float spin, float r) {
    return r * r - 2.0f * mass_geom * r + spin * spin;
}

}  // namespace singularity::physics

#endif  // SINGULARITY_PHYSICS_KERR_HPP

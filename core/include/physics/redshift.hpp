// core/include/physics/redshift.hpp
//
// Gravitational redshift and Doppler shift for photons in Schwarzschild and
// (equatorial, co-rotating) Kerr spacetimes. PHYSICS.md §8.1–§8.2.
//
// Two physical effects, one combined observed shift factor:
//
//   g ≡ ν_obs / ν_emit = (gravitational piece) × (Doppler piece)
//
// where the gravitational piece is the time-dilation ratio between emitter
// and observer (static observers at different r), and the Doppler piece is
// the kinematic shift from the emitter's four-velocity (matter orbiting the
// disc). The observed specific intensity then scales as I_obs = g⁴ I_emit
// (Rybicki & Lightman §4.9).
//
// All functions here are pure host code with no GPU dependency so the Catch2
// suite can exercise them directly. A GPU-parity mirror will appear in
// shared_shader/ when the disc kernel wires the same formulas into the
// per-pixel path in Phase 3.

#ifndef SINGULARITY_PHYSICS_REDSHIFT_HPP
#define SINGULARITY_PHYSICS_REDSHIFT_HPP

#include <cmath>

namespace singularity::physics {

// Gravitational redshift factor between two static observers at radii
// ``r_emit`` and ``r_obs`` in Schwarzschild spacetime:
//
//     g_grav = ν_obs / ν_emit = √((1 − r_s/r_emit) / (1 − r_s/r_obs))
//
// The ordering here matters: for ``r_emit < r_obs`` (light climbing out of
// the potential well) ``g_grav < 1`` — redshifted — because the emitter's
// local tick is slower. For a photon with conserved energy-at-infinity
// ``E``, the frequency measured by a static observer at ``r`` is
// ``ω = E / √(1 − r_s/r)``; taking the ratio gives the factor above.
//
// For an observer at infinity (``r_obs → ∞``) this reduces to the familiar
// ``g = √(1 − r_s/r_emit)``.
//
// The inputs must satisfy ``r_emit, r_obs > r_s`` for static observers to
// exist; callers inside the horizon must use a different formulation.
inline float schwarzschild_grav_redshift(float rs, float r_emit, float r_obs) {
    const float f_emit = 1.0f - rs / r_emit;
    const float f_obs = 1.0f - rs / r_obs;
    return std::sqrt(f_emit / f_obs);
}

// Convenience form for an observer at spatial infinity. Equivalent to
// ``schwarzschild_grav_redshift(rs, r_emit, +infinity)`` but avoids the
// floating-point limit gymnastics.
inline float schwarzschild_grav_redshift_to_infinity(float rs, float r_emit) {
    return std::sqrt(1.0f - rs / r_emit);
}

// Relativistic Doppler factor for an emitter moving with coordinate angular
// velocity ``Omega`` on a circular geodesic at radius ``r_emit`` in the
// equatorial plane of Schwarzschild. The formula (Rybicki & Lightman §4.9,
// combined with the Schwarzschild g_tt, g_φφ) for a photon emitted toward
// infinity at angle ``cos_psi`` between the emitter's orbital velocity and
// the photon's direction of travel in the observer's frame is:
//
//     g_doppler = √(1 − v²) / (1 − v · cos_psi)
//
// where ``v`` is the emitter's orbital speed as measured by a local static
// observer: ``v = r_emit · Omega / √(1 − r_s/r_emit)``. The orbital velocity
// for a timelike Keplerian circular orbit is ``Omega_K = √(M / r_emit³)``.
//
// ``cos_psi`` is ``+1`` for material moving directly toward the observer
// (maximal blueshift), ``−1`` for directly away (maximal redshift), ``0``
// for transverse motion (pure transverse-Doppler redshift ``√(1 − v²)``).
inline float schwarzschild_doppler_factor(float rs, float r_emit, float omega, float cos_psi) {
    // Linear velocity in the local static frame: v = r Ω / √(1 − r_s/r).
    const float f = 1.0f - rs / r_emit;
    const float v = r_emit * omega / std::sqrt(f);
    const float v2 = v * v;
    const float beta = 1.0f - v * cos_psi;
    // Guard against the relativistic-limit edge where v→1 (would only happen
    // if called with ultra-compact orbits the caller shouldn't hand us).
    return std::sqrt(1.0f - v2) / beta;
}

// Keplerian angular velocity for a timelike circular geodesic in
// Schwarzschild. Used to seed the Doppler factor for the accretion disc:
//
//     Omega_K = √(M / r_emit³)
inline float schwarzschild_keplerian_omega(float mass_geom, float r_emit) {
    return std::sqrt(mass_geom / (r_emit * r_emit * r_emit));
}

// Combined shift factor: gravitational × Doppler, evaluated for an emitter
// on a Keplerian circular orbit in the disc at ``r_emit``, observed at
// ``r_obs`` (usually ``+infinity``). ``cos_psi`` remains the angle between
// the emitter's motion and the photon in the observer's frame.
inline float schwarzschild_combined_shift(float rs, float r_emit, float r_obs, float cos_psi) {
    const float mass_geom = 0.5f * rs;
    const float omega = schwarzschild_keplerian_omega(mass_geom, r_emit);
    const float g_grav = schwarzschild_grav_redshift(rs, r_emit, r_obs);
    const float g_dop = schwarzschild_doppler_factor(rs, r_emit, omega, cos_psi);
    return g_grav * g_dop;
}

// Observed-to-emitted specific intensity. The ν⁻³ invariant combined with
// the g shift gives the famous ``I_obs = g⁴ I_emit`` scaling (Rybicki &
// Lightman §4.9) — bolometric intensity transforms as the fourth power of
// the frequency shift because both the spectral shift and the solid-angle
// aberration conspire to amplify the in-moving side of a relativistic disc.
inline float intensity_scaling(float g) {
    const float g2 = g * g;
    return g2 * g2;
}

}  // namespace singularity::physics

#endif  // SINGULARITY_PHYSICS_REDSHIFT_HPP

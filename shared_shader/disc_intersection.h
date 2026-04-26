// shared_shader/disc_intersection.h
//
// Portable equatorial-plane (θ = π/2) crossing detector for null geodesics
// that are stepped discretely by the RK4 (Schwarzschild) or Hamiltonian
// (Kerr) integrator. Compiles in MSL / HLSL / CUDA / host C++ via the
// macros in shader_compat.h so every backend uses the same interpolation
// rule. See docs/ARCHITECTURE.md §4 (shader sharing) and docs/PHYSICS.md
// §8 (thin-disc model).
//
// The crossing test is: at two consecutive integrator samples the sign of
// (θ − π/2) flips. When it does, linear interpolation gives the fraction
// of the step at which the ray pierced the equatorial plane; the caller
// then interpolates any scalar quantity it cares about (radius, Boyer–
// Lindquist azimuthal velocity, etc.) through the same fraction. Linear
// interpolation is correct to O(h²) which matches the RK4 step error at
// the step sizes the renderer uses.

#ifndef SINGULARITY_SHARED_SHADER_DISC_INTERSECTION_H
#define SINGULARITY_SHARED_SHADER_DISC_INTERSECTION_H

#include "shader_compat.h"

struct DiscCrossing {
    bool crossed;  // true if θ crossed π/2 between prev and curr
    float frac;    // interp fraction in [0,1] from prev → curr; undefined if !crossed
};

// Detect θ = π/2 crossings. Uses the sign product so equality at either
// endpoint doesn't generate a spurious crossing.
DEVICE INLINE DiscCrossing detect_equatorial_crossing(float theta_prev, float theta_curr) {
    // Canonical value of π/2 at float precision; avoids pulling kPi in at the
    // call site and keeps the header dependency-free.
    const float mid = 1.5707963267948966f;
    const float dp = theta_prev - mid;
    const float dc = theta_curr - mid;
    DiscCrossing result;
    result.crossed = (dp * dc) < 0.0f;
    // Guard the denominator — if both samples are on the same side, frac is
    // meaningless anyway; when they straddle, (theta_curr - theta_prev) is
    // strictly non-zero because (dp * dc < 0) forces opposite signs.
    const float denom = theta_curr - theta_prev;
    result.frac = result.crossed ? (mid - theta_prev) / denom : 0.0f;
    return result;
}

DEVICE INLINE float lerp_scalar(float prev, float curr, float frac) {
    return prev + frac * (curr - prev);
}

// Thin-disc annulus test: the ray hits the visible disc iff its crossing
// radius falls inside [r_inner, r_outer]. Inclusive on both endpoints so
// crossings at the ISCO and at the outer rim count.
DEVICE INLINE bool in_disc_annulus(float r_cross, float r_inner, float r_outer) {
    return (r_cross >= r_inner) && (r_cross <= r_outer);
}

#endif  // SINGULARITY_SHARED_SHADER_DISC_INTERSECTION_H

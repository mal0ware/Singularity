// shared_shader/kerr_math.h
//
// Kerr null-geodesic integration in the conserved-quantity (``E, L_z, Q``)
// form. PHYSICS.md §7.3.
//
// The rotating Kerr metric admits four Killing-style conserved quantities
// for a geodesic (``E, L_z, Q, m²``), so the four-component coordinate state
// ``(t, r, θ, φ)`` obeys ODEs whose right-hand sides are algebraic in
// ``(r, θ)`` and those four constants. We therefore never write down the 23
// non-zero Kerr Christoffel symbols — the explicit-Christoffel form is
// notoriously ill-conditioned near the inner horizon and about 3× slower
// on the GPU than this formulation (JMO 2015 §3).
//
// For photons (``m² = 0``) the conserved-quantity radial and polar
// "potentials" are:
//
//     P(r)   = E (r² + a²) − a L_z
//     R(r)   = P(r)² − Δ [(L_z − a E)² + Q]            [Σ² (dr/dλ)²]
//     Θ(θ)   = Q − cos²θ [L_z²/sin²θ − a² E²]          [Σ² (dθ/dλ)²]
//     Σ(r,θ) = r² + a² cos²θ
//     Δ(r)   = r² − 2 M r + a²
//
// and the geodesic equations become
//
//     Σ dt/dλ = (r² + a²) P(r)/Δ + a (L_z − a E sin²θ)
//     Σ dr/dλ = σ_r √R(r)
//     Σ dθ/dλ = σ_θ √Θ(θ)
//     Σ dφ/dλ = a P(r)/Δ + L_z/sin²θ − a E
//
// with ``σ_r, σ_θ ∈ {±1}`` tracking the direction of radial and polar motion.
// These signs flip at turning points where ``R`` or ``Θ`` cross zero; the
// stepper (in a separate header — this file just hosts the algebra) detects
// sign changes and flips accordingly. Between flips the RHS is smooth.
//
// The ``DEVICE INLINE`` macros from ``shader_compat.h`` make this header
// compile as MSL, HLSL, CUDA C++, and host C++ — the same algebra lights up
// every backend's Kerr kernel.

#ifndef SINGULARITY_SHARED_SHADER_KERR_MATH_H
#define SINGULARITY_SHARED_SHADER_KERR_MATH_H

#include "shader_compat.h"

// Constants characterising a null geodesic in Kerr. Held in a POD struct so
// the same bytes can cross the host/device boundary as a uniform upload.
struct KerrConserved {
    float E;    // energy at infinity (conserved)
    float L_z;  // axial angular momentum (conserved)
    float Q;    // Carter constant (conserved)
    float a;    // spin parameter, 0 ≤ a ≤ M
    float M;    // gravitational mass (geometrized units: horizon at M + √(M² − a²))
};

// 4-coordinate Kerr state: Boyer-Lindquist coordinates. Signs of the radial
// and polar branches live outside this struct so a single RK4 sub-stage can
// advance ``(t, r, θ, φ)`` smoothly while the surrounding stepper handles the
// branch flips separately.
struct KerrCoord {
    float t, r, theta, phi;
};

DEVICE INLINE KerrCoord kerr_coord_add(KerrCoord a, KerrCoord b) {
    KerrCoord r;
    r.t = a.t + b.t;
    r.r = a.r + b.r;
    r.theta = a.theta + b.theta;
    r.phi = a.phi + b.phi;
    return r;
}

DEVICE INLINE KerrCoord kerr_coord_scale(KerrCoord s, float k) {
    KerrCoord r;
    r.t = s.t * k;
    r.r = s.r * k;
    r.theta = s.theta * k;
    r.phi = s.phi * k;
    return r;
}

// --- Geometric scalars -----------------------------------------------------

DEVICE INLINE float kerr_Sigma(float r, float theta, float a) {
    float c = cos(theta);
    return r * r + a * a * c * c;
}

DEVICE INLINE float kerr_Delta(float r, float a, float M) {
    return r * r - 2.0f * M * r + a * a;
}

DEVICE INLINE float kerr_P(float r, float a, float E, float L_z) {
    return E * (r * r + a * a) - a * L_z;
}

// Radial "potential" — R(r) = P(r)² − Δ (L_z − a E)² − Δ Q (for photons, m=0).
DEVICE INLINE float kerr_R(float r, KerrConserved c) {
    float P = kerr_P(r, c.a, c.E, c.L_z);
    float Delta = kerr_Delta(r, c.a, c.M);
    float aE_Lz = c.L_z - c.a * c.E;
    return P * P - Delta * (aE_Lz * aE_Lz + c.Q);
}

// Polar "potential" — Θ(θ) = Q − cos²θ [L_z²/sin²θ − a² E²] for null geodesics.
// Near the poles sin²θ → 0 and the formula would spike; callers with L_z = 0
// can reach the pole safely (the L_z²/sin²θ term vanishes); callers with
// L_z ≠ 0 have a polar turning point Θ = 0 before sin²θ gets small.
DEVICE INLINE float kerr_Theta(float theta, KerrConserved c) {
    float s = sin(theta);
    float cth = cos(theta);
    float s2 = s * s;
    float c2 = cth * cth;
    // Avoid division-by-zero; caller should guarantee this isn't used at
    // exact poles with nonzero L_z.
    float safe_sin2 = (s2 < 1e-12f) ? 1e-12f : s2;
    return c.Q - c2 * (c.L_z * c.L_z / safe_sin2 - c.a * c.a * c.E * c.E);
}

// --- Full RHS (sign-tracking form) ----------------------------------------
//
// ``sigma_r`` and ``sigma_theta`` pick the branch of the square root for dr/dλ
// and dθ/dλ. We clamp the radicands at zero so numerical overshoot past a
// turning point does not produce NaN; the stepper must still detect that
// flip and invert the corresponding sign before the next step, or the
// integration will stall at the turning point.

DEVICE INLINE KerrCoord kerr_geodesic_rhs(KerrCoord state,
                                          KerrConserved c,
                                          float sigma_r,
                                          float sigma_theta) {
    float Sigma = kerr_Sigma(state.r, state.theta, c.a);
    float Delta = kerr_Delta(state.r, c.a, c.M);
    float P = kerr_P(state.r, c.a, c.E, c.L_z);

    float R_val = kerr_R(state.r, c);
    float T_val = kerr_Theta(state.theta, c);
    R_val = R_val > 0.0f ? R_val : 0.0f;
    T_val = T_val > 0.0f ? T_val : 0.0f;

    float s = sin(state.theta);
    float s2 = s * s;
    float safe_s2 = (s2 < 1e-12f) ? 1e-12f : s2;

    KerrCoord d;
    // dt/dλ and dφ/dλ use the closed algebraic form — no sign ambiguity.
    d.t = ((state.r * state.r + c.a * c.a) * P / Delta + c.a * (c.L_z - c.a * c.E * s2)) / Sigma;
    d.phi = (c.a * P / Delta + c.L_z / safe_s2 - c.a * c.E) / Sigma;
    // dr/dλ and dθ/dλ carry the ± branch through the sign vars.
    d.r = sigma_r * sqrt(R_val) / Sigma;
    d.theta = sigma_theta * sqrt(T_val) / Sigma;
    return d;
}

// --- RK4 on the 4-coordinate state ----------------------------------------
//
// Classical fixed-step RK4 advancing ``(t, r, θ, φ)`` through affine
// parameter ``h``. ``sigma_r`` and ``sigma_theta`` are held constant through
// the four stages — they flip at turning points, which is a step-level
// concern handled by ``kerr_rk4_step_tracking`` below.
DEVICE INLINE KerrCoord
kerr_rk4_coord_step(KerrCoord y, float h, KerrConserved c, float sigma_r, float sigma_theta) {
    KerrCoord k1 = kerr_geodesic_rhs(y, c, sigma_r, sigma_theta);
    KerrCoord k2 = kerr_geodesic_rhs(
        kerr_coord_add(y, kerr_coord_scale(k1, 0.5f * h)), c, sigma_r, sigma_theta);
    KerrCoord k3 = kerr_geodesic_rhs(
        kerr_coord_add(y, kerr_coord_scale(k2, 0.5f * h)), c, sigma_r, sigma_theta);
    KerrCoord k4 =
        kerr_geodesic_rhs(kerr_coord_add(y, kerr_coord_scale(k3, h)), c, sigma_r, sigma_theta);
    KerrCoord sum = kerr_coord_add(kerr_coord_add(k1, kerr_coord_scale(k2, 2.0f)),
                                   kerr_coord_add(kerr_coord_scale(k3, 2.0f), k4));
    return kerr_coord_add(y, kerr_coord_scale(sum, h / 6.0f));
}

// Bundled state for the turning-point-tracking stepper.
struct KerrStepperState {
    KerrCoord coord;
    float sigma_r;
    float sigma_theta;
};

// Full RK4 step that also detects turning points and flips the radial /
// polar signs when ``R`` or ``Θ`` have gone negative at the new point. This
// is the simple (non-bracketed) flavour: close to a turning point the RHS
// clamps the radicand at zero so the stepper stalls there for one or two
// iterations before the sign flips. For typical escape trajectories this
// costs a negligible amount of affine-parameter resolution.
//
// Known limitation: rays whose impact parameter sits near-critical (``|b|``
// a few percent above ``b_crit``) can wrap multiple times right at the
// photon sphere and get trapped by the stall — counted in diagnostic
// scripts as failures to escape inside the default step budget. Production
// Phase-6 rendering will move to the Hamiltonian (canonical-momenta) form
// which smooths through turning points without sign tracking; this simple
// form is enough for unit tests and the 2D toy visualiser.
DEVICE INLINE KerrStepperState kerr_rk4_step_tracking(KerrStepperState state,
                                                      float h,
                                                      KerrConserved c) {
    KerrStepperState next;
    next.coord = kerr_rk4_coord_step(state.coord, h, c, state.sigma_r, state.sigma_theta);
    next.sigma_r = state.sigma_r;
    next.sigma_theta = state.sigma_theta;
    // A negative R or Θ at the new coordinate is the signal the step carried
    // us past a turning point. Flip the offending sign; the RHS clamp at
    // zero prevents NaN during the step itself.
    if (kerr_R(next.coord.r, c) < 0.0f) {
        next.sigma_r = -state.sigma_r;
    }
    if (kerr_Theta(next.coord.theta, c) < 0.0f) {
        next.sigma_theta = -state.sigma_theta;
    }
    return next;
}

#endif  // SINGULARITY_SHARED_SHADER_KERR_MATH_H

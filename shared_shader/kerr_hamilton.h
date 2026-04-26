// shared_shader/kerr_hamilton.h
//
// Hamiltonian-form null geodesic integration for Kerr. Canonical coordinates
// ``(t, r, θ, φ)`` paired with canonical momenta ``(p_t, p_r, p_θ, p_φ)``.
// The two Killing isometries of Kerr make ``p_t = −E`` and ``p_φ = L_z``
// constants of motion, so we carry only the four dynamical variables plus
// ``p_r`` and ``p_θ`` forward in time. Six smooth ODEs; no sign flipping
// at turning points — the canonical momenta evolve through ``p_r = 0`` and
// ``p_θ = 0`` without branch ambiguity.
//
// This is the production-quality replacement for the sign-tracking form in
// ``kerr_math.h``. That earlier form is kept for unit-testing the algebra of
// the Carter-constant ``R`` and ``Θ`` potentials; the Hamiltonian form is
// what the GPU ray tracers consume once Phase 2 / Phase 4 begin.
//
// Method: Hamilton's equations advance ``(t, r, θ, φ, p_r, p_θ)``. The
// coordinate velocities come from the inverse metric directly; the momentum
// derivatives come from centred numerical differentiation of the
// Hamiltonian. At a step size of ``h = 0.1`` the dominant error is RK4
// truncation, not the O(h_diff²) numerical-differentiation error at
// ``h_diff = 1e−3``, so the approach is accurate to the float precision of
// the surrounding integrator.
//
// PHYSICS.md §7.3 references.

#ifndef SINGULARITY_SHARED_SHADER_KERR_HAMILTON_H
#define SINGULARITY_SHARED_SHADER_KERR_HAMILTON_H

#include "kerr_math.h"  // pulls in KerrConserved + geometric scalars
#include "shader_compat.h"

// Six-component canonical state advanced by the Hamiltonian form.
struct KerrHamState {
    float t, r, theta, phi;
    float p_r, p_theta;
};

DEVICE INLINE KerrHamState kerr_ham_state_add(KerrHamState a, KerrHamState b) {
    KerrHamState r;
    r.t = a.t + b.t;
    r.r = a.r + b.r;
    r.theta = a.theta + b.theta;
    r.phi = a.phi + b.phi;
    r.p_r = a.p_r + b.p_r;
    r.p_theta = a.p_theta + b.p_theta;
    return r;
}

DEVICE INLINE KerrHamState kerr_ham_state_scale(KerrHamState s, float k) {
    KerrHamState r;
    r.t = s.t * k;
    r.r = s.r * k;
    r.theta = s.theta * k;
    r.phi = s.phi * k;
    r.p_r = s.p_r * k;
    r.p_theta = s.p_theta * k;
    return r;
}

// 2·H(r, θ, p_r, p_θ) — Hamiltonian times two, with conserved (E, L_z, a, M)
// substituted in. For null geodesics this vanishes on-shell; for the
// momentum-derivative calculation we need the functional expression.
DEVICE INLINE float
kerr_ham_two_H(float r, float theta, float p_r, float p_theta, KerrConserved c) {
    const float st = sin(theta);
    const float s2 = st * st;
    // Guard at the poles — the φφ term carries a 1/sin²θ factor that we
    // smooth rather than let blow up at θ = 0 or π. The stepper should
    // never get exactly to the pole unless L_z = 0, in which case the
    // clamped term's contribution to 2H is zero anyway.
    const float safe_s2 = (s2 < 1e-10f) ? 1e-10f : s2;
    const float ct = cos(theta);
    const float c2 = ct * ct;
    const float r2 = r * r;
    const float a = c.a;
    const float M = c.M;
    const float a2 = a * a;
    const float Sigma = r2 + a2 * c2;
    const float Delta = r2 - 2.0f * M * r + a2;
    const float A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
    const float SD = Sigma * Delta;

    const float t_tt = -A * c.E * c.E / SD;
    const float t_tp = 4.0f * M * r * a * c.E * c.L_z / SD;
    const float t_pp = (Delta - a2 * safe_s2) * c.L_z * c.L_z / (SD * safe_s2);
    const float t_rr = Delta * p_r * p_r / Sigma;
    const float t_th = p_theta * p_theta / Sigma;

    return t_tt + t_tp + t_pp + t_rr + t_th;
}

// Full six-component RHS.
DEVICE INLINE KerrHamState kerr_ham_rhs(KerrHamState s, KerrConserved c) {
    const float r = s.r;
    const float theta = s.theta;
    const float st = sin(theta);
    const float s2 = st * st;
    const float safe_s2 = (s2 < 1e-10f) ? 1e-10f : s2;
    const float ct = cos(theta);
    const float c2 = ct * ct;
    const float r2 = r * r;
    const float a = c.a;
    const float M = c.M;
    const float a2 = a * a;
    const float Sigma = r2 + a2 * c2;
    const float Delta = r2 - 2.0f * M * r + a2;
    const float A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
    const float SD = Sigma * Delta;

    KerrHamState d;

    // Coordinate velocities from ∂H/∂p.
    d.t = (A * c.E - 2.0f * M * r * a * c.L_z) / SD;
    d.phi = (2.0f * M * r * a * c.E * s2 + (Delta - a2 * s2) * c.L_z) / (SD * safe_s2);
    d.r = Delta * s.p_r / Sigma;
    d.theta = s.p_theta / Sigma;

    // Momentum derivatives via centred numerical differentiation of 2H:
    //   dp_r/dλ     = −(1/2) ∂(2H)/∂r
    //   dp_θ/dλ     = −(1/2) ∂(2H)/∂θ
    // h_diff is chosen small enough that O(h²) truncation error is deep
    // below float precision, yet large enough that subtractive cancellation
    // in (f(x+h) − f(x−h)) stays negligible at float32.
    const float h_diff = 1.0e-3f;
    const float dH_dr = (kerr_ham_two_H(r + h_diff, theta, s.p_r, s.p_theta, c)
                         - kerr_ham_two_H(r - h_diff, theta, s.p_r, s.p_theta, c))
                        * (0.5f / h_diff);
    const float dH_dtheta = (kerr_ham_two_H(r, theta + h_diff, s.p_r, s.p_theta, c)
                             - kerr_ham_two_H(r, theta - h_diff, s.p_r, s.p_theta, c))
                            * (0.5f / h_diff);
    d.p_r = -0.5f * dH_dr;
    d.p_theta = -0.5f * dH_dtheta;

    return d;
}

// Classical fixed-step RK4 on the six-component state. No sign tracking.
DEVICE INLINE KerrHamState kerr_ham_rk4_step(KerrHamState y, float h, KerrConserved c) {
    KerrHamState k1 = kerr_ham_rhs(y, c);
    KerrHamState k2 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k1, 0.5f * h)), c);
    KerrHamState k3 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k2, 0.5f * h)), c);
    KerrHamState k4 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k3, h)), c);
    KerrHamState sum = kerr_ham_state_add(kerr_ham_state_add(k1, kerr_ham_state_scale(k2, 2.0f)),
                                          kerr_ham_state_add(kerr_ham_state_scale(k3, 2.0f), k4));
    return kerr_ham_state_add(y, kerr_ham_state_scale(sum, h / 6.0f));
}

// Initialise (p_r, p_θ) from a 4-velocity (u^r, u^θ) at a given point. Uses
// the covariant momenta definition p_i = g_iν u^ν restricted to the diagonal
// spatial pieces (which are g_rr = Σ/Δ and g_θθ = Σ).
//
// Host-only: MSL and HLSL both require explicit address-space qualifiers on
// pointer parameters, which would force this helper into a pile of per-
// backend overloads. The GPU kernels don't need it — they do the momentum
// projection inline when converting camera-frame velocities to canonical
// state. So we gate it on host C++ only.
#if !defined(__METAL_VERSION__) && !defined(__CUDACC__) && !defined(__HLSL_VERSION) \
    && !defined(_HLSL) && !defined(__SPIRV__)
INLINE void
kerr_ham_momenta_from_velocities(KerrHamState* state, float u_r, float u_theta, KerrConserved c) {
    const float r = state->r;
    const float theta = state->theta;
    const float ct = cos(theta);
    const float Sigma = r * r + c.a * c.a * ct * ct;
    const float Delta = r * r - 2.0f * c.M * r + c.a * c.a;
    state->p_r = Sigma * u_r / Delta;  // p_r = g_rr u^r = (Σ/Δ) u^r
    state->p_theta = Sigma * u_theta;  // p_θ = g_θθ u^θ = Σ u^θ
}
#endif

#endif  // SINGULARITY_SHARED_SHADER_KERR_HAMILTON_H

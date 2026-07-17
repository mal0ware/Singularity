// shared_shader/geodesic_math.h
//
// The actual physics. This single header is included by every backend's
// kernel boilerplate — Metal (MSL), Vulkan (HLSL → SPIR-V), CUDA, and the
// host C++ used by CLI and tests — via the DEVICE/INLINE macros in
// shader_compat.h. A bug fixed here is fixed everywhere simultaneously.
//
// Cross-references:
//   * PHYSICS.md §3  — Schwarzschild Christoffel symbols
//   * PHYSICS.md §4  — geodesic equation
//   * PHYSICS.md §6.2 — RK4
//
// The Christoffel expressions here are re-derived symbolically by
// verification/christoffel_sympy.py; any edit to the metric must keep that
// test passing.

#ifndef SINGULARITY_SHARED_SHADER_GEODESIC_MATH_H
#define SINGULARITY_SHARED_SHADER_GEODESIC_MATH_H

#include "shader_compat.h"

// 8-component geodesic state: coordinates (t, r, θ, φ) and 4-velocity
// (u^t, u^r, u^θ, u^φ). Deliberately POD so it is trivially copyable across
// host/device boundaries and byte-identical under every backend.
struct State {
    float t, r, theta, phi;
    float ut, ur, utheta, uphi;
};

DEVICE INLINE State state_add(State a, State b) {
    State r;
    r.t = a.t + b.t;
    r.r = a.r + b.r;
    r.theta = a.theta + b.theta;
    r.phi = a.phi + b.phi;
    r.ut = a.ut + b.ut;
    r.ur = a.ur + b.ur;
    r.utheta = a.utheta + b.utheta;
    r.uphi = a.uphi + b.uphi;
    return r;
}

DEVICE INLINE State state_scale(State s, float k) {
    State r;
    r.t = s.t * k;
    r.r = s.r * k;
    r.theta = s.theta * k;
    r.phi = s.phi * k;
    r.ut = s.ut * k;
    r.ur = s.ur * k;
    r.utheta = s.utheta * k;
    r.uphi = s.uphi * k;
    return r;
}

// Fused a*x + y — saves a temporary vs state_add(state_scale(x, a), y). Used
// heavily by the higher-order integrators below where each stage is a sum
// of 2-6 scaled k_i terms.
DEVICE INLINE State state_axpy(float a, State x, State y) {
    State r;
    r.t = a * x.t + y.t;
    r.r = a * x.r + y.r;
    r.theta = a * x.theta + y.theta;
    r.phi = a * x.phi + y.phi;
    r.ut = a * x.ut + y.ut;
    r.ur = a * x.ur + y.ur;
    r.utheta = a * x.utheta + y.utheta;
    r.uphi = a * x.uphi + y.uphi;
    return r;
}

// Radius-adaptive step size (gated by SING_FLAG_ADAPTIVE_STEP).
//
// Curvature falls off steeply with radius — the geodesic RHS terms scale
// like M/r² and the bending per unit affine parameter like M/r³ — so a
// fixed step tuned for the photon-sphere region wastes almost its entire
// budget crossing nearly-flat space. Grow the step linearly with r:
//
//   h(r) = h_base · clamp(r / 6M, 1, 40)
//
// Inside r ≤ 6M (photon sphere at 3M, ISCO at 6M for a = 0) integration is
// unchanged. At r = 30M — a typical disc outer edge — the step matches the
// desktop Draft preset (0.12 · 5 = 0.6), and the far field tops out at 40×.
// SIMT-friendly: a pure per-step function of the current state, no
// data-dependent loop divergence beyond what ray termination already has.
//
// Accuracy is validated on the host build of this exact header by
// tests/test_adaptive_step.cpp (deflection parity with a fine fixed-step
// reference + E/L conservation). Hand-ported to WGSL in
// render/webgpu/shaders/geodesic_kernel.wgsl — keep the two in sync.
DEVICE INLINE float adaptive_h(float h_base, float r, float mass_M) {
    float mult = r / (6.0f * mass_M);
    mult = (mult < 1.0f) ? 1.0f : mult;
    mult = (mult > 40.0f) ? 40.0f : mult;
    return h_base * mult;
}

// Schwarzschild geodesic RHS in spherical BL coordinates.
// du^μ/dλ = -Γ^μ_νσ u^ν u^σ with the non-zero Christoffels of PHYSICS.md §3.
// rs = 2M (Schwarzschild radius in geometrized units, PHYSICS.md §2).
DEVICE INLINE State geodesic_rhs_schwarzschild(State s, float rs) {
    float f = 1.0f - rs / s.r;
    float r2 = s.r * s.r;
    float sin_t = sin(s.theta);
    float cos_t = cos(s.theta);

    State d;
    d.t = s.ut;
    d.r = s.ur;
    d.theta = s.utheta;
    d.phi = s.uphi;

    d.ut = -(rs / (r2 * f)) * s.ut * s.ur;

    d.ur = -(rs * f / (2.0f * r2)) * s.ut * s.ut + (rs / (2.0f * r2 * f)) * s.ur * s.ur
           + s.r * f * (s.utheta * s.utheta + sin_t * sin_t * s.uphi * s.uphi);

    d.utheta = -2.0f * s.ur * s.utheta / s.r + sin_t * cos_t * s.uphi * s.uphi;

    d.uphi = -2.0f * s.ur * s.uphi / s.r - 2.0f * (cos_t / sin_t) * s.utheta * s.uphi;

    return d;
}

// Classical fixed-step RK4 (PHYSICS.md §6.2). O(h^4) global error.
DEVICE INLINE State rk4_step(State y, float h, float rs) {
    State k1 = geodesic_rhs_schwarzschild(y, rs);
    State k2 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k1, 0.5f * h)), rs);
    State k3 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k2, 0.5f * h)), rs);
    State k4 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k3, h)), rs);
    State sum =
        state_add(state_add(k1, state_scale(k2, 2.0f)), state_add(state_scale(k3, 2.0f), k4));
    return state_add(y, state_scale(sum, h / 6.0f));
}

// Forward Euler. First-order accurate — shipped only so the Phase 1 demo
// can illustrate why RK4 matters (PHYSICS.md §6.1). Do not use in production.
DEVICE INLINE State euler_step(State y, float h, float rs) {
    return state_add(y, state_scale(geodesic_rhs_schwarzschild(y, rs), h));
}

// Dormand-Prince 5(4) fixed-step — 5th-order accurate per step (O(h^6) local
// truncation vs RK4's O(h^5)). Butcher tableau from Dormand & Prince, J. Comp.
// Appl. Math. 6 (1980) 19-26. Seven RHS evaluations per step (~1.75x RK4's
// cost). The embedded 4(4) error estimate is discarded in this form — for
// host-side adaptive use, see core/include/physics/integrator.hpp.
//
// **Do not swap RK4 → DOPRI5 on the GPU path without re-checking conservation
// in float32.** Empirically (see the revert that points back here), at every
// step size we use in production (h ∈ [0.1M, 1.0M], 10K steps on the
// test_conserved.cpp photon path), RK4 conserves E, L, and the null residual
// 5–10× *better* than DOPRI5 in float32. DOPRI5's truncation advantage is
// dominated by accumulated round-off from its 7 stages with awkward
// coefficients (19372/6561, -25360/2187, etc.) — those cancellations bleed
// more bits than RK4's clean 1/2-and-2 weights ever do. The truncation win
// only emerges in double precision, where round-off floors out below
// truncation; the GPU path is float32, so RK4 is the right choice there.
// This routine is kept available for double-precision host-side use and as
// the building block for an eventual adaptive driver, not because it's
// drop-in-better than RK4 at fixed step.
//
// GPU SIMT rationale: even in double precision, adaptive step-size control
// (shrink h on error spike) breaks SIMT coherence because rays' step counts
// would diverge within a warp. Adaptive DOPRI5 belongs on the CPU path.
DEVICE INLINE State dopri5_step(State y, float h, float rs) {
    const float a21 = 1.0f / 5.0f;
    const float a31 = 3.0f / 40.0f, a32 = 9.0f / 40.0f;
    const float a41 = 44.0f / 45.0f, a42 = -56.0f / 15.0f, a43 = 32.0f / 9.0f;
    const float a51 = 19372.0f / 6561.0f, a52 = -25360.0f / 2187.0f, a53 = 64448.0f / 6561.0f,
                a54 = -212.0f / 729.0f;
    const float a61 = 9017.0f / 3168.0f, a62 = -355.0f / 33.0f, a63 = 46732.0f / 5247.0f,
                a64 = 49.0f / 176.0f, a65 = -5103.0f / 18656.0f;
    const float a71 = 35.0f / 384.0f, a73 = 500.0f / 1113.0f, a74 = 125.0f / 192.0f,
                a75 = -2187.0f / 6784.0f, a76 = 11.0f / 84.0f;

    State k1 = geodesic_rhs_schwarzschild(y, rs);

    State y2 = state_axpy(a21 * h, k1, y);
    State k2 = geodesic_rhs_schwarzschild(y2, rs);

    State y3 = state_axpy(a32 * h, k2, state_axpy(a31 * h, k1, y));
    State k3 = geodesic_rhs_schwarzschild(y3, rs);

    State y4 = state_axpy(a43 * h, k3, state_axpy(a42 * h, k2, state_axpy(a41 * h, k1, y)));
    State k4 = geodesic_rhs_schwarzschild(y4, rs);

    State y5 = state_axpy(
        a54 * h, k4, state_axpy(a53 * h, k3, state_axpy(a52 * h, k2, state_axpy(a51 * h, k1, y))));
    State k5 = geodesic_rhs_schwarzschild(y5, rs);

    State y6 = state_axpy(
        a65 * h,
        k5,
        state_axpy(a64 * h,
                   k4,
                   state_axpy(a63 * h, k3, state_axpy(a62 * h, k2, state_axpy(a61 * h, k1, y)))));
    State k6 = geodesic_rhs_schwarzschild(y6, rs);

    return state_axpy(
        a76 * h,
        k6,
        state_axpy(a75 * h,
                   k5,
                   state_axpy(a74 * h, k4, state_axpy(a73 * h, k3, state_axpy(a71 * h, k1, y)))));
}

#endif  // SINGULARITY_SHARED_SHADER_GEODESIC_MATH_H
